# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode on NVIDIA Blackwell: a decode executor over
the stock text backbone, enabled by
``additional_config={"decode_backend": "latency_specialized"}``.

The executor runs FULL-graph decode and verify steps; every other step runs the
stock forward. ``vllm bench forward`` captures its own FULL graphs through the
external launcher with vLLM CUDA graphs and ``torch.compile`` off, which is the
one accepted no-graphs deployment; a serving deployment with CUDA graphs off,
including ``--enforce-eager``, is rejected at startup rather than silently
running stock. It issues cross-rank collectives, so the deployment check below
is collective over the TP group and every per-step decision reads host state
that all TP ranks share.
"""

import importlib.metadata
from dataclasses import dataclass
from functools import cache

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

# Rows of the largest step the executor runs: 8 requests verifying 8 tokens
# each. Above it the stock layers are at least as fast today: at 128 rows the
# executor's own fused stages (all-reduce + dual norm, router, qkv) cost more
# than the stock kernels they replace, and the end-to-end step is 2.3-2.8 %
# slower (measured at concurrency 16, with and without adaptive verification).
# Raise it only with a measurement that beats stock at 128 rows.
MAX_ROWS = 64
# The flashinfer-python that requirements/cuda.txt pins. The executor calls
# FlashInfer's mnnvl_cutedsl internals and reads its attention metadata.
FLASHINFER_VERSION = "0.7.2.post1"


@dataclass(frozen=True)
class DecodeConfig:
    # Tune the executor's FlashInfer kernels at build
    # (kernel_config.enable_flashinfer_autotune is not False).
    autotune: bool


@cache
def _flashinfer_version() -> str | None:
    """The installed flashinfer-python version, or None."""
    try:
        return importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return None


def _has_multicast() -> bool:
    try:
        from flashinfer.comm.mnnvl import is_multicast_supported
    except ImportError:
        return False
    return bool(is_multicast_supported(torch.accelerator.current_device_index()))


def unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every requirement of the specialized decode this deployment misses."""
    parallel = vllm_config.parallel_config
    compilation = vllm_config.compilation_config
    cudagraph_mode = compilation.cudagraph_mode
    requirements = {
        "a Blackwell (SM100 family) GPU": (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ),
        "Model Runner V2": vllm_config.use_v2_model_runner,
        "tensor parallel size 4 without pipeline, data, expert or context "
        "parallelism or micro-batching": (
            parallel.tensor_parallel_size == 4
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and not parallel.use_ubatching
        ),
        "an NVFP4 (modelopt_fp4) checkpoint": isinstance(
            vllm_config.quant_config, ModelOptNvFp4Config
        ),
        # The prepared-NVFP4 experts need the resolved FlashInfer TRT-LLM
        # backend; `auto` is checked again against the built layers.
        "the FlashInfer TRT-LLM NVFP4 MoE backend (moe_backend auto or "
        "flashinfer_trtllm)": (
            vllm_config.kernel_config.moe_backend in ("auto", "flashinfer_trtllm")
        ),
        # Auto is checked again against the built attention layers.
        "the FLASHINFER attention backend (or auto resolving to it)": (
            vllm_config.attention_config.backend
            in (None, AttentionBackendEnum.FLASHINFER)
        ),
        f"flashinfer-python {FLASHINFER_VERSION} (found {_flashinfer_version()})": (
            _flashinfer_version() == FLASHINFER_VERSION
        ),
        # The executor runs only inside a FULL graph capture. vllm bench forward
        # (the external launcher) captures its own graphs with vLLM CUDA graphs
        # and torch.compile off; a serving deployment that turns both off would
        # silently run stock.
        "FULL CUDA graphs for decode batches (FULL_AND_PIECEWISE or "
        "FULL_DECODE_ONLY), or the external-capture benchmark mode (no CUDA "
        "graphs and no torch.compile)": (
            (
                cudagraph_mode is not None
                and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            )
            or (
                cudagraph_mode is not None
                and cudagraph_mode.decode_mode() == CUDAGraphMode.NONE
                and compilation.mode == CompilationMode.NONE
                and not vllm_config.model_config.enforce_eager
                and parallel.distributed_executor_backend == "external_launcher"
            )
        ),
        # The decode loop runs in forward; a compiled forward would trace it.
        "no torch.compile of the model": (
            compilation.mode != CompilationMode.STOCK_TORCH_COMPILE
        ),
        # The executor's kernels bind weight addresses.
        "no sleep mode": not vllm_config.model_config.enable_sleep_mode,
        "no LoRA, KV transfer or batch invariance": (
            vllm_config.lora_config is None
            and vllm_config.kv_transfer_config is None
            and not envs.VLLM_BATCH_INVARIANT
        ),
        "NVLink multicast for the fused collectives": _has_multicast(),
    }
    return [need for need, ok in requirements.items() if not ok]


def specialized_decode_config(vllm_config: VllmConfig) -> DecodeConfig | None:
    """The specialized decode settings, or None for the stock decode.

    Collective over the TP group when requested: every rank raises if any rank
    cannot run the specialized decode.
    """
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    missing = unsupported(vllm_config)
    supported = torch.tensor([0 if missing else 1], dtype=torch.int32)
    # The multicast check is rank-local, and ranks that disagree would hang.
    dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    if not supported.item():
        reason = "; ".join(missing) or "another tensor parallel rank to support it"
        raise ValueError(f"decode_backend=latency_specialized needs {reason}.")
    logger.info_once(
        "Muse Spark specialized decode on, for FULL-graph decode steps of up to "
        "%d rows.",
        MAX_ROWS,
    )
    return DecodeConfig(
        autotune=vllm_config.kernel_config.enable_flashinfer_autotune is not False
    )
