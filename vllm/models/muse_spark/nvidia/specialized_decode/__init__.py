# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode on NVIDIA Blackwell: a decode executor over
the stock text backbone, enabled by
``additional_config={"decode_backend": "latency_specialized"}``.

The executor runs FULL-graph decode and verify steps; every other step runs the
stock forward. It issues cross-rank collectives, so the deployment check below
is collective over the TP group and every per-step decision reads host state
that all TP ranks share.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Rows of the largest step the executor runs: 32 requests verifying 8 tokens
# each. Larger decode steps run the stock layers.
MAX_ROWS = 256


@dataclass(frozen=True)
class DecodeConfig:
    # Tune the executor's FlashInfer kernels at build
    # (kernel_config.enable_flashinfer_autotune is not False).
    autotune: bool


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
        "the default MoE backend (not deep_gemm_mega_moe)": (
            vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe"
        ),
        # Without CUDA graphs every step runs the stock layers; benchmarks that
        # capture their own graphs (vllm bench forward) still run the executor.
        "FULL CUDA graphs for decode batches, or none": (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() in (CUDAGraphMode.FULL, CUDAGraphMode.NONE)
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
    if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning_once(
            "Muse Spark specialized decode runs only in CUDA graph captures, "
            "and CUDA graphs are off."
        )
    return DecodeConfig(
        autotune=vllm_config.kernel_config.enable_flashinfer_autotune is not False
    )
