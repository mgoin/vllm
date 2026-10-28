# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The startup deployment gate of the Muse Spark specialized decode.

``unsupported`` is pure host code: it reads the resolved ``VllmConfig`` and
probes the rank-local multicast support. These tests replace the probe and the
platform with fakes, so they run without a GPU and check that every missing
requirement is reported rather than the first one.
"""

from types import SimpleNamespace

import pytest

import vllm.models.muse_spark.nvidia.specialized_decode as gate
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class _BlackwellPlatform:
    def is_cuda(self) -> bool:
        return True

    def is_device_capability_family(self, family: int) -> bool:
        return family == 100


@pytest.fixture
def config_gate(monkeypatch):
    """The gate with a Blackwell GPU and NVLink multicast on every rank."""
    monkeypatch.setattr(gate, "current_platform", _BlackwellPlatform())
    monkeypatch.setattr(gate, "_has_multicast", lambda: True)
    return gate


def _config(
    *,
    attention_backend=AttentionBackendEnum.FLASHINFER,
    moe_backend: str = "auto",
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_AND_PIECEWISE,
    compilation_mode: CompilationMode = CompilationMode.VLLM_COMPILE,
    enforce_eager: bool = False,
    executor_backend: str = "mp",
    tensor_parallel_size: int = 4,
    quantized: bool = True,
):
    parallel = SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        use_ubatching=False,
        distributed_executor_backend=executor_backend,
    )
    compilation = SimpleNamespace(
        mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
    )
    quant_config = (
        ModelOptNvFp4Config.__new__(ModelOptNvFp4Config) if quantized else None
    )
    return SimpleNamespace(
        parallel_config=parallel,
        attention_config=SimpleNamespace(backend=attention_backend),
        compilation_config=compilation,
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            enforce_eager=enforce_eager,
        ),
        quant_config=quant_config,
        use_v2_model_runner=True,
        lora_config=None,
        kv_transfer_config=None,
    )


def test_supported_deployment_has_no_missing_requirements(config_gate):
    assert config_gate.unsupported(_config()) == []


def test_external_capture_benchmark_is_allowed(config_gate):
    # vllm bench forward: no vLLM CUDA graphs or torch.compile, external launcher.
    config = _config(
        cudagraph_mode=CUDAGraphMode.NONE,
        compilation_mode=CompilationMode.NONE,
        executor_backend="external_launcher",
    )
    assert config_gate.unsupported(config) == []


def test_enforce_eager_is_rejected(config_gate):
    config = _config(
        cudagraph_mode=CUDAGraphMode.NONE,
        compilation_mode=CompilationMode.NONE,
        enforce_eager=True,
    )
    missing = config_gate.unsupported(config)
    assert any("FULL CUDA graphs" in need for need in missing)


def test_cuda_graphs_none_is_rejected_for_serving(config_gate):
    config = _config(cudagraph_mode=CUDAGraphMode.NONE)
    missing = config_gate.unsupported(config)
    assert any("FULL CUDA graphs" in need for need in missing)


def test_non_flashinfer_attention_backend_is_rejected(config_gate):
    config = _config(attention_backend=AttentionBackendEnum.TRITON_ATTN)
    missing = config_gate.unsupported(config)
    assert any("FLASHINFER attention" in need for need in missing)


def test_non_trtllm_moe_backend_is_rejected(config_gate):
    missing = config_gate.unsupported(_config(moe_backend="cutlass"))
    assert any("MoE backend" in need for need in missing)


def test_flashinfer_version_is_required(config_gate, monkeypatch):
    monkeypatch.setattr(config_gate, "_flashinfer_version", lambda: "0.0.0")
    missing = config_gate.unsupported(_config())
    assert any("flashinfer-python" in need for need in missing)


def test_every_missing_requirement_is_listed(config_gate):
    config = _config(
        cudagraph_mode=CUDAGraphMode.NONE,
        tensor_parallel_size=8,
        moe_backend="cutlass",
        quantized=False,
    )
    missing = config_gate.unsupported(config)
    for fragment in ("FULL CUDA graphs", "tensor parallel size 4", "MoE backend"):
        assert any(fragment in need for need in missing), (fragment, missing)
