# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark NVFP4 routed experts for decode batches of 1-64 tokens.

Replaces, for one MoE layer, the TRT-LLM-gen chain of
``MuseSparkTrtLlmNvFp4Experts.apply_routed`` (routing indices, GEMM1 +
SwiGLU, GEMM2; ``do_finalize=False``) with one persistent kernel
(``routed_experts_cute.py``) that reads the router's ``topk_ids`` directly::

    a1q, a1q_scale (norm_quant_trtllm_nvfp4) ─┐
    topk_ids int32, topk_weights FP32 ────────┤
                                              ▼
              RoutedExpertsKernel ─► GEMM2 rows [M * 8, 4096] BF16, route-major
                                              │
         UnfinalizedMoEOutput(route_major) ◄──┘  ─► MusePostExpertFinalize

The rows are bitwise those of the TRT-LLM-gen kernels (TP partial sums,
unweighted); row ``8 t + k`` is route ``(t, k)``, so the permute map is the
identity and ``route_major`` is set. The same weights, block scales and
global scales as the stock path are read in place: nothing is repacked.

Each layer owns a 1 KB workspace (the kernel's work and slot counters, which
it resets itself); the intermediate H and its scales live in a per-device
scratch shared by all layers, written only after the grid dependency wait.
Kernels compile when a layer binds (one per activation-tile width: 8, 16, 32
and 64 tokens) and every launch only passes pointers and scalars, so calls
can be captured in CUDA graphs with inputs first allocated inside the
capture.
"""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

from .routed_experts_cute import (
    EXPERTS,
    HIDDEN,
    HS_SLOT_BYTES,
    INTER,
    MAX_ROWS,
    TOP_K,
    WS_WORDS,
    RoutedExpertsKernel,
)

TILE_WIDTHS = (8, 16, 32, 64)
# Pipeline stages per tile width (shared memory: 20 KB + 128 B per token row
# per stage, plus 32 KB of resident input scales).
STAGES = {8: 8, 16: 8, 32: 7, 64: 6}


def tile_width(rows: int) -> int:
    """The activation-tile width a call of ``rows`` tokens runs with."""
    for width in TILE_WIDTHS:
        if rows <= width:
            return width
    raise ValueError(f"{rows} rows: at most {MAX_ROWS}")


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int = 16):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype: type[cutlass.Numeric], align: int = 16):
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


def _stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@cache
def _kernel(device_index: int, width: int, ids_ready: bool):
    """``RoutedExpertsKernel`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertsKernel(width, ids_ready, STAGES[width]),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN),
            _fake(cutlass.Float4E2M1FN),
            _fake(cutlass.Uint32),
            _fake(cutlass.Int32),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.Float32),
            _fake(cutlass.Float4E2M1FN),
            _fake(cutlass.Uint8),
            _fake(cutlass.Uint8),
            _fake(cutlass.Int32),
            _fake(cutlass.Uint32),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _scratch(device_index: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """H [E * width, 1024] E2M1 (bytes) and its per-slot scale atoms."""
    device = torch.device("cuda", device_index)
    h = torch.zeros(EXPERTS * width, INTER // 2, dtype=torch.uint8, device=device)
    hs = torch.zeros(EXPERTS * HS_SLOT_BYTES, dtype=torch.uint8, device=device)
    return h, hs


@cache
def _identity(device_index: int) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    return torch.arange(MAX_ROWS * TOP_K, dtype=torch.int32, device=device)


@cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def check_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    scales: tuple[torch.Tensor, ...],
) -> None:
    """Raise ``ValueError`` unless the tensors are the TRT-LLM-gen layout of
    this kernel's shapes."""
    for tensor, shape in (
        (w13, (EXPERTS, 2 * INTER, HIDDEN // 2)),
        (w13_scale, (EXPERTS, 2 * INTER, HIDDEN // 16)),
        (w2, (EXPERTS, HIDDEN, INTER // 2)),
        (w2_scale, (EXPERTS, HIDDEN, INTER // 16)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected a contiguous {shape} 1-byte tensor, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
    if w13.data_ptr() % 32 or w2.data_ptr() % 32:
        raise ValueError("packed expert weights must be 32-byte aligned")
    if w13_scale.data_ptr() % 16 or w2_scale.data_ptr() % 16:
        raise ValueError("weight scales must be 16-byte aligned")
    for scale in scales:
        if (
            scale.dtype != torch.float32
            or tuple(scale.shape) != (EXPERTS,)
            or not scale.is_contiguous()
        ):
            raise ValueError("per-expert scales must be contiguous FP32 [E]")


class MuseRoutedExperts:
    """The routed-expert kernel bound to one MoE layer's prepared NVFP4
    weights (module docstring).

    Args:
        w13, w13_scale, w2, w2_scale: the layer's ``w13_weight``,
            ``w13_weight_scale``, ``w2_weight`` and ``w2_weight_scale`` after
            ``MuseSparkModelOptNvFp4FusedMoE.process_weights_after_loading``.
        g1_alphas: FP32 [E] gate dequantization scale (``output1_scale_gate``).
        g1_scale_c: FP32 [E] up dequantization over the H global scale
            (``output1_scale``).
        g2_alphas: FP32 [E] GEMM2 output scale (``output2_scale``).
        ids_ready: ignored (kept for callers): the kernel always reads the
            routing after its grid dependency wait, because an event join
            captured into a CUDA graph also becomes a programmatic edge.

    """

    def __init__(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        g1_alphas: torch.Tensor,
        g1_scale_c: torch.Tensor,
        g2_alphas: torch.Tensor,
        *,
        ids_ready: bool = False,
    ) -> None:
        scales = (g1_alphas, g1_scale_c, g2_alphas)
        check_weights(w13, w13_scale, w2, w2_scale, scales)
        self.w13, self.w13_scale, self.w2, self.w2_scale = (
            w13,
            w13_scale,
            w2,
            w2_scale,
        )
        self.g1_alphas, self.g1_scale_c, self.g2_alphas = scales
        self.ids_ready = ids_ready
        index = w13.device.index
        self.device_index = (
            torch.accelerator.current_device_index() if index is None else index
        )
        self.workspace = torch.zeros(WS_WORDS, dtype=torch.int32, device=w13.device)
        for width in TILE_WIDTHS:
            _kernel(self.device_index, width, ids_ready)
            _scratch(self.device_index, width)
        _identity(self.device_index)

    def __call__(
        self,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> UnfinalizedMoEOutput:
        """GEMM2 rows of the routed experts.

        Args:
            a1q: [M, 2048] uint8 NVFP4 input (``nvfp4_input_global_scale``).
            a1q_scale: [M, 256] linear E4M3 block scales (uint8 or FP8).
            topk_ids: int32 [M, 8] expert ids in [0, 256).
            topk_weights: FP32 [M, 8], passed through to the finalize.
            out: optional BF16 [M * 8, 4096] destination.

        """
        rows = a1q.shape[0]
        width = tile_width(rows)
        assert a1q.is_contiguous() and a1q_scale.is_contiguous()
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
        assert tuple(topk_ids.shape) == (rows, TOP_K)
        if out is None:
            out = torch.empty(
                rows * TOP_K, HIDDEN, dtype=torch.bfloat16, device=a1q.device
            )
        h, hs = _scratch(self.device_index, width)
        _kernel(self.device_index, width, self.ids_ready)(
            _pointer(cutlass.Float4E2M1FN, self.w13, 32),
            _pointer(cutlass.Float8E4M3FN, self.w13_scale),
            _pointer(cutlass.Float4E2M1FN, self.w2, 32),
            _pointer(cutlass.Float8E4M3FN, self.w2_scale),
            _pointer(cutlass.Float4E2M1FN, a1q),
            _pointer(cutlass.Uint32, a1q_scale),
            _pointer(cutlass.Int32, topk_ids),
            _pointer(cutlass.Float32, self.g1_alphas),
            _pointer(cutlass.Float32, self.g1_scale_c),
            _pointer(cutlass.Float32, self.g2_alphas),
            _pointer(cutlass.Float4E2M1FN, h),
            _pointer(cutlass.Uint8, h),
            _pointer(cutlass.Uint8, hs),
            _pointer(cutlass.Int32, self.workspace),
            _pointer(cutlass.Uint32, out),
            Int32(rows),
            Int32(_sm_count(self.device_index)),
            _stream(),
        )
        permuted = _identity(self.device_index)[: rows * TOP_K].view(rows, TOP_K)
        return UnfinalizedMoEOutput(
            gemm2_permuted=out,
            expert_weights=topk_weights.view(rows, TOP_K),
            expanded_idx_to_permuted_idx=permuted,
            route_major=True,
        )


def build_routed_experts(
    runner: torch.nn.Module, ids_ready: bool = False
) -> MuseRoutedExperts | str:
    """The kernel bound to a ``MuseMoERunner``'s prepared NVFP4 experts, or
    why the layer keeps the stock path (BF16 layers, expert parallelism,
    other shapes or a SwiGLU clamp)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    experts = runner._prepared_nvfp4_experts
    if experts is None:
        return "experts are not the prepared-input TRT-LLM NVFP4 kernel"
    if (
        experts.local_num_experts != EXPERTS
        or experts.moe_config.moe_parallel_config.ep_size != 1
    ):
        return f"{experts.local_num_experts} local experts (needs {EXPERTS}, no EP)"
    if experts.topk != TOP_K or experts.intermediate_size_per_partition != INTER:
        return "top-k or intermediate size differ"
    if experts.moe_config.activation != MoEActivation.SILU:
        return f"activation {experts.moe_config.activation}"
    if any(
        t is not None
        for t in (experts.gemm1_alpha, experts.gemm1_beta, experts.gemm1_clamp_limit)
    ):
        return "SwiGLU alpha/beta/clamp"
    layer = runner.routed_experts
    quant = experts.quant_config
    try:
        return MuseRoutedExperts(
            layer.w13_weight,
            quant.w1_scale.view(torch.float8_e4m3fn),
            layer.w2_weight,
            quant.w2_scale.view(torch.float8_e4m3fn),
            quant.g1_alphas,
            experts.g1_scale_c,
            quant.g2_alphas,
            ids_ready=ids_ready,
        )
    except ValueError as error:
        return str(error)
