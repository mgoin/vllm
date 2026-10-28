# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts whose GEMM2 epilogue also publishes into the MoE tail.

``PublishingRoutedExperts`` is ``MuseRoutedExperts`` (``experts/``) with one
addition: every 8-byte store of a GEMM2 row (route ``8 t + k``, route-major)
is also multicast, negative zeros made +0, into this rank's slot of the tail's
``[3, tp, capacity * 8, 4096]`` published mailbox at the layer's parity. The
local rows stay bit for bit those of the stock kernel. The tail's finalize
(``MoeTail(..., parity=p)``) then starts at the poll: the rows reach the
peers while the experts still run, and nothing republishes them.

The epilogue is the experts kernel's own: while this kernel compiles, the
GEMM2 row store of ``routed_experts_cute`` is wrapped with the multicast. The
mailbox address is a compile-time constant (one kernel per parity), as is the
mailbox capacity; the rank follows from it.
"""

from __future__ import annotations

import contextlib
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.models.common.specialized_decode.primitives import (
    asm,
    no_neg_zero,
)

from ..experts import routed_experts as host
from ..experts import routed_experts_cute as device
from ..experts.routed_experts import MuseRoutedExperts
from .finalize import HIDDEN, TOP_K


@asm()
def _multicast_v2(address, a, b):
    code = "multimem.st.relaxed.sys.global.v2.bf16x2 [$0], {$1, $2};"
    return code, "l,r,r,~{memory}", (Int64(address), Uint32(a), Uint32(b))


class PublishingExpertsKernel(device.RoutedExpertsKernel):
    """``RoutedExpertsKernel`` publishing its GEMM2 rows at ``slot_base``, the
    byte address of this rank's slot of the parity's mailbox generation."""

    def __init__(self, num_stages: int, slot_base: int):
        super().__init__(num_stages)
        self.slot_base = slot_base
        self._out = None

    def epilogue(self, *args):
        # The GEMM2 rows' destination, for the row store's offsets. Only this
        # method's stores publish: the GEMM1 SwiGLU output shares `store_v2`.
        self._out = args[16].iterator
        with _publishing_rows(self):
            return device.RoutedExpertsKernel.epilogue(self, *args)

    def store_row(self, store, address, a, b, *, loc=None, ip=None):
        store(address, a, b, loc=loc, ip=ip)
        offset = Int64(address.toint()) - Int64(self._out.toint())
        _multicast_v2(Int64(self.slot_base) + offset, no_neg_zero(a), no_neg_zero(b))


@contextlib.contextmanager
def _publishing_rows(kernel: PublishingExpertsKernel):
    store = device.store_v2

    def store_row(address, a, b, *, loc=None, ip=None):
        kernel.store_row(store, address, a, b, loc=loc, ip=ip)

    device.store_v2 = store_row
    try:
        yield
    finally:
        device.store_v2 = store


@cache
def _kernel(device_index: int, slot_base: int):
    """``PublishingExpertsKernel`` compiled for a device and a mailbox slot."""
    kernel = PublishingExpertsKernel(host.STAGES, slot_base)
    fake = host._fake
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            kernel,
            fake(cutlass.Float4E2M1FN, 32),
            fake(cutlass.Float8E4M3FN),
            fake(cutlass.Float4E2M1FN, 32),
            fake(cutlass.Float8E4M3FN),
            fake(cutlass.Float4E2M1FN),
            fake(cutlass.Uint8),
            fake(cutlass.Int32),
            fake(cutlass.Float32),
            fake(cutlass.Float32),
            fake(cutlass.Float32),
            fake(cutlass.Float4E2M1FN, 32),
            fake(cutlass.Uint8),
            fake(cutlass.Float4E2M1FN, 32),
            fake(cutlass.Uint8),
            fake(cutlass.Int32),
            fake(cutlass.Uint32),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


class PublishingRoutedExperts(MuseRoutedExperts):
    """``MuseRoutedExperts`` that also publish into a tail's mailbox.

    Args:
        target: ``MoeTail.published_target`` (multicast address, rank and
            capacity of the published mailbox).
        tp: the TP group size.
        parities: the layer parities calls may use (``layer_parities``);
            kernels for them compile here.
        Other arguments as ``MuseRoutedExperts``.

    """

    def __init__(
        self,
        *args,
        target: tuple[int, int, int],
        tp: int,
        parities=(0, 1, 2),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.multicast, self.rank, self.capacity = target
        self.tp = tp
        for parity in parities:
            self._compiled(parity)

    def _slot_base(self, parity: int) -> int:
        slot = (parity * self.tp + self.rank) * self.capacity * TOP_K
        return self.multicast + slot * HIDDEN * 2

    def _compiled(self, parity: int):
        return _kernel(self.device_index, self._slot_base(parity))

    def __call__(  # type: ignore[override]
        self,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        parity: int,
    ) -> UnfinalizedMoEOutput:
        """``MuseRoutedExperts.__call__``, publishing at ``parity``."""
        rows = a1q.shape[0]
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed the mailbox's {self.capacity}")
        assert a1q.is_contiguous() and a1q_scale.is_contiguous()
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
        assert tuple(topk_ids.shape) == (rows, TOP_K)
        if out is None:
            out = torch.empty(
                rows * TOP_K, HIDDEN, dtype=torch.bfloat16, device=a1q.device
            )
        xc, xcs, h, hs = host._scratch(self.device_index)
        self._compiled(parity)(
            host._pointer(cutlass.Float4E2M1FN, self.w13, 32),
            host._pointer(cutlass.Float8E4M3FN, self.w13_scale),
            host._pointer(cutlass.Float4E2M1FN, self.w2, 32),
            host._pointer(cutlass.Float8E4M3FN, self.w2_scale),
            host._pointer(cutlass.Float4E2M1FN, a1q),
            host._pointer(cutlass.Uint8, a1q_scale),
            host._pointer(cutlass.Int32, topk_ids),
            host._pointer(cutlass.Float32, self.g1_alphas),
            host._pointer(cutlass.Float32, self.g1_scale_c),
            host._pointer(cutlass.Float32, self.g2_alphas),
            host._pointer(cutlass.Float4E2M1FN, xc, 32),
            host._pointer(cutlass.Uint8, xcs),
            host._pointer(cutlass.Float4E2M1FN, h, 32),
            host._pointer(cutlass.Uint8, hs),
            host._pointer(cutlass.Int32, self.workspace),
            host._pointer(cutlass.Uint32, out),
            Int32(rows),
            Int32(host._sm_count(self.device_index)),
            host._stream(),
        )
        permuted = host._identity(self.device_index)[: rows * TOP_K].view(rows, TOP_K)
        return UnfinalizedMoEOutput(
            gemm2_permuted=out,
            expert_weights=topk_weights.view(rows, TOP_K),
            expanded_idx_to_permuted_idx=permuted,
            route_major=True,
        )
