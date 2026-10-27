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
mailbox address, rank, capacity and parity are compile-time constants (one
kernel per tile width and parity).
"""

from __future__ import annotations

import contextlib
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    asm,
    no_neg_zero,
)

from ..experts import routed_experts as host
from ..experts import routed_experts_cute as device
from ..experts.routed_experts import MuseRoutedExperts, tile_width
from .finalize import HIDDEN, TOP_K


@asm()
def _multicast_v2(address, a, b):
    code = "multimem.st.relaxed.sys.global.v2.bf16x2 [$0], {$1, $2};"
    return code, "l,r,r,~{memory}", (Int64(address), Uint32(a), Uint32(b))


class PublishingExpertsKernel(device.RoutedExpertsKernel):
    """``RoutedExpertsKernel`` publishing its GEMM2 rows at ``slot_base``, the
    byte address of this rank's slot of the parity's mailbox generation."""

    def __init__(self, n: int, ids_ready: bool, num_stages: int, slot_base: int):
        super().__init__(n, ids_ready, num_stages)
        self.slot_base = slot_base
        self._out = None

    def epilogue(self, *args):
        # The GEMM2 rows' destination, for the row store's offsets.
        self._out = args[17].iterator
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
def _kernel(device_index: int, width: int, ids_ready: bool, slot_base: int):
    kernel = PublishingExpertsKernel(width, ids_ready, host.STAGES[width], slot_base)
    fake = host._fake
    with torch.accelerator.device_index(device_index), _publishing_rows(kernel):
        return cute.compile(
            kernel,
            fake(cutlass.Float4E2M1FN, 32),
            fake(cutlass.Float8E4M3FN),
            fake(cutlass.Float4E2M1FN, 32),
            fake(cutlass.Float8E4M3FN),
            fake(cutlass.Float4E2M1FN),
            fake(cutlass.Uint32),
            fake(cutlass.Int32),
            fake(cutlass.Float32),
            fake(cutlass.Float32),
            fake(cutlass.Float32),
            fake(cutlass.Float4E2M1FN),
            fake(cutlass.Uint8),
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
            for width in host.TILE_WIDTHS:
                if width <= max(8, self.capacity):
                    self._compiled(width, parity)

    def _slot_base(self, parity: int) -> int:
        slot = (parity * self.tp + self.rank) * self.capacity * TOP_K
        return self.multicast + slot * HIDDEN * 2

    def _compiled(self, width: int, parity: int):
        return _kernel(
            self.device_index, width, self.ids_ready, self._slot_base(parity)
        )

    def __call__(  # type: ignore[override]
        self,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        parity: int,
    ):
        """``MuseRoutedExperts.__call__``, publishing at ``parity``."""
        rows = a1q.shape[0]
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed the mailbox's {self.capacity}")
        width = tile_width(rows)
        compiled = self._compiled(width, parity)
        original = host._kernel
        host._kernel = lambda *_: compiled
        try:
            return super().__call__(a1q, a1q_scale, topk_ids, topk_weights, out)
        finally:
            host._kernel = original
