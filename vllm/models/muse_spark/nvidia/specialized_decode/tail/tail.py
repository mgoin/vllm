# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the Muse Spark small-batch MoE tail.

``MoeTail`` replaces, for decode steps of up to ``max_tokens`` rows, the
post-expert finalize (``MusePostExpertFinalize`` / FlashInfer's
``tp_moe_finalize_rmsnorm``), the column-sharded unpooler GEMM and the publish
kernel of the next boundary's all-gather::

    handle = tail(unfinalized, norm_weight, unpooler_shard, eps=..., gain_center=...)
    # the next boundary, the publisher's unchanged consumer:
    new_stream, ffn_input = publisher.gated_residual_next_rmsnorm(handle, ...)

Two PDL-chained kernels: the one-shot Lamport finalize (``finalize.py``) into
a pointer-stable ``[max_tokens, 4096]`` buffer, then the boundary package's
publishing GEMM, which streams the unpooler shard before its grid wait and
multicasts ``e @ unpooler_shard.T`` into the publisher's FlashInfer
all-gather mailbox.

With ``PublishingRoutedExperts`` (``experts.py``) the routed experts' GEMM2
epilogue has already multicast this rank's rows into ``published_box`` at the
layer's ``parity``; ``tail(..., parity=p)`` then starts at the poll.

Construction is collective over the TP group; ``precompile`` must run before
CUDA graph capture. Every rank must call with the same rows; calls launch
nothing but the two kernels and allocate nothing.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

from ..boundary.boundary import (
    AllGatherPublisher,
    LamportMailbox,
    Published,
    PublishingGemm,
)
from ..boundary.publishing_gemm import gmem_ptr
from .finalize import GENERATIONS, HIDDEN, TOP_K, FinalizeKernel, compile_finalize

# Largest decode step the tail beats the stock chain at (B300, TP4, us/layer of
# finalize + unpooler + all-gather publish and consume, tail vs stock: M=1 14.4
# vs 20.1, M=4 14.5 vs 22.4, M=8 15.3 vs 24.8, M=16 19.0 vs 23.5). Its one-shot
# publish moves three times the bytes of FlashInfer's owner-scatter finalize
# into each rank, which loses from about 32 rows on.
MAX_TOKENS = 16
CLUSTER = 8
GRID = 128


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def layer_parities(layers: int) -> list[int]:
    """Mailbox generations of consecutive layers for rows the experts publish:
    every layer's differs from the previous one's, also from the last layer to
    the first of the next step (a rank is at most one layer ahead of another)."""
    parities = [i % GENERATIONS for i in range(layers)]
    if layers > 1 and parities[-1] == parities[0]:
        parities[-1] = 1
    return parities


class Finalize:
    """The compiled finalize kernels of one rank over a mailbox it is handed:
    ``mailbox`` is the ``[3, tp, capacity * 8, 4096]`` BF16 buffer and
    ``multicast`` its multicast address (0 for the one-GPU stand-in, which
    stores only this rank's slot). With ``published`` the rows are already in
    the mailbox at the parity of each call (see ``finalize.py``)."""

    def __init__(
        self,
        *,
        tp: int,
        rank: int,
        capacity: int,
        mailbox: torch.Tensor,
        multicast: int,
        published: bool = False,
        cluster: int = CLUSTER,
        grid: int = GRID,
    ) -> None:
        if mailbox.shape != (GENERATIONS, tp, capacity * TOP_K, HIDDEN):
            raise ValueError("unexpected mailbox shape")
        self.tp, self.rank, self.capacity = tp, rank, capacity
        self.mailbox, self.multicast = mailbox, multicast
        self.published = published
        self.cluster, self.grid = cluster, grid
        self.generations = torch.zeros(grid, dtype=torch.int32, device=mailbox.device)
        self._compiled: dict[tuple[bool, bool], Any] = {}

    def _kernel(self, fp32_weights: bool, m: int):
        single = m <= self.grid // self.cluster
        compiled = self._compiled.get((fp32_weights, single))
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("precompile the MoE tail before graph capture")
            compiled = self._compiled[fp32_weights, single] = compile_finalize(
                FinalizeKernel(
                    tp=self.tp,
                    rank=self.rank,
                    capacity=self.capacity,
                    cluster=self.cluster,
                    grid=self.grid,
                    single=single,
                    fp32_weights=fp32_weights,
                    multicast=bool(self.multicast),
                    published=self.published,
                )
            )
        return compiled

    def precompile(self, weight_dtypes=(torch.float32, torch.bfloat16)) -> None:
        for dtype in weight_dtypes:
            for m in {1, self.capacity}:
                self._kernel(dtype == torch.float32, m)

    def __call__(
        self,
        gemm2_permuted: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        expert_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        out: torch.Tensor,
        *,
        eps: float,
        gain_center: float,
        parity: int = 0,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``e`` [M, 4096] BF16 into ``out``; see ``finalize.py``. Published
        rows need the routing's ``topk_ids``: the experts skip routes to no
        expert, whose (zero) rows the finalize publishes itself."""
        m = expanded_idx_to_permuted_idx.shape[0]
        if self.published:
            if topk_ids is None or topk_ids.shape != (m, TOP_K):
                raise ValueError("published rows need the [M, 8] topk_ids")
            if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
                raise ValueError("topk_ids must be contiguous int32")
        else:
            topk_ids = expanded_idx_to_permuted_idx
        if not 1 <= m <= self.capacity:
            raise ValueError(f"M={m} is outside [1, {self.capacity}]")
        if (
            gemm2_permuted.dtype != torch.bfloat16
            or gemm2_permuted.shape[-1] != HIDDEN
            or expanded_idx_to_permuted_idx.shape != (m, TOP_K)
            or expanded_idx_to_permuted_idx.dtype != torch.int32
            or expert_weights.numel() != m * TOP_K
            or expert_weights.dtype not in (torch.float32, torch.bfloat16)
            or norm_weight.shape != (HIDDEN,)
            or norm_weight.dtype != torch.bfloat16
            or out.shape[0] < m
            or out.shape[-1] != HIDDEN
            or out.dtype != torch.bfloat16
            or not 0 <= parity < GENERATIONS
        ):
            raise ValueError("unexpected MoE tail operands")
        for t in (
            gemm2_permuted,
            expanded_idx_to_permuted_idx,
            expert_weights,
            norm_weight,
            out,
        ):
            if not t.is_contiguous() or t.data_ptr() % 16:
                raise ValueError("MoE tail operands must be 16-byte aligned")
        fp32 = expert_weights.dtype == torch.float32
        self._kernel(fp32, m)(
            gmem_ptr(BFloat16, gemm2_permuted.data_ptr()),
            gmem_ptr(Int32, expanded_idx_to_permuted_idx.data_ptr()),
            gmem_ptr(Float32 if fp32 else BFloat16, expert_weights.data_ptr()),
            gmem_ptr(BFloat16, norm_weight.data_ptr()),
            gmem_ptr(BFloat16, out.data_ptr()),
            gmem_ptr(BFloat16, self.mailbox.data_ptr()),
            Int64(self.multicast),
            gmem_ptr(Int32, self.generations.data_ptr(), 4),
            Int32(parity),
            gmem_ptr(Int32, topk_ids.data_ptr(), 4),
            Int32(gemm2_permuted.shape[0]),
            Int32(m),
            Float32(eps),
            Float32(gain_center),
            _stream(),
        )
        return out[:m]


class MoeTail:
    """The MoE tail of a TP group; see the module docstring.

    ``publisher`` is the executor's ``AllGatherPublisher`` of the next
    boundary's FlashInfer ``MNNVLCuteDSLAllGatherDualNormWorkspace``;
    ``__call__`` returns its handle for its consumer
    (``gated_residual_next_rmsnorm``). Without one, ``__call__`` returns this
    rank's plain ``[M, n]`` unpooler block (for the stock all-gather) instead.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        publisher: AllGatherPublisher | None = None,
        shard_columns: int = 2048,
        max_tokens: int = MAX_TOKENS,
        device=None,
    ) -> None:
        device = torch.device(device or torch.accelerator.current_device_index())
        self.group = group
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.max_tokens = max_tokens
        shape = (GENERATIONS, self.tp, max_tokens * TOP_K, HIDDEN)
        self.box = LamportMailbox(group, shape, device)
        # Rows the experts publish (``PublishingRoutedExperts``), by layer
        # parity rather than by the finalize's own generations.
        self.published_box = LamportMailbox(group, shape, device)
        common = dict(tp=self.tp, rank=self.rank, capacity=max_tokens)
        self.finalize = Finalize(
            **common, mailbox=self.box.buffer, multicast=self.box.multicast
        )
        self.published_finalize = Finalize(
            **common,
            mailbox=self.published_box.buffer,
            multicast=self.published_box.multicast,
            published=True,
        )
        self.e = torch.empty(max_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
        self.publisher = publisher
        self.gemm: PublishingGemm | None = None
        if publisher is None:
            self.gemm = PublishingGemm(
                shard_columns, HIDDEN, None, max_tokens=max_tokens, store_c=True
            )
        elif publisher.gemm.k != HIDDEN or max_tokens > publisher.impl.capacity_m:
            raise ValueError("the publisher does not take this tail's rows")
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    @property
    def published_target(self) -> tuple[int, int, int]:
        """``(multicast address, rank, capacity)`` of the mailbox the experts
        publish their rows into."""
        return self.published_box.multicast, self.rank, self.max_tokens

    def precompile(self, weight_dtypes=(torch.float32, torch.bfloat16)) -> None:
        self.finalize.precompile(weight_dtypes)
        self.published_finalize.precompile(weight_dtypes)
        (self.publisher or self.gemm).precompile()

    def reset(self) -> None:
        """Re-arm the mailboxes after an aborted step (collective)."""
        self.box.arm()
        self.published_box.arm()
        self.finalize.generations.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)

    def supports(self, m: int) -> bool:
        return 1 <= m <= self.max_tokens

    def __call__(
        self,
        unfinalized: UnfinalizedMoEOutput,
        norm_weight: torch.Tensor,
        unpooler_shard: torch.Tensor,
        *,
        eps: float,
        gain_center: float,
        parity: int | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> Published | torch.Tensor:
        """Finalize ``unfinalized`` and publish ``e @ unpooler_shard.T`` into
        the all-gather workspace (the publisher's consumer takes the handle),
        or without a publisher return it. ``parity``: the experts already
        published the rows at this layer parity (``layer_parities``), routed
        by ``topk_ids``."""
        finalize = self.finalize if parity is None else self.published_finalize
        if parity is not None and not unfinalized.route_major:
            raise ValueError("published rows are route-major")
        e = finalize(
            unfinalized.gemm2_permuted,
            unfinalized.expanded_idx_to_permuted_idx,
            unfinalized.expert_weights,
            norm_weight,
            self.e,
            eps=eps,
            gain_center=gain_center,
            parity=parity or 0,
            topk_ids=topk_ids,
        )
        if self.publisher is not None:
            return self.publisher(e, unpooler_shard)
        assert self.gemm is not None
        branch = e.new_empty(e.shape[0], unpooler_shard.shape[0])
        self.gemm(e, unpooler_shard, branch)
        return branch
