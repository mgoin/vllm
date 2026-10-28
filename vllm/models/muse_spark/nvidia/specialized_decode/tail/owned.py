# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The per-route math follows FlashInfer's tp_moe_finalize_rmsnorm
# (flashinfer/comm/tp_moe_finalize_rmsnorm.py), flashinfer-python 0.7.2.post1,
# Copyright (c) 2026 by FlashInfer team, Apache License 2.0.

"""The Muse Spark MoE tail's finalize: FlashInfer's owner-scatter schedule.

Multicasting every rank's ``M * 8`` route rows into every rank (the schedule
this replaces) makes each rank receive ``3/4`` of the other ranks' partial
rows, ``3 * M * 64 KB`` per layer, and dies from 24 rows on. This kernel splits
the work instead, as FlashInfer's ``tp_moe_finalize_rmsnorm`` does:

1. **hop 1, scatter by token owner.** Rank ``owner(t) = t % tp`` owns token
   ``t``. Every rank pushes its partial rows for those tokens -- unicast -- into
   the owner's Lamport slot ``(t // tp) * 8 + k`` of its own source plane. Each
   rank sends ``3/4`` of ``M * 64 KB`` and receives the same.
2. **local finalize of the owned tokens.** The owner sums the four ranks' rows
   in rank order in FP32, rounds once to BF16, runs the input-scaled per-route
   RMSNorm over the full 4096 columns and the FP32 weighted sum over the eight
   routes -- the arithmetic of FlashInfer's finalize -- and multicasts the
   token's BF16 row into every rank's ``e`` mailbox, plane ``owner(t)``, row
   ``t``. The second hop is ``M / 4 * 8 KB`` out and ``3 * M / 4 * 8 KB`` in.
3. **copy.** Every rank copies the rows of the tokens it does not own out of its
   own ``e`` plane into its output block; the consumers' GEMM reads a
   contiguous ``[M, 4096]`` activation.

That is ``M * 54 KB`` received per rank against ``M * 192 KB`` for the
one-shot multicast, and the second hop carries BF16 rows rather than FP32
partial sums. Measured per layer at TP4 (B300, CUDA graph) the finalize is
8.7 us at 8 rows, 9.9 at 32 and 15.6 at 64, against 13.6, 12.8 and 18.7 for
the stock ``tp_moe_finalize_rmsnorm``; one schedule serves every row count, so
there is no gate to get wrong.

Lamport protocol: the hop-1 slots are armed with the negative-zero sentinel and
each reader re-arms the four slots of the token it consumed (its own plane
included, whose slot a sibling CTA of the same rank filled). The ``e`` rows are
written by their owner and read by the other three ranks, so the *writer* arms
its own plane's rows of generation ``g + 1`` (== ``g - 2``, two launches back) at
the start of each launch: a rank at generation ``g`` has necessarily completed
``g - 1``, which required every peer's push of ``g - 1``, so no reader is still
looking at generation ``g - 2``. A launch reads generation ``g`` and only touches
slots of ``g % 3``; the per-CTA counter is read at entry and advanced at exit, so
a captured graph replays correctly.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    map_shared_to_peer,
    store_shared_cluster_f32,
)

from vllm.models.common.specialized_decode.primitives import (
    add_rn,
    bf16x2,
    fma,
    load_vector,
    load_words,
    multicast_words,
    no_neg_zero,
    store_words,
    unpack_bf16,
)

from ..boundary.primitives import load_u64_at, store_global_u32x4_at
from ..boundary.publishing_gemm import gmem_ptr

HIDDEN = 4096
TOP_K = 8
VEC = 8
FRAGMENTS = HIDDEN // VEC
HALF = TOP_K // 2
# Lamport generations of both mailboxes; ``g % GENERATIONS`` selects one.
GENERATIONS = 3
SENTINEL = 0x80008000
CLUSTER = 8
GRID = 128


def _dirty(words) -> cutlass.Boolean:
    """Whether a BF16x8 fragment still holds the negative-zero sentinel."""
    dirty = cutlass.Boolean(False)
    for word in words:
        dirty = dirty | ((word & Uint32(0xFFFF)) == Uint32(0x8000))
        dirty = dirty | ((word >> Uint32(16)) == Uint32(0x8000))
    return dirty


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


class OwnedFinalizeKernel:
    """The owner-scatter finalize of one rank of a ``tp`` group.

    ``hop1`` is this rank's ``[3, tp, capacity_tp * 8, 4096]`` BF16 mailbox
    (``capacity_tp = ceil(capacity / tp)``), ``peers`` the device table of every
    rank's ``hop1`` address, ``e`` this rank's ``[3, tp, capacity, 4096]`` BF16
    ``e`` mailbox and ``e_multicast`` its multicast address (0 stores only into
    this rank's own plane, for the one-GPU stand-in).
    """

    def __init__(
        self,
        *,
        tp: int,
        rank: int,
        capacity: int,
        cluster: int = CLUSTER,
        grid: int = GRID,
        fp32_weights: bool = True,
        multicast: bool = True,
    ) -> None:
        if FRAGMENTS % cluster or (FRAGMENTS // cluster) % 32:
            raise ValueError(f"cluster={cluster} must split 512 fragments in warps")
        if grid % cluster:
            raise ValueError("grid must be a whole number of clusters")
        self.tp, self.rank = tp, rank
        self.capacity = capacity
        self.capacity_tp = -(-capacity // tp)
        self.cluster, self.grid = cluster, grid
        # Fragments of a route in one CTA, and warps per route.
        self.band = FRAGMENTS // cluster
        self.route_warps = self.band // 32
        self.threads = TOP_K * self.band
        self.warps = self.threads // 32
        self.fp32_weights = fp32_weights
        self.multicast = multicast

    def smem_bytes(self) -> int:
        # Half of the routes' normalized rows [HALF, band * VEC], small enough
        # to share an SM with the publishing GEMM, and two parities of the
        # [cluster, warps] sums of squares.
        return HALF * self.band * VEC * 4 + 2 * self.cluster * self.warps * 4 + 64

    @cute.jit
    def load_own(self, mine, rows, permuted_idx, num_rows, token, k, column):
        """This rank's fragment of route ``(token, k)`` into ``mine``, negative
        zero made +0; a missing route reads zero without touching its (possibly
        NaN) fallback row."""
        mine.fill(Uint32(0))
        row = permuted_idx[token * TOP_K + k]
        if row >= 0 and row < num_rows:
            words = load_vector(rows.iterator + (Int64(row) * HIDDEN + column))
            for j in cutlass.range_constexpr(4):
                mine[j] = no_neg_zero(words[j])

    @cute.jit
    def hop1_slot(self, generation, source, t_local, k, column):
        """Element offset of a route's fragment in the ``[3, tp, tokens, 8,
        4096]`` hop-1 mailbox."""
        routes = self.capacity_tp * TOP_K
        return (
            (Int64(generation) * self.tp + source) * routes + Int64(t_local) * TOP_K + k
        ) * HIDDEN + column

    @cute.jit
    def e_slot(self, generation, owner, token, column):
        """Element offset of a token's row in the ``[3, tp, tokens, 4096]``
        ``e`` mailbox."""
        return (
            (Int64(generation) * self.tp + owner) * self.capacity + Int64(token)
        ) * HIDDEN + column

    @cute.jit
    def scatter(
        self, mine, rows, permuted_idx, num_rows, hop1, peers, g, token, k, column
    ):
        """Push this rank's fragment of route ``(token, k)`` into the owner's
        slot of the hop-1 mailbox."""
        self.load_own(mine, rows, permuted_idx, num_rows, token, k, column)
        owner = Int64(token % self.tp)
        base = load_u64_at(peers + owner * 8)
        element = self.hop1_slot(g, self.rank, token // self.tp, k, column)
        store_global_u32x4_at(
            Int64(base) + element * 2, (mine[0], mine[1], mine[2], mine[3])
        )

    @cute.jit
    def own_row(
        self,
        hop1,
        weights,
        gamma,
        slots,
        normalized,
        out,
        ebox,
        ebox_multicast,
        g,
        previous,
        token,
        k,
        column,
        parity,
        eps,
    ):
        """Poll the four sources of an owned token, finalize it and publish its
        row: locally for the copy phase, and into every rank's ``e`` mailbox."""
        tp, rank, cluster, warps = self.tp, self.rank, self.cluster, self.warps
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        band = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        t_local = token // tp
        route_weights = cute.make_rmem_tensor(cute.make_layout((TOP_K,)), Float32)
        for r in cutlass.range_constexpr(TOP_K):
            route_weights[r] = weights[token * TOP_K + r].to(Float32)

        # The peers' rows may still be in flight; a slot that still reads as the
        # sentinel has not landed. Re-arm every source once it has been read.
        elements = [self.hop1_slot(g, s, t_local, k, column) for s in range(tp)]
        fragments = cute.make_rmem_tensor(cute.make_layout((tp, 4)), Uint32)
        pending = Int32(0)
        for i in cutlass.range_constexpr(tp):
            words = load_vector(hop1.iterator + elements[i], cache="volatile")
            for j in cutlass.range_constexpr(4):
                fragments[i, j] = words[j]
            if _dirty(words):
                pending = pending | (Int32(1) << i)
        while pending != 0:
            again = Int32(0)
            for i in cutlass.range_constexpr(tp):
                if ((pending >> i) & 1) != 0:
                    words = load_vector(hop1.iterator + elements[i], cache="volatile")
                    for j in cutlass.range_constexpr(4):
                        fragments[i, j] = words[j]
                    if _dirty(words):
                        again = again | (Int32(1) << i)
            pending = again
        for i in cutlass.range_constexpr(tp):
            store_words(hop1.iterator + elements[i], (Uint32(SENTINEL),) * 4)

        # This rank's own plane of the row two launches back is about to be
        # rewritten by its owner (this rank) in two launches: arm it now.
        previous_element = self.e_slot(previous, rank, token, column)
        if cutlass.const_expr(self.multicast):
            multicast_words(
                ebox_multicast + previous_element * 2,
                (Uint32(SENTINEL),) * 4,
                kind="bf16x2",
            )
        else:
            store_words(ebox.iterator + previous_element, (Uint32(SENTINEL),) * 4)

        # o: the four ranks' rows summed in FP32 in rank order, rounded once.
        scaled = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            low, high = Float32(0.0), Float32(0.0)
            for source in cutlass.range_constexpr(tp):
                a, b = unpack_bf16(fragments[source, j])
                low, high = add_rn(low, a), add_rn(high, b)
            low, high = unpack_bf16(bf16x2(low, high))
            low, high = low * gamma[2 * j], high * gamma[2 * j + 1]
            scaled[2 * j], scaled[2 * j + 1] = low, high
            square = fma(low, low, square)
            square = fma(high, high, square)
        for offset in (16, 8, 4, 2, 1):
            square = square + cute.arch.shuffle_sync_bfly(square, offset)

        # Every CTA of the cluster gets every (CTA, warp) partial.
        table = slots + parity * cluster * warps
        if lane < cluster:
            own = table + band * warps + warp
            store_shared_cluster_f32(map_shared_to_peer(own, lane), square)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        total = Float32(0.0)
        for b in cutlass.range_constexpr(cluster):
            for w in cutlass.range_constexpr(self.route_warps):
                slot = b * warps + k * self.route_warps + w
                total = total + cute.arch.load((table + slot).llvm_ptr, Float32)
        inverse = cute.rsqrt(total * Float32(1.0 / HIDDEN) + eps)

        # The weighted route sum, one column per thread, in route order, one
        # half of the routes at a time.
        fragment = tidx % self.band
        acc = Float32(0.0)
        for half in cutlass.range_constexpr(TOP_K // HALF):
            if half > 0:
                cute.arch.barrier()
            if k // HALF == half:
                for j in cutlass.range_constexpr(VEC):
                    normalized[k % HALF, fragment * VEC + j] = scaled[j] * inverse
            cute.arch.barrier()
            for r in cutlass.range_constexpr(HALF):
                acc = fma(normalized[r, tidx], route_weights[half * HALF + r], acc)
        partner = cute.arch.shuffle_sync_bfly(acc, 1)
        if lane % 2 == 0:
            local = Int64(band) * self.band * VEC + tidx
            pair = (bf16x2(acc, partner),)
            store_words(out.iterator + (Int64(token) * HIDDEN + local), pair)
            element = self.e_slot(g, rank, token, local)
            if cutlass.const_expr(self.multicast):
                multicast_words(ebox_multicast + element * 2, pair, kind="bf16x2")
            else:
                store_words(ebox.iterator + element, pair)

    @cute.jit
    def copy_row(self, ebox, out, g, token):
        """Poll an owner's ``e`` row, copy it into ``out`` and re-arm it."""
        tidx, _, _ = cute.arch.thread_idx()
        owner = Int64(token % self.tp)
        element = self.e_slot(g, owner, token, Int64(tidx) * VEC)
        words = load_vector(ebox.iterator + element, cache="volatile")
        while _dirty(words):
            words = load_vector(ebox.iterator + element, cache="volatile")
        store_words(ebox.iterator + element, (Uint32(SENTINEL),) * 4)
        store_words(out.iterator + (Int64(token) * HIDDEN + tidx * VEC), words)

    @cute.kernel
    def kernel(
        self,
        rows: cute.Tensor,
        permuted_idx: cute.Tensor,
        weights: cute.Tensor,
        norm_weight: cute.Tensor,
        out: cute.Tensor,
        hop1: cute.Tensor,
        peers: Int64,
        ebox: cute.Tensor,
        ebox_multicast: Int64,
        generations: cute.Tensor,
        owned_tokens: Int32,
        num_rows: Int32,
        m: Int32,
        eps: Float32,
        gain_center: Float32,
    ) -> None:
        cluster = self.cluster
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        band = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        k = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // self.route_warps
        first = cta // cluster
        clusters = self.grid // cluster
        column = Int64(band * self.band + tidx % self.band) * VEC
        # Peers must have started before the first remote store.
        cute.arch.cluster_arrive_relaxed()

        smem = cutlass.utils.SmemAllocator()
        width = self.band * VEC
        normalized = smem.allocate_tensor(
            Float32, cute.make_layout((HALF, width), stride=(width, 1)), 16
        )
        slots = smem.allocate_array(Float32, 2 * cluster * self.warps)

        # The norm weight does not depend on the previous kernel.
        gamma = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
        gamma_words = load_words(norm_weight.iterator + column, count=4)
        for j in cutlass.range_constexpr(4):
            low, high = unpack_bf16(gamma_words[j])
            gamma[2 * j] = low + gain_center
            gamma[2 * j + 1] = high + gain_center

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        generation = load_words(
            generations.iterator + cta, count=1, volatile=True, word_type="u32"
        )[0]
        g = generation % Uint32(GENERATIONS)
        previous = (generation + Uint32(GENERATIONS - 2)) % Uint32(GENERATIONS)

        mine = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        consume = (
            hop1,
            weights,
            gamma,
            slots,
            normalized,
            out,
            ebox,
            ebox_multicast,
            g,
            previous,
        )
        # Publish every owned token before waiting for any.
        for token in cutlass.range(first, m, clusters, unroll=1):
            self.scatter(
                mine, rows, permuted_idx, num_rows, hop1, peers, g, token, k, column
            )
        # The tokens this rank owns: t = rank + j * tp, j < owned_tokens.
        parity = Int32(0)
        for j in cutlass.range(first, owned_tokens, clusters, unroll=1):
            token = j * self.tp + self.rank
            self.own_row(*consume, token, k, column, parity, eps)
            parity = parity ^ 1
            # The normalized rows are rewritten next.
            cute.arch.barrier()
        # Every rank needs every row: copy the ones this rank does not own.
        for token in cutlass.range(cta, m, self.grid, unroll=1):
            if token % self.tp != self.rank:
                self.copy_row(ebox, out, g, token)

        cute.arch.barrier()
        if tidx == 0:
            store_words(generations.iterator + cta, (generation + Uint32(1),))

    @cute.jit
    def __call__(
        self,
        rows: cute.Pointer,
        permuted_idx: cute.Pointer,
        weights: cute.Pointer,
        norm_weight: cute.Pointer,
        out: cute.Pointer,
        hop1: cute.Pointer,
        peers: Int64,
        ebox: cute.Pointer,
        ebox_multicast: Int64,
        generations: cute.Pointer,
        owned_tokens: Int32,
        num_rows: Int32,
        m: Int32,
        eps: Float32,
        gain_center: Float32,
        stream: cuda.CUstream,
    ) -> None:
        flat = lambda pointer, size: cute.make_tensor(pointer, cute.make_layout(size))
        self.kernel(
            flat(rows, num_rows * HIDDEN),
            flat(permuted_idx, m * TOP_K),
            flat(weights, m * TOP_K),
            flat(norm_weight, HIDDEN),
            flat(out, m * HIDDEN),
            flat(hop1, GENERATIONS * self.tp * self.capacity_tp * TOP_K * HIDDEN),
            peers,
            flat(ebox, GENERATIONS * self.tp * self.capacity * HIDDEN),
            ebox_multicast,
            flat(generations, self.grid),
            owned_tokens,
            num_rows,
            m,
            eps,
            gain_center,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.cluster, 1, 1),
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=True,
            # 64 registers a thread: the publishing GEMM shares the SM.
            min_blocks_per_mp=2,
        )


def compile_owned(kernel: OwnedFinalizeKernel):
    return cute.compile(
        kernel,
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(Int32, 1024),
        gmem_ptr(Float32 if kernel.fp32_weights else BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        Int64(0),
        gmem_ptr(BFloat16, 1024),
        Int64(0),
        gmem_ptr(Int32, 1024, 4),
        Int32(1),
        Int32(1),
        Int32(1),
        Float32(1e-6),
        Float32(0.0),
        make_fake_stream(),
    )


class OwnedFinalize:
    """The compiled owner-scatter finalize kernels of one rank over the
    mailboxes it is handed: ``hop1`` (this rank's ``[3, tp, capacity_tp * 8,
    4096]`` BF16 buffer), ``peers`` (the device table of every rank's ``hop1``
    address), ``e`` (this rank's ``[3, tp, capacity, 4096]`` BF16 buffer) and
    ``e_multicast`` (0 for the one-GPU stand-in, which stores only into this
    rank's own plane and whose peers' planes and rows the caller fills)."""

    def __init__(
        self,
        *,
        tp: int,
        rank: int,
        capacity: int,
        hop1: torch.Tensor,
        peers: torch.Tensor,
        e: torch.Tensor,
        multicast: int,
        cluster: int = 8,
        grid: int = 128,
    ) -> None:
        capacity_tp = -(-capacity // tp)
        if hop1.shape != (GENERATIONS, tp, capacity_tp * TOP_K, HIDDEN):
            raise ValueError("unexpected hop-1 mailbox shape")
        if e.shape != (GENERATIONS, tp, capacity, HIDDEN):
            raise ValueError("unexpected e mailbox shape")
        if peers.shape != (tp,) or peers.dtype != torch.int64:
            raise ValueError("peers must be the [tp] int64 mailbox table")
        self.tp, self.rank, self.capacity = tp, rank, capacity
        self.hop1, self.peers = hop1, peers
        self.e, self.multicast = e, multicast
        self.cluster, self.grid = cluster, grid
        self.generations = torch.zeros(grid, dtype=torch.int32, device=hop1.device)
        self._compiled: dict[bool, Any] = {}

    def _kernel(self, fp32_weights: bool):
        compiled = self._compiled.get(fp32_weights)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("precompile the MoE tail before graph capture")
            compiled = self._compiled[fp32_weights] = compile_owned(
                OwnedFinalizeKernel(
                    tp=self.tp,
                    rank=self.rank,
                    capacity=self.capacity,
                    cluster=self.cluster,
                    grid=self.grid,
                    fp32_weights=fp32_weights,
                    multicast=bool(self.multicast),
                )
            )
        return compiled

    def precompile(self, weight_dtypes=(torch.float32, torch.bfloat16)) -> None:
        for dtype in weight_dtypes:
            self._kernel(dtype == torch.float32)

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
    ) -> torch.Tensor:
        m = expanded_idx_to_permuted_idx.shape[0]
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
        self._kernel(fp32)(
            gmem_ptr(BFloat16, gemm2_permuted.data_ptr()),
            gmem_ptr(Int32, expanded_idx_to_permuted_idx.data_ptr()),
            gmem_ptr(Float32 if fp32 else BFloat16, expert_weights.data_ptr()),
            gmem_ptr(BFloat16, norm_weight.data_ptr()),
            gmem_ptr(BFloat16, out.data_ptr()),
            gmem_ptr(BFloat16, self.hop1.data_ptr()),
            Int64(self.peers.data_ptr()),
            gmem_ptr(BFloat16, self.e.data_ptr()),
            Int64(self.multicast),
            gmem_ptr(Int32, self.generations.data_ptr(), 4),
            Int32((m - self.rank + self.tp - 1) // self.tp),
            Int32(gemm2_permuted.shape[0]),
            Int32(m),
            Float32(eps),
            Float32(gain_center),
            _stream(),
        )
        return out[:m]
