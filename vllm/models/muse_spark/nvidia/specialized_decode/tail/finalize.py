# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The per-route math follows FlashInfer's tp_moe_finalize_rmsnorm
# (flashinfer/comm/tp_moe_finalize_rmsnorm.py), flashinfer-python 0.7.2.post1,
# Copyright (c) 2026 by FlashInfer team, Apache License 2.0.

"""One-shot Lamport finalize of Muse Spark's TP-sharded routed experts.

Every rank holds the same ``M`` tokens and, per route ``(t, k)``, a GEMM2
partial row (the experts are sharded on their intermediate dimension) in the
permuted buffer at ``expanded_idx_to_permuted_idx[t, k]``. One kernel:

1. reads this rank's rows through the permute map (a missing or out-of-range
   slot is zero) and multicasts each 8-column fragment into this rank's slot of
   every rank's ``[3, tp, capacity * 8, 4096]`` BF16 mailbox, route-major;
2. polls the peers' fragments until none holds the negative-zero sentinel,
   re-arms them and sums the ranks' rows in FP32 in rank order, rounded once to
   BF16 (``o``);
3. per route, ``y = (o * (w + gain_center)) * rsqrt(mean((o * (w + gc))**2) +
   eps)`` (``MuseSpark1xRMSNormWithInputScale``), then
   ``e[t] = bf16(sum_k weight[t, k] * y_tk)`` in FP32, in route order.

A cluster of ``cluster`` CTAs owns a token and splits its 4096 columns; every
thread owns one 8-column fragment of one route. The per-route sums of squares
cross the cluster through distributed shared memory and the route sum goes
through shared memory, one column per thread. The grid is fixed (``grid`` CTAs;
clusters loop over
tokens) so that every CTA's generation counter advances on every launch: a
launch reads generation ``g`` (one counter per CTA, read at entry and advanced
at exit, so a captured graph replays correctly) and only touches slots of
``g % 3``. A peer can write generation ``g + 3`` only after it consumed this
rank's publish of ``g + 1``, which this rank issues after this launch (and its
re-arms) completed.

Publishing moves ``M * 8 * 4096 * 2`` bytes out of each rank and three times
that into it, against roughly a quarter of that for FlashInfer's owner-scatter
finalize, which needs a second hop; the one hop wins at small ``M`` only.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    map_shared_to_peer,
    store_shared_cluster_f32,
)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
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

from ..boundary.publishing_gemm import gmem_ptr

HIDDEN = 4096
TOP_K = 8
VEC = 8
FRAGMENTS = HIDDEN // VEC
GENERATIONS = 3
EXPERTS = 256
HALF = TOP_K // 2
SENTINEL = 0x80008000


def _dirty(words) -> cutlass.Boolean:
    """Whether a BF16x8 fragment still holds the negative-zero sentinel."""
    dirty = cutlass.Boolean(False)
    for word in words:
        dirty = dirty | ((word & Uint32(0xFFFF)) == Uint32(0x8000))
        dirty = dirty | ((word >> Uint32(16)) == Uint32(0x8000))
    return dirty


class FinalizeKernel:
    """The finalize of one rank of a ``tp`` group; see the module docstring.

    ``single`` compiles for at most one token per cluster (``M <= grid /
    cluster``), keeping this rank's rows in registers between publish and use.
    ``multicast=False`` stores the fragments only into this rank's own slot (a
    one-GPU stand-in whose peers' slots the caller fills).
    """

    def __init__(
        self,
        *,
        tp: int,
        rank: int,
        capacity: int,
        cluster: int = 8,
        grid: int = 128,
        single: bool = True,
        fp32_weights: bool = True,
        multicast: bool = True,
        published: bool = False,
    ) -> None:
        if FRAGMENTS % cluster or (FRAGMENTS // cluster) % 32:
            raise ValueError(f"cluster={cluster} must split 512 fragments in warps")
        if grid % cluster:
            raise ValueError("grid must be a whole number of clusters")
        self.tp, self.rank = tp, rank
        self.capacity = capacity
        self.cluster, self.grid = cluster, grid
        self.single = single
        # Fragments of a route in one CTA, and warps per route.
        self.band = FRAGMENTS // cluster
        self.route_warps = self.band // 32
        self.threads = TOP_K * self.band
        self.warps = self.threads // 32
        self.fp32_weights = fp32_weights
        self.multicast = multicast
        self.published = published

    def smem_bytes(self) -> int:
        # Half of the routes' normalized rows [HALF, band * VEC] (small enough
        # to share an SM with the publishing GEMM) and two parities of the
        # [cluster, warps] sums of squares.
        return HALF * self.band * VEC * 4 + 2 * self.cluster * self.warps * 4 + 64

    @cute.jit
    def __call__(
        self,
        rows: cute.Pointer,
        permuted_idx: cute.Pointer,
        weights: cute.Pointer,
        norm_weight: cute.Pointer,
        out: cute.Pointer,
        mailbox: cute.Pointer,
        mailbox_multicast: Int64,
        generations: cute.Pointer,
        layer_parity: Int32,
        topk_ids: cute.Pointer,
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
            flat(mailbox, GENERATIONS * self.tp * self.capacity * TOP_K * HIDDEN),
            mailbox_multicast,
            flat(generations, self.grid),
            layer_parity,
            flat(topk_ids, m * TOP_K),
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

    @cute.jit
    def slot(self, generation, source, token, k, column):
        """Element offset of a route's fragment in the ``[3, tp, routes, 4096]``
        mailbox."""
        routes = self.capacity * TOP_K
        return (
            (Int64(generation) * self.tp + source) * routes + Int64(token) * TOP_K + k
        ) * HIDDEN + column

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
    def publishes(self, topk_ids, token, k):
        """Whether this kernel publishes route ``(token, k)``: always, unless
        the experts did; they skip routes to no expert (ids outside [0, 256)),
        whose zero rows this kernel then publishes."""
        publishes = True
        if cutlass.const_expr(self.published):
            expert = topk_ids[token * TOP_K + k]
            publishes = expert < 0 or expert >= EXPERTS
        return publishes

    @cute.jit
    def publish(self, mine, mailbox, multicast, g, token, k, column):
        words = [mine[j] for j in range(4)]
        element = self.slot(g, self.rank, token, k, column)
        if cutlass.const_expr(self.multicast):
            multicast_words(multicast + element * 2, words, kind="bf16x2")
        else:
            store_words(mailbox.iterator + element, words)

    @cute.jit
    def consume(
        self,
        mine,
        weights,
        gamma,
        slots,
        normalized,
        mailbox,
        out,
        g,
        token,
        k,
        column,
        parity,
        eps,
    ):
        """Poll the peers' fragments of route ``(token, k)``, re-arm them, and
        store ``e[token]``'s columns of this CTA."""
        tp, rank, cluster, warps = self.tp, self.rank, self.cluster, self.warps
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        band = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        route_weights = cute.make_rmem_tensor(cute.make_layout((TOP_K,)), Float32)
        for r in cutlass.range_constexpr(TOP_K):
            route_weights[r] = weights[token * TOP_K + r].to(Float32)

        peers = [s for s in range(tp) if s != rank]
        count = len(peers)
        elements = [self.slot(g, peers[i], token, k, column) for i in range(count)]
        fragments = cute.make_rmem_tensor(cute.make_layout((count, 4)), Uint32)
        pending = Int32(0)
        for i in cutlass.range_constexpr(count):
            words = load_vector(mailbox.iterator + elements[i], cache="volatile")
            for j in cutlass.range_constexpr(4):
                fragments[i, j] = words[j]
            if _dirty(words):
                pending = pending | (Int32(1) << i)
        while pending != 0:
            again = Int32(0)
            for i in cutlass.range_constexpr(count):
                if ((pending >> i) & 1) != 0:
                    words = load_vector(
                        mailbox.iterator + elements[i], cache="volatile"
                    )
                    for j in cutlass.range_constexpr(4):
                        fragments[i, j] = words[j]
                    if _dirty(words):
                        again = again | (Int32(1) << i)
            pending = again
        for i in cutlass.range_constexpr(count):
            store_words(mailbox.iterator + elements[i], (Uint32(SENTINEL),) * 4)

        # o: the ranks' rows summed in FP32 in rank order, rounded once.
        scaled = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
        square = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            low, high = Float32(0.0), Float32(0.0)
            for source in cutlass.range_constexpr(tp):
                if cutlass.const_expr(source == rank):
                    word = mine[j]
                else:
                    word = fragments[peers.index(source), j]
                a, b = unpack_bf16(word)
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
            store_words(
                out.iterator + (Int64(token) * HIDDEN + local), (bf16x2(acc, partner),)
            )

    @cute.kernel
    def kernel(
        self,
        rows: cute.Tensor,
        permuted_idx: cute.Tensor,
        weights: cute.Tensor,
        norm_weight: cute.Tensor,
        out: cute.Tensor,
        mailbox: cute.Tensor,
        mailbox_multicast: Int64,
        generations: cute.Tensor,
        layer_parity: Int32,
        topk_ids: cute.Tensor,
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
        # Peers must have started before the first remote shared-memory store.
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
        if cutlass.const_expr(self.published):
            g = Uint32(layer_parity)

        mine = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        own = (rows, permuted_idx, num_rows)
        consume = (weights, gamma, slots, normalized, mailbox, out, g)
        if cutlass.const_expr(self.single):
            if first < m:
                self.load_own(mine, *own, first, k, column)
                if self.publishes(topk_ids, first, k):
                    self.publish(mine, mailbox, mailbox_multicast, g, first, k, column)
            cute.arch.cluster_wait()
            if first < m:
                self.consume(mine, *consume, first, k, column, Int32(0), eps)
        else:
            # Publish every owned token before waiting for any.
            for token in cutlass.range(first, m, clusters, unroll=1):
                if self.publishes(topk_ids, token, k):
                    self.load_own(mine, *own, token, k, column)
                    self.publish(mine, mailbox, mailbox_multicast, g, token, k, column)
            cute.arch.cluster_wait()
            parity = Int32(0)
            for token in cutlass.range(first, m, clusters, unroll=1):
                self.load_own(mine, *own, token, k, column)
                self.consume(mine, *consume, token, k, column, parity, eps)
                parity = parity ^ 1
                # The normalized rows are rewritten next.
                cute.arch.barrier()

        cute.arch.barrier()
        if tidx == 0:
            store_words(generations.iterator + cta, (generation + Uint32(1),))


def compile_finalize(kernel: FinalizeKernel):
    return cute.compile(
        kernel,
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(Int32, 1024),
        gmem_ptr(Float32 if kernel.fp32_weights else BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        Int64(0),
        gmem_ptr(Int32, 1024, 4),
        Int32(0),
        gmem_ptr(Int32, 1024),
        Int32(1),
        Int32(1),
        Float32(1e-6),
        Float32(0.0),
        make_fake_stream(),
    )
