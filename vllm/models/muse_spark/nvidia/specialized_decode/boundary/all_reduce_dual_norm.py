# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from FlashInfer's _AllGatherDualNormDeviceKernel
# (flashinfer/comm/mnnvl_cutedsl/kernel_ll/all_gather_dual_norm.py) and the
# row reduction of _dual_norm_kernel
# (flashinfer/norm/kernels/gated_residual_next_rmsnorm.py), flashinfer-python
# 0.7.2.post1, Copyright (c) 2026 by FlashInfer team, Apache License 2.0.

"""Lamport all-reduce of the attention output fused into Muse Spark's boundary.

The consumer of :class:`~.publishing_gemm.PublishingGemmKernel`'s mailbox
(FlashInfer's LL layout ``[3, tp, capacity, hidden]``). One-shot: every rank
polls all ranks' BF16 partials of its rows until no 16-byte fragment holds the
negative-zero sentinel. Two-shot (the GEMM scattered each ``hidden / tp``
column block to its owner rank): the owner polls the ranks' partials of its
block and multicasts their reduced BF16 sum into a ``[3, capacity, hidden]``
gather mailbox, which every rank then polls. Either way the partials are
summed in FP32 in rank order and rounded to BF16, as the unfused all-reduce
returns it, every consumed slot is re-armed and the generation advances like
FlashInfer's LL consumers. The kernel then computes
``gated_residual_next_rmsnorm`` of that branch::

    b = branch (* (post_weight + post_gain_center))
    new_stream = alpha * stream + (beta * b) * rsqrt(mean(b**2) + post_eps)
    router = new_stream * rsqrt(mean(new_stream**2) + next_eps)
             * (next_weight + next_gain_center)
    out = bf16(router)

with FlashInfer's thread mapping (a cluster of 8 x 128 threads, 8 columns
each, per row; a cluster handles ``tokens`` rows) and its exact reduction
tree, so for the same branch the outputs are bit-identical to
``gated_residual_next_rmsnorm`` (and ``router`` to its ``router_input``).

The two-shot grid stays within one wave (a cluster takes several rows), and
each CTA reduces all its owned rows before it waits for any gathered one, so
every row's owner work runs whatever the wait order.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    bf16x8_to_packed_u32x4,
    fragment_has_negative_zero,
    load_bf16x8_as_f32,
    load_f32x8,
    load_global_u32x4,
    load_volatile_u32,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    store_f32x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
)

from .primitives import fma_f32, multicast_u32x4
from .publishing_gemm import ACTIVE_STAGE, NEXT_STAGE, gmem_ptr

LAMPORT_GENERATIONS = 3
EARLY = bool(int(__import__("os").environ.get("EARLY", "1")))
HIDDEN = 8192
CLUSTER = 8
THREADS = 128
VEC = 8
WARPS = THREADS // 32
SLOTS = CLUSTER * WARPS
# Resident CTAs per SM the register budget must allow: a two-shot grid of
# 64 clusters (512 CTAs) fits 148 SMs in one wave.
MIN_BLOCKS_PER_SM = 4
assert CLUSTER * THREADS * VEC == HIDDEN and SLOTS == 32


@dataclass(frozen=True)
class DualNormConstants:
    post_eps: float
    next_eps: float
    next_gain_center: float
    has_post_weight: bool = False
    post_gain_center: float = 0.0
    router: bool = False


@cute.jit
def _row_sums(values, slots: cute.Pointer, cluster_rank: Int32):
    """FlashInfer's dual-norm row sum of each of ``values`` (one per row): a
    warp butterfly, one slot per (CTA, warp) in every CTA of the cluster, then a
    butterfly over the 32 slots."""
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    count = len(values)
    for i in cutlass.range_constexpr(count):
        value = values[i]
        for offset in (16, 8, 4, 2, 1):
            value = value + cute.arch.shuffle_sync_bfly(value, offset=offset)
        if lane < CLUSTER:
            slot = slots + (i * SLOTS + cluster_rank * WARPS + warp)
            store_shared_cluster_f32(map_shared_to_peer(slot, lane), value)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    totals = []
    for i in cutlass.range_constexpr(count):
        total = cute.arch.load((slots + (i * SLOTS + lane)).llvm_ptr, Float32)
        for offset in (16, 8, 4, 2, 1):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=offset)
        totals.append(total)
    return totals


@cute.jit
def _poll(mailbox: cute.Tensor, elements: list, valid: list, fragments: cute.Tensor):
    """Spin until the 16-byte fragment at each valid element is published,
    then re-arm the slots. A pass loads every unpublished fragment before
    checking any; between passes only one fragment is polled, which keeps the
    spin from flooding the memory system the peers write into."""
    count = len(elements)
    pointers = [
        cute.make_ptr(
            BFloat16,
            (mailbox.iterator + element).llvm_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for element in elements
    ]
    pending = Int32(0)
    for i in cutlass.range_constexpr(count):
        if valid[i]:
            pending = pending | (Int32(1) << i)
    while pending != 0:
        for i in cutlass.range_constexpr(count):
            if ((pending >> i) & 1) != 0:
                fragment = load_global_u32x4(pointers[i], volatile=True)
                for word in cutlass.range_constexpr(4):
                    fragments[i, word] = fragment[word]
        again = Int32(0)
        for i in cutlass.range_constexpr(count):
            if ((pending >> i) & 1) != 0:
                words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    words[word] = fragments[i, word]
                if fragment_has_negative_zero(words.load()):
                    again = again | (Int32(1) << i)
        pending = again
        # Wait for the first unpublished fragment alone.
        waiting = pending
        for i in cutlass.range_constexpr(count):
            if ((waiting >> i) & 1) != 0:
                dirty = waiting != 0
                while dirty:
                    fragment = load_global_u32x4(pointers[i], volatile=True)
                    dirty = fragment_has_negative_zero(fragment)
                waiting = Int32(0)
    for i in cutlass.range_constexpr(count):
        if valid[i]:
            store_lamport_sentinel_u32x4(Int64(pointers[i].toint()))


@cute.jit
def _rank_sum(
    fragments: cute.Tensor, first: cutlass.Constexpr[int], count: cutlass.Constexpr[int]
):
    """FP32 sum, in rank order, of ``count`` BF16x8 fragments."""
    total = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
    total.fill(Float32(0.0))
    for source in cutlass.range_constexpr(count):
        words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for word in cutlass.range_constexpr(4):
            words[word] = fragments[first + source, word]
        total.store(total.load() + packed_u32x4_to_bf16x8(words.load()).to(Float32))
    return total.load()


class AllReduceDualNormKernel:
    """The consumer for one TP group, mailbox capacity, constant set and
    protocol; a cluster handles ``tokens`` rows."""

    def __init__(
        self,
        *,
        tp: int,
        capacity: int,
        constants: DualNormConstants,
        two_shot: bool = False,
        tokens: int = 1,
    ):
        self.tp = tp
        self.capacity = capacity
        self.constants = constants
        self.two_shot = two_shot
        self.tokens = tokens
        self.chunk = HIDDEN // tp

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Pointer,
        gather: cute.Pointer,
        gather_multicast: Int64,
        stage_state: cute.Pointer,
        stream_in: cute.Pointer,
        alpha: cute.Pointer,
        beta: cute.Pointer,
        next_weight: cute.Pointer,
        post_weight: cute.Pointer,
        new_stream: cute.Pointer,
        out: cute.Pointer,
        router: cute.Pointer,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ) -> None:
        rows = cute.make_layout(m * HIDDEN)
        vector = cute.make_layout(HIDDEN)
        width = self.chunk if self.two_shot else HIDDEN
        self.kernel(
            cute.make_tensor(
                mailbox,
                cute.make_layout(LAMPORT_GENERATIONS * self.tp * self.capacity * width),
            ),
            cute.make_tensor(
                gather, cute.make_layout(LAMPORT_GENERATIONS * self.capacity * HIDDEN)
            ),
            gather_multicast,
            cute.make_tensor(stage_state, cute.make_layout(2)),
            cute.make_tensor(stream_in, rows),
            cute.make_tensor(alpha, vector),
            cute.make_tensor(beta, vector),
            cute.make_tensor(next_weight, vector),
            cute.make_tensor(post_weight, vector),
            cute.make_tensor(new_stream, rows),
            cute.make_tensor(out, rows),
            cute.make_tensor(router, rows),
            m,
            rank,
        ).launch(
            grid=((m + self.tokens - 1) // self.tokens, CLUSTER, 1),
            block=(THREADS, 1, 1),
            cluster=(1, CLUSTER, 1),
            smem=2 * self.tokens * SLOTS * 4,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=MIN_BLOCKS_PER_SM,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        gather: cute.Tensor,
        gather_multicast: Int64,
        stage_state: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        next_weight: cute.Tensor,
        post_weight: cute.Tensor,
        new_stream: cute.Tensor,
        out: cute.Tensor,
        router: cute.Tensor,
        m: Int32,
        rank: Int32,
    ) -> None:
        constants = self.constants
        tp, capacity, count = self.tp, self.capacity, self.tokens
        tidx, _, _ = cute.arch.thread_idx()
        first, _, _ = cute.arch.block_idx()
        clusters, _, _ = cute.arch.grid_dim()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        column = Int64(cluster_rank * THREADS + tidx) * VEC
        # Peers must have started before _row_sums stores into them.
        cute.arch.cluster_arrive_relaxed()

        smem = cutlass.memory.SmemAllocator()
        post_slots = smem.allocate_array(Float32, count * SLOTS)
        next_slots = smem.allocate_array(Float32, count * SLOTS)

        alpha_values = load_f32x8(alpha, column)
        beta_values = load_f32x8(beta, column)
        next_gamma = load_bf16x8_as_f32(next_weight, column)
        post_gamma = None
        if cutlass.const_expr(constants.has_post_weight):
            post_gamma = load_bf16x8_as_f32(post_weight, column) + Float32(
                constants.post_gain_center
            )
        tokens = [first + i * clusters for i in range(count)]
        # The residual stream does not depend on the publisher: load it early
        # unless it would cost the registers of a resident grid.
        preload = count <= 2
        streams = []
        for i in cutlass.range_constexpr(count if preload else 0):
            streams.append(self._stream(stream_in, tokens[i], m, column))

        # The publishing GEMM set ACTIVE (behind a fence) before releasing this
        # grid, and the data are self-validating: poll now, and wait for the
        # GEMM's grid (its memory flush) only before exiting. The next GEMM
        # may start streaming its weight now; it reads the stage state and
        # touches the mailboxes only after this grid completed.
        if cutlass.const_expr(not EARLY):
            cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        active = Int64(load_volatile_u32(stage_state.iterator + ACTIVE_STAGE))

        packed = cute.make_rmem_tensor(cute.make_layout((count, 4)), Uint32)
        packed.fill(Uint32(0))
        valid = [token < m for token in tokens]
        partials = cute.make_rmem_tensor(cute.make_layout((count * tp, 4)), Uint32)
        if cutlass.const_expr(self.two_shot):
            owner = Int32(column // self.chunk)
            if owner == rank:
                local = column - Int64(owner) * self.chunk
                _poll(
                    mailbox,
                    [
                        ((active * tp + s) * capacity + tokens[i]) * self.chunk + local
                        for i in range(count)
                        for s in range(tp)
                    ],
                    [valid[i] for i in range(count) for _ in range(tp)],
                    partials,
                )
                for i in cutlass.range_constexpr(count):
                    if valid[i]:
                        words = sanitize_negative_zero_u32x4(
                            bf16x8_to_packed_u32x4(
                                _rank_sum(partials, i * tp, tp).to(BFloat16)
                            )
                        )
                        element = (active * capacity + tokens[i]) * HIDDEN + column
                        multicast_u32x4(gather_multicast + element * 2, words)
            _poll(
                gather,
                [(active * capacity + token) * HIDDEN + column for token in tokens],
                valid,
                packed,
            )
        else:
            _poll(
                mailbox,
                [
                    ((active * tp + s) * capacity + tokens[i]) * HIDDEN + column
                    for i in range(count)
                    for s in range(tp)
                ],
                [valid[i] for i in range(count) for _ in range(tp)],
                partials,
            )
            for i in cutlass.range_constexpr(count):
                words = bf16x8_to_packed_u32x4(
                    _rank_sum(partials, i * tp, tp).to(BFloat16)
                )
                for word in cutlass.range_constexpr(4):
                    packed[i, word] = words[word]
        if first == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                Uint32((active + 1) % LAMPORT_GENERATIONS),
            )

        # Keep little per row live (the BF16 branch, then the updated stream):
        # the whole grid must be resident, see the module docstring.
        partials = []
        for i in cutlass.range_constexpr(count):
            b = self._branch(packed, i, post_gamma)
            partials.append((b * b).reduce(cute.ReductionOp.ADD, Float32(0.0), 0))

        cute.arch.cluster_wait()
        post_sums = _row_sums(partials, post_slots, cluster_rank)
        updates, partials = [], []
        for i in cutlass.range_constexpr(count):
            if cutlass.const_expr(preload):
                stream_values = streams[i]
            else:
                stream_values = self._stream(stream_in, tokens[i], m, column)
            u = alpha_values * stream_values.load()
            v = beta_values * self._branch(packed, i, post_gamma)
            r1 = cute.math.rsqrt(
                post_sums[i] * Float32(1.0 / HIDDEN) + Float32(constants.post_eps),
                fastmath=True,
            )
            # FlashInfer's u + v * r1 compiles to one FMA per element.
            fused = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
            for j in cutlass.range_constexpr(VEC):
                fused[j] = fma_f32(v[j], r1, u[j])
            updated = fused.load()
            if tokens[i] < m:
                store_f32x8(new_stream, Int64(tokens[i]) * HIDDEN + column, updated)
            updates.append(updated)
            partials.append(
                (updated * updated).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            )
        next_sums = _row_sums(partials, next_slots, cluster_rank)
        for i in cutlass.range_constexpr(count):
            r2 = cute.math.rsqrt(
                next_sums[i] * Float32(1.0 / HIDDEN) + Float32(constants.next_eps),
                fastmath=True,
            )
            result = (updates[i] * r2) * (
                next_gamma + Float32(constants.next_gain_center)
            )
            if tokens[i] < m:
                row_element = Int64(tokens[i]) * HIDDEN + column
                store_global_u32x4(
                    Int64((out.iterator + row_element).toint()),
                    bf16x8_to_packed_u32x4(result.to(BFloat16)),
                )
                if cutlass.const_expr(constants.router):
                    store_f32x8(router, row_element, result)
        # Complete only after the publishing grid: later kernels rely on PDL
        # completion being transitive.
        if cutlass.const_expr(EARLY):
            cute.arch.griddepcontrol_wait()

    @cute.jit
    def _stream(self, stream_in: cute.Tensor, token: Int32, m: Int32, column: Int64):
        values = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
        values.fill(Float32(0.0))
        if token < m:
            values.store(load_f32x8(stream_in, Int64(token) * HIDDEN + column))
        return values

    @cute.jit
    def _branch(self, packed: cute.Tensor, i: cutlass.Constexpr[int], post_gamma):
        words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for word in cutlass.range_constexpr(4):
            words[word] = packed[i, word]
        b = packed_u32x4_to_bf16x8(words.load()).to(Float32)
        if cutlass.const_expr(self.constants.has_post_weight):
            b = b * post_gamma
        return b


def compile_dual_norm(kernel: AllReduceDualNormKernel):
    return cute.compile(
        kernel,
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        Int64(0),
        gmem_ptr(Int32, 1024, 4),
        gmem_ptr(Float32, 1024, 32),
        gmem_ptr(Float32, 1024, 32),
        gmem_ptr(Float32, 1024, 32),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(Float32, 1024, 32),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(Float32, 1024, 32),
        Int32(1),
        Int32(0),
        make_fake_stream(),
    )
