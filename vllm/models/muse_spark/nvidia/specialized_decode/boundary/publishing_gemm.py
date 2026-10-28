# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M BF16 GEMM whose epilogue publishes C into a Lamport mailbox.

``C[M, N] = A[M, K] @ W[N, K]^T`` on ``tcgen05`` with the operands swapped:
the weight is the MMA's A operand (``bm`` output columns per CTA: one MMA of
64 or 128 rows, or two or four of 128) and at most ``bn`` activation rows
(tokens) are its B operand, accumulated in TMEM. One TMA warp streams the
weight, which does not depend on the previous kernel, through the shared-memory
ring before the programmatic grid wait (and prefetches the rest of its K range
into L2); a second TMA warp loads the activation after the wait and then
releases the dependent grid. ``cf`` neighbouring column tiles can share their
activation tile and ``ct`` neighbouring token tiles their weight tile through
TMA multicast in a cluster.

With ``split_k`` > 1 the ``split_k`` CTAs of a cluster each multiply a K slice.
Once the accumulator is final, every CTA stages it as FP32 in its drained ring,
``tc`` tokens at a time, and sends each other CTA of the split the tokens that
CTA owns (one shared-to-shared bulk copy each); the owner sums the slices in
slice order. Without split-K the staging only transposes the tile.

The epilogue rounds C to BF16 (negative zero made +0, so -0.0 stays the
Lamport sentinel) and multicasts each 8-column fragment into this rank's slot
of a ``[3, tp, capacity, width]`` BF16 mailbox on every rank, the layout of
FlashInfer's LL protocol: generation ``stage_state[0]``, row ``token``,
columns ``offset + n``. One thread then sets ``stage_state[1]`` to that
generation, as FlashInfer's publish kernel does, and the consumer launched as
this kernel's programmatic dependent reads it after its grid wait, that is
after this whole grid and its stores completed. For a two-shot all-reduce the
epilogue instead stores each block of ``width`` columns only into the mailbox
of the rank that owns it. It may also store C locally.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int16, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_ptr
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    bf16x2_to_packed_u32,
    bf16x8_to_packed_u32x4,
    packed_u32_to_bf16x2,
    packed_u32x4_to_bf16x8,
)

from .primitives import (
    EVICT_FIRST,
    bulk_commit_group,
    bulk_wait_group_read,
    copy_shared_to_peer,
    fence_tmem,
    load_shared_f32x4,
    load_shared_words,
    load_tmem_f32,
    load_u64_at,
    load_volatile_u32_at,
    map_shared_rank,
    multicast_u32x4,
    pack_bf16x2_published,
    shared_address,
    store_global_u32x4_at,
    store_shared_f32,
    store_shared_words,
    store_u32_at,
)

NEXT_STAGE = 0
ACTIVE_STAGE = 1
_THREADS = 256
_EPILOGUE_WARP = 4
# Leave room on each SM for the consumer cluster (and other small kernels)
# launched while this grid runs.
_SMEM_BYTES = utils.get_smem_capacity_in_bytes("sm_100") - 16 * 1024


@dataclass(frozen=True)
class GemmTactic:
    """``bm`` output columns (one MMA of 64 or 128, or 2-4 MMAs of 128) and
    ``bn`` tokens per CTA, ``split_k`` CTAs per cluster along K, ``stages``
    ring stages of ``bk`` K elements. ``cf`` CTAs of neighbouring column tiles
    share (multicast) the activation tile and ``ct`` CTAs of neighbouring token
    tiles the weight tile."""

    bm: int
    bn: int
    split_k: int
    stages: int
    bk: int = 64
    cf: int = 1
    ct: int = 1


def _stage_bytes(bm: int, bn: int, bk: int) -> int:
    return (bm + bn) * bk * 2


def _raw_stage_bytes(bn: int, bk: int) -> int:
    # The gate tile of the gated activation load.
    return bn * bk * 2


def max_stages(bm: int, bn: int, bk: int, gated: bool = False) -> int:
    stage = _stage_bytes(bm, bn, bk) + (_raw_stage_bytes(bn, bk) if gated else 0)
    # Barriers, TMEM pointer and alignment slack.
    return (_SMEM_BYTES - 1024) // stage


def _token_chunk(t: GemmTactic) -> int:
    """Tokens per FP32 exchange through the drained ring, which holds the
    staged tile and, with split-K, the slices received from the peers."""
    copies = 2 if t.split_k > 1 else 1
    chunk = t.bn
    while copies * t.bm * chunk * 4 > t.stages * _stage_bytes(t.bm, t.bn, t.bk):
        chunk //= 2
    return chunk


def validate_tactic(t: GemmTactic, n: int, k: int, gated: bool = False) -> None:
    if t.bm not in (64, 128, 256, 512):
        raise ValueError(f"bm={t.bm} must be 64, 128, 256 or 512")
    if not 8 <= t.bn <= 256 or t.bn % (8 if t.bm == 64 else 16):
        raise ValueError(f"bn={t.bn} is not an MMA N for bm={t.bm}")
    if max(1, t.bm // 128) * t.bn > 512:
        raise ValueError("the accumulators exceed the 512 TMEM columns")
    if n % t.bm:
        raise ValueError(f"N={n} must be a multiple of bm={t.bm}")
    if t.split_k not in (1, 2, 4, 8):
        raise ValueError(f"split_k={t.split_k} must be 1, 2, 4 or 8")
    if t.bk % 64 or k % (t.bk * t.split_k):
        raise ValueError(f"K={k} must split into split_k slices of bk={t.bk}")
    if not 1 <= t.stages <= max_stages(t.bm, t.bn, t.bk, gated):
        raise ValueError(f"stages={t.stages} does not fit shared memory")
    if _token_chunk(t) < 8 or _token_chunk(t) % t.split_k:
        raise ValueError("the ring cannot hold an FP32 exchange of the split")
    if (n // t.bm) % t.cf or (t.bn // t.cf) % 8 or (t.bm // t.ct) % 8:
        raise ValueError(f"cf={t.cf}, ct={t.ct} must split the tiles in 8 rows")
    if t.split_k * t.cf * t.ct > 8:
        raise ValueError("clusters hold at most 8 CTAs")


_LOG2E = 1.4426950408889634
_INV_HEAD_DIM = 1.0 / 64


def _gate_many(fragments, eps: float, vector_size, log2_lanes):
    """FlashInfer's rmsnorm_sigmoid_gate epilogue for several head rows of one
    lane (BF16 activation and gate elements each), the rows' reductions
    interleaved for ILP. Per row: a pairwise tree over the lane's squares,
    then a butterfly over its lanes, as FlashInfer's ``_row_sum``. Plain
    Python, unrolled while tracing, as FlashInfer's epilogue."""
    xs, sigmoids, sums = [], [], []
    for x_values, gate_values in fragments:
        x = x_values.to(Float32)
        gate = gate_values.to(Float32)
        sigmoids.append(
            cute.rcp(
                cute.exp2(gate * (-_LOG2E), fastmath=True) + 1.0, approx=True, ftz=True
            )
        )
        squares = cute.make_rmem_tensor((vector_size,), Float32)
        squares.store(x * x)
        values = [squares[i] for i in range(vector_size)]
        while len(values) > 1:
            values = [
                values[2 * i] + values[2 * i + 1] for i in range(len(values) // 2)
            ]
        xs.append(x)
        sums.append(values[0])
    for index in range(log2_lanes):
        sums = [
            value + cute.arch.shuffle_sync_bfly(value, offset=(1 << index))
            for value in sums
        ]
    outputs = []
    for x, sigmoid, total in zip(xs, sigmoids, sums):
        inv_rms = cute.rsqrt(total * _INV_HEAD_DIM + Float32(eps), fastmath=True)
        output = cute.make_rmem_tensor((vector_size,), BFloat16)
        output.store((x * inv_rms * sigmoid).to(BFloat16))
        outputs.append(output)
    return outputs


def _unpack(words: list, width: int):
    if width == 2:
        return packed_u32_to_bf16x2(words[0])
    packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
    for i in range(4):
        packed[i] = words[i]
    return packed_u32x4_to_bf16x8(packed.load())


@cute.jit
def _pack(output: cute.Tensor, width: cutlass.Constexpr[int]):
    if cutlass.const_expr(width == 2):
        return [bf16x2_to_packed_u32(output.load())]
    words = bf16x8_to_packed_u32x4(output.load())
    return [words[i] for i in range(4)]


class PublishingGemmKernel:
    """The kernel of one (N, K, tactic); see the module docstring.

    ``publish`` multicasts C into the mailbox, ``store_c`` stores it to ``c``.
    With ``scatter`` the ``mailbox`` argument is instead a device table of the
    ``tp`` ranks' mailbox addresses and each ``width``-column block of C goes
    to the rank that owns it only (the reduce-scatter of a two-shot
    all-reduce). The mailbox geometry (``tp``, ``capacity`` rows, row ``width``
    and column ``offset``) is compile-time; its addresses and the rank are
    arguments.
    """

    def __init__(
        self,
        n: int,
        k: int,
        tactic: GemmTactic,
        *,
        publish: bool,
        store_c: bool,
        scatter: bool = False,
        tp: int = 1,
        capacity: int = 1,
        width: int = 0,
        offset: int = 0,
        prefetch_l2: bool = True,
        gate_lanes: int = 0,
        gate_eps: float = 1e-5,
    ) -> None:
        validate_tactic(tactic, n, k, gated=gate_lanes > 0)
        if gate_lanes and (
            gate_lanes not in (8, 32) or tactic.cf > 1 or tactic.bn % (32 // gate_lanes)
        ):
            raise ValueError("the gated activation load needs cf=1 and whole passes")
        if publish and not scatter and (offset % 8 or offset + n > width):
            raise ValueError("the published columns must fit the mailbox row")
        if scatter and (offset or n != tp * width or width % tactic.bm):
            raise ValueError("scatter splits all N columns into tp owner blocks")
        self.n, self.k = n, k
        self.bm, self.bn, self.bk = tactic.bm, tactic.bn, tactic.bk
        self.split_k = tactic.split_k
        self.cf, self.ct = tactic.cf, tactic.ct
        self.multicast = tactic.cf * tactic.ct > 1
        self.stages = tactic.stages
        self.k_tiles = k // tactic.bk // tactic.split_k
        self.atom_m = min(tactic.bm, 128)
        self.mt = max(1, tactic.bm // 128)
        self.tmem_cols = max(32, 1 << (self.mt * tactic.bn - 1).bit_length())
        self.tc = _token_chunk(tactic)
        self.chunks = tactic.bn // self.tc
        # Each CTA of the split reduces and publishes tc / split_k tokens.
        self.tokens_own = self.tc // tactic.split_k
        self.load_cols = next(c for c in (32, 16, 8) if self.tc % c == 0)
        self.publish = publish
        self.scatter = scatter
        self.store_c = store_c
        self.tp, self.capacity = tp, capacity
        self.width, self.offset = width, offset
        self.prefetch_l2 = prefetch_l2 and self.k_tiles > self.stages
        self.gate_lanes = gate_lanes
        self.gate_eps = gate_eps

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        g_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        m: Int32,
        mailbox: Int64,
        stage_state: Int64,
        rank: Int32,
        stream: cuda.CUstream,
    ) -> None:
        w = cute.make_tensor(
            w_ptr, cute.make_layout((self.n, self.k), stride=(self.k, 1))
        )
        x = cute.make_tensor(x_ptr, cute.make_layout((m, self.k), stride=(self.k, 1)))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (self.atom_m, self.bn),
        )
        mma_tiler = (self.bm, self.bn, self.bk)
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, BFloat16, self.stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, BFloat16, self.stages
        )
        cluster_mn = (self.cf, self.ct)
        cluster_vmnk = cute.tiled_divide(
            cute.make_layout((*cluster_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_a, tma_w = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(cluster_mn, tiled_mma.thr_id),
            w,
            cute.slice_(a_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster_vmnk.shape,
        )
        tma_b, tma_x = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(cluster_mn, tiled_mma.thr_id),
            x,
            cute.slice_(b_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster_vmnk.shape,
        )
        tma_bg, tma_g = tma_b, tma_x
        if cutlass.const_expr(self.gate_lanes > 0):
            tma_bg, tma_g = cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils.cluster_shape_to_tma_atom_B(cluster_mn, tiled_mma.thr_id),
                cute.make_tensor(
                    g_ptr, cute.make_layout((m, self.k), stride=(self.k, 1))
                ),
                cute.slice_(b_layout, (None, None, None, 0)),
                mma_tiler,
                tiled_mma,
                cluster_vmnk.shape,
            )
        self.a_bytes = cute.size_in_bytes(
            BFloat16, cute.slice_(a_layout, (None, None, None, 0))
        )
        self.b_bytes = cute.size_in_bytes(
            BFloat16, cute.slice_(b_layout, (None, None, None, 0))
        )
        token_tiles = (m + self.bn - 1) // self.bn
        self.kernel(
            tiled_mma,
            tma_a,
            tma_w,
            tma_b,
            tma_x,
            tma_bg,
            tma_g,
            a_layout,
            b_layout,
            c_ptr,
            m,
            mailbox,
            stage_state,
            rank,
        ).launch(
            grid=(
                self.split_k,
                self.n // self.bm,
                (token_tiles + self.ct - 1) // self.ct * self.ct,
            ),
            block=(_THREADS, 1, 1),
            cluster=(self.split_k, self.cf, self.ct),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mW: cute.Tensor,
        tma_b: cute.CopyAtom,
        mX: cute.Tensor,
        tma_bg: cute.CopyAtom,
        mG: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        c_ptr: cute.Pointer,
        m: Int32,
        mailbox: Int64,
        stage_state: Int64,
        rank: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        split_rank, feat_tile, tok_tile = cute.arch.block_idx()
        stages = self.stages
        sk, cf, ct = self.split_k, self.cf, self.ct
        # Cluster ranks: split slice fastest, then column tile, then token tile.
        f_in = feat_tile % cf
        t_in = tok_tile % ct
        cluster_rank = split_rank + sk * (f_in + cf * t_in)
        w_mask = Int32(0)
        for t in cutlass.range_constexpr(ct):
            w_mask = w_mask | (Int32(1) << (split_rank + sk * (f_in + cf * t)))
        x_mask = Int32(0)
        for f in cutlass.range_constexpr(cf):
            x_mask = x_mask | (Int32(1) << (split_rank + sk * (f + cf * t_in)))

        smem = cutlass.memory.SmemAllocator()
        full = smem.allocate_array(Int64, stages)
        empty = smem.allocate_array(Int64, stages)
        acc_full = smem.allocate_array(Int64, 1)
        received = smem.allocate_array(Int64, 1)
        stage_ready = smem.allocate_array(Int64, 1)
        stage_slot = smem.allocate_array(Int32, 1)
        tmem_holder = smem.allocate_array(Int32, 1)
        sA = smem.allocate_tensor(
            BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner
        )
        sB = smem.allocate_tensor(
            BFloat16, b_layout.outer, byte_alignment=1024, swizzle=b_layout.inner
        )
        raw_full = full
        sG = sB
        if cutlass.const_expr(self.gate_lanes > 0):
            raw_full = smem.allocate_array(Int64, stages)
            sG = smem.allocate_tensor(
                BFloat16, b_layout.outer, byte_alignment=1024, swizzle=b_layout.inner
            )

        if warp == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(stages):
                    # The weight TMA and the activation TMA, or every thread of
                    # the gating warp.
                    cute.arch.mbarrier_init(
                        full + s, 1 + (32 if self.gate_lanes else 1)
                    )
                    if cutlass.const_expr(self.gate_lanes > 0):
                        cute.arch.mbarrier_init(raw_full + s, 1)
                    # Every CTA that multicast into this stage releases it.
                    cute.arch.mbarrier_init(empty + s, cf + ct - 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(received, 1)
                cute.arch.mbarrier_init(stage_ready, 1)
        if warp == 2:
            cute.arch.alloc_tmem(self.tmem_cols, tmem_holder)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.multicast):
            # Peers' barriers must be initialized before anything is
            # multicast into them.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        fence_tmem(before=True)
        cute.arch.barrier()
        fence_tmem()
        tmem_base = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_holder)
        tmem_address = cute.arch.load(tmem_holder.llvm_ptr, Int32)

        k_first = split_rank * self.k_tiles
        thr_mma = tiled_mma.get_slice(0)
        tiler = (self.bm, self.bn, self.bk)
        gW = cute.local_tile(mW, cute.slice_(tiler, (None, 0, None)), (None, None))
        gX = cute.local_tile(mX, cute.slice_(tiler, (0, None, None)), (None, None))
        tWsW, tWgW = cpasync.tma_partition(
            tma_a,
            t_in,
            cute.make_layout(ct),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gW), 0, 3),
        )
        tXsX, tXgX = cpasync.tma_partition(
            tma_b,
            f_in,
            cute.make_layout(cf),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gX), 0, 3),
        )
        tWgW = tWgW[(None, feat_tile, None)]
        tXgX = tXgX[(None, tok_tile, None)]
        tGsG, tGgG = tXsX, tXgX
        if cutlass.const_expr(self.gate_lanes > 0):
            gG = cute.local_tile(mG, cute.slice_(tiler, (0, None, None)), (None, None))
            tGsG, tGgG = cpasync.tma_partition(
                tma_bg,
                0,
                cute.make_layout(1),
                cute.group_modes(sG, 0, 3),
                cute.group_modes(thr_mma.partition_B(gG), 0, 3),
            )
            tGgG = tGgG[(None, tok_tile, None)]

        if warp == 0:
            # The weight does not depend on the previous kernel.
            if cutlass.const_expr(self.prefetch_l2):
                for kt in cutlass.range(stages, self.k_tiles, unroll=1):
                    cute.prefetch(tma_a, tWgW[(None, k_first + kt)])
            for kt in cutlass.range(self.k_tiles, unroll=1):
                s = kt % stages
                if kt >= stages:
                    cute.arch.mbarrier_wait(empty + s, (kt // stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.a_bytes)
                if cutlass.const_expr(ct > 1):
                    cute.copy(
                        tma_a,
                        tWgW[(None, k_first + kt)],
                        tWsW[(None, s)],
                        tma_bar_ptr=full + s,
                        mcast_mask=Int16(w_mask),
                        cache_policy=Int64(EVICT_FIRST),
                    )
                else:
                    cute.copy(
                        tma_a,
                        tWgW[(None, k_first + kt)],
                        tWsW[(None, s)],
                        tma_bar_ptr=full + s,
                        cache_policy=Int64(EVICT_FIRST),
                    )
            if cutlass.const_expr(self.multicast):
                # Peers release this CTA's stages remotely: exit only after the
                # last releases landed.
                for i in cutlass.range_constexpr(min(stages, self.k_tiles)):
                    kt = self.k_tiles - 1 - i
                    cute.arch.mbarrier_wait(empty + kt % stages, kt // stages % 2)
        elif warp == 1:
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.publish):
                self.claim_stage(
                    split_rank,
                    feat_tile,
                    tok_tile,
                    stage_state,
                    stage_slot,
                    stage_ready,
                )
            cute.arch.griddepcontrol_launch_dependents()
            for kt in cutlass.range(self.k_tiles, unroll=1):
                s = kt % stages
                if kt >= stages:
                    cute.arch.mbarrier_wait(empty + s, (kt // stages - 1) % 2)
                if cutlass.const_expr(self.gate_lanes > 0):
                    # The raw activation and gate tiles, for the gating warp.
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            raw_full + s, 2 * self.b_bytes
                        )
                    cute.copy(
                        tma_b,
                        tXgX[(None, k_first + kt)],
                        tXsX[(None, s)],
                        tma_bar_ptr=raw_full + s,
                    )
                    cute.copy(
                        tma_bg,
                        tGgG[(None, k_first + kt)],
                        tGsG[(None, s)],
                        tma_bar_ptr=raw_full + s,
                    )
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.b_bytes)
                    if cutlass.const_expr(cf > 1):
                        cute.copy(
                            tma_b,
                            tXgX[(None, k_first + kt)],
                            tXsX[(None, s)],
                            tma_bar_ptr=full + s,
                            mcast_mask=Int16(x_mask),
                        )
                    else:
                        cute.copy(
                            tma_b,
                            tXgX[(None, k_first + kt)],
                            tXsX[(None, s)],
                            tma_bar_ptr=full + s,
                        )
        elif warp == 2:
            acc = cute.make_tensor(
                tmem_base,
                tiled_mma.make_fragment_C(
                    tiled_mma.partition_shape_C((self.bm, self.bn))
                ).layout,
            )
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            for kt in cutlass.range(self.k_tiles, unroll=1):
                s = kt % stages
                cute.arch.mbarrier_wait(full + s, (kt // stages) % 2)
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, (kt + kb) != 0)
                    cute.gemm(
                        tiled_mma,
                        acc,
                        tCrA[(None, None, kb, s)],
                        tCrB[(None, None, kb, s)],
                        acc,
                    )
                with cute.arch.elect_one():
                    if cutlass.const_expr(self.multicast):
                        tcgen05.commit(empty + s, Int16(w_mask | x_mask))
                    else:
                        tcgen05.commit(empty + s)
            with cute.arch.elect_one():
                tcgen05.commit(acc_full)
        elif warp >= _EPILOGUE_WARP:
            if cutlass.const_expr(self.gate_lanes > 0):
                self.gate_producer(
                    tidx,
                    shared_address(sB.iterator),
                    shared_address(sG.iterator),
                    raw_full,
                    full,
                )
            self.epilogue(
                tidx,
                split_rank,
                cluster_rank,
                feat_tile,
                tok_tile,
                tmem_address,
                acc_full,
                received,
                stage_slot,
                stage_ready,
                # The FP32 exchange reuses the drained ring.
                shared_address(sA.iterator),
                c_ptr,
                m,
                mailbox,
                stage_state,
                rank,
            )

        if cutlass.const_expr(sk > 1):  # noqa: SIM102
            if warp < _EPILOGUE_WARP:
                # Every thread of the cluster joins the epilogue's exchange
                # barriers.
                for _ in cutlass.range_constexpr(self.chunks):
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

        fence_tmem(before=True)
        cute.arch.barrier()
        if warp == 2:
            fence_tmem()
            cute.arch.dealloc_tmem(tmem_base, self.tmem_cols)

    @cute.jit
    def gate_producer(
        self,
        tidx: Int32,
        ring: Int32,
        gate_ring: Int32,
        raw_full: cute.Pointer,
        full: cute.Pointer,
    ) -> None:
        """The activation operand gated in place: once the TMA landed a K
        tile's activation (in the B stage) and gate, each 64-column head is
        RMS-normalized and multiplied by ``sigmoid(gate)`` exactly as
        FlashInfer's ``rmsnorm_sigmoid_gate`` does (its lane split and
        reduction tree for the step's row count) and rounded to BF16. Rows past
        M are zero-filled by the TMA and stay zero."""
        epi = tidx - _EPILOGUE_WARP * 32
        # Each warp gates every fourth K tile: four tiles in flight.
        warp = epi // 32
        lanes = self.gate_lanes
        rows_per_pass = 32 // lanes
        width = 64 // lanes
        heads = self.bk // 64
        lane = epi % lanes
        for kt in cutlass.range(warp, self.k_tiles, 4, unroll=1):
            s = kt % self.stages
            cute.arch.mbarrier_wait(raw_full + s, (kt // self.stages) % 2)
            stage = s * self.bn * self.bk * 2
            # Every (head, row) of the tile this lane takes: loads first, then
            # the reductions interleaved, then the stores.
            # The 16-byte chunk (column // 8) is swizzled with the row in its
            # 8-row, 128-byte-row atom.
            chunk = (lane * width) // 8
            rows = [r0 + epi % 32 // lanes for r0 in range(0, self.bn, rows_per_pass)]
            offsets = [
                stage
                + h * self.bn * 128
                + row * 128
                + ((chunk ^ (row % 8)) * 16)
                + (lane * width % 8) * 2
                for h in range(heads)
                for row in rows
            ]
            fragments = [
                (
                    _unpack(load_shared_words(ring + offset, width // 2), width),
                    _unpack(load_shared_words(gate_ring + offset, width // 2), width),
                )
                for offset in offsets
            ]
            outputs = _gate_many(
                fragments, self.gate_eps, width, (lanes - 1).bit_length()
            )
            for offset, out in zip(offsets, outputs):
                store_shared_words(ring + offset, _pack(out, width))
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(full + s)

    @cute.jit
    def claim_stage(
        self,
        split_rank: Int32,
        feat_tile: Int32,
        tok_tile: Int32,
        stage_state: Int64,
        stage_slot: cute.Pointer,
        stage_ready: cute.Pointer,
    ) -> None:
        """Read the generation to publish into, right after the grid wait (the
        previous consumer, which advanced it, has completed), for the
        epilogue. One CTA also sets ``stage_state[1]`` and fences it before
        this grid releases its dependents, so a consumer may read it without
        waiting for the whole grid."""
        if cute.arch.lane_idx() == 0:
            stage = load_volatile_u32_at(stage_state + NEXT_STAGE * 4)
            cute.arch.store(stage_slot.llvm_ptr, Int32(stage))
            if split_rank == 0 and feat_tile == 0 and tok_tile == 0:
                store_u32_at(stage_state + ACTIVE_STAGE * 4, stage)
                cute.arch.fence_acq_rel_gpu()
            cute.arch.mbarrier_arrive(stage_ready)
        cute.arch.sync_warp()

    @cute.jit
    def epilogue(
        self,
        tidx: Int32,
        split_rank: Int32,
        cluster_rank: Int32,
        feat_tile: Int32,
        tok_tile: Int32,
        tmem_address: Int32,
        acc_full: cute.Pointer,
        received: cute.Pointer,
        stage_slot: cute.Pointer,
        stage_ready: cute.Pointer,
        ring: Int32,
        c_ptr: cute.Pointer,
        m: Int32,
        mailbox: Int64,
        stage_state: Int64,
        rank: Int32,
    ) -> None:
        """Stage this thread's accumulator rows (output columns) in the drained
        ring, ``tc`` tokens at a time and grouped by owning CTA, send every
        other CTA of the split its slice (one bulk copy each) and publish the
        own columns once the peers' slices arrived."""
        epi = tidx - _EPILOGUE_WARP * 32
        quadrant = epi // 32
        lane = epi % 32
        sk = self.split_k
        # [owner][token][row] FP32 slices (each CTA of the split owns
        # tokens_own tokens of a chunk): staged, then received from peers.
        slice_bytes = self.tokens_own * self.bm * 4
        incoming = ring + sk * slice_bytes

        cute.arch.mbarrier_wait(acc_full, 0)
        fence_tmem()
        stage = Uint32(0)
        if cutlass.const_expr(self.publish):
            cute.arch.mbarrier_wait(stage_ready, 0)
            stage = Uint32(cute.arch.load(stage_slot.llvm_ptr, Int32))
        tmem_lane = tmem_address + ((quadrant * 32) << 16)
        for c in cutlass.range_constexpr(self.chunks):
            for mi in cutlass.range_constexpr(self.mt):
                if cutlass.const_expr(self.bm == 64):
                    # M=64 accumulators use lanes 0-15 of each lane quadrant.
                    self.stage_row(
                        ring, tmem_lane, quadrant * 16 + lane, mi, c, lane < 16
                    )
                else:
                    self.stage_row(ring, tmem_lane, mi * 128 + epi, mi, c, True)
            if cutlass.const_expr(sk > 1):
                # The bulk copies read the staged slices through the async proxy.
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if cutlass.const_expr(sk > 1):
                # Every CTA of the split staged this chunk and read the last one.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if epi == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        received, (sk - 1) * slice_bytes
                    )
                    base = cluster_rank - split_rank
                    for s in cutlass.range_constexpr(1, sk):
                        peer = (split_rank + s) % sk
                        copy_shared_to_peer(
                            map_shared_rank(
                                incoming + split_rank * slice_bytes, base + peer
                            ),
                            ring + peer * slice_bytes,
                            map_shared_rank(shared_address(received), base + peer),
                            slice_bytes,
                        )
                    bulk_commit_group()
                cute.arch.mbarrier_wait(received, c % 2)
            self.publish_rows(
                epi,
                c,
                split_rank,
                feat_tile,
                tok_tile,
                ring,
                incoming,
                c_ptr,
                m,
                mailbox,
                stage,
                rank,
            )
            if cutlass.const_expr(sk > 1):  # noqa: SIM102
                if epi == 0:
                    bulk_wait_group_read()
            if cutlass.const_expr(c + 1 < self.chunks or sk > 1):
                # The staged slices may be overwritten (or the CTA exit).
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

    @cute.jit
    def stage_row(
        self,
        ring: Int32,
        tmem_lane: Int32,
        row: Int32,
        mi: cutlass.Constexpr[int],
        chunk: cutlass.Constexpr[int],
        row_valid,
    ) -> None:
        """Store accumulator row ``row`` (TMEM tile ``mi``) of a token chunk
        token-major into the ring, each owner's tokens contiguous. Tokens past
        M are staged too, and never read."""
        tc, owned = self.tc, self.tokens_own
        address = ring + row * 4
        for c0 in cutlass.range_constexpr(0, tc, self.load_cols):
            values = load_tmem_f32(
                tmem_lane + mi * self.bn + chunk * tc + c0, self.load_cols
            )
            if row_valid:
                for j in cutlass.range_constexpr(self.load_cols):
                    t = c0 + j
                    # Token t goes to owner t // owned: slices are contiguous.
                    store_shared_f32(
                        address + (t // owned * owned + t % owned) * self.bm * 4,
                        values[j],
                    )

    @cute.jit
    def publish_rows(
        self,
        epi: Int32,
        chunk: cutlass.Constexpr[int],
        split_rank: Int32,
        feat_tile: Int32,
        tok_tile: Int32,
        ring: Int32,
        incoming: Int32,
        c_ptr: cute.Pointer,
        m: Int32,
        mailbox: Int64,
        stage: Uint32,
        rank: Int32,
    ) -> None:
        """Sum the K slices of this CTA's tokens of a chunk in slice order (its
        own staged slice and those received), round to BF16 and publish (and
        store) them in 8-column fragments, consecutive threads on consecutive
        columns of a token."""
        tc = self.tc
        valid = cutlass.min(m - tok_tile * self.bn - chunk * tc, tc)
        if cutlass.const_expr(self.scatter):
            # A CTA's columns all belong to one owner rank.
            owner = feat_tile * self.bm // self.width
            owner_base = load_u64_at(mailbox + Int64(owner) * 8)
        owned = self.tokens_own
        groups = self.bm // 8
        fragments = groups * owned
        first = split_rank * owned
        iters = (fragments + 127) // 128
        # Issue 16 shared loads (2 x split_k per fragment) before summing.
        batch = max(1, 8 // self.split_k)
        for i0 in cutlass.range_constexpr(0, iters, batch):
            loaded = []
            for i in cutlass.range_constexpr(i0, min(i0 + batch, iters)):
                f = epi + 128 * i
                group = f % groups
                t = f // groups
                offset = (cutlass.min(t, owned - 1) * self.bm + group * 8) * 4
                slice_bytes = owned * self.bm * 4
                parts = []
                for s in cutlass.range_constexpr(self.split_k):
                    # Slice s: staged here if this CTA's own, else received.
                    base = incoming + s * slice_bytes + offset
                    if s == split_rank:
                        base = ring + s * slice_bytes + offset
                    for h in cutlass.range_constexpr(2):
                        parts.append(load_shared_f32x4(base + h * 16))
                loaded.append((f, group, t, parts))
            for f, group, t, parts in loaded:
                if f < fragments and first + t < valid:  # noqa: SIM102
                    values = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
                    for s in cutlass.range_constexpr(self.split_k):
                        for h in cutlass.range_constexpr(2):
                            for j in cutlass.range_constexpr(4):
                                part = parts[2 * s + h][j]
                                if cutlass.const_expr(s == 0):
                                    values[h * 4 + j] = part
                                else:
                                    values[h * 4 + j] = values[h * 4 + j] + part
                    words = [
                        pack_bf16x2_published(values[2 * j], values[2 * j + 1])
                        for j in range(4)
                    ]
                    column = feat_tile * self.bm + group * 8
                    token = tok_tile * self.bn + chunk * tc + first + t
                    if cutlass.const_expr(self.store_c):
                        store_global_u32x4_at(
                            Int64(c_ptr.toint()) + (Int64(token) * self.n + column) * 2,
                            words,
                        )
                    if cutlass.const_expr(self.publish):
                        slot = (Int64(stage) * self.tp + Int64(rank)) * self.capacity
                        if cutlass.const_expr(self.scatter):
                            element = (slot + Int64(token)) * self.width + (
                                column - owner * self.width
                            )
                            store_global_u32x4_at(owner_base + element * 2, words)
                        else:
                            element = (slot + Int64(token)) * self.width + (
                                self.offset + column
                            )
                            multicast_u32x4(mailbox + element * 2, words)


def gmem_ptr(dtype, address: int, align: int = 16):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def compile_gemm(kernel: PublishingGemmKernel):
    return cute.compile(
        kernel,
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        gmem_ptr(BFloat16, 1024),
        Int32(1),
        Int64(0),
        Int64(0),
        Int32(0),
        make_fake_stream(),
    )
