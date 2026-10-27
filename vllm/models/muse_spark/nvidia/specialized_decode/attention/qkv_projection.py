# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark decode QKV(+O-gate) projection with the QK-norm / RoPE / KV insert
in its epilogue: one tcgen05 kernel launch for 1..256 rows.

``x @ W^T`` ([M, 4608] at TP4: q | k | v | o_gate) like ``qkv_proj``, then
``ops.fused_muse_spark_qknorm_rope_kv_insert`` on it: q and k heads are
RMS-normalized (weightless) and, on sliding layers, rotated (GPT-J pairs, FP32
cos | sin table); k and v land in the paged KV cache at ``slot_mapping`` (slots
< 0 skipped). q and o_gate go to their own (e.g. contiguous) outputs; k and v
optionally also to the [M, 4608] ``out``.

Swap-AB split-K over 144 CTAs: a CTA owns 128 weight rows (two heads) and a K
quarter, and accumulates FP32 [128, tile] (tile = M padded to 16) with M=128
tcgen05 MMAs fed by a TMA ring of 128B-swizzled K blocks; the first ring of
weights loads before the PDL wait. The two CTAs of a cluster own adjacent row
tiles of the same K quarter and share the activation rows by TMA multicast.
(Clusters of the four K quarters would reduce on chip, but at most 33 clusters
of four are resident at once, and 36 are needed.)

The K quarters reduce-scatter through global mailboxes: quarter r finishes
tokens [r, r + 1) * tile / 4. It polls its mailbox words until none holds the
``SENTINEL`` its predecessor re-armed them with, sums in FP32, rounds once to
BF16 like the GEMM, and finishes the tokens exactly like the stock insert
kernel (same FMA order and approximate reciprocal square root).
"""

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from .primitives import (
    SENTINEL,
    acquire_tensormap,
    add,
    alloc,
    at,
    bits_float,
    commit,
    copy_g2s,
    cp_async_wait_all,
    empty_word,
    fence_tmem,
    fma,
    free_tmem,
    load_shared_f32x4,
    load_shared_i64,
    load_shared_x4,
    load_tma2,
    load_tma2_multicast,
    load_tmem,
    mma_bf16_block,
    mul,
    named_barrier,
    pack_bf16,
    rsqrt,
    smem_desc,
    store_global_f32x4,
    store_global_x4,
    store_shared_bf16,
    store_shared_i64,
    tensor_map,
    thread,
    unpack_bf16,
    wait,
)

HEAD = 64
ROWS = 128  # weight rows per CTA: two heads
PARTS = 4  # K quarters
BLOCK_K = 64  # K per ring stage: one 128-byte swizzle atom
A_BYTES = ROWS * BLOCK_K * 2
RING_BYTES = 200 * 1024
THREADS = 160  # warps 0-3: producer, prefetch and epilogue; warp 4: MMA
MAX_ROWS = 256


def row_tile(rows: int) -> int:
    """The MMA N (token tile) serving ``rows`` rows."""
    if not 0 < rows <= MAX_ROWS:
        raise ValueError(f"QkvProjection serves 1..{MAX_ROWS} rows, got {rows}")
    return (rows + 15) // 16 * 16


class QkvKernel:
    """Grid: 18 row-tile pairs x 4 K quarters x 2, in clusters of the two tiles
    of a pair. ``workspace``: the [36, 4 owners, 3 senders, tile / 16, 128, 4]
    FP32 mailboxes, ``SENTINEL`` between launches."""

    def __init__(self, tile, num_heads, num_kv_heads, hidden, rope, kv_out):
        assert tile % 16 == 0 and 16 <= tile <= MAX_ROWS
        assert (num_heads * HEAD) % ROWS == 0 and (num_kv_heads * HEAD) % ROWS == 0
        assert hidden % (BLOCK_K * PARTS) == 0
        self.tile, self.rope, self.kv_out = tile, rope, kv_out
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.tiles = 2 * (num_heads + num_kv_heads) * HEAD // ROWS
        self.width = tile // PARTS  # tokens each K quarter finishes
        self.quads = self.width // 4
        self.blocks = hidden // BLOCK_K // PARTS
        self.stage = A_BYTES + tile * BLOCK_K * 2
        self.stages = min(self.blocks, RING_BYTES // self.stage)
        self.columns = max(32, 1 << (tile - 1).bit_length())
        self.instruction = 1 << 4 | 1 << 7 | 1 << 10 | tile >> 3 << 17 | ROWS >> 4 << 24
        # Drained ring: the owned mailbox words, then the BF16 [token, row] tile.
        self.staging = (PARTS - 1) * self.quads * ROWS * 16
        assert self.staging + self.width * ROWS * 2 <= self.stages * self.stage

    @cute.jit
    def __call__(
        self, weight_map, x_map, out, q_out, og_out, kv, positions, slots, cos_sin,
        workspace,
        m: Int32, eps: Float32, block_size: Int32, s_block: Int64, s_head: Int64,
        s_token: Int64, stream,
    ):  # fmt: skip
        self.kernel(
            weight_map, x_map, out, q_out, og_out, kv, positions, slots, cos_sin,
            workspace, m, eps, block_size, s_block, s_head, s_token,
        ).launch(
            grid=[self.tiles * PARTS, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[2, 1, 1],
            stream=stream,
            use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(
        self, weight_map, x_map, out, q_out, og_out, kv, positions, slots, cos_sin,
        workspace,
        m: Int32, eps: Float32, block_size: Int32, s_block: Int64, s_head: Int64,
        s_token: Int64,
    ):  # fmt: skip
        tid, warp, _, cta = thread()
        half, pair = cta % 2, cta // 2
        part = pair % PARTS
        tile = pair // PARTS * 2 + half
        stages, stage, width = self.stages, self.stage, self.width
        smem = cutlass.memory.SmemAllocator()
        ring = alloc(smem, Uint8, stages * stage, 1024).iterator
        table = alloc(smem, Float32, width * HEAD if self.rope else 4, 16).iterator
        cache_rows = alloc(smem, Int64, width, 16).iterator
        bars = alloc(smem, Uint64, 2 * stages + 1, 8).iterator
        tmem_slot = alloc(smem, Uint32, 1, 4)
        full, empty, done = bars, bars + stages, bars + 2 * stages
        k0 = part * (self.blocks * BLOCK_K)
        row0 = tile * ROWS
        owns = part * width < m

        if tid == 0:
            for s in cutlass.range_constexpr(stages):
                cute.arch.mbarrier_init(full + s, 1)
                cute.arch.mbarrier_init(empty + s, 2)  # both CTAs' MMAs
            cute.arch.mbarrier_init(done, 1)
            cute.arch.mbarrier_init_fence()
            acquire_tensormap(weight_map.iterator)
            acquire_tensormap(x_map.iterator)
            # The weights do not depend on the predecessor: the first ring now.
            for kb in cutlass.range_constexpr(stages):
                cute.arch.mbarrier_arrive_and_expect_tx(full + kb, stage)
                column = k0 + kb * BLOCK_K
                w = weight_map.iterator
                load_tma2(ring + kb * stage, w, full + kb, column, row0)
        if warp == 4:
            cute.arch.alloc_tmem(self.columns, tmem_slot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()  # the peer's barriers are initialized
        tmem = tmem_slot[0]

        if warp < 4:
            cute.arch.griddepcontrol_wait()
            if tid == 0:
                self.produce(weight_map, x_map, ring, full, empty, k0, row0, half)
            elif warp > 0:
                self.prefetch(
                    positions, slots, cos_sin, table, cache_rows, m, part,
                    block_size, s_block, s_token,
                )  # fmt: skip
            cute.arch.griddepcontrol_launch_dependents()
            wait(done)
            fence_tmem()
            self.reduce(ring, workspace, tmem, m, part, tile, owns)
            cp_async_wait_all()
            named_barrier(1, 128)
            if owns:
                self.finish(
                    ring + self.staging, table, cache_rows, (out, q_out, og_out), kv,
                    m, part, tile, eps, s_head,
                )  # fmt: skip
        else:
            self.multiply(ring, full, empty, done, tmem)
        fence_tmem(before=True)
        cute.arch.sync_threads()
        if warp == 4:
            fence_tmem()
            free_tmem(tmem, self.columns)

    @cute.jit
    def produce(self, weight_map, x_map, ring, full, empty, k0, row0, half):
        """Thread 0: the weight blocks past the first ring and this CTA's half of
        every block's activation rows (rows past m read as zeros), multicast to
        the pair."""
        stages, stage = self.stages, self.stage
        rows: cutlass.Constexpr = self.tile // 2
        for kb in cutlass.range(self.blocks, unroll=1):
            s = kb % stages
            tile = ring + s * stage
            column = k0 + kb * BLOCK_K
            if kb >= stages:
                wait(empty + s, (kb // stages - 1) % 2)
                cute.arch.mbarrier_arrive_and_expect_tx(full + s, stage)
                load_tma2(tile, weight_map.iterator, full + s, column, row0)
            dst = tile + A_BYTES + half * (rows * BLOCK_K * 2)
            load_tma2_multicast(dst, x_map.iterator, full + s, column, half * rows, 3)

    @cute.jit
    def multiply(self, ring, full, empty, done, tmem):
        """Warp 4: four K16 MMAs per block, releasing the stage in both CTAs."""
        stages, stage = self.stages, self.stage
        for kb in cutlass.range(self.blocks, unroll=1):
            s = kb % stages
            wait(full + s, kb // stages % 2)
            fence_tmem()
            a = smem_desc(ring + s * stage)
            b = smem_desc(ring + s * stage + A_BYTES)
            release = Int32(kb + stages < self.blocks)
            mma_bf16_block(tmem, a, b, self.instruction, kb, empty + s, release, mask=3)
        with cute.arch.elect_one():
            commit(done)

    @cute.jit
    def prefetch(
        self, positions, slots, cos_sin, table, cache_rows, m, part, block_size,
        s_block, s_token,
    ):  # fmt: skip
        """Warps 1-3: the cache offsets (or -1) and cos | sin rows of this
        quarter's tokens into shared memory."""
        tid, _, _, _ = thread()
        per_token: cutlass.Constexpr = HEAD * 4 // 16 if self.rope else 1
        count: cutlass.Constexpr = self.width * per_token
        for i in cutlass.range_constexpr((count + 95) // 96):
            item = tid - 32 + 96 * i
            j, piece = item // per_token, item % per_token
            token = part * self.width + j
            if (item < count) & (token < m):
                if piece == 0:
                    slot = slots[token]
                    block = slot // Int64(block_size)
                    offset = block * s_block + (slot - block * block_size) * s_token
                    offset = cutlass.select_(slot >= 0, offset, Int64(-1))
                    store_shared_i64(cache_rows + j, offset)
                if cutlass.const_expr(self.rope):
                    src = at(cos_sin, Int32(positions[token]), piece * 4)
                    copy_g2s(table + j * HEAD + piece * 4, src)

    @cute.jit
    def reduce(self, ring, workspace, tmem, m, part, tile, owns):
        """Reduce-scatter of the FP32 partials through the mailboxes; the owned
        tokens leave as BF16 [token, row] in the staging tile. Every thread
        moves its own accumulator row only."""
        tid, warp, _, _ = thread()
        width, quads = self.width, self.quads
        lane_base = tmem + (Uint32(32 * warp) << 16)
        base = workspace.iterator.toint()
        send: cutlass.Constexpr = math.gcd(width, 32)
        for p in cutlass.range_constexpr(PARTS):
            if (part != p) & (p * width < m):
                index = cutlass.select_(part < p, part, part - 1)
                cell = ((tile * PARTS + p) * (PARTS - 1) + index) * quads
                for c in cutlass.range_constexpr(0, width, send):
                    values = load_tmem(lane_base + (p * width + c), send)
                    for i in cutlass.range_constexpr(0, send, 4):
                        quad = Int64((cell + (c + i) // 4) * ROWS + tid)
                        store_global_f32x4(base + quad * 16, values[i : i + 4])
        if owns:
            # Snapshot every owned word into the drained ring, then re-fetch the
            # quads still holding a sentinel, all in flight at once, until none.
            cell = (tile * PARTS + part) * (PARTS - 1) * quads
            total: cutlass.Constexpr = (PARTS - 1) * quads
            for q in cutlass.range_constexpr(total):
                src = base + Int64((cell + q) * ROWS + tid) * 16
                copy_g2s(ring + (q * ROWS + tid) * 16, src)
            cp_async_wait_all()
            pending = Int32(1)
            while pending > 0:
                pending = Int32(0)
                for q in cutlass.range_constexpr(total):
                    words = load_shared_x4(ring + (q * ROWS + tid) * 16)
                    if empty_word(words):
                        src = base + Int64((cell + q) * ROWS + tid) * 16
                        copy_g2s(ring + (q * ROWS + tid) * 16, src)
                        pending = pending + 1
                cp_async_wait_all()
            for q in cutlass.range_constexpr(total):
                src = base + Int64((cell + q) * ROWS + tid) * 16
                store_global_x4(src, (SENTINEL,) * 4)  # re-arm
            take: cutlass.Constexpr = math.gcd(width, 16)
            for c in cutlass.range_constexpr(0, width, take):
                values = list(load_tmem(lane_base + Uint32(part * width + c), take))
                for index in cutlass.range_constexpr(PARTS - 1):
                    for i in cutlass.range_constexpr(0, take, 4):
                        q = index * quads + (c + i) // 4
                        words = load_shared_x4(ring + (q * ROWS + tid) * 16)
                        for e in cutlass.range_constexpr(4):
                            values[i + e] = add(values[i + e], bits_float(words[e]))
                staging = ring + self.staging
                for e in cutlass.range_constexpr(take):
                    offset = ((c + e) * ROWS + tid) * 2
                    store_shared_bf16(staging + offset, values[e])

    @cute.jit
    def finish(
        self, staging, table, cache_rows, outs, kv, m, part, tile, eps, s_head
    ):  # fmt: skip
        """Eight lanes per (token, head) of the BF16 [token, row] ``staging``
        tile: the stock insert's norm, rotation and stores (q / k / v / o_gate by
        head)."""
        tid, _, _, _ = thread()
        group, lane8 = tid // 8, tid % 8
        width = self.width
        nq, nkv = self.num_heads, self.num_kv_heads
        items: cutlass.Constexpr = width * 2
        steps: cutlass.Constexpr = (items + 15) // 16
        # Heads [q | k | v | o_gate] of this tile; a tile never mixes kinds.
        g0 = tile * 2
        is_qk = g0 < nq + nkv
        is_kv = (g0 >= nq) & (g0 < nq + 2 * nkv)
        is_v = g0 >= nq + nkv
        kv_head0 = cutlass.select_(is_v, g0 - nq - nkv, g0 - nq)
        value_offset = cutlass.select_(is_v, Int32(HEAD), Int32(0))
        coords = []
        raws = []
        for it in cutlass.range_constexpr(steps):
            item = it * 16 + group
            j = cutlass.min(item // 2, width - 1)
            head = item % 2
            coords.append((item, j, head))
            offset = (j * ROWS + head * HEAD + lane8 * 8) * 2
            raws.append(load_shared_x4(staging + offset))
        for it in cutlass.range_constexpr(steps):
            item, j, head = coords[it]
            raw = raws[it]
            normed = self.normalize(raw, table + j * HEAD + lane8 * 4, eps)
            words = [cutlass.select_(is_qk, normed[w], raw[w]) for w in range(4)]
            token = part * width + j
            if (item < items) & (token < m):
                out, q_out, og_out = outs
                g = g0 + head
                if g < nq:
                    store_global_x4(at(q_out, token, g * HEAD + lane8 * 8), words)
                if g >= nq + 2 * nkv:
                    column = (g - nq - 2 * nkv) * HEAD + lane8 * 8
                    store_global_x4(at(og_out, token, column), words)
                if cutlass.const_expr(self.kv_out):  # noqa: SIM102
                    if is_kv:
                        column = g * HEAD + lane8 * 8
                        store_global_x4(at(out, token, column), words)
                if is_kv:
                    row = load_shared_i64(cache_rows + j)
                    if row >= 0:
                        element = (
                            row + (kv_head0 + head) * s_head + value_offset + lane8 * 8
                        )
                        store_global_x4(kv.iterator + element, words)

    @cute.jit
    def normalize(self, words, rope_row, eps):
        """The stock insert's math on one lane's eight elements of a head."""
        values = []
        for w in cutlass.range_constexpr(4):
            values += list(unpack_bf16(words[w]))
        total = mul(values[0], values[0])
        for e in cutlass.range_constexpr(1, 8):
            total = fma(values[e], values[e], total)
        for o in (4, 2, 1):
            total = add(total, cute.arch.shuffle_sync_bfly(total, o))
        scale = rsqrt(fma(total, 1.0 / HEAD, eps))
        values = [mul(v, scale) for v in values]
        if cutlass.const_expr(self.rope):
            cs = load_shared_f32x4(rope_row)
            sn = load_shared_f32x4(rope_row + HEAD // 2)
            for i in cutlass.range_constexpr(4):
                v0, v1 = values[2 * i], values[2 * i + 1]
                values[2 * i] = fma(v0, cs[i], -mul(v1, sn[i]))
                values[2 * i + 1] = fma(v0, sn[i], mul(v1, cs[i]))
        return [pack_bf16(values[2 * w], values[2 * w + 1]) for w in range(4)]


@functools.cache
def compile_qkv(tile, num_heads, num_kv_heads, hidden, rope, kv_out, device_index):
    """Code-only cache: every pointer and the row count are arguments."""
    sym = cute.sym_int

    def flat(dtype, align=16):
        return make_fake_tensor(dtype, (sym(),), (1,), assumed_align=align)

    def descriptor():
        return make_fake_tensor(Uint8, (128,), (1,), assumed_align=64)

    def rows(width):
        stride = (sym(divisibility=8), 1)
        return make_fake_tensor(BFloat16, (sym(), width), stride, assumed_align=16)

    q_width = num_heads * HEAD
    table = make_fake_tensor(Float32, (sym(), HEAD), (HEAD, 1), assumed_align=16)
    args = (
        descriptor(), descriptor(), rows(sym()), rows(q_width), rows(q_width),
        flat(BFloat16), flat(Int64, 8),
        flat(Int64, 8), table, flat(Float32), Int32(1), Float32(1e-5), Int32(16),
        Int64(0), Int64(0), Int64(0), make_fake_stream(use_tvm_ffi_env_stream=True),
    )  # fmt: skip
    kernel = QkvKernel(tile, num_heads, num_kv_heads, hidden, rope, kv_out)
    with torch.accelerator.device_index(device_index):
        return cute.compile(kernel, *args, options="--enable-tvm-ffi")


@functools.cache
def mailboxes(device: torch.device, n: int) -> torch.Tensor:
    """The FP32 mailboxes of every projection with ``n`` output rows on
    ``device``: one projection runs at a time in stream order, and its owners
    re-arm the words they read."""
    words = (n // ROWS) * PARTS * (PARTS - 1) * (MAX_ROWS // PARTS) * ROWS
    empty = torch.full((words,), -1, dtype=torch.int32, device=device)
    return empty.view(torch.float32)


_X_MAPS: dict[tuple, torch.Tensor] = {}


def _x_map(x: torch.Tensor, rows: int) -> torch.Tensor:
    """TMA map of ``x`` with [rows, 64] boxes, cached by address and shape (the
    decode inputs are pointer-stable)."""
    key = (x.data_ptr(), tuple(x.shape), x.stride(0), rows, x.device)
    descriptor = _X_MAPS.get(key)
    if descriptor is None:
        descriptor = _X_MAPS[key] = tensor_map(x, (BLOCK_K, rows))
    return descriptor


class QkvProjection:
    """One attention layer's fused projection; ``prepare`` compiles the row tiles
    ahead of CUDA graph capture.

    Args:
        weight: ``qkv_proj.weight`` BF16 [(2 * num_heads + 2 * num_kv_heads) *
            64, hidden] with rows [q | k | v | o_gate].
        num_heads: Query heads on this rank.
        num_kv_heads: KV heads on this rank.
        eps: QK-norm epsilon.
        rope: Whether the layer rotates (sliding window) or not (NoPE).

    """

    def __init__(self, weight, num_heads, num_kv_heads, eps, rope):
        n = 2 * (num_heads + num_kv_heads) * HEAD
        if weight.dtype != torch.bfloat16 or weight.dim() != 2 or weight.shape[0] != n:
            raise ValueError(f"expected a BF16 [{n}, K] weight, got {weight.shape}")
        if weight.stride(1) != 1 or (weight.stride(0) * 2) % 16:
            raise ValueError("weight rows need unit stride and a 16-byte pitch")
        self.weight, self.eps, self.rope = weight, eps, rope
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.hidden = weight.shape[1]
        self.device = weight.device
        self.weight_map = tensor_map(weight, (BLOCK_K, ROWS))
        self.mailboxes = mailboxes(self.device, n)
        self.no_rope_table = torch.zeros(1, HEAD, device=self.device)

    def kernel(self, rows, kv_out=False):
        return compile_qkv(
            row_tile(rows), self.num_heads, self.num_kv_heads, self.hidden,
            self.rope, kv_out, self.device.index,
        )  # fmt: skip

    def prepare(self, rows, kv_out=None):
        """Compile the kernels serving each of ``rows``: calls with ``out``
        (``kv_out`` True), without (False), or both (None)."""
        for r in rows:
            for flag in (False, True) if kv_out is None else (kv_out,):
                self.kernel(r, flag)
        return self

    def __call__(
        self,
        x: torch.Tensor,
        out: torch.Tensor | None,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor | None,
        q_out: torch.Tensor | None = None,
        og_out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Project, normalize, rotate and insert ``x``'s rows; returns ``out``.

        Args:
            x: BF16 [M, hidden] attention input, unit column stride.
            out: BF16 [M, 4608] (q | k | v | o_gate, like the stock ``qkv`` after
                the insert), or None when ``q_out`` and ``og_out`` are given;
                it receives k and v only when given.
            kv_cache: The layer's [num_blocks, num_kv_heads, block_size,
                2 * 64] cache view (K | V in the last dim), any block / head /
                token strides.
            slot_mapping: int64 [>= M]; negative slots are not written.
            positions: int64 [>= M].
            cos_sin_cache: FP32 [max_position, 64] (cos | sin), or None on NoPE
                layers (must match ``rope``).
            q_out: BF16 [M, num_heads * 64] for q (normed, rotated), unit column
                stride; default ``out``'s q columns.
            og_out: BF16 [M, num_heads * 64] for o_gate, unit column stride;
                default ``out``'s o_gate columns.

        """
        m = x.shape[0]
        q_width = self.num_heads * HEAD
        if out is not None:
            if q_out is None:
                q_out = out[:, :q_width]
            if og_out is None:
                og_out = out[:, -q_width:]
        assert q_out is not None and og_out is not None
        for t in (q_out, og_out):
            assert t.shape == (m, q_width) and t.stride(1) == 1 and t.stride(0) % 8 == 0
        assert (cos_sin_cache is not None) == self.rope
        assert kv_cache.dim() == 4 and kv_cache.stride(3) == 1
        assert kv_cache.shape[1] == self.num_kv_heads and kv_cache.shape[3] == 2 * HEAD
        table = cos_sin_cache if cos_sin_cache is not None else self.no_rope_table
        assert table.dtype == torch.float32 and table.shape[1] == HEAD
        base = torch.as_strided(kv_cache, (2 * HEAD,), (1,))
        self.kernel(m, out is not None)(
            self.weight_map, _x_map(x, row_tile(m) // 2),
            q_out if out is None else out, q_out, og_out, base, positions,
            slot_mapping, table, self.mailboxes, m, self.eps, kv_cache.shape[2],
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        )  # fmt: skip
        return out
