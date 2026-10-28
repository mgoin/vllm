# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark decode MoE router in one kernel launch: the FP32 router GEMM and
``fused_musespark1x_routing``'s selection for 1..256 rows.

``logits = x @ W^T`` with the FP32 router weight (output multiplier folded in)
and x BF16 or FP32. Then sigmoid scores, the expert bias added for selection
only, top-8 (ties to the lower expert), and weights = the selected unbiased
scores / (their sum + eps) * route_scale, as ``topk_ids`` int32 [M, 8] and
``topk_weights`` FP32 [M, 8] in selection order, bit for bit as the Triton
routing kernel computes them from the same logits.

The GEMM runs on tensor cores at near-FP32 accuracy: the weight splits into
three BF16 terms (hi + mid + lo carries its 24 bits; FP32 activations split
the same way) and each product of terms that matters is a BF16 MMA with FP32
accumulation (``mma.sync.m16n8k16``), never TF32.

128 CTAs of 256 threads: CTA (group g, part p) holds 16 experts' rows
[16 g, 16 g + 16) x K eighth p (64 KB, in registers before the PDL wait, split
into MMA fragments), stages the activation rows' K eighth in shared memory
(double-buffered, 8 or 16 rows a pass) and sums its eight warps' K slices in a
fixed order into its row of the partial-logits mailboxes. Warp w < 2 of CTA c
then routes row 128 w + c: it polls the row's eight partials until no word holds the
``SENTINEL``, adds them in part order, selects, writes and re-arms them. Every
sum runs in a fixed order, so all ranks of a TP group get the same bits.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..attention.primitives import (
    add,
    alloc,
    asm,
    bits_float,
    copy_g2s,
    load_shared_f32x4,
    mul,
    thread,
)

EXPERTS = 256
HIDDEN = 8192
TOP_K = 8
PARTS = 8  # K eighths
GROUP = 16  # experts per CTA: one MMA M tile
CHUNK = HIDDEN // PARTS
THREADS = 256
WARPS = THREADS // 32
WARP_K = CHUNK // WARPS  # K elements per warp
STEPS = WARP_K // 16  # K16 MMA steps per warp
TILE = 8  # MMA N: activation rows per tile
GATED_PASS_ROWS = 64  # above this, stage 32 BF16 rows a pass instead of 16
MAX_ROWS = 256
GROUPS = EXPERTS // GROUP
SENTINEL = -1  # an FP32 NaN no arithmetic produces: an empty mailbox word
NONE = -(2**31)  # below every score's key: taken
# Batcher's odd-even merge sort of eight: (i, j) puts the larger at i.
BATCHER_8 = (
    (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2),
    (5, 6), (0, 4), (1, 5), (2, 6), (3, 7), (2, 4), (3, 5), (1, 2), (3, 4),
    (5, 6),
)  # fmt: skip


@asm((Float32,) * 4)
def load_weights(address):
    """Four static FP32 values, L2 evict_first, not kept in L1."""
    code = "{ .reg .b64 p; createpolicy.fractional.L2::evict_first.b64 p,1.0; "
    code += "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.f32 "
    code += "{$0,$1,$2,$3},[$4],p; }"
    return code, "=f,=f,=f,=f,l", (address,)


@asm((Int32,) * 4)
def load_mailbox(address):
    code = "ld.relaxed.gpu.global.v4.b32 {$0,$1,$2,$3},[$4];"
    return code, "=r,=r,=r,=r,l,~{memory}", (address,)


@asm()
def empty_mailbox(address):
    """Four words back to ``SENTINEL``."""
    code = "st.global.v4.b32 [$0],{$1,$1,$1,$1};"
    return code, "l,r,~{memory}", (address, Int32(SENTINEL))


@asm((Uint32,) * 2)
def load_shared_x2(pointer):
    code = "ld.shared.v2.b32 {$0,$1},[$2];"
    return code, "=r,=r,r,~{memory}", (pointer,)


@asm(Float32)
def load_shared_f32(pointer):
    return "ld.shared.f32 $0,[$1];", "=f,r,~{memory}", (pointer,)


@asm()
def store_shared_f32(pointer, value):
    return "st.shared.f32 [$0],$1;", "r,f,~{memory}", (pointer, Float32(value))


@asm()
def store_global_f32(address, value):
    return "st.global.f32 [$0],$1;", "l,f,~{memory}", (address, Float32(value))


@asm()
def cp_async_commit():
    return "cp.async.commit_group;", "~{memory}", ()


@asm()
def cp_async_wait(pending):
    return f"cp.async.wait_group {pending};", "~{memory}", ()


@asm((Uint32,) * 3, pure=True)
def split3(a, b):
    """(a, b) as three BF16x2 terms (a in the low halves) whose FP32 sum is
    (a, b) to within 2^-24 relative."""
    code = "{ .reg .b32 h,m,ra,rb,u,v; "
    code += "cvt.rn.bf16x2.f32 h,$4,$3; "
    code += "shl.b32 u,h,16; and.b32 v,h,0xffff0000; "
    code += "sub.f32 ra,$3,u; sub.f32 rb,$4,v; "
    code += "cvt.rn.bf16x2.f32 m,rb,ra; "
    code += "shl.b32 u,m,16; and.b32 v,m,0xffff0000; "
    code += "sub.f32 ra,ra,u; sub.f32 rb,rb,v; "
    code += "cvt.rn.bf16x2.f32 $2,rb,ra; mov.b32 $0,h; mov.b32 $1,m; }"
    return code, "=r,=r,=r,f,f", (Float32(a), Float32(b))


@asm((Float32,) * 4, pure=True)
def mma(a, b, c):
    """D = A B + C, m16n8k16, BF16 in, FP32 accumulate."""
    code = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    code += "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
    operands = (*(Uint32(v) for v in a), *(Uint32(v) for v in b))
    operands += tuple(Float32(v) for v in c)
    return code, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", operands


@asm(Int32, pure=True)
def warp_max(value):
    return "redux.sync.max.s32 $0,$1,0xffffffff;", "=r,r", (Int32(value),)


@asm(Float32, pure=True)
def sigmoid(x):
    """1 / (1 + exp(-x)) as Triton's ``tl.sigmoid`` computes it."""
    code = "{ .reg .f32 t; mul.f32 t,$1,0fBFB8AA3B; ex2.approx.f32 t,t; "
    code += "add.f32 t,t,0f3F800000; div.full.f32 $0,0f3F800000,t; }"
    return code, "=f,f", (Float32(x),)


@asm(Float32, pure=True)
def divide(a, b):
    return "div.full.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Int32, pure=True)
def ordered(x):
    """A key whose signed order is the float order of ``x``."""
    code = "{ .reg .b32 m; shr.s32 m,$1,31; and.b32 m,m,0x7fffffff; "
    code += "xor.b32 $0,$1,m; }"
    return code, "=r,f", (Float32(x),)


class RouterKernel:
    """``tiles`` x 8 activation rows a pass; ``fp32``: FP32 activations;
    ``keep_logits``: also write the logits out."""

    def __init__(self, tiles: int, fp32: bool, keep_logits: bool):
        self.tiles, self.fp32, self.keep_logits = tiles, fp32, keep_logits
        self.rows = tiles * TILE
        self.size = 4 if fp32 else 2
        self.pitch = (CHUNK + 16) * self.size  # conflict-free fragment loads

    @cute.jit
    def __call__(
        self, x, weight, bias, mailbox, logits, topk_ids, topk_weights, m: Int32,
        eps: Float32, route_scale: Float32, stream,
    ):  # fmt: skip
        self.kernel(
            x, weight, bias, mailbox, logits, topk_ids, topk_weights, m, eps,
            route_scale,
        ).launch(
            grid=[GROUPS * PARTS, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(
        self, x, weight, bias, mailbox, logits, topk_ids, topk_weights, m: Int32,
        eps: Float32, route_scale: Float32,
    ):  # fmt: skip
        tid, warp, lane, cta = thread()
        part, group = cta % PARTS, cta // PARTS
        rows: cutlass.Constexpr = self.rows
        smem = cutlass.memory.SmemAllocator()
        # Sized for 16 FP32 rows either way: more than half the SM's shared
        # memory keeps one CTA per SM.
        staged = alloc(smem, Uint8, 2 * 2 * TILE * (CHUNK + 16) * 4, 128).iterator
        sums = alloc(smem, Float32, WARPS * GROUP * rows, 16).iterator

        # The weights do not depend on the predecessor: into registers now.
        # Lane (g, q) of warp v holds experts g and g + 8's K elements
        # 16 s + 4 q + [0, 4) of the warp's K slice, for each step s.
        g, q = lane >> 2, lane & 3
        column = part * CHUNK + warp * WARP_K + 4 * q
        first = Int64((group * GROUP + g) * HIDDEN + column) * 4
        source = weight.iterator.toint() + first
        w = []
        for s in cutlass.range_constexpr(STEPS):
            w.append(load_weights(source + s * 64))
            w.append(load_weights(source + 8 * HIDDEN * 4 + s * 64))
        source = bias.iterator.toint() + lane * 32
        biases = list(load_weights(source)) + list(load_weights(source + 16))
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        passes = (m + rows - 1) // rows
        self.stage(x, staged, m, part, 0)
        # A fragments of the three terms: row g is expert g, row g + 8 is
        # expert g + 8.
        terms = [[], [], []]
        for s in cutlass.range_constexpr(STEPS):
            top, bottom = w[2 * s], w[2 * s + 1]
            t01 = split3(top[0], top[1])
            t23 = split3(top[2], top[3])
            b01 = split3(bottom[0], bottom[1])
            b23 = split3(bottom[2], bottom[3])
            for i in cutlass.range_constexpr(3):
                terms[i].append((t01[i], b01[i], t23[i], b23[i]))
        for c in cutlass.range(passes, unroll=1):
            if c + 1 < passes:  # the next rows load while these multiply
                following = staged + (c + 1) % 2 * (rows * self.pitch)
                self.stage(x, following, m, part, c + 1)
                cp_async_wait(1)
            else:
                cp_async_wait(0)
            cute.arch.sync_threads()
            self.multiply(staged + c % 2 * (rows * self.pitch), terms, sums)
            cute.arch.sync_threads()
            # This CTA's partial logits of the pass, its warps' K slices added
            # in order.
            for item in cutlass.range_constexpr(
                (GROUP * rows + THREADS - 1) // THREADS
            ):
                index = tid + THREADS * item
                e, r = index % GROUP, index // GROUP
                token = c * rows + r
                if (index < GROUP * rows) & (token < m):
                    total = load_shared_f32(sums + e * rows + r)
                    for v in cutlass.range_constexpr(1, WARPS):
                        more = load_shared_f32(sums + (v * GROUP + e) * rows + r)
                        total = add(total, more)
                    cell = (part * MAX_ROWS + token) * EXPERTS + group * GROUP + e
                    store_global_f32(mailbox.iterator.toint() + Int64(cell) * 4, total)

        # One row a CTA (two past 128 rows): the polls stay off each other's
        # shared memory bandwidth.
        row = warp * (GROUPS * PARTS) + cta
        if (row < m) & (warp < 2):
            self.route(
                mailbox, logits, biases, topk_ids, topk_weights, row, eps,
                route_scale,
            )  # fmt: skip

    @cute.jit
    def stage(self, x, buffer, m, part, c):
        """Start loading activation rows [c, c + 1) * rows (those below m), K
        eighth ``part``, into shared ``buffer`` (one cp.async group)."""
        tid, _, _, _ = thread()
        pieces: cutlass.Constexpr = CHUNK * self.size // 16
        count: cutlass.Constexpr = self.rows * pieces
        for i in cutlass.range_constexpr((count + THREADS - 1) // THREADS):
            item = tid + THREADS * i
            r, piece = item // pieces, item % pieces
            token = c * self.rows + r
            if (item < count) & (token < m):
                src = x.iterator + (token * x.layout.stride[0] + part * CHUNK)
                address = src.toint() + piece * 16
                copy_g2s(buffer + r * self.pitch + piece * 16, address)
        cp_async_commit()

    @cute.jit
    def multiply(self, buffer, terms, sums):
        """This warp's K slice of 16 experts x the staged rows into ``sums``
        [warp, expert, row]: the high term's products in one accumulator, the
        smaller ones in another."""
        _, warp, lane, _ = thread()
        g, q = lane >> 2, lane & 3
        for n in cutlass.range_constexpr(self.tiles):
            high = [Float32(0)] * 4
            rest = [Float32(0)] * 4
            for s in cutlass.range_constexpr(STEPS):
                element = warp * WARP_K + s * 16 + 4 * q
                at_b = (n * TILE + g) * self.pitch + element * self.size
                if cutlass.const_expr(self.fp32):
                    xf = load_shared_f32x4(buffer + at_b)
                    x01 = split3(xf[0], xf[1])
                    x23 = split3(xf[2], xf[3])
                    hi_x, mid_x, lo_x = ((x01[i], x23[i]) for i in range(3))
                    high = list(mma(terms[0][s], hi_x, high))
                    rest = list(mma(terms[1][s], hi_x, rest))
                    rest = list(mma(terms[2][s], hi_x, rest))
                    rest = list(mma(terms[0][s], mid_x, rest))
                    rest = list(mma(terms[1][s], mid_x, rest))
                    rest = list(mma(terms[0][s], lo_x, rest))
                else:
                    b = load_shared_x2(buffer + at_b)
                    high = list(mma(terms[0][s], b, high))
                    rest = list(mma(terms[1][s], b, rest))
                    rest = list(mma(terms[2][s], b, rest))
            # c0, c1: expert g, rows 2 q, 2 q + 1; c2, c3: expert g + 8.
            for i in cutlass.range_constexpr(4):
                e = g + 8 * (i // 2)
                r = n * TILE + 2 * q + i % 2
                offset = (warp * GROUP + e) * self.rows + r
                store_shared_f32(sums + offset, add(high[i], rest[i]))

    @cute.jit
    def route(
        self, mailbox, logits, biases, topk_ids, topk_weights, row, eps, route_scale
    ):  # fmt: skip
        """One warp: ``fused_musespark1x_routing`` of one row once its eight
        partials hold no sentinel (then re-armed); lane l holds experts
        [8 l, 8 l + 8)."""
        _, _, lane, _ = thread()
        base = mailbox.iterator.toint() + Int64(row * EXPERTS + lane * 8) * 4
        stride: cutlass.Constexpr = MAX_ROWS * EXPERTS * 4
        words = self.fetch(base, stride)
        while self.any_empty(words):
            words = self.fetch(base, stride)
        for p in cutlass.range_constexpr(PARTS):
            empty_mailbox(base + p * stride)
            empty_mailbox(base + p * stride + 16)
        values = [bits_float(words[j]) for j in range(8)]
        for p in cutlass.range_constexpr(1, PARTS):
            for j in cutlass.range_constexpr(8):
                values[j] = add(values[j], bits_float(words[8 * p + j]))
        if cutlass.const_expr(self.keep_logits):
            out = logits.iterator.toint() + Int64(row * EXPERTS + lane * 8) * 4
            for j in cutlass.range_constexpr(8):
                store_global_f32(out + j * 4, values[j])
        scores = [sigmoid(v) for v in values]
        # Unique keys: the biased score's order, then the lower expert.
        keys = [
            (Int64(ordered(add(scores[j], biases[j]))) << 8) | (255 - lane * 8 - j)
            for j in range(8)
        ]
        for i, j in BATCHER_8:  # this lane's keys, descending
            swap = keys[j] > keys[i]
            keys[i], keys[j] = (
                cutlass.select_(swap, keys[j], keys[i]),
                cutlass.select_(swap, keys[i], keys[j]),
            )
            scores[i], scores[j] = (
                cutlass.select_(swap, scores[j], scores[i]),
                cutlass.select_(swap, scores[i], scores[j]),
            )
        winners, heads, picked = [], [], []
        for _ in cutlass.range_constexpr(TOP_K):
            head = keys[0]
            top = warp_max(Int32(head >> 8))
            ballot = cute.arch.vote_ballot_sync(Int32(head >> 8) == top)
            winner = Int32(cute.arch.popc((ballot & (0 - ballot)) - 1))
            winners.append(winner)
            heads.append(255 - Int32(head & 255))
            picked.append(scores[0])
            won = lane == winner
            for i in cutlass.range_constexpr(7):  # the winner's next key leads
                keys[i] = cutlass.select_(won, keys[i + 1], keys[i])
                scores[i] = cutlass.select_(won, scores[i + 1], scores[i])
            keys[7] = cutlass.select_(won, Int64(NONE) << 8, keys[7])
        total = Float32(0)
        my_id = Int32(0)
        my_score = Float32(0)
        for k in cutlass.range_constexpr(TOP_K):
            chosen = cute.arch.shuffle_sync(heads[k], winners[k])
            score = cute.arch.shuffle_sync(picked[k], winners[k])
            total = add(total, score)
            my_id = cutlass.select_(lane == k, chosen, my_id)
            my_score = cutlass.select_(lane == k, score, my_score)
        scale = divide(route_scale, add(total, eps))
        if lane < TOP_K:
            topk_ids[row, lane] = my_id
            topk_weights[row, lane] = mul(my_score, scale)

    @cute.jit
    def fetch(self, base, stride):
        """This lane's eight words of every part."""
        words = []
        for p in cutlass.range_constexpr(PARTS):
            words += list(load_mailbox(base + p * stride))
            words += list(load_mailbox(base + p * stride + 16))
        return words

    @cute.jit
    def any_empty(self, words):
        """Whether any lane still holds a sentinel."""
        hole = words[0] == SENTINEL
        for w in words[1:]:
            hole = hole | (w == SENTINEL)
        return cute.arch.vote_any_sync(hole)


@functools.cache
def compile_router(tiles: int, fp32: bool, keep_logits: bool, device_index: int):
    """Code-only cache: every pointer and the row count are arguments."""
    sym = cute.sym_int
    dtype = Float32 if fp32 else BFloat16

    def matrix(dt, cols, align=16):
        stride = (sym(divisibility=max(1, align * 8 // dt.width)), 1)
        return make_fake_tensor(dt, (sym(), cols), stride, assumed_align=align)

    def flat(dt):
        return make_fake_tensor(dt, (sym(),), (1,), assumed_align=16)

    args = (
        matrix(dtype, HIDDEN),
        make_fake_tensor(Float32, (EXPERTS, HIDDEN), (HIDDEN, 1), assumed_align=16),
        make_fake_tensor(Float32, (EXPERTS,), (1,), assumed_align=16),
        flat(Float32),
        flat(Float32),
        matrix(Int32, TOP_K, 4),
        matrix(Float32, TOP_K, 4),
        Int32(1),
        Float32(1e-15),
        Float32(1.0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    kernel = RouterKernel(tiles, fp32, keep_logits)
    with torch.accelerator.device_index(device_index):
        return cute.compile(kernel, *args, options="--enable-tvm-ffi")


@functools.cache
def mailboxes(device: torch.device) -> torch.Tensor:
    """The partial-logits mailboxes [8 parts, 256 rows, 256 experts] of every
    router on ``device``, empty (``SENTINEL``) between launches: one router runs
    at a time in stream order, and each routing warp re-arms its row."""
    empty = torch.full((PARTS * MAX_ROWS * EXPERTS,), SENTINEL, dtype=torch.int32)
    return empty.to(device).view(torch.float32)


def tiles_per_pass(rows: int, fp32: bool = False) -> int:
    """MMA tiles per staging pass: the shared staging buffer holds 32 BF16 rows
    (or 16 FP32 ones), and past ``GATED_PASS_ROWS`` a wider pass halves the
    number of barrier-and-staging round trips."""
    if not 0 < rows <= MAX_ROWS:
        raise ValueError(f"MuseRouter serves 1..{MAX_ROWS} rows, got {rows}")
    if rows <= TILE:
        return 1
    return 2 if fp32 or rows <= GATED_PASS_ROWS else 4


class MuseRouter:
    """One MoE layer's router; ``prepare`` compiles the kernels for the row
    counts and activation dtypes ahead of CUDA graph capture.

    Args:
        weight: FP32 [256, 8192] router weight (output multiplier folded in).
        expert_bias: FP32 [256] selection bias.
        top_k: Experts per row (8).
        route_scale: Scale of the renormalized weights.
        eps: Added to the selected scores' sum.

    """

    def __init__(
        self,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        top_k: int = TOP_K,
        route_scale: float = 1.0,
        eps: float = 1e-15,
    ):
        if top_k != TOP_K or weight.shape != (EXPERTS, HIDDEN):
            raise ValueError(f"expected top-{TOP_K} of [{EXPERTS}, {HIDDEN}]")
        if weight.dtype != torch.float32 or not weight.is_contiguous():
            raise ValueError("the router weight must be contiguous FP32")
        if expert_bias.dtype != torch.float32 or expert_bias.shape != (EXPERTS,):
            raise ValueError("the expert bias must be FP32 [256]")
        self.weight, self.expert_bias = weight, expert_bias.contiguous()
        self.route_scale, self.eps = route_scale, eps
        self.device = weight.device
        self.mailbox = mailboxes(self.device)

    def kernel(self, rows: int, dtype: torch.dtype, keep_logits: bool = False):
        fp32 = dtype == torch.float32
        tiles = tiles_per_pass(rows, fp32)
        return compile_router(tiles, fp32, keep_logits, self.device.index)

    def prepare(self, rows, dtypes=(torch.bfloat16, torch.float32)) -> "MuseRouter":
        """Compile the kernels serving each of ``rows`` and ``dtypes``."""
        for r in rows:
            for dtype in dtypes:
                self.kernel(r, dtype)
        return self

    def __call__(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route ``x``'s rows; returns ``(topk_ids, topk_weights)``.

        Args:
            x: BF16 or FP32 [M, 8192] router input, unit column stride.
            topk_ids: int32 [M, 8] output, unit column stride.
            topk_weights: FP32 [M, 8] output, unit column stride.
            logits: Optional FP32 [M, 256] contiguous output of the logits.

        """
        m = x.shape[0]
        assert x.shape[1] == HIDDEN and x.stride(1) == 1
        assert x.dtype in (torch.bfloat16, torch.float32)
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        assert topk_ids.shape == topk_weights.shape == (m, TOP_K)
        if logits is not None:
            assert logits.shape == (m, EXPERTS) and logits.is_contiguous()
        self.kernel(m, x.dtype, logits is not None)(
            x, self.weight, self.expert_bias, self.mailbox,
            self.mailbox if logits is None else logits.view(-1), topk_ids, topk_weights,
            m, self.eps, self.route_scale,
        )  # fmt: skip
        return topk_ids, topk_weights
