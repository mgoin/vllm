# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The operand and scale-factor layouts, TMA partitioning and MMA issue follow
# vLLM's DeepSeek-V4.1 routed-expert GEMM (routed_expert_gemm_cute.py), itself
# modified from LowLatencyBlockscaledGemmKernel in FlashInfer
# (flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py), which
# carries the notice below.

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Muse Spark NVFP4 routed experts (GEMM1, SwiGLU, GEMM2) in one persistent
CuTe DSL kernel for decode batches of 1-64 tokens at TP4.

Per rank: 256 experts, ``w13`` [2048, 4096] and ``w2`` [4096, 1024] NVFP4
(E2M1 with an E4M3 scale per 16 elements and FP32 per-expert global scales),
top-8 routing. The kernel reads the tensors vLLM prepares for the TRT-LLM-gen
NVFP4 MoE (rows shuffled for 128-row epilogue tiles, ``w13`` rows interleaving
up and gate, scales in the 128x4 layout) and computes, bit for bit, the GEMM2
rows of FlashInfer's ``trtllm_fp4_block_scale_routed_moe`` with
``do_finalize=False`` (its ``t128x8`` tactics), but in route-major order: row
``8 t + k`` holds route (token ``t``, slot ``k``).

Work: every CTA builds the call's routing table from ``topk_ids`` in shared
memory (the touched experts in ascending id order as *slots*, each slot's
tokens in ascending order). The work items are, for every slot, 16 GEMM1 items
(128 ``w13`` rows = 64 intermediate columns, K = 4096) then, after all GEMM1
items, 32 GEMM2 items (128 hidden rows, K = 1024); CTAs take them from a global
counter, so the GEMM2 weights of the experts whose intermediate is done stream
while the last GEMM1 items finish. An item multiplies its 128 weight rows by
``N`` activation rows (N = the call's token count rounded up to 8, 16, 32 or
64): GEMM1 by every token (the epilogue keeps the slot's tokens), GEMM2 by the
slot's intermediate rows. The MMAs are ``tcgen05`` ``kind::mxf4nvf4`` 128 x N x
64 in ascending K, without split-K, the order of the TRT-LLM-gen kernels.

GEMM1's epilogue applies SwiGLU with the per-expert gate and up scales and
quantizes the intermediate H to NVFP4 (E4M3 block scales of amax / 6), with
the instruction sequence of the TRT-LLM-gen epilogue, writes H and its scales
to a workspace and releases the slot's counter; the slot's GEMM2 items load H
(TMA) once the counter reaches 16. GEMM2's epilogue scales by the per-expert
output scale and stores BF16 rows.

Warps: 0 schedules the items and loads A (weights) with evict-first hints, 1
loads B (activations, H) and B's scales, 2 issues the MMAs, 4-7 run the
epilogue; at the start every warp builds the routing table and warps 3-7 stage
the input scales. Programmatic dependent launch: every CTA reads the routing
only after its grid dependency wait. Reading it earlier is unsafe even when
the router runs on another stream: inside a CUDA graph capture the edge from
an event-joined node also becomes programmatic, and without the aux stream the
router is the kernel right before this one. CTAs that read different ids
would build different tables and wait forever for each other's work.

Degenerate routing is tolerated: an id outside [0, 256) is a no-op route whose
row is zero; an expert repeated within a token's top-8 gets one H row, and
each of its routes the same GEMM2 row.

The last CTA to finish resets the work counter and the slot counters in the
workspace, so consecutive calls need no memset.
"""

from __future__ import annotations

import os
from functools import wraps

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op

HIDDEN = 4096
INTER = 1024
EXPERTS = 256
TOP_K = 8
MAX_ROWS = 64
CTA_M = 128
CTA_K = 256
SF_VEC = 16
G1_M_TILES = 2 * INTER // CTA_M  # 16: 64 intermediate columns each
G2_M_TILES = HIDDEN // CTA_M  # 32
G1_K_TILES = HIDDEN // CTA_K  # 16
G2_K_TILES = INTER // CTA_K  # 4
G1_ITEMS = G1_M_TILES
SLOT_ITEMS = G1_M_TILES + G2_M_TILES
# H block scales of one slot: one 128x4 atom (512 B) per 64 intermediate columns.
HS_SLOT_BYTES = INTER // 64 * 512
STAGE_SFB_BYTES = CTA_K // 64 * 512
RING = 8
# Workspace words: work counter, finished CTAs, then one counter per slot.
WS_QUEUE = 0
WS_DONE = 1
WS_FLAGS = 2
WS_WORDS = WS_FLAGS + EXPERTS
EVICT_FIRST = 0x12F0000000000000
THREADS = 256
EPI_WARP = 4


def _ptx(code, constraints, operands, result, effects, loc, ip):
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([t.mlir_type for t in result])
    else:
        ir_type = result.mlir_type
    value = llvm.inline_asm(
        ir_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        code,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            t(llvm.extractvalue(t.mlir_type, value, [i], loc=loc, ip=ip))
            for i, t in enumerate(result)
        )
    return result(value)


def asm(result=None, *, pure=False):
    """``f(*args) -> (code, constraints, operands)`` as one inline-PTX op.
    Pointer operands pass their address (32-bit for "r", 64-bit for "l")."""

    def wrap(fn):
        @dsl_user_op
        @wraps(fn)
        def op(*args, loc=None, ip=None, **kwargs):
            code, constraints, operands = fn(*args, **kwargs)
            kinds = [c for c in constraints.split(",") if c[:1] not in ("=", "~")]
            values = []
            for kind, v in zip(kinds, operands):
                if hasattr(v, "toint"):
                    v = v.toint(loc=loc, ip=ip)
                    v = Int32(v) if kind == "r" else v
                values.append(v)
            return _ptx(code, constraints, values, result, not pure, loc, ip)

        return op

    return wrap


@asm((Float32,) * 8)
def tmem_load_8x8(top, bottom):
    """Rows (lane / 4, + 8) of two 16-lane halves, eight columns: ``top`` and
    ``bottom`` are the halves' tensor-memory addresses."""
    code = (
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$8];\n"
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$4,$5,$6,$7}, [$9];\n"
        "tcgen05.wait::ld.sync.aligned;"
    )
    return code, "=f,=f,=f,=f,=f,=f,=f,=f,r,r", (Uint32(top), Uint32(bottom))


# SwiGLU of two tokens of one intermediate column, as the TRT-LLM-gen GEMM1
# epilogue computes it with no clamp limit (FLT_MAX), alpha 1 and beta 0:
# up * scale_gate * (gate * sigmoid(gate * scale_gate)). $6 is
# scale_gate * log2(e) (FTZ), $7 is scale_gate.
_SWIGLU_X2 = """{
.reg .pred p;
.reg .f32 g0, g1, u0, u1, a0, a1;
setp.gt.ftz.f32 p, $4, 0f7F7FFFFF;
selp.f32 g0, 0f7F7FFFFF, $4, p;
setp.geu.ftz.f32 p, g0, 0fFF7FFFFF;
selp.f32 g0, g0, 0fFF7FFFFF, p;
setp.gt.ftz.f32 p, $5, 0f7F7FFFFF;
selp.f32 g1, 0f7F7FFFFF, $5, p;
setp.geu.ftz.f32 p, g1, 0fFF7FFFFF;
selp.f32 g1, g1, 0fFF7FFFFF, p;
setp.gt.ftz.f32 p, $2, 0f7F7FFFFF;
selp.f32 u0, 0f7F7FFFFF, $2, p;
setp.geu.ftz.f32 p, u0, 0fFF7FFFFF;
selp.f32 u0, u0, 0fFF7FFFFF, p;
setp.gt.ftz.f32 p, $3, 0f7F7FFFFF;
selp.f32 u1, 0f7F7FFFFF, $3, p;
setp.geu.ftz.f32 p, u1, 0fFF7FFFFF;
selp.f32 u1, u1, 0fFF7FFFFF, p;
mul.rn.f32 a0, g0, $6;
mul.rn.f32 a1, g1, $6;
mul.rn.f32 a0, a0, 0fBF800000;
mul.rn.f32 a1, a1, 0fBF800000;
ex2.approx.ftz.f32 a0, a0;
ex2.approx.ftz.f32 a1, a1;
add.rn.f32 a0, a0, 0f3F800000;
add.rn.f32 a1, a1, 0f3F800000;
rcp.approx.ftz.f32 a0, a0;
rcp.approx.ftz.f32 a1, a1;
fma.rn.f32 u0, u0, $7, 0f00000000;
fma.rn.f32 u1, u1, $7, 0f00000000;
mul.rn.f32 a0, g0, a0;
mul.rn.f32 a1, g1, a1;
mul.rn.f32 $0, u0, a0;
mul.rn.f32 $1, u1, a1;
}"""


@asm((Float32, Float32), pure=True)
def swiglu_x2(up0, up1, gate0, gate1, k, scale):
    operands = tuple(Float32(v) for v in (up0, up1, gate0, gate1, k, scale))
    return _SWIGLU_X2, "=f,=f,f,f,f,f,f,f", operands


@asm(Float32, pure=True)
def mul_ftz(a, b):
    return "mul.ftz.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm((Float32, Float32), pure=True)
def block_amax_x2(a0, a1, b0, b1, scale):
    """Per token (a, b): ``max(0, |max(|x0|, |x1|) * scale|)``, FTZ maxima."""
    code = """{
.reg .f32 x, y;
abs.ftz.f32 x, $2;
abs.ftz.f32 y, $3;
max.ftz.f32 x, x, y;
abs.ftz.f32 y, $4;
abs.ftz.f32 $1, $5;
max.ftz.f32 y, y, $1;
mul.rn.f32 x, x, $6;
mul.rn.f32 y, y, $6;
abs.ftz.f32 x, x;
abs.ftz.f32 y, y;
max.ftz.f32 $0, x, 0f00000000;
max.ftz.f32 $1, y, 0f00000000;
}"""
    operands = tuple(Float32(v) for v in (a0, a1, b0, b1, scale))
    return code, "=f,=f,f,f,f,f,f", operands


@asm(Float32, pure=True)
def max_ftz(a, b):
    return "max.ftz.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm((Uint32, Float32, Float32), pure=True)
def block_scale_x2(amax_a, amax_b):
    """E4M3 block scales amax / 6 of two tokens (byte 0: a, byte 1: b) and
    their reciprocals (0 for a zero scale)."""
    code = """{
.reg .pred p;
.reg .b16 s;
.reg .b32 h;
.reg .f16 lo, hi;
.reg .f32 x, y;
mul.rn.f32 x, $3, 0f3E2AAAAB;
mul.rn.f32 y, $4, 0f3E2AAAAB;
cvt.rn.satfinite.e4m3x2.f32 s, y, x;
cvt.u32.u16 $0, s;
cvt.rn.f16x2.e4m3x2 h, s;
mov.b32 {lo, hi}, h;
cvt.f32.f16 x, lo;
cvt.f32.f16 y, hi;
rcp.approx.ftz.f32 $1, x;
setp.neu.ftz.f32 p, x, 0f00000000;
selp.f32 $1, $1, 0f00000000, p;
rcp.approx.ftz.f32 $2, y;
setp.neu.ftz.f32 p, y, 0f00000000;
selp.f32 $2, $2, 0f00000000, p;
}"""
    return code, "=r,=f,=f,f,f", (Float32(amax_a), Float32(amax_b))


@asm(Uint32, pure=True)
def quantize_e2m1x2(lo, hi, inverse, scale):
    """``(x * inverse) * scale`` of two values as one E2M1 byte, ``lo`` in
    the low nibble."""
    code = """{
.reg .b8 q;
.reg .f32 x, y;
mul.rn.f32 x, $1, $3;
mul.rn.f32 y, $2, $3;
mul.rn.f32 x, x, $4;
mul.rn.f32 y, y, $4;
cvt.rn.satfinite.e2m1x2.f32 q, y, x;
cvt.u32.u8 $0, q;
}"""
    operands = tuple(Float32(v) for v in (lo, hi, inverse, scale))
    return code, "=r,f,f,f,f", operands


@asm(Uint32, pure=True)
def scaled_bf16x2(lo, hi, scale):
    """``lo * scale`` and ``hi * scale`` as saturating BF16, ``lo`` low."""
    code = """{
.reg .f32 x, y;
mul.rn.f32 x, $1, $3;
mul.rn.f32 y, $2, $3;
cvt.rn.satfinite.bf16x2.f32 $0, y, x;
}"""
    return code, "=r,f,f,f", (Float32(lo), Float32(hi), Float32(scale))


@asm()
def store_u8(address, value):
    return "st.global.u8 [$0], $1;", "l,r,~{memory}", (address, Uint32(value))


@asm()
def store_v2(address, a, b):
    code = "st.global.v2.b32 [$0], {$1, $2};"
    return code, "l,r,r,~{memory}", (address, Uint32(a), Uint32(b))


@asm()
def store_zero_v2(address):
    return "st.global.v2.b32 [$0], {$1, $1};", "l,r,~{memory}", (address, Uint32(0))


@asm()
def trap():
    return "trap;", "~{memory}", ()


@asm(Int32)
def load_acquire(address):
    return "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", (address,)


@asm()
def release_add(address, value):
    """Make this thread's (and its CTA's synchronized) global writes visible
    to the async proxy and to GPU-scope acquirers, then add ``value``."""
    code = "fence.proxy.async.global;\nred.release.gpu.global.add.s32 [$0], $1;"
    return code, "l,r,~{memory}", (address, Int32(value))


@asm()
def fence_proxy_async_global():
    return "fence.proxy.async.global;", "~{memory}", ()


@asm(Int32)
def atomic_add_relaxed(address, value):
    code = "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;"
    return code, "=r,l,r,~{memory}", (address, Int32(value))


@asm(Int32)
def atomic_add_acq_rel(address, value):
    code = "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;"
    return code, "=r,l,r,~{memory}", (address, Int32(value))


@asm()
def store_relaxed(address, value):
    return (
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r,~{memory}",
        (address, Int32(value)),
    )


@asm()
def bulk_load(dst, src, size, barrier):
    """``size`` bytes global -> shared, completing on ``barrier``."""
    code = (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];"
    )
    return code, "r,l,r,r,~{memory}", (dst, src, Int32(size), barrier)


@asm()
def prefetch_l2(base, offset, size):
    code = (
        "{ .reg .b64 a; add.s64 a, $0, $1; cp.async.bulk.prefetch.L2.global [a], $2; }"
    )
    return code, "l,l,r", (base, Int64(offset), Int32(size))


@asm()
def store_shared_u32(address, value):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (address, Uint32(value))


@cute.jit
def tile_rank(mask: Uint64, token: Int32) -> Int32:
    """Tokens of ``mask`` below ``token``."""
    below = (Uint64(1) << Uint64(token)) - Uint64(1)
    return Int32(cute.arch.popc(mask & below))


class RoutedExpertsKernel:
    """The persistent kernel (module docstring) for N-column activation tiles
    (``n`` in 8, 16, 32, 64: calls of up to ``n`` tokens)."""

    def __init__(self, n: int, ids_ready: bool, num_stages: int) -> None:
        assert n in (8, 16, 32, 64)
        self.cta_n = n
        # Accepted for compatibility; the routing is always read after the
        # grid dependency wait (module docstring).
        self.ids_ready = ids_ready
        # Debug: trap after this many polls of a slot counter (0: never).
        self.spin_limit = int(os.environ.get("VLLM_MUSE_EXPERTS_SPIN_LIMIT", "0"))
        self.num_stages = num_stages
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.num_acc_stage = 2
        self.tmem_cols = 32
        while self.tmem_cols < 2 * n + 32:
            self.tmem_cols *= 2

    def _tiled_mma(self, mn: tuple) -> cute.TiledMma:
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            mn,
        )

    def _setup_attributes(self) -> None:
        tiled_mma = self._tiled_mma((CTA_M, self.cta_n))
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = CTA_K // mma_inst_shape_k
        self.num_acc_tmem_cols = self.cta_n * self.num_acc_stage
        self.num_sfa_tmem_cols = (CTA_M // 32) * self.mma_inst_tile_k
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        tile_mnk = (CTA_M, self.cta_n, CTA_K)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, tile_mnk, self.a_dtype, self.num_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, tile_mnk, self.b_dtype, self.num_stages
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, tile_mnk, SF_VEC, self.num_stages
        )
        # B's scales: one 128-row atom per 64 columns of K.
        self.mma_tiler_sfb = (CTA_M, 128, CTA_K)
        self.tiled_mma_sfb = self._tiled_mma((CTA_M, 128))
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma_sfb, self.mma_tiler_sfb, SF_VEC, self.num_stages
        )
        # The input's scales for every K tile of GEMM1, resident.
        self.sfx_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma_sfb, self.mma_tiler_sfb, SF_VEC, G1_K_TILES
        )

    @cute.jit
    def __call__(
        self,
        w13_ptr: cute.Pointer,
        w13s_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2s_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        g1a_ptr: cute.Pointer,
        g1c_ptr: cute.Pointer,
        g2a_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        hb_ptr: cute.Pointer,
        hs_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """``w13`` [E, 2048, 4096] / ``w2`` [E, 4096, 1024] E2M1 with E4M3
        scales in the 128x4 layout; ``x`` [rows, 4096] E2M1 with linear E4M3
        scales (``xs`` as Uint32 words); ``ids`` int32 [rows, 8]; FP32 [E]
        gate, up and output scales; H [E * N, 1024] E2M1 (``h`` and ``hb``
        the same buffer), HS [E, 8192] bytes; ``ws`` the int32 workspace;
        ``out`` [rows * 8, 4096] BF16 as Uint32 words."""
        self.a_dtype = w13_ptr.value_type
        self.b_dtype = x_ptr.value_type
        self.sf_dtype = w13s_ptr.value_type
        self.a_major_mode = tcgen05.OperandMajorMode.K
        self.b_major_mode = tcgen05.OperandMajorMode.K
        self._setup_attributes()
        tiled_mma = self._tiled_mma((CTA_M, self.cta_n))
        tile_mnk = (CTA_M, self.cta_n, CTA_K)

        w13 = cute.make_tensor(
            w13_ptr,
            cute.make_ordered_layout((2 * INTER, HIDDEN, EXPERTS), order=(1, 0, 2)),
        )
        w13s = cute.make_tensor(
            w13s_ptr, blockscaled_utils.tile_atom_to_shape_SF(w13.shape, SF_VEC)
        )
        w2 = cute.make_tensor(
            w2_ptr,
            cute.make_ordered_layout((HIDDEN, INTER, EXPERTS), order=(1, 0, 2)),
        )
        w2s = cute.make_tensor(
            w2s_ptr, blockscaled_utils.tile_atom_to_shape_SF(w2.shape, SF_VEC)
        )
        x = cute.make_tensor(
            x_ptr, cute.make_ordered_layout((rows, HIDDEN, 1), order=(1, 0, 2))
        )
        h = cute.make_tensor(
            h_ptr,
            cute.make_ordered_layout((EXPERTS * self.cta_n, INTER, 1), order=(1, 0, 2)),
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        shape = self.cluster_layout_vmnk.shape
        tma_a1, tma_tensor_a1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w13, a_smem_layout, tile_mnk, tiled_mma, shape
        )
        tma_sfa1, tma_tensor_sfa1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w13s,
            sfa_smem_layout,
            tile_mnk,
            tiled_mma,
            shape,
            internal_type=cutlass.Int16,
        )
        tma_a2, tma_tensor_a2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w2, a_smem_layout, tile_mnk, tiled_mma, shape
        )
        tma_sfa2, tma_tensor_sfa2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w2s,
            sfa_smem_layout,
            tile_mnk,
            tiled_mma,
            shape,
            internal_type=cutlass.Int16,
        )
        tma_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, x, b_smem_layout, tile_mnk, tiled_mma, shape
        )
        tma_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, h, b_smem_layout, tile_mnk, tiled_mma, shape
        )
        self.tma_bytes_a = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.tma_bytes_b = cute.size_in_bytes(self.b_dtype, b_smem_layout)

        xs = cute.make_tensor(xs_ptr, cute.make_layout(rows * (HIDDEN // 64)))
        ids = cute.make_tensor(ids_ptr, cute.make_layout(rows * TOP_K))
        g1a = cute.make_tensor(g1a_ptr, cute.make_layout(EXPERTS))
        g1c = cute.make_tensor(g1c_ptr, cute.make_layout(EXPERTS))
        g2a = cute.make_tensor(g2a_ptr, cute.make_layout(EXPERTS))
        hb = cute.make_tensor(
            hb_ptr, cute.make_layout(EXPERTS * self.cta_n * (INTER // 2))
        )
        hs = cute.make_tensor(hs_ptr, cute.make_layout(EXPERTS * HS_SLOT_BYTES))
        ws = cute.make_tensor(ws_ptr, cute.make_layout(WS_WORDS))
        out = cute.make_tensor(out_ptr, cute.make_layout(rows * TOP_K * (HIDDEN // 2)))
        self.kernel(
            tiled_mma,
            tma_a1,
            tma_tensor_a1,
            tma_sfa1,
            tma_tensor_sfa1,
            tma_a2,
            tma_tensor_a2,
            tma_sfa2,
            tma_tensor_sfa2,
            tma_b1,
            tma_tensor_b1,
            tma_b2,
            tma_tensor_b2,
            (xs, ids, g1a, g1c, g2a, hb, hs, ws, out),
            w2_ptr,
            w2s_ptr,
            rows,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfx_smem_layout,
        ).launch(
            grid=(ctas, 1, 1),
            block=[THREADS, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_sfa1: cute.CopyAtom,
        mSFA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_sfa2: cute.CopyAtom,
        mSFA2: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        tma_b2: cute.CopyAtom,
        mB2: cute.Tensor,
        side: tuple,
        w2_ptr: cute.Pointer,
        w2s_ptr: cute.Pointer,
        rows: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfx_smem_layout: cute.Layout,
    ):
        mXS, mIds, mG1a, mG1c, mG2a, mHB, mHS, mWS, mOut = side
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        grid, _, _ = cute.arch.grid_dim()
        n_cols = self.cta_n
        stages = self.num_stages

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a1)
            cpasync.prefetch_descriptor(tma_sfa1)
            cpasync.prefetch_descriptor(tma_a2)
            cpasync.prefetch_descriptor(tma_sfa2)
            cpasync.prefetch_descriptor(tma_b1)
            cpasync.prefetch_descriptor(tma_b2)

        @cute.struct
        class SharedStorage:
            ab_full: cute.struct.MemRange[Int64, stages]
            ab_empty: cute.struct.MemRange[Int64, stages]
            acc_full: cute.struct.MemRange[Int64, 2]
            acc_empty: cute.struct.MemRange[Int64, 2]
            ring_full: cute.struct.MemRange[Int64, RING]
            ring_empty: cute.struct.MemRange[Int64, RING]
            xs_ready: Int64
            tmem_bar: Int64
            tmem_base: cute.struct.MemRange[Int32, 1]
            ring: cute.struct.MemRange[Int32, RING]
            n_slots: cute.struct.MemRange[Int32, 1]
            # By expert: token masks (two words), H row base.
            mask_lo: cute.struct.MemRange[Uint32, EXPERTS]
            mask_hi: cute.struct.MemRange[Uint32, EXPERTS]
            base_of: cute.struct.MemRange[Int32, EXPERTS]
            # By slot: expert, compact row base, token mask.
            slot_expert: cute.struct.MemRange[Int32, EXPERTS]
            slot_base: cute.struct.MemRange[Int32, EXPERTS]
            slot_mask: cute.struct.MemRange[Uint64, EXPERTS]
            warp_sums: cute.struct.MemRange[Int32, 16]
            # Compact row (slot base + rank) -> route 8 t + k.
            row_route: cute.struct.MemRange[Int32, MAX_ROWS * TOP_K]
            # Route -> the next route of the same token and expert, or -1.
            dup_next: cute.struct.MemRange[Int32, MAX_ROWS * TOP_K]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_full = storage.ab_full.data_ptr()
        ab_empty = storage.ab_empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        ring_full = storage.ring_full.data_ptr()
        ring_empty = storage.ring_empty.data_ptr()
        xs_ready = storage.xs_ready.ptr
        tmem_bar = storage.tmem_bar.ptr
        ring = cute.make_tensor(storage.ring.data_ptr(), cute.make_layout(RING))
        mask_lo = cute.make_tensor(
            storage.mask_lo.data_ptr(), cute.make_layout(EXPERTS)
        )
        mask_hi = cute.make_tensor(
            storage.mask_hi.data_ptr(), cute.make_layout(EXPERTS)
        )
        base_of = cute.make_tensor(
            storage.base_of.data_ptr(), cute.make_layout(EXPERTS)
        )
        slot_expert = cute.make_tensor(
            storage.slot_expert.data_ptr(), cute.make_layout(EXPERTS)
        )
        slot_base = cute.make_tensor(
            storage.slot_base.data_ptr(), cute.make_layout(EXPERTS)
        )
        slot_mask = cute.make_tensor(
            storage.slot_mask.data_ptr(), cute.make_layout(EXPERTS)
        )
        warp_sums = cute.make_tensor(storage.warp_sums.data_ptr(), cute.make_layout(16))
        row_route = cute.make_tensor(
            storage.row_route.data_ptr(), cute.make_layout(MAX_ROWS * TOP_K)
        )
        dup_next = cute.make_tensor(
            storage.dup_next.data_ptr(), cute.make_layout(MAX_ROWS * TOP_K)
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(ab_full + s, 2)
                    cute.arch.mbarrier_init(ab_empty + s, 1)
                for s in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(acc_full + s, 1)
                    cute.arch.mbarrier_init(acc_empty + s, 128)
                for s in cutlass.range_constexpr(RING):
                    cute.arch.mbarrier_init(ring_full + s, 1)
                    cute.arch.mbarrier_init(ring_empty + s, 3)
                cute.arch.mbarrier_init(xs_ready, THREADS - 96)
                # The MMA warp allocates; the epilogue reads, then both
                # arrive again before the MMA warp frees it.
                cute.arch.mbarrier_init(tmem_bar, 32 + 128)
        cute.arch.mbarrier_init_fence()

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )
        sSFX = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfx_smem_layout,
            byte_alignment=128,
        )

        # The routing table, by every thread.
        cute.arch.griddepcontrol_wait()
        mask_lo[tidx] = Uint32(0)
        mask_hi[tidx] = Uint32(0)
        cute.arch.sync_threads()
        routes = rows * TOP_K
        for r in cutlass.range(tidx, routes, THREADS, unroll=1):
            expert = mIds[r]
            token = r // TOP_K
            if expert >= 0 and expert < EXPERTS:
                if token < 32:
                    cute.arch.atomic_or(
                        mask_lo.iterator + expert,
                        Uint32(1) << Uint32(token),
                        scope="cta",
                    )
                else:
                    cute.arch.atomic_or(
                        mask_hi.iterator + expert,
                        Uint32(1) << Uint32(token - 32),
                        scope="cta",
                    )
        cute.arch.sync_threads()
        my_mask = (Uint64(mask_hi[tidx]) << Uint64(32)) | Uint64(mask_lo[tidx])
        count = Int32(cute.arch.popc(my_mask))
        present = Int32(0)
        if count > 0:
            present = Int32(1)
        # Inclusive warp scans of (present, count), packed in one word.
        packed = present * 65536 + count
        for shift in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(packed, 1 << shift)
            if lane >= (1 << shift):
                packed = packed + other
        if lane == 31:
            warp_sums[warp_idx] = packed
        cute.arch.sync_threads()
        before = Int32(0)
        for w in cutlass.range_constexpr(THREADS // 32):
            if w < warp_idx:
                before = before + warp_sums[w]
        inclusive = before + packed
        slot = inclusive // 65536 - present
        base = inclusive % 65536 - count
        base_of[tidx] = base
        if present == 1:
            slot_expert[slot] = tidx
            slot_base[slot] = base
            slot_mask[slot] = my_mask
        n_slots_t = cute.make_tensor(storage.n_slots.data_ptr(), cute.make_layout(1))
        if tidx == THREADS - 1:
            n_slots_t[0] = inclusive // 65536
        cute.arch.sync_threads()
        for r in cutlass.range(tidx, routes, THREADS, unroll=1):
            dup_next[r] = Int32(-1)
        cute.arch.sync_threads()
        for r in cutlass.range(tidx, routes, THREADS, unroll=1):
            expert = mIds[r]
            if expert >= 0 and expert < EXPERTS:
                token = r // TOP_K
                # The nearest earlier route of this token to the same expert.
                earlier = Int32(-1)
                for k in cutlass.range(r - token * TOP_K, unroll=1):
                    if mIds[token * TOP_K + k] == expert:
                        earlier = token * TOP_K + k
                if earlier >= 0:
                    dup_next[earlier] = r
                else:
                    full = (Uint64(mask_hi[expert]) << Uint64(32)) | Uint64(
                        mask_lo[expert]
                    )
                    row_route[base_of[expert] + tile_rank(full, token)] = r
        cute.arch.sync_threads()
        n_slots = n_slots_t[0]
        g1_items = n_slots * G1_ITEMS
        total = n_slots * SLOT_ITEMS

        mma_coord = Int32(0)
        thr_mma = tiled_mma.get_slice(mma_coord)
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)

        if warp_idx == 0:
            # Scheduler and A (weights) loads.
            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            tAsA, tAgA1 = cpasync.tma_partition(
                tma_a1,
                block_coord[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(
                    thr_mma.partition_A(
                        cute.local_tile(mA1, (CTA_M, CTA_K), (None, None, None))
                    ),
                    0,
                    3,
                ),
            )
            _, tAgA2 = cpasync.tma_partition(
                tma_a2,
                block_coord[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(
                    thr_mma.partition_A(
                        cute.local_tile(mA2, (CTA_M, CTA_K), (None, None, None))
                    ),
                    0,
                    3,
                ),
            )
            tAsSFA, tAgSFA1 = cpasync.tma_partition(
                tma_sfa1,
                block_coord[2],
                a_cta_layout,
                cute.group_modes(sSFA, 0, 3),
                cute.group_modes(
                    thr_mma.partition_A(
                        cute.local_tile(mSFA1, (CTA_M, CTA_K), (None, None, None))
                    ),
                    0,
                    3,
                ),
            )
            _, tAgSFA2 = cpasync.tma_partition(
                tma_sfa2,
                block_coord[2],
                a_cta_layout,
                cute.group_modes(sSFA, 0, 3),
                cute.group_modes(
                    thr_mma.partition_A(
                        cute.local_tile(mSFA2, (CTA_M, CTA_K), (None, None, None))
                    ),
                    0,
                    3,
                ),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA1 = cute.filter_zeros(tAgSFA1)
            tAgSFA2 = cute.filter_zeros(tAgSFA2)
            policy = Int64(EVICT_FIRST)
            queue = mWS.iterator + WS_QUEUE
            stage = Int32(0)
            phase = Int32(1)
            published = Int32(0)
            item = Int32(0)
            if lane == 0:
                item = atomic_add_relaxed(queue, 1)
            item = cute.arch.shuffle_sync(item, 0)
            while item < total:
                self.publish(ring, ring_full, ring_empty, published, item)
                published += 1
                upcoming = Int32(0)
                if lane == 0:
                    upcoming = atomic_add_relaxed(queue, 1)
                if item < g1_items:
                    slot = item // G1_ITEMS
                    m_tile = item - slot * G1_ITEMS
                    expert = slot_expert[slot]
                    tAgA = tAgA1[(None, m_tile, None, expert)]
                    tAgSFA = tAgSFA1[(None, m_tile, None, expert)]
                    for k_tile in cutlass.range(G1_K_TILES, unroll=1):
                        cute.arch.mbarrier_wait(ab_empty + stage, phase)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                ab_full + stage, self.tma_bytes_a
                            )
                        cute.copy(
                            tma_a1,
                            tAgA[(None, k_tile)],
                            tAsA[(None, stage)],
                            tma_bar_ptr=ab_full + stage,
                            cache_policy=policy,
                        )
                        cute.copy(
                            tma_sfa1,
                            tAgSFA[(None, k_tile)],
                            tAsSFA[(None, stage)],
                            tma_bar_ptr=ab_full + stage,
                            cache_policy=policy,
                        )
                        stage += 1
                        if stage == stages:
                            stage = Int32(0)
                            phase ^= 1
                else:
                    j = item - g1_items
                    slot = j // G2_M_TILES
                    m_tile = j - slot * G2_M_TILES
                    expert = slot_expert[slot]
                    if lane == 0:
                        # The whole tile into L2 now: GEMM2 items may wait
                        # for their H with the stages full.
                        rows_off = Int64(expert) * HIDDEN + Int64(m_tile) * CTA_M
                        prefetch_l2(w2_ptr, rows_off * (INTER // 2), CTA_M * INTER // 2)
                        prefetch_l2(
                            w2s_ptr,
                            rows_off * (INTER // SF_VEC),
                            CTA_M * INTER // SF_VEC,
                        )
                    tAgA = tAgA2[(None, m_tile, None, expert)]
                    tAgSFA = tAgSFA2[(None, m_tile, None, expert)]
                    for k_tile in cutlass.range(G2_K_TILES, unroll=1):
                        cute.arch.mbarrier_wait(ab_empty + stage, phase)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                ab_full + stage, self.tma_bytes_a
                            )
                        cute.copy(
                            tma_a2,
                            tAgA[(None, k_tile)],
                            tAsA[(None, stage)],
                            tma_bar_ptr=ab_full + stage,
                            cache_policy=policy,
                        )
                        cute.copy(
                            tma_sfa2,
                            tAgSFA[(None, k_tile)],
                            tAsSFA[(None, stage)],
                            tma_bar_ptr=ab_full + stage,
                            cache_policy=policy,
                        )
                        stage += 1
                        if stage == stages:
                            stage = Int32(0)
                            phase ^= 1
                item = cute.arch.shuffle_sync(upcoming, 0)
            # The end marker.
            self.publish(ring, ring_full, ring_empty, published, total)
            cute.arch.griddepcontrol_launch_dependents()
            # Keep shared memory alive until the MMA warp releases every stage.
            for _ in cutlass.range(stages, unroll=1):
                cute.arch.mbarrier_wait(ab_empty + stage, phase)
                stage += 1
                if stage == stages:
                    stage = Int32(0)
                    phase ^= 1
        elif warp_idx == 1:
            # B (input or H) and B's scales.
            cute.arch.griddepcontrol_wait()
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            tBsB, tBgB1 = cpasync.tma_partition(
                tma_b1,
                block_coord[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(
                    thr_mma.partition_B(
                        cute.local_tile(mB1, (n_cols, CTA_K), (None, None, None))
                    ),
                    0,
                    3,
                ),
            )
            _, tBgB2 = cpasync.tma_partition(
                tma_b2,
                block_coord[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(
                    thr_mma.partition_B(
                        cute.local_tile(mB2, (n_cols, CTA_K), (None, None, None))
                    ),
                    0,
                    3,
                ),
            )
            stage = Int32(0)
            phase = Int32(1)
            consumed = Int32(0)
            item = self.consume(ring, ring_full, ring_empty, consumed, lane == 0)
            sfb_stage_bytes = cute.size_in_bytes(
                self.sf_dtype,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            while item < total:
                consumed += 1
                if item < g1_items:
                    tBgB = tBgB1[(None, 0, None, 0)]
                    for k_tile in cutlass.range(G1_K_TILES, unroll=1):
                        cute.arch.mbarrier_wait(ab_empty + stage, phase)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                ab_full + stage, self.tma_bytes_b
                            )
                        cute.copy(
                            tma_b1,
                            tBgB[(None, k_tile)],
                            tBsB[(None, stage)],
                            tma_bar_ptr=ab_full + stage,
                        )
                        stage += 1
                        if stage == stages:
                            stage = Int32(0)
                            phase ^= 1
                else:
                    j = item - g1_items
                    slot = j // G2_M_TILES
                    if lane == 0:
                        flag = mWS.iterator + WS_FLAGS + slot
                        polls = Int32(0)
                        while load_acquire(flag) < G1_ITEMS:
                            if cutlass.const_expr(self.spin_limit > 0):
                                polls += 1
                                if polls > self.spin_limit:
                                    cute.printf(
                                        "routed experts: CTA %d waits for slot %d "
                                        "(expert %d), counter %d\n",
                                        cute.arch.block_idx()[0],
                                        slot,
                                        slot_expert[slot],
                                        load_acquire(flag),
                                    )
                                    trap()
                        fence_proxy_async_global()
                    cute.arch.sync_warp()
                    tBgB = tBgB2[(None, slot, None, 0)]
                    for k_tile in cutlass.range(G2_K_TILES, unroll=1):
                        cute.arch.mbarrier_wait(ab_empty + stage, phase)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                ab_full + stage, self.tma_bytes_b + STAGE_SFB_BYTES
                            )
                            bulk_load(
                                sSFB.iterator + stage * sfb_stage_bytes,
                                mHS.iterator
                                + (slot * HS_SLOT_BYTES + k_tile * STAGE_SFB_BYTES),
                                STAGE_SFB_BYTES,
                                ab_full + stage,
                            )
                        cute.copy(
                            tma_b2,
                            tBgB[(None, k_tile)],
                            tBsB[(None, stage)],
                            tma_bar_ptr=ab_full + stage,
                        )
                        stage += 1
                        if stage == stages:
                            stage = Int32(0)
                            phase ^= 1
                item = self.consume(ring, ring_full, ring_empty, consumed, lane == 0)
        elif warp_idx == 2:
            self.mma_warp(
                storage,
                ab_full,
                ab_empty,
                acc_full,
                acc_empty,
                ring,
                ring_full,
                ring_empty,
                xs_ready,
                tmem_bar,
                tiled_mma,
                sA,
                sB,
                sSFA,
                sSFB,
                sSFX,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                g1_items,
                total,
            )
        if warp_idx >= 3:
            # The input's scales into GEMM1's resident B-scale atoms.
            cute.arch.griddepcontrol_wait()
            words = cute.make_tensor(
                cute.recast_ptr(sSFX.iterator, dtype=Uint32),
                cute.make_layout(G1_K_TILES * STAGE_SFB_BYTES // 4),
            )
            for w in cutlass.range(
                tidx - 96, rows * (HIDDEN // 64), THREADS - 96, unroll=1
            ):
                token = w // (HIDDEN // 64)
                chunk = w - token * (HIDDEN // 64)
                words[(chunk * 512 + (token % 32) * 16 + (token // 32) * 4) // 4] = mXS[
                    w
                ]
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(xs_ready)
            # Zero the rows of routes to no local expert (one CTA per route).
            cta, _, _ = cute.arch.block_idx()
            for r in cutlass.range(cta, rows * TOP_K, grid, unroll=1):
                expert = mIds[r]
                if expert < 0 or expert >= EXPERTS:
                    for w in cutlass.range(
                        (tidx - 96) * 2, HIDDEN // 2, (THREADS - 96) * 2, unroll=1
                    ):
                        store_zero_v2(mOut.iterator + (r * (HIDDEN // 2) + w))
        if warp_idx >= EPI_WARP:
            self.epilogue(
                storage,
                acc_full,
                acc_empty,
                ring,
                ring_full,
                ring_empty,
                tmem_bar,
                slot_expert,
                slot_base,
                slot_mask,
                row_route,
                mG1a,
                mG1c,
                mG2a,
                mHB,
                mHS,
                mWS,
                mOut,
                g1_items,
                total,
                dup_next,
            )
        cute.arch.sync_threads()
        if tidx == 0:
            # The last CTA to finish resets the workspace for the next call.
            done = atomic_add_acq_rel(mWS.iterator + WS_DONE, 1)
            if done == grid - 1:
                for s in cutlass.range(n_slots, unroll=1):
                    store_relaxed(mWS.iterator + WS_FLAGS + s, 0)
                store_relaxed(mWS.iterator + WS_QUEUE, 0)
                store_relaxed(mWS.iterator + WS_DONE, 0)

    @cute.jit
    def publish(self, ring, ring_full, ring_empty, index: Int32, item: Int32):
        """Warp 0: entry ``index`` of the item ring."""
        slot = index % RING
        cute.arch.mbarrier_wait(ring_empty + slot, ((index // RING) & 1) ^ 1)
        if cute.arch.lane_idx() == 0:
            ring[slot] = item
            cute.arch.mbarrier_arrive(ring_full + slot)

    @cute.jit
    def consume(self, ring, ring_full, ring_empty, index: Int32, releaser) -> Int32:
        """Entry ``index`` of the item ring; ``releaser`` frees the entry."""
        slot = index % RING
        cute.arch.mbarrier_wait(ring_full + slot, (index // RING) & 1)
        item = ring[slot]
        if releaser:
            cute.arch.mbarrier_arrive(ring_empty + slot)
        return item

    def _acc_layout(self, tiled_mma: cute.TiledMma) -> cute.Layout:
        acc_shape = tiled_mma.partition_shape_C((CTA_M, self.cta_n))
        return tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        ).layout

    def _s2t_copy(self, sSF: cute.Tensor, tSF: cute.Tensor):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, tCtSF_compact)
        thr_copy = tiled_copy.get_slice(0)
        tCsSF = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy, thr_copy.partition_S(tCsSF_compact)
        )
        return tiled_copy, tCsSF, thr_copy.partition_D(tCtSF_compact)

    @cute.jit
    def mma_warp(
        self,
        storage,
        ab_full,
        ab_empty,
        acc_full,
        acc_empty,
        ring,
        ring_full,
        ring_empty,
        xs_ready,
        tmem_bar,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sSFX: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        g1_items: Int32,
        total: Int32,
    ):
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tmem_base_ptr = storage.tmem_base.data_ptr()
        cute.arch.alloc_tmem(self.tmem_cols, tmem_base_ptr)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc_base = cute.make_tensor(tmem_ptr, self._acc_layout(tiled_mma))
        sfa_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self.num_acc_tmem_cols, dtype=self.sf_dtype
        )
        tCtSFA = cute.make_tensor(
            sfa_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                tiled_mma.shape_mnk,
                SF_VEC,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            ),
        )
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self.num_acc_tmem_cols + self.num_sfa_tmem_cols,
            dtype=self.sf_dtype,
        )
        tCtSFB = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                tiled_mma.shape_mnk,
                SF_VEC,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            ),
        )
        copy_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t_copy(sSFA, tCtSFA)
        copy_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t_copy(sSFB, tCtSFB)
        copy_sfx, tCsSFX_s2t, tCtSFX_s2t = self._s2t_copy(sSFX, tCtSFB)

        stage = Int32(0)
        phase = Int32(0)
        acc_stage = Int32(0)
        acc_phase = Int32(1)
        consumed = Int32(0)
        xs_waited = Int32(0)
        item = self.consume(
            ring, ring_full, ring_empty, consumed, cute.arch.lane_idx() == 0
        )
        while item < total:
            consumed += 1
            cute.arch.mbarrier_wait(acc_empty + acc_stage, acc_phase)
            tCtAcc = tCtAcc_base[(None, None, None, acc_stage)]
            if item < g1_items:
                if xs_waited == 0:
                    cute.arch.mbarrier_wait(xs_ready, 0)
                    xs_waited = Int32(1)
                for k_tile in cutlass.range(G1_K_TILES, unroll=1):
                    cute.arch.mbarrier_wait(ab_full + stage, phase)
                    cute.copy(
                        copy_sfa,
                        tCsSFA_s2t[(None, None, None, None, stage)],
                        tCtSFA_s2t,
                    )
                    cute.copy(
                        copy_sfx,
                        tCsSFX_s2t[(None, None, None, None, k_tile)],
                        tCtSFX_s2t,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    operand = (None, None, None, stage)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        [tCrA[operand], tCtSFA],
                        [tCrB[operand], tCtSFB],
                        tCtAcc,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(ab_empty + stage, None, self.cta_group)
                    stage += 1
                    if stage == self.num_stages:
                        stage = Int32(0)
                        phase ^= 1
            else:
                for k_tile in cutlass.range(G2_K_TILES, unroll=1):
                    cute.arch.mbarrier_wait(ab_full + stage, phase)
                    cute.copy(
                        copy_sfa,
                        tCsSFA_s2t[(None, None, None, None, stage)],
                        tCtSFA_s2t,
                    )
                    cute.copy(
                        copy_sfb,
                        tCsSFB_s2t[(None, None, None, None, stage)],
                        tCtSFB_s2t,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    operand = (None, None, None, stage)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        [tCrA[operand], tCtSFA],
                        [tCrB[operand], tCtSFB],
                        tCtAcc,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(ab_empty + stage, None, self.cta_group)
                    stage += 1
                    if stage == self.num_stages:
                        stage = Int32(0)
                        phase ^= 1
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + acc_stage, None, self.cta_group)
            acc_stage += 1
            if acc_stage == 2:
                acc_stage = Int32(0)
                acc_phase ^= 1
            item = self.consume(
                ring, ring_full, ring_empty, consumed, cute.arch.lane_idx() == 0
            )
        # Free the tensor memory once the epilogue has read it.
        cute.arch.mbarrier_wait(tmem_bar, 0)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 1)
        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)

    @cute.jit
    def epilogue(
        self,
        storage,
        acc_full,
        acc_empty,
        ring,
        ring_full,
        ring_empty,
        tmem_bar,
        slot_expert: cute.Tensor,
        slot_base: cute.Tensor,
        slot_mask: cute.Tensor,
        row_route: cute.Tensor,
        mG1a: cute.Tensor,
        mG1c: cute.Tensor,
        mG2a: cute.Tensor,
        mHB: cute.Tensor,
        mHS: cute.Tensor,
        mWS: cute.Tensor,
        mOut: cute.Tensor,
        g1_items: Int32,
        total: Int32,
        dup_next: cute.Tensor,
    ):
        """Thread (warp w, lane = 4 s + c) reads accumulator rows 32 w + s +
        8 i (i < 4) of columns 8 chunk + 2 c, + 1; with the 128-row shuffle
        they are rows 32 w + 4 s + i of the weight tile."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - EPI_WARP
        lane = cute.arch.lane_idx()
        tid = warp * 32 + lane
        s = lane // 4
        c = lane % 4
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 0)
        tmem = cute.make_tensor(storage.tmem_base.data_ptr(), cute.make_layout(1))[0]
        lanes = Uint32(warp * 32) << Uint32(16)
        acc_stage = Int32(0)
        acc_phase = Int32(0)
        consumed = Int32(0)
        item = self.consume(ring, ring_full, ring_empty, consumed, tid == 0)
        while item < total:
            consumed += 1
            cute.arch.mbarrier_wait(acc_full + acc_stage, acc_phase)
            column0 = Uint32(tmem) + Uint32(acc_stage * self.cta_n) + lanes
            if item < g1_items:
                slot = item // G1_ITEMS
                m_tile = item - slot * G1_ITEMS
                expert = slot_expert[slot]
                mask = slot_mask[slot]
                h_row0 = slot * self.cta_n
                scale_gate = mG1a[expert]
                scale_c = mG1c[expert]
                k = mul_ftz(scale_gate, Float32(1.4426950216293334961))
                h_col = m_tile * 32 + warp * 8 + s
                for chunk in cutlass.range_constexpr(self.cta_n // 8):
                    bits = (mask >> Uint64(chunk * 8)) & Uint64(255)
                    if bits != Uint64(0):
                        v = tmem_load_8x8(
                            column0 + Uint32(chunk * 8),
                            column0 + Uint32(chunk * 8) + (Uint32(16) << Uint32(16)),
                        )
                        # Columns p (rows s, s + 8) and p + 1 (rows s + 16,
                        # s + 24): up then gate, tokens a and a + 1.
                        ha0, hb0 = swiglu_x2(v[0], v[1], v[2], v[3], k, scale_gate)
                        ha1, hb1 = swiglu_x2(v[4], v[5], v[6], v[7], k, scale_gate)
                        amax_a, amax_b = block_amax_x2(ha0, ha1, hb0, hb1, scale_c)
                        for shift in cutlass.range_constexpr(4, 1, -1):
                            amax_a = max_ftz(
                                amax_a, cute.arch.shuffle_sync_bfly(amax_a, 1 << shift)
                            )
                            amax_b = max_ftz(
                                amax_b, cute.arch.shuffle_sync_bfly(amax_b, 1 << shift)
                            )
                        sf, inv_a, inv_b = block_scale_x2(amax_a, amax_b)
                        code_a = quantize_e2m1x2(ha0, ha1, inv_a, scale_c)
                        code_b = quantize_e2m1x2(hb0, hb1, inv_b, scale_c)
                        token_a = chunk * 8 + 2 * c
                        if (mask >> Uint64(token_a)) & Uint64(1) != Uint64(0):
                            rank = tile_rank(mask, token_a)
                            store_u8(
                                mHB.iterator + ((h_row0 + rank) * (INTER // 2) + h_col),
                                code_a,
                            )
                            if s == 0:
                                store_u8(
                                    mHS.iterator
                                    + (
                                        slot * HS_SLOT_BYTES
                                        + m_tile * 512
                                        + (rank % 32) * 16
                                        + (rank // 32) * 4
                                        + warp
                                    ),
                                    sf & Uint32(255),
                                )
                        if (mask >> Uint64(token_a + 1)) & Uint64(1) != Uint64(0):
                            rank = tile_rank(mask, token_a + 1)
                            store_u8(
                                mHB.iterator + ((h_row0 + rank) * (INTER // 2) + h_col),
                                code_b,
                            )
                            if s == 0:
                                store_u8(
                                    mHS.iterator
                                    + (
                                        slot * HS_SLOT_BYTES
                                        + m_tile * 512
                                        + (rank % 32) * 16
                                        + (rank // 32) * 4
                                        + warp
                                    ),
                                    sf >> Uint32(8),
                                )
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + acc_stage)
                # H of this item is out: count it for the slot's GEMM2 items.
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                if tid == 0:
                    release_add(mWS.iterator + WS_FLAGS + slot, 1)
            else:
                j = item - g1_items
                slot = j // G2_M_TILES
                m_tile = j - slot * G2_M_TILES
                expert = slot_expert[slot]
                mask = slot_mask[slot]
                count = Int32(cute.arch.popc(mask))
                base = slot_base[slot]
                scale = mG2a[expert]
                hidden = m_tile * CTA_M + warp * 32 + 4 * s
                for chunk in cutlass.range_constexpr(self.cta_n // 8):
                    if chunk * 8 < count:
                        v = tmem_load_8x8(
                            column0 + Uint32(chunk * 8),
                            column0 + Uint32(chunk * 8) + (Uint32(16) << Uint32(16)),
                        )
                        rank = chunk * 8 + 2 * c
                        if rank < count:
                            lo = scaled_bf16x2(v[0], v[2], scale)
                            hi = scaled_bf16x2(v[4], v[6], scale)
                            route = row_route[base + rank]
                            while route >= 0:
                                store_v2(
                                    mOut.iterator
                                    + (route * (HIDDEN // 2) + hidden // 2),
                                    lo,
                                    hi,
                                )
                                route = dup_next[route]
                        if rank + 1 < count:
                            lo = scaled_bf16x2(v[1], v[3], scale)
                            hi = scaled_bf16x2(v[5], v[7], scale)
                            route = row_route[base + rank + 1]
                            while route >= 0:
                                store_v2(
                                    mOut.iterator
                                    + (route * (HIDDEN // 2) + hidden // 2),
                                    lo,
                                    hi,
                                )
                                route = dup_next[route]
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + acc_stage)
            acc_stage += 1
            if acc_stage == 2:
                acc_stage = Int32(0)
                acc_phase ^= 1
            item = self.consume(ring, ring_full, ring_empty, consumed, tid == 0)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 1)
