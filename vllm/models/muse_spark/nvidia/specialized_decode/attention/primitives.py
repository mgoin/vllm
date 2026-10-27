# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline-PTX helpers of the attention kernels: TMA, tcgen05 BF16 MMA and TMEM
loads, mbarrier waits, mailbox words and BF16 packing."""

import ctypes
from functools import wraps

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint16, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

SENTINEL = 0xFFFFFFFF  # an FP32 NaN no arithmetic produces: an empty mailbox word


def ptx(code, constraints, operands, result=None, *, effects, loc=None, ip=None):
    """One inline-PTX block; a tuple ``result`` returns a tuple."""
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
    """``f(*args) -> (code, constraints, operands)`` as one inline-PTX op. Pointer
    operands pass their address: 32-bit for "r" (shared), 64-bit for "l"."""

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
            return ptx(
                code, constraints, values, result, effects=not pure, loc=loc, ip=ip
            )

        return op

    return wrap


def thread():
    """(thread, warp, lane, CTA) indices."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    return tid, tid // 32, tid % 32, cta


def alloc(allocator, dtype, size, align=16):
    return allocator.allocate_tensor(dtype, cute.make_layout(size), align)


def at(tensor, *coord):
    """Pointer to ``tensor[coord]``."""
    return tensor.iterator + cute.crd2idx(coord, tensor.layout)


def bits_float(bits):
    return Float32(llvm.bitcast(Float32.mlir_type, Uint32(bits).ir_value()))


def empty_word(words):
    """Whether any of four mailbox ``words`` is still ``SENTINEL``."""
    return (
        (words[0] == SENTINEL)
        | (words[1] == SENTINEL)
        | (words[2] == SENTINEL)
        | (words[3] == SENTINEL)
    )


@asm()
def acquire_tensormap(pointer):
    code = "fence.proxy.tensormap::generic.acquire.sys [$0],128; "
    return code + "prefetch.tensormap [$0];", "l", (pointer,)


@asm()
def load_tma2(dst, descriptor, barrier, column, row):
    """The 2D box at (``column``, ``row``), L2 evict_first."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier"
    code += "::complete_tx::bytes.L2::cache_hint [$0],[$1,{$3,$4}],[$2],pol; }"
    return code, "r,l,r,r,r", (dst, descriptor, barrier, Int32(column), Int32(row))


@asm()
def load_tma2_multicast(dst, descriptor, barrier, column, row, mask):
    """The 2D box at (``column``, ``row``) into every CTA of ``mask``, completing on
    each one's ``barrier``."""
    code = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier"
    code += "::complete_tx::bytes.multicast::cluster [$0],[$1,{$3,$4}],[$2],$5;"
    operands = (dst, descriptor, barrier, Int32(column), Int32(row), Uint16(mask))
    return code, "r,l,r,r,r,h", operands


@asm()
def copy_g2s(dst, src):
    """16 bytes global -> shared through L2 (cp.async.cg)."""
    return "cp.async.cg.shared.global [$0], [$1], 16;", "r,l", (dst, src)


@asm()
def cp_async_wait_all():
    return "cp.async.wait_all;", "~{memory}", ()


@asm(Int32)
def try_wait(barrier, phase, *, scope):
    code = f"{{ .reg .pred p; mbarrier.try_wait.parity.acquire.{scope}.shared::cta"
    code += ".b64 p,[$1],$2; selp.u32 $0,1,0,p; }"
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait(barrier, phase=0, scope: cutlass.Constexpr = "cta"):
    """Acquire wait for ``phase`` (``scope`` "cluster" for remote arrivals)."""
    done = try_wait(barrier, phase, scope=scope)
    while done == 0:
        done = try_wait(barrier, phase, scope=scope)


@asm()
def named_barrier(barrier_id, count):
    return f"bar.sync {barrier_id},{count};", "~{memory}", ()


@asm()
def fence_tmem(*, before=False):
    side = "before" if before else "after"
    return f"tcgen05.fence::{side}_thread_sync;", "~{memory}", ()


@asm()
def free_tmem(address, columns):
    code = f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0,{columns};"
    return code, "r", (Uint32(address),)


@asm()
def commit(barrier):
    code = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];"
    return code, "r", (barrier,)


@asm()
def mma_bf16_block(d, a, b, instruction, accumulate, empty, release, *, mask=0):
    """Warp-uniform issue of one 64-K block (K-major 128B-swizzled A and B): the
    elected lane runs the four K16 BF16 MMAs into ``d`` (the first accumulates iff
    ``accumulate`` is nonzero) and, if ``release``, commits ``empty`` (in every CTA
    of ``mask`` when nonzero)."""
    mma = "tcgen05.mma.cta_group::1.kind::f16"
    done = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
    done += f".multicast::cluster.b64 [$5],{mask};" if mask else ".b64 [$5];"
    code = "{ .reg .pred e,p0,pt,pc; .reg .b64 a1,a2,a3,b1,b2,b3; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 p0,$4,0; "
    code += "setp.ne.b32 pc,$6,0; and.pred pc,pc,e; "
    code += "add.s64 a1,$1,2; add.s64 a2,$1,4; add.s64 a3,$1,6; "
    code += "add.s64 b1,$2,2; add.s64 b2,$2,4; add.s64 b3,$2,6; "
    code += f"@e {mma} [$0],$1,$2,$3,p0; @e {mma} [$0],a1,b1,$3,pt; "
    code += f"@e {mma} [$0],a2,b2,$3,pt; @e {mma} [$0],a3,b3,$3,pt; "
    code += f"@pc {done} }}"
    operands = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    operands += (Uint32(accumulate), empty, Uint32(release))
    return code, "r,l,l,r,r,r,r", operands


@cute.jit
def smem_desc(pointer):
    """K-major 128B-swizzled UMMA descriptor (8-row groups 1024 B apart)."""
    address = Uint32(pointer.toint())
    value = Uint64((address & 262143) >> 4) | Uint64(1) << 46
    return value | Uint64(1) << 16 | Uint64(64) << 32 | Uint64(2) << 61


def load_tmem(address, count):
    """``count`` FP32 TMEM columns of this lane, waiting for the load."""

    @asm((Float32,) * count)
    def load(address):
        registers = ",".join(f"${i}" for i in range(count))
        code = f"tcgen05.ld.sync.aligned.32x32b.x{count}.b32 {{{registers}}},"
        code += f"[${count}]; tcgen05.wait::ld.sync.aligned;"
        constraints = ",".join(["=f"] * count + ["r", "~{memory}"])
        return code, constraints, (Uint32(address),)

    return load(address)


@asm((Float32,) * 4)
def load_shared_f32x4(pointer):
    code = "ld.shared.v4.f32 {$0,$1,$2,$3},[$4];"
    return code, "=f,=f,=f,=f,r,~{memory}", (pointer,)


@asm((Uint32,) * 4)
def load_shared_x4(pointer):
    code = "ld.shared.v4.b32 {$0,$1,$2,$3},[$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (pointer,)


@asm(cutlass.Int64)
def load_shared_i64(pointer):
    return "ld.shared.b64 $0,[$1];", "=l,r,~{memory}", (pointer,)


@asm()
def store_shared_i64(pointer, value):
    return "st.shared.b64 [$0],$1;", "r,l,~{memory}", (pointer, value)


@asm()
def store_shared_bf16(pointer, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h,$1; st.shared.b16 [$0],h; }"
    return code, "r,f,~{memory}", (pointer, Float32(value))


@asm()
def store_global_x4(address, words):
    code = "st.global.v4.b32 [$0],{$1,$2,$3,$4};"
    return code, "l,r,r,r,r,~{memory}", (address, *(Uint32(w) for w in words))


@asm()
def store_global_f32x4(address, values):
    code = "st.global.v4.f32 [$0],{$1,$2,$3,$4};"
    return code, "l,f,f,f,f,~{memory}", (address, *(Float32(v) for v in values))


@asm((Float32, Float32), pure=True)
def unpack_bf16(word):
    return "shl.b32 $0,$2,16; and.b32 $1,$2,0xffff0000;", "=f,=f,r", (Uint32(word),)


@asm(Uint32, pure=True)
def pack_bf16(low, high):
    return "cvt.rn.bf16x2.f32 $0,$2,$1;", "=r,f,f", (Float32(low), Float32(high))


@asm(Float32, pure=True)
def fma(a, b, c):
    return "fma.rn.f32 $0,$1,$2,$3;", "=f,f,f,f", (Float32(a), Float32(b), Float32(c))


@asm(Float32, pure=True)
def mul(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def add(a, b):
    return "add.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def rsqrt(x):
    return "rsqrt.approx.f32 $0,$1;", "=f,f", (Float32(x),)


_PINNED_MAPS: list[torch.Tensor] = []


def tensor_map(tensor, box):
    """Device TMA descriptor of a BF16 [rows, K] matrix (row pitch kept): 128B
    swizzle, ``box`` = (K elements, rows)."""
    rows, k = tensor.shape
    dims = [cuda.cuuint64_t(v) for v in (k, rows)]
    pitch = [cuda.cuuint64_t(tensor.stride(0) * tensor.element_size())]
    status, descriptor = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        tensor.data_ptr(),
        dims,
        pitch,
        [cuda.cuuint32_t(v) for v in box],
        [cuda.cuuint32_t(1)] * 2,
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {status}")
    encoded = bytes((ctypes.c_ubyte * 128).from_address(descriptor.getPtr()))
    # Pinned and kept alive: a map made during CUDA graph capture is a captured
    # copy that every replay repeats.
    host = torch.empty(128, dtype=torch.uint8, pin_memory=True)
    host.copy_(torch.frombuffer(bytearray(encoded), dtype=torch.uint8))
    _PINNED_MAPS.append(host)
    return host.to(tensor.device, non_blocking=True)
