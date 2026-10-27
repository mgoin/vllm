# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PTX helpers of the boundary kernels that DSV4.1's primitives lack."""

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as nvvm

from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.primitives import _asm

# TMA L2 cache hint (CUTLASS ``TMA::CacheHintSm90::EVICT_FIRST``).
EVICT_FIRST = 0x12F0000000000000


def fence_tmem(*, before: bool = False) -> None:
    nvvm.tcgen05_fence("before_thread_sync" if before else "after_thread_sync")


@dsl_user_op
def load_tmem_f32(address: Int32, count: int, *, loc=None, ip=None) -> list[Float32]:
    """``count`` FP32 columns of this thread's TMEM lane, waiting for the load."""
    registers = ",".join(f"${i}" for i in range(count))
    result = _asm(
        llvm.StructType.get_literal([T.f32()] * count),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        f"tcgen05.ld.sync.aligned.32x32b.x{count}.b32 {{{registers}}}, [${count}];\n\t"
        "tcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * count + ["r"]),
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(count)
    ]


@dsl_user_op
def store_shared_f32(address: Int32, value: Float32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_shared_words(address: Int32, words, *, loc=None, ip=None) -> None:
    """``st.shared`` of 1 or 4 32-bit words."""
    count = len(words)
    vector, registers = (".v4", "{$1, $2, $3, $4}") if count == 4 else ("", "$1")
    _asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        f"st.shared{vector}.b32 [$0], {registers};",
        ",".join(["r"] * (count + 1)),
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32x4(address: Int32, *, loc=None, ip=None) -> list[Float32]:
    result = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def pack_bf16x2_published(low: Float32, high: Float32, *, loc=None, ip=None) -> Uint32:
    """RNE BF16x2 of (low, high) with negative zero made +0, so no half is the
    Lamport sentinel."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Float32(low).ir_value(loc=loc, ip=ip),
                Float32(high).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 bits;\n\t.reg .pred p;\n\t"
            "cvt.rn.bf16x2.f32 $0, $2, $1;\n\t"
            "and.b32 bits, $0, 0xffff;\n\tsetp.eq.u32 p, bits, 0x8000;\n\t"
            "@p and.b32 $0, $0, 0xffff0000;\n\t"
            "shr.u32 bits, $0, 16;\n\tsetp.eq.u32 p, bits, 0x8000;\n\t"
            "@p and.b32 $0, $0, 0xffff;\n\t}",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_global_u32x4_at(address: Int64, words, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Uint32(words[i]).ir_value(loc=loc, ip=ip) for i in range(4)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def multicast_u32x4(address: Int64, words, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Uint32(words[i]).ir_value(loc=loc, ip=ip) for i in range(4)],
        "multimem.st.relaxed.sys.global.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_volatile_u32_at(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_u64_at(address: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_u32_at(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def copy_shared_to_peer(
    dst: Int32, src: Int32, barrier: Int32, size: int, *, loc=None, ip=None
) -> None:
    """Bulk copy ``size`` bytes of this CTA's shared memory to ``dst`` in a
    peer CTA (``shared::cluster`` addresses ``dst`` and ``barrier``), completing
    transaction bytes on the peer's mbarrier."""
    _asm(
        None,
        [Int32(v).ir_value(loc=loc, ip=ip) for v in (dst, src, barrier)],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        f"[$0], [$1], {size}, [$2];",
        "r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_words(address: Int32, count: int, *, loc=None, ip=None) -> list:
    """``ld.shared`` of 1 or 4 32-bit words."""
    if count == 1:
        return [
            Uint32(
                _asm(
                    T.i32(),
                    [Int32(address).ir_value(loc=loc, ip=ip)],
                    "ld.shared.b32 $0, [$1];",
                    "=r,r",
                    side_effects=True,
                    loc=loc,
                    ip=ip,
                )
            )
        ]
    result = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def bulk_commit_group(*, loc=None, ip=None) -> None:
    _asm(None, [], "cp.async.bulk.commit_group;", "", side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def bulk_wait_group_read(*, loc=None, ip=None) -> None:
    _asm(
        None,
        [],
        "cp.async.bulk.wait_group.read 0;",
        "",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_address(pointer: cute.Pointer, *, loc=None, ip=None) -> Int32:
    return Int32(pointer.toint(loc=loc, ip=ip))


@dsl_user_op
def map_shared_rank(address: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(address).ir_value(loc=loc, ip=ip),
                Int32(rank).ir_value(loc=loc, ip=ip),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )
