# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
"""Bytes, roofline and achieved bandwidth of one Muse Spark decode step.

Per rank at TP4, no expert parallelism: every rank holds all 256 experts at a
quarter of their width, so a step streams (distinct experts touched) x 6.29 MB
of expert weights, plus the dense projections and the KV cache it reads. Compute
is never the limit: even at 256 rows each expert GEMM has four rows.

python roofline.py [--rows 8 32 128] [--ctx 1024 16384] [--us per_layer_us]
"""

import argparse

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 8192
MOE_HIDDEN = 4096
TP = 4
NVFP4_BYTES = 0.5
# Block-scale factors of NVFP4: one FP8 scale and one FP32 global per 16 values.
SCALES_PER_PARAM = 1.0 / 16.0 * 1 + 1.0 / (16 * 4096)
# B300 HBM3e, sustained rather than peak.
PEAK_TBPS = 8.0
TARGET_TBPS = 7.5


def expert_bytes() -> float:
    w13 = MOE_HIDDEN * (2 * MOE_HIDDEN // TP)
    w2 = MOE_HIDDEN * (MOE_HIDDEN // TP)
    return (w13 + w2) * (NVFP4_BYTES + SCALES_PER_PARAM) / 1e6  # MB


def distinct_experts(rows: int) -> float:
    routes = rows * TOP_K
    return NUM_EXPERTS * (1.0 - (1.0 - 1.0 / NUM_EXPERTS) ** routes)


def per_layer_bytes(rows: int, ctx: int) -> float:
    """MB read per rank per layer."""
    mb = distinct_experts(rows) * expert_bytes()
    # qkv (128 q + 16 kv + 16 kv + 128 gate heads of 64) NVFP4, per rank.
    qkv = HIDDEN * ((128 + 16 + 16 + 128) * 64 // TP) * (NVFP4_BYTES + SCALES_PER_PARAM)
    o_proj = 128 * 64 // TP * HIDDEN * (NVFP4_BYTES + SCALES_PER_PARAM)
    # BF16 pooler + unpooler, column sharded.
    squeeze = 2 * HIDDEN * (MOE_HIDDEN // TP) * 2
    router = HIDDEN * NUM_EXPERTS * 4  # FP32, replicated
    dense = (qkv + o_proj + squeeze + router) / 1e6
    # KV reads: 46 sliding layers at 2048 tokens, 16 full layers at ctx, 16 kv
    # heads of 64 in BF16.
    kv_window = min(ctx, 2048) * 16 * 64 * 2 / 1e6
    kv_full = ctx * 16 * 64 * 2 / 1e6
    layers = 62
    kv = (46 * kv_window + 16 * kv_full) / layers
    return mb + dense + kv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--ctx", type=int, nargs="+", default=[1024, 8192, 16384])
    p.add_argument(
        "--us",
        type=float,
        nargs="*",
        default=None,
        help="measured per-layer us, one per --rows entry, same ctx as --ctx[0]",
    )
    a = p.parse_args()
    print(
        f"expert {expert_bytes():.2f} MB/rank  "
        f"(roofline at {TARGET_TBPS} TB/s)"
    )
    for ctx in a.ctx:
        print(f"\nctx {ctx}: per layer, per rank")
        print(
            f"{'rows':>5} {'experts':>8} {'MB':>9} {'roofline us':>12} "
            f"{'step ms':>8} {'us (meas)':>10} {'TB/s':>6}"
        )
        for i, rows in enumerate(a.rows):
            mb = per_layer_bytes(rows, ctx)
            roof = mb / TARGET_TBPS  # MB / (TB/s) == us
            meas = None
            if a.us and i < len(a.us):
                meas = a.us[i]
            tbps = "" if meas is None else f"{mb / meas:.2f}"
            print(
                f"{rows:5d} {distinct_experts(rows):8.1f} {mb:9.1f} {roof:12.1f} "
                f"{roof * 62 / 1e3:8.2f} {'' if meas is None else f'{meas:10.1f}'} "
                f"{tbps:>6}"
            )


if __name__ == "__main__":
    main()
