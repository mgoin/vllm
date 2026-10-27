# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
import collections
import gzip
import json
import regex as re
import sys

f = sys.argv[1]
ev = json.load(gzip.open(f))["traceEvents"]
k = [
    e
    for e in ev
    if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
]
k.sort(key=lambda e: e["ts"])
print("gpu events", len(k))
# steps: count fused qkv kernels (62 per target forward)
qkv = [e for e in k if "qkv_projection" in e["name"]]
print("qkv kernels", len(qkv), "-> target forwards", len(qkv) / 62)
# split timeline into steps at each first-layer qkv (every 62nd)
starts = [qkv[i]["ts"] for i in range(0, len(qkv), 62)]


def cls(n):
    pats = [
        ("target.qkv", r"qkv_projection"),
        ("target.attn", r"fmhaSm100f.*H64"),
        ("target.experts", r"bmm_E2m1|bmm_Bfloat16_E2m1|bmm_Bfloat16_Bfloat16"),
        (
            "target.finalize",
            r"kernel_cutlass_fused_kernel|kernel_cutlass_single_kernel",
        ),
        (
            "target.boundary",
            r"PublishingGemm|AllReduceDualNorm|AllGatherDualNorm|AllGatherNormQuant|SharedOnlyPublish",
        ),
        ("target.router", r"RouterKernel|routing|fp32_router"),
        ("target.gemm_kimi", r"SplitKGemm"),
        ("prefetch", r"L2Pref"),
        ("norm/gate", r"rmsnorm_sigmoid|dual_norm|_rmsnorm|gated_residual"),
        ("allreduce", r"[Aa]llreduce|all_reduce|nccl"),
        ("draft.attn", r"flash|fmha|Fmha"),
        ("gemm(other)", r"nvjet|cutlass|gemm|Gemm|sm100"),
        ("sample/topk", r"topk|TopK|sort|softmax|sample|argmax|gumbel|reject"),
        ("memcpy", r"Memcpy|Memset|memcpy|memset"),
        ("triton", r"^triton|_kernel$"),
    ]
    for c, p in pats:
        if re.search(p, n):
            return c
    return "other"


win_start, win_end = starts[1], starts[-1]  # full steps only
steps = len(starts) - 2
tot = collections.Counter()
busy = 0.0
last_end = win_start
idle = 0.0
names = collections.Counter()
for e in k:
    if e["ts"] < win_start or e["ts"] >= win_end:
        continue
    tot[cls(e["name"])] += e["dur"]
    names[(cls(e["name"]), e["name"][:70])] += e["dur"]
    if e["ts"] > last_end:
        idle += e["ts"] - last_end
    last_end = max(last_end, e["ts"] + e["dur"])
span = win_end - win_start
print(
    f"steps {steps}, span per step {span / steps:.0f} us, GPU idle per step {idle / steps:.0f} us"
)
for c, v in tot.most_common():
    print(f"  {c:18s} {v / steps:8.1f} us/step")
print("top non-target kernels:")
for (c, n), v in names.most_common(60):
    if not c.startswith("target") and c not in ("prefetch",):
        print(f"  {v / steps:7.1f} {c:14s} {n}")

print("\nper-step split:")
tgt_pat = re.compile(
    r"qkv_projection|fmhaSm100f.*H64|bmm_E2m1|bmm_Bfloat16_E2m1|bmm_Bfloat16_Bfloat16|kernel_cutlass_fused_kernel|PublishingGemm|AllReduceDualNorm|AllGatherDualNorm|AllGatherNormQuant|SharedOnlyPublish|RouterKernel|routingIndices|SplitKGemm|rmsnorm_sigmoid|dual_norm|L2Pref|oneshotAllreduce"
)
rows = []
for i in range(1, len(starts) - 1):
    s0, s1 = starts[i], starts[i + 1]
    inwin = [e for e in k if s0 <= e["ts"] < s1]
    tend = max(e["ts"] + e["dur"] for e in inwin if tgt_pat.search(e["name"]))
    # first non-target kernel after the target forward
    rest = [e for e in inwin if e["ts"] >= tend]
    rows.append((tend - s0, s1 - tend))
    if i == 1:
        rk = collections.Counter()
        for e in rest:
            rk[e["name"][:80]] += e["dur"]
        print("  first step: kernels after the target forward (us):")
        for n, v in rk.most_common(25):
            print(f"    {v:7.1f} {n}")
import statistics

print(
    f"  target forward {statistics.median(r[0] for r in rows):.0f} us, rest of step {statistics.median(r[1] for r in rows):.0f} us (median of {len(rows)})"
)
