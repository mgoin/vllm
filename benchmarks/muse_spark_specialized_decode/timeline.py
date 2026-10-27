# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
"""Kernel timeline of one profiled forward from a `vllm bench forward` trace.

python timeline.py DIR/<case>.rank0.trace.json.gz [--replay 1] [--layer-start REGEX]
    [--layers a,b] [--map]

Prints every GPU activity of one replay's forward in launch order: start offset,
duration, gap since the previous activity ended (idle GPU when > 0, overlap when
< 0), stream and short name. With --layer-start, the forward is cut into layers
at each activity whose short name matches REGEX, and per-layer span / busy /
idle are printed. --map labels each kernel with the Muse op it implements.
"""

import argparse
import gzip
import json
import os
import regex as re
import sys

sys.path.insert(0, os.environ.get("PYTHONPATH", ""))
from vllm.benchmarks.forward.profile import _GPU_CATEGORIES, short_name  # noqa: E402

# Ordered (regex on full kernel name, op label); first match wins. Dense GEMMs
# (nvjet/cuBLAS) are labeled by their position within the layer instead.
OP_MAP = [
    (r"vocab_parallel_embedding", "embedding"),
    (
        r"oneshotAllreduceFusion|twoshotAllreduce|AllreduceFusion",
        "all-reduce (o_proj / embed)",
    ),
    (r"AllGatherDualNorm", "all-gather unpooler + dual-norm boundary"),
    (r"AllGatherNormQuant", "all-gather pooler + pre-expert norm + fp4 quant"),
    (r"SharedOnlyPublish", "mnnvl all-gather publish"),
    (
        r"_dual_norm_kernel|gated_residual_next_rmsnorm",
        "dual-norm boundary (attn->ffn)",
    ),
    (r"rmsnorm_sigmoid_gate", "o-norm-sigmoid-gate"),
    (r"QKNormRopeKVInsert|qk_norm_rope", "qk-norm/rope/kv-insert"),
    (r"fmha|Fmha|BatchDecode|BatchPrefill|xqa", "attention"),
    (
        r"fp32_router_gemm|router_gemm|^_splitk_reduce_kernel",
        "router gemm (aux stream)",
    ),
    (r"^_routing_kernel|musespark1x_routing", "routing top-k (aux stream)"),
    (r"routingIndices|routingCustom", "moe routing/permute indices"),
    (r"bmm_E2m1_E2m1E2m1", "expert gemm1+swiglu (nvfp4)"),
    (r"bmm_Bfloat16_E2m1E2m1", "expert gemm2 (nvfp4)"),
    (r"bmm_Bfloat16_Bfloat16Bfloat16.*swiGlu", "expert gemm1+swiglu (bf16)"),
    (r"bmm_Bfloat16_Bfloat16Bfloat16", "expert gemm2 (bf16)"),
    (
        r"kernel_cutlass_fused_kernel|kernel_cutlass_single_kernel",
        "finalize (TP AR + post-expert norm + topk sum)",
    ),
    (r"mnnvl_lamport_all_gather", "all-gather (lamport)"),
    (r"direct_copy_kernel", "copy (q contiguous)"),
    (r"_rmsnorm_fwd_kernel", "rmsnorm"),
    (r"elementwise_kernel", "elementwise (torch)"),
    (r"Memcpy|Memset|memcpy|memset", "memcpy/memset"),
]
DENSE = r"nvjet|cublas|cutlass.*gemm|sm100_xmma|gemm_kernel"
DENSE_ORDER = ["qkv+ogate proj", "o_proj", "pooler", "unpooler"]


def label(name: str) -> str:
    for pat, op in OP_MAP:
        if re.search(pat, name):
            return op
    if re.search(DENSE, name):
        return "dense gemm"
    return "?"


def label_layer(events) -> list[str]:
    """Labels for one layer's events; dense GEMMs by order, splitK reduce
    inherits the GEMM before it."""
    out, n, last = [], 0, "?"
    for e in events:
        lab = label(e["name"])
        if "splitKreduce" in e["name"]:
            lab = last
        elif lab == "dense gemm":
            lab = DENSE_ORDER[n] if n < len(DENSE_ORDER) else f"dense gemm {n}"
            n += 1
        if e["tid"] == events[0]["tid"] or "aux" not in lab:
            last = lab if "splitKreduce" not in e["name"] else last
        out.append(lab)
    return out


def load_forward(path: str, replay: int):
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    acts = sorted(
        (e for e in events if e.get("cat") in _GPU_CATEGORIES and "dur" in e),
        key=lambda e: e["ts"],
    )
    summary_path = path.replace(".trace.json.gz", ".kernels.json")
    with open(summary_path) as f:
        summary = json.load(f)
    replays = summary["replays"]
    per = len(acts) // replays
    skip = per - summary["activities"]
    return acts[replay * per : (replay + 1) * per][skip:]


def union_us(acts) -> float:
    total, end = 0.0, None
    start = None
    for e in sorted(acts, key=lambda e: e["ts"]):
        s, t = e["ts"], e["ts"] + e["dur"]
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, t
        else:
            end = max(end, t)
    if end is not None:
        total += end - start
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--replay", type=int, default=1)
    p.add_argument(
        "--layer-end",
        default=r"AllGatherDualNorm",
        help="regex on the kernel name that ENDS a layer (inclusive)",
    )
    p.add_argument("--layers", help="print only these layers, e.g. 1,2")
    p.add_argument("--quiet", action="store_true", help="tables only")
    a = p.parse_args()
    fwd = load_forward(a.trace, a.replay)
    t0 = fwd[0]["ts"]
    groups, cur = [], []
    for e in fwd:
        cur.append(e)
        if re.search(a.layer_end, e["name"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    want = {int(x) for x in a.layers.split(",")} if a.layers else None
    end_prev = t0
    rows = []
    for li, ev in enumerate(groups):
        labs = label_layer(ev)
        for e, lab in zip(ev, labs):
            gap = e["ts"] - end_prev
            # Exclusive: the part of this kernel's interval not already covered
            # by earlier kernels (PDL lets kernels launch before the previous
            # one ends, so raw durations double-count). Sums to busy(union).
            excl = max(0.0, e["ts"] + e["dur"] - max(e["ts"], end_prev))
            end_prev = max(end_prev, e["ts"] + e["dur"])
            rows.append((li, e, lab, gap, excl))
    if not a.quiet:
        print(
            f"{'layer':>5} {'start':>8} {'dur':>7} {'gap':>6} {'tid':>6}  "
            f"{'excl':>6}  {'op':<46} kernel"
        )
        for li, e, lab, gap, excl in rows:
            if want is not None and li not in want:
                continue
            name = short_name(e["name"]) if e["cat"] == "kernel" else e["name"]
            print(
                f"{li:5d} {e['ts'] - t0:8.1f} {e['dur']:7.1f} {gap:6.1f} "
                f"{e['tid']!s:>6} {excl:6.1f}  {lab:<46} {name[:90]}"
            )
    span = fwd[-1]["ts"] + fwd[-1]["dur"] - t0
    print(
        f"\nforward span {span:.1f} us, busy(union) {union_us(fwd):.1f} us, "
        f"idle {span - union_us(fwd):.1f} us, {len(fwd)} activities, "
        f"{len(groups)} segments (segment i ends at its i-th '{a.layer_end}')"
    )
    print(f"{'seg':>4} {'n':>4} {'span':>8} {'busy':>8} {'idle':>7} {'sum dur':>8}")
    for li, ev in enumerate(groups):
        nxt = (
            groups[li + 1][0]["ts"]
            if li + 1 < len(groups)
            else (max(e["ts"] + e["dur"] for e in ev))
        )
        sp = nxt - ev[0]["ts"]
        busy = union_us(ev)
        print(
            f"{li:4d} {len(ev):4d} {sp:8.1f} {busy:8.1f} {sp - busy:7.1f} "
            f"{sum(e['dur'] for e in ev):8.1f}"
        )
    if want:
        for li in sorted(want):
            agg = {}
            for l2, e, lab, gap, excl in rows:
                if l2 == li:
                    d, x = agg.get(lab, (0.0, 0.0))
                    agg[lab] = (d + e["dur"], x + excl)
            print(
                f"\nsegment {li} by op: kernel dur / exclusive (critical "
                f"path share), us"
            )
            for lab, (d, x) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
                print(f"  {d:7.1f} {x:7.1f}  {lab}")
            print(
                f"  {sum(v[0] for v in agg.values()):7.1f} "
                f"{sum(v[1] for v in agg.values()):7.1f}  total"
            )


if __name__ == "__main__":
    main()
