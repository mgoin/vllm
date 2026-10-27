# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Summarize a repro.sh run: step time = median TPOT x acceptance length.
# ruff: noqa
import glob
import json
import os
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def m(p):
    d = {}
    if not os.path.exists(p):
        return d
    for line in open(p):
        k, v = line.rsplit(" ", 1)
        k = re.sub(r"\{.*\}", "", k)
        d[k] = d.get(k, 0) + float(v)
    return d


rows = []
for f in sorted(glob.glob(f"{D}/bench_*.json")):
    L, c = re.match(r".*bench_(\w+)_c(\d+)\.json", f).groups()
    b = json.load(open(f))
    a = m(f"{D}/m_{L}_c{c}_after.txt")
    bb = m(f"{D}/m_{L}_c{c}_before.txt")
    dd = {k: a[k] - bb.get(k, 0) for k in a}
    drafts = dd.get("vllm:spec_decode_num_drafts_total", 0)
    acc = dd.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    dt = dd.get("vllm:spec_decode_num_draft_tokens_total", 0)
    al = 1 + acc / drafts if drafts else float("nan")
    tpot = b["median_tpot_ms"]
    rows.append(
        f"{L:6s} c={c:>2} out_tok/s={b['output_throughput']:8.1f} "
        f"per-user={1000 / tpot:6.1f} medTPOT={tpot:6.2f}ms AL={al:5.2f} "
        f"drafted/step={dt / drafts if drafts else 0:5.2f} step={tpot * al:6.2f}ms "
        f"ttft_med={b['median_ttft_ms']:8.1f} dur={b['duration']:.0f}s"
    )
print("\n".join(rows))
