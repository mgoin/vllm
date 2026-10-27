# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
import glob
import json
import os
import regex as re

D = os.path.dirname(os.path.abspath(__file__))


def m(p):
    d = {}
    for l in open(p):
        k, v = l.rsplit(" ", 1)
        d[re.sub(r"\{.*\}", "", k)] = d.get(re.sub(r"\{.*\}", "", k), 0) + float(v)
    return d


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
    steps_per_req = drafts / int(c) if drafts else 0
    print(
        f"{L:5s} c={c:>2} out_tok/s={b['output_throughput']:8.1f} per-user={1000 / b['median_tpot_ms']:6.1f} medTPOT={b['median_tpot_ms']:.2f}ms AL={al:.2f} drafted/step={dt / drafts if drafts else 0:.2f} step≈{b['median_tpot_ms'] * al:.2f}ms dur={b['duration']:.0f}s"
    )
