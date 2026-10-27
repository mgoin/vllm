# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
"""Per-step GPU span of a serving trace, for either decode path.

python step_span.py TRACE.json.gz [--anchor REGEX] [--per-step N]

Splits the GPU timeline into steps at every ``per-step``-th anchor kernel
(default: the executor's fused QKV kernel, 62 per target forward; pass
``--anchor fusedMuseSparkQKNormRopeKVInsertKernel`` for the stock path) and
prints, per step, the target-forward span and the span of everything after it
(drafter, sampling, logits) up to the next step's start.
"""

import argparse
import gzip
import json
import re


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--anchor", default="qkv_projection")
    p.add_argument("--per-step", type=int, default=62)
    a = p.parse_args()
    ev = json.load(gzip.open(a.trace))["traceEvents"]
    k = [
        e
        for e in ev
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    k.sort(key=lambda e: e["ts"])
    idx = [i for i, e in enumerate(k) if re.search(a.anchor, e["name"])]
    starts = [idx[i] for i in range(0, len(idx), a.per_step)]
    if len(starts) < 3:
        print(f"only {len(starts)} steps found with anchor {a.anchor!r}")
        return

    def span(i, j):
        seg = k[i:j]
        if not seg:
            return 0.0
        return max(e["ts"] + e["dur"] for e in seg) - seg[0]["ts"]

    fwd, rest, total = [], [], []
    for n, i in enumerate(starts[:-1]):
        j = starts[n + 1]
        f = span(i, i + a.per_step) if i + a.per_step <= len(k) else 0.0
        r = span(i + a.per_step, j) if i + a.per_step < j else 0.0
        fwd.append(f)
        rest.append(r)
        total.append(span(i, j))
    fwd, rest, total = fwd[1:], rest[1:], total[1:]  # drop the warm-up step
    med = lambda v: sorted(v)[len(v) // 2]
    print(
        f"steps {len(total)}  forward {med(fwd):.1f}us  after {med(rest):.1f}us  "
        f"step {med(total):.1f}us"
    )


if __name__ == "__main__":
    main()
