# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
"""Per-layer time from two `vllm bench forward --output-json` runs.

python per_layer.py l4.json l8.json [--full-layers 62]

per_layer = (T8 - T4) / (L8 - L4); overhead = T4 - L4 * per_layer (includes
layer 0's extra BF16-expert cost); projection = overhead + 62 * per_layer.
"""

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("short")
    p.add_argument("long")
    p.add_argument("--short-layers", type=int, default=4)
    p.add_argument("--long-layers", type=int, default=8)
    p.add_argument("--full-layers", type=int, default=62)
    a = p.parse_args()
    s = {r["case"]: r["median_us"] for r in json.load(open(a.short))}
    l = {r["case"]: r["median_us"] for r in json.load(open(a.long))}
    dl = a.long_layers - a.short_layers
    print(
        f"{'case':<12} {'T4 us':>9} {'T8 us':>9} {'layer us':>9} "
        f"{'overhead':>9} {'62L ms':>8}"
    )
    for case in s:
        if case not in l:
            continue
        per = (l[case] - s[case]) / dl
        over = s[case] - a.short_layers * per
        proj = (over + a.full_layers * per) / 1000
        print(
            f"{case:<12} {s[case]:9.1f} {l[case]:9.1f} {per:9.1f} "
            f"{over:9.1f} {proj:8.2f}"
        )


if __name__ == "__main__":
    main()
