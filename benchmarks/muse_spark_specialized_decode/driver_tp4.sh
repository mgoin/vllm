#!/usr/bin/env bash
# Production-shape (TP4, no EP) per-layer table + kernel timelines, dummy weights.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p res
export EP=0 AUTOTUNE=0
LAYERS=4 CASES=verify OUT=res/tp4_l4_v.json ./run_microbench.sh
LAYERS=8 CASES=verify OUT=res/tp4_l8_v.json ./run_microbench.sh
echo "=== TP4 VERIFY (per layer) ==="
python per_layer.py res/tp4_l4_v.json res/tp4_l8_v.json
rm -rf prof_tp4
PROFILE=prof_tp4 LAYERS=8 CASES="--case v1x8_1k=1*8@1024:7 --case v8x8_1k=8*8@1024:7 --case v32x8_1k=32*8@1024:7" \
  OUT=res/tp4_prof.json ./run_microbench.sh
echo TP4_DONE
