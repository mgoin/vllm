#!/usr/bin/env bash
# Baseline per-layer roofline table: dummy weights, L=4 vs L=8, verify + decode.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p res
LAYERS=4 CASES=verify OUT=res/l4_v.json ./run_microbench.sh
LAYERS=8 CASES=verify OUT=res/l8_v.json ./run_microbench.sh
LAYERS=4 CASES=decode OUT=res/l4_d.json ./run_microbench.sh
LAYERS=8 CASES=decode OUT=res/l8_d.json ./run_microbench.sh
echo "=== VERIFY (per layer) ==="
python per_layer.py res/l4_v.json res/l8_v.json
echo "=== DECODE (per layer) ==="
python per_layer.py res/l4_d.json res/l8_d.json
echo BASE_DONE
