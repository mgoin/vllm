#!/bin/bash
# One-command validation of a snapshot: per-layer at 8/16/32/64 rows, then the
# end-to-end step time at c=1 and c=4, then a 400-question GSM8K A/B.
#
#   chg run -g=4 --timeout 90m -- env SNAP=/home/mgoin/code/vllm-snap V=v3 bash validate.sh
#
# Writes into ./val_$V/ (per-layer jsons, serve jsons, gsm8k logs).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SNAP=${SNAP:-/home/mgoin/code/vllm-snap}
V=${V:-adhoc}
D=$HERE/val_$V; mkdir -p "$D"
export SNAP PATH=/home/mgoin/code/vllm/.venv/bin:$PATH HF_HOME=/models/mgoin
PY=/home/mgoin/code/vllm/.venv/bin/python
MODEL=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
SPEC='{"method":"dflash","model":"RadixArk/Muse-Spark-Open-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
SD='{"decode_backend":"latency_specialized"}'

echo "=== per layer (executor), $SNAP"
for L in 4 8; do
  EP=0 AUTOTUNE=0 LAYERS=$L CASES=verify OUT="$D/l$L.json" \
    EXTRA="--additional-config $SD" ./run_microbench.sh >"$D/micro_l$L.log" 2>&1
done
$PY per_layer.py "$D/l4.json" "$D/l8.json" | tee "$D/per_layer.txt"

echo "=== end to end"
rm -rf "$D/serve"; OUT_DIR=$D/serve CONCS='1:16 4:32' SNAP=$SNAP PORT=8152 ./repro.sh tp4 tp4sd >"$D/serve.log" 2>&1
$PY summ.py "$D/serve" | tee "$D/serve.txt"

echo "=== gsm8k (400 questions, 5-shot)"
for L in tp4 tp4sd; do
  EXTRA=(); [ "$L" = tp4sd ] && EXTRA=(--additional-config "$SD")
  setsid vllm serve $MODEL --port 8153 --tensor-parallel-size 4 "${EXTRA[@]}" \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --speculative-config "$SPEC" \
    >"$D/server_$L.log" 2>&1 &
  SP=$!
  until curl -sf localhost:8153/health >/dev/null; do kill -0 $SP 2>/dev/null || break; sleep 5; done
  $PY /home/mgoin/code/vllm-muse-sd/tests/evals/gsm8k/gsm8k_eval.py --port 8153 \
    --num-questions 400 --max-concurrency 32 --save-results "$D/gsm8k_$L.json" \
    >"$D/gsm8k_$L.log" 2>&1
  echo "[$L] $(tr '\r' '\n' <"$D/gsm8k_$L.log" | grep -a -A2 '^Results' | tr '\n' ' ')"
  kill -TERM -- -$SP 2>/dev/null; sleep 15; kill -KILL -- -$SP 2>/dev/null; sleep 5
done
echo VALIDATE_DONE
