#!/bin/bash
# Reproduction of the handoff numbers on this box (dgx-b300-1, HF cache /models/mgoin).
# usage: chg run -g=4 -- bash repro.sh [layout ...]     layouts: tp4 tp4sd
# Each layout starts its own server, benches c=1/4/16 on mixed64.jsonl, stops.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D=${OUT_DIR:-$HERE/repro_out}; mkdir -p "$D"
export PATH=/home/mgoin/code/vllm/.venv/bin:$PATH
export PYTHONPATH=${SNAP:-/home/mgoin/code/vllm-muse-sd}
export HF_HOME=/models/mgoin
MODEL=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
PORT=${PORT:-8141}
DATA=${DATA:-/home/mgoin/muse_eval/mixed64.jsonl}
OUTLEN=${OUTLEN:-1024}
SPEC='{"method":"dflash","model":"RadixArk/Muse-Spark-Open-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then
  echo "FATAL: something already serves on port $PORT"; exit 1
fi
python -c "import vllm; print('vllm from', vllm.__file__)"
metrics() { curl -s localhost:$PORT/metrics | grep -E "^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|^vllm:generation_tokens_total"; }

for L in "$@"; do
  EXTRA=()
  [ "$L" = tep4 ] && EXTRA=(--enable-expert-parallel)
  [ "$L" = tp4sd ] && EXTRA=(--additional-config '{"decode_backend":"latency_specialized"}')
  t0=$(date +%s)
  setsid vllm serve $MODEL --port $PORT --tensor-parallel-size 4 "${EXTRA[@]}" \
    --max-model-len 32768 \
    --reasoning-parser muse --enable-auto-tool-choice --tool-call-parser muse \
    --gpu-memory-utilization 0.85 --speculative-config "$SPEC" > "$D/server_$L.log" 2>&1 &
  SP=$!
  until curl -sf localhost:$PORT/health >/dev/null; do
    if ! kill -0 $SP 2>/dev/null; then echo "[$L] server died"; break; fi
    sleep 5
  done
  if kill -0 $SP 2>/dev/null; then
    echo "[$L] ready after $(( $(date +%s) - t0 ))s"
    for CN in ${CONCS:-"1 16" "4 32" "16 64"}; do
      set -- $CN; C=$1; N=$2
      metrics > "$D/m_${L}_c${C}_before.txt"
      vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --port $PORT \
        --model $MODEL --dataset-name custom --dataset-path $DATA \
        --custom-output-len $OUTLEN --num-prompts $N --max-concurrency $C \
        --temperature 0 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
        --save-result --result-dir "$D" --result-filename "bench_${L}_c${C}.json" \
        > "$D/bench_${L}_c${C}.log" 2>&1
      metrics > "$D/m_${L}_c${C}_after.txt"
      echo "[$L] c=$C done: $(grep -E 'Output token throughput|Median TPOT' "$D/bench_${L}_c${C}.log" | tr -s ' ' | tr '\n' ';')"
    done
  fi
  kill -TERM -- -$SP 2>/dev/null; sleep 15; kill -KILL -- -$SP 2>/dev/null
  wait $SP 2>/dev/null
  sleep 5
done
echo ALL_DONE
