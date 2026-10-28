#!/bin/bash
# Two TP4 replicas on one node against one TP8 replica: aggregate throughput.
#   chg run -g=8 --timeout 45m -- bash dual_tp4.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D=${OUT_DIR:-$HERE/dual_out}; mkdir -p "$D"
export PATH=/home/mgoin/code/vllm/.venv/bin:$PATH
export PYTHONPATH=${SNAP:-/home/mgoin/code/vllm-muse-sd}
export HF_HOME=/models/mgoin
MODEL=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
DATA=${DATA:-/home/mgoin/muse_eval/mixed64.jsonl}
SPEC='{"method":"dflash","model":"RadixArk/Muse-Spark-Open-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
SD='{"decode_backend":"latency_specialized"}'
metrics() { curl -s "localhost:$1/metrics" | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total"; }

start() {  # port gpus extra_args...
  local port=$1 gpus=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpus setsid vllm serve $MODEL --port $port --tensor-parallel-size 4 "$@" \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --speculative-config "$SPEC" \
    > "$D/server_$port.log" 2>&1 &
  echo $!
}

bench() {  # port concurrency prompts tag
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --port $1 \
    --model $MODEL --dataset-name custom --dataset-path "$DATA" \
    --custom-output-len 1024 --num-prompts $3 --max-concurrency $2 \
    --temperature 0 --ignore-eos --percentile-metrics ttft,tpot --metric-percentiles 50 \
    --result-dir "$D" --result-filename "bench_$4.json" > "$D/bench_$4.log" 2>&1
}

# One TP8 replica, the reference.
setsid vllm serve $MODEL --port 8160 --tensor-parallel-size 8 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --speculative-config "$SPEC" \
  > "$D/server_tp8.log" 2>&1 &
P8=$!
until curl -sf localhost:8160/health >/dev/null; do kill -0 $P8 2>/dev/null || { echo "tp8 died"; break; }; sleep 5; done
echo "[tp8] ready"
bench 8160 1 16 tp8_c1
bench 8160 16 64 tp8_c16
kill -TERM -- -$P8 2>/dev/null; sleep 20; kill -KILL -- -$P8 2>/dev/null; sleep 10

# Two TP4 replicas on the same node, executor on.
A=$(start 8161 0,1,2,3 --additional-config "$SD")
B=$(start 8162 4,5,6,7 --additional-config "$SD")
for p in 8161 8162; do
  until curl -sf localhost:$p/health >/dev/null; do kill -0 $A 2>/dev/null || { echo "replica $p died"; break; }; sleep 5; done
done
echo "[2x tp4] ready"
# Concurrent: one request per replica, then 8 per replica.
bench 8161 1 16 dual_c1_a & bench 8162 1 16 dual_c1_b & wait
bench 8161 8 32 dual_c8_a & bench 8162 8 32 dual_c8_b & wait
kill -TERM -- -$A -- -$B 2>/dev/null; sleep 20; kill -KILL -- -$A -- -$B 2>/dev/null
echo DUAL_DONE
