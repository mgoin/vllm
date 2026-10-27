#!/bin/bash
# Torch-profile steady decode steps of one layout at a chosen concurrency.
# usage: chg run -g=4 --timeout 45m -- bash profile_serve_conc.sh tp4 16
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=${1:?layout}; C=${2:-16}
D=${OUT_DIR:-$HERE/serve_prof/$L}
mkdir -p "$D"; rm -rf "$D/prof"; mkdir -p "$D/prof"
export PATH=/home/mgoin/code/vllm/.venv/bin:$PATH
export PYTHONPATH=${SNAP:-/home/mgoin/code/vllm-muse-sd}
export HF_HOME=/models/mgoin
MODEL=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
PORT=${PORT:-8143}
DATA=${DATA:-/home/mgoin/muse_eval/mixed64.jsonl}
SPEC='{"method":"dflash","model":"RadixArk/Muse-Spark-Open-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
EXTRA=()
[ "$L" = tp4sd ] && EXTRA=(--additional-config '{"decode_backend":"latency_specialized"}')
setsid vllm serve $MODEL --port $PORT --tensor-parallel-size 4 "${EXTRA[@]}" --max-model-len 32768 \
  --reasoning-parser muse --enable-auto-tool-choice --tool-call-parser muse \
  --gpu-memory-utilization 0.85 --speculative-config "$SPEC" \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$D/prof\",\"ignore_frontend\":true,\"torch_profiler_with_stack\":false,\"max_iterations\":24,\"delay_iterations\":30}" \
  > "$D/server.log" 2>&1 &
SP=$!
until curl -sf localhost:$PORT/health >/dev/null; do
  kill -0 $SP 2>/dev/null || { echo "server died"; exit 1; }; sleep 5
done
echo "[$L c=$C] ready"
curl -s -X POST localhost:$PORT/start_profile
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --port $PORT --model $MODEL \
  --dataset-name custom --dataset-path "$DATA" --custom-output-len 512 \
  --num-prompts $((C * 3)) --max-concurrency $C --temperature 0 --ignore-eos > "$D/bench.log" 2>&1
curl -s -X POST localhost:$PORT/stop_profile
sleep 60
kill -TERM -- -$SP 2>/dev/null; sleep 15; kill -KILL -- -$SP 2>/dev/null
ls -la "$D/prof" | head -5
echo PROF_DONE
