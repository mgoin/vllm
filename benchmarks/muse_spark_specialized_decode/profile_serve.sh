#!/bin/bash
# Real-weight specialized server: torch-profile a few steady-state decode steps at c=1.
D=${OUT_DIR:-$(pwd)/serve_out}; mkdir -p $D
export PATH=/home/mgoin/code/vllm/.venv/bin:$PATH
export PYTHONPATH=${SNAP:-/home/mgoin/code/vllm-muse-sd}
export HF_HOME=/data/mgoin HF_HUB_OFFLINE=1
MODEL=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
PORT=8131
SPEC='{"method":"dflash","model":"RadixArk/Muse-Spark-Open-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
rm -rf $D/prof/*
setsid vllm serve $MODEL --port $PORT --tensor-parallel-size 4 \
  --additional-config '{"decode_backend":"latency_specialized"}' \
  --gpu-memory-utilization 0.85 --speculative-config "$SPEC" \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$D/prof\",\"ignore_frontend\":true,\"torch_profiler_with_stack\":false,\"max_iterations\":12,\"delay_iterations\":40}" \
  > $D/server_prof.log 2>&1 &
SP=$!
until curl -sf localhost:$PORT/health >/dev/null; do kill -0 $SP 2>/dev/null || { echo died; exit 1; }; sleep 5; done
echo ready
curl -s -X POST localhost:$PORT/start_profile
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --port $PORT --model $MODEL \
  --dataset-name custom --dataset-path /home/mgoin/nightly-test/mixed64.jsonl --custom-output-len 512 \
  --num-prompts 2 --max-concurrency 1 --temperature 0 --ignore-eos > $D/prof_bench.log 2>&1
curl -s -X POST localhost:$PORT/stop_profile
sleep 60
ls -la $D/prof | head
kill -TERM -- -$SP; sleep 15; kill -KILL -- -$SP 2>/dev/null
echo PROF_DONE
