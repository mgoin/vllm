#!/usr/bin/env bash
# Muse Spark Open per-layer forward microbench (dummy weights, truncated depth).
#
#   LAYERS=4|8      truncate to the first N layers of the real layer pattern
#   EP=1|0          TP4 with/without --enable-expert-parallel
#   CASES=verify|decode|all|prof|pool|"<custom --case args>"
#   PROFILE=DIR     add --profile-dir DIR (torch profiler kernels + traces)
#   OUT=file.json   --output-json
#   EXTRA="..."     extra bench/engine args (e.g. --moe-backend deep_gemm_mega_moe)
#   MUSE_ROUTING=random|balanced|pool:K, MUSE_ROUTING_DIAG=1 (see muse_bench.py);
#     a case named *_pK routes from a K-expert pool regardless (CASES=pool sweep)
#   MUSE_TUNE_EXACT=1  autotune FlashInfer on each case's exact shapes first
#
# Full sweep in one reservation: driver2.sh (logs/, res/, prof_*/).
# Tables: python per_layer.py res/l4_ep0.json res/l8_ep0.json
# Kernels: python timeline.py prof_l4_ep0/v1x8_1k.rank0.trace.json.gz --layers 1,2
#   (segment 1 = full/NoPE layer, 2 = sliding layer; both NVFP4 experts)
# After a kill, run ./cleanup.sh: torchrun workers have their own process group.
#
# Example: LAYERS=8 EP=1 OUT=l8_ep.json ./run_microbench.sh
# Per-layer time = (T[L=8] - T[L=4]) / 4, layers 4..7 = [swa, full, swa, swa],
# all NVFP4 experts (layer 0 has BF16 experts in the checkpoint).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYERS=${LAYERS:-4}
EP=${EP:-1}
CASES=${CASES:-all}
TP=${TP:-4}
MODEL=meta-models/Muse-Spark-Open-816B-A42B-NVFP4

export PATH=/home/mgoin/code/vllm/.venv/bin:$PATH
export PYTHONPATH=${SNAP:-/home/mgoin/code/vllm-muse-sd}
export HF_HOME=${HF_HOME:-/models/mgoin}
export MUSE_ROUTING=${MUSE_ROUTING:-random}

OVR=$(python - "$LAYERS" <<'EOF'
import json, sys
from huggingface_hub import hf_hub_download
n = int(sys.argv[1])
t = json.load(open(hf_hub_download("meta-models/Muse-Spark-Open-816B-A42B-NVFP4", "config.json")))["text_config"]
print(json.dumps({"text_config": {"num_hidden_layers": n, **{k: t[k][:n] for k in ("layer_types", "mlp_layer_types", "layer_rope_theta")}}}))
EOF
)

verify_cases=()
for ctx in 1024 16384; do
  for b in 1 2 4 8 16 32; do
    verify_cases+=(--case "v${b}x8_$((ctx / 1024))k=${b}*8@${ctx}:7")
  done
done
decode_cases=()
for ctx in 1024 16384; do
  for b in 1 8 64; do
    decode_cases+=(--case "d${b}x1_$((ctx / 1024))k=${b}*1@${ctx}")
  done
done
case "$CASES" in
  verify) case_args=("${verify_cases[@]}") ;;
  decode) case_args=("${decode_cases[@]}") ;;
  all) case_args=("${verify_cases[@]}" "${decode_cases[@]}") ;;
  prof) case_args=(--case "v1x8_1k=1*8@1024:7" --case "v8x8_1k=8*8@1024:7"
                   --case "v1x8_16k=1*8@16384:7" --case "v8x8_16k=8*8@16384:7") ;;
  pool) case_args=()  # routing pool sweep: case suffix _pK = K-expert pool
        for k in 8 16 32; do case_args+=(--case "v1x8_1k_p${k}=1*8@1024:7"); done
        for k in 8 16 32 64 128; do case_args+=(--case "v8x8_1k_p${k}=8*8@1024:7"); done
        case_args+=(--case "v1x8_1k=1*8@1024:7" --case "v8x8_1k=8*8@1024:7") ;;
  *) read -r -a case_args <<<"$CASES" ;;
esac

args=(bench forward --model "$MODEL" --language-model-only
  -tp "$TP"
  --speculative-config "{\"method\":\"dflash\",\"model\":\"RadixArk/Muse-Spark-Open-DFlash2\",\"num_speculative_tokens\":7}" --exclude-drafter
  --kv-len-jitter 0 --max-model-len 20480
  "${case_args[@]}")
[[ "$EP" == 1 ]] && args+=(--enable-expert-parallel)
[[ -n "${PROFILE:-}" ]] && args+=(--profile-dir "$PROFILE")
# AUTOTUNE=0 skips FlashInfer autotuning (minutes of startup per run) for fast
# kernel iteration; measure with the default (1) before you report numbers.
if [ "${AUTOTUNE:-1}" = "0" ]; then args+=(--disable-flashinfer-autotune); fi
[[ -n "${OUT:-}" ]] && args+=(--output-json "$OUT")
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && args+=($EXTRA)

exec torchrun --nproc-per-node="$TP" "$HERE/muse_bench.py" "${args[@]}"
