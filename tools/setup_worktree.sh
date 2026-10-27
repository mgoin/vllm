#!/usr/bin/env bash
# Create a Muse Spark specialized-decode worktree from a branch and make it
# importable with the precompiled extensions of the main checkout.
#
#   tools/setup_worktree.sh <path> [<ref>]      default ref: fork/mgoin/muse-spark-specialized-decode
#
# The worktree shares the main checkout's compiled .so files and its generated
# CUTLASS/FlashInfer third_party trees, so no build is needed.
set -euo pipefail
MAIN=/home/mgoin/code/vllm
DEST=${1:?usage: setup_worktree.sh <path> [ref]}
REF=${2:-fork/mgoin/muse-spark-specialized-decode}

if [ ! -d "$DEST" ]; then
  git -C "$MAIN" worktree add "$DEST" "$REF"
fi

# Compiled extensions (gitignored: built once in the main checkout).
while IFS= read -r f; do
  src="$MAIN/$f"
  dst="$DEST/$f"
  if [ -e "$dst" ]; then continue; fi
  mkdir -p "$(dirname "$dst")"
  ln -s "$src" "$dst"
done < <(cd "$MAIN" && find vllm -name "*.so")

# Generated third-party Python trees (gitignored).
for d in vllm/third_party/deep_gemm vllm/third_party/fmha_sm100 \
         vllm/third_party/tml_fa4 vllm/third_party/triton_kernels; do
  if [ -e "$DEST/$d" ]; then continue; fi
  mkdir -p "$(dirname "$DEST/$d")"
  ln -s "$MAIN/$d" "$DEST/$d"
done
f=vllm/third_party/flashmla/flash_mla_interface.py
if [ ! -e "$DEST/$f" ]; then ln -s "$MAIN/$f" "$DEST/$f"; fi

# Import check from a neutral cwd, so the editable install of the main checkout
# cannot mask the worktree.
cd /tmp
export PYTHONPATH="$DEST"
"$MAIN/.venv/bin/python" -c 'import vllm, vllm.entrypoints.openai.api_server
from vllm.models.muse_spark.nvidia.specialized_decode import specialized_decode_config
print("worktree ok:", vllm.__file__)'
