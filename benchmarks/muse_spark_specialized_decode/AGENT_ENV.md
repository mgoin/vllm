# Environment and workflow facts (Muse Spark specialized decode)

Read this before running anything. Written for agents working on the
`muse-spark-specialized-decode` branch on `dgx-b300-1`.

## Box

- 8 × B300 SXM6 (275 GB each, SM103), 256 CPU cores, ~2 TB RAM.
- Shared box discipline: **every GPU command goes through chg**:
  `chg run -g=4 --timeout 3600 -- <cmd>` (adds `CUDA_VISIBLE_DEVICES`, queues
  FCFS when the GPUs are taken, and releases on exit). Never run CUDA without
  it. At most one 4-GPU job per agent at a time; the orchestrator and the other
  agents are using the rest.
- Do not start long servers without `--timeout`; orphan servers block GPUs.

## Files

| what | where |
| --- | --- |
| main checkout (reference, **read-only for you**) | `/home/mgoin/code/vllm` |
| integration worktree (branch) | `/home/mgoin/code/vllm-muse-sd` |
| your worktree | create with `tools/setup_worktree.sh /home/mgoin/code/vllm-<task> fork/mgoin/muse-spark-specialized-decode` |
| Python | `/home/mgoin/code/vllm/.venv/bin/python` (never system python; never pip install) |
| torchrun | `/home/mgoin/code/vllm/.venv/bin/torchrun` |
| HF cache (model + draft) | `HF_HOME=/models/mgoin` |
| model | `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` |
| draft | `RadixArk/Muse-Spark-Open-DFlash2` (DFlash2, 8 GB, cached) |
| bench prompts | `/home/mgoin/muse_eval/mixed64.jsonl` (64 chat prompts, mixed lengths) |
| harness | `<worktree>/benchmarks/muse_spark_specialized_decode/` |

Each worktree needs its own symlinks to the precompiled extensions and the
generated third-party trees; `tools/setup_worktree.sh` does that and import
checks the result. Always run with `PYTHONPATH=<your worktree>`; the venv's
editable install otherwise points at the main checkout.

## Measuring

Fast loop (dummy weights, truncated depth, exact shapes, kernel breakdown):

```bash
cd benchmarks/muse_spark_specialized_decode
LAYERS=4 CASES=verify OUT=res/l4.json ./run_microbench.sh     # ~4 min
LAYERS=8 CASES=verify OUT=res/l8.json ./run_microbench.sh
python per_layer.py res/l4.json res/l8.json                   # per-layer us
PROFILE=prof_l4 LAYERS=4 CASES="--case v8x8_1k=8*8@1024:7" ./run_microbench.sh
python timeline.py prof_l4/v8x8_1k.rank0.trace.json.gz --layers 1,2
```

`EXTRA='--additional-config {"decode_backend":"latency_specialized"}'` runs the
executor. Note the dummy loader skips `load_weights`, and the harness installs
the post-load hooks Muse needs (`muse_bench.py`).

End-to-end (real weights, server): `repro.sh <layout...>` with layouts `tp4`
(stock) / `tp4sd` (executor); `python summ.py repro_out` prints
step/TPOT/AL per case. Accuracy: `tests/evals/gsm8k/gsm8k_eval.py --port ...`.

## Roofline (the number that matters)

A decode step streams weights, nothing else. Per rank at TP4:

- one expert = 6.29 MB NVFP4 (`w13` 4096×2048 + `w2` 4096×1024, 0.5 B/param);
  256 experts per layer = 1.61 GB if every expert is touched;
- qkv 18.9 MB, o_proj 8.4 MB, pooler+unpooler 33.6 MB, router 8.4 MB per layer;
- KV reads at ctx 8192: 0.46 GB per step (46 SWA + 16 full layers).

Distinct experts touched per layer ≈ `256 (1 - (1 - 1/256)^(rows*8))`.
At 7.5 TB/s: 8 rows ≈ 27 GB ≈ 3.6 ms; 32 rows ≈ 68 GB ≈ 9.1 ms; 128 rows
≈ 104 GB ≈ 13.9 ms. Report achieved GB/s, not just ms.

## Pitfalls (learned the hard way)

- The dummy loader skips `MuseSpark1xForCausalLM.load_weights`, so residual-gate
  alpha/beta are never materialized and boundaries fall back to ~25 unfused
  kernels; the harness and tests call the post-load hooks.
- Never H2D-copy from non-pinned memory inside a call: calls run during graph
  capture with fresh graph-pool tensors. Build TMA descriptors ahead.
- Never read routing ids (or any producer output) before
  `griddepcontrol.wait`: this hung the real server in adaptive-verification
  profiling replays.
- Persistent kernels whose CTAs wait on each other need co-residency; side
  streams (prefetch, router) must never wait on a collective.
- `--gpu-memory-utilization 0.85` avoids a stock startup OOM in sampler warmup.
- First server start of a new config pays ~4-8 min of FlashInfer autotune
  (`~/.cache/vllm/flashinfer_autotune_cache`, keyed by the full config hash);
  keep configs identical between A/B runs to reuse it.
- Advisory: two 4-GPU jobs can run at once, but do not A/B two variants at the
  same time on the same GPUs — measure alone for final numbers.
