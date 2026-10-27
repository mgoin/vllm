# Muse Spark Open specialized decode (TP4, Blackwell): status and handoff

Branch `mgoin/muse-spark-specialized-decode` (fork `mgoin/vllm`), work-in-progress.
Design doc of the package: `vllm/models/muse_spark/nvidia/specialized_decode/AGENTS.md`.
Benchmark/profiling harness: this directory.

## Goal

A Kimi K3 / DeepSeek-V4.1-Flash style latency-specialized decode path for
`meta-models/Muse-Spark-Open-816B-A42B-NVFP4` on B300/B200, TP4, coexisting with the
stock model (prefill and mixed steps stay stock; FULL-graph decode/verify steps run the
executor). Deployment being optimized (the user's best config):

```bash
vllm serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 --tensor-parallel-size 4 \
  --reasoning-parser muse --enable-auto-tool-choice --tool-call-parser muse \
  --speculative-config '{"method":"dflash","model":"RadixArk/Muse-Spark-Open-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --additional-config '{"decode_backend":"latency_specialized"}'
```

TP4 vs TEP4 (`--enable-expert-parallel`) was measured on main: same per-step time
(TP4 1-5% faster at c=1/4), so the executor targets TP4 without EP (experts
TP-sharded on the intermediate dim, every rank holds all 256 experts).

## Results so far (real weights, B300, TP4, mixed64.jsonl prompts, 1024 output tokens)

| | c=1 step / tok/s | c=4 step / tok/s | c=16 step / tok/s | GSM8K |
| --- | --- | --- | --- | --- |
| stock TP4 | 10.07 ms / 467 | 14.19 ms / 1276 | 20.12 ms / 3388 | 0.959 |
| specialized (current) | **7.83 ms / 602** (−22%, +29%) | **11.71 ms / 1357** (−17.5%) | 20.98 ms / 3342 (+4%) | 0.958 |

Step = median TPOT x acceptance length (AL); AL swings ±0.12 between server launches
(probabilistic drafting), so compare step times and TPOT, not tok/s, across runs.
GSM8K: `tests/evals/gsm8k/gsm8k_eval.py --max-concurrency 32` (5-shot, 1319 q).

Progression at c=1 (step ms): stock 10.07 → split-K GEMMs 9.50 → + fused QKV,
router, prefetch 8.86 → + boundary kernels ~8.9 (AL noise) → + o-gate fold, persistent
experts, drafter round 2 8.47 → + fused MoE tail 7.83.

Real-weight step profile at c=1 (before the experts kernel and tail):
target forward ~7.24 ms + ~1.6 ms after it (target lm_head + logits gather + rejection
sampling + DFlash2 drafter); GPU idle ~0.1 ms/step, so the step is GPU-bound and host
overhead is hidden by async scheduling.

## Architecture (what runs on a specialized step)

Enabled by `additional_config.decode_backend = "latency_specialized"`; checked once at
startup on every rank (MIN all-reduce of the requirements; see `__init__.py`). The
model state (`runtime/model_state.py`) builds `SpecializedDecode` (`runtime/decode.py`)
after weights load and before memory profiling/capture; every kernel compiles there.
`MuseSpark1xForCausalLM.forward` asks `plan()` per forward: only FULL CUDA graph
captures of pure decode/verify steps with <= 256 rows run the executor (so it is baked
into the FULL graphs; eager/PIECEWISE/prefill run stock). Adaptive verification's
ragged per-request query lengths only matter to the attention core, which uses the
stock FlashInfer trtllm-gen metadata. The executor keeps the stock FP32 aux hidden
states for the drafter.

Per layer, by stage (gates are rows = tokens of the step):

| Stage | Kernel(s) | Files | Gate |
| --- | --- | --- | --- |
| Attention-in boundary | FlashInfer AG dual-norm consumer of the previous layer's published unpooler | `boundary/boundary.py` (`AllGatherPublisher.gated_residual_next_rmsnorm`) | when the previous layer published |
| QKV + QK-norm + RoPE + KV insert | one CuTe tcgen05 split-K kernel, q and o_gate to contiguous buffers | `attention/qkv_projection.py` | <= 192 rows, BF16 KV cache |
| Attention core | stock FlashInfer trtllm-gen decode | — | — |
| o-gate + o_proj + AR + dual norm (+ FP32 router input) | publishing GEMM (gate folded in for <= 8 rows) + Lamport one/two-shot AR + dual-norm consumer | `boundary/{publishing_gemm,all_reduce_dual_norm,boundary}.py` | <= 128 rows |
| Router + top-8 | one CuTe kernel (BF16x3 split weights, exact) on the aux stream | `moe/router.py` | 8..128 rows, BF16 router input |
| Pooler shard + AG + pre-expert norm + NVFP4 quant | publishing GEMM into FlashInfer's AG norm-quant workspace + its consumer | `boundary/boundary.py` | all rows |
| Routed experts | persistent CuTe NVFP4 GEMM1+SwiGLU+quant+GEMM2, no permute kernel, bitwise = TRT-LLM-gen | `experts/routed_experts{,_cute}.py` | 2..16 rows (stock TRT-LLM above) |
| MoE tail | experts publish GEMM2 rows into a Lamport mailbox; one-shot finalize (sum, per-route norm, weighted sum) + publishing unpooler | `tail/{experts,finalize,tail}.py` | <= 16 rows, layers not feeding the drafter's aux states |
| Unpooler shard + publish | publishing GEMM into FlashInfer's AG dual-norm workspace (consumed by the next boundary) | `boundary/boundary.py` | <= 128 rows |
| Small BF16 projections otherwise | Kimi K3 split-K CuTe GEMM with per-(shape, rows) tactics | `runtime/linear.py` | <= 64 rows (pooler 32) |
| L2 weight prefetch | Kimi `PreparedWeightRanges` on a side stream: o_proj during attention, router+pooler during the AR, next layer's QKV head during the next boundary | `runtime/prefetch.py` | <= 64 rows |

BF16-expert layers (0 and 61) and the 5 layers feeding the drafter's aux hidden states
run the stock MoE/unpooler paths. Mailbox protocol (FlashInfer LL compatible):
symmetric-memory BF16 `[3 generations, tp, capacity, width]`, -0.0 sentinel,
producer writes after `griddepcontrol.wait` with multimem stores, consumer polls,
re-arms its slots and advances `[NEXT, ACTIVE]` stage state.

DFlash2 drafter (stock code path, all models): `fc` column-sharded with the custom
MNNVL all-gather, small-M GEMMs on the Kimi split-K kernel (exact Muse TP4 shapes only),
fused aux concat/cast, fused candidate head (~20 kernels → 6), fused Q/K norm + RoPE,
prepare-inputs padding fix (duplicates open PR #58633 — land that instead). Optional
FP8 candidate preselection via the existing `draft_lm_head_quantization` (default off;
not bit-exact, ~10% of rows see a different top-16, no measured AL cost; −60 µs/step).
Draft step at c=1: ~1262 → ~1030 µs.

## Validation

- Kernel tests (1 GPU): `test_attention_kernels.py`, `test_moe_kernels.py`,
  `test_boundary_kernels.py`, `test_expert_kernels.py` (includes degenerate routing
  under a watchdog), `test_tail_kernels.py` — 100+ tests, mostly bitwise vs the stock op.
- Cross-rank (4 GPU): `test_boundary_numerics.py`, `test_tail_numerics.py --replays 1000`
  (late rank, 1000 graph replays, re-arm checks).
- Engine (4 GPU): `test_layers.py` — 4-layer random-weight model + random DFlash2
  drafter, `latency_specialized` vs `auto`, with/without adaptive verification.
  Random weights are chaotic (NVFP4 activation quantization turns 1-ulp differences
  into percent-level changes), so it asserts layer-0 agreement and output median, not
  full-depth equality. Not re-run since the experts kernel and tail were integrated.
- Real weights: GSM8K within noise of stock on every round; AL unchanged within noise.
- Run every GPU command through `chg run -g=N --timeout ... --` (shared box).

Commands: run pytest from the worktree with
`PYTHONPATH=/home/mgoin/code/vllm-muse-sd` and `/home/mgoin/code/vllm/.venv`
(the worktree uses the main checkout's precompiled `.so`s via symlinks).

## Harness (this directory)

- `run_microbench.sh` + `muse_bench.py`: `vllm bench forward` on a dummy-weight model cut
  to `LAYERS` layers at TP4 (ngram k=7 verify shapes, `--exclude-drafter`), routing
  fixture (uniform/pool) installed on Muse's `router`, post-load setup Muse needs.
  Per-layer = (T8 - T4) / 4 via `per_layer.py`. `EXTRA='--additional-config
  {"decode_backend":"latency_specialized"}'` runs the executor; its "graph vs eager
  rel rms" compares executor (graph) vs stock (eager) on the same batch.
  Caveat: the executor's fused router bypasses the fixture; `muse_bench.py` disables it.
- `timeline.py`: per-kernel timeline of a profiled case (`PROFILE=dir`).
- `serve_bench.sh <layouts>`: real-weight server per layout (`tp4`, `tep4`, `tp4sd`),
  `vllm bench serve` at c=1/4/16 on `~/nightly-test/mixed64.jsonl` + GSM8K;
  `summarize_serve.py` prints step time / AL from the metrics.
- `profile_serve.sh` + `analyze_step_profile.py`: torch profile of steady decode steps on
  real weights, split into target forward vs the rest.
- `SNAP=<dir>`: run from a frozen rsync copy of `vllm/` (`rsync -a --exclude __pycache__
  vllm <dir>/`) so concurrent edits don't leak into a run; import-check it first.
- Temporary A/B env toggles in `runtime/decode.py` (remove before a PR):
  `MUSE_SD_QKV`, `MUSE_SD_PREFETCH`, `MUSE_SD_ROUTER`, `MUSE_SD_EXPERTS`,
  `MUSE_SD_TAIL`, `MUSE_SD_BND` (`ar,pool,unpool`), all default on.

## Pitfalls learned

- The dummy loader skips `MuseSpark1xForCausalLM.load_weights`, so residual-gate
  alpha/beta are never materialized and boundaries fall back to ~25 unfused kernels;
  the harness and tests call the post-load hooks.
- Never H2D-copy from non-pinned memory inside a call (calls happen during graph
  capture with fresh graph-pool tensors); build TMA descriptors ahead or by value.
- Never read routing ids (or any producer output) before `griddepcontrol.wait`: the
  router can be the PDL predecessor and graph capture makes event joins programmatic.
  This hung the real server in adaptive-verification profiling replays; the bench
  harness (static ids) never showed it.
- Persistent kernels whose CTAs wait on each other need co-residency; side-stream
  kernels (prefetch, router) only delay them as long as they never wait on a collective.
- Routed-expert L2 prefetch after routing did not help (routing finishes too late);
  removed.
- Stock FlashInfer autotune "No tuned config covers trtllm_fp4_block_scale_moe" for the
  routed path is real but costs nothing measurable.
- `--gpu-memory-utilization 0.85` avoids a stock startup OOM in the sampler warmup.

## Open issues

1. c=16 (128 rows) is 2-4% slower than stock in every run, although most stages fall back
   to stock-equivalent kernels there. A/B per stage at 128 rows (fused QKV, boundary,
   router) or cap `MAX_ROWS` / per-stage gates at 96.
2. Concurrency 4-8 (32-64 rows) misses the experts kernel and tail (<= 16 rows):
   - tail: one-shot multicast of M*8*4096 rows stops paying above ~16 rows; needs an
     owner-scatter / two-shot variant (FlashInfer `tp_moe_finalize_rmsnorm` is the
     reference design).
   - experts: N=32/64 activation tile reloaded per GEMM1 item; gather per-expert token
     tiles.
3. Experts kernel lost ~1.6 µs when it moved the routing read after the PDL wait; it can
   stream weights before the wait again (weights don't depend on routing).
4. Drafter still ~1.0 ms/step (~2x its weight-read bound): conv `kernel_projection` +
   gate_up/down dominate; more fusion possible. Logit scale vs soft-cap order in
   DFlash2 candidate scoring differs from the target's `LogitsProcessor` — confirm
   against training.
5. Target lm_head + logits gather: an NCCL ring all-gather (~89 µs/step) remained in the
   step profile outside the drafter; route through the custom all-gather.
6. Muse's FP32 router input (32 < rows <= 128) runs a TF32 GEMM in stock (`F.linear`),
   which defeats the FP32 input — confirm intent with model owners.
7. Not yet done: B200 (SM100) runs of every kernel, Buildkite wiring of the new tests,
   docs, removal of the debug toggles, TP8.
8. Lint debt (the WIP commit bypassed pre-commit): mypy errors in
   `attention/primitives.py`, `moe/router.py`, `tail/{tail,experts}.py`,
   `runtime/decode.py` (router eps may be None, `LayerPlan.index` shadows
   `tuple.index`, tail parity Optional), `qwen3_dflash2.py` (quantization literal),
   `tests/v1/spec_decode/test_dflash2.py`, `test_tail_numerics.py`; shellcheck on the
   harness scripts; the new tests are not wired into Buildkite. Run
   `pre-commit run --files $(git diff --name-only origin/main)` to see them.

## Plan (prioritized)

1. Close the c=16 regression (issue 1) — cheap, protects the high-concurrency case.
2. Extend experts + tail to 32-64 rows (issue 2) — the c=4/8 win.
3. Experts weight streaming before the PDL wait (issue 3), and the tail's remaining
   finalize latency (target < 8 µs/layer at 8 rows, currently ~15).
4. MoE front: router fused into the pooler publishing GEMM (BF16x3 router rows sharded
   64 per rank, logits all-gathered in the same mailbox) + one consumer for top-k +
   pre-expert norm + NVFP4 quant — removes the aux-stream router and a kernel boundary.
5. Drafter: remaining GEMMs and per-layer fusions (issue 4), logits gather (issue 5).
6. Clean-up for PRs: split into (a) drafter changes (stock path, separate PRs: fc
   sharding + custom all-gather, fused candidate head, FP8 preselection flag, split-K
   GEMM via a shared layer), (b) the Muse executor; move the split-K CuTe GEMM out of
   `kimi_k3` into a shared module; remove toggles; docs; CI wiring; B200 validation.
7. TP8 (and TEP8 with replicated experts, per the roofline study) after TP4 is solid.
