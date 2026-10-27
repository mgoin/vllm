# Muse Spark specialized decode — critique and plan (round 2)

Base: branch `fork/mgoin/muse-spark-specialized-decode` @ 22ee9766c6, worktree
`/home/mgoin/code/vllm-muse-sd`.

## What exists (from the handoff + code read)

- Opt-in `decode_backend=latency_specialized`, validated at startup with a MIN
  all-reduce, executor built in a model state before profiling/capture,
  `plan()` gates on host state only, stock layers untouched. Model-side hooks
  are small and in the right place (mirrors DSV4.1).
- Kernel set: fused QKV+QK-norm+RoPE+KV-insert (≤192 rows), o_proj+AR+dual-norm
  publishing boundary + Lamport AR (≤128), fused router (8..128), pooler AG
  norm+NVFP4 quant, persistent routed experts (2..16), fused MoE tail (≤16),
  unpooler publisher (≤128), L2 prefetch, split-K CuTe GEMMs.
- Outside the package: DFlash2 drafter optimizations (fc sharding, fused
  candidate head, FP8 preselection, split-K GEMM) + tests + a docs row.

## Measured in this environment (real weights, B300, TP4, DFlash2, mixed64 prompts, 1024 out)

| case | stock step / tok/s / AL | specialized step / tok/s / AL | delta |
| --- | --- | --- | --- |
| c=1 | 9.52 ms / 437 / 4.31 | 8.02 ms / 538 / 4.40 | -15.8 % step, +23 % tok/s |
| c=4 | 13.40 ms / 1108 / 4.08 | 11.20 ms / 1243 / 3.64 | -16.4 % step, +12 % tok/s |
| c=16 | 20.17 ms / 2741 / 3.93 | 20.74 ms / 2773 / 4.05 | +2.8 % step, +1 % tok/s |

Per-layer forward, dummy weights, TP4, no EP (`vllm bench forward`, per_layer.py):

| rows | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| stock us | 156 | 205 | 282 | 346 | 403 | 466 |
| executor us | 117 | 162 | 239 | 318 | 385 | 463 |
| achieved TB/s (stock) | 3.05 | 3.85 | 4.33 | 4.75 | 4.60 | 4.05 |
| roofline us @7.5 TB/s | 63.5 | 105 | 163 | 219 | 247 | 251 |

The layer-level win fades to nothing by 128 rows, yet the executor's forward at 128 rows is itself
3 % faster than stock, so the E2E c=16 step is not losing inside the layers: it is the drafter /
adaptive-verification balance (AL 3.93 -> 4.05 with the step growing by the same 3 %). Confirm with
adaptive verification off on both layouts before chasing it.

## Handoff numbers (previous agent, same box class)

| case | stock step / tok/s | specialized step / tok/s |
| --- | --- | --- |
| c=1 | 10.07 ms / 467 | 7.83 ms / 602 |
| c=4 | 14.19 ms / 1276 | 11.71 ms / 1357 |
| c=16 | 20.12 ms / 3388 | 20.98 ms / 3342 |

## Roofline (per rank, TP4, NVFP4, ctx 8192, ambition: ~7.5 TB/s)

Bytes a step must stream, per rank and layer:

- experts, all 256 touched: 256 × (w13 4096×2048 + w2 4096×1024) × 0.5 B =
  1.61 GB; with ~22 % of experts touched at 8 rows: 0.36 GB
- qkv 18.9 MB, o_proj 8.4 MB, pooler+unpooler 33.6 MB, router 8.4 MB
- KV reads: 46 SWA layers × 4.2 MB + 16 full layers × 16.8 MB = 0.46 GB

| rows | distinct experts/layer | bytes/rank/step | roofline @7.5 TB/s | measured target part | gap |
| --- | --- | --- | --- | --- | --- |
| 8 (c=1) | ~57/256 | ~27 GB | 3.6 ms | ~6.5 ms | 1.8× |
| 32 (c=4) | ~162/256 | ~68 GB | 9.1 ms | ~10.6 ms | 1.16× |
| 128 (c=16) | 256/256 | ~104 GB | 13.9 ms | ~19.9 ms | 1.43× |

The step is a pure weight-streaming problem at every batch size. Compute is
never the limit: even at 128 rows each expert GEMM is M=4.

## Critique

1. **No roofline accounting.** The handoff reports ms but not achieved GB/s, so
   "we are close" was never checkable. It is not close: 1.8× at c=1, 1.43× at
   c=16. Get the bandwidth number into the harness output (per case: bytes,
   ms, GB/s).
2. **The fast path covers the least valuable shapes.** Persistent experts and
   the fused tail stop at 16 rows, so c=4 (32 rows) and c=16 (128 rows) fall
   back to stock kernels — exactly where a 4-GPU deployment spends its time.
   c=16 is a *regression* vs stock (+4 %), which is the worst outcome: the
   executor must never be slower than stock, by gate at worst.
3. **Serialization, not fusion, is the remaining c=1 cost.** Per layer ~15-20
   launches at 8 rows: 62 × (kernel latency + gaps). The fused kernels cut the
   count but the chain is still: norm → qkv → attn → o-norm-gate → AR → dual
   norm → pool/quant → router → experts(GEMM1, swiglu, quant, GEMM2) → tail →
   unpooler → boundary. Each hop is a full-dependency stall at 8 rows.
4. **Scope creep into shared spec-decode code.** qwen3_dflash{,2}.py,
   dflash/speculator.py, config/speculative.py, docs. Real wins (~230 µs/step)
   but they are model-general changes with accuracy surface (FP8 preselection
   is not bit-exact) and belong in their own series with their own eval.
5. **Debug toggles in the step path** (`MUSE_SD_*`, `import os` per layer) and
   hand-written bisect code left in `_layer`/`_moe`/`_experts`. Non-standard and
   a hidden per-step host cost.
6. **Tail/experts protocol is single-shot and multicast-based**, which is why it
   stops at 16 rows; above that the stock finalize wins. Growing it needs an
   owner-scatter or two-shot design (FlashInfer `tp_moe_finalize_rmsnorm`).
7. **Unvalidated surfaces**: B200 (SM100) has never run; TP8 absent; tests not
   wired to CI; mypy/shellcheck debt; docs table missing; no eval on the final
   configuration beyond GSM8K once.

## Plan (priority order)

P0 — instrument and hold the floor
- Add achieved-GB/s accounting (bytes model above) to `summ.py`/`per_layer.py`.
- Make `latency_specialized` never slower than stock: per-stage gates must be
  A/B-verified at 32/64/128 rows, and MAX_ROWS lowered if needed.

P1 — c=16 and c=4 coverage (the throughput cases)
- Persistent experts: rework the 32/64-row tile so the activation tile and the
  per-expert token gather are loaded once per expert, weights streamed once;
  target ≥6.5 TB/s at 128 rows (19.9 → ~15 ms).
- Fused tail: owner-scatter / two-shot finalize + publishing unpooler for
  128 rows; remove the ≤16-row gate.
- Re-A/B every stage at 32/64/128 rows (qkv, boundary, router, pooler) and set
  gates from measurement.

P2 — c=1 latency (the interactive case)
- Collapse the per-layer chain: fuse pooler+quant into the dual-norm consumer,
  router into the pooler publisher, and the experts' GEMM2 epilogue into the
  tail (already designed in the handoff, partly done).
- Drafter: keep the wins, but move them to a separate commit series; only land
  what is bit-exact (no FP8 preselection) unless a model owner signs off.
- lm_head + logits gather (~89 µs) through the custom all-gather.

P3 — cleanliness and coverage
- Remove `MUSE_SD_*` toggles and bisect code; single measured gate table.
- Split the branch: (1) drafter series, (2) executor package, (3) docs/tests.
- Wire tests into Buildkite; mypy/shellcheck clean; B200 run; TP8 study.

## Orchestration

Independent work items, each in its own worktree, merged by the orchestrator:

| id | item | owner |
| --- | --- | --- |
| E1 | experts kernel 32/64/128 rows + gate | agent `experts` |
| E2 | tail 32/64/128 rows (owner-scatter/two-shot) | agent `tail` |
| E3 | c=16/c=4 stage A/B matrix + gate fixes | agent `gates` |
| E4 | c=1 chain collapse (pooler/quant/router/MoE front) | agent `front` |
| E5 | drafter split-out + lm_head/logits gather | agent `drafter` |
| R1 | integration-quality review (vLLM standards) | agent `reviewer` |

GPU discipline: every GPU command goes through `chg run -g=4 --timeout ... --`;
at most one 4-GPU job per agent at a time (the node has 8).
