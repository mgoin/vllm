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

### In-situ serving profile at c=16 (128 rows), same 24 steps, rank 0, AV off

| | stock | executor |
| --- | --- | --- |
| GPU span | 527 ms | 576 ms |
| GPU busy | 589 ms | **737 ms** |
| kernels issued | 37320 | 25967 |

The executor issues 30 % fewer kernels but does **25 % more GPU work** per step. Delta by kernel
(24 steps): its own kernels appear (+161 ms: `PublishingGemm` 4248 calls 58 ms, `Router` 1440 calls
47 ms, `QkvKernel` 1488 calls 37 ms, `AllReduceDualNorm` 1488 calls 19 ms), and the stock kernels they
replace go away (two-shot AR 1776 -> 288 calls, nvjet split-K GEMMs, `dual_norm_kernel`,
`fused_moe_router`, a 128x32 bf16 GEMM). Net: the fused stages are **slower than stock at 128 rows**:
router 32.5 us/call at 128 rows (0.26 TB/s of weights), qkv 25 us, publishing GEMM 13.7 us x 2.86/layer,
plus 31 ms of `bmm_Bfloat16...t128x16` that only the executor runs (the `torch.mm` fallback of
`DecodeProjection` above its 32/64-row tactic tables).

Action: gate each fused stage to the row range where it is measured to win (qkv, router, o_proj
boundary), and extend the split-K tactic tables to 256 rows so nothing falls back to cuBLAS. Then the
experts/tail work has to earn the 32-128-row win on top of parity, not on top of a deficit.

Adaptive verification off, both layouts, same prompts: c=4 stock 14.79 ms vs executor 13.56 ms
(-8.3 %), c=16 stock 21.45 ms vs executor 21.95 ms (+2.3 %). So the c=16 difference is *not* the
adaptive-verification balance (that hypothesis is dead) and it is *not* inside the layers (the
executor's forward at 128 rows measures 3 % faster than stock in `vllm bench forward`). It has to be
the part of the step outside the target forward: drafter, sampling, or host-side step overhead at
128 rows. `profile_serve_conc.sh` + `step_span.py` measure exactly that split in situ.

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

### The 8-row chain, exclusive critical path per layer (front agent, PROFILE replay)

| stage | us/layer | share |
| --- | --- | --- |
| tail experts publish (persistent NVFP4 experts) | 58.2 | 48 % |
| boundary publish GEMMs (o_proj, pooler, unpooler) | 11.4 | within 1.3x of roofline |
| fused QKV (qk-norm + rope + kv insert) | 11.1 | ~10 us roofline |
| router + top-k (side stream) | 8.7 | |
| o_proj all-reduce + dual norm | 8.6 | latency-bound on the slowest rank |
| all-gather unpooler + dual norm (FlashInfer) | 7.4 | |
| tail finalize | 5.4 | |
| L2 prefetch (side stream) | 4.5 | |
| attention core | 4.3 | |
| all-gather pooler + norm + fp4 quant | 0.5 | fully overlapped |

At 16 rows the layer is 164 us and the tail experts share is 61 %. So the chain is not in the
boundary/attention code: at 8 rows 48 % is the expert stream and the remaining ~70 us is spread over
ten stages that are each within ~1.3x of their own roofline. The remaining c=1 levers are the expert
kernel's stream rate, the router's 8.7 us, and the cross-rank wait inside the o_proj boundary.

**Expert stream, measured with ncu (experts agent):** one expert is 7.08 MB — 6.29 MB of E2M1
weights *plus 0.79 MB of E4M3 block scales* — so every roofline that used 6.29 MB understated the
stream by 12 %. At 128 rows ncu counts 1.79 GB read against 1 798 MB computed from the routing: the
reads are exactly once, no per-N-tile re-read, and the only duplication is the 1 % of experts that
hold more than 8 tokens. The kernel moves that at 6.12 TB/s (6.45 under ncu-locked clocks) against
stock's 6.79 TB/s for the same bytes; the residual 11 % is in-flight depth (128 KB per CTA with 8
stages versus stock's 160 KB), proven by the stage sweep 3/4/8 = 5.33/5.70/5.93 TB/s, and request
granularity is *not* the cause (CTA_K=512 is bit-exactly equal at equal in-flight bytes).

**Bandwidth calibration (experts agent, nsys + ncu):** a pure streaming read on this box runs at
5.5-5.8 TB/s (80 % of a 7.66 TB/s peak), and the stock TRT-LLM expert GEMMs reach 5.99 TB/s at 128
rows. The 7.5 TB/s figure I used first was speculative; `roofline.py` now targets 6.0 TB/s. With that
target the executor is at 3.67 TB/s (8 rows), 5.07 (32), 5.00 (64), 4.60 (128) of a 6.0 ceiling.

## Final end-to-end table (medians of three paired server runs, c=1/4/8/16)

| case | stock step | executor step | tok/s | note |
| --- | --- | --- | --- | --- |
| c=1 (8 rows) | 9.80 ms | 8.18 ms (−17 %) | 419 -> 515 (+23 %) | AL 4.17-4.31 |
| c=4 (32 rows) | 14.04 ms | 12.00 ms (−15 %) | 1057 -> 1223 (+16 %) | |
| c=8 (64 rows) | 16.85 ms | 16.44 ms (−2 %) | 1740 -> 1852 (+6 %) | at the cap |
| c=16 (128 rows) | 20.29 ms | 20.29 ms | 2761 -> 2796 | stock layers (above MAX_ROWS) |

Per layer the executor is ahead at every row count it serves, including 128 rows (385.9 against
stock's 397.0), yet at concurrency 16 it is 3.4 % behind end to end in all three pairs. The gap is
therefore outside the target forward and it is the one open performance question; until it is found
`MAX_ROWS` stays 64 so 128-row steps run stock.

## Measurement noise (important)

Two server launches of the *same* configuration differ by up to 3 % on step time at
concurrency 8-16 here, and acceptance length swings by up to 10 % (c=2: AL 4.54 vs 4.11
for stock vs executor on the same prompts). Everything at the 1-3 % level needs repeat
runs. In particular the c=16 "regression" I chased (2.3 % with adaptive verification
off, 2.8 % on) is inside that band, and the stage-off A/B (below) says the executor is
*not* losing at 128 rows in the harness: 3773 us per 8-layer forward against stock's
3868, i.e. 11.8 us/layer better.

## Integration state (experts kernel merged, fused tail parked, MAX_ROWS=64)

Per-layer forward, dummy weights, TP4, no EP, executor vs the stock numbers measured the same way:

| rows | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| stock us | 155.6 | 204.8 | 276.4 | 340.7 | 397.0 | 461.7 |
| executor, old expert kernel + tail | 116.7 | 162.3 | 239.2 | 317.9 | 385.2 | 463.3 |
| executor, new expert kernel, tail parked | 127.5 | 171.0 | 233.7 | 314.4 | 397.6 | 459.4 |
| executor, fused tail (one-shot <=16, owner-scatter above) | 122.4 | 165.1 | 240.4 | 317.9 | 385.9 | 463.4 |

The fused tail is worth ~5 us/layer at 8/16 rows and is not resolvable at 32/64 (my runs put it
3-7 us behind the no-tail path there, the tail agent's AUTOTUNE=1 runs put it 2 us ahead; the two
schedules cost the same below 24 rows, so the end state is one owner-scatter schedule and no one-shot
path at all). At 128 rows the executor is 11 us/layer *ahead* of stock, which is what the stage-off
A/B predicted.

The new expert kernel is a win at every row count it serves and improves 32/64 rows over the old one
(239 -> 234, 318 -> 314). Parking the fused tail costs 11 us/layer at 8 rows, which is the size of the
prize for rebinding it. 128 rows and above run stock: parity by construction.

## Executor kernel profile (8 rows and 128 rows, 8-layer truncated model, TP4, no EP)

Per forward of 8 layers, rank 0 (`vllm bench forward --profile-dir`):

| 8 rows, 1165 us total | us | calls | us/call |
| --- | --- | --- | --- |
| PublishingExpertsKernel (persistent NVFP4 experts) | 428 | 7 | 61 |
| PublishingGemmKernel (o_proj / pooler / unpooler) | 238 | 22 | 10.8 |
| L2Prefetch (side stream) | 173 | 23 | 7.5 |
| lm_head GEMM | 145 | 1 | 145 |
| AllReduceDualNorm (Lamport AR + dual norm) | 128 | 8 | 16.0 |
| QkvKernel (fused qkv + qk-norm + rope + kv insert) | 122 | 8 | 15.2 |
| RouterKernel (side stream) | 113 | 7 | 16.1 |
| attention (trtllm-gen) | 103 | 8 | 12.8 |

| 128 rows, 3783 us total | us | calls | us/call |
| --- | --- | --- | --- |
| expert gemm1+swiglu (stock TRT-LLM, gate is <=16 rows) | 1248 | 7 | 178 |
| expert gemm2 (stock) | 626 | 7 | 89 |
| lm_head GEMM | 598 | 1 | 598 |
| PublishingGemmKernel | 300 | 22 | 13.7 |
| o_proj (torch.mm, above the 64-row tactic) | 300 | 1 | 300 |
| attention | 227 | 8 | 28.4 |
| RouterKernel | 203 | 7 | 29.0 |
| QkvKernel | 202 | 8 | 25.3 |
| AllGatherDualNorm | 111 | 8 | 13.8 |

Read: at 8 rows the dense projections run at 0.8-1.5 TB/s (10.8 us for 8.4 MB in the o_proj publish, 15 us
for 19 MB of qkv), the Lamport all-reduce + dual norm costs 16 us, and the L2 prefetch kernels burn 25 us/layer
of SM time on a side stream. At 128 rows the executor's own kernels are fine, the MoE is still stock
(the <=16-row gate), and the `PublishingGemmKernel` at 13.7 us for 60.9 MB/layer is ~4.5 TB/s.

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
