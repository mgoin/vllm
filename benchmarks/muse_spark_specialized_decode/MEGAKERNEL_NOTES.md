# Why a Kimi-style megakernel is the wrong lever here, and what is left

Measured on the current tree (`muse-sd`), 8 rows, TP4, B300, 6-layer profile,
torch-profiler trace of a CUDA-graph replay.

## The GPU is never idle

| | value |
| --- | --- |
| per-layer span (critical path) | 120-128 us |
| union of kernel busy time | >= span (idle = -2 to -3 us per layer) |
| sum of kernel durations | ~212 us |
| overlap factor | 1.7x across 4 streams |

So the layer already overlaps everything it can: Muse runs ~9 kernels per layer
with PDL chaining and mailboxes, and the GPU is saturated on the critical path.
A megakernel's usual prize — reclaiming launch bubbles and barrier gaps — is not
there to take. Kimi K3 needed one because its stock chain was ~25 kernels with
NCCL all-reduce kernels between them; Muse's chain is already fused.

## The real byte budget (corrected)

The checkpoint excludes `self_attn` from NVFP4, so qkv, o_proj, pooler and
unpooler are **BF16**: 2 B per parameter, four times the NVFP4 expert stream for
the same parameter count. Earlier rooflines in this directory undercounted them.

Per layer at 8 rows, per rank: experts 403 MB (57 of 256 experts x 7.08 MB),
qkv 75.5 MB, o_proj 33.5 MB, pooler 16.8 MB, unpooler 16.8 MB, router 8.4 MB,
KV ~7.4 MB -> **554.6 MB**. The box's achievable *cold* stream read, measured with
L2 flushed (`sum` over 512 MB), is **5.31 TB/s** — not the 6.0 assumed earlier —
so the layer's floor is **104.4 us**, plus ~4 us of attention work that is not a
stream and ~8 us of all-reduce latency that cannot be hidden at c=1: a realistic
floor of **~116 us**.

Stock 155.6 us = 3.56 TB/s effective. Executor 122.4 us = 4.53 TB/s. So the
executor is at 87 % of the achievable stream rate and **within ~5 % of the
realistic floor**; the whole 1.33x I hoped for was an artifact of the 6 TB/s
assumption. There is no 1.5-2x left in the kernels at c=1 on TP4.

## Where the remaining 30 us/layer is

Kernel durations per layer (trace, 15 layer instances divided out) and their
bandwidth with the corrected byte counts:

| kernel | us/layer | MB | TB/s | headroom to 6 TB/s |
| --- | --- | --- | --- | --- |
| expert stream | 63.8 | 403 | 6.3 | at the ceiling |
| publish GEMMs (o_proj, pooler, unpooler) | **36.8** | 67 | **1.8** | 3.3x |
| fused QKV | 19.8 | 75.5 | 3.8 | 1.6x |
| attention | 15.6 | ~7.4 | latency-bound | — |
| router | **14.1** | 8.4 | **0.6** | 10x |
| tail finalize | 13.7 | small | latency-bound | — |
| all-reduce + dual norm | 8.4 | small | latency-bound | — |
| all-gather dual norm | 8.0 | 33 | 4.1 | 1.5x |
| all-gather pooler | 5.8 | 16 | 2.8 | 2.1x |
| L2 prefetch (side stream) | 26.5 | — | — | measured neutral (A/B: 122.9 vs 123.1 us/layer at 8 rows, 240.4 vs 240.9 at 32) |

**Do not read the TB/s column as a kernel deficiency.** Those numbers divide the
in-layer *duration* by bytes, and in-layer durations are inflated by bandwidth
sharing with the concurrent expert, prefetch and publish streams. Measured in
isolation (one GPU, cold weights, copies rotated past L2, graph replay) the three
publish GEMMs run at 4.76 TB/s (o_proj, 128 CTAs), 3.85 (pooler, 64 CTAs) and
3.49-3.98 (unpooler) — i.e. 3.5-4.8 of the 5.31 achievable — and their in-layer
*exclusive critical-path* shares are 5.6 + 0.4 + 3.9 = **9.9 us/layer**, not the
34.6 us of duration. An A/B of the fused QKV's K-split (PARTS 4 -> 8, doubling the
CTAs) changed the 8-row layer by 0.2 us, i.e. nothing.

So the small GEMMs are close to their shape limit: with n/64 column tiles they top
out at 64 CTAs (pooler, unpooler) or 128 (o_proj), the split-K exchange costs
0.3-0.6 us per added slice, and deepening the ring by a stage changes nothing.

## Plan

Tune the three kernels per shape rather than restructure the layer:

What is actually left, in order:

| item | measured now | ceiling | realistic win |
| --- | --- | --- | --- |
| unpooler publish, split-K 4 -> 2 at 8 rows | 3.49 TB/s | 3.98 | ~0.4 us/layer |
| all-reduce latency at c=1 | 8 us of critical path | unhideable without a second request | 0 |
| attention + tail + router latency | ~20 us of critical path | ~10 | a few us |
| fused QKV concurrency | 3.8 TB/s in layer | 4.8 isolated | 0 (A/B: 0.2 us) |

Realistic total: ~5-8 us/layer, i.e. 122 -> ~115 us, step 8.18 -> ~7.7 ms, 1.27x
over stock. **Conclusion: at c=1 on TP4 this design is within ~5 % of its floor.**
The 1.5-2x the user asked about is not available from kernels; it comes from
deployment shape (two TP4 replicas: 2x aggregate at c=1) or from reading fewer
bytes (the BF16 attention projections are 151 MB of the 554 MB per layer; FP8
would cut ~75 MB and buy ~11 %, at an accuracy cost that needs an eval).

## Where the 1.5-2x actually is: two replicas, measured

One node, 8x B300, DFlash2, same prompts, 1024 output tokens.

| deployment | c=1 per request | aggregate tok/s | c=16 aggregate |
| --- | --- | --- | --- |
| one TP8 replica | 1.72 ms TPOT, 558.6 tok/s | 558.6 | 3685 |
| two TP4 replicas, executor on | 1.81 / 1.99 ms TPOT | **1020** (525.9 + 494.2) | 3527 (at 8 each) |

Two TP4 replicas give **1.83x** the node's throughput of one TP8 replica at c=1
with the same per-request latency, and are within 4 % at 16 concurrent requests.
With stock TP4 the same comparison is 1.5x (2 x 419 = 838), so the executor is
what makes the two-replica shape the better deployment: it buys 23 % more node
throughput at c=1 than the same shape did before.

The other lever is bytes: the BF16 attention projections (qkv 75.5 MB, o_proj
33.5, pooler 16.8, unpooler 16.8) are 151 MB of the 554 MB a layer streams. FP8
would halve them (-75 MB, ~14 us/layer, ~1.36x at c=1 from stock); NVFP4 would
cut 113 MB (~21 us, ~1.44x). Both are accuracy decisions, not kernel work.

## Landed this round, with the A/B that justified it

- **FP32 router input on the fused kernel** (`[Muse] Router: run the FP32 router
  input on the fused kernel`): at 32 < rows <= 128 the executor used to fall back
  to the stock TF32 GEMM + Triton routing while still paying to produce the FP32
  input. Back-to-back A/B, same session: 8 rows 122.9 vs 123.8 us/layer, 32 rows
  241.9 vs 241.6, 64 rows 328.3 vs 327.7 — neutral, so the change is kept for the
  consistency, not for speed. 19 router tests pass, 6 of them new.
- **Cold-cache publish-GEMM sweep** (`bench_publish.py`) and its result: the eight-row
  tactics in `TACTICS` are already the best of ~50 measured configurations; the
  three kernels run at 4.0-4.7 TB/s isolated, which is the layer average (4.45),
  so there is nothing concentrated to fix there.
- **An A/B of the fused QKV's K-split** (PARTS 4 -> 8, doubling the CTAs): 0.2 us
  at 8 rows, i.e. nothing.
- **An A/B of the L2 prefetch**: 122.9 vs 123.1 us/layer at 8 rows and 240.4 vs
  240.9 at 32 rows — neutral, so it neither helps nor hurts and is a candidate for
  deletion on simplicity grounds, not performance grounds.

## Calibration warning for anyone reusing these numbers

Cross-session variance at 64 rows is ~4 % (314.4 us in one session, 328.0 in
another, same code). Compare inside one `chg` job, never across sessions.
