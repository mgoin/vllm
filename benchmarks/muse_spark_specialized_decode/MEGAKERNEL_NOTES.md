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
KV ~7.4 MB -> **554.6 MB**. At the box's achievable 6 TB/s that is **92.4 us**.

Stock 155.6 us = 3.56 TB/s. Executor 122.4 us = 4.53 TB/s. A perfect layer would
be 1.68x stock; the executor is at 1.27x, so **1.33x remains**, all of it in
per-link efficiency, not in structure.

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

The three small GEMMs are the whole story: they hold 151 MB of the 554 MB and run
at 0.6-3.8 TB/s. That is a concurrency problem at M=8 (too few CTAs, too little
in flight per SM), which is exactly what Kimi's kernels are tuned for — their
o_proj producer runs 112 CTAs of 64 rows with register-resident weight slices and
in-order K reduction, and their router GEMM is weight-stationary with the top-k
consumer in the same kernel.

## Plan

Tune the three kernels per shape rather than restructure the layer:

| item | now | target | expected critical-path win |
| --- | --- | --- | --- |
| publish GEMM tactics at M<=32 | 36.8 us, 1.8 TB/s | >=4.5 TB/s, ~15 us | 8-10 us/layer |
| router tiling at 8-64 rows | 14.1 us, 0.6 TB/s | <=4 us | 5-6 us/layer |
| fused QKV at 8-32 rows | 19.8 us, 3.8 TB/s | >=5.5 TB/s, ~13 us | 3-4 us/layer |

Together ~16-20 us/layer, i.e. 122 -> ~103 us, step 8.18 -> ~7.0 ms at c=1, and
1.40x over stock (from 1.20x). Beyond that the only levers left are the unhideable
cross-rank latency (8 us/layer of all-reduce at c=1, physics) and request-level
parallelism, which does not exist at c=1.
