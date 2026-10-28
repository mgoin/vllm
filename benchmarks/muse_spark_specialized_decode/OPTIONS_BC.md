# Scoping (b) FP8 dense weights, (c) TP8, and a (d) that looks better

All byte counts are per rank, per layer, at 8 rows, TP4, and priced at the box's
measured 5.31 TB/s cold-stream rate. The layer is 554.6 MB / 122.9 us today.

## (b) FP8 on the BF16 dense weights

| weight | MB/layer at TP4 | note |
| --- | --- | --- |
| qkv (8192 x 4608) | 75.5 | excluded from NVFP4 in the checkpoint |
| o_proj (2048 x 8192) | 33.5 | excluded from NVFP4 |
| pooler (8192 x 1024) | 16.8 | the recipe keeps squeeze/expand in full precision **by design** |
| unpooler (4096 x 2048) | 16.8 | same |

So the candidates are qkv + o_proj = **109 MB (20.5 us/layer)**. FP8 halves them:
**-10.3 us/layer = -8.4 %** at c=1 (122.9 -> 112.6, step 8.18 -> 7.5 ms). Adding the
squeeze/expand matrices against the recipe's intent gets to -13.4 us (-11 %);
NVFP4 on all four gets to -20 us (-16 %), which is the 1.36x figure I quoted
earlier — it assumed all four, and two of them are meant to stay BF16.

Cost: neither kernel has an FP8 path. The fused QKV kernel
(`attention/qkv_projection.py`) and the publishing GEMM are BF16-only; the shared
`primitives.mma(..., fp8=True)` exists (from Kimi) but neither Muse kernel uses it.
That means a load-time quantize hook plus an FP8 weight variant of two CuTe
kernels, their bitwise tests rewritten against the quantized reference, and an
accuracy eval (GSM8K + GPQA) because the checkpoint authors deliberately left the
attention projections exact.

Verdict: **8 % at c=1 for two kernel rewrites and a quality risk. Not the best
use of the node.**

## (c) TP8

What blocks it today:

1. `unsupported()` requires `tensor_parallel_size == 4`.
2. The persistent experts kernel hardcodes `INTER = 1024`, which is
   `moe_intermediate / TP`. At TP8 it is 512, and the derived tile counts change
   (`G1_M_TILES` 16 -> 8, `G2_K_TILES` 16 -> 8), as does `check_weights`
   (`(E, 2*INTER, HIDDEN//2)`). That is the bulk of the work: `INTER` has to
   become a per-deployment compile-time constant in a 1 568-line CuTe kernel,
   with the bitwise suite re-run at the new shape.
3. The four tactic tables in `runtime/linear.py` are keyed by TP4 shapes
   (8192x2048, 2048x4096, 1024x8192, ...) and need TP8 entries: qkv 2 304 output
   rows, o_proj 1 024 input columns, pooler 512, unpooler 1 024. Small, but they
   need a fresh sweep per shape.
4. Mailboxes, the Lamport all-reduce, the all-gather publishers and the tail are
   already parameterised by the group size, so they should carry over; that is an
   assumption to test, not a fact.

Payoff, measured: stock TP8 is 7.44 ms / 558.6 tok/s at c=1 against the executor's
8.18 ms / 515 tok/s at TP4. A TP8 executor that keeps the same relative win lands
near 6.2-6.5 ms, i.e. **~25 % lower single-stream step time** — but it consumes
the whole node. Two TP4 replicas with the executor deliver 1 020 tok/s aggregate
at c=1 against one TP8's 558.6, so TP8 is a latency play, not a throughput play:
it is the right choice only if the workload is one stream that must feel fast.

Verdict: **the only way to lower single-request latency materially, at a large
port cost and a 1.5-1.8x loss of node throughput.**

## (d) The drafter, which neither of the above covers

DFlash2 is 9.77 GB of BF16 weights, 2.44 GB per rank at TP4 -> a **0.46 ms**
weight-read floor. Its measured step is **~1.03 ms** (previous agent, this box,
after their drafter work). That is ~0.57 ms of slack in a 8.18 ms c=1 step:
**~7 %**, bit-exact, no accuracy question, and it does not touch the target model.

The drafter still runs its own small GEMMs, its candidate head and its logits
gather; the previous agent measured it at ~2x its weight-read bound and left
notes on what was fused and what was not. The work is kernel fusion inside
DFlash2 (shared spec-decode code, so it needs its own PR), which is exactly the
Kimi-style kernel work that *does* pay here — because the drafter's chain is
serial and has no weight stream to hide behind.

Verdict: **best value per unit of work of the three, and the one I would do.**

## Order I would take

1. (d) drafter fusion: ~7 % at c=1, bit-exact, contained in the drafter.
2. (a) consolidate the branch into the `PR_SPLIT.md` series and re-run the
   acceptance suite as the gate.
3. (b) FP8 dense weights only if an eval shows the attention projections tolerate
   it — 8 % is not worth risking without that.
4. (c) TP8 if and only if single-stream latency becomes the product requirement.
