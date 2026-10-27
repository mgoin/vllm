# Muse Spark specialized decode (NVIDIA)

A decode executor for small decode and verify batches of Muse Spark Open on
Blackwell, enabled with
`--additional-config '{"decode_backend": "latency_specialized"}'` (the
default, `auto`, keeps the stock decode). With it off, the model runs exactly
as without this package. The executor runs the target model; the draft model
(DFlash/DFlash2, DSpark) keeps its stock paths.

## Scope

`specialized_decode_config` (`__init__.py`) reads the setting once, when the
target model is built. Any deployment other than the one below raises at
startup on every TP rank with the missing requirements:

- GPU: Blackwell (SM100 family).
- Runner: Model Runner V2.
- Parallelism: TP=4; no pipeline, data, expert or context parallelism, no
  micro-batching.
- Checkpoint: NVFP4 (modelopt), with the default MoE backend.
- CUDA graphs: FULL graphs for decode batches (or none, for benchmarks such as
  `vllm bench forward` that capture their own).
- No `torch.compile` of the model, no sleep mode, no LoRA, KV transfer or
  batch invariance.
- NVLink multicast.

The ranks' results are combined with a MIN all-reduce over the TP group, so a
rank without NVLink multicast makes every rank raise rather than run a path the
others do not.

Speculative decoding of any method is supported, with or without adaptive
verification: verify steps may give each request a different query length,
which only the attention core reads (from the stock metadata, on the device).
Adaptive verification times its cost curves by replaying the captured graphs,
so the curves describe the executor.

## Structure

The model state (`runtime/model_state.py`) builds the executor
(`runtime/decode.py`) once the weights are loaded, before memory profiling and
CUDA graph capture; every kernel compiles then. For each forward the model asks
the executor for a `DecodeStep`; only FULL CUDA graph captures of pure decode
or verify steps of up to `MAX_ROWS` (256) rows get one, and the model then hands
its embedding to the step. Every other forward runs the stock layers, which the
executor does not modify.

| Package | Contents |
| --- | --- |
| `runtime/` | The executor and its per-step gate, the model state, the low-batch projections, L2 weight prefetch |
| `attention/` | The QKV projection fused with the QK-norm, RoPE and KV-cache insert |
| `boundary/` | Publishing GEMMs (o_proj with the folded output gate, pooler, unpooler) and the all-reduce + dual-norm consumer |
| `moe/` | The router GEMM fused with the top-k selection |
| `experts/` | The persistent NVFP4 routed experts (GEMM1, SwiGLU, quantization, GEMM2) |
| `tail/` | The MoE tail: experts publishing their rows, the one-shot finalize and the publishing unpooler |

## Paths

| Path | Steps |
| --- | --- |
| QKV projection, QK-norm, RoPE and KV-cache insert in one kernel; q and the output gate in their own contiguous buffers | up to 192 rows, BF16 KV cache |
| o_proj published into a one-shot (two-shot from 24 rows) all-reduce whose consumer runs the dual norm and the FP32 router input; the output gate folded into the o_proj | up to 128 rows; gate fold up to 8 |
| Router GEMM and top-8 selection in one kernel on the router's aux stream | 8 to 128 rows of BF16 router input |
| Pooler shard published into the all-gather + pre-expert norm + NVFP4 quantization workspace | all rows |
| Routed experts in one persistent kernel | 2 to 16 rows |
| Experts publishing their rows into the MoE tail, which finalizes and publishes the unpooler shard | up to 16 rows, layers not feeding the draft model's aux hidden states |
| Unpooler shard published into the next boundary's all-gather | up to 128 rows |
| Other small projections on Kimi K3's split-K CuTe DSL GEMM, with a tactic per row count | up to 64 rows (pooler 32), else cuBLAS |
| L2 prefetch on a side stream: o_proj during the attention core, router and pooler during the all-reduce, the next layer's QKV head during the next boundary | up to 64 rows |

## Dispatch

The executor issues cross-rank collectives, so a rank that diverged would hang
rather than fall back. Every per-step gate reads only host state that all TP
ranks share (row counts, graph mode, the attention metadata's host fields);
nothing is read back from the GPU.

## Tests

`tests/models/muse_spark/nvidia/specialized_decode/`:

- `test_attention_kernels.py`, `test_moe_kernels.py`, `test_boundary_kernels.py`,
  `test_expert_kernels.py`, `test_tail_kernels.py` (one GPU): each kernel against
  the stock op it replaces, eagerly and on CUDA graph replay.
- `test_boundary_numerics.py`, `test_tail_numerics.py` (four GPUs): the cross-rank
  kernels, with a late rank and many graph replays.
- `test_layers.py` (four GPUs): a four-layer random-weight model through the
  engine with the DFlash2 draft, `latency_specialized` against `auto`, with and
  without adaptive verification.
