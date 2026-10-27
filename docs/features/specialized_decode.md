# Specialized decode for Muse Spark

A *specialized decode* is an opt-in decode executor: instead of running the
stock decoder layers, vLLM runs a set of small-batch CUDA kernels fused around
the attention core and the MoE. It is a latency optimization for interactive
serving, not a throughput one. This page covers Muse Spark Open on NVIDIA
Blackwell. DeepSeek-V4.1 and Kimi-K3 have their own packages with the same
`decode_backend` setting and their own deployment requirements.

The executor runs only the decoder layers of a **pure decode or verify step
captured into a FULL CUDA graph** of at most 256 rows. Prefill, mixed
prefill/decode, eager execution and PIECEWISE captures run the stock layers,
which the package does not modify. Speculative decoding of any method
(DFlash, DFlash2, DSpark) is supported, with or without adaptive verification;
the draft model keeps its stock paths.

## Supported models

| Model | Package | Notes |
| --- | --- | --- |
| Muse Spark Open | `vllm/models/muse_spark/nvidia/specialized_decode/` | this page |
| DeepSeek-V4.1 | `vllm/models/deepseek_v41/nvidia/specialized_decode/` | GB300; its own startup checks |
| Kimi-K3 | `vllm/models/kimi_k3/nvidia/specialized_decode/`, `vllm/models/kimi_k3/amd/specialized_decode/` | see its own `README.md` |

## Enabling it

The default, `auto`, keeps the stock decode:

```bash
vllm serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tensor-parallel-size 4 \
    --additional-config '{"decode_backend": "latency_specialized"}'
```

The setting is read once, when the target model is built. It is validated
collectively over the tensor-parallel group: if any rank misses a requirement,
every rank refuses to start and the error lists every missing requirement.
Ranks must agree because the executor issues cross-rank collectives, and a rank
running the stock layers while its peers run the specialized ones would hang.

### Supported deployment

| Requirement | Value |
| --- | --- |
| Model | Muse Spark Open, NVFP4 (`modelopt_fp4`) checkpoint |
| GPU | Blackwell, SM100 family (B200 or B300) |
| Runner | Model Runner V2 (default in vLLM V1) |
| Parallelism | tensor parallel 4; no pipeline, data, expert or context parallelism, no micro-batching |
| Attention | the FlashInfer backend; flashinfer-python `0.7.2.post1` (the version `requirements/cuda.txt` pins) |
| MoE | the FlashInfer TRT-LLM NVFP4 backend (`--moe-backend auto` resolves to it, or name `flashinfer_trtllm`) |
| CUDA graphs | FULL graphs for decode batches (`FULL_AND_PIECEWISE` or `FULL_DECODE_ONLY`) |
| Interconnect | NVLink multicast for the fused collectives (a single node) |
| Disabled | `torch.compile`, sleep mode, LoRA, KV transfer, batch invariance |

For example, the deployment the executor was measured on:

```bash
vllm serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --tensor-parallel-size 4 \
    --speculative-config '{"method": "dflash", "model": "RadixArk/Muse-Spark-Open-DFlash2", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic", "rejection_sample_method": "block", "enable_adaptive_verification": true}' \
    --additional-config '{"decode_backend": "latency_specialized"}'
```

`--enforce-eager` is rejected: with no CUDA graphs the executor would never run
and the model would silently serve on the stock layers. `vllm bench forward`
runs through the external launcher and captures its own FULL graphs with vLLM
CUDA graphs and `torch.compile` off; that combination is the one accepted
no-graphs deployment. It serves no requests, so it is not a serving mode.

## What it does not support

- **Serving without CUDA graphs.** Use FULL decode graphs. An eager server is
  rejected at startup rather than silently running stock.
- **Rows above 256**, or non-uniform (prefill) steps. Those forward passes run
  the stock layers.
- **FP8 (or other non-BF16) KV caches.** The fused QKV projection with the
  KV-cache insert is skipped and the stock projection and cache insert run
  instead; the rest of the executor still applies.
- **Some layers and stages.** BF16-expert layers, layers whose FFN feeds the
  draft model's auxiliary hidden states, decode steps above 128 rows for the
  attention/output boundary and the unpooler, routed experts above 16 rows,
  and the fused MoE tail above 16 rows all fall back to their stock
  equivalents. Every fallback is decided from host state that all TP ranks
  share, so the ranks never disagree.
- **Other attention backends, TP sizes, GPUs, MoE backends or a different
  flashinfer-python** than the table above. Startup fails with the list of
  what is missing.

These configurations are also rejected: sleep mode, LoRA, KV transfer,
batch invariance and `torch.compile` of the model.

## Measured effect

Real weights, B300, TP4, mixed-length chat prompts, 1024 output tokens,
DFlash2 drafting seven tokens. "Step" is the median TPOT
times the acceptance length. Acceptance length swings by about ±0.12 between
server launches with probabilistic drafting, so compare step times rather than
tokens/s across runs.

| Concurrency | Stock step | Specialized step |
| --- | --- | --- |
| 1 | 10.07 ms | 7.83 ms (−22%) |
| 4 | 14.19 ms | 11.71 ms (−17.5%) |
| 16 | 20.12 ms | 20.98 ms (+4%) |

GSM8K is within noise of stock (0.959 stock, 0.958 specialized on 1319
questions at concurrency 32).

The win is at the latency-sensitive batch sizes. As the batch grows the stock
kernels catch up, which is why the executor's per-stage gates hand stages back
rather than losing to the stock path. These numbers are from B300;
B200 (SM100) is accepted by the deployment check but has not been measured.

## Verifying it

The startup log prints one line when the executor is enabled:

```text
Muse Spark specialized decode on, for FULL-graph decode steps of up to 256 rows.
```

`vllm bench serve` against the same model with and without the setting shows
the step-time change above. For accuracy, run the usual evaluation (for
example `tests/evals/gsm8k/gsm8k_eval.py`) with the executor on and off; the
fused kernels are tested bitwise against the stock ops where the arithmetic
allows it, and the measured GSM8K difference is within noise.

The kernels themselves are covered by
`tests/models/muse_spark/nvidia/specialized_decode/`: one-GPU kernel tests
against the stock ops, and four-GPU numerics tests of the cross-rank kernels.
