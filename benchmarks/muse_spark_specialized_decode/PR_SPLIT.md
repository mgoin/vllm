# Landing this work: the series is built and verified

The split exists as three branches on `mgoin/vllm`, each verified on this box:

| branch | commits | content | verified |
| --- | --- | --- | --- |
| `muse-pr1-shared` | 1 | the shared primitives and the Kimi K3 import move | imports clean |
| `muse-pr2-drafter` | 1 | the DFlash/DFlash2 drafter kernels and the opt-in quantized candidate head | 115 tests pass |
| `muse-pr3-executor` | 1 | the Muse Spark executor, its tests, docs and the nightly CI job | 8 config-gate + 166 kernel/numerics + 2 engine-equivalence tests pass |

Each is a single commit on top of the previous one, so they must land in that
order (PR3 imports what PR1 adds; PR2 rewrites the file whose import PR1 changes).
The `vllm/` tree at the head of `muse-pr3-executor` is byte-identical to the
integration branch.

## 0. Shared primitives (prerequisite, no behaviour change)

`vllm/models/common/specialized_decode/` (primitives, host, splitk, projection,
prefetch), Kimi K3 importing from it with re-export shims only for
`recoverssm/`, Muse and DFlash2 importing from it, and DFlash2's split-K
registry keyed by a per-projection id instead of `weight.data_ptr()`.

Files: `vllm/models/common/specialized_decode/*`,
`vllm/models/kimi_k3/nvidia/specialized_decode/**` (imports only),
`vllm/model_executor/models/qwen3_dflash2.py` (import only).

## 1. DFlash drafter: separate change, separate evidence

`fc` column sharding with the custom all-gather, the fused aux concat, the fused
candidate head, and the padding fix that duplicates open PR #58633 (drop that
hunk if the PR has landed). These affect every DFlash/DFlash2 user, so they need
their own acceptance-length and accuracy table on a DFlash2-supported model.

Files: `vllm/v1/worker/gpu/spec_decode/dflash/{speculator,utils}.py`,
`vllm/model_executor/models/qwen3_dflash{,.py}`, `tests/v1/spec_decode/test_dflash_prepare_inputs.py`.

## 2. DFlash2 candidate preselection: opt-in, off by default

`draft_lm_head_quantization` for DFlash2 (FP8/NVFP4 preselection, rescored
against the unquantized head) plus its docs row and `test_dflash2.py`. Not
bit-exact, so it stays off unless a model owner signs off.

Files: `vllm/model_executor/models/qwen3_dflash2.py`,
`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`,
`vllm/config/speculative.py`, `docs/features/speculative_decoding/README.md`,
`tests/v1/spec_decode/test_dflash2.py`.

## 3. The Muse Spark executor, by subsystem

One commit per directory, each with its kernel tests, so a reviewer can follow
the argument:

1. `attention/` — fused QKV with QK-norm, RoPE and KV insert; no host copy on a
   captured call.
2. `boundary/` — publishing GEMMs (o_proj with the folded gate, pooler,
   unpooler) and the all-reduce + dual-norm consumer.
3. `moe/` — the router GEMM with top-k.
4. `experts/` — the persistent NVFP4 routed experts (one 8-column tile kernel
   for every row count), bitwise equal to TRT-LLM-gen.
5. `tail/` — the MoE tail: the owner-scatter finalize (one schedule, 2-16 rows)
   and the publishing unpooler.
6. `runtime/` — the executor, its gates, the model state, warm-up, the L2
   prefetch.
7. The model hooks: `get_model_state_cls`, `build_specialized_decode`, the
   `plan()` call in `MuseSpark1xForCausalLM.forward`, and `MAX_ROWS` /
   `MAX_EXPERT_ROWS` with the measurements behind them.

Tests: `tests/models/muse_spark/nvidia/specialized_decode/` (config gate,
five one-GPU kernel suites, two four-GPU numerics suites), wired into
`.buildkite/test_areas/models_basic.yaml` as the nightly B200 job.

## 4. Docs and the feature itself

`docs/features/specialized_decode.md` and the package `AGENTS.md`. Ships with
the measured table and the row cap, so the feature is enabled by the same change
that documents what it does not cover.

## Not part of any PR

`benchmarks/muse_spark_specialized_decode/` (dev harness), `tools/setup_worktree.sh`
and every `/home/mgoin`, `/models/mgoin`, `chg`, `dgx-b300-1` path: machine-specific,
keep them out of the branch or in a separate `benchmarks/` change if the team
wants the harness.
