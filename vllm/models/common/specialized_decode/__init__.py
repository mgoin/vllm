# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch primitives shared by the NVIDIA specialized-decode executors.

A specialized decode executor runs decode-sized CUDA kernels that the stock
layers do not have: split-K BF16 projections, L2 weight prefetch and inline
PTX/CuTe helpers. This package owns those model-independent pieces so that an
executor under ``vllm/models/<model>/nvidia/specialized_decode/`` never imports
another model's private package.

- ``primitives``: inline PTX, memory, synchronization and math helpers.
- ``host``: host-side helpers for the prepared CuTe providers.
- ``splitk``: the FlashInfer split-K dense BF16 GEMM core.
- ``projection``: the prepared split-K projection with per-row-count tactics.
- ``prefetch``: the L2 weight prefetch kernel and its prepared weight ranges.

Nothing here depends on a model package; behaviour is fixed by the caller's
weights, tactics and row counts.
"""
