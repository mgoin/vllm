# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility re-exports for Kimi K3 code outside ``specialized_decode``.

The primitives live in ``vllm/models/common/specialized_decode/primitives.py``;
modules under ``nvidia/specialized_decode`` import them from there.
"""

from vllm.models.common.specialized_decode.primitives import (
    alloc,
    asm,
    at,
    bf16,
    bf16x2,
    copy_g2s,
    fma,
    init_barriers,
    launch,
    load_words,
    prefetch_bytes,
    shared_address,
    sigmoid,
    store_words,
    wait,
    warp_sum,
)

__all__ = [
    "alloc",
    "asm",
    "at",
    "bf16",
    "bf16x2",
    "copy_g2s",
    "fma",
    "init_barriers",
    "launch",
    "load_words",
    "prefetch_bytes",
    "shared_address",
    "sigmoid",
    "store_words",
    "wait",
    "warp_sum",
]
