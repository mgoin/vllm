# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility re-exports for Kimi K3 code outside ``specialized_decode``.

The helpers live in ``vllm/models/common/specialized_decode/host.py``; modules
under ``nvidia/specialized_decode`` import them from there.
"""

from vllm.models.common.specialized_decode.host import (
    _checked,
    architecture,
    check_residency,
    dynamic_fake,
    dynamic_view,
    fake,
    layout,
    prepare_shared_carveout,
    resolve_kernel,
    stream,
    tensor_maps,
    to_cute,
)

__all__ = [
    "_checked",
    "architecture",
    "check_residency",
    "dynamic_fake",
    "dynamic_view",
    "fake",
    "layout",
    "prepare_shared_carveout",
    "resolve_kernel",
    "stream",
    "tensor_maps",
    "to_cute",
]
