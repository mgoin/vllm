# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch BF16 projections of the executor.

cuBLAS runs the decode-sized projections as a split-K GEMM plus a separate
reduction kernel, without PDL. Up to ``max_rows`` rows, ``DecodeProjection``
runs Kimi K3's split-K CuTe DSL GEMM instead (FlashInfer's SM100 split-K core
with a DSMEM reduction and PDL), with a tactic per row count measured on B300
for each Muse Spark TP4 shape; above that it calls ``torch.mm``.
"""

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
    PreparedProjection,
)

# (M tile, N tile, split K, stages) by the largest row count it serves.
Tactic = tuple[int, int, int, int]
TacticTable = tuple[tuple[int, Tactic], ...]

QKV_TACTICS: TacticTable = (
    (16, (64, 16, 2, 8)),
    (32, (64, 32, 2, 8)),
    (64, (64, 64, 2, 5)),
)
O_PROJ_TACTICS: TacticTable = (
    (8, (64, 8, 1, 8)),
    (16, (64, 16, 1, 8)),
    (32, (64, 32, 1, 8)),
    (64, (64, 64, 1, 5)),
)
POOLER_TACTICS: TacticTable = (
    (8, (64, 8, 4, 8)),
    (16, (64, 16, 4, 8)),
    (32, (64, 32, 4, 8)),
)
UNPOOLER_TACTICS: TacticTable = (
    (8, (128, 8, 4, 6)),
    (16, (64, 16, 2, 8)),
    (32, (64, 32, 2, 8)),
    (64, (64, 64, 2, 5)),
)


def tactic_policies(table: TacticTable) -> dict[int, Tactic]:
    """The tactic of every row count up to the table's largest."""
    policies: dict[int, Tactic] = {}
    first = 1
    for max_rows, tactic in table:
        policies.update(dict.fromkeys(range(first, max_rows + 1), tactic))
        first = max_rows + 1
    return policies


class DecodeProjection:
    """``x @ weight.T`` for decode-sized ``x``; ``weight`` must stay in place."""

    def __init__(self, weight: torch.Tensor, table: TacticTable) -> None:
        self.weight = weight
        self.max_rows = table[-1][0]
        self.prepared = PreparedProjection(weight, tactic_policies(table)).prepare()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        if rows > self.max_rows:
            return torch.mm(x, self.weight.t())
        out = x.new_empty(rows, self.weight.shape[0])
        return self.prepared.run(x.contiguous(), out)


def column_shard(weight: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """This rank's contiguous block of ``weight``'s output rows."""
    shard = weight.shape[0] // world_size
    return weight.narrow(0, rank * shard, shard)
