# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the Muse Spark small-batch MoE tail.

``MoeTail`` replaces, for decode steps of up to ``max_tokens`` rows, the
post-expert finalize (``MusePostExpertFinalize`` / FlashInfer's
``tp_moe_finalize_rmsnorm``), the column-sharded unpooler GEMM and the publish
kernel of the next boundary's all-gather::

    handle = tail(unfinalized, norm_weight, unpooler_shard, eps=..., gain_center=...)
    # the next boundary, the publisher's unchanged consumer:
    new_stream, ffn_input = publisher.gated_residual_next_rmsnorm(handle, ...)

Two PDL-chained kernels: the owner-scatter finalize of ``owned.py`` into a
pointer-stable ``[max_tokens, 4096]`` buffer, then the boundary package's
publishing GEMM, which streams the unpooler shard before its grid wait and
multicasts ``e @ unpooler_shard.T`` into the publisher's FlashInfer
all-gather mailbox.

The finalize scatters this rank's partial route rows to the token's owner, the
owner finalizes the tokens it owns and multicasts their rows, and every rank
copies the rest: each rank receives ``M * 54 KB`` per layer rather than the
``M * 192 KB`` of a multicast-to-everyone schedule, which is what lets the tail
pay from 8 to 64 rows. ``MAX_TOKENS`` is the routed experts' row range, so it
is the largest step whose GEMM2 rows the experts can hand over; larger steps run
the stock finalize and unpooler.

Construction is collective over the TP group; ``precompile`` must run before
CUDA graph capture. Every rank must call with the same rows; calls launch
nothing but the two kernels and allocate nothing.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

from .. import MAX_EXPERT_ROWS
from ..boundary.boundary import (
    AllGatherPublisher,
    LamportMailbox,
    Published,
    PublishingGemm,
)
from ..boundary.publishing_gemm import gmem_ptr
from .owned import CLUSTER, GENERATIONS, GRID, HIDDEN, TOP_K, OwnedFinalize

# The routed experts serve this many rows (``EXPERTS_ROWS``), so it is the
# largest step whose GEMM2 rows the tail can take.
MAX_TOKENS = MAX_EXPERT_ROWS


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


class MoeTail:
    """The MoE tail of a TP group; see the module docstring.

    ``publisher`` is the executor's ``AllGatherPublisher`` of the next
    boundary's FlashInfer ``MNNVLCuteDSLAllGatherDualNormWorkspace``;
    ``__call__`` returns its handle for its consumer
    (``gated_residual_next_rmsnorm``). Without one, ``__call__`` returns this
    rank's plain ``[M, n]`` unpooler block (for the stock all-gather) instead.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        publisher: AllGatherPublisher | None = None,
        shard_columns: int = 2048,
        max_tokens: int = MAX_TOKENS,
        device=None,
        cluster: int = CLUSTER,
        grid: int = GRID,
    ) -> None:
        device = torch.device(device or torch.accelerator.current_device_index())
        self.group = group
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.max_tokens = max_tokens
        capacity_tp = -(-max_tokens // self.tp)
        # Hop 1: this rank's partial rows for the tokens its peers own, and the
        # final rows of the tokens it owns.
        self.hop1 = LamportMailbox(
            group, (GENERATIONS, self.tp, capacity_tp * TOP_K, HIDDEN), device
        )
        self.e_box = LamportMailbox(
            group, (GENERATIONS, self.tp, max_tokens, HIDDEN), device
        )
        self.finalize = OwnedFinalize(
            tp=self.tp,
            rank=self.rank,
            capacity=max_tokens,
            hop1=self.hop1.buffer,
            peers=self.hop1.peers,
            e=self.e_box.buffer,
            multicast=self.e_box.multicast,
            cluster=cluster,
            grid=grid,
        )
        self.e = torch.empty(max_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
        self.publisher = publisher
        self.gemm: PublishingGemm | None = None
        if publisher is None:
            self.gemm = PublishingGemm(
                shard_columns, HIDDEN, None, max_tokens=max_tokens, store_c=True
            )
        elif publisher.gemm.k != HIDDEN or max_tokens > publisher.impl.capacity_m:
            raise ValueError("the publisher does not take this tail's rows")
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def precompile(self, weight_dtypes=(torch.float32, torch.bfloat16)) -> None:
        self.finalize.precompile(weight_dtypes)
        (self.publisher or self.gemm).precompile()

    def reset(self) -> None:
        """Re-arm the mailboxes after an aborted step (collective)."""
        self.hop1.arm()
        self.e_box.arm()
        self.finalize.generations.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)

    def supports(self, m: int) -> bool:
        return 1 <= m <= self.max_tokens

    def __call__(
        self,
        unfinalized: UnfinalizedMoEOutput,
        norm_weight: torch.Tensor,
        unpooler_shard: torch.Tensor,
        *,
        eps: float,
        gain_center: float,
    ) -> Published | torch.Tensor:
        """Finalize ``unfinalized`` and publish ``e @ unpooler_shard.T`` into
        the all-gather workspace (the publisher's consumer takes the handle),
        or without a publisher return it."""
        e = self.finalize(
            unfinalized.gemm2_permuted,
            unfinalized.expanded_idx_to_permuted_idx,
            unfinalized.expert_weights,
            norm_weight,
            self.e,
            eps=eps,
            gain_center=gain_center,
        )
        if self.publisher is not None:
            return self.publisher(e, unpooler_shard)
        assert self.gemm is not None
        branch = e.new_empty(e.shape[0], unpooler_shard.shape[0])
        self.gemm(e, unpooler_shard, branch)
        return branch
