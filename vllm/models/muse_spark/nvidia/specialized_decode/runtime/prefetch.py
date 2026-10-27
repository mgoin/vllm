# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for the specialized decode.

Small decode steps are bound by HBM bandwidth, but some kernels leave it idle:
the attention core, the collectives and the norms. At fixed points of each
layer a small kernel on a side stream pulls the byte ranges of the weights the
next kernels read into L2 (Kimi K3's ``PreparedWeightRanges``), so they start
warm:

- ``attention``: after the QKV projection, this layer's o_proj, read once the
  attention core is done;
- ``moe_front``: after the o_proj, this layer's router and pooler shard, read
  after the all-reduce and the FFN norm;
- ``next_layer``: after the unpooler, the leading bytes of the next layer's QKV
  projection, read after the next boundary.

Each fork records an event on the main stream that the side stream waits on,
so a prefetch starts only after the work it overlaps, and ``join`` makes the
main stream wait for the side stream so a captured graph rejoins it.
"""

from functools import partial

import cuda.bindings.driver as cuda
import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.prefetch import (
    PreparedWeightRanges,
)

# Most rows a prefetch runs for: above it the HBM-bound experts dominate.
MAX_PREFETCH_ROWS = 64
# Bytes of the next layer's QKV projection the unpooler boundary prefetches.
NEXT_LAYER_BYTES = 24 * 1024**2


class PrefetchController:
    """Prefetch launches per (point, layer) on one side stream."""

    def __init__(self, plans: list) -> None:
        device = plans[0].qkv_proj.weight.device
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.pending = False
        self.events: dict[tuple[str, int], torch.cuda.Event] = {}
        self.tables: dict[tuple[str, int], PreparedWeightRanges] = {}
        urgent = partial(PreparedWeightRanges, priority=True)
        for index, plan in enumerate(plans):
            self._add("attention", index, urgent((plan.o_proj.weight,), grid=32))
            front = (plan.moe.router.weight, plan.pooler.weight)
            self._add("moe_front", index, urgent(front, grid=32))
            if index + 1 < len(plans):
                ranges = PreparedWeightRanges(
                    (plans[index + 1].qkv_proj.weight,),
                    byte_cap=NEXT_LAYER_BYTES,
                    grid=32,
                )
                self._add("next_layer", index, ranges)

    def _add(self, point: str, layer: int, ranges: PreparedWeightRanges) -> None:
        self.events[point, layer] = torch.cuda.Event()
        self.tables[point, layer] = ranges

    def fork(self, point: str, layer: int, rows: int) -> None:
        ranges = self.tables.get((point, layer))
        if ranges is None or rows > MAX_PREFETCH_ROWS:
            return
        event = self.events[point, layer]
        event.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(event)
        self.pending = True
        ranges.launch(cuda.CUstream(self.stream.cuda_stream))

    def join(self) -> None:
        if self.pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.pending = False
