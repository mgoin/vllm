# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for the specialized decode.

Small-batch decode is bound by HBM bandwidth, and each kernel that starts on weights
not yet in L2 stalls on DRAM. Some kernels leave bandwidth idle, such as the attention
core and the all-reduces. At fixed points of each layer, this module launches a small
kernel (``cutedsl/common/prefetch.py``) on a side stream. It pulls the byte ranges of
the weights the next kernels read into L2, so that they start warm.

Prefetch points (``fork(point, layer, batch)``), each with its batches per TP:

- ``oproj_down``: after the o-projection (B3-8), the MoE front's packed
  router + latent-down rows.
- ``attention_moe``: at TP8 only, after the KDA input projection (B5-8) or the MLA
  core (B8), the same front rows plus the shared expert's weights.
- ``next_attention``: once the routed experts are launched, the next layer's
  attention weights, capped in MiB per batch (``NEXT_ATTENTION``). Not above B8,
  where the HBM-bound experts lose more bandwidth than the attention gains.

The front and shared-expert ranges are on the critical path. They launch with the
highest stream priority and the maximum shared-memory carveout, which needs a driver
launch because CuTe drops both attributes.

Each fork records an event on the main stream that the side stream waits on, so a
prefetch starts only after the work it overlaps. ``join()`` at the end of the step
makes the main stream wait for the side stream, so a captured graph always rejoins
it. This lives in ``runtime/`` rather than ``cutedsl/`` because it is host-side
orchestration across layers: a stream, events, and per-(point, layer, batch) tables
built from the model's layers.
"""

from functools import partial

import cuda.bindings.driver as cuda
import torch

from vllm.models.common.specialized_decode.prefetch import PreparedWeightRanges

from ..cutedsl.common.policy import batch_table

# Per-TP batch -> MiB cap (or cap.carveout) of the next layer's attention weights,
# prefetched once the routed experts finish.
# Not at B>8: the HBM-bound experts lose more bandwidth than the attention gains.
NEXT_ATTENTION = {8: "1:96.100,2-3:96,4-7:64,8:32", 16: "1-8:1024"}


def attention_weights(attention):
    """Next-attention weights in their consumption order."""
    names = (
        *("in_proj_qkvgfab", "fused_qkv_a_g_proj", "fused_qkv_a_proj", "q_proj"),
        *("kv_a_proj_with_mqa", "f_b_proj", "q_b_proj", "g_proj"),
    )
    weights = [getattr(getattr(attention, n, None), "weight", None) for n in names]
    weights += [getattr(attention, "W_UK_T", None), getattr(attention, "W_UV", None)]
    weights = [w for w in weights if isinstance(w, torch.Tensor)]
    return (*weights, attention.o_proj.weight)


class PrefetchController:
    """Prefetch launches per (point, layer, batch) on one side stream."""

    def __init__(self, runtime, layers, packed):
        self.device, tp = runtime.device, runtime.tp
        self.stream = torch.cuda.Stream(device=self.device)
        self.events, self.pending, self.tables = {}, False, {}
        urgent = partial(PreparedWeightRanges, carveout=100, priority=True)
        following = batch_table(NEXT_ATTENTION[tp])
        for position, layer in enumerate(layers):
            if not layer.is_moe_layer:
                continue
            index, moe, attention = layer.layer_idx, layer.mlp, layer.self_attn
            # The front GEMM reads [router shard; down shard] from the packed copy.
            ranges = urgent((packed[index],), grid=8)
            self._add("oproj_down", index, range(3, 9), ranges)
            if tp == 8:
                shared = moe.shared_experts
                shared = shared.gate_up_proj.weight, shared.down_proj.weight
                ranges = urgent((packed[index], *shared), grid=32)
                batches = range(5, 9) if hasattr(attention, "in_proj_qkvgfab") else (8,)
                self._add("attention_moe", index, batches, ranges)
            if position + 1 < len(layers):
                weights = attention_weights(layers[position + 1].self_attn)
                self._add_capped("next_attention", index, weights, following)

    def _add_capped(self, point, layer, weights, table):
        groups: dict[tuple, list[int]] = {}
        for batch, spec in table.items():
            cap, carveout = spec if isinstance(spec, tuple) else (spec, None)
            if cap:
                groups.setdefault((cap, carveout), []).append(batch)
        for (cap, carveout), batches in groups.items():
            ranges = PreparedWeightRanges(
                weights, byte_cap=cap * 1024**2, grid=32, carveout=carveout
            )
            self._add(point, layer, batches, ranges)

    def _add(self, point, layer, batches, ranges):
        key = point, layer
        if key not in self.events:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.events[key] = event
        for batch in batches:
            self.tables[point, layer, batch] = ranges

    def fork(self, point, layer, batch):
        ranges = self.tables.get((point, layer, batch))
        if ranges is None:
            return False
        event = self.events[point, layer]
        event.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(event)
        self.pending = True
        ranges.launch(cuda.CUstream(self.stream.cuda_stream))
        return True

    def join(self):
        if self.pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.pending = False
