# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the Muse Spark decode boundaries' publishing GEMMs.

``BoundaryAllReduce`` replaces ``o_proj`` + ``tensor_model_parallel_all_reduce``
+ ``norm_residual_then_norm`` at the attention boundary::

    handle = boundary.o_proj(attn_output, o_proj.weight)
    out, new_stream, router = boundary.dual_norm(handle, stream, alpha, beta, ...)

``AllGatherPublisher`` replaces the column-sharded pooler / unpooler GEMM and
the publish kernel of an existing FlashInfer ``MNNVLCuteDSLAllGatherDualNorm``
or ``MNNVLCuteDSLAllGatherNormQuant`` workspace; its consumers run unchanged.

The GEMM publishes into the mailbox and the consumer must come next on the
same stream, on every TP rank with the same M (``o_proj`` and ``dual_norm`` are
collective). Construction is collective over the TP group. Kernels compile for
every M up to ``max_tokens`` in ``precompile``, which must run before CUDA graph
capture; calls are pointer-stable and launch nothing else.

On B300 at TP4 the o_proj boundary beats the stock chain up to
``O_PROJ_MAX_TOKENS`` rows; above, the stock chain is faster (the GEMM is bound
by each SM's L2 ingress and the all-reduce by NVLink), so callers route larger
steps there.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import cuda.bindings.driver as cuda
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64

from .all_reduce_dual_norm import (
    HIDDEN,
    LAMPORT_GENERATIONS,
    AllReduceDualNormKernel,
    DualNormConstants,
    compile_dual_norm,
)
from .publishing_gemm import (
    GemmTactic,
    PublishingGemmKernel,
    compile_gemm,
    gmem_ptr,
    max_stages,
)

# Rows from which the two-shot all-reduce beats the one-shot one (B300, TP4).
TWO_SHOT_TOKENS = 24
# Largest step the o_proj boundary beats the stock chain at (B300, TP4).
O_PROJ_MAX_TOKENS = 128
# Largest step whose o-norm gate the o_proj GEMM computes in its activation
# load; every CTA gates its whole K slice, so larger steps gate separately.
GATED_TOKENS = 8
# Consumer clusters of the two-shot all-reduce: one wave on 148 SMs.
TWO_SHOT_CLUSTERS = 64


def _gate_lanes(m: int) -> int:
    """FlashInfer's rmsnorm_sigmoid_gate splits a head over 32 lanes up to 256
    head rows (8 tokens of 32 heads), else over 8: the lane split decides its
    reduction tree, which the gated GEMM reproduces."""
    return 32 if m * 32 <= 256 else 8


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


# Measured on B300 (cold weights, CUDA graphs): per (N, K), the largest M of a
# bucket and its (bm, bn, split_k, bk, cf). Small M streams the weight from HBM
# with ~128 CTAs; from M ~ 64 each SM's L2 ingress bounds the tile choice.
TACTICS: dict[tuple[int, int], list[tuple[int, ...]]] = {
    # o_proj
    (8192, 2048): [
        (8, 64, 8, 1, 128, 1),
        (16, 64, 16, 1, 128, 1),
        (32, 64, 32, 1, 128, 2),
        (48, 64, 48, 1, 128, 1),
        (64, 64, 64, 1, 128, 2),
        (96, 64, 96, 1, 64, 2),
        (128, 128, 128, 2, 64, 2),
        (192, 128, 96, 1, 64, 1),
        (256, 128, 128, 1, 64, 2),
    ],
    # unpooler shard
    (2048, 4096): [
        (16, 128, 16, 4, 64, 1),
        (32, 128, 32, 4, 64, 1),
        (48, 64, 48, 4, 128, 1),
        (64, 64, 64, 4, 64, 1),
        (96, 64, 96, 4, 64, 1),
        (128, 64, 64, 2, 64, 1),
        (192, 64, 96, 2, 64, 1),
        (256, 128, 128, 4, 64, 1),
    ],
    # pooler shard
    (1024, 8192): [
        (8, 64, 8, 4, 128, 1),
        (16, 128, 16, 8, 64, 1),
        (32, 64, 32, 4, 64, 1),
        (48, 128, 48, 8, 64, 1),
        (64, 128, 64, 8, 64, 1),
        (96, 64, 48, 4, 128, 1),
        (128, 64, 64, 4, 64, 1),
        (192, 64, 96, 4, 64, 1),
        (256, 64, 128, 4, 64, 1),
    ],
}


def _tactic(n: int, k: int, bm: int, bn: int, split_k: int, bk: int, cf: int):
    stages = min(max_stages(bm, bn, bk), k // bk // split_k)
    return GemmTactic(bm, bn, split_k, stages, bk, cf)


def default_tactic(n: int, k: int, m: int) -> GemmTactic:
    """The measured tactic of ``m``'s bucket, else one wave of at most 128
    CTAs (split K before larger tiles: the weight streams from HBM)."""
    for max_m, *config in TACTICS.get((n, k), []):
        if m <= max_m:
            return _tactic(n, k, *config)
    bn = 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    bm = 64 if bn <= 64 else 128
    tiles = n // bm * -(-m // bn)
    split_k = 1
    while (
        split_k < 8
        and tiles * split_k * 2 <= 148
        and k % (64 * split_k * 2) == 0
        and bm // (split_k * 2) >= 8
    ):
        split_k *= 2
    bk = 128 if k // split_k >= 1024 and bn <= 32 else 64
    return _tactic(n, k, bm, bn, split_k, bk, 1)


@dataclass(frozen=True)
class PublishTarget:
    """Where a publishing GEMM stores C: FlashInfer's LL mailbox layout
    ``[3, tp, capacity, width]`` (BF16, multicast address ``address``) at
    columns ``offset + n``, and its ``[next, active]`` int32 stage state. With
    ``scatter``, ``address`` is a device table of the ranks' mailboxes and the
    ``width``-column blocks of C go to their owner ranks."""

    address: int
    stage_state: int
    tp: int
    rank: int
    capacity: int
    width: int
    offset: int = 0
    scatter: bool = False


class PublishingGemm:
    """``x @ weight.T`` for one (N, K) whose epilogue publishes into ``target``
    (and with ``store_c`` also stores C), one compiled kernel per tactic."""

    def __init__(
        self,
        n: int,
        k: int,
        target: PublishTarget | None,
        *,
        max_tokens: int,
        store_c: bool = False,
        tactics: dict[int, GemmTactic] | None = None,
        gate_eps: float = 1e-5,
    ) -> None:
        if target is not None and max_tokens > target.capacity:
            raise ValueError("max_tokens exceeds the mailbox capacity")
        self.n, self.k = n, k
        self.target = target
        self.max_tokens = max_tokens
        self.store_c = store_c
        self.gate_eps = gate_eps
        self._tactics = tactics or {}
        self._compiled: dict[tuple[GemmTactic, int], Any] = {}
        self._args = (
            (Int64(target.address), Int64(target.stage_state), Int32(target.rank))
            if target is not None
            else (Int64(0), Int64(0), Int32(0))
        )

    def tactic(self, m: int, gated: bool = False) -> GemmTactic:
        tactic = self._tactics.get(m) or default_tactic(self.n, self.k, m)
        if gated:
            # The gating producer replaces the (multicast) activation TMA and
            # stages the raw activation and gate tiles too.
            stages = max_stages(tactic.bm, tactic.bn, tactic.bk, gated=True)
            tactic = dataclasses.replace(
                tactic, cf=1, stages=min(tactic.stages, stages)
            )
        return tactic

    def precompile(self, gated_tokens: int = 0) -> None:
        for m in range(1, self.max_tokens + 1):
            self._kernel(self.tactic(m), 0)
        for m in range(1, min(gated_tokens, self.max_tokens) + 1):
            self._kernel(self.tactic(m, True), _gate_lanes(m))

    def _kernel(self, tactic: GemmTactic, gate_lanes: int):
        key = (tactic, gate_lanes)
        compiled = self._compiled.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("precompile PublishingGemm before graph capture")
            t = self.target
            if t is None:
                kernel = PublishingGemmKernel(
                    self.n,
                    self.k,
                    tactic,
                    publish=False,
                    store_c=self.store_c,
                    gate_lanes=gate_lanes,
                    gate_eps=self.gate_eps,
                )
            else:
                kernel = PublishingGemmKernel(
                    self.n,
                    self.k,
                    tactic,
                    publish=True,
                    store_c=self.store_c,
                    scatter=t.scatter,
                    tp=t.tp,
                    capacity=t.capacity,
                    width=t.width,
                    offset=t.offset,
                    gate_lanes=gate_lanes,
                    gate_eps=self.gate_eps,
                )
            compiled = self._compiled[key] = compile_gemm(kernel)
        return compiled

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """``x @ weight.T``; with ``gate``, ``x`` is first gated per 64-column
        head as by FlashInfer's ``rmsnorm_sigmoid_gate(x, gate, gate_eps)``
        (``x`` itself is not modified)."""
        m = x.shape[0]
        if not 1 <= m <= self.max_tokens:
            raise ValueError(f"M={m} is outside [1, {self.max_tokens}]")
        if x.shape != (m, self.k) or weight.shape != (self.n, self.k):
            raise ValueError("unexpected operand shapes")
        if gate is not None and gate.shape != x.shape:
            raise ValueError("gate must match x")
        for t in (x, weight) + tuple(v for v in (out, gate) if v is not None):
            if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16:
                raise ValueError("operands must be 16-byte aligned contiguous BF16")
        if self.store_c and (out is None or out.shape != (m, self.n)):
            raise ValueError("store_c needs an [M, N] output")
        gated = gate is not None
        self._kernel(self.tactic(m, gated), _gate_lanes(m) if gated else 0)(
            gmem_ptr(BFloat16, weight.data_ptr()),
            gmem_ptr(BFloat16, x.data_ptr()),
            gmem_ptr(BFloat16, (gate if gate is not None else x).data_ptr()),
            gmem_ptr(BFloat16, out.data_ptr() if out is not None else x.data_ptr()),
            Int32(m),
            *self._args,
            _stream(),
        )
        return out


class LamportMailbox:
    """A sentinel-armed BF16 symmetric buffer of ``shape`` with its multicast
    address and a device table of every rank's address; construction is
    collective over ``group``."""

    def __init__(self, group: dist.ProcessGroup, shape: tuple[int, ...], device):
        self.group = group
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
        handle = symm_mem.rendezvous(self.buffer, group)
        self.multicast = int(handle.multicast_ptr or 0)
        if not self.multicast or self.multicast % 16:
            raise RuntimeError("NVLink multicast mapping is unavailable")
        self.peers = torch.tensor(
            list(handle.buffer_ptrs), dtype=torch.int64, device=device
        )
        self.arm()

    def arm(self) -> None:
        self.buffer.view(torch.int16).fill_(-32768)


@dataclass(frozen=True)
class Published:
    """A GEMM output that lives only in a mailbox, awaiting its consumer."""

    owner: object
    m: int


class BoundaryAllReduce:
    """``o_proj`` (row-parallel, BF16) all-reduced into Muse Spark's
    post-attention boundary; see the module docstring.

    Up to ``two_shot_tokens - 1`` rows the all-reduce is one-shot (every rank
    multicasts its whole partial); from ``two_shot_tokens`` rows on it is
    two-shot (each rank sends each owner its column block, the owner reduces it
    and multicasts the sum), which moves about half the NVLink bytes.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        k: int,
        max_tokens: int,
        constants: list[DualNormConstants],
        device=None,
        two_shot_tokens: int = TWO_SHOT_TOKENS,
        gated_tokens: int = GATED_TOKENS,
        gate_eps: float = 1e-5,
        tactics: dict[int, GemmTactic] | None = None,
    ) -> None:
        device = torch.device(device or torch.accelerator.current_device_index())
        self.gated_tokens = gated_tokens
        self.gate_eps = gate_eps
        self.group = group
        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.max_tokens = max_tokens
        self.two_shot_tokens = two_shot_tokens
        chunk = HIDDEN // self.tp
        shape = (LAMPORT_GENERATIONS, self.tp, max_tokens)
        self.mailbox = LamportMailbox(group, (*shape, HIDDEN), device)
        self.stage_state = torch.zeros(2, dtype=torch.int32, device=device)
        target = PublishTarget(
            self.mailbox.multicast,
            self.stage_state.data_ptr(),
            self.tp,
            self.rank,
            max_tokens,
            HIDDEN,
        )
        self.gemm = PublishingGemm(
            HIDDEN, k, target, max_tokens=max_tokens, tactics=tactics, gate_eps=gate_eps
        )
        self.scatter: LamportMailbox | None = None
        self.gather: LamportMailbox | None = None
        self.scatter_gemm: PublishingGemm | None = None
        if two_shot_tokens <= max_tokens:
            self.scatter = LamportMailbox(group, (*shape, chunk), device)
            self.gather = LamportMailbox(
                group, (LAMPORT_GENERATIONS, max_tokens, HIDDEN), device
            )
            self.scatter_gemm = PublishingGemm(
                HIDDEN,
                k,
                PublishTarget(
                    self.scatter.peers.data_ptr(),
                    self.stage_state.data_ptr(),
                    self.tp,
                    self.rank,
                    max_tokens,
                    chunk,
                    scatter=True,
                ),
                max_tokens=max_tokens,
                tactics=tactics,
                gate_eps=gate_eps,
            )
        self._constants = constants
        self._consumers: dict[tuple[DualNormConstants, bool, int], Any] = {}
        for m in range(1, max_tokens + 1):
            for c in constants:
                self._consumer(c, m)
        torch.accelerator.synchronize()
        dist.barrier(group=group)
        self._pending: Published | None = None

    @staticmethod
    def _tokens_per_cluster(m: int) -> int:
        # One wave of consumer clusters: see all_reduce_dual_norm.
        return -(-m // TWO_SHOT_CLUSTERS)

    def _consumer(self, constants: DualNormConstants, m: int):
        two_shot = m >= self.two_shot_tokens
        tokens = self._tokens_per_cluster(m) if two_shot else 1
        key = (constants, two_shot, tokens)
        consumer = self._consumers.get(key)
        if consumer is None:
            consumer = self._consumers[key] = compile_dual_norm(
                AllReduceDualNormKernel(
                    tp=self.tp,
                    capacity=self.max_tokens,
                    constants=constants,
                    two_shot=two_shot,
                    tokens=tokens,
                )
            )
        return consumer

    def precompile(self) -> None:
        self.gemm.precompile(self.gated_tokens)
        if self.scatter_gemm is not None:
            self.scatter_gemm.precompile(self.gated_tokens)

    def reset(self) -> None:
        """Drop a publish an aborted step left unconsumed and re-arm the
        mailboxes (collective)."""
        self._pending = None
        for mailbox in (self.mailbox, self.scatter, self.gather):
            if mailbox is not None:
                mailbox.arm()
        self.stage_state.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)

    def o_proj(
        self,
        attn_output: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> Published:
        """Publish this rank's ``attn_output @ weight.T`` partial.

        With ``gate`` (BF16, like ``attn_output``), ``attn_output`` is first
        gated as by FlashInfer's ``rmsnorm_sigmoid_gate(attn_output, gate,
        gate_eps)``: inside the GEMM's activation load up to ``gated_tokens``
        rows (``attn_output`` is then left as is), else by that kernel, in
        place.
        """
        if self._pending is not None:
            raise RuntimeError("a published o_proj is still unconsumed")
        m = attn_output.shape[0]
        gemm = self.scatter_gemm if m >= self.two_shot_tokens else self.gemm
        assert gemm is not None
        if gate is not None and m > self.gated_tokens:
            from flashinfer.norm import rmsnorm_sigmoid_gate

            heads = attn_output.shape[1] // 64
            rmsnorm_sigmoid_gate(
                attn_output.view(m, heads, 64),
                gate.view(m, heads, 64),
                self.gate_eps,
                enable_pdl=True,
            )
            gate = None
        gemm(attn_output, weight, gate=gate)
        self._pending = Published(self, m)
        return self._pending

    def dual_norm(
        self,
        handle: Published,
        stream: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_weight: torch.Tensor,
        *,
        post_eps: float,
        next_eps: float,
        next_gain_center: float,
        post_weight: torch.Tensor | None = None,
        post_gain_center: float = 0.0,
        router: bool = False,
        out: torch.Tensor | None = None,
        new_stream: torch.Tensor | None = None,
        router_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """All-reduce the published partial and run the dual norm.

        Args:
            handle: ``o_proj``'s result.
            stream: [M, 8192] FP32 residual stream.
            alpha: [8192] FP32 stream gate.
            beta: [8192] FP32 branch gate.
            pre_weight: [8192] BF16 next-norm weight.
            post_eps: Epsilon of the branch RMSNorm.
            next_eps: Epsilon of the next RMSNorm.
            next_gain_center: Constant added to ``pre_weight``.
            post_weight: Optional [8192] BF16 branch-norm weight.
            post_gain_center: Constant added to ``post_weight``.
            router: Also return the unrounded FP32 next-norm output.
            out: Optional preallocated [M, 8192] BF16 output.
            new_stream: Optional preallocated [M, 8192] FP32 stream output.
            router_input: Optional preallocated [M, 8192] FP32 router output
                (implies ``router``).

            The norm constants must be among those given at construction.

        Returns:
            ``(out, new_stream, router_input or None)``.

        """
        if handle is not self._pending:
            raise RuntimeError("consuming a handle other than the pending publish")
        self._pending = None
        m = handle.m
        router = router or router_input is not None
        constants = DualNormConstants(
            float(post_eps),
            float(next_eps),
            float(next_gain_center),
            post_weight is not None,
            float(post_gain_center) if post_weight is not None else 0.0,
            router,
        )
        if constants not in self._constants:
            raise ValueError(f"{constants} was not compiled at construction")
        consumer = self._consumer(constants, m)
        if stream.shape != (m, HIDDEN) or stream.dtype != torch.float32:
            raise ValueError("stream must be FP32 [M, 8192]")
        device = stream.device
        if out is None:
            out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
        if new_stream is None:
            new_stream = torch.empty((m, HIDDEN), dtype=torch.float32, device=device)
        if router and router_input is None:
            router_input = torch.empty((m, HIDDEN), dtype=torch.float32, device=device)
        for t in (stream, alpha, beta, new_stream) + (
            (router_input,) if router_input is not None else ()
        ):
            if not t.is_contiguous() or t.data_ptr() % 32:
                raise ValueError("FP32 operands must be 32-byte aligned and contiguous")
        two_shot = m >= self.two_shot_tokens
        mailbox = self.scatter if two_shot else self.mailbox
        gather = self.gather if two_shot else self.mailbox
        assert mailbox is not None and gather is not None
        consumer(
            gmem_ptr(BFloat16, mailbox.buffer.data_ptr()),
            gmem_ptr(BFloat16, gather.buffer.data_ptr()),
            Int64(gather.multicast),
            gmem_ptr(Int32, self.stage_state.data_ptr(), 4),
            gmem_ptr(Float32, stream.data_ptr(), 32),
            gmem_ptr(Float32, alpha.data_ptr(), 32),
            gmem_ptr(Float32, beta.data_ptr(), 32),
            gmem_ptr(BFloat16, pre_weight.data_ptr()),
            gmem_ptr(
                BFloat16,
                (post_weight if post_weight is not None else pre_weight).data_ptr(),
            ),
            gmem_ptr(Float32, new_stream.data_ptr(), 32),
            gmem_ptr(BFloat16, out.data_ptr()),
            gmem_ptr(
                Float32,
                (router_input if router_input is not None else new_stream).data_ptr(),
                32,
            ),
            Int32(m),
            Int32(self.rank),
            _stream(),
        )
        return out, new_stream, router_input


class AllGatherPublisher:
    """A column-sharded linear published into a FlashInfer all-gather
    workspace (``MNNVLCuteDSLAllGatherDualNormWorkspace`` or
    ``MNNVLCuteDSLAllGatherNormQuantWorkspace``) in place of its publish
    kernel; the workspace's consumer then runs unchanged."""

    def __init__(
        self,
        workspace: Any,
        group: dist.ProcessGroup,
        *,
        k: int,
        tactics: dict[int, GemmTactic] | None = None,
    ) -> None:
        impl = workspace.impl
        state = impl.state
        self.workspace = workspace
        self.group = group
        self.impl = impl
        self.hidden = impl.hidden_size
        self.hidden_local = impl.hidden_local
        target = PublishTarget(
            int(state.contribution_mailbox.multicast_address),
            state.stage_state.data_ptr(),
            impl.tp_size,
            dist.get_rank(group),
            impl.capacity_m,
            impl.hidden_local,
        )
        self.gemm = PublishingGemm(
            self.hidden_local,
            k,
            target,
            max_tokens=impl.capacity_m,
            tactics=tactics,
        )
        self._pending: Published | None = None

    def precompile(self) -> None:
        self.gemm.precompile()

    def reset(self) -> None:
        """Drop an unconsumed publish and re-zero the workspace's stage state
        (collective)."""
        self._pending = None
        self.impl.state.stage_state.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> Published:
        """Publish this rank's ``x @ weight.T`` column block."""
        if self._pending is not None:
            raise RuntimeError("a published column block is still unconsumed")
        self.gemm(x, weight)
        self._pending = Published(self, x.shape[0])
        return self._pending

    def _take(self, handle: Published) -> int:
        if handle is not self._pending:
            raise RuntimeError("consuming a handle other than the pending publish")
        self._pending = None
        return handle.m

    def gated_residual_next_rmsnorm(
        self,
        handle: Published,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        *,
        post_eps: float,
        next_eps: float,
        next_gain_center: float = 0.0,
        post_weight: torch.Tensor | None = None,
        post_gain_center: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FlashInfer's ``all_gather_gated_residual_next_rmsnorm`` consumer on
        the published block; returns ``(residual_out, ffn_input)``."""
        from flashinfer.comm.mnnvl_cutedsl.runtime import to_cute, to_cute_dynamic
        from flashinfer.comm.mnnvl_cutedsl_ag import _norm_constants

        m = self._take(handle)
        hidden = self.hidden
        residual_out = torch.empty_like(residual)
        ffn_input = torch.empty_like(residual, dtype=torch.bfloat16)
        collective = self.impl.compiled(
            _norm_constants(
                post_eps,
                next_eps,
                next_gain_center,
                post_weight is not None,
                post_gain_center,
            )
        )
        collective(
            self.impl._mailbox,
            to_cute_dynamic(residual.flatten(), 32, divisibility=hidden),
            to_cute(alpha, 32),
            to_cute(beta, 32),
            to_cute(post_weight if post_weight is not None else next_weight, 16),
            to_cute(next_weight, 16),
            to_cute_dynamic(residual_out.flatten(), 32, divisibility=hidden),
            to_cute_dynamic(ffn_input.flatten(), 16, divisibility=hidden),
            self.impl._stage_state,
            Int32(m),
            _stream(),
        )
        return residual_out, ffn_input

    def rmsnorm_fp4_quantize(
        self,
        handle: Published,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        *,
        eps: float,
        gain_center: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FlashInfer's ``all_gather_rmsnorm_fp4_quantize`` consumer on the
        published block; returns ``(packed, scales)``."""
        from flashinfer.comm.mnnvl_cutedsl.kernel_ll.all_gather_norm_quant import (
            _NormQuantConstants,
        )
        from flashinfer.comm.mnnvl_cutedsl.runtime import to_cute, to_cute_dynamic

        m = self._take(handle)
        hidden = self.hidden
        device = weight.device
        packed = torch.empty((m, hidden // 2), dtype=torch.uint8, device=device)
        scales = torch.empty((m, hidden // 16), dtype=torch.uint8, device=device)
        collective = self.impl.compiled(
            _NormQuantConstants(float(eps), float(gain_center))
        )
        collective(
            self.impl._mailbox,
            to_cute(weight, 16),
            to_cute(global_scale.reshape(1), 4),
            to_cute_dynamic(packed.flatten(), 16, divisibility=hidden // 2),
            to_cute_dynamic(scales.flatten(), 4, divisibility=hidden // 16),
            self.impl._stage_state,
            Int32(m),
            _stream(),
        )
        return packed, scales
