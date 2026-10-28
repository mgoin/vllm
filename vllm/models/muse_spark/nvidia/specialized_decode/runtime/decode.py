# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor.

``SpecializedDecode`` runs the decoder layers of ``MuseSpark1xModel`` for the
steps captured into FULL CUDA graphs of pure decode or verify batches of up to
``MAX_ROWS`` rows. Its model state builds it once the weights are loaded.
``plan`` decides per forward whether it runs; the model's forward then hands
the embedding to ``DecodeStep.run``. Every other forward (prefill, mixed,
eager, PIECEWISE) runs the stock layers, which the executor leaves unchanged.

Verify steps may give each request a different query length (adaptive
verification): only the attention core reads per-request lengths, from the
stock metadata on the device, so the executor works on rows alone.

The executor issues TP collectives, so every decision reads host state that
all TP ranks share: the step's rows and the attention metadata's host fields.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, cast

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xResidualStream,
    norm_then_gated_residual,
)
from vllm.models.muse_spark.nvidia.ops.prepare_trtllm_nvfp4 import (
    norm_quant_trtllm_nvfp4,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from ...model import (
    MuseSpark1xColumnShardedStream,
    MuseSpark1xMoE,
    norm_residual_then_norm,
    o_norm_sigmoid_gate,
)
from .. import MAX_ROWS, DecodeConfig
from .linear import (
    O_PROJ_TACTICS,
    POOLER_TACTICS,
    QKV_TACTICS,
    UNPOOLER_TACTICS,
    DecodeProjection,
    column_shard,
)
from .prefetch import PrefetchController

if TYPE_CHECKING:
    from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadata

    from ...model import MuseSpark1xAttention, MuseSpark1xDecoderLayer, MuseSpark1xModel
    from ..attention.qkv_projection import QkvProjection
    from ..boundary.boundary import Published
    from ..experts.routed_experts import MuseRoutedExperts
    from ..moe.router import MuseRouter
    from ..tail.experts import PublishingRoutedExperts
    from ..tail.tail import MoeTail

DecodeOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]


class DecodeStep(NamedTuple):
    """The host-side gates of one captured decode step, by its rows."""

    rows: int
    executor: "SpecializedDecode"

    def run(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> DecodeOutput:
        """The backbone's output after its embedding."""
        return self.executor.run(self, hidden_states, positions)


# Most rows of the published unpooler; above it the stock all-gather is as fast.
UNPOOLER_MAX_ROWS = 128

# The experts kernel took one 8-column tile width and a 16-pointer signature;
# `tail/experts.py` still subclasses it with the previous shape, so the fused
# tail waits for its rebind. Off, the layers run the plain expert kernel and
# keep the stock finalize and unpooler: correct, just not fused.
TAIL_BOUND = False


@dataclass(slots=True)
class PublishedBranch:
    """A layer's residual stream whose FFN branch (the unpooler's column
    blocks) lives in the all-gather mailbox, for the next boundary to gather."""

    stream: torch.Tensor
    handle: "Published"
    layer: "MuseSpark1xDecoderLayer"


class Boundaries:
    """The executor's collectives fused with the projections that feed them:
    the o_proj publishes into the all-reduce of the attention boundary, and
    the pooler and unpooler publish their column blocks into the stock
    all-gather workspaces, whose consumers run unchanged."""

    def __init__(self, layers: list["MuseSpark1xDecoderLayer"]) -> None:
        from ..boundary.all_reduce_dual_norm import DualNormConstants
        from ..boundary.boundary import (
            O_PROJ_MAX_TOKENS,
            AllGatherPublisher,
            BoundaryAllReduce,
        )

        group = get_tp_group().device_group
        self.o_proj_max_rows = O_PROJ_MAX_TOKENS
        constants = {
            DualNormConstants(
                float(layer.post_attn_norm.variance_epsilon),
                float(layer.pre_ffn_norm.variance_epsilon),
                float(layer.pre_ffn_norm.fused_gain_center),
                router=router,
            )
            for layer in layers
            for router in (False, True)
        }
        self.attention = BoundaryAllReduce(
            group,
            k=layers[0].self_attn.q_size,
            max_tokens=O_PROJ_MAX_TOKENS,
            constants=sorted(constants, key=repr),
            gate_eps=layers[0].self_attn.rms_norm_eps,
        )
        self.attention.precompile()
        moes = [cast(MuseSpark1xMoE, layer.feed_forward) for layer in layers]
        unpooler = next(
            (
                layer.all_gather_dual_norm
                for layer in layers
                if layer.all_gather_dual_norm
            ),
            None,
        )
        pooler = next(
            (moe.all_gather_norm_quant for moe in moes if moe.all_gather_norm_quant),
            None,
        )
        self.unpooler = (
            None
            if unpooler is None
            else AllGatherPublisher(unpooler, group, k=moes[0].intermediate_size)
        )
        self.pooler = (
            None
            if pooler is None
            else AllGatherPublisher(pooler, group, k=layers[0].hidden_size)
        )
        for publisher in (self.unpooler, self.pooler):
            if publisher is not None:
                publisher.precompile()

    def gather_branch(
        self, state: PublishedBranch, next_norm: "MuseSpark1xRMSNorm"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The published FFN branch's boundary into ``next_norm``:
        ``(next norm output, residual stream)``."""
        assert self.unpooler is not None
        layer = state.layer
        post_norm, connection = layer.post_ffn_norm, layer.add_ffn
        stream, out = self.unpooler.gated_residual_next_rmsnorm(
            state.handle,
            state.stream,
            connection.alpha,
            connection.beta,
            next_norm.weight,
            post_eps=post_norm.variance_epsilon,
            next_eps=next_norm.variance_epsilon,
            next_gain_center=next_norm.fused_gain_center,
            post_weight=post_norm.weight if post_norm.has_weight else None,
            post_gain_center=post_norm.fused_gain_center,
        )
        return out, stream


# Most rows of the fused QKV projection; above it cuBLAS and the stock insert
# are faster (B300).
QKV_FUSED_MAX_ROWS = 192


def fused_qkv_projection(attn: "MuseSpark1xAttention") -> "QkvProjection | None":
    """The fused QKV projection of ``attn``, compiled for every row count it
    serves, or None where the stock insert must run (an FP8 KV cache)."""
    from ..attention.qkv_projection import QkvProjection

    if not attn.fused_qk_norm_rope_kv_insert or attn.attn.kv_cache_dtype not in (
        "auto",
        "bfloat16",
    ):
        return None
    projection = QkvProjection(
        attn.qkv_proj.weight,
        num_heads=attn.num_heads,
        num_kv_heads=attn.num_kv_heads,
        eps=attn.rms_norm_eps,
        rope=attn.use_rope,
    )
    return projection.prepare(range(1, QKV_FUSED_MAX_ROWS + 1), kv_out=False)


# Rows of BF16 router input the fused router serves (faster than the stock
# router GEMM and routing kernel there, on B300).
ROUTER_ROWS = range(8, 129)


def fused_router(moe: MuseSpark1xMoE) -> "MuseRouter | None":
    """The fused router of an NVFP4 MoE layer whose routing runs ahead of the
    experts, compiled for BF16 input."""
    from ..moe.router import MuseRouter

    if not (
        moe.aux_stream_routing and moe.experts.nvfp4_input_global_scale is not None
    ):
        return None
    router = moe.moe_router
    if router.router_output_multiplier is not None or router.route_scale != 1.0:
        return None
    return MuseRouter(
        moe.router.weight,
        router.expert_bias,
        top_k=router.top_k,
        route_scale=router.route_scale,
        eps=router.router_score_division_eps,
    ).prepare(ROUTER_ROWS, dtypes=(torch.bfloat16,))


# Rows of the persistent routed experts: faster than TRT-LLM-gen there; one
# row and 32 rows of uniform routing are slower (B300).
EXPERTS_ROWS = range(2, 17)


def routed_experts(moe: MuseSpark1xMoE) -> "MuseRoutedExperts | None":
    """The persistent routed experts of an NVFP4 MoE layer, which read the
    routing that the aux stream finished."""
    from ...muse_moe_runner import MuseSparkModelOptNvFp4FusedMoE
    from ..experts.routed_experts import MuseRoutedExperts, build_routed_experts

    experts = build_routed_experts(moe.experts, ids_ready=True)
    if isinstance(experts, MuseRoutedExperts):
        return experts
    # BF16 layers keep the stock experts; an NVFP4 layer without the prepared
    # TRT-LLM kernel would silently run a mixed path (or assert in the runner).
    if isinstance(
        moe.experts.routed_experts.quant_method, MuseSparkModelOptNvFp4FusedMoE
    ):
        raise ValueError(
            "decode_backend=latency_specialized needs the resolved FlashInfer "
            f"TRT-LLM NVFP4 MoE backend for every NVFP4 layer: {experts}."
        )
    return None


class LayerPlan(NamedTuple):
    """The executor's kernels over one layer's weights."""

    index: int
    layer: "MuseSpark1xDecoderLayer"
    moe: MuseSpark1xMoE
    qkv_proj: DecodeProjection
    # The QKV projection fused with the QK-norm, RoPE and KV-cache insert.
    qkv_fused: "QkvProjection | None"
    o_proj: DecodeProjection
    # Router GEMM and top-k in one kernel; None for the BF16-expert layers.
    router: "MuseRouter | None"
    # Routed experts in one persistent kernel; None for the BF16-expert layers.
    experts: "MuseRoutedExperts | None"
    # This rank's output columns of the MoE projections.
    pooler: DecodeProjection
    unpooler: DecodeProjection
    # The routed experts publishing into the MoE tail at this layer's parity,
    # for the layers whose FFN branch the tail publishes.
    tail_experts: "PublishingRoutedExperts | None" = None
    tail_parity: int | None = None


def captures_pure_decode(layer_names: list[str]) -> bool:
    """Whether the forward being captured is a FULL graph of a pure decode or
    verify step: not a PIECEWISE capture (Model Runner V2 captures FULL graphs
    with the forward context's mode NONE, and PIECEWISE ones with PIECEWISE),
    and no prefills in any attention group's metadata."""
    context = get_forward_context()
    attn_metadata = context.attn_metadata
    if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE or not isinstance(
        attn_metadata, dict
    ):
        return False
    for name in layer_names:
        metadata = cast("FlashInferMetadata | None", attn_metadata.get(name))
        if metadata is None or metadata.num_prefills > 0:
            return False
    return True


class SpecializedDecode:
    """The decode executor over one backbone's layers."""

    def __init__(self, backbone: "MuseSpark1xModel", config: DecodeConfig) -> None:
        if backbone.start_layer != 0 or backbone.end_layer != len(backbone.layers):
            raise ValueError("decode_backend=latency_specialized needs every layer")
        if backbone.is_sequence_parallel or not backbone.fuse_across_layers:
            raise ValueError(
                "decode_backend=latency_specialized needs the replicated residual "
                "stream fused across layers"
            )
        self.backbone = backbone
        self.config = config
        self.layers = [
            cast("MuseSpark1xDecoderLayer", layer) for layer in backbone.layers
        ]
        for layer in self.layers:
            attn = layer.self_attn
            if not isinstance(layer.feed_forward, MuseSpark1xMoE):
                raise ValueError(
                    "decode_backend=latency_specialized needs every layer to be MoE"
                )
            if not (attn.fuse_o_norm_gate and attn.cute_o_norm_gate):
                raise ValueError(
                    "decode_backend=latency_specialized needs FlashInfer's "
                    "rmsnorm_sigmoid_gate for the attention output gate"
                )
            if attn.attn.backend != AttentionBackendEnum.FLASHINFER:
                raise ValueError(
                    "decode_backend=latency_specialized needs the FLASHINFER "
                    f"attention backend, not {attn.attn.backend.name}."
                )
        rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.plans = []
        for index, layer in enumerate(self.layers):
            moe = cast(MuseSpark1xMoE, layer.feed_forward)
            assert moe.pooler is not None and moe.unpooler is not None
            self.plans.append(
                LayerPlan(
                    index=index,
                    layer=layer,
                    moe=moe,
                    qkv_proj=DecodeProjection(
                        layer.self_attn.qkv_proj.weight, QKV_TACTICS
                    ),
                    qkv_fused=fused_qkv_projection(layer.self_attn),
                    router=fused_router(moe),
                    experts=routed_experts(moe),
                    o_proj=DecodeProjection(
                        layer.self_attn.o_proj.weight, O_PROJ_TACTICS
                    ),
                    pooler=DecodeProjection(
                        column_shard(moe.pooler.weight, rank, tp_size),
                        POOLER_TACTICS,
                    ),
                    unpooler=DecodeProjection(
                        column_shard(moe.unpooler.weight, rank, tp_size),
                        UNPOOLER_TACTICS,
                    ),
                )
            )
        self.boundaries = Boundaries(self.layers)
        self.tail = self._build_tail(backbone.aux_hidden_state_layers)
        self.prefetch = PrefetchController(self.plans)
        # One layer per attention KV-cache group is enough, but the names are
        # few and every one is checked.
        self._layer_names = [layer.self_attn.attn.layer_name for layer in self.layers]
        # FULL graph captures that run the executor.
        self.captured_steps = 0

    def _build_tail(self, aux_layers) -> "MoeTail | None":
        """The fused MoE tail and the publishing experts of the layers it
        serves: NVFP4 layers whose FFN branch goes to the next boundary rather
        than to the draft model's aux hidden states."""
        if self.boundaries.unpooler is None:
            return None
        if not TAIL_BOUND:
            return None
        from ..tail.experts import PublishingRoutedExperts
        from ..tail.tail import MoeTail, layer_parities

        served = [
            i
            for i, plan in enumerate(self.plans)
            if plan.experts is not None and i + 1 not in aux_layers
        ]
        if not served:
            return None
        group = get_tp_group().device_group
        tail = MoeTail(group, publisher=self.boundaries.unpooler, max_tokens=16)
        tail.precompile()
        parities = layer_parities(len(served))
        for i, parity in zip(served, parities):
            experts = self.plans[i].experts
            assert experts is not None
            self.plans[i] = self.plans[i]._replace(
                tail_experts=PublishingRoutedExperts(
                    experts.w13,
                    experts.w13_scale,
                    experts.w2,
                    experts.w2_scale,
                    experts.g1_alphas,
                    experts.g1_scale_c,
                    experts.g2_alphas,
                    target=tail.published_target,
                    tp=tail.tp,
                    parities=(parity,),
                    ids_ready=True,
                ),
                tail_parity=parity,
            )
        return tail

    def rearm(self) -> None:
        """Re-arm the mailboxes and stage state after the memory-profiling
        capture, which runs against a throwaway KV cache.

        ``initialize_kv_cache`` calls this before the real capture, so the
        executor starts from the state of a fresh one. The resets synchronize
        the TP group, and ``initialize_kv_cache`` is collective over it.
        """
        self.boundaries.attention.reset()
        for publisher in (self.boundaries.pooler, self.boundaries.unpooler):
            if publisher is not None:
                publisher.reset()
        if self.tail is not None:
            self.tail.reset()

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        """The step this forward runs, or None to run the stock layers: only
        FULL CUDA graph captures of pure decode steps of up to ``MAX_ROWS``
        rows run the executor."""
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= MAX_ROWS
            or not is_forward_context_available()
            # The eager run before each capture runs the stock layers.
            or not torch.cuda.is_current_stream_capturing()
            or not captures_pure_decode(self._layer_names)
        ):
            return None
        self.captured_steps += 1
        return DecodeStep(rows=rows, executor=self)

    def run(
        self, step: DecodeStep, x: torch.Tensor, positions: torch.Tensor
    ) -> DecodeOutput:
        """The decoder layers and the final norm of the step, on the embedding
        ``x``; returns what ``MuseSpark1xModel.forward`` returns."""
        backbone = self.backbone
        aux_layers = backbone.aux_hidden_state_layers
        aux: list[torch.Tensor] = []
        state: MuseSpark1xResidualStream | torch.Tensor = x.to(backbone.residual_dtype)
        if 0 in aux_layers:
            aux.append(cast(torch.Tensor, state))
        try:
            for idx, plan in enumerate(self.plans):
                state = self._layer(plan, state, positions, idx + 1 not in aux_layers)
                if idx + 1 in aux_layers:
                    if isinstance(state, MuseSpark1xColumnShardedStream):
                        state = state.materialize()
                    aux.append(self._materialize(state))
            output = self._final_norm(state)
        finally:
            # A capture must end joined, also when it raises.
            self.prefetch.join()
        if aux:
            return output, aux
        return output

    def _layer(
        self,
        plan: LayerPlan,
        state: PublishedBranch | MuseSpark1xResidualStream | torch.Tensor,
        positions: torch.Tensor,
        may_publish: bool,
    ) -> PublishedBranch | MuseSpark1xResidualStream:
        """``MuseSpark1xDecoderLayer.forward`` for the replicated stream; with
        ``may_publish``, the FFN branch may stay in the all-gather mailbox."""
        layer, moe = plan.layer, plan.moe
        attn_input, stream = self._attention_input(layer, state)
        attn_output, og = self._attention(plan, attn_input, positions)
        rows = attn_output.shape[0]
        wants_router = moe.wants_fp32_router_input(rows)
        attn = layer.self_attn
        if rows <= self.boundaries.o_proj_max_rows:
            boundary = self.boundaries.attention
            # The boundary applies the output gate, in the GEMM for few rows.
            handle = boundary.o_proj(attn_output, attn.o_proj.weight, gate=og)
            self._prefetch("moe_front", plan, rows)
            post_norm, pre_norm = layer.post_attn_norm, layer.pre_ffn_norm
            ffn_input, stream, router_input = boundary.dual_norm(
                handle,
                stream,
                layer.add_attn.alpha,
                layer.add_attn.beta,
                pre_norm.weight,
                post_eps=post_norm.variance_epsilon,
                next_eps=pre_norm.variance_epsilon,
                next_gain_center=pre_norm.fused_gain_center,
                router=wants_router,
            )
        else:
            assert o_norm_sigmoid_gate is not None
            o_norm_sigmoid_gate(
                attn_output.view(-1, attn.num_heads, attn.head_dim),
                og.view(-1, attn.num_heads, attn.head_dim),
                attn.rms_norm_eps,
                enable_pdl=True,
            )
            attn_out = tensor_model_parallel_all_reduce(plan.o_proj(attn_output))
            self._prefetch("moe_front", plan, rows)
            router_input = None
            if wants_router:
                router_input = torch.empty_like(attn_out, dtype=torch.float32)
            ffn_input, stream = norm_residual_then_norm(
                MuseSpark1xResidualStream(
                    stream=stream,
                    branch=attn_out,
                    post_norm=layer.post_attn_norm,
                    connection=layer.add_attn,
                ),
                layer.pre_ffn_norm,
                out_dtype=layer.default_dtype,
                router_input=router_input,
            )
        local_columns = layer._defers_column_gather(rows)
        publish = (
            may_publish
            and local_columns
            and rows <= UNPOOLER_MAX_ROWS
            and self.boundaries.unpooler is not None
        )
        branch = self._moe(plan, ffn_input, router_input, local_columns, publish)
        self._prefetch("next_layer", plan, rows)
        if isinstance(branch, torch.Tensor):
            if local_columns:
                return MuseSpark1xColumnShardedStream(
                    stream=stream,
                    branch=branch,
                    post_norm=layer.post_ffn_norm,
                    connection=layer.add_ffn,
                    all_gather=layer.all_gather_dual_norm,
                )
            return MuseSpark1xResidualStream(
                stream=stream,
                branch=branch,
                post_norm=layer.post_ffn_norm,
                connection=layer.add_ffn,
            )
        return PublishedBranch(stream=stream, handle=branch, layer=layer)

    def _attention_input(
        self,
        layer: "MuseSpark1xDecoderLayer",
        state: PublishedBranch | MuseSpark1xResidualStream | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The previous layer's FFN boundary and this layer's attention norm:
        ``(attention input, residual stream)``."""
        pre_norm = layer.self_attn.pre_qkv_proj_norm
        if isinstance(state, PublishedBranch):
            return self.boundaries.gather_branch(state, pre_norm)
        if isinstance(state, torch.Tensor):
            return pre_norm(state, out_dtype=layer.default_dtype), state
        return norm_residual_then_norm(state, pre_norm, out_dtype=layer.default_dtype)

    def _attention(
        self,
        plan: LayerPlan,
        attn_input: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``MuseSpark1xAttention.forward`` up to its output gate:
        ``(attention output, output gate)``."""
        attn = plan.layer.self_attn
        rows = attn_input.shape[0]
        fused = self._fused_qkv(plan, attn_input, positions)
        if fused is not None:
            self._prefetch("attention", plan, rows)
            q, og = fused
            attn_output = attn.attn(q, None, None, output_dtype=q.dtype)
        else:
            qkv = plan.qkv_proj(attn_input)
            self._prefetch("attention", plan, rows)
            q, k, v, og = attn._qk_norm_rope_kv_insert(qkv, positions)
            if attn.fused_qk_norm_rope_kv_insert:
                attn_output = attn.attn(q, None, None, output_dtype=qkv.dtype)
            else:
                attn_output = attn.attn(q, k, v)
        assert og is not None
        width = attn.num_heads * attn.head_dim
        return attn_output.reshape(-1, width), og.reshape(-1, width)

    def _fused_qkv(
        self, plan: LayerPlan, attn_input: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The fused QKV projection's ``(q, o_gate)`` with the KV cache written,
        or None to run it unfused."""
        fused = plan.qkv_fused
        rows = attn_input.shape[0]
        if fused is None or rows > QKV_FUSED_MAX_ROWS:
            return None
        attn = plan.layer.self_attn
        _, _, kv_cache, slot_mapping = get_attention_context(attn.attn.layer_name)
        if slot_mapping is None or kv_cache.numel() == 0:
            # No KV cache bound yet (memory profiling).
            return None
        rotary = attn.rotary_emb
        q = attn_input.new_empty(rows, attn.q_size)
        og = attn_input.new_empty(rows, attn.q_size)
        fused(
            attn_input,
            None,
            kv_cache,
            slot_mapping,
            positions,
            None if rotary is None else rotary.cos_sin_cache,
            q_out=q,
            og_out=og,
        )
        return q, og

    def _experts(
        self,
        plan: LayerPlan,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor,
        routing: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """``MuseMoERunner.forward_prepared_nvfp4``, on the persistent routed
        experts where they serve the step."""
        runner = plan.moe.experts
        if (
            plan.experts is None
            or not isinstance(routing, tuple)
            or a1q.shape[0] not in EXPERTS_ROWS
        ):
            return runner.forward_prepared_nvfp4(a1q, a1q_scale, routing)
        topk_ids, topk_weights = routing
        unfinalized = plan.experts(a1q, a1q_scale, topk_ids, topk_weights)
        output = runner._prepared_nvfp4_experts._run_post_gemm2_hook(unfinalized)
        return runner._maybe_reduce_final_output(
            output, None, runner._fused_output_is_reduced
        )

    def _prefetch(self, point: str, plan: LayerPlan, rows: int) -> None:
        if self.prefetch is not None:
            self.prefetch.fork(point, plan.index, rows)

    def _moe(
        self,
        plan: LayerPlan,
        hidden_states: torch.Tensor,
        router_input: torch.Tensor | None,
        local_columns: bool,
        publish: bool,
    ) -> "torch.Tensor | Published":
        """``MuseSpark1xMoE.forward``: the unpooler output, or with
        ``local_columns`` this rank's block of its columns, which ``publish``
        leaves in the all-gather mailbox."""
        moe = plan.moe
        num_tokens = hidden_states.shape[0]
        if not (moe.fused_input_quant and moe._prepares_nvfp4_inputs(num_tokens)):
            # The BF16-expert layers.
            return moe(
                hidden_states,
                router_input=router_input,
                defer_column_gather=local_columns,
            )
        if router_input is None:
            router_input = hidden_states
        norm, pooler = moe.pre_expert_norm, moe.pooler
        assert norm is not None and pooler is not None
        global_scale = moe.experts.nvfp4_input_global_scale

        pooler_publisher = self.boundaries.pooler

        def pool_and_quantize() -> tuple[torch.Tensor, torch.Tensor]:
            if moe.shards_pooler(num_tokens) and pooler_publisher is not None:
                handle = pooler_publisher(hidden_states, plan.pooler.weight)
                return pooler_publisher.rmsnorm_fp4_quantize(
                    handle,
                    norm.weight,
                    global_scale,
                    eps=norm.variance_epsilon,
                    gain_center=norm.fused_gain_center,
                )
            if moe.shards_pooler(num_tokens):
                pooled = plan.pooler(hidden_states)
                if fused := moe._all_gather_norm_quant(pooled, norm, global_scale):
                    return fused
                pooled = sp_all_gather(pooled)
            else:
                pooled = pooler(hidden_states)[0]
            return norm_quant_trtllm_nvfp4(
                pooled,
                num_tokens,
                norm.weight,
                norm.variance_epsilon,
                norm.fused_gain_center,
                global_scale,
            )

        def route() -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if (
                plan.router is not None
                and num_tokens in ROUTER_ROWS
                and router_input.dtype == torch.bfloat16
            ):
                top_k = moe.moe_router.top_k
                ids = router_input.new_empty(num_tokens, top_k, dtype=torch.int32)
                weights = router_input.new_empty(num_tokens, top_k, dtype=torch.float32)
                return plan.router(router_input, ids, weights)
            logits, _ = moe.router(router_input)
            if not moe.aux_stream_routing:
                return logits
            weights, ids = moe.moe_router._compute_routing(
                hidden_states, logits, torch.int32
            )
            return ids, weights

        (a1q, a1q_scale), routing = maybe_execute_in_parallel(
            pool_and_quantize,
            route,
            moe._router_events[0],
            moe._router_events[1],
            moe._router_aux_stream,
        )
        if (
            publish
            and self.tail is not None
            and plan.tail_experts is not None
            and isinstance(routing, tuple)
            and self.tail.supports(num_tokens)
        ):
            parity = plan.tail_parity
            unfinalized = plan.tail_experts(
                a1q, a1q_scale, routing[0], routing[1], parity=parity
            )
            post_norm = moe.post_expert_norm
            assert post_norm is not None
            return self.tail(
                unfinalized,
                post_norm.weight,
                plan.unpooler.weight,
                eps=post_norm.variance_epsilon,
                gain_center=post_norm.fused_gain_center,
                parity=parity,
                topk_ids=routing[0],
            )
        experts_out = self._experts(plan, a1q, a1q_scale, routing)
        if publish:
            assert self.boundaries.unpooler is not None
            return self.boundaries.unpooler(experts_out, plan.unpooler.weight)
        if local_columns:
            return plan.unpooler(experts_out)
        assert moe.unpooler is not None
        return moe._project(moe.unpooler, experts_out, moe.shards_unpooler(num_tokens))

    def _materialize(
        self, state: MuseSpark1xResidualStream | torch.Tensor
    ) -> torch.Tensor:
        """The residual stream after a layer, for the drafter's aux inputs."""
        if isinstance(state, torch.Tensor):
            return state
        return norm_then_gated_residual(state)

    def _final_norm(
        self, state: PublishedBranch | MuseSpark1xResidualStream | torch.Tensor
    ) -> torch.Tensor:
        backbone = self.backbone
        if isinstance(state, PublishedBranch):
            return self.boundaries.gather_branch(state, backbone.norm)[0]
        if isinstance(state, torch.Tensor):
            return backbone.norm(state, out_dtype=backbone.default_dtype)
        output, _ = norm_residual_then_norm(
            state, backbone.norm, out_dtype=backbone.default_dtype
        )
        return output
