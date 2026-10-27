# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import weakref
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor, _topk
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    tp_all_gather_rows,
)
from .utils import maybe_prefix

logger = init_logger(__name__)


@triton.jit
def _dflash2_grouped_conv_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    output_ptr,
    x_stride_row,
    delta_stride_row,
    delta_stride_tap,
    base_stride_tap,
    output_stride_row,
    NUM_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    ELEMENT_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0) // triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    col_block = tl.program_id(0) % triton.cdiv(NUM_CHANNELS, ELEMENT_BLOCK)
    channels = col_block * ELEMENT_BLOCK + tl.arange(0, ELEMENT_BLOCK)
    mask = channels < NUM_CHANNELS
    groups = channels // GROUP_SIZE
    position = row % BLOCK_SIZE

    delta_row = delta_ptr + row * delta_stride_row
    x_row = x_ptr + row * x_stride_row
    accumulator = (
        tl.load(base_ptr + channels, mask=mask, other=0.0).to(tl.float32)
        + tl.load(delta_row + groups, mask=mask, other=0.0).to(tl.float32)
    ) * tl.load(x_row + channels, mask=mask, other=0.0).to(tl.float32)

    for tap in tl.static_range(1, TAPS):
        tap_mask = mask & (position >= tap)
        coefficient = tl.load(
            base_ptr + tap * base_stride_tap + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32) + tl.load(
            delta_row + tap * delta_stride_tap + groups,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + (row - tap) * x_stride_row + channels,
            mask=tap_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += coefficient * x

    tl.store(
        output_ptr + row * output_stride_row + channels,
        accumulator,
        mask=mask,
    )


def dflash2_grouped_conv_impl(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """DFlash2 grouped convolution adapted from LightSeek TokenSpeed PR #1399."""
    num_rows, num_channels = x.shape
    output = torch.empty_like(x)
    if num_rows == 0:
        return output

    element_block = 1024 if num_rows >= 128 and num_channels % 1024 == 0 else 512
    grid = (num_rows * triton.cdiv(num_channels, element_block),)
    _dflash2_grouped_conv_kernel[grid](
        x,
        delta,
        base,
        output,
        x.stride(0),
        delta.stride(0),
        delta.stride(1),
        base.stride(0),
        output.stride(0),
        NUM_CHANNELS=num_channels,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        TAPS=base.shape[0],
        ELEMENT_BLOCK=element_block,
        num_warps=4,
    )
    return output


def dflash2_grouped_conv_fake(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="dflash2_grouped_conv",
    op_func=dflash2_grouped_conv_impl,
    fake_impl=dflash2_grouped_conv_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def dflash2_grouped_conv(
    x: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    return torch.ops.vllm.dflash2_grouped_conv(x, delta, base, block_size, group_size)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    if hidden_states.is_cuda:
        return dflash2_grouped_conv(hidden_states, delta, base, block_size, group_size)

    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


@triton.jit
def _rescore_candidates_kernel(
    hidden_ptr,
    weight_ptr,
    ids_ptr,
    out_ptr,
    hidden_stride,
    weight_stride,
    NUM_CANDIDATES: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    row = pid // NUM_CANDIDATES
    token = tl.load(ids_ptr + pid).to(tl.int64)
    accumulator = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, HIDDEN_SIZE, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < HIDDEN_SIZE
        x = tl.load(hidden_ptr + row * hidden_stride + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + token * weight_stride + offsets, mask=mask, other=0.0)
        accumulator += x.to(tl.float32) * w.to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(accumulator))


@triton.jit
def _rank_desc(values, valid, BLOCK: tl.constexpr):
    """Each entry's position in a descending sort; ties go to the lower index.

    NaN ranks as -inf, so every position below the valid count is filled even
    for the garbage rows of capture and padding.
    """
    values = tl.where(values != values, -float("inf"), values)
    index = tl.arange(0, BLOCK)
    other = tl.reshape(values, (1, BLOCK))
    mine = tl.reshape(values, (BLOCK, 1))
    ahead = (other > mine) | ((other == mine) & (index[None, :] < index[:, None]))
    ahead = ahead & tl.reshape(valid, (1, BLOCK))
    return tl.sum(ahead.to(tl.int32), axis=1)


@triton.jit
def _local_top_k_kernel(
    values_ptr,
    ids_ptr,
    packed_ptr,
    vocab_start,
    NUM_CANDIDATES: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Top-k of a row's candidates, packed as fp32 (values, global ids)."""
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    valid = index < NUM_CANDIDATES
    offsets = row * NUM_CANDIDATES + index
    values = tl.load(values_ptr + offsets, mask=valid, other=-float("inf"))
    values = values.to(tl.float32)
    ids = tl.load(ids_ptr + offsets, mask=valid, other=0).to(tl.int64) + vocab_start
    rank = _rank_desc(values, valid, BLOCK)
    keep = valid & (rank < TOP_K)
    out = packed_ptr + row * 2 * TOP_K + rank
    tl.store(out, values, mask=keep)
    tl.store(out + TOP_K, ids.to(tl.float32), mask=keep)


@triton.jit
def _merge_top_k_kernel(
    packed_ptr,
    ids_ptr,
    values_ptr,
    num_rows,
    scale,
    soft_cap,
    TP_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_SOFT_CAP: tl.constexpr,
) -> None:
    """Global top-k from the rank-major gather of every rank's packed top-k,
    with the logits processor's scale and soft cap applied."""
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    valid = index < TP_SIZE * TOP_K
    rank_of_entry = index // TOP_K
    offsets = (rank_of_entry * num_rows + row) * 2 * TOP_K + index % TOP_K
    values = tl.load(packed_ptr + offsets, mask=valid, other=-float("inf"))
    ids = tl.load(packed_ptr + offsets + TOP_K, mask=valid, other=0.0)
    rank = _rank_desc(values, valid, BLOCK)
    keep = valid & (rank < TOP_K)
    values = values * scale
    if HAS_SOFT_CAP:
        # tanh(x) = 1 - 2 / (exp(2x) + 1)
        values = soft_cap * (1.0 - 2.0 / (tl.exp(2.0 * values / soft_cap) + 1.0))
    tl.store(values_ptr + row * TOP_K + rank, values, mask=keep)
    tl.store(ids_ptr + row * TOP_K + rank, ids.to(tl.int64), mask=keep)


class CandidateHead:
    """Vocab-parallel top-k candidates of the drafter's LM head.

    Each rank takes its shard's top-k, packs values and ids into one fp32
    tensor for a single all-gather, and one kernel merges the ranks and
    applies the logits processor's scale and soft cap.

    With ``preselect_quantization``, a quantized copy of the head shard ranks
    the vocabulary instead, and its top ``preselect`` ids are rescored against
    the BF16 rows, so candidate logits stay the BF16 head's. A true top-k id
    ranked below ``preselect`` by the copy (or tied with it) can be missed.
    """

    def __init__(
        self,
        lm_head: VocabParallelEmbedding,
        processor: LogitsProcessor,
        top_k: int,
        preselect_quantization: str | None = None,
        preselect: int = 0,
    ):
        assert processor.scale > 0
        self.lm_head = lm_head
        self.processor = processor
        self.top_k = top_k
        self.preselect = preselect
        self.preselect_head: nn.Module | None = None
        if preselect_quantization is not None:
            from vllm.model_executor.layers.quantization.online.lm_head import (
                quantized_lm_head_copy,
            )

            self.preselect_head = quantized_lm_head_copy(
                lm_head, preselect_quantization
            )

    @staticmethod
    def supports(lm_head: nn.Module) -> bool:
        weight = getattr(lm_head, "weight", None)
        return (
            isinstance(lm_head, VocabParallelEmbedding)
            # A quant config that skips the head hands out the linear method.
            and isinstance(
                lm_head.quant_method,
                (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
            )
            and weight is not None
            and weight.is_cuda
            and lm_head.org_vocab_size <= 2**24
        )

    def _local_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This shard's (values, local ids) to choose the top-k from."""
        lm_head = self.lm_head
        head = self.preselect_head
        if head is None:
            logits = self.processor._apply_head(lm_head, hidden_states, None)
        else:
            logits = head.quant_method.apply(head, hidden_states)
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        if head is None:
            return _topk(logits, self.top_k)

        _, preselected = _topk(logits, self.preselect)
        # Exact logits of the preselected ids, rounded like the BF16 GEMM's.
        exact = torch.empty(
            preselected.shape, dtype=lm_head.weight.dtype, device=preselected.device
        )
        num_rows, hidden_size = hidden_states.shape
        if num_rows > 0:
            _rescore_candidates_kernel[(num_rows * self.preselect,)](
                hidden_states,
                lm_head.weight,
                preselected,
                exact,
                hidden_states.stride(0),
                lm_head.weight.stride(0),
                NUM_CANDIDATES=self.preselect,
                HIDDEN_SIZE=hidden_size,
                BLOCK=1024,
                num_warps=4,
            )
        return exact, preselected

    def __call__(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, local_ids = self._local_candidates(hidden_states)
        num_rows, num_candidates = values.shape
        k, tp_size = self.top_k, self.lm_head.tp_size
        packed = torch.empty(num_rows, 2 * k, dtype=torch.float32, device=values.device)
        ids = torch.empty(num_rows, k, dtype=torch.int64, device=values.device)
        out = torch.empty(num_rows, k, dtype=torch.float32, device=values.device)
        if num_rows == 0:
            return ids, out
        _local_top_k_kernel[(num_rows,)](
            values.contiguous(),
            local_ids.contiguous(),
            packed,
            self.lm_head.shard_indices.org_vocab_start_index,
            NUM_CANDIDATES=num_candidates,
            TOP_K=k,
            BLOCK=triton.next_power_of_2(num_candidates),
            num_warps=1,
        )
        if tp_size > 1:
            # Ids below 2**24 are exact in fp32, so one collective carries both.
            packed = tp_all_gather_rows(packed)
        soft_cap = self.processor.soft_cap
        _merge_top_k_kernel[(num_rows,)](
            packed,
            ids,
            out,
            num_rows,
            self.processor.scale,
            soft_cap or 1.0,
            TP_SIZE=tp_size,
            TOP_K=k,
            BLOCK=triton.next_power_of_2(tp_size * k),
            HAS_SOFT_CAP=soft_cap is not None,
            num_warps=1,
        )
        return ids, out


# Split-K CuTe tactics (M tile, row tile, split K, stages) for the drafter's
# decode-sized BF16 projections, by (N, K) per rank and the largest padded
# row count each serves. Measured on B300 for Muse Spark Open DFlash2 at TP4,
# where they beat cuBLAS's split-K GEMM plus reduction kernel; other shapes
# and larger batches stay on cuBLAS.
_SPLIT_K_TACTICS: dict[tuple[int, int], tuple[tuple[int, tuple], ...]] = {
    # fc (column shard)
    (2048, 40960): ((8, (64, 8, 4, 8)), (16, (64, 16, 4, 8)), (32, (64, 32, 4, 8))),
    # conv kernel_projection
    (2048, 8192): (
        (8, (64, 8, 4, 8)),
        (16, (64, 16, 4, 8)),
        (32, (64, 32, 4, 8)),
        (64, (64, 64, 2, 6)),
    ),
    # qkv_proj
    (2560, 8192): (
        (8, (128, 8, 4, 6)),
        (16, (64, 16, 2, 8)),
        (32, (64, 32, 2, 8)),
        (64, (64, 64, 2, 6)),
    ),
    # o_proj
    (8192, 2048): (
        (8, (64, 8, 1, 8)),
        (16, (64, 16, 1, 8)),
        (32, (64, 32, 1, 8)),
        (64, (64, 64, 1, 6)),
    ),
    # gate_up_proj
    (14336, 8192): (
        (8, (128, 8, 1, 6)),
        (16, (128, 16, 1, 6)),
        (32, (128, 32, 1, 4)),
        (64, (128, 64, 1, 4)),
    ),
    # down_proj
    (8192, 7168): ((8, (64, 8, 1, 8)), (16, (64, 16, 1, 8)), (32, (64, 32, 1, 8))),
}

# Prepared split-K projections by projection id, for the custom op below. The
# id is minted per switched layer from a monotonic counter and never reused, so
# a freed layer cannot leave an entry that a later weight (reusing its address)
# would hit; a finalizer drops the entry with the layer it belongs to. Keying by
# `weight.data_ptr()` would be unsound: the prepared kernels pin the weight, but
# another module can reuse the address after the original is freed.
_SPLIT_K_PROJECTIONS: dict[int, Any] = {}
_PROJECTION_IDS = itertools.count()


def _split_k_linear_impl(
    x: torch.Tensor, weight: torch.Tensor, projection: int
) -> torch.Tensor:
    prepared = _SPLIT_K_PROJECTIONS.get(projection)
    if prepared is None or x.shape[0] not in prepared.kernels:
        return F.linear(x, weight)
    out = x.new_empty(x.shape[0], weight.shape[0])
    return prepared.run(x.contiguous(), out)


def _split_k_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, projection: int
) -> torch.Tensor:
    return x.new_empty(x.shape[0], weight.shape[0])


direct_register_custom_op(
    op_name="dflash2_split_k_linear",
    op_func=_split_k_linear_impl,
    fake_impl=_split_k_linear_fake,
)


class _SplitKLinearMethod(UnquantizedLinearMethod):
    """The unquantized method, with decode-sized rows on the split-K GEMM.

    Owns its projection and its registry id for as long as the layer lives.
    """

    def __init__(
        self,
        base: UnquantizedLinearMethod,
        prepared: Any,
        projection: int,
    ) -> None:
        self.__dict__.update(base.__dict__)
        self.prepared = prepared
        self.projection = projection

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None or x.dim() != 2:
            return super().apply(layer, x, bias)
        return torch.ops.vllm.dflash2_split_k_linear(x, layer.weight, self.projection)


def use_split_k_projections(model: nn.Module) -> int:
    """Route the drafter's decode-sized BF16 projections with a tuned shape to
    the shared split-K CuTe GEMM, compiled now so CUDA graph capture only
    replays it. Returns the number of projections switched."""
    if envs.VLLM_BATCH_INVARIANT or not current_platform.is_device_capability_family(
        100
    ):
        return 0
    from vllm.models.common.specialized_decode.projection import (
        PreparedProjection,
        tactic_policies,
    )

    switched = 0
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        weight = getattr(module, "weight", None)
        if (
            type(method) is not UnquantizedLinearMethod
            or weight is None
            or weight.dtype != torch.bfloat16
            or tuple(weight.shape) not in _SPLIT_K_TACTICS
        ):
            continue
        policies = tactic_policies(_SPLIT_K_TACTICS[tuple(weight.shape)])
        prepared = PreparedProjection(weight, policies).prepare()
        projection = next(_PROJECTION_IDS)
        _SPLIT_K_PROJECTIONS[projection] = prepared
        weakref.finalize(module, _SPLIT_K_PROJECTIONS.pop, projection, None)
        module.quant_method = _SplitKLinearMethod(method, prepared, projection)
        switched += 1
    return switched


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model
    # Candidates preselected per rank by a quantized head, as a multiple of top-k.
    candidate_preselect_factor = 4

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.preselect_quantization = speculative_config.draft_lm_head_quantization
        self.candidate_head: CandidateHead | None = None

    def init_candidate_head(self) -> None:
        """Build the candidate head once the LM head is final (after the
        drafter may have adopted the target's)."""
        if not CandidateHead.supports(self.lm_head):
            if self.preselect_quantization is not None:
                raise ValueError(
                    "draft_lm_head_quantization for DFlash2 needs an unquantized "
                    "vocab-parallel LM head on CUDA."
                )
            return
        top_k = self.model.candidate_selector.top_k
        self.candidate_head = CandidateHead(
            self.lm_head,
            self.candidate_logits_processor,
            top_k,
            self.preselect_quantization,
            preselect=self.candidate_preselect_factor * top_k,
        )
        if self.preselect_quantization is not None:
            # A submodule, so refresh_quantized_lm_heads re-derives it.
            self.candidate_preselect_head = self.candidate_head.preselect_head
            logger.info_once(
                "DFlash2 preselects %d candidates per rank with a %s copy of the "
                "LM head and rescores them against the %s head.",
                self.candidate_head.preselect,
                self.preselect_quantization,
                self.lm_head.weight.dtype,
            )

    def init_split_k_projections(self) -> None:
        switched = use_split_k_projections(self.model)
        if switched:
            logger.info_once(
                "DFlash2 runs %d decode-sized projections on the split-K CuTe GEMM.",
                switched,
            )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.candidate_head is None:
            return self.candidate_logits_processor.get_top_k_tokens(
                self.lm_head, hidden_states, self.model.candidate_selector.top_k
            )
        return self.candidate_head(hidden_states)


EntryClass = DFlash2Qwen3ForCausalLM
