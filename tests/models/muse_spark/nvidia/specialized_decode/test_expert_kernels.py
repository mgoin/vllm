# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's routed-expert kernel against the stock chain it
replaces: ``MuseSparkTrtLlmNvFp4Experts.apply_routed``'s FlashInfer
``trtllm_fp4_block_scale_routed_moe`` (routing indices, GEMM1 + SwiGLU,
GEMM2, ``do_finalize=False``) on the same prepared NVFP4 weights and scales.

The kernel computes the TRT-LLM-gen arithmetic instruction for instruction, so
every GEMM2 row matches bit for bit; only the row order differs (route-major
here, expert-major there), which the permute maps undo.

Needs one Blackwell GPU.
"""

import os
import subprocess
import sys

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the expert kernel targets Blackwell (SM100 family) GPUs",
)

EXPERTS, HIDDEN, INTER, TOP_K = 256, 4096, 1024, 8
ROWS = (1, 2, 4, 8, 16, 32, 64)


def _nvfp4(w: torch.Tensor, global_scale: torch.Tensor):
    from vllm import _custom_ops as ops

    q, s = ops.scaled_fp4_quant(w, global_scale, is_sf_swizzled_layout=False)
    return q.view(torch.uint8), s.view(torch.uint8).reshape(w.shape[0], -1)


@pytest.fixture(scope="module")
def layer():
    return make_layer()


def make_layer():
    """Random experts in the ModelOpt checkpoint layout, prepared as
    ``MuseSparkModelOptNvFp4FusedMoE.process_weights_after_loading`` prepares
    them for the TRT-LLM-gen kernels (w1/w3 reordered, rows shuffled, scales
    interleaved; the split gate/up global scales folded)."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    w13 = torch.empty(EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.uint8, device="cuda")
    w13_scale = torch.empty(
        EXPERTS, 2 * INTER, HIDDEN // 16, dtype=torch.uint8, device="cuda"
    )
    w2 = torch.empty(EXPERTS, HIDDEN, INTER // 2, dtype=torch.uint8, device="cuda")
    w2_scale = torch.empty(
        EXPERTS, HIDDEN, INTER // 16, dtype=torch.uint8, device="cuda"
    )
    w13_scale_2 = torch.empty(EXPERTS, 2, device="cuda")
    w2_scale_2 = torch.empty(EXPERTS, device="cuda")
    for e in range(EXPERTS):
        std = 0.02 * (0.5 + torch.rand((), generator=g, device="cuda"))
        parts = (
            torch.randn(INTER, HIDDEN, generator=g, device="cuda") * std,
            torch.randn(INTER, HIDDEN, generator=g, device="cuda") * std * 1.3,
        )
        for i, w in enumerate(parts):
            w13_scale_2[e, i] = w.abs().max() / (6 * 448)
            rows = slice(i * INTER, (i + 1) * INTER)
            w13[e, rows], w13_scale[e, rows] = _nvfp4(
                w.bfloat16(), (1 / w13_scale_2[e, i]).reshape(1)
            )
        w = torch.randn(HIDDEN, INTER, generator=g, device="cuda") * std
        w2_scale_2[e] = w.abs().max() / (6 * 448)
        w2[e], w2_scale[e] = _nvfp4(w.bfloat16(), (1 / w2_scale_2[e]).reshape(1))
    w13, w13_scale = reorder_w1w3_to_w3w1(w13, w13_scale.view(torch.float8_e4m3fn))
    w13, w13_scale, w2, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
        w13,
        w2,
        w13_scale,
        w2_scale.view(torch.float8_e4m3fn),
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_experts=EXPERTS,
        is_gated_activation=True,
    )
    # Input and intermediate global scales (amax / (6 * 448)) of a typical
    # layer: inputs up to ~4, H up to ~6.
    a13 = torch.tensor([4.0 / (6 * 448)], device="cuda")
    a2 = torch.tensor([6.0 / (6 * 448)], device="cuda")
    return dict(
        w13=w13,
        w13_scale=w13_scale.view(torch.float8_e4m3fn),
        w2=w2,
        w2_scale=w2_scale.view(torch.float8_e4m3fn),
        g1_alphas=(a13 * w13_scale_2[:, 0]).contiguous(),
        g1_scale_c=(a13 * w13_scale_2[:, 1] / a2).contiguous(),
        g2_alphas=(a2 * w2_scale_2).contiguous(),
        a1_gscale=(1.0 / a13).contiguous(),
    )


def _kernel(layer, ids_ready=False):
    from vllm.models.muse_spark.nvidia.specialized_decode.experts.routed_experts import (  # noqa: E501
        MuseRoutedExperts,
    )

    return MuseRoutedExperts(
        layer["w13"],
        layer["w13_scale"],
        layer["w2"],
        layer["w2_scale"],
        layer["g1_alphas"],
        layer["g1_scale_c"],
        layer["g2_alphas"],
        ids_ready=ids_ready,
    )


@pytest.fixture(scope="module")
def experts(layer):
    return _kernel(layer)


def stock(layer, a1q, a1q_scale, ids, weights):
    """``apply_routed``'s FlashInfer call; GEMM2 rows in route-major order."""
    import flashinfer

    m = a1q.shape[0]
    rows, _, permuted = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, weights),
        routing_bias=None,
        hidden_states=a1q,
        hidden_states_scale=a1q_scale.view(torch.float8_e4m3fn).reshape(m, -1),
        gemm1_weights=layer["w13"],
        gemm1_weights_scale=layer["w13_scale"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer["w2"],
        gemm2_weights_scale=layer["w2_scale"],
        gemm2_bias=None,
        output1_scale_scalar=layer["g1_scale_c"],
        output1_scale_gate_scalar=layer["g1_alphas"],
        output2_scale_scalar=layer["g2_alphas"],
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=7,  # MiniMax2, as Muse; unused with given ids
        do_finalize=False,
        activation_type=3,  # SwiGLU
    )
    return rows[permuted.reshape(-1).long()]


def inputs(layer, m, routing, seed):
    """NVFP4 input as ``norm_quant_trtllm_nvfp4`` lays it out, and top-8
    routing: ``uniform`` over all experts, ``pool16`` from 16 experts, or
    ``hot`` (four experts shared by every token, four uniform)."""
    from vllm import _custom_ops as ops

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, generator=g, device="cuda").bfloat16()
    a1q, a1q_scale = ops.scaled_fp4_quant(
        x, layer["a1_gscale"], is_sf_swizzled_layout=False
    )
    cpu = torch.Generator().manual_seed(seed)
    order = torch.randperm(EXPERTS, generator=cpu)
    if routing == "uniform":
        ids = torch.stack(
            [torch.randperm(EXPERTS, generator=cpu)[:TOP_K] for _ in range(m)]
        )
    elif routing == "pool16":
        ids = torch.stack(
            [order[torch.randperm(16, generator=cpu)[:TOP_K]] for _ in range(m)]
        )
    else:
        rest = order[4:]
        ids = torch.stack(
            [
                torch.cat(
                    [order[:4], rest[torch.randperm(rest.numel(), generator=cpu)[:4]]]
                )[torch.randperm(TOP_K, generator=cpu)]
                for _ in range(m)
            ]
        )
    weights = torch.rand(m, TOP_K, generator=cpu) + 0.1
    weights /= weights.sum(-1, keepdim=True)
    return (
        a1q.view(torch.uint8),
        a1q_scale.view(torch.uint8).reshape(m, HIDDEN // 16),
        ids.to(torch.int32).cuda(),
        weights.cuda(),
    )


def run(experts, a1q, a1q_scale, ids, weights):
    m = a1q.shape[0]
    out = torch.full((m * TOP_K, HIDDEN), float("nan"), device="cuda").bfloat16()
    result = experts(a1q, a1q_scale, ids, weights, out)
    assert result.gemm2_permuted.data_ptr() == out.data_ptr()
    return result


@pytest.mark.parametrize("routing", ["uniform", "pool16", "hot"])
@pytest.mark.parametrize("m", ROWS)
def test_matches_stock(layer, experts, m, routing):
    """Every GEMM2 row bit for bit; the output contract of the finalize."""
    a1q, a1q_scale, ids, weights = inputs(layer, m, routing, seed=m)
    result = run(experts, a1q, a1q_scale, ids, weights)
    ref = stock(layer, a1q, a1q_scale, ids, weights)
    assert ref.abs().max() > 0
    torch.testing.assert_close(result.gemm2_permuted, ref, atol=0, rtol=0)
    assert result.route_major
    assert result.expert_weights.data_ptr() == weights.data_ptr()
    torch.testing.assert_close(
        result.expanded_idx_to_permuted_idx,
        torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").view(m, TOP_K),
    )


def test_ids_ready_matches(layer):
    """The variant that reads the routing before its grid dependency wait."""
    experts = _kernel(layer, ids_ready=True)
    for m in (3, 8, 40):
        a1q, a1q_scale, ids, weights = inputs(layer, m, "uniform", seed=100 + m)
        result = run(experts, a1q, a1q_scale, ids, weights)
        ref = stock(layer, a1q, a1q_scale, ids, weights)
        torch.testing.assert_close(result.gemm2_permuted, ref, atol=0, rtol=0)


def test_back_to_back_calls(layer, experts):
    """The kernel resets its workspace: calls of every tile width in a row,
    without synchronizing, each match their own reference."""
    cases = [inputs(layer, m, "uniform", seed=200 + m) for m in (64, 1, 16, 9, 33)]
    results = [run(experts, *case) for case in cases]
    for case, result in zip(cases, results):
        ref = stock(layer, *case)
        torch.testing.assert_close(result.gemm2_permuted, ref, atol=0, rtol=0)


def test_cuda_graph_replay(layer, experts):
    """Captured per row count, replayed on new inputs in the same buffers,
    interleaved."""
    graphs = {}
    for m in (1, 8, 64):
        case = inputs(layer, m, "uniform", seed=300 + m)
        out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
        experts(*case, out)  # warm up
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts(*case, out)
        graphs[m] = (graph, case, out)
    for seed in (1, 2):
        for m, (graph, case, out) in graphs.items():
            fresh = inputs(layer, m, "hot" if seed == 2 else "uniform", seed=seed)
            for buffer, value in zip(case, fresh):
                buffer.copy_(value)
            out.fill_(float("nan"))
            graph.replay()
            ref = stock(layer, *case)
            torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_capture_with_new_input_addresses(layer, experts):
    """Inputs and output first allocated inside the capture (graph-pool
    addresses): the launch makes no host copies or allocations of its own."""
    m = 16
    source = inputs(layer, m, "pool16", seed=400)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case = tuple(t.clone() for t in source)
        result = experts(*case)
    for buffer, value in zip(source, inputs(layer, m, "uniform", seed=401)):
        buffer.copy_(value)
    graph.replay()
    ref = stock(layer, *source)
    torch.testing.assert_close(result.gemm2_permuted, ref, atol=0, rtol=0)


# Degenerate routing, as dummy runs and padded rows of a captured graph produce
# it from garbage activations. Every case must complete (a hang would block the
# GPU, so they run in a subprocess with a timeout).


def _routes_reference(layer, case, ids):
    """Each route's row from the stock kernel on sanitized ids: an invalid id
    becomes an unused expert, a repeat of an earlier slot's expert becomes
    another unused expert; the expected row of a repeat is the earlier
    slot's row, of an invalid id zero."""
    a1q, a1q_scale, _, weights = case
    m = ids.shape[0]
    clean = ids.clone()
    source = torch.arange(m * TOP_K).view(m, TOP_K)
    invalid = (ids < 0) | (ids >= EXPERTS)
    for t in range(m):
        taken = set(ids[t].tolist())
        spare = iter(e for e in range(EXPERTS) if e not in taken)
        for k in range(TOP_K):
            e = int(ids[t, k])
            if invalid[t, k]:
                clean[t, k] = next(spare)
                continue
            for j in range(k):
                if int(ids[t, j]) == e:
                    clean[t, k] = next(spare)
                    source[t, k] = t * TOP_K + j
                    break
    ref = stock(layer, a1q, a1q_scale, clean.int().cuda().contiguous(), weights)
    expected = ref[source.view(-1).cuda()]
    expected[invalid.view(-1).cuda()] = 0
    return expected


def _degenerate_cases(layer, m, seed):
    a1q, a1q_scale, ids, weights = inputs(layer, m, "uniform", seed)
    ids = ids.cpu()
    g = torch.Generator().manual_seed(seed)
    bad = ids.clone()
    pick = torch.rand(m, TOP_K, generator=g) < 0.3
    values = torch.tensor([-1, 256, 1000, -(2**31)], dtype=torch.int32)
    bad[pick] = values[torch.randint(0, 4, (int(pick.sum()),), generator=g)]
    dup = ids.clone()
    dup[:, 5] = dup[:, 1]
    dup[:, 7] = dup[:, 1]
    one = torch.full((m, TOP_K), 17, dtype=torch.int32)
    one[:, 0] = 3
    same = torch.full((m, TOP_K), 9, dtype=torch.int32)
    return [(a1q, a1q_scale, x.contiguous(), weights) for x in (bad, dup, one, same)]


def _robustness_main():
    from vllm.models.muse_spark.nvidia.specialized_decode.experts.routed_experts import (  # noqa: E501
        MuseRoutedExperts,
    )

    layer = make_layer()
    experts = _kernel(layer)
    assert isinstance(experts, MuseRoutedExperts)
    for m in range(1, 33):
        for case in _degenerate_cases(layer, m, seed=500 + m):
            ids = case[2]
            got = run(experts, case[0], case[1], ids.cuda(), case[3])
            expected = _routes_reference(layer, case, ids)
            torch.testing.assert_close(got.gemm2_permuted, expected, atol=0, rtol=0)
    # NaN input scales (E4M3 0x7F) on the first token: the same rows as the
    # stock kernel, bit for bit (H's saturating quantization keeps them finite).
    for m in (1, 8, 32):
        a1q, a1q_scale, ids, weights = inputs(layer, m, "uniform", seed=600 + m)
        a1q_scale[0] = 0x7F
        got = run(experts, a1q, a1q_scale, ids, weights).gemm2_permuted
        ref = stock(layer, a1q, a1q_scale, ids, weights)
        torch.testing.assert_close(got, ref, atol=0, rtol=0, equal_nan=True)
    # Back to back across widths inside graphs, degenerate and NaN inputs mixed.
    graphs = []
    for m in (3, 12, 30, 1, 20):
        case = _degenerate_cases(layer, m, seed=700 + m)[m % 4]
        case = (case[0], case[1].clone(), case[2].cuda(), case[3])
        out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
        experts(*case, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts(*case, out)
        graphs.append((graph, case, out))
    for _ in range(20):
        for graph, _, _ in graphs:
            graph.replay()
    torch.accelerator.synchronize()
    for _, case, out in graphs:
        expected = _routes_reference(layer, case, case[2].cpu())
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    print("robustness ok")


def test_degenerate_routing_completes():
    """Invalid ids (-1, 256, 1000, INT_MIN) give zero rows, an expert repeated
    within a token gives each route the expert's row, all tokens on one or two
    experts, NaN input scales; M = 1..32, eager and graph replays back to back
    across widths. A hang fails the test after the timeout."""
    env = dict(os.environ, VLLM_MUSE_EXPERTS_SPIN_LIMIT="200000000")
    result = subprocess.run(
        [sys.executable, __file__],
        env=env,
        timeout=900,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "robustness ok" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:]
    )


if __name__ == "__main__":
    _robustness_main()
