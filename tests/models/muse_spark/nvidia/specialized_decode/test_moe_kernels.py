# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's MoE router against the stock chain it replaces:
``GateLinear`` (FP32 router weight) then ``fused_musespark1x_routing`` (sigmoid,
expert bias for selection, top-8, renormalized with eps 1e-15).

The logits come from a split-BF16 tensor-core GEMM in another summation
order, so they match an FP64 reference to a few 1e-7 (relative to the largest
logit) and the experts match except at near-ties. Given its own logits, the
selection matches the Triton kernel bit for bit.

Needs one Blackwell GPU.
"""

import pytest
import torch

from vllm.models.muse_spark.common.ops.fused_routing import fused_musespark1x_routing
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the router kernel targets Blackwell (SM100 family) GPUs",
)

EXPERTS, HIDDEN, TOP_K, EPS = 256, 8192, 8, 1e-15
# Largest logit error relative to the largest |logit|, vs FP64.
LOGIT_RTOL = {torch.bfloat16: 1e-6, torch.float32: 3e-6}
TIE_GAP = 1e-5


@pytest.fixture(scope="module")
def router():
    from vllm.models.muse_spark.nvidia.specialized_decode.moe.router import MuseRouter

    g = torch.Generator(device="cuda").manual_seed(0)
    # The checkpoint's scale, output multiplier 0.177 folded in.
    weight = torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") * 0.0035
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.05
    return MuseRouter(weight, bias)


def stock(router, x, tf32=False):
    """GateLinear's FP32-weight dispatch on Blackwell, then the Triton router.

    Past the stock exact kernel's 32-token limit an FP32 input reaches
    ``F.linear``, so ``tf32`` is the deployment's TF32 GEMM there
    (``VLLM_FLOAT32_MATMUL_PRECISION`` is what makes the model ask for the FP32
    input in the first place).
    """
    import vllm.model_executor.layers.fused_moe.router.gate_linear  # noqa: F401

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        logits = torch.ops.vllm.fp32_router_gemm_dispatch(x, router.weight, True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    weights, ids = fused_musespark1x_routing(
        logits, router.expert_bias, TOP_K, 1.0, router_score_division_eps=EPS
    )
    return logits, ids, weights


def run(router, x):
    m = x.shape[0]
    ids = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    weights = torch.full((m, TOP_K), float("nan"), device="cuda")
    logits = torch.full((m, EXPERTS), float("nan"), device="cuda")
    router(x, ids, weights, logits)
    return logits, ids, weights


def activations(m, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, HIDDEN, generator=g, device="cuda").to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m", [1, 8, 33, 64, 256])
def test_matches_stock(router, m, dtype):
    x = activations(m, dtype, m)
    logits, ids, weights = run(router, x)
    exact = x.double() @ router.weight.double().t()
    error = (logits.double() - exact).abs().max() / exact.abs().max()
    assert error < LOGIT_RTOL[dtype]

    # The selection, bit for bit, from the same logits.
    ref_weights, ref_ids = fused_musespark1x_routing(
        logits.clone(), router.expert_bias, TOP_K, 1.0, router_score_division_eps=EPS
    )
    torch.testing.assert_close(ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, ref_weights, atol=0, rtol=0)

    # Against the stock chain: same experts unless the 8th and 9th biased
    # scores nearly tie; weights to the logits' precision.
    _, stock_ids, stock_weights = stock(router, x)
    scores = torch.sigmoid(exact) + router.expert_bias.double()
    top = scores.topk(TOP_K + 1, dim=-1).values
    clear = (top[:, TOP_K - 1] - top[:, TOP_K]) > TIE_GAP
    assert clear.float().mean() > 0.9
    torch.testing.assert_close(ids[clear], stock_ids[clear], atol=0, rtol=0)
    torch.testing.assert_close(
        weights[clear], stock_weights[clear], atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("m", [33, 40, 64, 96, 128])
def test_fp32_router_input(router, m):
    """The deployment's FP32 router input (32 < rows <= 128, the window where
    ``wants_fp32_router_input`` asks for one): the fused kernel takes it and
    stays exact, where the stock chain lands on a TF32 GEMM past 32 rows."""
    x = activations(m, torch.float32, m)
    logits, ids, weights = run(router, x)
    exact = x.double() @ router.weight.double().t()
    error = (logits.double() - exact).abs().max() / exact.abs().max()
    assert error < LOGIT_RTOL[torch.float32]

    # The selection, bit for bit, from the same logits.
    ref_weights, ref_ids = fused_musespark1x_routing(
        logits.clone(), router.expert_bias, TOP_K, 1.0, router_score_division_eps=EPS
    )
    torch.testing.assert_close(ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, ref_weights, atol=0, rtol=0)

    # The FP64 reference's own top-8: by biased score, ties to the lower expert.
    scores = torch.sigmoid(exact) + router.expert_bias.double()
    top = scores.topk(TOP_K + 1, dim=-1).values
    clear = (top[:, TOP_K - 1] - top[:, TOP_K]) > TIE_GAP
    expected = torch.argsort(scores, dim=-1, descending=True, stable=True)
    expected = expected[:, :TOP_K].to(torch.int32)
    assert clear.float().mean() > 0.5
    torch.testing.assert_close(ids[clear], expected[clear], atol=0, rtol=0)

    # The stock chain on the same input, as the deployment runs it (TF32 past
    # the stock exact kernel's 32-token limit): ~1500x coarser logits, and it
    # flips experts exactly where the fused kernel still gets the reference
    # order right, i.e. at 8th/9th biased-score gaps up to ~3e-3.
    tf32_logits, tf32_ids, tf32_weights = stock(router, x, tf32=True)
    tf32_error = (tf32_logits.double() - exact).abs().max() / exact.abs().max()
    assert tf32_error > 100 * error
    flipped = (tf32_ids != ids).any(-1)
    print(
        f"m={m}: exact {error:.2e}, tf32 {tf32_error:.2e}, "
        f"{clear.float().mean():.0%} clear rows, tf32 flips "
        f"{flipped.float().mean():.1%} of them"
    )
    assert flipped[clear].float().mean() < 0.1
    torch.testing.assert_close(
        weights[clear][~flipped[clear]],
        tf32_weights[clear][~flipped[clear]],
        atol=1e-5,
        rtol=1e-4,
    )


def test_fused_router_serves_fp32_input():
    """The executor's gate takes the fused path for both router input dtypes,
    the FP32 one included (it used to fall back to the stock chain)."""
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime import decode

    assert decode.ROUTER_DTYPES == (torch.bfloat16, torch.float32)
    assert 33 in decode.ROUTER_ROWS and 128 in decode.ROUTER_ROWS


def test_deterministic(router):
    """Two launches give the same bits (fixed reduction order)."""
    x = activations(64, torch.bfloat16, 1)
    first = run(router, x)
    second = run(router, x)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cuda_graph_replay(router):
    """Captured per row count and dtype, replayed on new inputs in the same
    buffers, interleaved (the logits mailboxes are shared)."""
    graphs = {}
    for m, dtype in ((8, torch.bfloat16), (64, torch.float32), (200, torch.bfloat16)):
        x = activations(m, dtype, 0)
        ids = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")
        weights = torch.empty(m, TOP_K, device="cuda")
        router(x, ids, weights)  # compile and warm up
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            router(x, ids, weights)
        graphs[m] = (graph, x, ids, weights)
    for seed in (1, 2):
        for m, (graph, x, ids, weights) in graphs.items():
            x.copy_(activations(m, x.dtype, seed * 1000 + m))
            graph.replay()
            _, ref_ids, ref_weights = run(router, x)
            torch.testing.assert_close(ids, ref_ids, atol=0, rtol=0)
            torch.testing.assert_close(weights, ref_weights, atol=0, rtol=0)


def test_capture_with_new_input_address(router):
    """An input first seen inside the capture (a graph-pool address): the
    router launch makes no host copies or allocations."""
    m = 16
    router.prepare([m], [torch.float32])
    source = activations(m, torch.bfloat16, 3)
    ids = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.empty(m, TOP_K, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x = source.float()  # allocated by the capture
        router(x, ids, weights)
    source.copy_(activations(m, torch.bfloat16, 4))
    graph.replay()
    _, ref_ids, ref_weights = run(router, source.float())
    torch.testing.assert_close(ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(weights, ref_weights, atol=0, rtol=0)
