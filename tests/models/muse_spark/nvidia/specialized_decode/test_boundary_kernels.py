# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU checks of the Muse Spark boundary's publishing GEMM, with its
local C store in place of the mailbox: numerics against ``torch.mm`` for the
three boundary shapes (o_proj, pooler and unpooler shards) at decode and verify
sizes, every tactic family (split-K, multicast, several MMA tiles, chunked
split-K exchange, partial token tiles), negative-zero sanitizing, and CUDA graph
capture and replay. The publish itself is checked across ranks in
``test_boundary_numerics.py``.

Needs one SM100-family GPU; about a minute.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="needs SM100",
)

SHAPES = {"o_proj": (8192, 2048), "pooler": (1024, 8192), "unpooler": (2048, 4096)}


def _operands(n: int, k: int, m: int, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(n, k, generator=generator, device="cuda") * 0.02
    x = torch.randn(m, k, generator=generator, device="cuda")
    return x.bfloat16(), weight.bfloat16()


def _gemm(n: int, k: int, m: int, tactic=None):
    from vllm.models.muse_spark.nvidia.specialized_decode.boundary.boundary import (
        PublishingGemm,
    )

    return PublishingGemm(
        n, k, None, max_tokens=m, store_c=True, tactics={m: tactic} if tactic else None
    )


def _check(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> None:
    """Our rounding error stays within 1.25x of cuBLAS's (both round an FP32
    sum once to BF16; only the accumulation order differs)."""
    ref = x.float() @ weight.float().t()
    stock = torch.mm(x, weight.t())
    error = (out.float() - ref).abs().max().item()
    stock_error = (stock.float() - ref).abs().max().item()
    equal = (out.view(torch.int16) == stock.view(torch.int16)).float().mean().item()
    print(f"max |err| {error:.3g} (torch.mm {stock_error:.3g}), {equal:.4f} bitwise")
    assert not out.isnan().any()
    assert error <= 1.25 * stock_error + 1e-6


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("m", [1, 8, 13, 24, 64, 100, 256])
def test_default_tactics_match_mm(shape: str, m: int):
    n, k = SHAPES[shape]
    x, weight = _operands(n, k, m)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    _gemm(n, k, m)(x, weight, out)
    _check(out, x, weight)


@pytest.mark.parametrize(
    "shape, m, tactic",
    [
        # One MMA of 64 rows, the whole K.
        ("o_proj", 8, (64, 8, 1, 11, 128, 1, 1)),
        # Split-K 2 and 8 through the cluster's bulk copies.
        ("o_proj", 13, (128, 16, 2, 11, 64, 1, 1)),
        ("pooler", 3, (64, 16, 8, 16, 64, 1, 1)),
        # Activation multicast over 4 column tiles.
        ("o_proj", 64, (64, 64, 1, 13, 64, 4, 1)),
        # Weight and activation multicast with split-K.
        ("o_proj", 70, (128, 64, 2, 7, 64, 2, 2)),
        # Two and four 128-row MMAs, chunked split-K exchange.
        ("o_proj", 200, (256, 256, 4, 3, 64, 1, 1)),
        ("o_proj", 100, (512, 128, 8, 2, 64, 1, 1)),
        # Token tile not a power of two.
        ("unpooler", 33, (128, 48, 2, 9, 64, 1, 1)),
        ("unpooler", 256, (128, 128, 4, 6, 64, 1, 1)),
    ],
)
def test_tactics_match_mm(shape: str, m: int, tactic: tuple):
    from vllm.models.muse_spark.nvidia.specialized_decode.boundary.publishing_gemm import (  # noqa: E501
        GemmTactic,
    )

    n, k = SHAPES[shape]
    x, weight = _operands(n, k, m, seed=1)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    _gemm(n, k, m, GemmTactic(*tactic))(x, weight, out)
    _check(out, x, weight)


def test_negative_zero_is_published_as_positive_zero():
    """-0.0 is the Lamport sentinel: an output that rounds to -0 must not be
    published (or stored) as -0."""
    n, k, m = 8192, 2048, 8
    x = torch.full((m, k), 1e-30, dtype=torch.bfloat16, device="cuda")
    weight = torch.full((n, k), -1e-30, dtype=torch.bfloat16, device="cuda")
    weight[:64] = 1.0
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    _gemm(n, k, m)(x, weight, out)
    torch.accelerator.synchronize()
    assert (out[:, 64:].float() == 0).all()
    assert not (out.view(torch.int16) == -32768).any()


def test_cuda_graph_replay():
    """Pointer-stable launches replay in a graph with fresh activations."""
    n, k = SHAPES["o_proj"]
    gemms = {m: _gemm(n, k, m) for m in (8, 64)}
    static = {m: _operands(n, k, m, seed=m) for m in gemms}
    outs = {m: torch.empty(m, n, dtype=torch.bfloat16, device="cuda") for m in gemms}
    for m, gemm in gemms.items():
        gemm.precompile()
        gemm(*static[m], outs[m])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for m, gemm in gemms.items():
            gemm(*static[m], outs[m])
    for step in range(3):
        for m in gemms:
            x, _ = _operands(n, k, m, seed=100 + step)
            static[m][0].copy_(x)
        graph.replay()
        for m in gemms:
            _check(outs[m], *static[m])


@pytest.mark.parametrize("m", [1, 5, 8, 9, 16, 32])
def test_gated_activation_matches_flashinfer(m: int):
    """The o-norm gate folded into the GEMM's activation load is bit for bit
    FlashInfer's ``rmsnorm_sigmoid_gate`` (both of its reduction trees: up to
    8 and above 8 rows) followed by the same GEMM, and leaves the input as
    is."""
    from flashinfer.norm import rmsnorm_sigmoid_gate

    n, k = SHAPES["o_proj"]
    x, weight = _operands(n, k, m, seed=2)
    gate = torch.randn(m, k, device="cuda").bfloat16()
    original = x.clone()
    gemm = _gemm(n, k, m)
    fused = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    gemm(x, weight, fused, gate=gate)
    gated = x.clone()
    rmsnorm_sigmoid_gate(gated.view(m, -1, 64), gate.view(m, -1, 64), 1e-5)
    want = torch.empty_like(fused)
    _gemm(n, k, m, gemm.tactic(m, gated=True))(gated, weight, want)
    torch.accelerator.synchronize()
    assert torch.equal(x, original)
    assert torch.equal(fused.view(torch.int16), want.view(torch.int16))
