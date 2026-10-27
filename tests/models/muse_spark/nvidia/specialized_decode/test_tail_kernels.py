# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU checks of the Muse Spark MoE tail's one-shot finalize, with its
peers played by mailbox slots the test fills (the kernel stores this rank's
fragments into its own slot instead of multicasting them):

* the per-route math against a torch reference with the same roundings (FP32
  rank sum in rank order rounded to BF16, input-scaled RMSNorm, FP32 weighted
  route sum), for FP32 and BF16 routing weights, missing and out-of-range
  routes over NaN rows, negative zeros, and more tokens than clusters;
* the published layout, the re-armed peer slots and the generation counters
  across all three mailbox generations;
* CUDA graph capture and replay.

The cross-rank kernel is checked in ``test_tail_numerics.py``.

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

TP = 4
HIDDEN = 4096
TOP_K = 8
CAPACITY = 64
EPS = 1e-8
SENTINEL = -32768


def _finalize(rank: int, published: bool = False):
    from vllm.models.muse_spark.nvidia.specialized_decode.tail.tail import Finalize

    mailbox = torch.full(
        (3, TP, CAPACITY * TOP_K, HIDDEN), SENTINEL, dtype=torch.int16, device="cuda"
    ).view(torch.bfloat16)
    finalize = Finalize(
        tp=TP,
        rank=rank,
        capacity=CAPACITY,
        mailbox=mailbox,
        multicast=0,
        published=published,
    )
    return finalize, mailbox


def _scrub(x: torch.Tensor) -> torch.Tensor:
    """BF16 with negative zeros made +0, as the kernel publishes it."""
    bits = x.view(torch.int16)
    return torch.where(bits == SENTINEL, 0, bits).view(torch.bfloat16)


def _inputs(m: int, rank: int, fp32_weights: bool, seed: int):
    """This rank's permuted GEMM2 buffer (padded, NaN outside the routes) with
    its map, every rank's [M, 8, 4096] route rows, weights and norm weight."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    routes = torch.randn(TP, m, TOP_K, HIDDEN, generator=generator, device="cuda")
    routes = routes.bfloat16()
    # Exact zeros of both signs must survive the sentinel scrub.
    routes[:, :, :, :16] = 0.0
    routes[:, :, :, 16:32] = -0.0
    num_rows = m * TOP_K + 5
    order = torch.randperm(m * TOP_K, generator=generator, device="cuda")
    idx = order.view(m, TOP_K).int()
    # A missing route and one past the buffer read zero, never their NaN rows.
    idx[0, 3] = -1
    if m > 1:
        idx[1, 6] = num_rows + 7
    rows = torch.full((num_rows, HIDDEN), float("nan"), device="cuda").bfloat16()
    valid = (idx >= 0) & (idx < num_rows)
    rows[idx[valid].long()] = routes[rank][valid]
    routes[:, ~valid] = 0.0
    weights = torch.rand(m, TOP_K, generator=generator, device="cuda")
    if not fp32_weights:
        weights = weights.bfloat16()
    norm_weight = torch.randn(HIDDEN, generator=generator, device="cuda") * 0.3
    return rows, idx, routes, weights, norm_weight.bfloat16()


def _reference(routes, weights, norm_weight, gain_center):
    total = torch.zeros(routes.shape[1:], device="cuda")
    for source in range(TP):
        total += routes[source].float()
    scaled = total.bfloat16().float() * (norm_weight.float() + gain_center)
    inverse = torch.rsqrt(scaled.square().mean(-1, keepdim=True) + EPS)
    y = scaled * inverse
    acc = torch.zeros(y.shape[0], HIDDEN, device="cuda")
    for k in range(TOP_K):
        acc = torch.addcmul(acc, y[:, k], weights[:, k : k + 1].float())
    return acc.bfloat16()


def _fill_peers(mailbox, routes, generation: int, rank: int) -> None:
    m = routes.shape[1]
    for source in range(TP):
        if source != rank:
            mailbox[generation, source, : m * TOP_K] = _scrub(
                routes[source].reshape(-1, HIDDEN)
            )


def _check(out, want, what: str) -> None:
    equal = (out.view(torch.int16) == want.view(torch.int16)).float().mean().item()
    rel = (
        (out.float() - want.float()).abs() / want.float().abs().clamp_min(1e-3)
    ).max()
    print(f"{what}: {equal:.5f} bitwise, max rel {rel.item():.3g}")
    assert not out.isnan().any()
    assert equal > 0.99, what
    torch.testing.assert_close(out.float(), want.float(), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("fp32_weights", [True, False])
@pytest.mark.parametrize("m", [1, 3, 8, 16, 40])
def test_finalize_matches_reference(m: int, fp32_weights: bool, published: bool):
    """Four launches cycle the mailbox generations (their own, or with
    ``published`` rows the given layer parities); each one re-arms the peers'
    slots it consumed and advances every CTA's generation."""
    rank = 2
    finalize, mailbox = _finalize(rank, published)
    out = torch.empty(CAPACITY, HIDDEN, dtype=torch.bfloat16, device="cuda")
    for launch in range(4):
        gain_center = float(launch % 2)
        rows, idx, routes, weights, norm_weight = _inputs(
            m, rank, fp32_weights, seed=launch
        )
        generation = (2, 0, 1, 2)[launch] if published else launch % 3
        _fill_peers(mailbox, routes, generation, rank)
        # Route (0, 3) has no expert (and no row): the experts publish none,
        # so the finalize publishes its zero row itself.
        ids = torch.zeros(m, TOP_K, dtype=torch.int32, device="cuda")
        ids[0, 3] = -1
        got = finalize(
            rows,
            idx,
            weights,
            norm_weight,
            out,
            eps=EPS,
            gain_center=gain_center,
            parity=generation,
            topk_ids=ids if published else None,
        )
        torch.accelerator.synchronize()
        _check(got, _reference(routes, weights, norm_weight, gain_center), f"M={m}")
        if published:
            assert not mailbox[generation, rank, 3].view(torch.int16).any()
        else:
            own = mailbox[generation, rank, : m * TOP_K].view(m, TOP_K, HIDDEN)
            assert torch.equal(
                own.view(torch.int16), _scrub(routes[rank]).view(torch.int16)
            )
        peers = [s for s in range(TP) if s != rank]
        assert (mailbox[:, peers].view(torch.int16) == SENTINEL).all(), "re-arm"
        assert (finalize.generations == launch + 1).all()


def test_finalize_graph_replay():
    rank = 0
    m = 8
    finalize, mailbox = _finalize(rank)
    finalize.precompile()
    out = torch.empty(CAPACITY, HIDDEN, dtype=torch.bfloat16, device="cuda")
    rows, idx, routes, weights, norm_weight = _inputs(m, rank, True, seed=0)
    static = (rows, idx, weights)

    def run():
        return finalize(*static, norm_weight, out, eps=EPS, gain_center=1.0)

    _fill_peers(mailbox, routes, 0, rank)
    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = run()
    for replay in range(6):
        rows, idx, routes, weights, _ = _inputs(m, rank, True, seed=replay + 1)
        for dst, src in zip(static, (rows, idx, weights)):
            dst.copy_(src)
        _fill_peers(mailbox, routes, (replay + 1) % 3, rank)
        graph.replay()
        torch.accelerator.synchronize()
        _check(got, _reference(routes, weights, norm_weight, 1.0), f"replay {replay}")
