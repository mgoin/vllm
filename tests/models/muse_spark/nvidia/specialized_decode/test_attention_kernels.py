# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's fused QKV projection against the stock chain it
replaces: ``x @ W^T`` then ``ops.fused_muse_spark_qknorm_rope_kv_insert`` into
a real paged cache, at the TP4 shapes (32 q / 4 kv heads of 64, hidden 8192).

With integer-valued inputs every GEMM summation order gives the same FP32 sum,
so the fused kernel must match the stock chain bit for bit (outputs and cache).
With random inputs only the GEMM's summation order differs.

Needs one Blackwell GPU.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="tcgen05 kernels need a Blackwell (SM100 family) GPU",
)

NUM_HEADS, NUM_KV_HEADS, HEAD, HIDDEN = 32, 4, 64, 8192
N = 2 * (NUM_HEADS + NUM_KV_HEADS) * HEAD
Q = NUM_HEADS * HEAD
EPS = 1e-5
BLOCKS, BLOCK_SIZE = 64, 32
MAX_POSITION = 4096


def rope_table(theta: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / theta ** (torch.arange(0, HEAD, 2, dtype=torch.float) / HEAD)
    freqs = torch.outer(torch.arange(MAX_POSITION, dtype=torch.float), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).cuda()


def kv_cache(layout: str) -> torch.Tensor:
    """A [blocks, kv heads, block size, K | V] view of an HND or NHD cache."""
    if layout == "HND":
        shape = (BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD)
        return torch.randn(shape, device="cuda").to(torch.bfloat16)
    shape = (BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 2 * HEAD)
    return torch.randn(shape, device="cuda").to(torch.bfloat16).transpose(1, 2)


def inputs(m: int, exact: bool, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if exact:
        x = torch.randint(-1, 2, (m, HIDDEN), generator=g, device="cuda")
        x = x.to(torch.bfloat16)
    else:
        x = torch.randn(m, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    positions = torch.randint(0, MAX_POSITION, (m,), generator=g, device="cuda")
    slots = torch.randperm(BLOCKS * BLOCK_SIZE, generator=g, device="cuda")[:m]
    slots[::3] = -1  # CUDA graph padding rows
    return x, positions, slots


def stock(x, weight, cache, slots, positions, table):
    qkv = torch.mm(x, weight.t())
    ops.fused_muse_spark_qknorm_rope_kv_insert(
        qkv, positions, NUM_HEADS, NUM_KV_HEADS, HEAD, EPS,
        cos_sin_cache=table, rotary_dim=HEAD if table is not None else 0,
        slot_mapping=slots, kv_cache=cache, block_size=BLOCK_SIZE,
    )  # fmt: skip
    return qkv


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    exact = torch.randint(-2, 3, (N, HIDDEN), generator=g, device="cuda") / 64
    noisy = torch.randn(N, HIDDEN, generator=g, device="cuda") * 0.02
    return {True: exact.to(torch.bfloat16), False: noisy.to(torch.bfloat16)}


@pytest.fixture(scope="module")
def projections(weights):
    from vllm.models.muse_spark.nvidia.specialized_decode.attention.qkv_projection import (  # noqa: E501
        QkvProjection,
    )

    return {
        (exact, rope): QkvProjection(w, NUM_HEADS, NUM_KV_HEADS, EPS, rope)
        for exact, w in weights.items()
        for rope in (True, False)
    }


def run_both(weights, projections, m, exact, rope, layout, split=False, seed=1):
    """(fused, stock) [q | k | v | o_gate] and caches. ``split``: q into rows of
    a larger contiguous buffer and o_gate into its own, nothing else (k and v
    compared from the cache only)."""
    x, positions, slots = inputs(m, exact, seed)
    table = rope_table() if rope else None
    cache = kv_cache(layout)
    ref_cache = cache.clone()
    ref = stock(x, weights[exact], ref_cache, slots, positions, table)
    projection = projections[exact, rope]
    if not split:
        out = torch.full_like(ref, float("nan"))
        projection(x, out, cache, slots, positions, table)
        return out, ref, cache, ref_cache
    q_out = torch.full((256, Q), float("nan"), device="cuda").to(torch.bfloat16)[:m]
    og_out = torch.full((m, Q), float("nan"), device="cuda").to(torch.bfloat16)
    projection(x, None, cache, slots, positions, table, q_out=q_out, og_out=og_out)
    out = torch.cat([q_out, ref[:, Q:-Q], og_out], dim=1)
    return out, ref, cache, ref_cache


@pytest.mark.parametrize("split", [False, True], ids=["qkv", "split"])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("rope", [True, False], ids=["sliding", "nope"])
@pytest.mark.parametrize("m", [1, 7, 8, 16, 64, 256])
def test_bitwise_with_exact_gemm(weights, projections, m, rope, layout, split):
    """q, k (normed, rotated), v, o_gate and the cache entries, padding slots
    skipped, exactly as the stock chain writes them."""
    out, ref, cache, ref_cache = run_both(
        weights, projections, m, True, rope, layout, split
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    torch.testing.assert_close(cache, ref_cache, atol=0, rtol=0)


@pytest.mark.parametrize("rope", [True, False], ids=["sliding", "nope"])
@pytest.mark.parametrize("m", [1, 16, 64, 256])
def test_random_inputs(weights, projections, m, rope):
    """Random weights and activations: only the GEMM's FP32 summation order
    differs, so a BF16 output ulp at most, on a small share of elements."""
    out, ref, cache, ref_cache = run_both(weights, projections, m, False, rope, "HND")
    for got, want in ((out, ref), (cache, ref_cache)):
        torch.testing.assert_close(got, want, atol=0.07, rtol=0.016)
        assert (got == want).float().mean() > 0.99


def test_cuda_graph_replay(weights, projections):
    """Captured once per row count (contiguous q and o_gate outputs), replayed
    on new contents of the same buffers, row counts interleaved (the K-split
    mailboxes are shared)."""
    table = rope_table()
    weight, projection = weights[True], projections[True, True]
    graphs = {}
    for m in (8, 256, 24):
        x, positions, slots = inputs(m, True, seed=m)
        cache = kv_cache("HND")
        q, og = (torch.empty(m, Q, dtype=torch.bfloat16, device="cuda") for _ in "qo")
        args = (x, None, cache, slots, positions, table)
        projection(*args, q_out=q, og_out=og)  # compile and warm up
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection(*args, q_out=q, og_out=og)
        graphs[m] = (graph, x, positions, slots, cache, q, og)
    for seed in (5, 6):
        for m, (graph, x, positions, slots, cache, q, og) in graphs.items():
            new_x, new_positions, new_slots = inputs(m, True, seed=seed * 1000 + m)
            x.copy_(new_x)
            positions.copy_(new_positions)
            slots.copy_(new_slots)
            ref_cache = cache.clone()
            ref = stock(x, weight, ref_cache, slots, positions, table)
            graph.replay()
            torch.testing.assert_close(q, ref[:, :Q], atol=0, rtol=0)
            torch.testing.assert_close(og, ref[:, -Q:], atol=0, rtol=0)
            torch.testing.assert_close(cache, ref_cache, atol=0, rtol=0)


def replay_memcpys(graph) -> list[str]:
    """The copies a replay of ``graph`` issues. The projection prepares its TMA
    descriptors when it is bound and the kernel points its own copy at the call's
    input, so neither the capture nor a replay may copy anything from the host."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    return [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and "Memcpy" in event.name
    ]


def test_capture_with_new_input_address(weights, projections):
    """An input first seen inside the capture (a graph-pool address, as in the
    model's capture) gets pointed at by the kernel's own descriptor copy, and a
    replay of that graph copies nothing from the host."""
    table = rope_table()
    weight, projection = weights[True], projections[True, True]
    m = 16
    projection.prepare([m])
    source, positions, slots = inputs(m, True, seed=7)
    cache = kv_cache("HND")
    q, og = (torch.empty(m, Q, dtype=torch.bfloat16, device="cuda") for _ in "qo")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x = source * 1  # allocated by the capture
        projection(x, None, cache, slots, positions, table, q_out=q, og_out=og)
    assert not replay_memcpys(graph)
    for seed in (8, 9):
        new_x, new_positions, new_slots = inputs(m, True, seed=seed)
        source.copy_(new_x)
        positions.copy_(new_positions)
        slots.copy_(new_slots)
        ref_cache = cache.clone()
        ref = stock(source, weight, ref_cache, slots, positions, table)
        graph.replay()
        torch.testing.assert_close(q, ref[:, :Q], atol=0, rtol=0)
        torch.testing.assert_close(og, ref[:, -Q:], atol=0, rtol=0)
        torch.testing.assert_close(cache, ref_cache, atol=0, rtol=0)


def test_bind_does_not_leak_descriptors(weights, projections):
    """Binding the projection once serves any number of captures: no capture or
    replay stages or copies a descriptor, so nothing pinned accumulates."""
    projection = projections[True, True]
    m = 8
    x, positions, slots = inputs(m, True, seed=11)
    cache = kv_cache("HND")
    table = rope_table()
    q, og = (torch.empty(m, Q, dtype=torch.bfloat16, device="cuda") for _ in "qo")
    args = (x, None, cache, slots, positions, table)
    graphs = []
    for _ in range(2):  # a second capture must not add host staging either
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection(*args, q_out=q, og_out=og)
        graphs.append(graph)
    for graph in graphs:
        assert not replay_memcpys(graph)
