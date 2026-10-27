# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank numerics of the Muse Spark boundary publishing GEMMs at TP4.

Every check runs one rank late (a different one each step), so a missing
Lamport wait reads a peer's stale data:

* ``BoundaryAllReduce`` (o_proj publishing GEMM + all-reduce + dual norm, one-
  and two-shot) at 1, 8, 24, 64 and 256 rows, with and without the FP32 router
  input: bitwise against the stock chain fed the same GEMM partials (sum in
  FP32 in rank order, BF16, FlashInfer's ``gated_residual_next_rmsnorm``), and
  against the stock chain on ``torch.mm`` partials (bitwise fraction and max
  error reported);
* the same with the o-norm gate (``o_proj(..., gate=og)``), folded into the
  GEMM or not, against FlashInfer's ``rmsnorm_sigmoid_gate`` + the stock chain;
* the pooler and unpooler shards published into FlashInfer all-gather
  workspaces, consumed by FlashInfer's unchanged consumers, bitwise against
  FlashInfer's own publish of the same shard;
* one CUDA graph of boundaries at several sizes replayed ``REPLAYS`` times with
  fresh inputs, bitwise, after which every mailbox slot must be re-armed.

Needs four SM100-family GPUs with NVLink multicast; about 2 minutes. For a
race stress run, run this file as a module with ``--replays 10000``.
"""

import argparse
import os
import tempfile

import pytest
import torch

from vllm.platforms import current_platform

TP = 4
HIDDEN = 8192
K_O_PROJ = 2048
CAPACITY = 256
REPLAYS = 50
LATE_CYCLES = 20_000
POST_EPS, NEXT_EPS, GAIN = 1e-8, 1e-5, 1.0


def _late(rank: int, step: int) -> None:
    if rank == step % TP:
        torch.cuda._sleep(LATE_CYCLES)


def _boundary_inputs(m: int, rank: int, seed: int):
    """This rank's attention output and o_proj shard, and the replicated
    residual-stream inputs."""
    own = torch.Generator(device="cuda").manual_seed(seed + 97 * rank)
    attn = torch.randn(m, K_O_PROJ, generator=own, device="cuda").bfloat16()
    shared = torch.Generator(device="cuda").manual_seed(seed)
    return attn, dict(
        stream=torch.randn(m, HIDDEN, generator=shared, device="cuda") * 3,
        alpha=torch.rand(HIDDEN, generator=shared, device="cuda") + 0.5,
        beta=torch.rand(HIDDEN, generator=shared, device="cuda"),
        pre_weight=(
            torch.randn(HIDDEN, generator=shared, device="cuda") * 0.1
        ).bfloat16(),
    )


def _stock_dual_norm(partial: torch.Tensor, group, inputs: dict, router: bool):
    """The stock chain from this rank's partial: all-reduce (FP32 sum in rank
    order, BF16) and FlashInfer's dual norm."""
    from flashinfer.norm.kernels.gated_residual_next_rmsnorm import run

    parts = [torch.empty_like(partial) for _ in range(TP)]
    torch.distributed.all_gather(parts, partial, group=group)
    total = torch.zeros(partial.shape, device="cuda")
    for part in parts:
        total += part.float()
    stream = inputs["stream"]
    new_stream = torch.empty_like(stream)
    out = torch.empty(stream.shape, dtype=torch.bfloat16, device="cuda")
    router_input = torch.empty_like(stream) if router else None
    run(
        total.bfloat16(),
        stream,
        inputs["alpha"],
        inputs["beta"],
        inputs["pre_weight"],
        POST_EPS,
        NEXT_EPS,
        GAIN,
        new_stream,
        out,
        router_input=router_input,
    )
    return out, new_stream, router_input


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _report(rank: int, what: str, got, want) -> None:
    for name, a, b in zip(("out", "new_stream", "router"), got, want):
        if a is None:
            continue
        bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        equal = (a.view(bits) == b.view(bits)).float().mean().item()
        diff = (a.float() - b.float()).abs()
        rel = (diff / b.float().abs().clamp_min(1e-2)).max().item()
        assert torch.allclose(a.float(), b.float(), rtol=0.05, atol=0.05), what
        if rank == 0:
            print(
                f"{what} {name}: {equal:.4f} bitwise, max abs {diff.max().item():.3g}"
                f", max rel {rel:.3g}"
            )


def _check_all_reduce(boundary, gemm_for, group, rank: int) -> None:
    for step, m in enumerate((1, 8, 24, 64, 256)):
        attn, inputs = _boundary_inputs(m, rank, seed=m)
        generator = torch.Generator(device="cuda").manual_seed(7 + rank)
        weight = (
            torch.randn(HIDDEN, K_O_PROJ, generator=generator, device="cuda") * 0.02
        ).bfloat16()
        for router in (False, True):
            _late(rank, step + router)
            handle = boundary.o_proj(attn, weight)
            got = boundary.dual_norm(
                handle,
                inputs["stream"],
                inputs["alpha"],
                inputs["beta"],
                inputs["pre_weight"],
                post_eps=POST_EPS,
                next_eps=NEXT_EPS,
                next_gain_center=GAIN,
                router=router,
            )
            own = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
            gemm_for(m)(attn, weight, own)
            want = _stock_dual_norm(own, group, inputs, router)
            torch.accelerator.synchronize()
            for a, b in zip(got, want):
                assert (a is None) == (b is None)
                if a is not None:
                    assert _equal(a, b), f"M={m} router={router}"
            stock = _stock_dual_norm(torch.mm(attn, weight.t()), group, inputs, router)
            _report(
                rank, f"o_proj M={m} router={router} vs torch.mm chain:", got, stock
            )


def _check_gated(boundary, gemm_for, group, rank: int) -> None:
    """``o_proj(..., gate=og)``: the o-norm gate folded into the GEMM (small M)
    or run by FlashInfer first, bitwise against FlashInfer's gate, the GEMM
    and the stock all-reduce + dual norm."""
    from flashinfer.norm import rmsnorm_sigmoid_gate

    for step, m in enumerate((1, 8, 24, 64)):
        attn, inputs = _boundary_inputs(m, rank, seed=50 + m)
        generator = torch.Generator(device="cuda").manual_seed(70 + rank)
        weight = (
            torch.randn(HIDDEN, K_O_PROJ, generator=generator, device="cuda") * 0.02
        ).bfloat16()
        gate = torch.randn(m, K_O_PROJ, generator=generator, device="cuda").bfloat16()
        gated = attn.clone()
        rmsnorm_sigmoid_gate(gated.view(m, -1, 64), gate.view(m, -1, 64), 1e-5)
        _late(rank, step)
        handle = boundary.o_proj(attn, weight, gate=gate)
        got = boundary.dual_norm(
            handle,
            inputs["stream"],
            inputs["alpha"],
            inputs["beta"],
            inputs["pre_weight"],
            post_eps=POST_EPS,
            next_eps=NEXT_EPS,
            next_gain_center=GAIN,
        )
        own = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        gemm_for(m)(gated, weight, own)
        want = _stock_dual_norm(own, group, inputs, False)
        torch.accelerator.synchronize()
        for a, b in zip(got, want):
            if a is not None:
                assert _equal(a, b), f"gated M={m}"
        if rank == 0:
            print(f"gated o_proj M={m}: bitwise equal to the gate + stock chain")


def _check_all_gather(group, rank: int) -> None:
    """Pooler and unpooler shards published into FlashInfer's workspaces."""
    import flashinfer.comm.mnnvl_cutedsl_ag as ag

    from vllm.models.muse_spark.nvidia.specialized_decode.boundary.boundary import (
        AllGatherPublisher,
        PublishingGemm,
    )

    for name, n, k, hidden in (
        ("unpooler", 2048, 4096, 8192),
        ("pooler", 1024, 8192, 4096),
    ):
        if name == "unpooler":
            make = ag.MNNVLCuteDSLAllGatherDualNormWorkspace
        else:
            make = ag.MNNVLCuteDSLAllGatherNormQuantWorkspace
        workspaces = [make(TP, rank, CAPACITY, hidden, group=group) for _ in range(2)]
        for workspace in workspaces:
            if name == "unpooler":
                workspace.compile(
                    post_eps=POST_EPS, next_eps=NEXT_EPS, next_gain_center=GAIN
                )
            else:
                workspace.compile(eps=NEXT_EPS, gain_center=GAIN)
        publisher = AllGatherPublisher(workspaces[0], group, k=k)
        generator = torch.Generator(device="cuda").manual_seed(11 + rank)
        weight = (
            torch.randn(n, k, generator=generator, device="cuda") * 0.02
        ).bfloat16()
        shared = torch.Generator(device="cuda").manual_seed(5)
        norm_weight = (
            torch.randn(hidden, generator=shared, device="cuda") * 0.1
        ).bfloat16()
        alpha = torch.rand(hidden, generator=shared, device="cuda") + 0.5
        beta = torch.rand(hidden, generator=shared, device="cuda")
        scale = torch.tensor([3.0], device="cuda")
        for step, m in enumerate((1, 8, 64, 256)):
            x = torch.randn(m, k, generator=generator, device="cuda").bfloat16()
            residual = torch.randn(m, hidden, generator=shared, device="cuda")
            shard = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            PublishingGemm(
                n,
                k,
                None,
                max_tokens=m,
                store_c=True,
                tactics={m: publisher.gemm.tactic(m)},
            )(x, weight, shard)
            _late(rank, step)
            handle = publisher(x, weight)
            if name == "unpooler":
                got = publisher.gated_residual_next_rmsnorm(
                    handle,
                    residual,
                    alpha,
                    beta,
                    norm_weight,
                    post_eps=POST_EPS,
                    next_eps=NEXT_EPS,
                    next_gain_center=GAIN,
                )
                want = ag.all_gather_gated_residual_next_rmsnorm(
                    shard,
                    residual,
                    alpha,
                    beta,
                    norm_weight,
                    workspaces[1],
                    post_eps=POST_EPS,
                    next_eps=NEXT_EPS,
                    next_gain_center=GAIN,
                )
            else:
                got = publisher.rmsnorm_fp4_quantize(
                    handle, norm_weight, scale, eps=NEXT_EPS, gain_center=GAIN
                )
                want = ag.all_gather_rmsnorm_fp4_quantize(
                    shard,
                    norm_weight,
                    scale,
                    workspaces[1],
                    eps=NEXT_EPS,
                    gain_center=GAIN,
                )
            torch.accelerator.synchronize()
            for a, b in zip(got, want):
                assert _equal(a, b), f"{name} M={m}"
            if rank == 0:
                print(f"{name} M={m}: bitwise equal to FlashInfer's publish")


def _check_replays(boundary, gemm_for, group, rank: int, replays: int) -> None:
    """Boundaries at several sizes (one- and two-shot) in one graph, replayed
    with fresh inputs; afterwards every mailbox slot is armed again."""
    sizes = (8, 24, 64, 200)
    generator = torch.Generator(device="cuda").manual_seed(3 + rank)
    weights = [
        (
            torch.randn(HIDDEN, K_O_PROJ, generator=generator, device="cuda") * 0.02
        ).bfloat16()
        for _ in sizes
    ]
    static = [_boundary_inputs(m, rank, seed=m) for m in sizes]
    results: list[tuple] = []

    def step():
        results.clear()
        for (attn, inputs), weight in zip(static, weights):
            handle = boundary.o_proj(attn, weight)
            results.append(
                boundary.dual_norm(
                    handle,
                    inputs["stream"],
                    inputs["alpha"],
                    inputs["beta"],
                    inputs["pre_weight"],
                    post_eps=POST_EPS,
                    next_eps=NEXT_EPS,
                    next_gain_center=GAIN,
                    router=attn.shape[0] == 64,
                )
            )

    step()
    torch.accelerator.synchronize()
    torch.distributed.barrier(group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for replay in range(replays):
        for i, m in enumerate(sizes):
            attn, inputs = _boundary_inputs(m, rank, seed=replay * 31 + m)
            static[i][0].copy_(attn)
            static[i][1]["stream"].copy_(inputs["stream"])
        _late(rank, replay)
        graph.replay()
        if replay % 10 == 0 or replay == replays - 1:
            for (attn, inputs), weight, got in zip(static, weights, results):
                m = attn.shape[0]
                own = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
                gemm_for(m)(attn, weight, own)
                want = _stock_dual_norm(own, group, inputs, got[2] is not None)
                for a, b in zip(got, want):
                    if a is not None:
                        assert _equal(a, b), f"replay {replay} M={m}"
    torch.accelerator.synchronize()
    for mailbox in (boundary.mailbox, boundary.scatter, boundary.gather):
        if mailbox is not None:
            assert (mailbox.buffer.view(torch.int16) == -32768).all(), "not re-armed"
    if rank == 0:
        print(f"{replays} graph replays bitwise; every mailbox slot re-armed")


def _worker(rank: int, port: int, replays: int, skip_path: str) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        init_test_distributed_environment(TP, 1, rank, str(port), local_rank=rank)
        from flashinfer.comm.mnnvl import is_multicast_supported

        supported = torch.tensor([int(is_multicast_supported(rank))])
        torch.distributed.all_reduce(
            supported, torch.distributed.ReduceOp.MIN, get_tp_group().cpu_group
        )
        if not supported.item():
            if rank == 0:
                with open(skip_path, "w") as f:
                    f.write("no NVLink multicast")
            cleanup_dist_env_and_memory()
            return
        from vllm.models.muse_spark.nvidia.specialized_decode.boundary.all_reduce_dual_norm import (  # noqa: E501
            DualNormConstants,
        )
        from vllm.models.muse_spark.nvidia.specialized_decode.boundary.boundary import (
            BoundaryAllReduce,
            PublishingGemm,
        )

        group = get_tp_group().device_group
        boundary = BoundaryAllReduce(
            group,
            k=K_O_PROJ,
            max_tokens=CAPACITY,
            constants=[
                DualNormConstants(POST_EPS, NEXT_EPS, GAIN, router=router)
                for router in (False, True)
            ],
        )
        boundary.precompile()
        gemms: dict = {}

        def gemm_for(m: int) -> PublishingGemm:
            """A store-only GEMM with the publishing one's tactic: the same
            partials, bit for bit."""
            if m not in gemms:
                gemms[m] = PublishingGemm(
                    HIDDEN,
                    K_O_PROJ,
                    None,
                    max_tokens=CAPACITY,
                    store_c=True,
                    tactics={m: boundary.gemm.tactic(m)},
                )
            return gemms[m]

        _check_all_reduce(boundary, gemm_for, group, rank)
        _check_gated(boundary, gemm_for, group, rank)
        _check_all_gather(group, rank)
        _check_replays(boundary, gemm_for, group, rank, replays)
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


def run(replays: int) -> str | None:
    """Run the checks on four GPUs; returns why they were skipped, if so."""
    from vllm.utils.network_utils import get_open_port

    with tempfile.TemporaryDirectory() as tmp:
        skip_path = os.path.join(tmp, "skipped")
        torch.multiprocessing.spawn(
            _worker, args=(get_open_port(), replays, skip_path), nprocs=TP
        )
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                return f.read()
    return None


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="needs SM100"
)
def test_boundary_numerics():
    if torch.accelerator.device_count() < TP:
        pytest.skip("needs four GPUs")
    skipped = run(REPLAYS)
    if skipped:
        pytest.skip(skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replays", type=int, default=REPLAYS)
    skipped = run(parser.parse_args().replays)
    print(f"skipped: {skipped}" if skipped else "passed")
