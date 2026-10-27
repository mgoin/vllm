# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank numerics of the Muse Spark MoE tail at TP4.

Each rank gets its own permuted GEMM2 buffer (its own row order, NaN padding,
a missing route) of the same tokens. The tail (one-shot finalize + publishing
unpooler GEMM, consumed by FlashInfer's unchanged all-gather dual norm) runs
against the stock chain on identical inputs: FlashInfer's
``tp_moe_finalize_rmsnorm``, ``torch.mm`` with the unpooler shard and
FlashInfer's own all-gather publish. For M in 1, 2, 8, 16, 32 and 64 rows:

* ``e`` against the stock finalize (bitwise fraction, max relative error);
* the boundary outputs bitwise against the stock all-gather of the same ``e``
  through a store-only GEMM of the same tactic (so the publish is exact), and
  against the full stock chain (bitwise fraction, max relative error);
* one rank late each step (a different one), so a missing Lamport wait reads
  stale data;
* one CUDA graph of tails at several sizes, with the stock finalize captured
  beside them, replayed ``REPLAYS`` times with fresh inputs and a late rank;
  mismatches are counted on the device; afterwards every peer slot is
  re-armed and every generation counter agrees;
* ``PublishingRoutedExperts`` (random NVFP4 experts per rank) at 1, 2, 4, 8 and
  16 rows, the layer parities rotating: its GEMM2 rows bitwise those of
  ``MuseRoutedExperts``, and the tail on the rows it published bitwise the
  tail that publishes them itself (``e`` and the boundary outputs); then a
  CUDA graph of three such layers replayed ``REPLAYS`` times with fresh
  inputs and a late rank, checked the same way on the device; and degenerate
  routing (ids -1, 256, 1000 and INT_MIN, repeated experts, NaN input scales)
  in that mode, which must finish within a watchdog timeout.

Needs four SM100-family GPUs with NVLink multicast; about 3 minutes. For a
race stress run, run this file as a module with ``--replays 10000``.
"""

import argparse
import faulthandler
import os
import tempfile

import pytest
import torch

from vllm.platforms import current_platform

TP = 4
HIDDEN = 4096
BOUNDARY = 8192
TOP_K = 8
MAX_TOKENS = 64
SIZES = (1, 2, 8, 16, 32, 64)
GRAPH_SIZES = (1, 8, 16, 64)
REPLAYS = 1000
LATE_CYCLES = 20_000
EPS, POST_EPS, NEXT_EPS, GAIN = 1e-8, 1e-8, 1e-5, 1.0


def _late(rank: int, step: int) -> None:
    if rank == step % TP:
        torch.cuda._sleep(LATE_CYCLES)


class Inputs:
    """One rank's MoE tail inputs for ``m`` tokens; ``refill`` draws new
    values in place (the route map stays)."""

    def __init__(self, m: int, rank: int, seed: int):
        self.m, self.rank = m, rank
        own = torch.Generator(device="cuda").manual_seed(seed * 131 + rank)
        num_rows = m * TOP_K + 9
        order = torch.randperm(num_rows, generator=own, device="cuda")[: m * TOP_K]
        self.idx = order.view(m, TOP_K).int()
        self.idx[0, 5] = -1
        self.rows = torch.full(
            (num_rows, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        self.weights = torch.empty(m, TOP_K, device="cuda")
        self.residual = torch.empty(m, BOUNDARY, device="cuda")
        self.refill(seed)

    def refill(self, seed: int) -> None:
        own = torch.Generator(device="cuda").manual_seed(seed * 7919 + self.rank)
        valid = self.idx[self.idx >= 0].long()
        self.rows[valid] = torch.randn(
            valid.numel(), HIDDEN, generator=own, device="cuda"
        ).bfloat16()
        shared = torch.Generator(device="cuda").manual_seed(seed * 7919 + 1000)
        self.weights.copy_(torch.rand(self.m, TOP_K, generator=shared, device="cuda"))
        self.residual.copy_(
            torch.randn(self.m, BOUNDARY, generator=shared, device="cuda") * 3
        )

    def unfinalized(self):
        from vllm.model_executor.layers.fused_moe.moe_output import (
            UnfinalizedMoEOutput,
        )

        return UnfinalizedMoEOutput(
            gemm2_permuted=self.rows,
            expert_weights=self.weights,
            expanded_idx_to_permuted_idx=self.idx,
        )


class Model:
    """Weights shared by the ranks: the post-expert norm, this rank's unpooler
    shard and the boundary's gates and next-norm weight."""

    def __init__(self, rank: int):
        shared = torch.Generator(device="cuda").manual_seed(5)
        self.norm_weight = (
            torch.randn(HIDDEN, generator=shared, device="cuda") * 0.3
        ).bfloat16()
        unpooler = torch.randn(BOUNDARY, HIDDEN, generator=shared, device="cuda")
        shard = BOUNDARY // TP
        self.unpooler_shard = (unpooler[rank * shard : (rank + 1) * shard] * 0.02).to(
            torch.bfloat16
        )
        self.alpha = torch.rand(BOUNDARY, generator=shared, device="cuda") + 0.5
        self.beta = torch.rand(BOUNDARY, generator=shared, device="cuda")
        self.next_weight = (
            torch.randn(BOUNDARY, generator=shared, device="cuda") * 0.1
        ).bfloat16()


def _boundary_args(model: Model, inputs: Inputs) -> tuple:
    return (inputs.residual, model.alpha, model.beta, model.next_weight)


_NORM = dict(post_eps=POST_EPS, next_eps=NEXT_EPS, next_gain_center=GAIN)


def _stock_finalize(inputs: Inputs, model: Model, out: torch.Tensor) -> None:
    from flashinfer.comm.tp_moe_finalize_rmsnorm import tp_moe_finalize_rmsnorm

    tp_moe_finalize_rmsnorm(
        inputs.rows,
        inputs.weights,
        inputs.idx,
        model.norm_weight,
        out,
        eps=EPS,
        gain_center=GAIN,
        weight_before_norm=True,
    )


def _stock_boundary(branch, inputs: Inputs, model: Model, workspace):
    import flashinfer.comm.mnnvl_cutedsl_ag as ag

    return ag.all_gather_gated_residual_next_rmsnorm(
        branch, *_boundary_args(model, inputs), workspace, **_NORM
    )


def _tail(tail, inputs: Inputs, model: Model):
    handle = tail(
        inputs.unfinalized(),
        model.norm_weight,
        model.unpooler_shard,
        eps=EPS,
        gain_center=GAIN,
    )
    return tail.publisher.gated_residual_next_rmsnorm(
        handle, *_boundary_args(model, inputs), **_NORM
    )


def _mismatches(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return (a.view(bits) != b.view(bits)).sum()


def _compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    equal = (a.view(bits) == b.view(bits)).float().mean().item()
    rel = (a.float() - b.float()).abs() / b.float().abs().clamp_min(1e-2)
    return equal, rel.max().item()


def _check_chain(tail, gemm_for, workspace, model: Model, rank: int) -> None:
    for step, m in enumerate(SIZES):
        inputs = Inputs(m, rank, seed=m)
        stock_e = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        _stock_finalize(inputs, model, stock_e)
        stock_branch = torch.mm(stock_e, model.unpooler_shard.t())
        stock = _stock_boundary(stock_branch, inputs, model, workspace)

        _late(rank, step)
        got = _tail(tail, inputs, model)
        e = tail.e[:m].clone()
        branch = torch.empty(m, BOUNDARY // TP, dtype=torch.bfloat16, device="cuda")
        gemm_for(m)(e, model.unpooler_shard, branch)
        same_e = _stock_boundary(branch, inputs, model, workspace)
        torch.accelerator.synchronize()

        equal, rel = _compare(e, stock_e)
        assert not e.isnan().any()
        assert equal > 0.99 and rel < 0.02, f"M={m}: e {equal} bitwise, rel {rel}"
        for a, b in zip(got, same_e):
            assert _mismatches(a, b).item() == 0, f"M={m}"
        if rank == 0:
            print(f"M={m}: e vs stock finalize {equal:.5f} bitwise, max rel {rel:.3g}")
        for name, a, b in zip(("new_stream", "ffn_input"), got, stock):
            equal, rel = _compare(a, b)
            assert torch.allclose(a.float(), b.float(), rtol=0.05, atol=0.05)
            if rank == 0:
                print(f"  {name} vs stock chain {equal:.5f} bitwise, max rel {rel:.3g}")


def _check_replays(tail, gemm_for, workspace, model: Model, rank: int, replays: int):
    """Tails at several sizes in one graph beside the stock finalize and the
    stock all-gather of their own ``e``; fresh inputs every replay."""
    inputs = [Inputs(m, rank, seed=100 + m) for m in GRAPH_SIZES]
    stock_e = [
        torch.empty(i.m, HIDDEN, dtype=torch.bfloat16, device="cuda") for i in inputs
    ]
    branches = [
        torch.empty(i.m, BOUNDARY // TP, dtype=torch.bfloat16, device="cuda")
        for i in inputs
    ]
    mismatches = torch.zeros(3, dtype=torch.int64, device="cuda")
    outputs: list = []

    def step():
        outputs.clear()
        for i, e_ref, branch in zip(inputs, stock_e, branches):
            _stock_finalize(i, model, e_ref)
            got = _tail(tail, i, model)
            e = tail.e[: i.m]
            gemm_for(i.m)(e, model.unpooler_shard, branch)
            want = _stock_boundary(branch, i, model, workspace)
            outputs.append((e, e_ref, got, want))

    # Boundary mismatches and max deviation per (replay, size, output).
    log = torch.zeros(replays, len(GRAPH_SIZES), 2, 2, device="cuda")

    def count(replay: int):
        for size, (e, e_ref, got, want) in enumerate(outputs):
            close = (e.float() - e_ref.float()).abs() <= (
                0.02 * e_ref.float().abs() + 1e-3
            )
            mismatches[0] += (~close).sum()
            mismatches[1] += (e.view(torch.int16) != e_ref.view(torch.int16)).sum()
            for o, (a, b) in enumerate(zip(got, want)):
                off = _mismatches(a, b)
                mismatches[2] += off
                log[replay, size, o, 0] = off
                log[replay, size, o, 1] = (a.float() - b.float()).abs().max()

    step()
    torch.accelerator.synchronize()
    torch.distributed.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for replay in range(replays):
        for i in inputs:
            i.refill(1000 + replay)
        _late(rank, replay)
        graph.replay()
        count(replay)
    torch.accelerator.synchronize()
    total = sum(i.m for i in inputs) * HIDDEN * replays
    bad, flips, boundary = mismatches.tolist()
    assert bad == 0, f"{bad} elements of e off the stock finalize"
    if boundary:
        for replay, size, o in (log[..., 0] > 0).nonzero().tolist():
            print(
                f"rank {rank} replay {replay} M={GRAPH_SIZES[size]} output {o}: "
                f"{int(log[replay, size, o, 0])} off, max |diff| "
                f"{log[replay, size, o, 1].item():.4g}"
            )
    assert boundary == 0, f"{boundary} boundary outputs off the stock all-gather"
    assert flips < 0.01 * total
    peers = [s for s in range(TP) if s != rank]
    armed = tail.box.buffer[:, peers].view(torch.int16) == -32768
    assert armed.all(), "peer slots not re-armed"
    generations = tail.finalize.generations
    assert (generations == generations[0]).all()
    if rank == 0:
        print(
            f"{replays} graph replays: e {1 - flips / total:.5f} bitwise to the "
            "stock finalize, boundary exact; every peer slot re-armed"
        )


def _build(seed: int) -> dict:
    from tests.models.muse_spark.nvidia.specialized_decode.test_expert_kernels import (
        _nvfp4,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.experts import (
        routed_experts_cute as k,
    )

    experts, inter = k.EXPERTS, k.INTER
    g = torch.Generator(device="cuda").manual_seed(seed)
    u8 = dict(dtype=torch.uint8, device="cuda")
    w13 = torch.empty(experts, 2 * inter, HIDDEN // 2, **u8)
    w13_scale = torch.empty(experts, 2 * inter, HIDDEN // 16, **u8)
    w2 = torch.empty(experts, HIDDEN, inter // 2, **u8)
    w2_scale = torch.empty(experts, HIDDEN, inter // 16, **u8)
    w13_scale_2 = torch.empty(experts, 2, device="cuda")
    w2_scale_2 = torch.empty(experts, device="cuda")
    for e in range(experts):
        for i in range(2):
            w = torch.randn(inter, HIDDEN, generator=g, device="cuda") * 0.02
            w13_scale_2[e, i] = w.abs().max() / (6 * 448)
            rows = slice(i * inter, (i + 1) * inter)
            w13[e, rows], w13_scale[e, rows] = _nvfp4(
                w.bfloat16(), (1 / w13_scale_2[e, i]).reshape(1)
            )
        w = torch.randn(HIDDEN, inter, generator=g, device="cuda") * 0.02
        w2_scale_2[e] = w.abs().max() / (6 * 448)
        w2[e], w2_scale[e] = _nvfp4(w.bfloat16(), (1 / w2_scale_2[e]).reshape(1))
    w13, w13_scale = reorder_w1w3_to_w3w1(w13, w13_scale.view(torch.float8_e4m3fn))
    w13, w13_scale, w2, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
        w13,
        w2,
        w13_scale,
        w2_scale.view(torch.float8_e4m3fn),
        hidden_size=HIDDEN,
        intermediate_size=inter,
        num_experts=experts,
        is_gated_activation=True,
    )
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


def _check_published_experts(group, publisher, gemm_for, model, rank, replays):
    """``PublishingRoutedExperts`` feeding the tail; see the module docstring."""
    from tests.models.muse_spark.nvidia.specialized_decode.test_expert_kernels import (
        inputs,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.experts.routed_experts import (  # noqa: E501
        MuseRoutedExperts,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.tail.experts import (
        PublishingRoutedExperts,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.tail.tail import MoeTail

    layer = _build(seed=11 + rank)
    weights = [layer[n] for n in ("w13", "w13_scale", "w2", "w2_scale")]
    weights += [layer[n] for n in ("g1_alphas", "g1_scale_c", "g2_alphas")]
    tail = MoeTail(group, publisher=publisher, max_tokens=16)
    tail.precompile()
    stock = MuseRoutedExperts(*weights)
    experts = PublishingRoutedExperts(*weights, target=tail.published_target, tp=TP)
    norm = dict(eps=EPS, gain_center=GAIN)

    def layer_step(case, parity, out, ref, e_ref):
        """Publishing experts and their tail, and beside them the stock experts
        and the self-publishing tail on the same rows."""
        m = case[0].shape[0]
        _late(rank, parity)
        stock(*case, ref)
        unfinalized = experts(*case, out, parity=parity)
        handle = tail(
            unfinalized,
            model.norm_weight,
            model.unpooler_shard,
            parity=parity,
            topk_ids=case[2],
            **norm,
        )
        e = tail.e[:m].clone()
        idx = unfinalized.expanded_idx_to_permuted_idx
        tail.finalize(ref, idx, case[3], model.norm_weight, e_ref, **norm)
        return handle, e

    for step, m in enumerate((1, 2, 4, 8, 16)):
        case = inputs(layer, m, "uniform", seed=500 + m)
        out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ref = torch.empty_like(out)
        e_ref = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        handle, e = layer_step(case, step % 3, out, ref, e_ref)
        tail.publisher.gated_residual_next_rmsnorm(
            handle,
            torch.zeros(m, BOUNDARY, device="cuda"),
            model.alpha,
            model.beta,
            model.next_weight,
            **_NORM,
        )
        torch.accelerator.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), f"M={m}"
        assert torch.equal(e.view(torch.int16), e_ref.view(torch.int16)), f"M={m}"
        if rank == 0:
            print(f"published experts M={m}: rows and e bitwise")

    # Degenerate routing (ids outside [0, 256), repeated experts, NaN input
    # scales) must finish, under a watchdog, and match the stock experts with
    # the self-publishing tail bit for bit (NaNs included).
    faulthandler.dump_traceback_later(300, exit=True)
    for step, m in enumerate((4, 16, 1)):
        case = inputs(layer, m, "uniform", seed=800 + m)
        ids = case[2]
        ids[0] = torch.tensor(
            [-1, 256, 1000, -(2**31), 5, 5, 5, 7], dtype=torch.int32, device="cuda"
        )
        if m > 1:
            ids[1] = torch.tensor([3] * 8, dtype=torch.int32, device="cuda")
            ids[m - 1, :4] = -1
            case[1][m - 1, :8] = 0x7F  # E4M3 NaN block scales
        out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ref = torch.empty_like(out)
        e_ref = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        handle, e = layer_step(case, (step + 2) % 3, out, ref, e_ref)
        tail.publisher.gated_residual_next_rmsnorm(
            handle,
            torch.zeros(m, BOUNDARY, device="cuda"),
            model.alpha,
            model.beta,
            model.next_weight,
            **_NORM,
        )
        torch.accelerator.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), f"M={m}"
        assert torch.equal(e.view(torch.int16), e_ref.view(torch.int16)), f"M={m}"
        assert not out[:4].any(), "routes to no expert must be zero rows"
        if rank == 0:
            print(f"degenerate routing M={m}: finished, rows and e bitwise")
    faulthandler.cancel_dump_traceback_later()

    sizes = (1, 8, 16)
    cases = [inputs(layer, m, "uniform", seed=600 + m) for m in sizes]
    outs = [
        torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for m in sizes
    ]
    refs = [torch.empty_like(o) for o in outs]
    e_refs = [
        torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda") for m in sizes
    ]
    residuals = [torch.randn(m, BOUNDARY, device="cuda") for m in sizes]
    branches = [
        torch.empty(m, BOUNDARY // TP, dtype=torch.bfloat16, device="cuda")
        for m in sizes
    ]
    mismatches = torch.zeros(3, dtype=torch.int64, device="cuda")
    results: list = []

    def graph_step():
        results.clear()
        for i, m in enumerate(sizes):
            handle, e = layer_step(cases[i], i, outs[i], refs[i], e_refs[i])
            got = tail.publisher.gated_residual_next_rmsnorm(
                handle,
                residuals[i],
                model.alpha,
                model.beta,
                model.next_weight,
                **_NORM,
            )
            gemm_for(m)(e_refs[i], model.unpooler_shard, branches[i])
            want = _stock_boundary(
                branches[i], _Residual(residuals[i]), model, gemm_for.workspace
            )
            results.append((outs[i], refs[i], e, e_refs[i], got, want))

    graph_step()
    torch.accelerator.synchronize()
    torch.distributed.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_step()
    generator = torch.Generator(device="cuda").manual_seed(rank)
    for replay in range(replays):
        for i, m in enumerate(sizes):
            fresh = inputs(layer, m, "uniform", seed=700 + replay * 3 + i)
            for buffer, value in zip(cases[i], fresh):
                buffer.copy_(value)
            residuals[i].normal_(generator=generator)
        _late(rank, replay)
        graph.replay()
        for out, ref, e, e_ref, got, want in results:
            mismatches[0] += _mismatches(out, ref)
            mismatches[1] += _mismatches(e, e_ref)
            for a, b in zip(got, want):
                mismatches[2] += _mismatches(a, b)
    torch.accelerator.synchronize()
    rows, es, boundary = mismatches.tolist()
    assert rows == 0 and es == 0 and boundary == 0, (rows, es, boundary)
    peers = [s for s in range(TP) if s != rank]
    assert (tail.published_box.buffer[:, peers].view(torch.int16) == -32768).all()
    if rank == 0:
        print(
            f"{replays} graph replays of published experts + tail: rows, e and "
            "boundary bitwise; every peer slot re-armed"
        )


class _Residual:
    """The boundary inputs ``_stock_boundary`` reads from an ``Inputs``."""

    def __init__(self, residual: torch.Tensor):
        self.residual = residual


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
        import flashinfer.comm.mnnvl_cutedsl_ag as ag
        from flashinfer.comm.tp_moe_finalize_rmsnorm import (
            init_tp_moe_finalize_rmsnorm,
        )

        from vllm.models.muse_spark.nvidia.specialized_decode.boundary.boundary import (
            AllGatherPublisher,
            PublishingGemm,
        )
        from vllm.models.muse_spark.nvidia.specialized_decode.tail.tail import MoeTail

        group = get_tp_group().device_group
        assert init_tp_moe_finalize_rmsnorm(group)
        workspaces = [
            ag.MNNVLCuteDSLAllGatherDualNormWorkspace(
                TP, rank, MAX_TOKENS, BOUNDARY, group=group
            )
            for _ in range(2)
        ]
        for workspace in workspaces:
            workspace.compile(**_NORM)
        publisher = AllGatherPublisher(workspaces[0], group, k=HIDDEN)
        tail = MoeTail(group, publisher=publisher, max_tokens=MAX_TOKENS)
        tail.precompile()
        gemms: dict = {}

        def gemm_for(m: int) -> PublishingGemm:
            """A store-only GEMM with the publishing one's tactic."""
            if m not in gemms:
                gemms[m] = PublishingGemm(
                    BOUNDARY // TP,
                    HIDDEN,
                    None,
                    max_tokens=MAX_TOKENS,
                    store_c=True,
                    tactics={m: tail.publisher.gemm.tactic(m)},
                )
            return gemms[m]

        model = Model(rank)
        _check_chain(tail, gemm_for, workspaces[1], model, rank)
        _check_replays(tail, gemm_for, workspaces[1], model, rank, replays)
        gemm_for.workspace = workspaces[1]
        _check_published_experts(group, publisher, gemm_for, model, rank, replays)
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
def test_tail_numerics():
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
