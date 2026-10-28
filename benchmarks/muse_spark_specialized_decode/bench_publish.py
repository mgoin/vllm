# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
"""Isolated cold-cache tactic sweep of the boundary publishing GEMM (one GPU).

The kernel publishes into the mailbox through a multimem store, so it cannot
run alone on one GPU; this harness runs it with its local C store in place of
the mailbox (the epilogue work and the 8-column fragments are the same, only
the store is ``st.global`` instead of ``multimem.st``) and rotates the weight
over copies whose total footprint is far past L2, so every launch streams cold
weights the way a layer does. Timed with CUDA-graph replay, events around the
graph, one launch per sample, median over samples.

    python bench_publish.py --grid current --m 8           # what the layer runs
    python bench_publish.py --grid current --m 8 --gate    # o_proj's gated kernel
    python bench_publish.py --grid sweep --m 8             # every measured tactic
    python bench_publish.py --grid current --m 8 --reserve-kb 8  # 12 ring stages

``--reserve-kb`` shrinks the kernel's ``_SMEM_BYTES`` reservation, which buys a
deeper weight ring (11 -> 12 stages at 8 rows); ``--no-prefetch`` drops the
``prefetch_l2`` of the K tail.

Measured on B300 (cold, 8 rows), against the tactic the table selects for them:

    kernel     bytes   tactic                           us    TB/s   CTAs
    o_proj     33.5MB  bm64 bn8 sk1   bk128 stages11  7.04   4.77   128
      (gated, the kernel the layer actually runs)    8.26   4.06   128
    pooler     16.8MB  bm64 bn8 sk4   bk128 stages11  4.30   3.90    64
    unpooler   16.8MB  bm128 bn16 sk4 bk64  stages11  3.95   4.24    64

Aggregate 15.3 us (4.39 TB/s) ungated, 16.6 us (4.05 TB/s) with o_proj gated,
against a box cold-read ceiling of 6.14 TB/s (``sum`` over 4 GiB; 5.3 TB/s at
0.5 GiB, because a 7 us kernel pays ~1.3 us of ramp). No entry in ``TACTICS``
beat the current one in ~50 measured configurations, so the table was left as
it was:

* the pooler and unpooler have 16 column tiles, i.e. 64 CTAs, and a CTA streams
  60-73 GB/s; every split-K variant that adds CTAs (pooler sk8, unpooler sk4 or
  sk8) loses 20-45% to the cluster exchange, which costs 0.3-2.5 us per extra
  slice at 8 rows;
* o_proj already has 128 CTAs at 4.77 TB/s (78% of the ceiling). bm=128/bk=64
  (128 CTAs, sk2) is 4.66; bm=128 (64 CTAs) is 3.8-3.9;
* in-flight bytes are not the limiter: a 12th ring stage (+9%) changes nothing
  (o_proj 7.04 -> 7.02-7.15, pooler 4.30 -> 4.52, unpooler 3.95 -> 4.22);
* prefetch_l2 is worth +1.6% on o_proj and -8% on pooler/unpooler; bk=64 is
  better for the unpooler's 128-row tiles and worse everywhere else;
* the gate fold costs 1.22 us of o_proj (17%): it stages the raw activation and
  gate and gates each K tile inside the ring, on the MMA's critical path.
"""

import argparse
import statistics
import sys

import torch

from vllm.models.muse_spark.nvidia.specialized_decode.boundary import (
    publishing_gemm,
)
from vllm.models.muse_spark.nvidia.specialized_decode.boundary.boundary import (
    PublishingGemm,
)
from vllm.models.muse_spark.nvidia.specialized_decode.boundary.publishing_gemm import (
    GemmTactic,
    max_stages,
    validate_tactic,
)

SHAPES = {"o_proj": (8192, 2048), "pooler": (1024, 8192), "unpooler": (2048, 4096)}
# Far past L2 (B300: 126 MB), so a copy is evicted before it comes round again.
ROTATE_BYTES = 512 * 1024**2

# (bm, bn, split_k, bk, cf, ct, stages); stages=None takes the deepest ring.
GRIDS = {
    # The buckets `default_tactic` selects for 8 rows: o_proj and pooler from
    # their (8, ...) entry, unpooler from its (16, ...) one -- buckets are
    # "m <= max_m", so nothing falls through to the generic tactic.
    "current": {
        "o_proj": [(64, 8, 1, 128, 1, 1, None)],
        "pooler": [(64, 8, 4, 128, 1, 1, None)],
        "unpooler": [(128, 16, 4, 64, 1, 1, None)],
    },
    "sweep": {
        "o_proj": [
            (64, 8, 1, 128, 1, 1, None),
            (64, 8, 1, 128, 1, 1, 8),
            (64, 8, 1, 128, 1, 1, 4),
            (64, 16, 1, 128, 1, 1, None),
            (64, 8, 1, 64, 1, 1, None),
            (64, 8, 1, 256, 1, 1, None),
            (64, 8, 1, 512, 1, 1, None),
            (64, 8, 2, 128, 1, 1, None),
            (64, 8, 2, 128, 1, 1, 6),
            (64, 8, 2, 128, 1, 1, 4),
            (64, 8, 2, 256, 1, 1, None),
            (64, 8, 4, 128, 1, 1, None),
            (64, 8, 4, 128, 1, 1, 11),
            (128, 16, 1, 128, 1, 1, None),
            (128, 16, 1, 64, 1, 1, None),
            (128, 16, 2, 128, 1, 1, None),
            (128, 16, 2, 64, 1, 1, None),
        ],
        "pooler": [
            (64, 8, 4, 128, 1, 1, None),
            (64, 8, 4, 128, 1, 1, 16),
            (64, 8, 4, 128, 1, 1, 6),
            (64, 8, 4, 256, 1, 1, None),
            (64, 8, 4, 512, 1, 1, None),
            (64, 8, 4, 64, 1, 1, None),
            (64, 8, 2, 128, 1, 1, None),
            (64, 8, 2, 256, 1, 1, None),
            (64, 8, 2, 64, 1, 1, None),
            (64, 8, 1, 128, 1, 1, None),
            (64, 8, 8, 128, 1, 1, None),
            (64, 8, 8, 256, 1, 1, None),
            (64, 8, 8, 512, 1, 1, None),
            (64, 8, 8, 64, 1, 1, None),
            (128, 16, 8, 128, 1, 1, None),
            (128, 16, 8, 64, 1, 1, None),
            (128, 16, 8, 256, 1, 1, None),
            (128, 16, 4, 64, 1, 1, None),
            (128, 16, 2, 128, 1, 1, None),
        ],
        "unpooler": [
            (128, 16, 4, 64, 1, 1, None),
            (128, 16, 4, 64, 1, 1, 8),
            (128, 16, 4, 128, 1, 1, None),
            (128, 16, 2, 64, 1, 1, None),
            (128, 16, 8, 64, 1, 1, None),
            (128, 16, 8, 128, 1, 1, None),
            (128, 16, 1, 128, 1, 1, None),
            (64, 8, 2, 128, 1, 1, None),
            (64, 8, 2, 64, 1, 1, None),
            (64, 8, 4, 128, 1, 1, None),
            (64, 8, 4, 64, 1, 1, None),
            (64, 8, 4, 256, 1, 1, None),
            (64, 8, 2, 256, 1, 1, None),
            (64, 8, 1, 128, 1, 1, None),
            (64, 8, 8, 128, 1, 1, None),
        ],
    },
}


def _tactic(n: int, k: int, entry):
    bm, bn, split_k, bk, cf, ct, stages = entry
    if stages is None:
        stages = max_stages(bm, bn, bk)
    tactic = GemmTactic(bm, bn, split_k, stages, bk, cf, ct)
    validate_tactic(tactic, n, k)
    return tactic


def run(shape, m, entries, reps, reserve_kb, gate=None, no_prefetch=False):
    publishing_gemm._SMEM_BYTES = publishing_gemm.utils.get_smem_capacity_in_bytes(
        "sm_100"
    ) - reserve_kb * 1024
    n, k = SHAPES[shape]
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = (torch.randn(n, k, generator=generator, device="cuda") * 0.02).bfloat16()
    x = torch.randn(m, k, generator=generator, device="cuda").bfloat16()
    gate_arg, reference = None, x
    if gate is not None:
        from flashinfer.norm import rmsnorm_sigmoid_gate

        reference = x.clone()
        rmsnorm_sigmoid_gate(reference.view(m, -1, 64), gate.view(m, -1, 64), 1e-5)
        gate_arg = gate
    ref = reference.float() @ weight.float().t()
    stock = torch.mm(reference, weight.t())
    stock_error = (stock.float() - ref).abs().max().item()
    copies = max(2, -(-ROTATE_BYTES // weight.numel() // 2))
    weights = [weight.clone() for _ in range(copies)]
    outs = [
        torch.empty(m, n, dtype=torch.bfloat16, device="cuda") for _ in range(copies)
    ]
    weight_bytes = weight.numel() * 2
    if no_prefetch:
        original = publishing_gemm.PublishingGemmKernel.__init__

        def patched(self, *args, **kwargs):
            kwargs["prefetch_l2"] = False
            original(self, *args, **kwargs)

        publishing_gemm.PublishingGemmKernel.__init__ = patched
    rows = []
    for entry in entries:
        try:
            tactic = _tactic(n, k, entry)
        except ValueError as error:
            print(f"{shape:9s} {entry} REJECTED {error}")
            continue
        gemm = PublishingGemm(
            n, k, None, max_tokens=m, store_c=True, tactics={m: tactic}
        )
        try:
            gemm(x, weight, outs[0], gate=gate_arg)
            torch.accelerator.synchronize()
        except Exception as error:  # noqa: BLE001
            print(f"{shape:9s} {tactic} FAILED {error}")
            continue
        error = (outs[0].float() - ref).abs().max().item()
        equal = (outs[0].view(torch.int16) == stock.view(torch.int16)).float().mean()
        assert not outs[0].isnan().any(), "nan"
        # Both round one FP32 sum to BF16; only the accumulation order differs.
        assert error <= 1.25 * stock_error + 1e-6, f"{tactic} wrong: {error}"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for w, out in zip(weights, outs):
                gemm(x, w, out, gate=gate_arg)
        times = []
        for _ in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / copies)
        us = statistics.median(times)
        ctas = n // tactic.bm * tactic.split_k * (-(-m // tactic.bn))
        rows.append((us, tactic, ctas, error, equal))
    rows.sort(key=lambda row: row[0])
    for us, tactic, ctas, error, equal in rows:
        print(
            f"{shape:9s} m={m:<3d} res={reserve_kb:<2d} bm={tactic.bm:<4d} "
            f"bn={tactic.bn:<4d} sk={tactic.split_k:<2d} bk={tactic.bk:<5d} "
            f"cf={tactic.cf} ct={tactic.ct} stages={tactic.stages:<3d} "
            f"ctas={ctas:<4d} {us:8.2f} us {weight_bytes / us / 1e6:6.2f} TB/s "
            f"err={error:.3g} {equal:.3f} bitwise"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="o_proj,pooler,unpooler")
    parser.add_argument("--m", type=int, default=8, help="rows (tokens)")
    parser.add_argument("--grid", default="current", choices=sorted(GRIDS))
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--reserve-kb", type=int, default=16)
    parser.add_argument("--gate", action="store_true", help="o_proj's gated kernel")
    parser.add_argument("--no-prefetch", action="store_true")
    args = parser.parse_args()
    print(
        f"GPU {torch.cuda.get_device_name()} L2 "
        f"{torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MB"
    )
    for shape in args.cases.split(","):
        gate = None
        if args.gate and shape == "o_proj":
            generator = torch.Generator(device="cuda").manual_seed(7)
            gate = torch.randn(
                args.m, SHAPES[shape][1], generator=generator, device="cuda"
            ).bfloat16()
        run(
            shape,
            args.m,
            GRIDS[args.grid][shape],
            args.reps,
            args.reserve_kb,
            gate,
            args.no_prefetch,
        )


if __name__ == "__main__":
    sys.exit(main())
