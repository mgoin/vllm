# SPDX-License-Identifier: Apache-2.0
"""Time the LUT-B MoE GEMMs at decode sizes and report HBM bandwidth.

Measured case: one grouped GEMM (ops.lut_b_moe_gemm vs
ops.lut_b_moe_gemm_stream), MXFP8 activations x LUT-B weights,
tcgen05.mma.kind::mxf8f6f4.block_scale.decompress::lut::b.block32,
uniform random top-k routing, cold L2 (weights rotated through enough copies
to exceed L2, plus FlashInfer's cold-L2 flush when available).

Bandwidth counts the weight tiles of the experts actually hit plus the
activation and output bytes, divided by kernel time.
"""

import argparse
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    lut_b_sm107_pair_tiles,
)

TILE_BYTES = 6912


def make_tiles(
    experts: int, n: int, k: int, copies: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Random tiles with unit scales and finite codebooks, in both layouts."""
    out = []
    for _ in range(copies):
        t = torch.randint(
            0,
            256,
            (experts, n // 128, k // 128, TILE_BYTES),
            dtype=torch.uint8,
            device="cuda",
        )
        t[..., 6144:6656] = 127  # unit UE8M0 scales
        t[..., 6656:] &= 0x77  # finite E4M3 codebook entries
        out.append((t, lut_b_sm107_pair_tiles(t)))
    return out


def timer():
    try:
        from flashinfer.testing import bench_gpu_time

        def run(fn, iters):
            try:
                times = bench_gpu_time(
                    fn,
                    dry_run_iters=5,
                    repeat_iters=iters,
                    enable_cupti=True,
                    cold_l2_cache=True,
                )
            except TypeError:
                times = bench_gpu_time(
                    fn, dry_run_iters=5, repeat_iters=iters, enable_cupti=True
                )
            return statistics.median(times) * 1e3  # ms -> us

        return run, "flashinfer-cupti"
    except ImportError:
        flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")

        def run(fn, iters):
            times = []
            for _ in range(iters):
                flush.zero_()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e) * 1e3)
            return statistics.median(times)

        return run, "cuda-events+flush"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--shapes", default="3072x2048,2048x1536", help="NxK list")
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256")
    p.add_argument("--peak-tbs", type=float, default=19.2)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--copies", type=int, default=0, help="0: enough for 1 GB")
    p.add_argument("--split-k", default="0", help="list; 0 = auto")
    args = p.parse_args()
    run, timer_name = timer()
    torch.manual_seed(0)
    print(
        f"timer={timer_name} device={torch.cuda.get_device_name()} "
        f"sms={torch.cuda.get_device_properties(0).multi_processor_count}"
    )
    print("shape,tokens,experts_hit,split,kernel,us,weight_MB,TB/s,%peak")
    for shape in args.shapes.split(","):
        n, k = map(int, shape.split("x"))
        per_copy = args.experts * n * k * TILE_BYTES // (128 * 128)
        copies = args.copies or max(2, (1 << 30) // (2 * per_copy) + 1)
        tiles = make_tiles(args.experts, n, k, copies)
        for m in map(int, args.tokens.split(",")):
            topk_ids = torch.stack(
                [
                    torch.randperm(args.experts, device="cuda")[: args.top_k]
                    for _ in range(m)
                ]
            ).to(torch.int32)
            topk_w = torch.rand(m, args.top_k, device="cuda").flatten()
            a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
            a_s = torch.full((m, k // 32), 127, dtype=torch.uint8, device="cuda")
            routing = moe_align_block_size(topk_ids, 128, args.experts)
            routing32 = moe_align_block_size(topk_ids, 32, args.experts)
            c = torch.empty(m * args.top_k, n, device="cuda", dtype=torch.bfloat16)
            hit = topk_ids.unique().numel()
            wbytes = hit * (n // 128) * (k // 128) * TILE_BYTES
            total = wbytes + a.numel() + a_s.numel() + c.numel() * 2
            state = {"i": 0}  # rotates through the weight copies

            def simple(
                a=a, a_s=a_s, c=c, w=topk_w, r=routing, tiles=tiles, state=state
            ):
                t = tiles[state["i"] % len(tiles)][0]
                state["i"] += 1
                ops.lut_b_moe_gemm(a, a_s, t, c, w, *r, args.top_k)

            variants = [("simple", 1, simple)]
            for sk in map(int, args.split_k.split(",")):

                def stream(
                    sk=sk,
                    a=a,
                    a_s=a_s,
                    c=c,
                    w=topk_w,
                    r=routing32,
                    tiles=tiles,
                    state=state,
                ):
                    t = tiles[state["i"] % len(tiles)][1]
                    state["i"] += 1
                    ops.lut_b_moe_gemm_stream(
                        a, a_s, t, c, w, *r, args.top_k, split_k=sk
                    )

                variants.append(("stream", sk, stream))
            for name, sk, fn in variants:
                us = run(fn, args.iters)
                tbs = total / us / 1e6
                print(
                    f"{n}x{k},{m},{hit},{sk},{name},{us:.2f},"
                    f"{wbytes / 1e6:.1f},{tbs:.2f},{100 * tbs / args.peak_tbs:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
