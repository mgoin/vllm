# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Development harness for the Muse Spark specialized decode; not a library.
# ruff: noqa
"""`vllm bench forward` wrapper for Muse Spark with fixture MoE routing.

Muse Spark's MoE names its gate `router`, so the stock balanced-routing fixture
(which looks for `module.gate`) silently falls back to real routing. This
wrapper teaches it the `router` attribute and adds a uniform-random mode.

Env:
  MUSE_ROUTING=random    (default) each token picks top_k distinct experts
                         uniformly at random (natural EP load imbalance).
  MUSE_ROUTING=balanced  stock fixture: equal load per EP rank.
  MUSE_ROUTING_DIAG=1    print, per MoE layer, the routing the dummy router
                         actually produces (before substitution) and whether
                         layer outputs are finite. Eager passes only.

Usage: torchrun --nproc-per-node=4 muse_bench.py bench forward <args>
"""

import os
import sys

import numpy as np
import torch

import vllm.benchmarks.forward.routing as routing
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

MODE = os.environ.get("MUSE_ROUTING", "random")
DIAG = os.environ.get("MUSE_ROUTING_DIAG", "0") == "1"

_orig_moe_layers = routing._moe_layers
_orig_balanced_ids = routing.balanced_topk_ids
_orig_install = routing.install_routing


def _moe_layers(model):
    layers = _orig_moe_layers(model)
    seen = {id(g) for _, g in layers}
    for module in model.modules():
        gate = getattr(module, "router", None)
        if isinstance(gate, GateLinear) and id(gate) not in seen:
            if routing._top_k(module) is not None:
                seen.add(id(gate))
                layers.append((module, gate))
    return layers


def _random_topk_ids(num_tokens, num_experts, top_k, ep_size, rng):
    return np.argsort(rng.random((num_tokens, num_experts)), axis=1)[:, :top_k]


def _pool_topk_ids(pool_size):
    """Every token draws top_k distinct experts from one random pool per layer,
    so a step touches at most `pool_size` distinct experts."""

    def ids(num_tokens, num_experts, top_k, ep_size, rng):
        assert top_k <= pool_size <= num_experts
        pool = rng.permutation(num_experts)[:pool_size]
        pick = np.argsort(rng.random((num_tokens, pool_size)), axis=1)[:, :top_k]
        return pool[pick]

    return ids


_diag_done: set = set()


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _install_diag(model):
    for idx, (module, gate) in enumerate(_moe_layers(model)):
        if getattr(gate, "_muse_diag", False):
            continue
        gate._muse_diag = True
        bias = getattr(module, "expert_bias", None)

        def hook(mod, inputs, output, idx=idx, bias=bias, module=module):
            if torch.cuda.is_current_stream_capturing() or _rank() != 0:
                return
            logits = (output[0] if isinstance(output, tuple) else output).float()
            key = (idx, logits.shape[0])
            if key in _diag_done:
                return
            _diag_done.add(key)
            mult = getattr(module.moe_router, "router_output_multiplier", None) or 1.0
            scores = torch.sigmoid(logits * mult)
            if bias is not None:
                scores = scores + bias.float()
            ids = scores.topk(module.top_k, dim=-1).indices
            counts = torch.bincount(ids.flatten(), minlength=logits.shape[1])
            n_slots = ids.numel()
            print(
                f"[muse-diag] layer {idx} tokens={logits.shape[0]} "
                f"logit std={logits.std().item():.3g} finite="
                f"{torch.isfinite(logits).all().item()} distinct experts="
                f"{(counts > 0).sum().item()}/{logits.shape[1]} max expert load="
                f"{counts.max().item()}/{n_slots} (uniform ~{n_slots / logits.shape[1]:.2f})"
                f" tokens sharing token0's top-k set="
                f"{(ids.sort(1).values == ids[:1].sort(1).values).all(1).sum().item()}",
                flush=True,
            )

        gate.register_forward_hook(hook)

    for idx, layer in enumerate(model.model.layers):
        if getattr(layer, "_muse_diag", False) or not hasattr(layer, "self_attn"):
            continue
        layer._muse_diag = True

        def lhook(mod, inputs, output, idx=idx):
            if torch.cuda.is_current_stream_capturing() or _rank() != 0:
                return
            key = ("layer", idx)
            if key in _diag_done:
                return
            _diag_done.add(key)
            parts = {
                k: getattr(output, k)
                for k in ("stream", "branch")
                if isinstance(getattr(output, k, None), torch.Tensor)
            }
            msg = " ".join(
                f"{k}: finite={torch.isfinite(v).all().item()} "
                f"rms={v.float().square().mean().sqrt().item():.3g}"
                for k, v in parts.items()
            )
            print(f"[muse-diag] decoder layer {idx} out {msg}", flush=True)

        layer.register_forward_hook(lhook)


def install_routing(runner, requested, dummy_weights, num_tokens, seed):
    if DIAG:
        _install_diag(runner.model)
    mode, fixture = _orig_install(runner, requested, dummy_weights, num_tokens, seed)
    if fixture is not None:
        # Muse keeps its selection bias in `expert_bias`, which the stock
        # fixture doesn't know about; zero it while the fixture is active.
        for module, _ in fixture._layers:
            bias = getattr(module, "expert_bias", None)
            if isinstance(bias, torch.Tensor):
                fixture._biases.append((bias, bias.detach().clone()))
                bias.data.zero_()
        mode = f"{mode}:{MODE}"
    return mode, fixture


routing._moe_layers = _moe_layers
routing.install_routing = install_routing
import regex as _re

import vllm.benchmarks.forward.main as _fwd_main

_current_case = [""]
_orig_run_case = _fwd_main.run_case


def _run_case(runner, case, args):
    _current_case[0] = case.name
    return _orig_run_case(runner, case, args)


_fwd_main.run_case = _run_case


def _by_case_name(default):
    """A case named `..._p<K>` routes from a K-expert pool, else `default`."""

    def ids(num_tokens, num_experts, top_k, ep_size, rng):
        m = _re.search(r"_p(\d+)$", _current_case[0])
        fn = _pool_topk_ids(int(m.group(1))) if m else default
        return fn(num_tokens, num_experts, top_k, ep_size, rng)

    return ids


if MODE == "random":
    routing.balanced_topk_ids = _by_case_name(_random_topk_ids)
elif MODE.startswith("pool:"):
    routing.balanced_topk_ids = _pool_topk_ids(int(MODE.split(":")[1]))
elif MODE != "balanced":
    raise ValueError(f"MUSE_ROUTING={MODE!r}")

# The dummy loader never calls the model's load_weights, which is where Muse
# materializes its gated-residual alpha/beta (enabling the fused dual-norm
# boundary kernels), folds the router multiplier and finalizes MegaMoE
# weights. Without this the boundary runs ~25 unfused elementwise kernels.
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

_orig_dummy_load = DummyModelLoader.load_weights


def _dummy_load_then_finalize(self, model, model_config):
    _orig_dummy_load(self, model, model_config)
    if os.environ.get("MUSE_SKIP_FINALIZE", "0") == "1":
        return
    from vllm.models.muse_spark.common.layers import (
        materialize_residual_path_constants,
    )

    for module in model.modules():
        fin = getattr(module, "finalize_mega_moe_weights", None)
        if callable(fin) and hasattr(module, "layers"):
            fin()
    n = materialize_residual_path_constants(model)
    print(f"[muse-bench] materialized {n} gated residual constants", flush=True)


DummyModelLoader.load_weights = _dummy_load_then_finalize

if os.environ.get("MUSE_TUNE_EXACT", "0") == "1":
    # Run each case's eager warmup under FlashInfer autotune, so ops whose
    # exact call signature the startup autotune missed (e.g. the routed NVFP4
    # MoE on prepared inputs) get profiled before the graph is captured.
    from flashinfer.autotuner import autotune

    from vllm.benchmarks.forward.capture import ForwardBench

    _orig_forward = ForwardBench._forward

    def _tuned_forward(self, **inputs):
        if self.phase != "warmup":
            return _orig_forward(self, **inputs)
        with autotune(tune_mode=True):
            return _orig_forward(self, **inputs)

    ForwardBench._forward = _tuned_forward

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    sys.argv[0] = "vllm"
    main()


# The executor's fused router computes logits from the weight, past the routing
# fixture's patched gate; keep the fixture's routing for comparable timings.
if MODE != "real":
    try:
        import vllm.models.muse_spark.nvidia.specialized_decode.runtime.decode as _sd

        _sd.ROUTER_ROWS = range(0)
    except ImportError:
        pass
