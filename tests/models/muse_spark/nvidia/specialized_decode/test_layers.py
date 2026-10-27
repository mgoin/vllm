# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark specialized decode through the real model: a four-layer Muse
Spark Open on random weights at TP4, served with ``decode_backend``
``latency_specialized`` and ``auto``.

The layers follow the checkpoint's pattern: sliding window, full attention
(NoPE), then two sliding-window layers; layer 0 has BF16 experts, the others
NVFP4. The DFlash2 draft model of the deployment, also on random weights,
drafts 7 tokens from the target's FP32 aux hidden states of every layer, with
and without adaptive verification (which gives requests ragged query lengths).
Twelve requests decode greedily and run FULL decode graphs of up to 12
requests of 8 rows.

The target's final hidden state is kept for every decode row, keyed by prompt,
position and the tokens so far, and the two runs are compared within the
tolerance below.

Needs four Blackwell GPUs with NVLink multicast; a few minutes, mostly two
engine starts.
"""

import zlib
from typing import TYPE_CHECKING

import pytest
import torch

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

MODEL = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DRAFT_MODEL = "RadixArk/Muse-Spark-Open-DFlash2"
LOAD_FORMAT = "muse_spark_nvidia_random"
NUM_LAYERS = 4
DRAFTS = 7
MAX_ROWS = 256

# The target models this worker loaded.
_LOADED: list[torch.nn.Module] = []


def truncate(config):
    """``hf_overrides`` for the target and, being callable, the DFlash2 draft
    too: the target keeps its first ``NUM_LAYERS`` layers and the draft reads
    the aux hidden states of all of them. Module level, so engine processes
    unpickle it."""
    text = getattr(config, "text_config", None)
    if text is not None and hasattr(text, "layer_types"):
        text.num_hidden_layers = NUM_LAYERS
        for key in ("layer_types", "mlp_layer_types", "layer_rope_theta"):
            setattr(text, key, getattr(text, key)[:NUM_LAYERS])
    dflash = getattr(config, "dflash_config", None)
    if dflash is not None:
        dflash["target_layer_ids"] = list(range(NUM_LAYERS))
        config.num_target_layers = NUM_LAYERS
    return config


@torch.no_grad()
def _fill(name: str, t: torch.Tensor, norm_shift: float = 0.0) -> None:
    """A random value for every weight, seeded by its name (alike on every
    rank and in every run), in the ranges the checkpoint's formats hold."""
    if t.device.type == "meta" or t.numel() == 0:
        return
    g = torch.Generator(device=t.device).manual_seed(zlib.crc32(name.encode()))

    def randn(scale=1.0, shift=0.0):
        return torch.randn(t.shape, generator=g, device=t.device) * scale + shift

    if t.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(0, 256, u8.shape, generator=g, device=t.device))
    elif t.dtype == torch.float8_e4m3fn:
        # NVFP4 block scales.
        t.copy_(randn(0.25, 1.0).clamp(0.25, 2.0))
    elif not torch.is_floating_point(t):
        return
    elif name.endswith("weight_scale_2"):
        # FP4 codes have a standard deviation of about 3.
        t.fill_(0.005)
    elif name.endswith("input_scale"):
        t.fill_(4.0 / (448.0 * 6.0))
    elif "norm" in name:
        t.copy_(randn(0.1, norm_shift))
    elif name.endswith((".add_attn.gate", ".add_ffn.gate")):
        t.copy_(randn(0.5))
    elif name.endswith("expert_bias"):
        t.copy_(randn(0.01))
    elif name.endswith("embed_tokens.weight"):
        t.copy_(randn())
    else:
        t.copy_(randn(0.02))


@register_model_loader(LOAD_FORMAT)
class RandomWeightLoader(DummyModelLoader):
    """Dummy loading with weights in the checkpoint's ranges, followed by the
    post-load steps of ``MuseSpark1xForCausalLM.load_weights``."""

    def load_weights(self, model, model_config) -> None:
        from vllm.models.muse_spark.common.layers import (
            materialize_residual_path_constants,
        )
        from vllm.models.muse_spark.nvidia.model import MuseSpark1xForCausalLM

        super().load_weights(model, model_config)
        is_target = isinstance(model, MuseSpark1xForCausalLM)
        # The draft's norms multiply by their weight; the target's are
        # zero-centered.
        norm_shift = 0.0 if is_target else 1.0
        for name, t in model.named_parameters():
            _fill(name, t, norm_shift)
        if not is_target:
            return
        for name, t in model.named_buffers():
            if name.endswith("expert_bias"):
                _fill(name, t)
        model.model.finalize_mega_moe_weights()
        materialize_residual_path_constants(model)
        _LOADED.append(model)
        model._probe = _Probe(model)


class _Probe:
    """Persistent rows of the target's final hidden states and FP32 aux hidden
    states, copied by its forward. The copies are part of the forward, so CUDA
    graph replays fill them too."""

    def __init__(self, model: torch.nn.Module):
        hidden = model.config.get_text_config().hidden_size
        device = next(model.parameters()).device
        self.output = torch.zeros(MAX_ROWS, hidden, dtype=torch.bfloat16, device=device)
        self.aux = [
            torch.zeros(MAX_ROWS, hidden, dtype=torch.float32, device=device)
            for _ in range(NUM_LAYERS)
        ]
        forward = model.forward

        def forward_and_keep(*args, **kwargs):
            out = forward(*args, **kwargs)
            hidden_states, aux = out if isinstance(out, tuple) else (out, [])
            rows = hidden_states.shape[0]
            if rows <= MAX_ROWS:
                self.output[:rows].copy_(hidden_states)
                for buffer, state in zip(self.aux, aux):
                    buffer[:rows].copy_(state)
            return out

        model.forward = forward_and_keep


class StepRecorder:
    """Worker extension (``worker_extension_cls``): it also brings this module,
    and so the loader, into the workers. Rank 0 keeps every step's rows."""

    model_runner: "GPUModelRunner"
    rank: int

    def paths(self) -> dict:
        model = _LOADED[-1]
        decode = model.specialized_decode
        return {
            "model_state": type(self.model_runner.model_state).__name__,
            "decode": type(decode).__name__,
            "captured_steps": getattr(decode, "captured_steps", 0),
        }

    def record_steps(self, prompt_lens: list[int]) -> None:
        if self.rank != 0:
            return
        runner = self.model_runner
        self._prompt_lens = prompt_lens
        self._rows: dict[tuple, torch.Tensor] = {}
        self._full_steps = 0
        run_fullgraph = runner.cudagraph_manager.run_fullgraph

        def run_and_count(desc):
            self._full_steps += 1
            return run_fullgraph(desc)

        runner.cudagraph_manager.run_fullgraph = run_and_count
        self._modes: dict[str, int] = {}
        dispatch = runner.cudagraph_manager.dispatch

        def dispatch_and_count(*args, **kwargs):
            desc = dispatch(*args, **kwargs)
            mode = f"{desc.cg_mode.name}:{desc.num_tokens}"
            self._modes[mode] = self._modes.get(mode, 0) + 1
            return desc

        runner.cudagraph_manager.dispatch = dispatch_and_count
        self._executed = 0
        execute_model = runner.execute_model

        def execute_and_record(*args, **kwargs):
            self._executed += 1
            output = execute_model(*args, **kwargs)
            state = runner.execute_model_state
            if not kwargs.get("dummy_run") and state is not None:
                self._record(state.input_batch)
            return output

        runner.execute_model = execute_and_record

    def _record(self, batch) -> None:
        torch.accelerator.synchronize()
        n = batch.num_tokens
        if batch.has_prefill or n > MAX_ROWS:
            return
        probe = _LOADED[-1]._probe
        hidden = torch.cat([probe.output[:n].float(), *(a[:n] for a in probe.aux)], -1)
        hidden = hidden.cpu()
        positions = batch.positions[:n].tolist()
        input_ids = batch.input_ids[:n].tolist()
        for i in range(batch.num_reqs):
            prompt = self._prompt_lens.index(int(batch.prefill_len_np[i]))
            start, end = batch.query_start_loc_np[i : i + 2]
            for row in range(start, end):
                key = (prompt, positions[row], tuple(input_ids[start : row + 1]))
                self._rows[key] = hidden[row]

    def save_steps(self, path: str) -> None:
        if self.rank == 0:
            torch.save(
                {
                    "rows": self._rows,
                    "full_steps": self._full_steps,
                    "modes": self._modes,
                    "executed": self._executed,
                },
                path,
            )


def _serve(
    vllm_runner, specialized_decode: bool, adaptive: bool, prompts, max_tokens, path
):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    with vllm_runner(
        MODEL,
        tensor_parallel_size=4,
        load_format=LOAD_FORMAT,
        hf_overrides=truncate,
        worker_extension_cls=f"{__name__}.StepRecorder",
        additional_config={
            "decode_backend": "latency_specialized" if specialized_decode else "auto"
        },
        speculative_config={
            "method": "dflash",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": DRAFTS,
            # Greedy drafts keep both runs' verify rows alike.
            "enable_adaptive_verification": adaptive,
        },
        # Autotuning times each run apart and may pick other tactics for the
        # stock paths; FlashInfer's defaults are the same in both.
        enable_flashinfer_autotune=False,
        # Verify steps of 1 to 16 requests of 8 rows.
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [8, 16, 24, 32, 48, 64, 96, 128],
        },
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=16,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.6,
    ) as runner:
        llm = runner.llm
        paths = llm.collective_rpc("paths")
        llm.collective_rpc("record_steps", args=([len(p) for p in prompts],))
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            [
                SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
                for n in max_tokens
            ],
        )
        llm.collective_rpc("save_steps", args=(path,))
        paths_after = llm.collective_rpc("paths")
    steps = torch.load(path)
    tokens = [list(o.outputs[0].token_ids) for o in outputs]
    return tokens, paths, paths_after, steps


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or torch.accelerator.device_count() < 4,
    reason="needs four Blackwell GPUs",
)
@pytest.mark.parametrize("adaptive", [False, True])
def test_specialized_decode_matches_stock(vllm_runner, tmp_path, adaptive: bool):
    prompts = [[1000 + (j * 7 + i) % 97 for j in range(32 + 8 * i)] for i in range(12)]
    # Requests finish one by one: decode steps of 12 down to 1 requests.
    max_tokens = [24 + 8 * i for i in range(len(prompts))]
    stock = _serve(
        vllm_runner, False, adaptive, prompts, max_tokens, str(tmp_path / "stock.pt")
    )
    special = _serve(
        vllm_runner, True, adaptive, prompts, max_tokens, str(tmp_path / "special.pt")
    )

    _, stock_paths, _, stock_steps = stock
    _, special_paths, special_after, special_steps = special
    print("stock", {k: v for k, v in stock_steps.items() if k != "rows"})
    print("special", {k: v for k, v in special_steps.items() if k != "rows"})
    assert {p["decode"] for p in stock_paths} == {"NoneType"}
    assert {p["model_state"] for p in special_paths} == {
        "MuseSparkSpecializedDecodeModelState"
    }
    # Every rank captured the executor into its FULL decode graphs.
    assert all(p["captured_steps"] > 0 for p in special_after)
    assert special_steps["full_steps"] > 0
    # Steps of fewer than 12 requests ran the executor's low-batch kernels.
    assert any(
        mode.startswith("FULL:") and int(mode.split(":")[1]) <= 64
        for mode in special_steps["modes"]
    )

    stock_rows, special_rows = stock_steps["rows"], special_steps["rows"]
    common = stock_rows.keys() & special_rows.keys()
    # Random weights leave the logits nearly flat, so the two runs' greedy
    # tokens drift apart once their batches differ. Without adaptive
    # verification both runs take the same steps; with it, the drafts each
    # run verifies follow its own timing-based budget.
    assert len(common) >= (sum(max_tokens) // 2 if not adaptive else 1)

    def relative(key, segment: int | None = None) -> float:
        a, b = stock_rows[key], special_rows[key]
        if segment is not None:
            width = a.shape[0] // (NUM_LAYERS + 1)
            a = a[segment * width : (segment + 1) * width]
            b = b[segment * width : (segment + 1) * width]
        return ((b - a).abs().max() / a.abs().max().clamp_min(1e-6)).item()

    # The executor's kernels may round differently from the stock ones. The
    # NVFP4 expert input quantization turns a last-bit difference into a few
    # percent on a value whose 16-element block scale changes, and random
    # weights carry no signal to dominate it, so later layers drift. The
    # residual stream after layer 0 (BF16 experts) must match closely, and
    # most rows of the output too.
    first_layer = max(relative(k, segment=1) for k in common)
    output = sorted(relative(k, segment=0) for k in common)
    median = output[len(output) // 2]
    print(
        f"adaptive={adaptive}: {len(common)} rows compared, "
        f"{special_steps['full_steps']} FULL steps; relative difference after "
        f"layer 0 max {first_layer:.2e}; output median {median:.2e}, "
        f"max {output[-1]:.2e}"
    )
    assert first_layer <= 2e-2
    # Adaptive verification's runs share few rows, from histories that have
    # already drifted.
    assert adaptive or median <= 1e-2
