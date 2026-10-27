# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model state of the specialized decode: ``DefaultModelState`` plus the
decode executor, built once the weights are loaded (the model runner creates
the model state after loading the target and draft models, and before memory
profiling and CUDA graph capture)."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class MuseSparkSpecializedDecodeModelState(DefaultModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.decode = model.get_language_model().build_specialized_decode()
