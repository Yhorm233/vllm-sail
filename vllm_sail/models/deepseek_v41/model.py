# SPDX-License-Identifier: Apache-2.0
"""Preserve the V4.1 VL wrapper and adjust weight-scale mapping only for PPU W8A8 instances."""

from copy import deepcopy

from vllm.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV41ForCausalLM as UpstreamDeepseekV41ForCausalLM,
)
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41.config import set_int8_w8a8, uses_int8_w8a8


def int8_weights_mapper(mapper):
    """Keep weight_scale_inv for Engram FP32 row scales; use weight_scale for linear layers and experts."""
    mapper = deepcopy(mapper)
    mapper.orig_to_new_regex = {
        pattern: replacement
        if "engram" in pattern.pattern
        else replacement.replace("weight_scale_inv", "weight_scale")
        for pattern, replacement in mapper.orig_to_new_regex.items()
    }
    return mapper


class DeepseekV41ForCausalLM(UpstreamDeepseekV41ForCausalLM):
    """PPU V4.1 model.

    Like the PPU V4 model, the quantization contract is resolved from the
    constructor's ``vllm_config`` only; forward stays upstream and never reads
    the global config.
    """

    def __init__(self, *, vllm_config, prefix=""):
        use_int8 = current_platform.is_ppu() and uses_int8_w8a8(vllm_config)
        # Publish before layer construction so the decision covers this
        # model's warmup, profiling and draft forwards sharing its caches.
        set_int8_w8a8(use_int8)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Install on this instance after the upstream initializer; mutating
        # the class would leak one checkpoint's mapping into later instances.
        if use_int8:
            self.hf_to_vllm_mapper = int8_weights_mapper(self.hf_to_vllm_mapper)
