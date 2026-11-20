# SPDX-License-Identifier: Apache-2.0
"""Preserve the V4.1 VL wrapper and its checkpoint quantization contract."""

from copy import deepcopy

from vllm.model_executor.models.utils import WeightsMapper
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

    # configure_quant_config reads this before __init__ constructs any layers.
    # The upstream VL wrapper only installs an instance mapper after constructing
    # its language model, which is too late to remap compressed-tensors ignores
    # such as layers.1.engram.wkv into language_model.model.layers.1.engram.wkv.
    # Only map module prefixes here, independently of expert dtype or scale
    # naming. The upstream initializer still installs the checkpoint-specific
    # instance mapper for loading weights.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": "language_model.model.layers.",
            "embed.": "language_model.model.embed.",
            "norm.": "language_model.model.norm.",
            "hc_head": "language_model.model.hc_head",
            "mtp.": "language_model.model.mtp.",
        }
    )

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
