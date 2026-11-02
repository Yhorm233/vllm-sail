# SPDX-License-Identifier: Apache-2.0
"""Preserve the V4.1 VL wrapper and adjust weight-scale mapping only for 810E W8A8 instances."""

from contextlib import nullcontext
from copy import deepcopy

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV41ForCausalLM as UpstreamDeepseekV41ForCausalLM,
)
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.models.deepseek_v41.config import uses_int8_w8a8


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
    def __init__(self, *, vllm_config, prefix=""):
        directory = envs.VLLM_SAIL_V41_AUDIT_DIR
        if directory and (
            not current_platform.is_ppu()
            or not current_platform.is_device_capability((8, 0))
            or not vllm_config.model_config.enforce_eager
        ):
            raise ValueError("V4.1 runtime auditing requires an 810E PPU eager service")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._v41_int8_config = None
        if (
            current_platform.is_ppu()
            and current_platform.is_device_capability((8, 0))
            and uses_int8_w8a8(vllm_config)
        ):
            self.hf_to_vllm_mapper = int8_weights_mapper(self.hf_to_vllm_mapper)
            self._v41_int8_config = vllm_config
        self._v41_audit = None
        if directory:
            from .runtime_audit import RuntimeAudit

            self._v41_audit = RuntimeAudit(directory, self, vllm_config)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
        **kwargs,
    ):
        forward = super().forward
        args = (input_ids, positions, intermediate_tensors, inputs_embeds)
        kwargs["lookback_token_ids"] = lookback_token_ids
        # Profiling and regular forward may lack an initialized config context.
        # Scope config to verified 810E W8A8 instances; restore it on exit or error.
        context = (
            set_current_vllm_config(self._v41_int8_config)
            if self._v41_int8_config is not None
            else nullcontext()
        )
        with context:
            if self._v41_audit is None:
                return forward(*args, **kwargs)
            return self._v41_audit.run(forward, *args, **kwargs)
