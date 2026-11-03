# SPDX-License-Identifier: Apache-2.0
"""Opt-in modular MoE instrumentation."""

import functools

from vllm.model_executor.layers.fused_moe import modular_kernel as _modular

from vllm_sail.patch.utils import patch
from vllm_sail.profiling.moe import moe_range

_original = _modular.FusedMoEKernelModularImpl.apply


@patch(
    _modular.__name__,
    "FusedMoEKernelModularImpl.apply",
    reason="Opt-in PPU profiling includes modular MoE dispatch.",
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="Upstream exposes equivalent per-dispatch MoE tracing.",
)
@functools.wraps(_original)
def apply(self, hidden_states, w1, w2, topk_ids, topk_weights, *args, **kwargs):
    with moe_range(hidden_states, w1, topk_ids):
        return _original(
            self, hidden_states, w1, w2, topk_ids, topk_weights, *args, **kwargs
        )
