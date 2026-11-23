# SPDX-License-Identifier: Apache-2.0
"""Keep compressed-tensors Linear targets valid for renamed linear subclasses."""

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as _ct,
)
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_original = _ct.find_matched_target


@patch(
    _ct.__name__,
    "find_matched_target",
    reason=(
        "DeepseekV4GroupedWoA inherits ColumnParallelLinear but its class name "
        "no longer contains Linear. Compressed-tensors silently leaves it "
        "unquantized, so PPU INT8 checkpoints cannot load its weight_scale. "
        "Honor the literal Linear target through the actual base class after "
        "upstream's explicit name, regex and type matching."
    ),
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="Compressed-tensors Linear target matching recognizes LinearBase subclasses.",
)
def find_matched_target(layer_name, module, targets, fused_mapping=None):
    targets = tuple(targets)
    result = _original(layer_name, module, targets, fused_mapping or {})
    if (
        result is None
        and current_platform.is_ppu()
        and "Linear" in targets
        and isinstance(module, LinearBase)
    ):
        return "Linear"
    return result
