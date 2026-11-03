# SPDX-License-Identifier: Apache-2.0
"""Disable optional CUDA-only kernel libraries that do not work on PPU.

``PPUPlatform`` declares ``PlatformEnum.CUDA``, which is what makes the CUDA code
paths work — but it also means capability probes that only ask "is this CUDA?"
answer yes for libraries PPU cannot actually run. ``has_cutedsl()`` and
``has_humming()`` are pure ``_has_module()`` checks upstream, so on a PPU host
that happens to have those wheels installed, vLLM would select kernels that fail
at launch.

Both patches delegate to the upstream implementation first, so if upstream adds a
real capability check we keep it and only add the PPU exclusion.
"""

from __future__ import annotations

import sys

from vllm.platforms import current_platform
from vllm.utils import import_utils

from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.31.0,<0.32.0"

_upstream_has_cutedsl = import_utils.has_cutedsl
_upstream_has_humming = import_utils.has_humming


@patch(
    "vllm.utils.import_utils",
    "has_cutedsl",
    reason=(
        "CuTeDSL kernels do not run on PPU. Upstream only checks that the "
        "`cutlass` module imports, and PPU passes the platform gate because it "
        "declares PlatformEnum.CUDA, so vLLM would select CuTeDSL kernels that "
        "fail at launch."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU gains CuTeDSL support, or upstream gates this on a real device capability.",
)
def has_cutedsl() -> bool:
    if not _upstream_has_cutedsl():
        return False
    return not current_platform.is_ppu()


@patch(
    "vllm.utils.import_utils",
    "has_humming",
    reason=("Humming kernels do not run on PPU; same reasoning as has_cutedsl above."),
    affected_versions=_AFFECTED,
    remove_when="PPU gains Humming support, or upstream gates this on a real device capability.",
)
def has_humming() -> bool:
    if not _upstream_has_humming():
        return False
    return not current_platform.is_ppu()


_CONSUMERS = {
    "has_cutedsl": (
        "vllm.models.deepseek_v4.common.ops.cache_utils",
        "vllm.models.deepseek_v41.common.ops.cache_utils",
        "vllm.models.deepseek_v4.common.ops.fused_indexer_q",
        "vllm.model_executor.layers.sparse_attn_indexer",
        "vllm.models.deepseek_v41.common.ops.cache_utils",
    ),
    "has_humming": (
        "vllm.model_executor.kernels.linear.mixed_precision.humming",
        "vllm.model_executor.kernels.linear.mxfp4.humming",
        "vllm.model_executor.kernels.linear.mxfp6.humming",
        "vllm.model_executor.kernels.linear.mxfp8.humming",
        "vllm.model_executor.kernels.linear.nvfp4.humming",
        "vllm.model_executor.kernels.linear.scaled_mm.humming",
        "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe",
        "vllm.model_executor.layers.quantization.utils.humming.moe",
        "vllm.model_executor.layers.quantization.utils.humming.schema",
    ),
}


def _rebind_loaded_consumers():
    for replacement in (has_cutedsl, has_humming):
        name = replacement.__name__
        original = getattr(replacement, PATCH_MARKER)[f"vllm.utils.import_utils.{name}"]
        for consumer in _CONSUMERS[name]:
            module = sys.modules.get(consumer)
            if module is None or getattr(module, name, None) is not original:
                continue
            patch(
                consumer,
                name,
                reason="A preloaded consumer must observe the PPU optional-kernel gate.",
                affected_versions=_AFFECTED,
                remove_when="vLLM resolves optional-kernel probes through their provider module.",
            )(replacement)


_rebind_loaded_consumers()
