# SPDX-License-Identifier: Apache-2.0
"""Keep DSv4.1 NVIDIA-specific decode and column-gather paths off PPU."""

import sys

from vllm.models.deepseek_v41.nvidia.specialized_decode import ampere, hopper
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_REASON = "PPU reuses CUDA capability numbers but cannot run DSv4.1 NVIDIA-specific decode kernels."
_AFFECTED = "upstream main e83d82a"
_REMOVE_WHEN = "The upstream specialized decode predicates distinguish PPU from NVIDIA devices."
_ampere = ampere.is_ampere
_hopper = hopper.is_hopper


@patch(
    "vllm.models.deepseek_v41.nvidia.specialized_decode.ampere",
    "is_ampere",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def is_ampere() -> bool:
    return not current_platform.is_ppu() and _ampere()


@patch(
    "vllm.models.deepseek_v41.nvidia.specialized_decode.hopper",
    "is_hopper",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def is_hopper() -> bool:
    return not current_platform.is_ppu() and _hopper()


for _name, _original, _guarded in (
    ("is_ampere", _ampere, is_ampere),
    ("is_hopper", _hopper, is_hopper),
):
    for _path in (
        "vllm.models.deepseek_v41.nvidia.model",
        "vllm.models.deepseek_v41.nvidia.dspark",
        "vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.mhc",
    ):
        if getattr(sys.modules.get(_path), _name, None) is _original:
            patch(
                _path, _name, reason=_REASON,
                affected_versions=_AFFECTED, remove_when=_REMOVE_WHEN,
            )(_guarded)
