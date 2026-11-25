# SPDX-License-Identifier: Apache-2.0
"""PPU device capability aliases must not activate NVIDIA DSv4.1 code."""

import pytest

from tests.support.modules import load_patch


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("sm", [80, 89, 90])
def test_specialized_decode_keeps_native_and_preloaded_consumers_consistent(
    modules, ppu, sm
):
    from types import SimpleNamespace as NS

    prefix = "vllm.models.deepseek_v41.nvidia."
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: ppu))
    ampere = modules(prefix + "specialized_decode.ampere", is_ampere=lambda: sm == 80)
    hopper = modules(prefix + "specialized_decode.hopper", is_hopper=lambda: sm == 90)
    model = modules(
        prefix + "model", is_ampere=ampere.is_ampere, is_hopper=hopper.is_hopper
    )
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_device.py")
    assert model.is_ampere is ampere.is_ampere and model.is_hopper is hopper.is_hopper
    assert model.is_ampere() is (not ppu and sm == 80)
    assert model.is_hopper() is (not ppu and sm == 90)
