# SPDX-License-Identifier: Apache-2.0
"""A grouped projection must retain the checkpoint's INT8 scale parameter."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import function


@pytest.fixture
def matching(modules, patch_utils_module):
    class LinearBase:
        pass

    class DeepseekV4GroupedWoA(LinearBase):
        pass

    platform = NS(ppu=True)
    platform.is_ppu = lambda: platform.ppu
    modules("vllm.platforms", current_platform=platform)
    modules("vllm.model_executor.layers.linear", LinearBase=LinearBase)
    calls = []
    original = NS(result=None)

    def match(*args):
        calls.append(args)
        return original.result

    provider = modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        find_matched_target=match,
    )
    load_patch("vllm_sail/patch/enhancement/linear_quant_target.py")
    return NS(
        provider=provider,
        platform=platform,
        layer=DeepseekV4GroupedWoA(),
        original=original,
        calls=calls,
    )


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("targets", [("Linear",), ("re:Linear",), ()])
def test_literal_linear_target_only(matching, ppu, linear, targets):
    matching.platform.ppu = ppu
    layer = matching.layer if linear else object()
    result = matching.provider.find_matched_target(
        "layers.0.attn.wo_a", layer, iter(targets)
    )
    assert result == ("Linear" if ppu and linear and "Linear" in targets else None)
    assert matching.calls[0][2] == targets


@pytest.mark.parametrize("target", ["layers.0.attn.wo_a", "re:.*wo_a", "Grouped"])
def test_existing_specific_target_wins(matching, target):
    matching.original.result = target
    assert (
        matching.provider.find_matched_target(
            "layers.0.attn.wo_a", matching.layer, ("Linear", target)
        )
        == target
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("ignored", [False, True])
def test_real_scheme_lookup_preserves_ignore(matching, upstream_source_root, ignored):
    path = upstream_source_root / (
        "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
    )
    get_scheme = function(
        path,
        "CompressedTensorsConfig.get_scheme_dict",
        {
            "find_matched_target": matching.provider.find_matched_target,
            "should_ignore_layer": lambda *args, **kwargs: ignored,
        },
    )
    scheme = {"weights": object(), "format": "int-quantized"}
    config = NS(
        ignore=[], packed_modules_mapping={}, target_scheme_map={"Linear": scheme}
    )
    result = get_scheme(config, matching.layer, "layers.0.attn.wo_a")
    assert result is (None if ignored else scheme)
