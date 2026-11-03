# SPDX-License-Identifier: Apache-2.0
"""Channelwise quantization matching against upstream fused-layer semantics."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, definition, function


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "relative,name",
    [
        (
            "vllm_sail/models/deepseek_v4/quant_config.py",
            "DeepseekV4FP8Config.get_quant_method",
        ),
        (
            "vllm_sail/patch/enhancement/mxfp4.py",
            "install._get_quant_method",
        ),
        (
            "vllm_sail/patch/enhancement/compressed_tensors_ppu.py",
            "install._is_fp8_channelwise_layer",
        ),
    ],
)
def test_channelwise_matching_with_upstream_fused_layers(
    upstream_source_root, relative, name
):
    matcher = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/quant_utils.py",
        "is_layer_skipped",
        {"MappingProxyType": dict},
    )
    node = definition(ROOT / relative, name)
    call = next(
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "is_layer_skipped"
        and any(
            (
                isinstance(kw.value, ast.Attribute)
                and kw.value.attr == "fp8_channelwise_layers"
            )
            or (isinstance(kw.value, ast.Name) and kw.value.id == "patterns")
            for kw in n.keywords
        )
    )
    expression = compile(ast.Expression(call), relative, "eval")
    config = SimpleNamespace(
        fp8_channelwise_layers=["attn.wq_a", "attn.wkv"],
        packed_modules_mapping={"fused_wqa_wkv": ["wq_a", "wkv"]},
    )
    namespace = {
        "self": config,
        "is_layer_skipped": matcher,
        "match_mode": "substring",
        "patterns": config.fp8_channelwise_layers,
    }
    for prefix in (
        "model.layers.0.attn.fused_wqa_wkv",
        "model.mtp.1.attn.fused_wqa_wkv",
    ):
        namespace.update(prefix=prefix, layer_name=prefix)
        assert eval(expression, namespace)
    config.fp8_channelwise_layers = ["attn.unrelated"]
    namespace["patterns"] = config.fp8_channelwise_layers
    assert not eval(expression, namespace)
    config.fp8_channelwise_layers = ["attn.wq_a"]
    namespace["patterns"] = config.fp8_channelwise_layers
    with pytest.raises(ValueError, match="some but not all shards"):
        eval(expression, namespace)
