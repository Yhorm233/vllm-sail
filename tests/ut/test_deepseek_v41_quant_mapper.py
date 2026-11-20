# SPDX-License-Identifier: Apache-2.0
"""Check pre-construction quantization mapping without importing torch or vLLM."""

import re
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.source import function


@pytest.mark.upstream_source
@pytest.mark.parametrize("quant_type", ["compressed-tensors", "fp8"])
@pytest.mark.parametrize("bf16_wkv", [False, True])
def test_checkpoint_quantization_paths_are_mapped_before_construction(
    modules, upstream_source_root, bf16_wkv, quant_type
):
    # Execute the real upstream mapper and configuration functions. No model or
    # tensor stubs implement the name transformation being checked here.
    source = upstream_source_root / "vllm"
    namespace = dict(__name__=__name__, re=re, field=field, replace=replace)
    namespace["WeightsMapper"] = dataclass(
        function(source / "model_executor/models/utils.py", "WeightsMapper", namespace)
    )
    modules(
        "vllm.model_executor.models.utils", WeightsMapper=namespace["WeightsMapper"]
    )
    base = type("UpstreamVL", (), {})  # Upstream VL has no class-level mapper.
    modules(
        "vllm.models.deepseek_v41.nvidia.vl_model",
        DeepseekV41ForCausalLM=base,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace())
    modules(
        "vllm_sail.models.deepseek_v41.config",
        set_int8_w8a8=lambda value: None,
        uses_int8_w8a8=lambda config: True,
    )
    model_class = load_patch(
        "vllm_sail/models/deepseek_v41/model.py"
    ).DeepseekV41ForCausalLM
    mapper = model_class.hf_to_vllm_mapper
    assert mapper.orig_to_new_regex == {}
    assert mapper.orig_to_new_suffix == {}
    assert mapper.orig_to_new_substr == {}
    assert mapper.orig_to_new_stacked == {}
    assert mapper.orig_to_new_renaming == []
    # The module-prefix rules must stay aligned with the upstream VL wrapper,
    # without borrowing its expert-dtype-dependent weight or scale rules.
    factory_source = function(
        source / "models/deepseek_v41/nvidia/vl_model.py",
        "_make_deepseek_v4_vl_weights_mapper",
        {
            "WeightsMapper": namespace["WeightsMapper"],
            "_make_deepseek_v4_weights_mapper": lambda *args: SimpleNamespace(
                orig_to_new_regex={}
            ),
        },
    )
    assert (
        mapper.orig_to_new_prefix
        == factory_source("unused", "unused").orig_to_new_prefix
    )
    # Names that look like weights/scales retain every suffix. Already rooted
    # paths are also unchanged, so prefix mapping does not apply twice.
    for name in (
        "layers.0.ffn.experts.0.w1.scale",
        "layers.1.engram.embed.scale",
        "layers.0.ffn.shared_experts.w2",
        "layers.0.ffn.gate.bias",
    ):
        assert mapper.map_name(name) == f"language_model.model.{name}"
    rooted = "language_model.model.layers.1.engram.wkv"
    assert mapper.map_name(rooted) == rooted
    quant_source = (
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
        if quant_type == "compressed-tensors"
        else "model_executor/layers/quantization/fp8.py"
    )
    quant_class_name = (
        "CompressedTensorsConfig" if quant_type == "compressed-tensors" else "Fp8Config"
    )
    apply_mapper = function(
        source / quant_source,
        f"{quant_class_name}.apply_vllm_mapper",
        namespace,
    )
    quant_class = type("QuantConfig", (), {"apply_vllm_mapper": apply_mapper})
    quant = quant_class()
    quant.kv_cache_scheme = None
    wkv_names = [f"layers.{layer}.engram.wkv" for layer in (1, 14)]
    checkpoint_ignore = ["layers.0.attn.wq", "mtp.0.head"]
    if bf16_wkv:
        checkpoint_ignore += wkv_names
    quant.ignore = list(checkpoint_ignore)
    quant.ignored_layers = list(checkpoint_ignore)
    # Explicit paths, class targets, and regex targets follow distinct upstream
    # rules. MTP ignores must survive the weight mapper's drop rules too.
    quant.target_scheme_map = {
        "Linear": "w8a8",
        "layers.0.attn.wkv": "w8a8",
        "re:.*ffn.*": "w8a8",
    }
    configure = function(
        source / "model_executor/model_loader/utils.py",
        "configure_quant_config",
        {"SupportsQuant": type("SupportsQuant", (), {})},
    )
    # This is the engine's hook before the first model instance is constructed.
    configure(quant, model_class)
    ignored = (
        quant.ignore if quant_type == "compressed-tensors" else quant.ignored_layers
    )
    for name in wkv_names:
        runtime_name = f"language_model.model.{name}"
        assert (runtime_name in ignored) is bf16_wkv
    assert ignored == [f"language_model.model.{n}" for n in checkpoint_ignore]
    if quant_type == "compressed-tensors":
        assert quant.target_scheme_map == {
            "Linear": "w8a8",
            "language_model.model.layers.0.attn.wkv": "w8a8",
            "re:.*ffn.*": "w8a8",
        }
    assert checkpoint_ignore[0] == "layers.0.attn.wq"
    assert model_class.hf_to_vllm_mapper is mapper
    assert not hasattr(base, "hf_to_vllm_mapper")
