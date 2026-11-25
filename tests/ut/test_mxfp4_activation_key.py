# SPDX-License-Identifier: Apache-2.0
"""Carry checkpoint activation schemas through the MXFP4 converter seam."""

import ast
import inspect
from types import SimpleNamespace as NS

import pytest

from tests.support.source import ROOT, definition, function

PATCH = "vllm_sail/patch/enhancement/mxfp4.py"
WNA16 = "vllm_sail/registry/moe_backends/int_wna16.py"


@pytest.mark.parametrize("gpt_oss", [False, True])
def test_ppu_w4a16_conversion_no_longer_reads_removed_marlin_activation_env(
    modules, gpt_oss
):
    def forbidden(name):
        raise AssertionError("删除的 Marlin activation 变量不能再被读取: " + name)

    envs = modules("vllm.envs")
    envs.__getattr__ = forbidden
    calls = []
    result = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    convert = function(
        PATCH,
        "install._convert_gpt_oss_weight" if gpt_oss else "install._convert_weight",
        {
            "_ppu_backends": lambda kind: ("PPU",) if kind == "w4a16" else (),
            "oracle": NS(Mxfp4MoeBackend=NS(MARLIN="marlin")),
            "_upstream_convert_weight": original,
            "_upstream_convert_gpt_oss": original,
        },
    )
    inputs = tuple(object() for _ in range(7))
    assert convert("PPU", *inputs) is result
    assert calls[0][0][:8] == ("marlin", *inputs)
    assert calls[0][1] == (
        {}
        if gpt_oss
        else dict(
            activation=None,
            use_separated_a4w4=False,
            activation_key=None,
        )
    )


@pytest.mark.parametrize("backend", ["ppu", "batched_ppu"])
def test_ppu_wna16_admission_keeps_layout_guards_without_removed_env(modules, backend):
    quant_cls = type("QuantizationArgs", (), {})
    modules("compressed_tensors.quantization", QuantizationArgs=quant_cls)
    modules(
        "vllm.model_executor.layers.quantization.auto_gptq",
        AutoGPTQConfig=type("GPTQ", (), {}),
    )
    modules(
        "vllm_sail.patch.enhancement.models.moe_marlin_gate",
        supports_marlin_layout=lambda *a: True,
    )
    reason = function(
        WNA16, "_backend_incompatibility_reason", {"_PPU": ("ppu", "batched_ppu")}
    )
    quant = quant_cls()
    quant.group_size = 32
    config = NS(has_bias=False)
    assert reason(backend, config, quant, False, False, False) is None
    assert "zero points" in reason(backend, config, quant, True, False, False)
    quant.group_size = 64
    assert "group32" in reason(backend, config, quant, False, False, False)
    quant.group_size, quant.desc_act = 32, True
    assert "activation ordering" in reason(backend, config, quant, False, False, False)


@pytest.mark.upstream_source
def test_wna16_complete_kernel_factory_retains_only_ppu_expert_additions(
    upstream_source_root,
):
    copied = definition(ROOT / WNA16, "make_wna16_moe_kernel")
    original = definition(
        upstream_source_root
        / "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
        "make_wna16_moe_kernel",
    )
    copied.body = [
        node
        for node in copied.body
        if not (
            isinstance(node, ast.ImportFrom) and node.module.startswith("vllm_sail.")
        )
    ]
    allowed = next(
        node
        for node in copied.body
        if isinstance(node, ast.AnnAssign)
        and ast.unparse(node.target) == "allowed_experts"
    )
    assert [ast.unparse(value) for value in allowed.value.elts].count(
        "PPUDeepGemmExperts"
    ) == 1
    assert [ast.unparse(value) for value in allowed.value.elts].count(
        "PPUBatchedDeepGemmExperts"
    ) == 1
    allowed.value.elts = [
        node
        for node in allowed.value.elts
        if ast.unparse(node)
        not in (
            "PPUDeepGemmExperts",
            "PPUBatchedDeepGemmExperts",
        )
    ]
    assert ast.dump(copied, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("backend", ["ppu", "batched_ppu", "marlin"])
def test_real_wna16_converter_redirects_backend_and_keeps_eight_return_slots(
    modules,
    upstream_source_root,
    backend,
):
    quant_cls = type("QuantizationArgs", (), {})
    modules(
        "vllm.model_executor.layers.quantization.auto_awq",
        AutoAWQConfig=type("AWQ", (), {}),
    )
    modules(
        "vllm.model_executor.layers.quantization.auto_gptq",
        AutoGPTQConfig=type("GPTQ", (), {}),
    )
    calls = []
    output = tuple(object() for _ in range(8))

    def process(*args):
        calls.append(args)
        return output

    original = function(
        upstream_source_root
        / "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
        "convert_to_wna16_moe_kernel_format",
        {
            "WNA16MoEBackend": NS(
                HUMMING="humming", MARLIN="marlin", BATCHED_MARLIN="batched_marlin"
            ),
            "QuantizationArgs": quant_cls,
            "WNA16WeightFormat": type("WNA16WeightFormat", (), {}),
            "_process_weights_marlin": process,
        },
    )
    converter = function(
        WNA16,
        "convert_to_wna16_moe_kernel_format",
        {
            "inspect": inspect,
            "_upstream_convert": original,
            "_PPU": ("ppu", "batched_ppu"),
            "WNA16MoEBackend": NS(MARLIN="marlin"),
        },
    )
    quant, layer = quant_cls(), object()
    quant.num_bits, quant.group_size = 4, 32
    weights = tuple(object() for _ in range(8))
    result = converter(backend, layer, quant, *weights)
    assert result is output and len(result) == 8
    assert calls == [(layer, 4, 8, 32, *weights)]
    with pytest.raises(TypeError):
        converter(backend, layer, quant, *weights, input_dtype="int8")
    assert len(calls) == 1


@pytest.mark.parametrize("separated", [False, True])
@pytest.mark.parametrize("with_key", [False, True])
def test_non_ppu_converter_preserves_checkpoint_activation_and_layout(
    separated, with_key
):
    calls = []
    expected, activation, key = object(), object(), object() if with_key else None
    inputs = tuple(object() for _ in range(8))

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    convert = function(
        PATCH,
        "install._convert_weight",
        {
            "_ppu_backends": lambda _: (),
            "_upstream_convert_weight": upstream,
        },
    )
    assert (
        convert(
            "HUMMING",
            *inputs,
            activation=activation,
            activation_key=key,
            use_separated_a4w4=separated,
        )
        is expected
    )
    assert calls == [
        (
            ("HUMMING", *inputs),
            dict(
                activation=activation, use_separated_a4w4=separated, activation_key=key
            ),
        )
    ]


def test_ppu_w4a4_converter_keeps_native_weight_conversion_with_activation_key():
    expected, key = object(), object()
    weights = tuple(object() for _ in range(6))
    calls = []

    def native(*args):
        calls.append(args)
        return expected

    convert = function(
        PATCH,
        "install._convert_weight",
        {
            "_ppu_backends": lambda kind: ("PPU",) if kind == "w4a4" else (),
            "_convert_ppu_weights": native,
        },
    )
    assert convert("PPU", object(), *weights, activation_key=key) is expected
    assert calls == [weights]


@pytest.mark.upstream_source
@pytest.mark.parametrize("override", ["", "fp8"])
def test_humming_converter_retains_checkpoint_key_unless_explicit_override(
    monkeypatch, upstream_source_root, override
):
    import sys
    from types import ModuleType

    path = upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
    backend = NS(
        CUTLASS_MXFP4_MXFP4="cutlass", DEEPGEMM_MXFP4="deepgemm", HUMMING="humming"
    )
    key, schema = object(), object()
    key_calls, conversions, staged, order = [], [], [], []
    weights, transformed = (
        tuple(object() for _ in range(4)),
        tuple(object() for _ in range(4)),
    )
    module = ModuleType("vllm.model_executor.layers.quantization.utils.humming")
    names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")

    def stage(layer, *actual):
        assert actual == weights
        staged.append((layer, actual))
        order.append("stage")
        for name, value in zip(names, actual, strict=True):
            setattr(layer, name, value)

    def convert(layer, **kwargs):
        assert tuple(getattr(layer, name) for name in names) == weights
        conversions.append((layer, kwargs))
        order.append("convert")
        for name, value in zip(names, transformed, strict=True):
            setattr(layer, name, value)

    def to_schema(value):
        key_calls.append(value)
        return schema

    module.convert_to_humming_moe_kernel_format = convert
    module.stage_moe_weights_for_humming = stage
    module.quant_key_to_input_schema = to_schema
    monkeypatch.setitem(sys.modules, module.__name__, module)
    layer = NS(
        w13_weight=object(),
        w2_weight=object(),
        w13_weight_scale=object(),
        w2_weight_scale=object(),
    )
    namespace = {
        "Mxfp4MoeBackend": backend,
        "kMxfp8Dynamic": object(),
        "envs": NS(VLLM_HUMMING_INPUT_QUANT_CONFIG=override),
        "current_platform": NS(is_rocm=lambda: False),
        "B12X_BACKENDS": (),
        "user_activation_override": lambda feature, config: (None, False),
        "get_current_vllm_config": lambda: NS(),
        "get_current_vllm_config_or_none": lambda: NS(),
    }
    for name in (
        "_user_moe_activation_override",
        "_user_moe_activation_override_is_set",
        "_resolve_activation_key",
        "_humming_input_schema",
    ):
        function(path, name, namespace)
    converter = function(path, "convert_weight_to_mxfp4_moe_kernel_format", namespace)
    result = converter("humming", layer, *weights, activation_key=key)
    assert result == (*transformed, None, None)
    assert staged == [(layer, weights)] and order == ["stage", "convert"]
    assert key_calls == ([] if override else [key])
    assert conversions == [
        (
            layer,
            dict(
                quant_config={"quant_method": "mxfp4"},
                input_schema=None if override else schema,
                allow_input_schema_fallback=False,
            ),
        )
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
def test_w4a8_checkpoint_method_retains_inherited_activation_state(
    monkeypatch, upstream_source_root, ppu
):
    import ast
    import sys
    from types import ModuleType

    from tests.support.source import definition

    key = object()

    class Mxfp4MoEMethod:
        def __init__(self, activation_key):
            self.activation_key = activation_key

    path = (
        upstream_source_root
        / "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8_mxfp4.py"
    )
    node = definition(path, "CompressedTensorsW4A8Mxfp4MoEMethod")
    namespace = {"Mxfp4MoEMethod": Mxfp4MoEMethod}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    cls = namespace["CompressedTensorsW4A8Mxfp4MoEMethod"]
    assert cls.__bases__ == (Mxfp4MoEMethod,)
    method = cls(key)
    platform = ModuleType("vllm.platforms")
    platform.current_platform = NS(is_ppu=lambda: ppu)
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    old_class_module = ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4"
    )
    old_class_module.CompressedTensorsW4A4Mxfp4MoEMethod = type("W4A4", (), {})
    monkeypatch.setitem(sys.modules, old_class_module.__name__, old_class_module)
    calls = []

    def original(*args):
        calls.append(args)
        return method

    get_method = function(
        "vllm_sail/patch/enhancement/compressed_tensors_ppu.py",
        "install._get_moe_method",
        {"_upstream_get_moe_method": original},
    )
    quant, layer = object(), object()
    assert get_method(quant, layer, "model.experts") is method
    assert method.activation_key is key
    assert calls == [(quant, layer, "model.experts")]
