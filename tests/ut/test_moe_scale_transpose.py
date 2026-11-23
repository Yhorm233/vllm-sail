# SPDX-License-Identifier: Apache-2.0
"""Preserve the activation-scale layout selected by the FP8 backend."""

from enum import Enum
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/fused_moe_ppu.py"


def platform(monkeypatch, ppu):
    import sys

    module = ModuleType("vllm.platforms")
    module.current_platform = NS(is_ppu=lambda: ppu, fp8_dtype=lambda: "fp8")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.current_platform


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_quantizer_forwards_new_layout_keyword(monkeypatch, ppu, transpose):
    platform(monkeypatch, ppu)
    expected, tensor, scale = object(), object(), object()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    quantize = function(
        PATCH,
        "install._moe_kernel_quantize_input",
        {
            "_upstream_quantize_input": original,
        },
    )
    assert (
        quantize(tensor, scale, "fp8", False, [128, 128], is_scale_transposed=transpose)
        is expected
    )
    assert calls == [
        (
            (tensor, scale, "fp8", False),
            dict(block_shape=[128, 128], is_scale_transposed=transpose),
        )
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize(
    "backend,block,transpose",
    [
        ("FLASHINFER_TRTLLM", [128, 128], True),
        ("FLASHINFER_TRTLLM", [64, 128], False),
        ("TRITON", [128, 128], False),
        ("FLASHINFER_CUTLASS", [128, 128], False),
    ],
)
def test_actual_fp8_config_layout_reaches_the_group_quantizer(
    monkeypatch, upstream_source_root, ppu, backend, block, transpose
):
    device = platform(monkeypatch, ppu)
    oracle_path = (
        upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/fp8.py"
    )
    utils_path = upstream_source_root / "vllm/model_executor/layers/fused_moe/utils.py"
    backend_type = function(oracle_path, "Fp8MoeBackend", {"Enum": Enum})
    original_config = function(
        oracle_path,
        "make_fp8_moe_quant_config",
        {
            "Fp8MoeBackend": backend_type,
            "fp8_w8a8_moe_quant_config": lambda **kwargs: NS(**kwargs),
        },
    )
    make_config = function(
        "vllm_sail/registry/moe_backends/fp8.py",
        "make_fp8_moe_quant_config",
        {"_upstream_quant_config": original_config},
    )
    alpha, beta = object(), object()
    config = make_config(
        getattr(backend_type, backend),
        object(),
        object(),
        None,
        None,
        block_shape=block,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
    )
    assert config.is_scale_transposed is transpose
    assert config.gemm1_alpha is alpha and config.gemm1_beta is beta
    calls = []
    values = NS(size=lambda axis: 512)
    scales = NS(size=lambda axis: 4)

    def group_quant(tensor, size, **kwargs):
        calls.append((tensor, size, kwargs))
        return values, scales

    namespace = {
        "current_platform": device,
        "per_token_group_quant_fp8": group_quant,
        "cdiv": lambda a, b: (a + b - 1) // b,
    }
    function(utils_path, "_fp8_quantize", namespace)
    upstream_quantizer = function(utils_path, "moe_kernel_quantize_input", namespace)
    quantize = function(
        PATCH,
        "install._moe_kernel_quantize_input",
        {"_upstream_quantize_input": upstream_quantizer},
    )
    tensor = object()
    assert quantize(
        tensor,
        None,
        "fp8",
        False,
        block,
        is_scale_transposed=config.is_scale_transposed,
    ) == (values, scales)
    assert calls == [(tensor, 128, {"column_major_scales": transpose})]


@pytest.mark.requires_runtime
@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("backend", ["FLASHINFER_CUTLASS", "HPC"])
def test_real_per_tensor_config_refills_derived_storage(
    monkeypatch, upstream_source_root, ppu, backend
):
    torch = pytest.importorskip("torch")
    from vllm.model_executor.utils import refill_derived, replace_parameter

    platform(monkeypatch, ppu)
    oracle_path = (
        upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/fp8.py"
    )
    backend_type = function(oracle_path, "Fp8MoeBackend", {"Enum": Enum})
    original = function(
        oracle_path,
        "make_fp8_moe_quant_config",
        {
            "Fp8MoeBackend": backend_type,
            "replace_parameter": replace_parameter,
            "refill_derived": refill_derived,
            "fp8_w8a8_moe_quant_config": lambda **kwargs: NS(**kwargs),
        },
    )
    make_config = function(
        "vllm_sail/registry/moe_backends/fp8.py",
        "make_fp8_moe_quant_config",
        {"_upstream_quant_config": original},
    )
    layer = torch.nn.Module()
    alpha, beta = 1.7, 0.4

    def configure(multiplier):
        return make_config(
            getattr(backend_type, backend),
            torch.tensor([2.0, 3.0]) * multiplier,
            torch.tensor([5.0, 7.0]) * multiplier,
            torch.tensor([0.5, 0.25]) * multiplier,
            torch.tensor([0.125, 0.0625]) * multiplier,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            layer=layer,
        )

    first = configure(1)
    names = ("g1_alphas", "g2_alphas", "_a1_gscale", "_a2_gscale")
    held = [getattr(layer, name) for name in names]
    pointers = [value.data_ptr() for value in held]
    loader = object()
    layer.g1_alphas.weight_loader = loader
    second = configure(2)
    assert [getattr(layer, name).data_ptr() for name in names] == pointers
    assert all(
        getattr(layer, name) is value for name, value in zip(names, held, strict=False)
    )
    assert layer.g1_alphas.weight_loader is loader
    for field, value in zip(
        ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"), held, strict=False
    ):
        assert getattr(first, field) is getattr(second, field) is value
    expected = ([4.0, 3.0], [2.5, 1.75], [1.0, 2.0], [4.0, 8.0])
    for value, values in zip(held, expected, strict=False):
        assert torch.equal(value, torch.tensor(values))
    assert "_a1_gscale" not in layer._parameters
    assert "_a2_gscale" not in layer._buffers
    if ppu:
        assert second.gemm1_alpha == alpha and second.gemm1_beta == beta


@pytest.mark.requires_runtime
@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "kind,inverse", [("fp8", False), ("fp8", True), ("mxfp4", False)]
)
@pytest.mark.parametrize("with_inputs", [False, True])
def test_humming_delegation_stages_canonical_parameters_and_preserves_loader_aliases(
    modules, upstream_source_root, kind, inverse, with_inputs
):
    torch = pytest.importorskip("torch")
    utils = upstream_source_root / "vllm/model_executor/utils.py"
    helpers = {"torch": torch}
    replace = function(utils, "replace_parameter", helpers)
    move = function(utils, "move_parameter", helpers)
    set_kernel = function(utils, "set_kernel_parameter", helpers)
    stage = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/humming/moe.py",
        "stage_moe_weights_for_humming",
        {"replace_parameter": replace},
    )
    layer = torch.nn.Module()
    scale_suffix = "weight_scale_inv" if inverse else "weight_scale"
    if inverse:
        layer.weight_block_size = [1, 2]
    names = ["w13_weight", "w2_weight", f"w13_{scale_suffix}", f"w2_{scale_suffix}"]
    shapes = [(1, 2, 2), (1, 2, 2), (1, 2, 1), (1, 2, 1)]
    if with_inputs and kind == "fp8":
        names += ["w13_input_scale", "w2_input_scale"]
        shapes += [(1,), (1,)]
    held, pointers, loaders = {}, {}, {}
    for name, shape in zip(names, shapes, strict=True):
        param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        param.weight_loader = loaders[name] = object()
        param._ckpt = name
        layer.register_parameter(name, param)
        held[name], pointers[name] = param, param.data_ptr()
    conversions = []
    canonical = []

    def convert(target, **kwargs):
        # 在实际 Humming 计算边界使用 double；staging 与 alias 工具来自真实 pin。
        assert target is layer
        for name, tensor in zip(names, canonical, strict=True):
            assert getattr(layer, name) is held[name]
            assert torch.equal(getattr(layer, name), tensor)
        conversions.append(kwargs)
        for prefix in ("w13", "w2"):
            value = getattr(layer, f"{prefix}_{scale_suffix}") * 2
            if inverse:
                layer.register_parameter(
                    f"{prefix}_weight_scale",
                    torch.nn.Parameter(value, requires_grad=False),
                )
            else:
                set_kernel(layer, f"{prefix}_weight_scale", value)

    modules(
        "vllm.model_executor.layers.quantization.utils.humming",
        stage_moe_weights_for_humming=stage,
        convert_to_humming_moe_kernel_format=convert,
    )
    oracle = (
        upstream_source_root / f"vllm/model_executor/layers/fused_moe/oracle/{kind}.py"
    )
    enum_name = "Fp8MoeBackend" if kind == "fp8" else "Mxfp4MoeBackend"
    backend = function(oracle, enum_name, {"Enum": Enum})
    if kind == "fp8":
        original_namespace = {
            "torch": torch,
            "Fp8MoeBackend": backend,
            "move_parameter": move,
            "set_kernel_parameter": set_kernel,
        }
        function(oracle, "_humming_fp8_weight_schema", original_namespace)
        original = function(
            oracle, "convert_to_fp8_moe_kernel_format", original_namespace
        )
        adapt = function(
            "vllm_sail/registry/moe_backends/fp8.py",
            "convert_to_fp8_moe_kernel_format",
            {
                "_upstream_convert": original,
                "Fp8MoeBackend": NS(
                    PPU_DEEPGEMM=object(), BATCHED_PPU_DEEPGEMM=object()
                ),
            },
        )
    else:
        original = function(
            oracle,
            "convert_weight_to_mxfp4_moe_kernel_format",
            {
                "current_platform": NS(is_rocm=lambda: False),
                "B12X_BACKENDS": (),
                "Mxfp4MoeBackend": backend,
                "_humming_input_schema": lambda key: key,
            },
        )
        adapt = function(
            "vllm_sail/patch/enhancement/mxfp4.py",
            "install._convert_weight",
            {"_upstream_convert_weight": original, "_ppu_backends": lambda kind: ()},
        )
    activation = object()
    for multiplier in (1, 3):
        canonical[:] = [torch.full(shape, float(multiplier)) for shape in shapes]
        if kind == "fp8":
            inputs = canonical[4:] if with_inputs else [None, None]
            result = adapt(backend.HUMMING, layer, *canonical[:4], *inputs)
        else:
            result = adapt(
                backend.HUMMING, layer, *canonical[:4], activation_key=activation
            )
            assert result[4:] == (None, None)
        for name in names:
            assert getattr(layer, name) is held[name]
            assert getattr(layer, name).data_ptr() == pointers[name]
            assert getattr(layer, name).weight_loader is loaders[name]
            assert getattr(layer, name)._ckpt == name
        for prefix, runtime in zip(("w13", "w2"), result[2:4], strict=True):
            source = getattr(layer, f"{prefix}_{scale_suffix}")
            assert runtime is getattr(layer, f"{prefix}_weight_scale") is source
            assert torch.equal(source, torch.full(source.shape, float(multiplier * 2)))
            if inverse:
                assert source.runtime_name == f"{prefix}_weight_scale"
        assert result[0] is held["w13_weight"] and result[1] is held["w2_weight"]
    assert len(conversions) == 2
    if kind == "mxfp4":
        assert (
            conversions
            == [
                dict(
                    quant_config={"quant_method": "mxfp4"},
                    input_schema=activation,
                    allow_input_schema_fallback=False,
                )
            ]
            * 2
        )
    elif inverse:
        assert (
            conversions
            == [dict(quant_config={"quant_method": "fp8", "weight_block_size": [1, 2]})]
            * 2
        )
    else:
        assert all(
            value["quant_config"]["strategy"] == "channel" for value in conversions
        )
