# SPDX-License-Identifier: Apache-2.0
"""Qlean V4.1 metadata contracts, without torch, vLLM or PPU dependencies."""

from types import SimpleNamespace

import pytest

from tests.support.source import function
from tests.ut import test_ppu_kernel_capabilities as harness
from vllm_sail.models.deepseek_v41.config import (
    checkpoint_layer_name,
    normalize_expert_dtype,
)

modules = harness.modules
load_patch = harness.load_patch


@pytest.mark.parametrize("kind", ["deepseek_v41", "deepseek_v41_text"])
def test_exported_expert_alias_is_normalized_for_target_and_draft(kind):
    quant = {
        "quant_method": "mxfp4",
        "expert_dtype": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wq_a"],
    }
    target = SimpleNamespace(
        model_type=kind, expert_dtype="mxfp4", quantization_config=quant
    )
    draft = SimpleNamespace(
        model_type=kind, expert_dtype="mxfp4", quantization_config=quant
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=target),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft)
        ),
    )
    normalize_expert_dtype(config)
    normalize_expert_dtype(config)
    assert target.expert_dtype == draft.expert_dtype == "fp4"
    assert quant["expert_dtype"] == "mxfp4"


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("language_model.model.layers.14.engram.wkv", "layers.14.engram.wkv"),
        ("model.layers.40.attn.wo_a", "mtp.0.attn.wo_a"),
        ("model.layers.42.ffn.shared_experts.down_proj", "mtp.2.ffn.shared_experts.w2"),
        ("model.main_proj", "mtp.0.main_proj"),
    ],
)
def test_checkpoint_names_preserve_target_and_draft_indices(prefix, expected):
    assert checkpoint_layer_name(prefix, 40) == expected


@pytest.mark.upstream_source
def test_export_dispatch_preserves_bf16_and_per_layer_precision(
    upstream_source_root, modules
):
    class Linear:
        pass

    matcher = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/quant_utils.py",
        "is_layer_skipped",
        {"MappingProxyType": dict},
    )
    hf = SimpleNamespace(model_type="deepseek_v41", num_hidden_layers=40)
    modules(
        "vllm.config",
        get_current_vllm_config=lambda: SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf)
        ),
    )
    modules(
        "vllm_sail.models.deepseek_v41.config",
        checkpoint_layer_name=checkpoint_layer_name,
    )
    modules(
        "compressed_tensors.quantization",
        QuantizationArgs=lambda **kw: kw,
        QuantizationStrategy=SimpleNamespace(CHANNEL="channel"),
        QuantizationType=SimpleNamespace(FLOAT="float"),
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        CompressedTensorsLinearMethod=lambda config: "channel",
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8",
        CompressedTensorsW8A8Fp8=lambda **kwargs: kwargs,
    )
    dispatch = function(
        "vllm_sail/models/deepseek_v41/quant_config.py",
        "DeepseekV41FP8Config.get_quant_method",
        {
            "LinearBase": Linear,
            "is_layer_skipped": matcher,
            "current_platform": SimpleNamespace(is_ppu=lambda: True),
            "UnquantizedLinearMethod": lambda: "bf16",
        },
    )
    config = SimpleNamespace(
        fp8_channelwise_layers=[
            "layers.0.attn.wq_a",
            "layers.0.attn.wkv",
            "layers.14.engram.wkv",
            "mtp.0.attn.wo_a",
            "mtp.0.main_proj",
            "layers.0.ffn.shared_experts.w2",
        ],
        _checkpoint_ignored_layers=["layers.14.attn.compressor.wkv"],
        packed_modules_mapping={"fused_wqa_wkv": ["wq_a", "wkv"]},
    )
    for prefix in (
        "language_model.model.layers.0.attn.fused_wqa_wkv",
        "language_model.model.layers.14.engram.wkv",
        "model.layers.40.attn.wo_a",
        "model.main_proj",
        "language_model.model.layers.0.ffn.shared_experts.down_proj",
    ):
        layer = Linear()
        assert dispatch(config, layer, prefix) == "channel"
        assert layer.scheme["weight_quant"]["strategy"] == "channel"
        assert not layer.scheme["is_static_input_scheme"]
    for prefix in (
        "language_model.model.layers.1.attn.wq_a",  # do not broaden layer 0 to all layers
        "language_model.model.layers.14.attn.compressor.wkv",
        "model.markov_head.head",
    ):
        assert dispatch(config, Linear(), prefix) == "bf16"
    config.fp8_channelwise_layers.remove("layers.0.attn.wkv")
    with pytest.raises(ValueError, match="some but not all shards"):
        dispatch(config, Linear(), "language_model.model.layers.0.attn.fused_wqa_wkv")


@pytest.mark.parametrize("ppu,cap", [(True, (8, 0)), (True, (8, 9)), (False, (8, 0))])
@pytest.mark.parametrize("bias", [False, True])
def test_fp8_linear_810e_scale_axes_and_delegation(ppu, cap, bias):
    class Tensor:
        def __init__(self, label, shape=(2, 3), count=1):
            self.label, self.shape, self.count = label, shape, count

        def numel(self):
            return self.count

        def float(self):
            return self

        def reshape(self, *shape):
            return Tensor(("reshape", self.label, shape))

        def __mul__(self, other):
            return Tensor(("mul", self.label, other.label))

        def __add__(self, other):
            return Tensor(("add", self.label, other.label))

        def to(self, dtype):
            return Tensor(("to", self.label, dtype))

        def view(self, *shape):
            return ("view", self.label, shape)

    calls = []

    def mm(a, b):
        calls.append(("mm", a.label, b.label))
        return Tensor("product")

    def native(a, b, **kwargs):
        calls.append(("native", a, b, kwargs))
        return Tensor("native")

    apply = function(
        "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py",
        "PPUCutlassFP8ScaledMMLinearKernel.apply_scaled_mm",
        {
            "torch": SimpleNamespace(mm=mm),
            "ops": SimpleNamespace(cutlass_scaled_mm=native),
            "current_platform": SimpleNamespace(
                is_ppu=lambda: ppu,
                is_device_capability=lambda expected: cap == expected,
            ),
        },
    )
    a, b = Tensor("a", (2, 4)), Tensor("b", (4, 3))
    as_, bs = Tensor("as", count=2), Tensor("bs", count=3)
    bias_tensor = Tensor("bias") if bias else None
    kwargs = dict(
        A=a,
        B=b,
        As=as_,
        Bs=bs,
        bias=bias_tensor,
        out_dtype="bf16",
        output_shape=[1, 2, 3],
    )
    result = apply(None, **kwargs)
    if ppu and cap == (8, 0):
        assert calls == [("mm", "a", "b")]
        scaled = (
            "mul",
            ("mul", "product", ("reshape", "as", (-1, 1))),
            ("reshape", "bs", (1, -1)),
        )
        expected = ("add", scaled, "bias") if bias else scaled
        assert result == ("view", ("to", expected, "bf16"), (1, 2, 3))
        kwargs["As"] = Tensor("blockscale", count=4)
        with pytest.raises(ValueError, match="tensor/token"):
            apply(None, **kwargs)
        assert len(calls) == 1
    else:
        assert calls == [
            (
                "native",
                a,
                b,
                dict(out_dtype="bf16", scale_a=as_, scale_b=bs, bias=bias_tensor),
            )
        ]
        assert result == ("view", "native", (1, 2, 3))


@pytest.mark.parametrize("ppu,cap", [(True, (8, 0)), (True, (8, 9)), (False, (8, 0))])
@pytest.mark.parametrize("clean", [False, True])
def test_fp8_paged_logits_keeps_sdk_route_outside_810e(ppu, cap, clean):
    calls = []

    def software(*args):
        calls.append(("software", args))
        return "software"

    def sdk(*args, **kwargs):
        calls.append(("sdk", args, kwargs))
        return "sdk"

    wrapper = function(
        "vllm_sail/utils/deep_gemm.py",
        "fp8_paged_mqa_logits",
        {
            "current_platform": SimpleNamespace(
                is_ppu=lambda: ppu,
                is_device_capability=lambda expected: cap == expected,
            ),
            "_fp8_paged_mqa_logits_ppu10": software,
            "_lazy_init": lambda: calls.append(("lazy",)),
            "_fp8_paged_mqa_logits_impl": sdk,
        },
    )
    args = tuple(range(7))
    if ppu and cap == (8, 0):
        assert wrapper(*args, clean) == "software"
        assert calls == [("software", args[:5] + (args[6],))]
    else:
        assert wrapper(*args, clean) == "sdk"
        assert calls == [("lazy",), ("sdk", args, dict(clean_logits=clean))]
