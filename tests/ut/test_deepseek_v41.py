# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

import ast
import re
from types import SimpleNamespace

import pytest

from tests.support.source import definition, function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("record_bytes", [528, 584])
def test_cache_gather_dispatch_after_preload(
    modules, upstream_source_root, ppu, record_bytes
):
    calls = []
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.utils.import_utils", has_cutedsl=lambda: True, has_humming=lambda: True
    )
    provider = modules("vllm.utils.import_utils")
    cache = modules(
        "vllm.models.deepseek_v41.common.ops.cache_utils",
        has_cutedsl=provider.has_cutedsl,
        V41_BYTES_PER_TOKEN=528,
        V41_NVFP4_BYTES_PER_TOKEN=288,
        V41_QUANT_BLOCK=32,
        dequantize_and_gather_k_cache_triton=lambda *a, **kw: calls.append("triton"),
    )
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl",
        _DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL=lambda **kw: calls.append("cute"),
    )
    gather = function(
        upstream_source_root / "vllm/models/deepseek_v41/common/ops/cache_utils.py",
        "dequantize_and_gather_k_cache",
        cache.__dict__,
    )
    load_patch("vllm_sail/patch/enhancement/import_gates.py")
    gather(None, SimpleNamespace(shape=(4, 64, record_bytes)), None, None, None, 64, 3)
    assert calls == ["triton" if ppu else "cute"]


@pytest.fixture
def quant_module(modules):
    calls = []

    class Upstream:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def override_quantization_method(cls, *args):
            calls.append(("recognize", args))
            return "upstream-method"

        @classmethod
        def from_config(cls, config):
            calls.append(("config", config))
            return cls()

        def get_quant_method(self, layer, prefix):
            calls.append(("dispatch", layer, prefix))
            return "upstream-kernel"

    modules("vllm.models.deepseek_v41.quant_config", DeepseekV4FP8Config=Upstream)
    modules("vllm.model_executor.layers.fused_moe", RoutedExperts=type("MoE", (), {}))
    modules(
        "vllm.model_executor.layers.linear",
        LinearBase=type("Linear", (), {}),
        UnquantizedLinearMethod=object,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=lambda **kwargs: False,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    return load_patch("vllm_sail/models/deepseek_v41/quant_config.py"), calls


def test_standard_quantization_delegates_without_rewriting_config(quant_module):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    checkpoint = {"quant_method": "fp8", "weight_block_size": [32, 32]}
    hf = SimpleNamespace(model_type="deepseek_v41")
    assert cls.override_quantization_method(checkpoint, None, hf) == "deepseek_v41_fp8"
    config = cls.from_config(checkpoint)
    layer = object()
    assert config.get_quant_method(layer, "model.ffn") == "upstream-kernel"
    assert calls == [
        ("recognize", (checkpoint, None, hf)),
        ("config", checkpoint),
        ("dispatch", layer, "model.ffn"),
    ]
    assert calls[1][1] is checkpoint


@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"]),
        ("deepseek_v41_text", []),
        # SpeculativeConfig verifies this temporary config before rewriting
        # it to deepseek_v41 / DSparkV41DraftModel.
        ("deepseek_mtp", ["DeepseekV41ForCausalLM"]),
    ],
)
def test_qlean_metadata_is_translated_without_editing_checkpoint(
    quant_module, model_type, architectures
):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    cls.__bases__[0].override_quantization_method = classmethod(lambda *args: None)
    checkpoint = {
        "quant_method": "mxfp4",
        "expert_dtype": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wo_a"],
        "ignore": ["layers.0.attn.compressor.wkv"],
    }
    assert (
        cls.override_quantization_method(
            checkpoint,
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        == "deepseek_v41_fp8"
    )
    result = cls.from_config(checkpoint)
    translated = calls[-1][1]
    assert translated["quant_method"] == "fp8"
    assert translated["activation_scheme"] == "dynamic"
    assert translated["ignored_layers"] == checkpoint["ignore"]
    assert result.fp8_channelwise_layers == checkpoint["fp8_channelwise_layers"]
    assert (
        checkpoint["quant_method"] == "mxfp4" and "activation_scheme" not in checkpoint
    )


@pytest.mark.parametrize("quant_method", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "ppu,model_type,architectures",
    [
        (False, "deepseek_v41", ["DeepseekV41ForCausalLM"]),
        (True, "deepseek_v4", ["DeepseekV4ForCausalLM"]),
        (True, "deepseek_v4_text", []),
        (True, "deepseek_mtp", ["DeepSeekV4MTPModel"]),
        (True, "deepseek_mtp", ["DeepSeekMTPModel"]),
        (True, "other", []),
    ],
)
def test_v41_quant_override_leaves_other_models_and_platforms_untouched(
    quant_module, quant_method, ppu, model_type, architectures
):
    module, calls = quant_module
    module.current_platform.is_ppu = lambda: ppu
    assert (
        module.DeepseekV41FP8Config.override_quantization_method(
            {
                "quant_method": quant_method,
                "fp8_channelwise_layers": ["layers.0.attn.wq_a"],
            },
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        is None
    )
    assert calls == []


@pytest.mark.parametrize("ppu", [False, True])
def test_quant_registration_is_lazy_idempotent_and_preserves_v4(modules, ppu):
    calls = []
    modules("vllm.logger", init_logger=lambda _: SimpleNamespace(debug=lambda *a: None))
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.model_executor.layers.quantization",
        QUANTIZATION_METHODS=[],
        register_quantization_config=lambda name: lambda cls: calls.append((name, cls)),
    )
    mixed = type("Mixed", (), {"get_name": staticmethod(lambda: "mixed_precision_w4")})
    v4 = type("V4", (), {})
    v41 = type("V41", (), {"get_name": staticmethod(lambda: "deepseek_v41_fp8")})
    modules(
        "vllm_sail.model_executor.layers.quantization.mixed_precision_w4",
        MixedPrecisionW4Config=mixed,
    )
    modules("vllm_sail.models.deepseek_v4.quant_config", DeepseekV4FP8Config=v4)
    modules("vllm_sail.models.deepseek_v41.quant_config", DeepseekV41FP8Config=v41)
    registry = load_patch("vllm_sail/registry/quant_config/__init__.py")
    assert calls == []
    registry.register()
    registry.register()
    assert calls == [("mixed_precision_w4", mixed)] + (
        [("deepseek_v4_fp8", v4), ("deepseek_v41_fp8", v41)] if ppu else []
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("quant_method", ["fp8", "deepseek_v4_fp8", "mxfp4"])
@pytest.mark.parametrize(
    "model_type,architectures,expected",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"], "deepseek_v41_fp8"),
        ("deepseek_v41_text", [], "deepseek_v41_fp8"),
        ("deepseek_mtp", ["DeepseekV41ForCausalLM"], "deepseek_v41_fp8"),
        ("deepseek_v4", ["DeepseekV4ForCausalLM"], "deepseek_v4_fp8"),
    ],
)
def test_upstream_model_config_selects_model_specific_quantization(
    quant_module,
    modules,
    upstream_source_root,
    quant_method,
    model_type,
    architectures,
    expected,
):
    """Exercise upstream recognition and override ordering, not a reimplemented selector."""
    from typing import Literal, cast, get_args

    module, _ = quant_module
    cls = module.DeepseekV41FP8Config
    upstream = modules("vllm.models.deepseek_v41.quant_config").DeepseekV4FP8Config
    assert cls.__bases__ == (upstream,)
    upstream.override_quantization_method = classmethod(
        function(
            upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
            "DeepseekV4FP8Config.override_quantization_method",
            {},
        )
    )

    class UpstreamV4(upstream):
        pass

    UpstreamV4.override_quantization_method = classmethod(
        function(
            upstream_source_root / "vllm/models/deepseek_v4/quant_config.py",
            "DeepseekV4FP8Config.override_quantization_method",
            {},
        )
    )
    modules("vllm.models.deepseek_v4.quant_config", DeepseekV4FP8Config=UpstreamV4)
    v4 = load_patch("vllm_sail/models/deepseek_v4/quant_config.py").DeepseekV4FP8Config
    assert v4.__bases__ == (UpstreamV4,)
    platform = module.current_platform
    platform.supported_quantization = ["fp8", "mxfp4", "deepseek_v4_fp8"]
    verified = []
    platform.verify_quantization = verified.append
    registry = {}
    names = list(platform.supported_quantization)
    register = function(
        upstream_source_root / "vllm/model_executor/layers/quantization/__init__.py",
        "register_quantization_config",
        {
            "QUANTIZATION_METHODS": names,
            "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG": registry,
            "QuantizationConfig": upstream,
            "current_platform": platform,
        },
    )
    register(cls.get_name())(cls)
    assert cls.get_name() in platform.supported_quantization
    registry["deepseek_v4_fp8"] = v4
    no_override = SimpleNamespace(override_quantization_method=lambda *a, **kw: None)
    quant = SimpleNamespace(
        QUANTIZATION_METHODS=names,
        QuantizationMethods=Literal["fp8", "mxfp4", "deepseek_v4_fp8"],
        DEPRECATED_QUANTIZATION_METHODS=[],
        get_quantization_config=lambda name: registry.get(name, no_override),
    )
    verify = function(
        upstream_source_root / "vllm/config/model.py",
        "ModelConfig._verify_quantization",
        {
            "me_quant": quant,
            "get_args": get_args,
            "cast": cast,
            "current_platform": platform,
        },
    )
    checkpoint = {"quant_method": quant_method}
    if quant_method == "mxfp4":
        checkpoint["fp8_channelwise_layers"] = ["layers.0.attn.wq_a"]
    config = SimpleNamespace(
        quantization=None,
        model_arch_config=SimpleNamespace(quantization_config=checkpoint),
        hf_config=SimpleNamespace(model_type=model_type, architectures=architectures),
    )
    verify(config)
    assert config.quantization == expected
    assert verified == [expected]
    assert registry[config.quantization] is (cls if expected == cls.get_name() else v4)
    assert checkpoint["quant_method"] == quant_method


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "model_type",
    ["deepseek_v4", "deepseek_v4_text", "deepseek_v41", "deepseek_v41_text"],
)
def test_upstream_recognizes_target_and_text_configs(upstream_source_root, model_type):
    recognize = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.override_quantization_method",
        {},
    )
    assert (
        recognize(
            object,
            {"quant_method": "fp8"},
            None,
            SimpleNamespace(model_type=model_type),
        )
        == "deepseek_v4_fp8"
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
def test_upstream_mxfp4_dispatch_preserves_target_and_draft_moe(
    upstream_source_root, modules, experts, topk
):
    class RoutedExperts:
        moe_config = SimpleNamespace(num_experts=experts, experts_per_token=topk)

    expected = object()
    modules(
        "vllm.model_executor.layers.quantization.mxfp4",
        Mxfp4MoEMethod=lambda cfg: expected,
    )
    dispatch = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.get_quant_method",
        {
            "LinearBase": type("Linear", (), {}),
            "RoutedExperts": RoutedExperts,
            "is_layer_skipped": lambda **kwargs: False,
            "Mxfp4MoEMethod": lambda cfg: (expected, cfg),
        },
    )
    config = SimpleNamespace(
        ignored_layers=[],
        packed_modules_mapping={},
        expert_dtype="fp4",
        moe_quant_algo="",
    )
    layer = RoutedExperts()
    assert dispatch(config, layer, "model.ffn.experts") == (expected, layer.moe_config)


def test_v41_selector_preserves_cuda_and_other_backends(modules):
    upstream, ppu, other = (type(name, (), {}) for name in ("Upstream", "PPU", "Other"))
    platform = SimpleNamespace(ppu=True)
    modules(
        "vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: platform.ppu)
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=upstream
    )
    modules("vllm_sail.models.deepseek_v41.flashmla", DeepseekV41FlashMLAAttention=ppu)

    def original(config):
        return config

    provider = modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _select_dsv4_attn_cls=original,
        _linear_scale_param_name=lambda *args: "weight_scale_inv",
    )
    original_scale = provider._linear_scale_param_name
    draft = modules(
        "vllm.models.deepseek_v41.nvidia.dspark",
        _linear_scale_param_name=original_scale,
    )
    vl = modules(
        "vllm.models.deepseek_v41.nvidia.vl_model",
        _linear_scale_param_name=original_scale,
    )
    replacement = load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")
    assert provider._select_dsv4_attn_cls is replacement._select_dsv4_attn_cls
    marker = getattr(provider._select_dsv4_attn_cls, replacement.PATCH_MARKER)
    assert marker[replacement._TARGET] is original
    assert provider._select_dsv4_attn_cls(upstream) is ppu
    assert provider._select_dsv4_attn_cls(other) is other
    config = SimpleNamespace(
        quant_config=SimpleNamespace(fp8_channelwise_layers=["attn.wo_a"])
    )
    assert (
        draft._linear_scale_param_name
        is vl._linear_scale_param_name
        is provider._linear_scale_param_name
    )
    assert draft._linear_scale_param_name(config, "fp4") == "weight_scale"
    platform.ppu = False
    assert provider._select_dsv4_attn_cls(upstream) is upstream
    assert draft._linear_scale_param_name(config, "fp4") == "weight_scale_inv"
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")


@pytest.mark.parametrize("dequant_at_load", [True, False])
def test_v41_projection_preserves_weights_and_uses_bf16_helper(
    modules, dequant_at_load
):
    calls = []
    loaded = SimpleNamespace(element_size=lambda: 2 if dequant_at_load else 1)
    bf16 = loaded if dequant_at_load else object()
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda *args: bf16,
    )
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.o_proj",
        deep_gemm_fp8_o_proj=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    class Upstream:
        forward_mqa = object()
        _forward_prefill = object()

    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=Upstream
    )
    modules("vllm_sail.models.deepseek_v4.flashmla", DeepseekV4FlashMLAAttention=object)
    modules("vllm.models.deepseek_v41.sparse_mla", DeepseekV4FlashMLABackend=object)
    modules("torch", int8="int8")
    modules("vllm_sail.models.deepseek_v41.config", int8_w8a8_enabled=lambda: False)
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: True, is_device_capability=lambda cap: False
        ),
    )
    module = load_patch("vllm_sail/models/deepseek_v41/flashmla.py")
    attn = module.DeepseekV41FlashMLAAttention()
    attn.__dict__.update(
        _o_proj_block_size=32,
        wo_a=SimpleNamespace(weight=loaded, weight_scale=object()),
        wo_b=object(),
        rotary_emb=SimpleNamespace(cos_sin_cache=object()),
        n_local_groups=2,
        n_local_heads=16,
        nope_head_dim=448,
        rope_head_dim=64,
        o_lora_rank=1024,
    )
    o, pos = object(), object()
    attn._o_proj(o, pos)
    args, kwargs = calls.pop()
    assert args[:2] == (o, pos) and args[3].weight is bf16 and args[4] is attn.wo_b
    assert kwargs["heads_per_group"] == 8 and kwargs["einsum_recipe"] == (1, 1, 32)
    assert attn.wo_a.weight is loaded
    assert attn.forward_mqa is Upstream.forward_mqa
    assert attn._forward_prefill is Upstream._forward_prefill


@pytest.mark.parametrize(
    "ppu,capability,contract",
    [
        (True, (8, 0), True),
        (True, (8, 0), False),
        (True, (8, 9), True),
        (True, (8, 9), False),
        (True, (9, 0), True),
        (False, (8, 0), True),
        (False, (8, 9), True),
    ],
)
@pytest.mark.parametrize("weight_dtype", ["int8", "bf16"])
@pytest.mark.parametrize("num_heads", [2, 4, 8, 16, 32, 64])
def test_v41_projection_dispatches_by_quantization(
    modules, ppu, capability, contract, weight_dtype, num_heads
):
    calls = []

    class Upstream:
        @classmethod
        def get_padded_num_q_heads(cls, heads):
            return 64

        def _o_proj(self, *args):
            calls.append(("upstream", args, {}))
            return "upstream"

    class Backend:
        @classmethod
        def supports_compute_capability(cls, cap):
            return cap.major in (9, 10)

    class PPUV4:
        @classmethod
        def get_padded_num_q_heads(cls, heads):
            return heads + 128

        def _o_proj(self, *args):
            calls.append(("ppuv4", args, {}))
            return "ppuv4"

    class Output:
        def __getitem__(self, slices):
            assert slices == (slice(None), slice(None, num_heads))
            return "actual_heads"

    def int8(*args, **kw):
        calls.append(("int8", args, kw))
        return "int8"

    def unavailable(*args):
        raise AssertionError("FlashMLA must reuse the model's published decision")

    modules("torch", int8="int8")
    modules("vllm.config", get_current_vllm_config=unavailable)
    # DeepseekV41ForCausalLM publishes is_ppu() and uses_int8_w8a8(config).
    published = SimpleNamespace(value=ppu and contract)
    modules(
        "vllm_sail.models.deepseek_v41.config",
        uses_int8_w8a8=unavailable,
        int8_w8a8_enabled=lambda: published.value,
    )
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(is_ppu=lambda: ppu),
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda *args: None,
    )
    modules(
        "vllm.models.deepseek_v4.nvidia.ops.o_proj",
        deep_gemm_fp8_o_proj=lambda *args: None,
    )
    modules("vllm_sail.models.deepseek_v4.ops.o_proj", deep_gemm_int8_o_proj=int8)
    modules("vllm_sail.models.deepseek_v4.flashmla", DeepseekV4FlashMLAAttention=PPUV4)
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla", DeepseekV4FlashMLAAttention=Upstream
    )
    modules("vllm.models.deepseek_v41.sparse_mla", DeepseekV4FlashMLABackend=Backend)
    module = load_patch("vllm_sail/models/deepseek_v41/flashmla.py")
    cls = module.DeepseekV41FlashMLAAttention
    attn = cls()
    # _o_proj uses the construction-time snapshot, not the current flag.
    published.value = not published.value
    attn.__dict__.update(
        _o_proj_block_size=128,
        wo_a=SimpleNamespace(weight=SimpleNamespace(dtype=weight_dtype)),
        _wo_b_proj=object(),
        rotary_emb=SimpleNamespace(cos_sin_cache=object()),
        n_local_groups=2,
        n_local_heads=num_heads,
        nope_head_dim=448,
        rope_head_dim=64,
        o_lora_rank=1024,
    )
    expected = (
        "int8"
        if ppu and contract and weight_dtype == "int8"
        else "ppuv4"
        if ppu
        else "upstream"
    )
    assert attn._o_proj(Output(), object()) == expected
    if expected == "int8":
        _, args, kw = calls[-1]
        assert (
            args[0] == "actual_heads"
            and args[3] is attn.wo_a
            and args[4] is attn._wo_b_proj
        )
        assert kw["einsum_recipe"] == (1, 1, 128)
        assert kw["heads_per_group"] == num_heads // attn.n_local_groups
    # PPU always uses the PPU V4 padding; the INT8 projection slices actual heads.
    expected_heads = num_heads + 128 if ppu else 64
    assert cls.get_padded_num_q_heads(num_heads) == expected_heads
    backend = module.PPUDeepseekV41FlashMLABackend
    for major, minor in ((8, 0), (8, 9), (9, 0), (10, 0)):
        assert backend.supports_compute_capability(
            SimpleNamespace(major=major, minor=minor)
        ) is (ppu or major in (9, 10))


@pytest.fixture
def v41_draft(modules):
    class Upstream:
        load_weights = object()
        forward = object()

    class SupportsQuant:
        pass

    mapper_args = []
    mapper = object()

    def make_mapper(*args):
        mapper_args.append(args)
        return mapper

    modules("vllm.model_executor.models.interfaces", SupportsQuant=SupportsQuant)
    provider = modules(
        "vllm.models.deepseek_v41.nvidia.dspark", DSparkDeepseekV4ForCausalLM=Upstream
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _make_deepseek_v4_weights_mapper=make_mapper,
    )
    module = load_patch("vllm_sail/models/deepseek_v41/dspark.py")
    draft = module.DSparkDeepseekV41ForCausalLM
    assert issubclass(draft, SupportsQuant)
    assert draft.load_weights is not Upstream.load_weights
    assert draft.load_weights.__globals__ is provider.__dict__
    assert draft.forward is Upstream.forward
    assert draft.hf_to_vllm_mapper is mapper
    assert mapper_args == [("fp4", "weight_scale")]
    assert draft.packed_modules_mapping["fused_wqa_wkv"] == ["wq_a", "wkv"]
    return draft, provider


def test_v41_draft_preserves_its_own_execution(v41_draft):
    assert "forward" not in vars(v41_draft[0])


@pytest.mark.parametrize("expert_dtype", ["int8", "int4", "fp4", "fp8"])
@pytest.mark.parametrize("dense_scale", ["weight_scale", "weight_scale_inv"])
@pytest.mark.parametrize("mega", [False, True])
@pytest.mark.parametrize("draft_experts", [0, 2])
def test_v41_draft_loads_scales_from_actual_destinations(
    v41_draft, expert_dtype, dense_scale, mega, draft_experts
):
    """Reproduce the fused attention KeyError and cover experts, shared layers and heads."""
    cls, provider = v41_draft
    calls, counts = [], []

    def mapping(count):
        counts.append(count)
        return [
            ("experts.w13_", "experts.0.w1.", 0, "w1"),
            ("experts.w13_", "experts.0.w3.", 0, "w3"),
            ("experts.w2_", "experts.0.w2.", 0, "w2"),
        ]

    provider.make_deepseek_v4_expert_params_mapping = mapping
    provider.fused_moe_make_expert_params_mapping = lambda _, **kw: mapping(
        kw["num_experts"]
    )
    provider._EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")
    provider.torch = SimpleNamespace(float8_e8m0fnu="e8m0", uint8="uint8")
    provider.get_tensor_model_parallel_world_size = lambda: 8
    provider.get_tensor_model_parallel_rank = lambda: 1
    provider.default_weight_loader = lambda p, w: calls.append((p.name, w, (), {}))
    provider.logger = SimpleNamespace(info_once=lambda *args: None)

    def parameter(name):
        def load(param, weight, *args, **kw):
            calls.append((param.name, weight, args, kw))
            return True

        return SimpleNamespace(name=name, weight_loader=load)

    expert_scale = "weight_scale_inv" if expert_dtype == "fp8" else "weight_scale"
    pairs = [
        (
            "mtp.0.attn.wq_a.scale",
            f"model.layers.0.attn.fused_wqa_wkv.{dense_scale}",
            0,
        ),
        ("mtp.0.attn.wkv.scale", f"model.layers.0.attn.fused_wqa_wkv.{dense_scale}", 1),
        ("mtp.0.attn.wq_a.weight", "model.layers.0.attn.fused_wqa_wkv.weight", 0),
        (
            "mtp.0.ffn.shared_experts.w1.scale",
            f"model.layers.0.ffn.shared_experts.gate_up_proj.{dense_scale}",
            0,
        ),
        (
            "mtp.0.ffn.shared_experts.w3.scale",
            f"model.layers.0.ffn.shared_experts.gate_up_proj.{dense_scale}",
            1,
        ),
        (
            "mtp.0.ffn.shared_experts.w2.scale",
            f"model.layers.0.ffn.shared_experts.down_proj.{dense_scale}",
            None,
        ),
        # Mixed quantization: the first scale suffix cannot represent the entire model.
        ("mtp.0.attn.wo_a.scale", "model.layers.0.attn.wo_a.weight_scale_inv", None),
        ("mtp.0.main_proj.scale", "model.main_proj.weight_scale", None),
        ("mtp.2.markov_head.embed.weight", "model.markov_head.markov_w1.weight", None),
        (
            "mtp.0.ffn.experts.0.w1.scale",
            f"model.layers.0.ffn.experts.w13_{expert_scale}",
            "w1",
        ),
        (
            "mtp.0.ffn.experts.0.w3.scale",
            f"model.layers.0.ffn.experts.w13_{expert_scale}",
            "w3",
        ),
        (
            "mtp.0.ffn.experts.0.w2.scale",
            f"model.layers.0.ffn.experts.w2_{expert_scale}",
            "w2",
        ),
        ("mtp.0.ffn.experts.0.w2.weight", "model.layers.0.ffn.experts.w2_weight", "w2"),
        (
            "mtp.0.ffn.gate.bias",
            "model.layers.0.ffn.gate.e_score_correction_bias",
            None,
        ),
    ]
    params = {target: parameter(target) for _, target, _ in pairs}
    model = cls()
    model.config = SimpleNamespace(
        n_routed_experts=384,
        dspark_n_routed_experts=draft_experts,
        expert_dtype=expert_dtype,
        num_attention_heads=64,
    )
    model.model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=mega))],
        confidence_head=object(),
    )
    model.linear_scale_name = "weight_scale_inv"
    model.pad_shared_expert = False
    model.named_parameters = lambda: params.items()
    processed = []
    model.process_weights_after_loading = lambda: processed.append(True)

    def remap(name):
        match = re.fullmatch(r"mtp\.(\d+)\.(.*)", name)
        if not match:
            return None
        stage, rest = match.groups()
        if rest.startswith("markov_head.embed."):
            return "model.markov_head.markov_w1." + rest.removeprefix(
                "markov_head.embed."
            )
        if rest.startswith("main_proj."):
            return "model." + rest
        return f"model.layers.{stage}.{rest}"

    model._remap_dspark_name = remap
    weights = [SimpleNamespace(dtype="int8") for _ in pairs]
    stream = [(name, weights[i]) for i, (name, _, _) in enumerate(pairs)]
    stream.insert(0, ("layers.0.attn.wq_a.scale", object()))
    assert model.load_weights(iter(stream)) == set(params)
    assert counts == [draft_experts or 384]
    assert processed == [True]
    assert model.model.confidence_head is None
    assert len(calls) == len(pairs)
    for call, (_, target, shard), weight in zip(calls, pairs, weights, strict=False):
        assert call[:2] == (target, weight)
        if isinstance(shard, str):
            assert call[2] == (target,)
            assert call[3] == dict(shard_id=shard, expert_id=0, return_success=True)
        else:
            assert call[2] == (() if shard is None else (shard,))
            assert call[3] == {}


def test_v41_scale_resolution_preserves_existing_and_unknown_names(v41_draft):
    resolve = v41_draft[0]._resolve_scale_name
    params = {"layer.weight_scale": object(), "layer.weight_scale_inv": object()}
    for name in (
        *params,
        "missing.weight_scale_inv",
        "layer.input_scale",
        "layer.weight",
    ):
        assert resolve(name, params) == name
    assert (
        resolve("layer.weight_scale_inv", {"layer.weight_scale": 1})
        == "layer.weight_scale"
    )
    assert (
        resolve("layer.weight_scale", {"layer.weight_scale_inv": 1})
        == "layer.weight_scale_inv"
    )


@pytest.mark.upstream_source
def test_v41_draft_loader_keeps_upstream_body_except_scale_resolution(
    upstream_source_root,
):
    """Match the installed V4.1 loader after removing marked changes, without V4 context-KV."""
    path = harness.ROOT / "vllm_sail/models/deepseek_v41/dspark.py"
    lines, inside = [], False
    for line in path.read_text().splitlines():
        if "PPU MODIFICATION: begin" in line:
            inside = True
        elif "PPU MODIFICATION: end" in line:
            inside = False
        elif not inside:
            lines.append(line)
    tree = ast.parse("\n".join(lines))
    local = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    upstream = definition(
        upstream_source_root / "vllm/models/deepseek_v41/nvidia/dspark.py",
        "DSparkDeepseekV4ForCausalLM.load_weights",
    )
    # Moving the function out of its class changes docstring indentation, not logic.
    local.body[0] = upstream.body[0]
    assert ast.dump(local, include_attributes=False) == ast.dump(
        upstream, include_attributes=False
    )


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("int8", [False, True])
def test_v41_model_resolves_int8_at_construction_like_v4(
    modules, monkeypatch, ppu, int8
):
    calls = []
    published = []
    selected = ppu and int8
    original_mapper, mapped = object(), object()
    config = SimpleNamespace(quant_config=None)

    class Base:
        hf_to_vllm_mapper = original_mapper

        def __init__(self, **kwargs):
            # Upstream layers are built here and must already see the decision.
            calls.append(("construct", kwargs, list(published)))

        def forward(self, *args, **kwargs):
            return 17

    def recognize(value):
        assert value is config
        calls.append(("recognize",))
        return int8

    def map_weights(mapper):
        assert mapper is original_mapper
        calls.append(("map",))
        return mapped

    def unavailable(*args, **kwargs):
        raise AssertionError("the V4.1 model must not touch the global config")

    modules(
        "vllm.config",
        set_current_vllm_config=unavailable,
        get_current_vllm_config=unavailable,
    )
    modules("vllm.models.deepseek_v41.nvidia.vl_model", DeepseekV41ForCausalLM=Base)
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm_sail.models.deepseek_v41.config",
        uses_int8_w8a8=recognize,
        set_int8_w8a8=published.append,
    )
    module = load_patch("vllm_sail/models/deepseek_v41/model.py")
    monkeypatch.setattr(module, "int8_weights_mapper", map_weights)
    model = module.DeepseekV41ForCausalLM(vllm_config=config, prefix="model")
    assert [call[0] for call in calls] == (
        (["recognize"] if ppu else []) + ["construct"] + (["map"] if selected else [])
    )
    construct = next(call for call in calls if call[0] == "construct")
    assert construct[1:] == (dict(vllm_config=config, prefix="model"), [selected])
    assert published == [selected]
    assert model.hf_to_vllm_mapper is (mapped if selected else original_mapper)
    assert Base.hf_to_vllm_mapper is original_mapper
    # Like the PPU V4 model, forward is inherited unchanged.
    assert "forward" not in vars(module.DeepseekV41ForCausalLM)
    assert model.forward(1, 2, 3, 4, lookback_token_ids=5) == 17
    assert not hasattr(model, "_v41_int8_config")
