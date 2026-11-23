# SPDX-License-Identifier: Apache-2.0
"""Exercise installed GDN selection patches and their upstream call sites."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import function
from vllm_sail import envs
from vllm_sail.attention import gdn_decode


@pytest.fixture
def gdn_provider(modules, monkeypatch, upstream_source_root):
    monkeypatch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    monkeypatch.delenv("VLLM_PPU_USE_PLA", raising=False)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")
    modules("vllm_sail.envs", **envs.__dict__)
    modules("vllm_sail.attention.gdn_decode", **gdn_decode.__dict__)
    calls = []
    platform = NS(
        ppu=True,
        is_ppu=lambda: platform.ppu,
        is_rocm=lambda: False,
        has_device_capability=lambda _: True,
    )
    modules("vllm.platforms", current_platform=platform)
    logger = NS(
        info_once=lambda *a: None,
        warning_once=lambda *a: None,
        warning=lambda *a: pytest.fail(f"意外的环境因子警告：{a}"),
    )
    modules("vllm.logger", init_logger=lambda _: logger)
    torch = NS(
        float32="fp32",
        bfloat16="bf16",
        int32="int32",
        _C=NS(_dispatch_has_kernel_for_dispatch_key=lambda *a: False),
        ops=NS(_C=NS(fused_gdn_decode_post_conv_mtp=lambda *a: None)),
    )
    modules("torch", **torch.__dict__)
    modules(
        "vllm_sail.attention.pla_decode",
        get_sail_cuda_pla_k_last=lambda: calls.append("pla probe") or object(),
        get_sail_cuda_pla_k_last_packed=lambda: object(),
    )

    def original_getter():
        import os

        value = os.getenv("VLLM_GDN_DECODE_KERNEL", "cuda")
        if value not in ("cuda", "triton"):
            raise ValueError("upstream choice rejected")
        return value

    def ignored_getter():
        pytest.fail("上游忽略的缓存路径及凭据 getter 不应求值")

    env_module = modules(
        "vllm.envs",
        environment_variables={
            "VLLM_GDN_DECODE_KERNEL": original_getter,
            "SAIL_TEST_UNRELATED": lambda: 7,
            "VLLM_CACHE_ROOT": ignored_getter,
            "VLLM_API_KEY": ignored_getter,
        },
        os=os,
        logger=logger,
    )
    env_module.__getattr__ = lambda name: env_module.environment_variables[name]()
    # 真实 normalize_value 的 primitive 分支；不导入 Torch/config 全模块。
    modules(
        "vllm.config.utils",
        normalize_value=function(
            upstream_source_root / "vllm/config/utils.py", "normalize_value", {}
        ),
    )
    original_compile_factors = function(
        upstream_source_root / "vllm/envs.py", "compile_factors", env_module.__dict__
    )
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    namespace = dict(
        torch=torch,
        current_platform=platform,
        FUSED_GDN_STATE_DTYPES=("fp32", "bf16"),
        MAX_FUSED_GDN_MTP_TOKENS=8,
    )

    supports = function(path, "supports_fused_mtp_decode", namespace)

    class Layer:
        def __init__(self, config, vllm_config, prefix=""):
            self.head_k_dim = self.head_v_dim = 128
            self.num_k_heads, self.num_v_heads = 2, 4
            self.gqa_interleaved_layout = False
            self._ba_stream = None
            self._fused_gdn_gated_rms_norm = None
            self.norm = NS(activation="silu")
            self.state_dtype = vllm_config.state_dtype
            self.speculative_config = vllm_config.speculative_config
            self.gdn_decode_kernel = env_module.VLLM_GDN_DECODE_KERNEL
            if self.gdn_decode_kernel == "cuda":
                reason = self._fused_gdn_decode_unsupported_reason(vllm_config)
                if reason:
                    raise ValueError(reason)
            self.enable_fused_gdn_spec_decode = (
                self.gdn_decode_kernel == "cuda" and self.speculative_config is not None
            )

        def get_state_dtype(self):
            return "bf16", self.state_dtype

    for name in (
        "_fused_gdn_decode_unsupported_reason",
        "_can_use_fused_gdn_mtp_decode",
    ):
        setattr(
            Layer, name, function(path, f"QwenGatedDeltaNetAttention.{name}", namespace)
        )
    provider = modules(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        QwenGatedDeltaNetAttention=Layer,
        supports_fused_mtp_decode=supports,
    )
    originals = {
        name: getattr(Layer, name)
        for name in (
            "__init__",
            "_fused_gdn_decode_unsupported_reason",
            "_can_use_fused_gdn_mtp_decode",
        )
    }
    original_getters = env_module.environment_variables
    load_patch("vllm_sail/patch/enhancement/attention/gdn_decode.py")
    return NS(
        layer=Layer,
        env=env_module,
        torch=torch,
        platform=platform,
        calls=calls,
        path=path,
        config=NS(
            state_dtype="fp32",
            model_config=NS(dtype="bf16"),
            speculative_config=object(),
        ),
        provider=provider,
        originals=originals,
        original_getters=original_getters,
        original_compile_factors=original_compile_factors,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "attribute",
    [
        "environment_variables",
        "compile_factors",
        "__init__",
        "_fused_gdn_decode_unsupported_reason",
        "_can_use_fused_gdn_mtp_decode",
    ],
)
def test_decode_patches_record_originals_and_reject_reapplication(
    gdn_provider, attribute
):
    from vllm_sail.patch.utils import PATCH_MARKER, patch

    f = gdn_provider
    if attribute in ("environment_variables", "compile_factors"):
        module, target = "vllm.envs", attribute
        replacement = getattr(f.env, attribute)
        original = (
            f.original_getters
            if attribute == "environment_variables"
            else f.original_compile_factors
        )
    else:
        module = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
        target = f"QwenGatedDeltaNetAttention.{attribute}"
        replacement, original = getattr(f.layer, attribute), f.originals[attribute]
    assert replacement is not original
    assert getattr(replacement, PATCH_MARKER)[f"{module}.{target}"] is original
    with pytest.raises(RuntimeError, match="already patched"):
        patch(
            module,
            target,
            reason="GDN lifecycle regression",
            affected_versions=">=0.31.0,<0.32.0",
            remove_when="upstream registers platform GDN backends",
        )(lambda *a, **kw: None)


@pytest.mark.upstream_source
def test_stub_cannot_select_native_mtp_even_with_registered_schema(
    gdn_provider, monkeypatch
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    assert f.provider.supports_fused_mtp_decode() is True
    layer = f.layer(None, f.config)
    assert layer.gdn_decode_kernel == "triton"
    assert not layer.enable_fused_gdn_spec_decode
    layer.gdn_decode_kernel = "cuda"  # Also guard stale/prebuilt layer state.
    layer.kv_cache = (None, NS(dtype="fp32"))
    metadata = NS(
        spec_state_indices_tensor=NS(size=lambda _: 3),
        spec_sequence_masks=object(),
        num_decodes=0,
        num_spec_decodes=1,
    )
    assert not layer._can_use_fused_gdn_mtp_decode(metadata)


@pytest.mark.upstream_source
@pytest.mark.parametrize("requested", ["auto", "pla", "triton"])
def test_extended_choices_reach_layer_initialization(
    gdn_provider, monkeypatch, requested
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", requested)
    if requested == "pla":
        monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    layer = f.layer(None, f.config)
    assert f.env.VLLM_GDN_DECODE_KERNEL == requested
    assert f.env.environment_variables["VLLM_GDN_DECODE_KERNEL"]() == requested
    assert layer.gdn_decode_kernel == ("pla" if requested == "auto" else requested)
    assert not layer.enable_fused_gdn_spec_decode
    assert bool(f.calls) is (requested != "triton")


@pytest.mark.upstream_source
def test_explicit_cuda_requires_real_dispatch_kernel(gdn_provider, monkeypatch):
    f = gdn_provider
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "cuda")
    with pytest.raises(ValueError, match="CUDA implementation"):
        f.layer(None, f.config)
    f.torch._C._dispatch_has_kernel_for_dispatch_key = lambda *a: True
    layer = f.layer(None, f.config)
    assert layer.gdn_decode_kernel == "cuda"
    assert layer.enable_fused_gdn_spec_decode
    assert not f.calls


@pytest.mark.upstream_source
def test_explicit_pla_rejects_bf16_state_without_changing_cache_dtype(
    gdn_provider, monkeypatch
):
    f = gdn_provider
    f.config.state_dtype = "bf16"
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "pla")
    with pytest.raises(ValueError, match="float32.*mamba-ssm-cache-dtype"):
        f.layer(None, f.config)
    assert f.config.state_dtype == "bf16"
    assert not f.calls


@pytest.mark.upstream_source
def test_non_ppu_retains_upstream_parser_and_selection(gdn_provider, monkeypatch):
    f = gdn_provider
    f.platform.ppu = False
    f.torch._C._dispatch_has_kernel_for_dispatch_key = lambda *a: True
    assert f.layer(None, f.config).gdn_decode_kernel == "cuda"
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "pla")
    with pytest.raises(ValueError, match="upstream choice rejected"):
        f.layer(None, f.config)


@pytest.mark.upstream_source
def test_stub_falls_through_real_upstream_forward_to_normal_core(
    gdn_provider, monkeypatch
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    layer = f.layer(None, f.config)
    _exercise_real_gdn_forward(f, layer, None)


def _exercise_real_gdn_forward(f, layer, qkvz_input):
    calls = []
    projections = []

    class Tensor:
        dtype = "bf16"
        device = "cuda"

        def size(self, dim):
            return 3

        def split(self, sizes, **kwargs):
            return tuple(Tensor() for _ in sizes)

        def reshape(self, *shape):
            return self

        def contiguous(self):
            return self

    def fused(*args, **kwargs):
        raise AssertionError("The unsupported fused GDN op must never be reached")

    f.torch.ops.vllm = NS(
        qwen_gdn_attention_core=lambda *a, **kw: calls.append("normal core"),
        qwen_gdn_attention_core_fused_norm_packed=fused,
    )
    f.torch.zeros = lambda *a, **kw: Tensor()
    f.torch.empty = lambda *a, **kw: Tensor()

    def project(name, value):
        projections.append((name, value))
        return Tensor(), None

    layer.in_proj_qkvz = lambda value: project("qkvz", value)
    layer.in_proj_ba = lambda value: project("ba", value)
    layer.split_ba = lambda _: (Tensor(), Tensor())
    layer._output_projection = lambda *a: "norm + output"
    layer.key_dim, layer.value_dim, layer.tp_size, layer.prefix = 256, 512, 1, "gdn"
    forward = function(
        f.path,
        "QwenGatedDeltaNetAttention.forward_cuda",
        dict(torch=f.torch, _encode_layer_name=lambda name: name),
    )
    hidden = Tensor()
    assert forward(layer, hidden, qkvz_input=qkvz_input) == "norm + output"
    assert calls == ["normal core"]
    assert projections == [
        ("qkvz", hidden if qkvz_input is None else qkvz_input),
        ("ba", hidden),
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("prequantized", [False, True])
def test_qwen35_direct_cuda_entry_preserves_projection_inputs_and_ppu_core(
    gdn_provider, monkeypatch, prequantized
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    layer = f.layer(None, f.config)
    quantized = object() if prequantized else None
    _exercise_real_gdn_forward(f, layer, quantized)


@pytest.mark.upstream_source
@pytest.mark.parametrize("speculative", [False, True])
def test_auto_native_selection_updates_the_spec_decode_flag(
    gdn_provider, monkeypatch, speculative
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "0")
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "auto")
    f.torch._C._dispatch_has_kernel_for_dispatch_key = lambda *a: True
    f.config.speculative_config = object() if speculative else None
    layer = f.layer(None, f.config)
    assert layer.gdn_decode_kernel == "cuda"
    assert layer.enable_fused_gdn_spec_decode is speculative


def _real_compile_cache_helpers(f, upstream_source_root, mega):
    # Config 和 Inductor 是明确边界替身；执行真实共享 helper 和 AOT 调用者。
    hash_factors = function(
        upstream_source_root / "vllm/config/utils.py",
        "hash_factors",
        dict(hashlib=hashlib, json=json),
    )
    namespace = dict(
        envs=f.env,
        hash_factors=hash_factors,
        get_inductor_factors=lambda: ["test-inductor-boundary"],
    )
    path = upstream_source_root / "vllm/compilation/caching.py"
    shared = function(path, "compute_env_and_config_hashes", namespace)
    aot = function(path, "aot_compile_hash_factors", namespace)
    f.env.VLLM_USE_MEGA_AOT_ARTIFACT = mega
    config_factors = {"model": {"dtype": "bf16"}}
    config = NS(compile_factors=lambda: config_factors)
    return hash_factors, shared, aot, config, config_factors


@pytest.mark.upstream_source
@pytest.mark.parametrize("requested", ["auto", "cuda", "pla", "triton"])
@pytest.mark.parametrize("mega", [False, True])
def test_real_jit_aot_factors_track_decode_and_resolved_pla_preference(
    gdn_provider, monkeypatch, upstream_source_root, requested, mega
):
    f = gdn_provider
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", requested)
    f.torch._C._dispatch_has_kernel_for_dispatch_key = lambda *a: True
    hash_factors, shared, aot, config, config_factors = _real_compile_cache_helpers(
        f, upstream_source_root, mega
    )
    for key, getter in f.original_getters.items():
        if key != "VLLM_GDN_DECODE_KERNEL":
            assert f.env.environment_variables[key] is getter
    assert f.original_compile_factors.__globals__ is f.env.__dict__
    hashes = {}
    # 规范名、legacy alias 以及显式 false/空值覆盖 alias 的优先级。
    for canonical, legacy, use_pla in (
        ("1", None, True),
        (None, "1", True),
        ("0", "1", False),
        ("", "1", False),
        (None, "0", False),
    ):
        for name, value in (
            ("VLLM_SAIL_USE_PLA", canonical),
            ("VLLM_PPU_USE_PLA", legacy),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        env_hash, config_hash, raw_env, raw_config = shared(config)
        assert raw_env["VLLM_GDN_DECODE_KERNEL"] == requested
        assert raw_env["VLLM_SAIL_USE_PLA"] is use_pla
        assert "VLLM_PPU_USE_PLA" not in raw_env
        assert raw_env["SAIL_TEST_UNRELATED"] == 7
        assert "VLLM_CACHE_ROOT" not in raw_env
        assert "VLLM_API_KEY" not in raw_env
        assert raw_config is config_factors
        assert config_hash == hash_factors(config_factors)
        assert env_hash == hash_factors(raw_env)
        expected_aot = [env_hash, config_hash]
        if mega:
            expected_aot.append("test-inductor-boundary")
        assert aot(config) == expected_aot
        layer = f.layer(None, f.config)
        active = "pla" if requested == "auto" and use_pla else requested
        if active == "auto":
            active = "cuda"
        assert layer.gdn_decode_kernel == active
        assert layer.enable_fused_gdn_spec_decode is (active == "cuda")
        if use_pla in hashes:
            assert env_hash == hashes[use_pla]
        hashes[use_pla] = env_hash
    assert hashes[True] != hashes[False]


@pytest.mark.upstream_source
@pytest.mark.parametrize("mega", [False, True])
def test_non_ppu_jit_aot_factors_preserve_upstream_values_and_result_ownership(
    gdn_provider, monkeypatch, upstream_source_root, mega
):
    f = gdn_provider
    f.platform.ppu = False
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    hash_factors, shared, aot, config, config_factors = _real_compile_cache_helpers(
        f, upstream_source_root, mega
    )
    for value in ("0", "1"):
        monkeypatch.setenv("VLLM_SAIL_USE_PLA", value)
        original = f.original_compile_factors()
        assert "VLLM_SAIL_USE_PLA" not in original
        result = shared(config)
        assert result == (
            hash_factors(original),
            hash_factors(config_factors),
            original,
            config_factors,
        )
        expected = list(result[:2])
        if mega:
            expected.append("test-inductor-boundary")
        assert aot(config) == expected
    # 非 PPU wrapper 必须直接返回 provider 的对象，不复制或追加因子。
    original = f.original_compile_factors()
    monkeypatch.setitem(
        f.env.compile_factors.__globals__, "_upstream_compile_factors", lambda: original
    )
    assert f.env.compile_factors() is original
