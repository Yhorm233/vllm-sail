# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

import ast
import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.source import ROOT, definition, function

_MEGAMOE_IMPL_MODULE = (
    "vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe"
)
_MEGAMOE_IMPL = ROOT / (_MEGAMOE_IMPL_MODULE.replace(".", "/") + ".py")
_MEGAMOE_PATCH = ROOT / "vllm_sail/patch/enhancement/mega_moe.py"
_MEGAMOE_API = ROOT / "vllm_sail/utils/deep_gemm_mega.py"
_MEGAMOE_MODEL = "vllm.models.deepseek_v4.nvidia.model"
_MEGAMOE_PROVIDER = "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe"
_MEGAMOE_EXPERTS = _MEGAMOE_PROVIDER.replace(".", "/") + ".py"
_MEGAMOE_GATE_PROVIDER = "vllm.utils.deep_gemm"
_KIMI_MODEL = "vllm.models.kimi_k3.nvidia.model"
_MINIMAX_MODEL = "vllm.models.minimax_m3.nvidia.model"
_MEGAMOE_CLASS = "PPUDeepGemmMegaMoEExperts"
_MEGAMOE_COMMON_APIS = (
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch",
)
_MEGAMOE_MODE_APIS = {
    "fp4xfp4": ("fp4_fp4_mega_moe",),
    "fp8xfp4": ("fp8_fp4_mega_moe", "transform_weights_for_w4a8_mega_moe"),
    "bf16xfp4": ("bf16_fp4_mega_moe", "transform_weights_for_w4a16_mega_moe"),
    "int8xint8": ("int8_int8_mega_moe", "transform_weights_for_mega_moe"),
}


@pytest.fixture
def megamoe_patch(modules):
    class NativeExperts:
        pass

    class PPUExperts(NativeExperts):
        pass

    class MoE:
        def forward(self):
            return "native forward"

        def _init_mega_moe_experts(self):
            self.experts = model.DeepGemmMegaMoEExperts()

    def native_gate(*args, **kwargs):
        return "native gate"

    platform = SimpleNamespace(ppu=True)
    platform.is_ppu = lambda: platform.ppu
    modules("vllm.platforms", current_platform=platform)
    provider = modules(_MEGAMOE_PROVIDER, DeepGemmMegaMoEExperts=NativeExperts)
    gate_provider = modules(_MEGAMOE_GATE_PROVIDER, bf16_mega_gate=native_gate)
    model = modules(
        _MEGAMOE_MODEL,
        DeepseekV4MoE=MoE,
        DeepGemmMegaMoEExperts=NativeExperts,
        bf16_mega_gate=native_gate,
    )
    modules(_MEGAMOE_IMPL_MODULE, PPUDeepGemmMegaMoEExperts=PPUExperts)
    return SimpleNamespace(
        platform=platform,
        model=model,
        native=NativeExperts,
        ppu=PPUExperts,
        provider=provider,
        gate_provider=gate_provider,
        native_gate=native_gate,
        original_init=MoE._init_mega_moe_experts,
        original_forward=MoE.forward,
        patch=load_patch(_MEGAMOE_PATCH),
    )


def test_megamoe_redirects_provider_and_preloaded_models(megamoe_patch, modules):
    harness = megamoe_patch
    kimi = modules(_KIMI_MODEL, DeepGemmMegaMoEExperts=harness.native)
    minimax = modules(_MINIMAX_MODEL, DeepGemmMegaMoEExperts=harness.native)
    harness.patch.install()
    harness.patch.install()

    for owner in (harness.provider, harness.model, kimi, minimax):
        assert owner.DeepGemmMegaMoEExperts is harness.ppu
    assert harness.model.DeepseekV4MoE._init_mega_moe_experts is harness.original_init
    assert harness.model.DeepseekV4MoE.forward is harness.original_forward
    moe = harness.model.DeepseekV4MoE()
    moe._init_mega_moe_experts()
    assert type(moe.experts) is harness.ppu
    assert moe.forward() == "native forward"


def test_megamoe_redirects_late_imports_and_preserves_other_bindings(
    megamoe_patch, modules
):
    harness = megamoe_patch
    other_experts = object()
    kimi = modules(_KIMI_MODEL, DeepGemmMegaMoEExperts=other_experts)
    harness.patch.install()

    minimax = modules(_MINIMAX_MODEL)
    exec(f"from {_MEGAMOE_PROVIDER} import DeepGemmMegaMoEExperts", vars(minimax))
    assert minimax.DeepGemmMegaMoEExperts is harness.ppu
    assert kimi.DeepGemmMegaMoEExperts is other_experts


def test_megamoe_leaves_native_gate_unmodified(megamoe_patch):
    harness = megamoe_patch
    harness.patch.install()

    assert harness.gate_provider.bf16_mega_gate is harness.native_gate
    assert harness.model.bf16_mega_gate is harness.native_gate


def test_megamoe_leaves_non_ppu_unmodified(megamoe_patch, monkeypatch):
    harness = megamoe_patch
    harness.platform.ppu = False
    monkeypatch.setitem(sys.modules, _MEGAMOE_IMPL_MODULE, None)
    harness.patch.install()

    assert harness.provider.DeepGemmMegaMoEExperts is harness.native
    assert harness.model.DeepGemmMegaMoEExperts is harness.native
    assert harness.gate_provider.bf16_mega_gate is harness.native_gate
    assert harness.model.bf16_mega_gate is harness.native_gate


def test_megamoe_inherits_native_expert_mapping_and_eplb_interfaces():
    local = definition(_MEGAMOE_IMPL, _MEGAMOE_CLASS)
    assert [ast.unparse(base) for base in local.bases] == ["DeepGemmMegaMoEExperts"]
    local_methods = {
        node.name for node in local.body if isinstance(node, ast.FunctionDef)
    }
    assert not local_methods.intersection(
        {
            "_map_global_expert_id",
            "set_eplb_state",
            "get_expert_weights",
            "update_expert_map",
            "has_fused_shared_experts",
            "layer_id",
        }
    )


@pytest.mark.upstream_source
def test_megamoe_targets_and_inherited_interfaces_exist_in_main(upstream_source_root):
    upstream_model = upstream_source_root / "vllm/models/deepseek_v4/nvidia/model.py"
    for name in (
        "_map_global_expert_id",
        "set_eplb_state",
        "get_expert_weights",
        "update_expert_map",
        "has_fused_shared_experts",
        "layer_id",
    ):
        definition(
            upstream_source_root / _MEGAMOE_EXPERTS, f"DeepGemmMegaMoEExperts.{name}"
        )
    definition(upstream_model, "DeepseekV4MoE._init_mega_moe_experts")


@pytest.mark.upstream_source
@pytest.mark.parametrize("method", ["__init__", "forward", "weight_loader"])
def test_megamoe_signatures_accept_upstream_keywords(upstream_source_root, method):
    upstream = function(
        upstream_source_root / _MEGAMOE_EXPERTS,
        f"DeepGemmMegaMoEExperts.{method}",
        {},
    )
    local = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}.{method}", {})
    offered = inspect.signature(upstream)
    accepted = inspect.signature(local)
    keyword_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    for include_defaults in (False, True):
        kwargs = {
            name: object()
            for name, param in offered.parameters.items()
            if param.kind in keyword_kinds
            and (include_defaults or param.default is inspect.Parameter.empty)
        }
        accepted.bind(**kwargs)
    for kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        if any(param.kind == kind for param in offered.parameters.values()):
            assert any(param.kind == kind for param in accepted.parameters.values())


@pytest.fixture
def megamoe_loader():
    loader = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}.weight_loader", {})

    def prepare(local_ids):
        mapped, indexed, loaded = [], [], []
        slots = {slot: object() for slot in local_ids}

        class Data:
            def __getitem__(self, slot):
                indexed.append(slot)
                return slots[slot]

        def map_expert(expert_id):
            mapped.append(expert_id)
            return local_ids

        expert = SimpleNamespace(
            _map_global_expert_id=map_expert,
            _load_weight=lambda *args: loaded.append(args),
        )
        return SimpleNamespace(
            loader=loader,
            expert=expert,
            param=SimpleNamespace(data=Data()),
            mapped=mapped,
            indexed=indexed,
            loaded=loaded,
            slots=slots,
        )

    return prepare


@pytest.mark.parametrize("shard_id", ["w1", "w3", "w2"])
@pytest.mark.parametrize("is_scale", [False, True])
@pytest.mark.parametrize("local_ids", [(0, 4), ()], ids=["replicas", "nonlocal"])
@pytest.mark.parametrize("return_success", [False, None, True])
def test_megamoe_weight_loader_routes_replicas_and_skips_nonlocal(
    megamoe_loader, shard_id, is_scale, local_ids, return_success
):
    harness = megamoe_loader(local_ids)
    bank = "w2" if shard_id == "w2" else "w13"
    weight_name = f"model.layers.3.ffn.experts.{bank}_weight"
    if is_scale:
        weight_name += "_scale"
    source = object()
    result = harness.loader(
        harness.expert,
        param=harness.param,
        loaded_weight=source,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=2,
        return_success=return_success,
    )
    assert result is (bool(local_ids) if return_success else None)
    assert harness.mapped == [2]
    assert harness.indexed == list(local_ids)
    assert harness.loaded == [
        (harness.slots[slot], source, shard_id, is_scale) for slot in local_ids
    ]


@pytest.mark.parametrize(
    "shard_id,wrong_bank", [("w1", "w2"), ("w3", "w2"), ("w2", "w13")]
)
@pytest.mark.parametrize("return_success", [False, True])
def test_megamoe_weight_loader_skips_mismatched_parameter_bank(
    megamoe_loader, shard_id, wrong_bank, return_success
):
    harness = megamoe_loader((0, 4))
    result = harness.loader(
        harness.expert,
        harness.param,
        object(),
        f"{wrong_bank}_weight",
        shard_id,
        2,
        return_success=return_success,
    )
    assert result is (False if return_success else None)
    assert harness.indexed == harness.loaded == []


def test_megamoe_weight_loader_defaults_to_none_and_rejects_unknown_shards(
    megamoe_loader,
):
    harness = megamoe_loader((0,))
    result = harness.loader(
        harness.expert, harness.param, object(), "w2_weight", "w2", 2
    )
    assert result is None
    assert harness.indexed == [0]
    harness.indexed.clear()
    harness.loaded.clear()
    with pytest.raises(ValueError, match="Unsupported expert shard id"):
        harness.loader(harness.expert, harness.param, object(), "w13_weight", "w9", 2)
    assert harness.indexed == harness.loaded == []


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "start,end,logical_id,local_ids",
    [(6, 12, 2, (0, 4)), (6, 12, 0, (2,)), (8, 10, 2, ())],
    ids=["redundant-replicas", "offset-slot", "nonlocal"],
)
def test_megamoe_weight_loader_uses_upstream_logical_to_physical_map(
    megamoe_loader, upstream_source_root, start, end, logical_id, local_ids
):
    harness = megamoe_loader(local_ids)
    mapping = function(
        upstream_source_root / _MEGAMOE_EXPERTS,
        "DeepGemmMegaMoEExperts._map_global_expert_id",
        {},
    )
    harness.expert.__dict__.update(
        experts_start_idx=start, experts_end_idx=end, num_logical_experts=4
    )
    harness.expert._map_global_expert_id = lambda expert_id: mapping(
        harness.expert, expert_id
    )
    source = object()
    result = harness.loader(
        harness.expert,
        harness.param,
        source,
        "w13_weight",
        "w1",
        logical_id,
        return_success=True,
    )
    assert result is bool(local_ids)
    assert harness.indexed == list(local_ids)
    assert harness.loaded == [
        (harness.slots[slot], source, "w1", False) for slot in local_ids
    ]


@pytest.fixture
def megamoe_api_module(monkeypatch):
    calls = []
    original_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "deep_gemm.mega":
            calls.append(name)
            raise ImportError("DeepGEMM is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    for name in ("deep_gemm", "torch", "vllm"):
        monkeypatch.setitem(sys.modules, name, None)
    module = load_patch(_MEGAMOE_API)
    assert calls == []
    return module, calls


def test_megamoe_api_import_is_lazy_and_reports_missing_package(megamoe_api_module):
    module, calls = megamoe_api_module
    with pytest.raises(
        RuntimeError, match=r"DeepGEMM-for-sail.*deep_gemm\.mega"
    ) as error:
        module.mega_moe_api("fp8xfp4")
    assert isinstance(error.value.__cause__, ImportError)
    assert calls == ["deep_gemm.mega"]


@pytest.mark.parametrize(
    "mode,missing",
    [
        (mode, api)
        for mode, mode_apis in _MEGAMOE_MODE_APIS.items()
        for api in (*_MEGAMOE_COMMON_APIS, *mode_apis)
    ],
)
@pytest.mark.parametrize("noncallable", [False, True], ids=["absent", "noncallable"])
def test_megamoe_api_reports_missing_required_functions_and_allows_retry(
    megamoe_api_module, monkeypatch, mode, missing, noncallable
):
    module, calls = megamoe_api_module
    required = (*_MEGAMOE_COMMON_APIS, *_MEGAMOE_MODE_APIS[mode])
    api = SimpleNamespace(**{name: lambda: None for name in required})
    if noncallable:
        setattr(api, missing, object())
    else:
        delattr(api, missing)

    def import_module(name):
        calls.append(name)
        return api

    monkeypatch.setattr(
        module, "importlib", SimpleNamespace(import_module=import_module)
    )
    with pytest.raises(RuntimeError, match=missing):
        module.mega_moe_api(mode)
    setattr(api, missing, lambda: None)
    view = module.mega_moe_api(mode)
    assert module.mega_moe_api(mode) is view
    if mode == "int8xint8":
        # INT8 wraps the module to pin the pre-dispatch group size.
        assert view.int8_int8_mega_moe is api.int8_int8_mega_moe
    else:
        assert view is api
    assert calls == ["deep_gemm.mega", "deep_gemm.mega"]


@pytest.mark.parametrize("mode", _MEGAMOE_MODE_APIS)
def test_megamoe_api_reports_all_missing_functions(
    megamoe_api_module, monkeypatch, mode
):
    module, _ = megamoe_api_module
    monkeypatch.setattr(
        module, "importlib", SimpleNamespace(import_module=lambda _: SimpleNamespace())
    )
    with pytest.raises(RuntimeError) as error:
        module.mega_moe_api(mode)
    for name in (*_MEGAMOE_COMMON_APIS, *_MEGAMOE_MODE_APIS[mode]):
        assert name in str(error.value)


@pytest.mark.parametrize(
    "mode,hidden,intermediate,supported",
    [
        ("fp4xfp4", 2048, 64, True),
        ("fp4xfp4", 2560, 128, True),
        ("fp4xfp4", 512, 128, False),
        ("fp4xfp4", 2048, 0, False),
        ("fp4xfp4", 2048, 96, False),
        ("fp4xfp4", 2304, 128, False),
        ("fp8xfp4", 2304, 128, True),
        ("fp8xfp4", 2048, 64, False),
        ("fp8xfp4", 2176, 128, False),
        ("bf16xfp4", 2304, 128, True),
        ("bf16xfp4", 2048, 0, False),
        ("bf16xfp4", 0, 128, False),
        ("int8xint8", 2048, 256, True),
        ("int8xint8", 5120, 2304, True),
        ("int8xint8", 1024, 256, False),
        ("int8xint8", 2048, 128, False),
        ("fp4xfp4", 1, None, True),
        ("fp8xfp4", 1, None, True),
        ("bf16xfp4", 1, None, True),
        ("int8xint8", 1, None, True),
    ],
)
def test_megamoe_geometry_matches_wheel_contract(
    modules, monkeypatch, mode, hidden, intermediate, supported
):
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: True,
            has_device_capability=lambda capability, device_id: capability == 89,
            get_device_capability=lambda device_id: SimpleNamespace(major=8, minor=0),
        ),
    )
    modules("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: True)
    module = load_patch(_MEGAMOE_API)
    calls = []
    monkeypatch.setattr(module, "mega_moe_api", calls.append)
    device = SimpleNamespace(index=0)
    if supported:
        module.check_mega_moe_supported(device, hidden, intermediate, mode)
        assert calls == [mode]
    else:
        with pytest.raises(ValueError):
            module.check_mega_moe_supported(device, hidden, intermediate, mode)
        assert not calls


@pytest.mark.parametrize("missing", ["ppu", "m890", "deep_gemm", "api"])
def test_megamoe_unknown_geometry_still_checks_hardware_and_api(modules, missing):
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: missing != "ppu",
            has_device_capability=lambda *args, **kwargs: missing != "m890",
            get_device_capability=lambda device_id: SimpleNamespace(major=8, minor=9),
        ),
    )
    modules(
        "vllm_sail.utils.deep_gemm",
        is_deep_gemm_supported=lambda: missing != "deep_gemm",
    )
    modules("deep_gemm.mega")
    module = load_patch(_MEGAMOE_API)
    error = RuntimeError if missing == "api" else NotImplementedError
    match = "API is missing" if missing == "api" else "requires Zhenwu M890"
    with pytest.raises(error, match=match):
        module.check_mega_moe_supported(SimpleNamespace(index=0), 1, None, "fp8xfp4")


@pytest.fixture
def megamoe_platform_gate(modules):
    calls = []
    modules("vllm_sail.ops")
    modules(
        "vllm_sail.models.deepseek_v41.config",
        normalize_expert_dtype=lambda config: None,
        uses_int8_w8a8=lambda config: False,
    )
    envs = modules("vllm_sail.envs", VLLM_SAIL_MEGA_MOE_MMA_TYPE="fp8xfp4")
    api = load_patch(_MEGAMOE_API)
    modules(
        "vllm_sail.utils.deep_gemm_mega",
        check_mega_moe_supported=lambda *args: calls.append(args),
        resolve_expert_dtype=api.resolve_expert_dtype,
        mma_type_for_expert_dtype=api.mma_type_for_expert_dtype,
    )
    modules(
        "torch",
        device=lambda kind, index: (kind, index),
        cuda=SimpleNamespace(current_device=lambda: 0),
    )
    gate = function(
        "vllm_sail/platform.py",
        "PPUPlatform.apply_config_platform_defaults",
        {"logger": SimpleNamespace(info_once=lambda *args: None)},
    )
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        model_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(enable_eplb=True),
        compilation_config=SimpleNamespace(custom_ops=[]),
        attention_config=SimpleNamespace(),
        speculative_config=None,
    )
    return gate, config, envs, calls


@pytest.mark.parametrize("model_type", ["deepseek_v41", "kimi_k31", "other_moe"])
@pytest.mark.parametrize("mode", _MEGAMOE_MODE_APIS)
@pytest.mark.parametrize("intermediate", [None, 128])
def test_megamoe_platform_gate_accepts_models_and_eplb(
    megamoe_platform_gate, model_type, mode, intermediate
):
    gate, config, envs, calls = megamoe_platform_gate
    text_config = SimpleNamespace(model_type=model_type, hidden_size=2048)
    if mode == "int8xint8":
        text_config.expert_dtype = "int8"
    if intermediate is not None:
        text_config.moe_intermediate_size = intermediate
    config.model_config.hf_text_config = text_config
    envs.VLLM_SAIL_MEGA_MOE_MMA_TYPE = mode
    gate(None, config)
    assert calls == [(("cuda", 0), 2048, intermediate, mode)]


def test_megamoe_platform_gate_rejects_declared_non_fp4_experts(megamoe_platform_gate):
    gate, config, _, calls = megamoe_platform_gate
    config.model_config.hf_text_config = SimpleNamespace(
        model_type="other_moe", hidden_size=2048, expert_dtype="fp8"
    )
    with pytest.raises(ValueError, match="requires MXFP4 experts"):
        gate(None, config)
    assert calls == []


def test_megamoe_platform_gate_pins_int8_mode_for_int8_experts(megamoe_platform_gate):
    gate, config, envs, calls = megamoe_platform_gate
    # The configured MXFP4 mode must not leak into an INT8 checkpoint.
    envs.VLLM_SAIL_MEGA_MOE_MMA_TYPE = "fp8xfp4"
    config.model_config.hf_text_config = SimpleNamespace(
        model_type="deepseek_v41", hidden_size=5120, expert_dtype="int8"
    )
    gate(None, config)
    assert calls == [(("cuda", 0), 5120, None, "int8xint8")]


def test_megamoe_mma_type_resolution(modules, monkeypatch):
    modules(
        "vllm_sail.models.deepseek_v41.config",
        uses_int8_w8a8=lambda config: getattr(config, "w8a8", False),
    )
    envs = modules("vllm_sail.envs", VLLM_SAIL_MEGA_MOE_MMA_TYPE="fp8xfp4")
    module = load_patch(_MEGAMOE_API)
    w8a8 = SimpleNamespace(w8a8=True)
    fp4 = SimpleNamespace(
        w8a8=False,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(expert_dtype="fp4")
        ),
    )
    assert module.resolve_expert_dtype(w8a8) == "int8"
    assert module.resolve_expert_dtype(fp4) == "fp4"
    assert module.resolve_expert_dtype(SimpleNamespace()) == "fp4"
    assert module.mma_type_for_expert_dtype("int8") == "int8xint8"
    assert module.mma_type_for_expert_dtype("fp4") == "fp8xfp4"
    envs.VLLM_SAIL_MEGA_MOE_MMA_TYPE = "int8xint8"
    with pytest.raises(ValueError, match="requires INT8 experts"):
        module.mma_type_for_expert_dtype("fp4")


def test_megamoe_api_int8_pins_pre_dispatch_group_size(modules, monkeypatch):
    module = load_patch(_MEGAMOE_API)
    seen = []

    def dispatch(*args, **kwargs):
        seen.append(kwargs)

    required = (
        *_MEGAMOE_COMMON_APIS,
        *_MEGAMOE_MODE_APIS["int8xint8"],
    )
    api = SimpleNamespace(**{name: lambda: None for name in required})
    api.mega_moe_pre_dispatch = dispatch
    monkeypatch.setattr(
        module, "importlib", SimpleNamespace(import_module=lambda _: api)
    )
    view = module.mega_moe_api("int8xint8")
    assert view.int8_int8_mega_moe is api.int8_int8_mega_moe
    view.mega_moe_pre_dispatch()
    view.mega_moe_pre_dispatch(group_size=7)
    assert seen == [{"group_size": 0}, {"group_size": 7}]
    # MXFP4 modes keep the raw module so the shared forward stays untouched.
    mxfp4 = SimpleNamespace(
        **{
            name: lambda: None
            for name in (*_MEGAMOE_COMMON_APIS, *_MEGAMOE_MODE_APIS["fp8xfp4"])
        }
    )
    monkeypatch.setattr(
        module, "importlib", SimpleNamespace(import_module=lambda _: mxfp4)
    )
    assert module.mega_moe_api("fp8xfp4") is mxfp4


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("with_shared", [False, True])
@pytest.mark.parametrize(
    "mode,activation",
    [
        ("fp4xfp4", "swiglu"),
        ("fp8xfp4", "swiglu"),
        ("fp4xfp4", "situ"),
        ("fp8xfp4", "situ"),
        ("bf16xfp4", "situ"),
    ],
)
def test_expert_forward_uses_sail_dispatch_and_main_shared_contract(
    tokens, with_shared, mode, activation
):
    calls = []

    class Tensor:
        shape = (tokens, 5120)

        def to(self, dtype):
            calls.append(("dtype", dtype))
            return self

        def contiguous(self):
            return self

        def __iadd__(self, other):
            assert other == "shared-output"
            calls.append("add-shared")
            return self

    hidden, ids, weights, output = Tensor(), Tensor(), Tensor(), Tensor()
    buffer = SimpleNamespace(
        x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
    )
    l1, l2 = object(), object()
    kernel_name = _MEGAMOE_MODE_APIS[mode][0]
    expected_kwargs = {"fast_math": False, "activation": activation}
    if activation == "situ":
        expected_kwargs.update(situ_beta=1.7, situ_linear_beta=-0.2)
    else:
        expected_kwargs["activation_clamp"] = 7.0
    if mode == "fp4xfp4":
        expected_kwargs["recipe"] = (1, 1, 32)

    def dispatch(
        x,
        routed_ids,
        routed_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        num_tokens,
        group_size=32,
        mma_type="fp4xfp4",
    ):
        assert (x, routed_ids, routed_weights) == (hidden, ids, weights)
        assert buf_x is buffer.x
        assert buf_x_sf is buffer.x_sf
        assert buf_topk_idx is buffer.topk_idx
        assert buf_topk_weights is buffer.topk_weights
        assert num_tokens == tokens
        assert group_size == 32 and mma_type == mode
        calls.append("dispatch")

    def kernel(y, first, second, symm, **kwargs):
        assert (y, first, second, symm) == (output, l1, l2, buffer)
        assert kwargs == expected_kwargs
        calls.append("kernel")

    def shared(x):
        assert x is hidden
        calls.append("shared")
        return "shared-output"

    api = SimpleNamespace(mega_moe_pre_dispatch=dispatch, **{kernel_name: kernel})

    def get_api(mma_type):
        assert mma_type == mode
        return api

    def transform_weights(actual_api):
        assert actual_api is api
        calls.append("transform")

    def kernel_weights(name):
        assert name in ("w13", "w2")
        return l1 if name == "w13" else l2

    class Expert(SimpleNamespace):
        @property
        def l1_weights(self):
            return self._kernel_weights("w13")

        @property
        def l2_weights(self):
            return self._kernel_weights("w2")

    ns = {
        "torch": SimpleNamespace(
            int32="i32",
            float32="fp32",
            bfloat16="bf16",
            empty_like=lambda x, dtype: output,
        ),
        "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
        "mega_moe_api": get_api,
    }
    forward = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}.forward", ns)
    kernel_kwargs = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}._kernel_kwargs", {})
    expert = Expert(
        max_num_tokens=4,
        _synchronize_first_launch=lambda: calls.append("sync"),
        capture_fn=lambda value: calls.append(("capture", value)),
        eplb_state=SimpleNamespace(logical_to_physical_map=None),
        get_symm_buffer=lambda: buffer,
        mma_type=mode,
        mma_format=SimpleNamespace(
            kernel=kernel_name,
            transform=None if mode == "fp4xfp4" else "transform",
        ),
        activation=SimpleNamespace(name=activation, alpha=1.7, beta=-0.2, clamp=7.0),
        fast_math=False,
        _kernel_weights=kernel_weights,
        _transform_weights=transform_weights,
    )
    expert._kernel_kwargs = lambda: kernel_kwargs(expert)
    assert (
        forward(expert, hidden, weights, ids, shared if with_shared else None) is output
    )
    assert calls == [
        "sync",
        ("capture", ids),
        "transform",
        ("dtype", "i32"),
        ("dtype", "fp32"),
        "dispatch",
        "kernel",
    ] + (["shared", "add-shared"] if with_shared else [])
    expert.max_num_tokens = -1
    with pytest.raises(ValueError, match="exceeds capacity"):
        forward(expert, hidden, weights, ids)


def test_expert_rejects_reload_after_in_place_transform():
    load_weight = function(_MEGAMOE_IMPL, f"{_MEGAMOE_CLASS}._load_weight", {})
    expert = SimpleNamespace(_transformed=True)
    with pytest.raises(RuntimeError, match="cannot be reloaded"):
        load_weight(expert, None, None, "w2", False)
