# SPDX-License-Identifier: Apache-2.0
"""CPU-safe contract tests for the four PPU MoE oracle extensions."""

from __future__ import annotations

import ast
import importlib
from enum import Enum
from types import SimpleNamespace

import pytest

from tests.support.source import function, load_moe_oracle
from vllm_sail.registry.moe_backends._extend import extend_enum


def test_moe_backend_package_is_importable_without_vllm() -> None:
    package = importlib.import_module("vllm_sail.registry.moe_backends")
    assert callable(package.extend_enum)
    assert callable(package.register)


_MOE_SOURCE_CASES = [
    ("fp8", "Fp8MoeBackend", "Fp8MoEKernelOracle", "select_fp8_moe_backend"),
    ("int8", "Int8MoeBackend", "Int8MoEKernelOracle", "select_int8_moe_backend"),
    (
        "unquantized",
        "UnquantizedMoeBackend",
        "UnquantizedMoEKernelOracle",
        "select_unquantized_moe_backend",
    ),
    (
        "int_wna16",
        "WNA16MoEBackend",
        "WNA16MoEKernelOracle",
        "select_wna16_moe_backend",
    ),
]


@pytest.fixture
def moe_source_boundary(modules, upstream_source_root):
    """读取真实 getter、格式及支持检查；其他准入谓词为边界替身，不证明设备能力。"""
    ns = SimpleNamespace
    modules(
        "vllm.model_executor.layers.quantization.online.requantization",
        is_requantization_source=lambda: False,
    )
    root = upstream_source_root / "vllm/model_executor/layers/fused_moe"
    formats = function(
        root / "modular_kernel.py", "FusedMoEActivationFormat", {"Enum": Enum}
    )
    modules(
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        FusedMoEActivationFormat=formats,
    )
    config_cls = type(
        "SourceMoEConfig",
        (ns,),
        {
            name: property(function(root / "config.py", "FusedMoEConfig." + name, {}))
            for name in ("activation_format", "use_batched_activation_format")
        },
    )

    def config(format_name, backend="auto"):
        return config_cls(
            moe_backend=backend,
            moe_parallel_config=ns(
                use_deepep_ll_kernels=format_name == "BatchedExperts",
                use_nixl_ep_kernels=False,
                use_deepep_v2_kernels=format_name == "PaddedStandard",
                use_moonep_kernels=False,
                use_alltoall_batched_kernels=False,
            ),
            is_act_and_mul=True,
            activation="silu",
            has_hash_routing=False,
            require_decomposed_backend=False,
            routing_method="topk",
            router_logits_dtype=None,
            hidden_dim=128,
            is_lora_enabled=False,
            has_bias=False,
        )

    admission = function(
        root / "modular_kernel.py",
        "FusedMoEExperts.is_supported_config",
        {
            "envs": ns(VLLM_BATCH_INVARIANT=False),
            "check_activation_quant_fallback": function(
                root.parent / "quantization/utils/quant_utils.py",
                "check_activation_quant_fallback",
                {},
            ),
        },
    )

    def expert(format_name):
        return type(
            "SourceExperts" + format_name,
            (),
            {
                "activation_format": staticmethod(lambda: formats[format_name]),
                "is_supported_config": staticmethod(admission),
                "executed_activation_key": staticmethod(
                    function(
                        root / "modular_kernel.py",
                        "FusedMoEExperts.executed_activation_key",
                        {},
                    )
                ),
                "is_monolithic": staticmethod(lambda: False),
                **{
                    name: staticmethod(lambda *args: True)
                    for name in (
                        "_supports_current_device",
                        "_supports_no_act_and_mul",
                        "_supports_activation",
                        "_supports_quant_scheme",
                        "_supports_parallel_config",
                        "_supports_routing_method",
                        "_supports_router_logits_dtype",
                        "_supports_shape",
                    )
                },
            },
        )

    experts = {}
    for module, cls_name, format_name in (
        ("deep_gemm_moe", "PPUDeepGemmExperts", "Standard"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExperts", "BatchedExperts"),
    ):
        cls = expert(format_name)
        cls.__name__ = cls_name
        cls.activation_format = staticmethod(
            function(
                "vllm_sail/model_executor/layers/fused_moe/experts/" + module + ".py",
                cls_name + ".activation_format",
                {"mk": ns(FusedMoEActivationFormat=formats)},
            )
        )
        experts[format_name] = cls
    return ns(root=root, formats=formats, config=config, expert=expert, experts=experts)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "ll,nixl,v2",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
    ],
)
@pytest.mark.parametrize("backend", ["auto", "batched_triton"])
def test_real_moe_config_getter_owns_activation_format(
    moe_source_boundary, ll, nixl, v2, backend
):
    boundary = moe_source_boundary
    config = boundary.config("Standard", backend)
    parallel = config.moe_parallel_config
    parallel.use_deepep_ll_kernels = ll
    parallel.use_nixl_ep_kernels = nixl
    parallel.use_deepep_v2_kernels = v2
    expected = (
        "BatchedExperts"
        if ll or nixl or backend == "batched_triton"
        else "PaddedStandard"
        if v2
        else "Standard"
    )
    assert not hasattr(parallel, "use_batched_activation_format")
    assert config.activation_format is boundary.formats[expected]
    assert config.use_batched_activation_format is (expected == "BatchedExperts")


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "expert_format", ["Standard", "BatchedExperts", "PaddedStandard"]
)
@pytest.mark.parametrize(
    "input_format", ["Standard", "BatchedExperts", "PaddedStandard"]
)
def test_real_expert_admission_uses_directional_format_relation(
    moe_source_boundary, expert_format, input_format
):
    boundary = moe_source_boundary
    cls = boundary.expert(expert_format)
    supported, reason = cls.is_supported_config(
        cls, boundary.config(input_format), None, None, boundary.formats[input_format]
    )
    expected = expert_format == input_format or (expert_format, input_format) == (
        "PaddedStandard",
        "Standard",
    )
    assert supported is expected
    assert reason is None if expected else "activation format" in reason


@pytest.mark.upstream_source
@pytest.mark.parametrize("case", _MOE_SOURCE_CASES)
@pytest.mark.parametrize(
    "format_name", ["Standard", "BatchedExperts", "PaddedStandard"]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_real_public_moe_selectors_preserve_ppu_format_boundary(
    modules, moe_source_boundary, case, format_name, explicit
):
    """执行真实 adapter、委托 selector 和共享 oracle；优先级及非格式准入单独验证。"""
    boundary = moe_source_boundary
    module, enum_name, oracle_name, select_name = case
    path = boundary.root / "oracle" / (module + ".py")
    backend = function(path, enum_name, {"Enum": Enum})
    for name in ("PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM"):
        extend_enum(backend, name, name)
    if module in ("int8", "unquantized"):
        extend_enum(backend, "ACEXT", "ACEXT")
    config = boundary.config(format_name, "ppu_deep_gemm" if explicit else "auto")
    candidates = [backend.PPU_DEEPGEMM, backend.BATCHED_PPU_DEEPGEMM]
    if module == "fp8":
        candidates += [backend.VLLM_CUTLASS, backend.BATCHED_VLLM_CUTLASS]
    rejected, selected = [], []
    logger = SimpleNamespace(
        info_once=lambda *a, **kw: selected.append(a),
        debug_once=lambda *a, **kw: rejected.append(a),
    )
    platform = SimpleNamespace(
        is_ppu=lambda: True,
        is_cuda=lambda: True,
        is_rocm=lambda: False,
        is_tpu=lambda: False,
        is_xpu=lambda: False,
        is_out_of_tree=lambda: False,
    )
    modules("vllm.platforms", current_platform=platform)
    namespace = {
        enum_name: backend,
        "mk": SimpleNamespace(FusedMoEActivationFormat=boundary.formats),
        "current_platform": platform,
        "envs": SimpleNamespace(is_set=lambda name: False),
        "ppu_envs": SimpleNamespace(is_set=lambda name: False),
        "logger": logger,
        "oracle": SimpleNamespace(
            kInt8StaticChannelSym="weight",
            kInt8DynamicTokenSym="activation",
            logger=logger,
        ),
        "kInt8StaticChannelSym": "weight",
        "kInt8DynamicTokenSym": "activation",
        "_get_priority_backends": lambda *args: list(candidates),
        "backend_to_kernel_cls": lambda item: [
            boundary.experts[
                "BatchedExperts" if item == backend.BATCHED_PPU_DEEPGEMM else "Standard"
            ]
        ],
        "get_current_vllm_config": lambda: SimpleNamespace(model_config=None),
        "_backend_incompatibility_reason": lambda *args: None,
        "_PPU_ALIASES": ("ppu_deep_gemm", "ppu_deep_gemm_w4a16"),
    }
    namespace[
        "_upstream_map_" + ("wna16" if module == "int_wna16" else module) + "_backend"
    ] = function(
        path,
        "map_" + ("wna16" if module == "int_wna16" else module) + "_backend",
        namespace,
    )
    patch_path = "vllm_sail/registry/moe_backends/" + module + ".py"
    function(
        patch_path,
        "map_" + ("wna16" if module == "int_wna16" else module) + "_backend",
        namespace,
    )
    if module == "int_wna16":
        namespace.update(
            user_activation_override=lambda *args: (None, None),
            get_current_vllm_config_or_none=lambda: None,
            check_activation_quant_fallback=lambda *args: None,
        )
        for helper in (
            "_requested_activation_key",
            "_backend_activation_key",
            "_filter_by_activation",
            "_activation_fallback_reason",
        ):
            function(path, helper, namespace)
    resolved_oracle = load_moe_oracle(path, oracle_name, namespace)
    setattr(namespace["oracle"], oracle_name, resolved_oracle)
    if module == "int_wna16":
        namespace["oracle"]._requested_activation_key = namespace[
            "_requested_activation_key"
        ]
        namespace["oracle"]._activation_fallback_reason = namespace[
            "_activation_fallback_reason"
        ]
    namespace["_upstream_" + select_name] = function(path, select_name, namespace)
    if module in ("fp8", "int_wna16"):
        function(
            patch_path,
            "_select_ppu_backend" if module == "fp8" else "_select_batched_ppu_backend",
            namespace,
        )
    select = function(patch_path, select_name, namespace)
    args = (
        (config,)
        if module == "unquantized"
        else (config, "weight", None, False, False)
        if module == "int_wna16"
        else (config, "weight", "activation")
    )
    if format_name == "PaddedStandard":
        with pytest.raises(ValueError if explicit else NotImplementedError):
            select(*args)
        assert selected == []
        if module != "fp8" or not explicit:
            assert any("activation format" in entry[0] for entry in rejected)
    else:
        result = select(*args)
        expected_backend = (
            backend.BATCHED_PPU_DEEPGEMM
            if format_name == "BatchedExperts"
            else backend.PPU_DEEPGEMM
        )
        assert result == (expected_backend, boundary.experts[format_name])


@pytest.mark.parametrize("case", _MOE_SOURCE_CASES)
def test_explicit_batched_ppu_requantization_source_never_selects_compute(case):
    module, enum_name, oracle_name, select_name = case
    placeholder = (object(), object())
    config = SimpleNamespace(
        moe_backend="ppu_deep_gemm",
        activation_format=SimpleNamespace(is_batched=True),
        is_lora_enabled=False,
    )
    oracle = SimpleNamespace(maybe_skip_for_requantization_source=lambda: placeholder)
    provider = SimpleNamespace(kInt8StaticChannelSym=None, kInt8DynamicTokenSym=None)
    setattr(provider, oracle_name, oracle)
    env = {
        "oracle": provider,
        enum_name: SimpleNamespace(PPU_DEEPGEMM=object()),
        "_PPU_ALIASES": ("ppu_deep_gemm",),
    }
    path = f"vllm_sail/registry/moe_backends/{module}.py"
    if module == "fp8":
        function(path, "_select_ppu_backend", env)
    if module == "int_wna16":
        function(path, "_select_batched_ppu_backend", env)
    args = {
        "unquantized": (),
        "fp8": (None, None),
        "int8": (None, None),
        "int_wna16": (None, None, False, False),
    }[module]
    assert function(path, select_name, env)(config, *args) is placeholder


def _real_unquantized_lora_selector(modules, boundary, accelerated):
    """读取真实 selector/共享 oracle；仅替换设备准入及专家计算边界。"""
    path = boundary.root / "oracle/unquantized.py"
    backend = function(path, "UnquantizedMoeBackend", {"Enum": Enum})
    events, logs = [], []
    platform = SimpleNamespace(
        is_tpu=lambda: False,
        is_xpu=lambda: False,
        is_out_of_tree=lambda: False,
    )

    def lora_supported(config):
        events.append(("lora-support", config))
        return accelerated

    class LoRAExperts:
        @staticmethod
        def is_supported_config(cls, config, weight, activation, fmt):
            events.append(("expert", cls, config, weight, activation, fmt))
            return True, None

    modules(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe",
        TrtLlmBf16LoRAExperts=LoRAExperts,
    )

    def mapping(item):
        events.append(("mapping", item))
        assert item is backend.TRITON
        return [LoRAExperts]

    namespace = {
        "UnquantizedMoeBackend": backend,
        "mk": SimpleNamespace(FusedMoEActivationFormat=boundary.formats),
        "current_platform": platform,
        "logger": SimpleNamespace(
            info_once=lambda *a, **kw: logs.append(a),
            debug_once=lambda *a, **kw: logs.append(a),
        ),
        "_trtllm_bf16_lora_supported": lora_supported,
        "backend_to_kernel_cls": mapping,
    }
    load_moe_oracle(path, "UnquantizedMoEKernelOracle", namespace)
    upstream = function(path, "select_unquantized_moe_backend", namespace)
    namespace["_upstream_select_unquantized_moe_backend"] = upstream
    select = function(
        "vllm_sail/registry/moe_backends/unquantized.py",
        "select_unquantized_moe_backend",
        namespace,
    )
    return select, backend, LoRAExperts, events, logs


@pytest.mark.upstream_source
@pytest.mark.parametrize("format_name", ["BatchedExperts", "PaddedStandard"])
@pytest.mark.parametrize("backend_name", ["auto", "ppu_deep_gemm", "ppu_acext"])
def test_real_unquantized_lora_rejects_nonstandard_before_ppu_probe(
    modules, moe_source_boundary, format_name, backend_name
):
    boundary = moe_source_boundary
    select, _, _, events, logs = _real_unquantized_lora_selector(
        modules, boundary, accelerated=True
    )
    config = boundary.config(format_name, backend_name)
    config.is_lora_enabled = True
    config.moe_parallel_config.all2all_backend = "source-boundary"
    with pytest.raises(
        ValueError, match="MoE LoRA requires a Standard activation layout"
    ) as error:
        select(config)
    message = str(error.value)
    assert "source-boundary" in message and backend_name in message
    assert config.activation_format.value[0] in message
    assert events == [] and logs == []


@pytest.mark.upstream_source
@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("backend_name", ["auto", "ppu_deep_gemm", "ppu_acext"])
def test_real_unquantized_lora_standard_preserves_upstream_expert_choice(
    modules, moe_source_boundary, accelerated, backend_name
):
    boundary = moe_source_boundary
    select, backend, cls, events, logs = _real_unquantized_lora_selector(
        modules, boundary, accelerated
    )
    config = boundary.config("Standard", backend_name)
    config.is_lora_enabled = True
    expected = backend.FLASHINFER_TRTLLM if accelerated else backend.TRITON
    assert select(config) == (expected, cls)
    assert events == [
        ("lora-support", config),
        *(
            []
            if accelerated
            else [
                ("mapping", backend.TRITON),
                ("expert", cls, config, None, None, config.activation_format),
            ]
        ),
    ]
    assert len(logs) == 1 and "LoRA" in logs[0][0]


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "case",
    _MOE_SOURCE_CASES
    + [
        ("mxfp4", "Mxfp4MoeBackend", "Mxfp4MoEKernelOracle", "select_mxfp4_moe_backend")
    ],
)
def test_real_shared_oracle_observes_live_mapping_and_bf16_override(
    moe_source_boundary, case
):
    boundary = moe_source_boundary
    module, enum_name, oracle_name, _ = case
    path = boundary.root / "oracle" / (module + ".py")
    backend = function(path, enum_name, {"Enum": Enum})
    candidate = next(iter(backend))
    probes = []
    first, second = type("First", (), {}), type("Second", (), {})
    for cls in (first, second):
        cls.is_supported_config = staticmethod(
            lambda *args: (probes.append(args) or True, None)
        )
    namespace = {
        enum_name: backend,
        "logger": SimpleNamespace(debug_once=lambda *a: None),
        "backend_to_kernel_cls": lambda item: [first],
    }
    oracle_cls = load_moe_oracle(path, oracle_name, namespace)
    config = boundary.config("Standard")
    oracle = oracle_cls(config, "weights", "checkpoint-activation")
    assert oracle.try_backend(candidate) == (first, [])
    namespace["backend_to_kernel_cls"] = lambda item: [second]
    assert oracle.try_backend(candidate, activation_key=None) == (second, [])
    assert oracle.try_backend(candidate, activation_key="override") == (second, [])
    assert [args[0] for args in probes] == [first, second, second]
    assert [args[2:] for args in probes] == [
        ("weights", key, config.activation_format)
        for key in ("checkpoint-activation", None, "override")
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("expert_format", ["Standard", "BatchedExperts"])
@pytest.mark.parametrize(
    "input_format", ["Standard", "BatchedExperts", "PaddedStandard"]
)
def test_wna16_bound_maker_uses_shared_format_guard_before_expert_construction(
    modules, moe_source_boundary, expert_format, input_format
):
    """绑定真实 copied maker 和共享 helper；不声称完整模块安装或设备计算通过。"""
    from vllm_sail.patch.bodies import bind_body

    boundary = moe_source_boundary
    ns = SimpleNamespace
    cls = boundary.experts[expert_format]
    built, prepare_calls = [], []
    cls.__init__ = lambda self, **kwargs: built.append(kwargs)
    prepare = ns(
        activation_format=boundary.formats[input_format],
        max_num_tokens_per_rank=lambda: 37,
        num_dispatchers=lambda: 3,
    )
    config, quant, routing = (
        boundary.config(input_format, "ppu_deep_gemm"),
        object(),
        object(),
    )

    def prepare_finalize(**kwargs):
        prepare_calls.append(kwargs)
        return prepare

    shared = function(
        boundary.root / "oracle/base.py",
        "make_moe_kernel",
        {
            "maybe_make_prepare_finalize": prepare_finalize,
            "logger": ns(info_once=lambda *a: None, debug_once=lambda *a: None),
            "mk": ns(
                FusedMoEKernel=lambda pf, experts: ns(
                    prepare_finalize=pf, experts=experts
                )
            ),
        },
    )
    for module, name in (
        ("cpu_moe", "CPUExpertsInt4"),
        ("int4_emulation_moe", "Int4EmulationTritonExperts"),
        ("rdna3_moe", "Rdna3WNA16Experts"),
        ("rdna3_int4_int8_moe", "Rdna3Int4Int8Experts"),
        ("xpu_moe", "XPUExpertsWNA16"),
        ("zentorch_moe", "ZentorchExpertsWNA16"),
    ):
        modules(
            "vllm.model_executor.layers.fused_moe.experts." + module,
            **{name: type(name, (), {})},
        )
    modules(
        "vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe",
        PPUDeepGemmExperts=boundary.experts["Standard"],
    )
    modules(
        "vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe",
        PPUBatchedDeepGemmExperts=boundary.experts["BatchedExperts"],
    )
    backend = function(
        boundary.root / "oracle/int_wna16.py", "WNA16MoEBackend", {"Enum": Enum}
    )
    provider = modules(
        "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
        WNA16MoEBackend=backend,
        make_moe_kernel=shared,
        **{
            name: type(name, (), {})
            for name in (
                "MarlinExperts",
                "BatchedMarlinExperts",
                "TritonWNA16Experts",
                "TrtLlmMxint4ExpertsMonolithic",
            )
        },
    )
    copied = function(
        "vllm_sail/registry/moe_backends/int_wna16.py",
        "make_wna16_moe_kernel",
        {"WNA16MoEBackend": backend},
    )
    maker = bind_body(copied, provider)
    assert maker.__globals__ is provider.__dict__
    if expert_format != input_format:
        with pytest.raises(ValueError, match="expects .* activations"):
            maker(quant, config, cls, routing_tables=routing)
        assert built == []
    else:
        result = maker(quant, config, cls, routing_tables=routing)
        assert result.prepare_finalize is prepare and isinstance(result.experts, cls)
        assert built == [
            dict(
                moe_config=config,
                quant_config=quant,
                **(
                    dict(max_num_tokens=37, num_dispatchers=3)
                    if input_format == "BatchedExperts"
                    else {}
                ),
            )
        ]
    assert prepare_calls == [
        dict(
            moe=config,
            quant_config=quant,
            routing_tables=routing,
            allow_new_interface=True,
            use_monolithic=False,
        )
    ]


@pytest.fixture
def nvfp4_factory_boundary(moe_source_boundary):
    """真实 factory 与共享 helper；PF、专家构造和 tensor 计算是明确的外部边界。"""
    boundary = moe_source_boundary
    ns = SimpleNamespace
    path = boundary.root / "oracle/nvfp4.py"
    backend = function(path, "NvFp4MoeBackend", {"Enum": Enum})
    assignments = [
        node.value
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PER_TOKEN_NVFP4_MOE_BACKENDS"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    value = assignments[0]
    assert isinstance(value, ast.Call) and ast.unparse(value.func) == "frozenset"
    assert not value.keywords and len(value.args) == 1
    assert isinstance(value.args[0], ast.Set)
    elements = value.args[0].elts
    assert all(
        isinstance(element, ast.Attribute)
        and ast.unparse(element.value) == "NvFp4MoeBackend"
        for element in elements
    )
    allowed = frozenset(backend[element.attr] for element in elements)
    assert {item.name for item in allowed} == {
        "FLASHINFER_CUTEDSL",
        "FLASHINFER_TRTLLM",
    }
    events = []
    prepare = ns(
        activation_format=boundary.formats.Standard,
        max_num_tokens_per_rank=lambda: 37,
        num_dispatchers=lambda: 3,
    )

    def prepare_finalize(**kwargs):
        events.append(("prepare", kwargs))
        return prepare

    def assemble(pf, experts):
        events.append(("kernel", pf, experts))
        return ns(prepare_finalize=pf, experts=experts)

    shared = function(
        boundary.root / "oracle/base.py",
        "make_moe_kernel",
        {
            "maybe_make_prepare_finalize": prepare_finalize,
            "logger": ns(info_once=lambda *a: None, debug_once=lambda *a: None),
            "mk": ns(FusedMoEKernel=assemble),
        },
    )
    maker = function(
        path,
        "make_nvfp4_moe_kernel",
        {
            "NvFp4MoeBackend": backend,
            "PER_TOKEN_NVFP4_MOE_BACKENDS": allowed,
            "make_moe_kernel": shared,
        },
    )
    cls = boundary.expert("Standard")
    cls.__init__ = lambda self, **kwargs: events.append(("experts", kwargs))
    config = boundary.config("Standard")
    config.in_dtype = object()
    return ns(
        backend=backend,
        allowed=allowed,
        maker=maker,
        shared=shared,
        config=config,
        quant=ns(use_nvfp4_w4a16=False),
        routing=object(),
        cls=cls,
        prepare=prepare,
        events=events,
        boundary=boundary,
    )


_NVFP4_BACKENDS = [
    "NONE",
    "B12X",
    "FLASHINFER_TRTLLM",
    "FLASHINFER_CUTLASS",
    "FLASHINFER_CUTEDSL",
    "FLASHINFER_CUTEDSL_BATCHED",
    "FLASHINFER_B12X",
    "VLLM_CUTLASS",
    "MARLIN",
    "HUMMING",
    "EMULATION",
    "FLASHINFER_MOE_EP_CUTEDSL",
]


@pytest.mark.upstream_source
@pytest.mark.parametrize("backend_name", _NVFP4_BACKENDS)
@pytest.mark.parametrize("per_token", [False, True])
def test_nvfp4_real_factory_rejects_unsafe_per_token_before_prepare(
    nvfp4_factory_boundary, backend_name, per_token
):
    f = nvfp4_factory_boundary
    assert {item.name for item in f.backend} == set(_NVFP4_BACKENDS)
    backend = f.backend[backend_name]
    if per_token and backend not in f.allowed:
        with pytest.raises(ValueError) as raised:
            f.maker(f.quant, f.config, f.cls, backend, f.routing, per_token)
        assert str(raised.value) == (
            f"Per-token NVFP4 activation scales are not supported by {backend.value}."
        )
        assert f.events == []
        return
    result = f.maker(f.quant, f.config, f.cls, backend, f.routing, per_token)
    assert result.prepare_finalize is f.prepare and isinstance(result.experts, f.cls)
    assert f.events == [
        (
            "prepare",
            dict(
                moe=f.config,
                quant_config=f.quant,
                routing_tables=f.routing,
                allow_new_interface=True,
                use_monolithic=False,
                input_dtype=f.config.in_dtype if per_token else None,
            ),
        ),
        (
            "experts",
            dict(
                moe_config=f.config,
                quant_config=f.quant,
                **({"per_token_activation": True} if per_token else {}),
            ),
        ),
        ("kernel", f.prepare, result.experts),
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("per_token", [False, True])
def test_nvfp4_marlin_a16_drops_per_token_metadata(nvfp4_factory_boundary, per_token):
    f = nvfp4_factory_boundary
    f.quant.use_nvfp4_w4a16 = True
    result = f.maker(f.quant, f.config, f.cls, f.backend.MARLIN, f.routing, per_token)
    assert f.events[0][1]["input_dtype"] is None
    assert f.events[1] == ("experts", dict(moe_config=f.config, quant_config=f.quant))
    assert result.prepare_finalize is f.prepare and isinstance(result.experts, f.cls)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "expert_format", ["Standard", "BatchedExperts", "PaddedStandard"]
)
@pytest.mark.parametrize(
    "input_format", ["Standard", "BatchedExperts", "PaddedStandard"]
)
def test_nvfp4_factory_keeps_shared_directional_format_guard(
    nvfp4_factory_boundary, expert_format, input_format
):
    f = nvfp4_factory_boundary
    f.prepare.activation_format = f.boundary.formats[input_format]
    cls = f.boundary.expert(expert_format)
    cls.__init__ = lambda self, **kwargs: f.events.append(("experts", kwargs))
    accepted = expert_format == input_format or (expert_format, input_format) == (
        "PaddedStandard",
        "Standard",
    )
    if not accepted:
        with pytest.raises(ValueError, match="expects .* activations"):
            f.maker(f.quant, f.config, cls, f.backend.VLLM_CUTLASS, f.routing)
        assert [event[0] for event in f.events] == ["prepare"]
        return
    result = f.maker(f.quant, f.config, cls, f.backend.VLLM_CUTLASS, f.routing)
    assert result.prepare_finalize is f.prepare and isinstance(result.experts, cls)
    assert f.events[1] == (
        "experts",
        dict(
            moe_config=f.config,
            quant_config=f.quant,
            **(
                dict(max_num_tokens=37, num_dispatchers=3)
                if input_format == "BatchedExperts"
                else {}
            ),
        ),
    )
    assert [event[0] for event in f.events] == ["prepare", "experts", "kernel"]


@pytest.mark.upstream_source
@pytest.mark.parametrize("per_token", [False, True])
def test_nvfp4_factory_preserves_prepare_failure_identity(
    nvfp4_factory_boundary, per_token
):
    f = nvfp4_factory_boundary
    error = RuntimeError("prepare/finalize provider failed")

    def fail(**kwargs):
        f.events.append(("prepare-failed", kwargs))
        raise error

    f.shared.__globals__["maybe_make_prepare_finalize"] = fail
    with pytest.raises(RuntimeError) as raised:
        f.maker(
            f.quant, f.config, f.cls, f.backend.FLASHINFER_TRTLLM, f.routing, per_token
        )
    assert raised.value is error
    assert [event[0] for event in f.events] == ["prepare-failed"]
    assert f.events[0][1]["routing_tables"] is f.routing


@pytest.fixture(scope="module")
def oracles():
    """Load the real oracle stack, or skip when its real deps are unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    from vllm.platforms import current_platform

    from vllm_sail.registry import moe_backends

    moe_backends.register()
    from vllm_sail.registry.moe_backends import fp8, int8, int_wna16, mxfp4, unquantized

    return SimpleNamespace(
        current_platform=current_platform,
        fp8=fp8,
        int8=int8,
        int_wna16=int_wna16,
        mxfp4=mxfp4,
        unquantized=unquantized,
    )


@pytest.mark.requires_runtime
@pytest.mark.parametrize("backend_name", _NVFP4_BACKENDS)
def test_imported_nvfp4_factory_has_real_shared_provider_and_safe_early_rejection(
    oracles, monkeypatch, backend_name
):
    """真实模块及绑定，PF/计算不执行；在兼容 Torch 环境运行才有 provider 证据。"""
    from vllm.model_executor.layers.fused_moe.oracle import base, nvfp4

    from vllm_sail.patch.utils import PATCH_MARKER

    maker = nvfp4.make_nvfp4_moe_kernel
    assert maker.__globals__ is nvfp4.__dict__
    assert maker.__globals__["make_moe_kernel"] is base.make_moe_kernel
    assert (
        oracles.int_wna16.make_wna16_moe_kernel.__globals__
        is oracles.int_wna16.oracle.__dict__
    )
    assert oracles.int_wna16.oracle.make_moe_kernel is base.make_moe_kernel
    assert getattr(oracles.int_wna16.oracle.make_wna16_moe_kernel, PATCH_MARKER)
    backend = nvfp4.NvFp4MoeBackend[backend_name]
    observed = []

    def forbidden_prepare(**kwargs):
        pytest.fail("不支持的 per-token backend 不得创建 prepare/finalize")

    monkeypatch.setattr(base, "maybe_make_prepare_finalize", forbidden_prepare)
    if backend in nvfp4.PER_TOKEN_NVFP4_MOE_BACKENDS:
        sentinel = object()

        def shared(quant, config, cls, routing, **kwargs):
            observed.append((quant, config, cls, routing, kwargs))
            return sentinel

        monkeypatch.setattr(nvfp4, "make_moe_kernel", shared)
        config, quant, cls, routing = (
            SimpleNamespace(in_dtype=object()),
            SimpleNamespace(use_nvfp4_w4a16=False),
            object(),
            object(),
        )
        assert maker(quant, config, cls, backend, routing, True) is sentinel
        assert observed == [
            (
                quant,
                config,
                cls,
                routing,
                dict(
                    prepare_finalize_kwargs={"input_dtype": config.in_dtype},
                    experts_kwargs={"per_token_activation": True},
                ),
            )
        ]
    else:
        with pytest.raises(ValueError) as raised:
            maker(
                SimpleNamespace(use_nvfp4_w4a16=False),
                object(),
                object(),
                backend,
                object(),
                True,
            )
        assert str(raised.value) == (
            f"Per-token NVFP4 activation scales are not supported by {backend.value}."
        )
        assert observed == []


@pytest.fixture
def nvfp4_cpu_runtime(oracles):
    """真实导入与 CPU tensor；不证明 installed wheel 归属、PPU 图或 native 计算。"""
    import torch
    from vllm.model_executor import utils
    from vllm.model_executor.layers.fused_moe import config
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4

    assert nvfp4.make_nvfp4_moe_quant_config.__globals__ is nvfp4.__dict__
    assert nvfp4.refill_derived is utils.refill_derived
    assert nvfp4.nvfp4_moe_quant_config is config.nvfp4_moe_quant_config
    assert config.nvfp4_moe_quant_config.__globals__ is config.__dict__
    layer = torch.nn.Module()
    for name, values in (
        ("w13_scale", [1.0, 2.0]),
        ("w2_scale", [2.0, 3.0]),
        ("w13_scale_2", [3.0, 6.0]),
        ("w2_scale_2", [4.0, 8.0]),
        ("a13_scale", [2.0, 4.0]),
        ("a2_scale", [5.0, 10.0]),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.tensor(values, dtype=torch.float32, device="cpu"),
                requires_grad=False,
            ),
        )

    def make(backend_name):
        return nvfp4.make_nvfp4_moe_quant_config(
            nvfp4.NvFp4MoeBackend[backend_name],
            layer.w13_scale,
            layer.w2_scale,
            layer.w13_scale_2,
            layer.w2_scale_2,
            layer.a13_scale,
            layer.a2_scale,
            swiglu_limit=7.0,
            swiglu_alpha=1.5,
            swiglu_beta=0.25,
            layer=layer,
        )

    return SimpleNamespace(
        torch=torch, utils=utils, nvfp4=nvfp4, layer=layer, make=make
    )


_NVFP4_CPU_QUANT_BACKENDS = [
    "VLLM_CUTLASS",
    "FLASHINFER_CUTLASS",
    "FLASHINFER_TRTLLM",
]


@pytest.mark.requires_runtime
@pytest.mark.parametrize("backend_name", _NVFP4_CPU_QUANT_BACKENDS)
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_imported_nvfp4_reload_preserves_derived_address_and_alpha_ownership(
    nvfp4_cpu_runtime, backend_name, dtype_name
):
    f = nvfp4_cpu_runtime
    layer, torch = f.layer, f.torch
    layer.to(dtype=getattr(torch, dtype_name))
    parameters = dict(layer.named_parameters())
    addresses = {name: value.data_ptr() for name, value in parameters.items()}
    first = f.make(backend_name)
    held = (first.a1_gscale, first.a2_gscale)
    held_addresses = tuple(value.data_ptr() for value in held)
    assert held[0] is layer._a1_gscale and held[1] is layer._a2_gscale
    for derived, source in zip(held, (layer.a13_scale, layer.a2_scale), strict=False):
        assert derived.device.type == "cpu" and derived.dtype == source.dtype
        assert (
            derived.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(derived, 1.0 / source)
    for name, parameter in parameters.items():
        new = torch.flip(parameter.detach(), dims=[0]).contiguous() * 2.0
        f.utils.replace_parameter(layer, name, new)
        assert getattr(layer, name) is parameter
        assert parameter.data_ptr() == addresses[name]
        torch.testing.assert_close(parameter, new)
    second = f.make(backend_name)
    for previous, current, address, source in zip(
        held,
        (second.a1_gscale, second.a2_gscale),
        held_addresses,
        (layer.a13_scale, layer.a2_scale),
        strict=False,
    ):
        assert current is previous and current.data_ptr() == address
        torch.testing.assert_close(current, 1.0 / source)
    for quant in (first, second):
        assert quant.g1_alphas is layer.w13_scale_2
        assert quant.g2_alphas is layer.w2_scale_2
        assert quant.w1_scale is layer.w13_scale and quant.w2_scale is layer.w2_scale
        assert quant.is_scale_swizzled is (backend_name != "FLASHINFER_TRTLLM")
        assert (quant.gemm1_alpha, quant.gemm1_beta, quant.gemm1_clamp_limit) == (
            1.5,
            0.25,
            7.0,
        )
    assert set(layer._parameters) == set(parameters) and not layer._buffers
    assert "_a1_gscale" not in layer.state_dict()
    assert "_a2_gscale" not in layer.state_dict()


@pytest.mark.requires_runtime
@pytest.mark.parametrize("backend_name", _NVFP4_CPU_QUANT_BACKENDS)
@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_imported_nvfp4_incompatible_derived_reload_rebinds_without_changing_old_config(
    nvfp4_cpu_runtime, backend_name, change
):
    f = nvfp4_cpu_runtime
    layer, torch = f.layer, f.torch
    first = f.make(backend_name)
    old = (first.a1_gscale, first.a2_gscale)
    saved = tuple(value.clone() for value in old)
    for name in ("a13_scale", "a2_scale"):
        source = getattr(layer, name)
        new = torch.full(
            (3,) if change == "shape" else source.shape,
            8.0,
            dtype=torch.float64 if change == "dtype" else source.dtype,
            device="cpu",
        )
        # 本例只改变 checkpoint 输入；derived 属性由真实 factory/refill 管理。
        setattr(layer, name, torch.nn.Parameter(new, requires_grad=False))
    second = f.make(backend_name)
    for previous, snapshot, current, source, attr in zip(
        old,
        saved,
        (second.a1_gscale, second.a2_gscale),
        (layer.a13_scale, layer.a2_scale),
        ("_a1_gscale", "_a2_gscale"),
        strict=False,
    ):
        assert current is getattr(layer, attr) and current is not previous
        assert (
            current.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(current, 1.0 / source)
        torch.testing.assert_close(previous, snapshot)
    assert first.a1_gscale is old[0] and first.a2_gscale is old[1]
    assert first.g1_alphas is second.g1_alphas is layer.w13_scale_2
    assert first.g2_alphas is second.g2_alphas is layer.w2_scale_2
    assert not layer._buffers and not {"_a1_gscale", "_a2_gscale"} & set(
        layer._parameters
    )


@pytest.mark.requires_runtime
def test_imported_refill_derived_none_clears_only_the_plain_attribute(
    nvfp4_cpu_runtime,
):
    f = nvfp4_cpu_runtime
    first = f.make("VLLM_CUTLASS")
    saved = first.a1_gscale.clone()
    parameters = dict(f.layer.named_parameters())
    assert f.utils.refill_derived(f.layer, "_a1_gscale", None) is None
    assert f.layer._a1_gscale is None
    torch = f.torch
    torch.testing.assert_close(first.a1_gscale, saved)
    assert dict(f.layer.named_parameters()).keys() == parameters.keys()
    assert all(getattr(f.layer, name) is value for name, value in parameters.items())
    second = f.make("VLLM_CUTLASS")
    assert (
        second.a1_gscale is f.layer._a1_gscale
        and second.a1_gscale is not first.a1_gscale
    )
    assert second.a2_gscale is first.a2_gscale
    torch.testing.assert_close(second.a1_gscale, 1.0 / f.layer.a13_scale)


@pytest.mark.requires_runtime
@pytest.mark.parametrize("fail_inside", ["inner", "outer"])
def test_imported_requantization_scope_restores_context_and_skips_nvfp4_selection(
    oracles, monkeypatch, fail_inside
):
    from contextvars import Context

    from vllm.model_executor.layers.fused_moe.oracle import nvfp4
    from vllm.model_executor.layers.fused_moe.oracle.base import (
        RequantizationSourceExperts,
    )
    from vllm.model_executor.layers.quantization.online import requantization as req

    def forbidden(*args, **kwargs):
        pytest.fail("requantization source 不得进入普通 backend selection")

    monkeypatch.setattr(nvfp4, "prioritize_humming", forbidden)
    error = RuntimeError("requantization source scope failed")
    assert not req.is_requantization_source()
    with pytest.raises(RuntimeError) as raised:
        with req.skip_checkpoint_backend_selection():
            assert req.is_requantization_source()
            assert not Context().run(req.is_requantization_source)
            assert nvfp4.select_nvfp4_moe_backend(object(), object(), object()) == (
                nvfp4.NvFp4MoeBackend.NONE,
                RequantizationSourceExperts,
            )
            if fail_inside == "inner":
                with pytest.raises(RuntimeError) as nested:
                    with req.skip_checkpoint_backend_selection():
                        assert req.is_requantization_source()
                        raise error
                assert nested.value is error and req.is_requantization_source()
            else:
                with req.skip_checkpoint_backend_selection():
                    assert req.is_requantization_source()
                assert req.is_requantization_source()
            raise error
    assert raised.value is error and not req.is_requantization_source()


@pytest.mark.requires_runtime
@pytest.mark.parametrize("action", ["unchanged", "replaced", "removed"])
def test_imported_requantization_release_keeps_runtime_parameters_and_loader(
    nvfp4_cpu_runtime, action
):
    from vllm.model_executor.layers.quantization.online.requantization import (
        OnlineRequantizationMixin,
    )

    f = nvfp4_cpu_runtime
    layer, torch = f.layer, f.torch
    original = dict(layer.named_parameters())
    layer.register_parameter("empty", None)
    method = OnlineRequantizationMixin()
    assert not method.create_requantization_source_weights(layer)
    assert method.requantization_source_parameters == {}
    calls = []

    def loader(param, weight):
        param.data.copy_(weight)

    def create_weights(target, *args, **kwargs):
        calls.append((target, args, kwargs))
        for name in ("serialized", "obsolete"):
            value = torch.nn.Parameter(torch.ones(2, device="cpu"), requires_grad=False)
            value.weight_loader = loader
            target.register_parameter(name, value)

    source = SimpleNamespace(create_weights=create_weights)
    method.set_requantization_source(source)
    assert method.requantization_source is source
    assert method.create_requantization_source_weights(layer, 17, marker="checkpoint")
    assert calls == [(layer, (17,), {"marker": "checkpoint"})]
    tracked = dict(method.requantization_source_parameters)
    assert set(tracked) == {"serialized", "obsolete"}
    assert all(value.is_meta for value in tracked.values())
    assert all(getattr(layer, name) is value for name, value in tracked.items())
    assert all(value.requantization_source is True for value in tracked.values())
    assert all(
        not getattr(value, "requantization_source", False)
        for value in original.values()
    )
    runtime = None
    if action == "replaced":
        new = torch.tensor([3.0, 6.0], device="cpu")
        f.utils.replace_parameter(layer, "serialized", new)
        runtime = layer.serialized
        assert runtime is not tracked["serialized"]
        assert runtime.weight_loader is tracked["serialized"].weight_loader
        assert runtime.weight_loader.__wrapped__ is loader
        assert not getattr(runtime, "requantization_source", False)
        torch.testing.assert_close(runtime, new)
    elif action == "removed":
        delattr(layer, "serialized")
    method.release_requantization_source_weights(layer)
    assert not method.requantization_source_parameters and not hasattr(
        layer, "obsolete"
    )
    if runtime is not None:
        assert layer.serialized is runtime
    else:
        assert not hasattr(layer, "serialized")
    assert "empty" in layer._parameters and layer.empty is None
    assert all(getattr(layer, name) is value for name, value in original.items())
    method.release_requantization_source_weights(layer)
    assert not method.requantization_source_parameters


def _config(oracles):
    parallel = SimpleNamespace(
        use_deepep_v2_kernels=False,
        dp_size=1,
        ep_size=1,
    )
    return SimpleNamespace(
        moe_parallel_config=parallel,
        moe_backend="auto",
        activation_format=oracles.unquantized.mk.FusedMoEActivationFormat.Standard,
        activation=oracles.unquantized.oracle.MoEActivation.SILU,
        is_lora_enabled=False,
        has_bias=False,
    )


def _set_platform(monkeypatch: pytest.MonkeyPatch, oracles, *, ppu: bool) -> None:
    platform_type = type(oracles.current_platform)
    monkeypatch.setattr(platform_type, "is_ppu", lambda self: ppu)
    monkeypatch.setattr(platform_type, "is_cuda", lambda self: True)
    monkeypatch.setattr(platform_type, "is_rocm", lambda self: False)
    monkeypatch.setattr(platform_type, "is_xpu", lambda self: False)
    monkeypatch.setattr(platform_type, "is_cpu", lambda self: False)
    monkeypatch.setattr(platform_type, "is_device_capability", lambda *_: False)
    monkeypatch.setattr(platform_type, "is_device_capability_family", lambda *_: False)


@pytest.mark.parametrize(
    ("oracle_name", "members"),
    [
        (
            "unquantized",
            {
                "PPU_DEEPGEMM": "PPU_DEEPGEMM",
                "BATCHED_PPU_DEEPGEMM": "BATCHED_PPU_DEEPGEMM",
                "ACEXT": "ACEXT",
            },
        ),
        (
            "fp8",
            {
                "PPU_DEEPGEMM": "PPU_DEEPGEMM",
                "BATCHED_PPU_DEEPGEMM": "BATCHED_PPU_DEEPGEMM",
            },
        ),
        (
            "int8",
            {
                "PPU_DEEPGEMM": "PPU_DEEPGEMM",
                "BATCHED_PPU_DEEPGEMM": "BATCHED_PPU_DEEPGEMM",
                "ACEXT": "ACEXT",
            },
        ),
        (
            "mxfp4",
            {
                "PPU_DEEPGEMM_MXFP4": "PPU_DEEPGEMM_MXFP4",
                "BATCHED_PPU_DEEPGEMM_MXFP4": ("BATCHED_PPU_DEEPGEMM_MXFP4"),
            },
        ),
    ],
)
@pytest.mark.requires_runtime
def test_enum_members_cover_all_lookup_paths(oracles, oracle_name, members) -> None:
    plugin = getattr(oracles, oracle_name)
    enum_cls = getattr(
        plugin,
        {
            "unquantized": "UnquantizedMoeBackend",
            "fp8": "Fp8MoeBackend",
            "int8": "Int8MoeBackend",
            "mxfp4": "Mxfp4MoeBackend",
        }[oracle_name],
    )
    for name, value in members.items():
        member = getattr(enum_cls, name)
        assert member.name == name
        assert member.value == value
        assert enum_cls[name] is member
        assert enum_cls(value) is member
        assert member in list(enum_cls)


@pytest.mark.requires_runtime
def test_extend_enum_stays_idempotent_after_helper_reimport(oracles) -> None:
    member = oracles.fp8.Fp8MoeBackend.PPU_DEEPGEMM
    extend_module = importlib.reload(
        importlib.import_module("vllm_sail.registry.moe_backends._extend")
    )
    assert (
        extend_module.extend_enum(oracles.fp8.Fp8MoeBackend, member.name, member.value)
        is member
    )


@pytest.mark.parametrize("oracle_name", ["unquantized", "fp8", "int8", "mxfp4"])
@pytest.mark.requires_runtime
def test_backend_to_kernel_cls_delegates_upstream(
    monkeypatch: pytest.MonkeyPatch, oracles, oracle_name
) -> None:
    plugin = getattr(oracles, oracle_name)

    class UpstreamExperts:
        pass

    monkeypatch.setattr(
        plugin, "_upstream_backend_to_kernel_cls", lambda backend: [UpstreamExperts]
    )
    assert plugin.backend_to_kernel_cls(
        plugin.oracle.__dict__[
            {
                "unquantized": "UnquantizedMoeBackend",
                "fp8": "Fp8MoeBackend",
                "int8": "Int8MoeBackend",
                "mxfp4": "Mxfp4MoeBackend",
            }[oracle_name]
        ].TRITON
    ) == [UpstreamExperts]


@pytest.mark.parametrize(
    ("oracle_name", "member_name", "module_name", "class_name"),
    [
        ("unquantized", "PPU_DEEPGEMM", "deep_gemm_moe", "PPUDeepGemmExperts"),
        (
            "unquantized",
            "BATCHED_PPU_DEEPGEMM",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExperts",
        ),
        ("fp8", "PPU_DEEPGEMM", "deep_gemm_moe", "PPUDeepGemmExperts"),
        (
            "fp8",
            "BATCHED_PPU_DEEPGEMM",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExperts",
        ),
        ("int8", "PPU_DEEPGEMM", "deep_gemm_moe", "PPUDeepGemmExperts"),
        (
            "int8",
            "BATCHED_PPU_DEEPGEMM",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExperts",
        ),
        (
            "mxfp4",
            "PPU_DEEPGEMM_MXFP4",
            "deep_gemm_moe",
            "PPUDeepGemmExpertsMXFP4",
        ),
        (
            "mxfp4",
            "BATCHED_PPU_DEEPGEMM_MXFP4",
            "batched_deep_gemm_moe",
            "PPUBatchedDeepGemmExpertsMXFP4",
        ),
    ],
)
@pytest.mark.requires_runtime
def test_backend_to_kernel_cls_returns_ppu_expert(
    oracles, oracle_name, member_name, module_name, class_name
) -> None:
    pytest.importorskip("triton", reason="PPU DeepGEMM expert modules need Triton")
    plugin = getattr(oracles, oracle_name)
    enum_cls = getattr(
        plugin,
        {
            "unquantized": "UnquantizedMoeBackend",
            "fp8": "Fp8MoeBackend",
            "int8": "Int8MoeBackend",
            "mxfp4": "Mxfp4MoeBackend",
        }[oracle_name],
    )
    experts = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts." + module_name
    )
    assert plugin.backend_to_kernel_cls(getattr(enum_cls, member_name)) == [
        getattr(experts, class_name)
    ]


@pytest.mark.parametrize("oracle_name", ["unquantized", "int8"])
@pytest.mark.requires_runtime
def test_backend_to_kernel_cls_returns_acext_expert(oracles, oracle_name) -> None:
    from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

    plugin = getattr(oracles, oracle_name)
    assert plugin.backend_to_kernel_cls(
        plugin.__dict__[
            "UnquantizedMoeBackend"
            if oracle_name == "unquantized"
            else "Int8MoeBackend"
        ].ACEXT
    ) == [AcextExperts]


@pytest.mark.requires_runtime
def test_map_functions_add_ppu_names_and_delegate_upstream_names(oracles) -> None:
    unquantized, fp8, int8, mxfp4 = (
        oracles.unquantized,
        oracles.fp8,
        oracles.int8,
        oracles.mxfp4,
    )
    assert unquantized.map_unquantized_backend("ppu_deep_gemm") is (
        unquantized.UnquantizedMoeBackend.PPU_DEEPGEMM
    )
    assert unquantized.map_unquantized_backend("ppu_acext") is (
        unquantized.UnquantizedMoeBackend.ACEXT
    )
    assert fp8.map_fp8_backend("ppu_deep_gemm") is fp8.Fp8MoeBackend.PPU_DEEPGEMM
    assert int8.map_int8_backend("ppu_acext") is int8.Int8MoeBackend.ACEXT
    assert mxfp4.map_mxfp4_backend("ppu_deep_gemm") == [
        mxfp4.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4,
        mxfp4.Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4,
    ]
    assert unquantized.map_unquantized_backend("triton") is (
        unquantized._upstream_map_unquantized_backend("triton")
    )
    assert fp8.map_fp8_backend("triton") is fp8._upstream_map_fp8_backend("triton")
    assert int8.map_int8_backend("triton") is int8._upstream_map_int8_backend("triton")
    assert mxfp4.map_mxfp4_backend("triton") == (
        mxfp4._upstream_map_mxfp4_backend("triton")
    )


@pytest.mark.requires_runtime
def test_priority_lists_include_ppu_members(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    _set_platform(monkeypatch, oracles, ppu=True)
    monkeypatch.delenv("VLLM_PPU_MOE_BACKEND", raising=False)
    deep_gemm = importlib.import_module("vllm_sail.utils.deep_gemm")
    acext = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts.acext"
    )
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(acext, "is_acext_supported", lambda: True)
    config = _config(oracles)

    assert oracles.unquantized.UnquantizedMoeBackend.PPU_DEEPGEMM in (
        oracles.unquantized._get_priority_backends(config)
    )
    assert oracles.fp8.Fp8MoeBackend.PPU_DEEPGEMM in (
        oracles.fp8._get_priority_backends(config, None, None)
    )
    assert oracles.int8.Int8MoeBackend.ACEXT in (
        oracles.int8._get_priority_backends(config)
    )
    assert oracles.mxfp4.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 in (
        oracles.mxfp4._get_priority_backends()
    )
    assert oracles.mxfp4.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4 in (
        oracles.mxfp4._get_priority_backends_for_gpt_oss()
    )


@pytest.mark.requires_runtime
def test_priority_lists_are_unchanged_off_ppu(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    _set_platform(monkeypatch, oracles, ppu=False)
    config = _config(oracles)

    assert oracles.unquantized._get_priority_backends(config) == (
        oracles.unquantized._upstream_get_priority_backends(config)
    )
    assert oracles.fp8._get_priority_backends(config, None, None) == (
        oracles.fp8._upstream_get_priority_backends(config, None, None)
    )
    assert oracles.int8._get_priority_backends(config) == (
        oracles.int8._upstream_get_priority_backends(config)
    )
    assert oracles.mxfp4._get_priority_backends() == (
        oracles.mxfp4._upstream_get_priority_backends()
    )
    assert oracles.mxfp4._get_priority_backends_for_gpt_oss() == (
        oracles.mxfp4._upstream_get_priority_backends_for_gpt_oss()
    )


@pytest.mark.requires_runtime
def test_every_oracle_patch_records_original_and_rejects_reapplication(
    oracles,
) -> None:
    from vllm_sail.patch.utils import PATCH_REGISTRY, original_of, patch

    records = [record for record in PATCH_REGISTRY if ".oracle." in record.target]
    # 18 from the MoE-backend registry; the enhancement chain's mxfp4 and
    # fused_moe_ppu modules add five more when they ran in this process.
    assert len(records) >= 18
    for record in records:
        module_name, attribute = record.target.rsplit(".", 1)
        replacement = getattr(importlib.import_module(module_name), attribute)
        assert original_of(replacement, record.target) is not replacement
        with pytest.raises(RuntimeError, match="already patched"):
            patch(
                module_name,
                attribute,
                reason=record.reason,
                affected_versions=record.affected_versions,
                remove_when=record.remove_when,
            )(replacement)


# ---------------------------------------------------------------------------
# Selection tests with mocked is_ppu() -- the second half of the Phase-3 exit
# criteria. Expert classes are faked so the tests do not need the PPU SDK or
# Triton; the selection *logic* is what is under test here.
# ---------------------------------------------------------------------------


class _FakeExperts:
    @staticmethod
    def is_supported_config(cls, moe_config, weight_key, activation_key, fmt):
        return True, None


def _select_config(oracles, *, backend: str, batched: bool):
    parallel = SimpleNamespace(
        use_deepep_v2_kernels=False,
        dp_size=1,
        ep_size=1,
    )
    return SimpleNamespace(
        moe_parallel_config=parallel,
        moe_backend=backend,
        activation_format=(
            oracles.unquantized.mk.FusedMoEActivationFormat.BatchedExperts
            if batched
            else oracles.unquantized.mk.FusedMoEActivationFormat.Standard
        ),
        activation=oracles.unquantized.oracle.MoEActivation.SILU,
        is_lora_enabled=False,
        has_bias=False,
    )


@pytest.mark.requires_runtime
def test_select_unquantized_batched_ppu_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.unquantized
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    backend, kernel_cls = plugin.select_unquantized_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=True)
    )
    assert backend is plugin.UnquantizedMoeBackend.BATCHED_PPU_DEEPGEMM
    assert kernel_cls is _FakeExperts


@pytest.mark.requires_runtime
def test_select_unquantized_batched_ppu_deep_gemm_raises_when_unsupported(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.unquantized

    class _Unsupported(_FakeExperts):
        @staticmethod
        def is_supported_config(cls, moe_config, weight_key, activation_key, fmt):
            return False, "testing"

    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_Unsupported])
    with pytest.raises(ValueError, match="unsupported"):
        plugin.select_unquantized_moe_backend(
            _select_config(oracles, backend="ppu_deep_gemm", batched=True)
        )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.requires_runtime
def test_select_unquantized_delegates_for_non_ppu_backends(
    monkeypatch: pytest.MonkeyPatch, oracles, batched
) -> None:
    plugin = oracles.unquantized
    sentinel = object()
    monkeypatch.setattr(
        plugin, "_upstream_select_unquantized_moe_backend", lambda config: sentinel
    )
    assert (
        plugin.select_unquantized_moe_backend(
            _select_config(oracles, backend="auto", batched=batched)
        )
        is sentinel
    )


@pytest.mark.requires_runtime
@pytest.mark.parametrize(
    "format_name", ["Standard", "BatchedExperts", "PaddedStandard"]
)
@pytest.mark.parametrize("backend_name", ["auto", "ppu_deep_gemm", "ppu_acext"])
def test_installed_unquantized_lora_preserves_real_selector_layout_guard(
    monkeypatch, oracles, format_name, backend_name
):
    """真实安装的 provider/adapter 交界；专家替身不证明 LoRA 设备计算。"""
    plugin = oracles.unquantized
    _set_platform(monkeypatch, oracles, ppu=True)
    platform_type = type(oracles.current_platform)
    monkeypatch.setattr(platform_type, "is_tpu", lambda self: False)
    monkeypatch.setattr(platform_type, "is_out_of_tree", lambda self: False)
    config = _select_config(oracles, backend=backend_name, batched=False)
    config.activation_format = plugin.mk.FusedMoEActivationFormat[format_name]
    config.is_lora_enabled = True
    config.moe_parallel_config.all2all_backend = "installed-boundary"
    events = []

    def lora_supported(config):
        events.append(("lora-support", config))
        return False

    def mapping(item):
        events.append(("mapping", item))
        assert item is plugin.UnquantizedMoeBackend.TRITON
        return [_FakeExperts]

    monkeypatch.setattr(plugin.oracle, "_trtllm_bf16_lora_supported", lora_supported)
    monkeypatch.setattr(plugin.oracle, "backend_to_kernel_cls", mapping)
    if format_name != "Standard":
        with pytest.raises(
            ValueError, match="MoE LoRA requires a Standard activation layout"
        ):
            plugin.select_unquantized_moe_backend(config)
        assert events == []
    else:
        assert plugin.select_unquantized_moe_backend(config) == (
            plugin.UnquantizedMoeBackend.TRITON,
            _FakeExperts,
        )
        assert events == [
            ("lora-support", config),
            ("mapping", plugin.UnquantizedMoeBackend.TRITON),
        ]


@pytest.mark.parametrize(
    ("batched", "expected_member"),
    [
        (False, "PPU_DEEPGEMM"),
        (True, "BATCHED_PPU_DEEPGEMM"),
    ],
)
@pytest.mark.requires_runtime
def test_select_fp8_explicit_ppu_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, oracles, batched, expected_member
) -> None:
    plugin = oracles.fp8
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    backend, kernel_cls = plugin.select_fp8_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=batched), None, None
    )
    assert backend is getattr(plugin.Fp8MoeBackend, expected_member)
    assert kernel_cls is _FakeExperts


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
@pytest.mark.requires_runtime
def test_select_fp8_env_deepgemm_selects_ppu_only_on_ppu(
    monkeypatch: pytest.MonkeyPatch, oracles, prefix: str
) -> None:
    plugin = oracles.fp8
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    monkeypatch.delenv("VLLM_SAIL_MOE_BACKEND", raising=False)
    monkeypatch.setenv(f"{prefix}_MOE_BACKEND", "deepgemm")

    _set_platform(monkeypatch, oracles, ppu=True)
    backend, _ = plugin.select_fp8_moe_backend(
        _select_config(oracles, backend="auto", batched=False), None, None
    )
    assert backend is plugin.Fp8MoeBackend.PPU_DEEPGEMM

    _set_platform(monkeypatch, oracles, ppu=False)
    sentinel = object()
    monkeypatch.setattr(
        plugin,
        "_upstream_select_fp8_moe_backend",
        lambda *args, **kwargs: sentinel,
    )
    assert (
        plugin.select_fp8_moe_backend(
            _select_config(oracles, backend="auto", batched=False), None, None
        )
        is sentinel
    )


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
@pytest.mark.requires_runtime
def test_select_fp8_env_acext_delegates_upstream(
    monkeypatch: pytest.MonkeyPatch, oracles, prefix: str
) -> None:
    plugin = oracles.fp8
    monkeypatch.delenv("VLLM_SAIL_MOE_BACKEND", raising=False)
    monkeypatch.setenv(f"{prefix}_MOE_BACKEND", "acext")
    _set_platform(monkeypatch, oracles, ppu=True)
    sentinel = object()
    monkeypatch.setattr(
        plugin,
        "_upstream_select_fp8_moe_backend",
        lambda *args, **kwargs: sentinel,
    )
    assert (
        plugin.select_fp8_moe_backend(
            _select_config(oracles, backend="auto", batched=False), None, None
        )
        is sentinel
    )


@pytest.mark.requires_runtime
def test_select_int8_batched_ppu_deep_gemm(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.int8
    monkeypatch.setattr(plugin, "backend_to_kernel_cls", lambda backend: [_FakeExperts])
    backend, kernel_cls = plugin.select_int8_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=True)
    )
    assert backend is plugin.Int8MoeBackend.BATCHED_PPU_DEEPGEMM
    assert kernel_cls is _FakeExperts


@pytest.mark.parametrize(
    "backend_name",
    ["PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM", "ACEXT"],
)
@pytest.mark.requires_runtime
def test_convert_int8_ppu_weights_preserves_canonical_layout(
    oracles, backend_name
) -> None:
    """PPU INT8 experts consume the checkpoint's canonical weight layout."""
    import torch

    plugin = oracles.int8
    w13 = torch.empty((2, 4, 8), dtype=torch.int8)
    w2 = torch.empty((2, 8, 2), dtype=torch.int8)

    converted = plugin.oracle.convert_to_int8_moe_kernel_format(
        getattr(plugin.Int8MoeBackend, backend_name),
        w13,
        w2,
    )

    assert converted[0] is w13
    assert converted[1] is w2


@pytest.mark.requires_runtime
def test_convert_int8_upstream_backends_delegate_unchanged(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.int8
    w13 = object()
    w2 = object()
    layer = object()
    w13_scale = object()
    sentinel = (object(), object())
    captured = {}

    def _upstream(backend, arg_w13, arg_w2, *, layer, w13_scale):
        captured.update(
            backend=backend,
            w13=arg_w13,
            w2=arg_w2,
            layer=layer,
            w13_scale=w13_scale,
        )
        return sentinel

    monkeypatch.setattr(
        plugin,
        "_upstream_convert_to_int8_moe_kernel_format",
        _upstream,
    )

    assert (
        plugin.convert_to_int8_moe_kernel_format(
            plugin.Int8MoeBackend.TRITON,
            w13,
            w2,
            layer=layer,
            w13_scale=w13_scale,
        )
        is sentinel
    )
    assert captured == {
        "backend": plugin.Int8MoeBackend.TRITON,
        "w13": w13,
        "w2": w2,
        "layer": layer,
        "w13_scale": w13_scale,
    }


@pytest.mark.parametrize(
    ("backend", "match"),
    [
        ("triton", "Triton MoE backend is disabled"),
        ("ppu_acext", "ACEXT MoE backend is disabled"),
    ],
)
@pytest.mark.requires_runtime
def test_select_int8_batched_rejects_standard_only_backends(
    monkeypatch: pytest.MonkeyPatch, oracles, backend, match
) -> None:
    plugin = oracles.int8
    with pytest.raises(ValueError, match=match):
        plugin.select_int8_moe_backend(
            _select_config(oracles, backend=backend, batched=True)
        )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.requires_runtime
def test_select_int8_auto_delegates_upstream(
    monkeypatch: pytest.MonkeyPatch, oracles, batched
) -> None:
    plugin = oracles.int8
    sentinel = object()
    monkeypatch.setattr(
        plugin,
        "_upstream_select_int8_moe_backend",
        lambda *args, **kwargs: sentinel,
    )
    assert (
        plugin.select_int8_moe_backend(
            _select_config(oracles, backend="auto", batched=batched)
        )
        is sentinel
    )


@pytest.mark.requires_runtime
def test_select_mxfp4_batched_ppu_deep_gemm_uses_batched_member(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.mxfp4
    captured = {}

    def _capture(self, backend, *, kernel_classes, activation_key):
        config = self.moe_config
        weight_key = self.weight_key
        activation_format = config.activation_format
        assert kernel_classes == [_FakeExperts]
        # map_mxfp4_backend() lists the standard PPU member followed by its
        # batched sibling and lets the shared is_supported_config() format check
        # drop the one that does not match the MoE layout; this stub mirrors it.
        own_format = (
            plugin.mk.FusedMoEActivationFormat.BatchedExperts
            if backend.name.startswith("BATCHED_")
            else plugin.mk.FusedMoEActivationFormat.Standard
        )
        if activation_format != own_format:
            raise ValueError(f"{backend.value} rejected: activation format")
        captured.update(
            backend=backend,
            weight_key=weight_key,
            activation_key=activation_key,
            activation_format=activation_format,
        )
        return backend, _FakeExperts

    monkeypatch.setattr(plugin.oracle.Mxfp4MoEKernelOracle, "return_or_raise", _capture)
    monkeypatch.setattr(
        plugin.oracle, "_requested_kernel_cls", lambda backend, config: [_FakeExperts]
    )
    monkeypatch.setattr(plugin.oracle, "_resolve_activation_key", lambda key: key)
    backend, kernel_cls = plugin.select_mxfp4_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=True)
    )
    assert backend is plugin.Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4
    assert kernel_cls is _FakeExperts
    assert captured["backend"] is plugin.Mxfp4MoeBackend.BATCHED_PPU_DEEPGEMM_MXFP4
    assert captured["activation_key"] is plugin.oracle.kMxfp4Dynamic
    assert (
        captured["activation_format"]
        is plugin.mk.FusedMoEActivationFormat.BatchedExperts
    )


@pytest.mark.requires_runtime
def test_select_mxfp4_non_batched_ppu_deep_gemm_uses_standard_member(
    monkeypatch: pytest.MonkeyPatch, oracles
) -> None:
    plugin = oracles.mxfp4
    captured = {}

    def _capture(self, backend, *, kernel_classes, activation_key):
        config = self.moe_config
        weight_key = self.weight_key
        activation_format = config.activation_format
        assert kernel_classes == [_FakeExperts]
        # map_mxfp4_backend() lists the standard PPU member followed by its
        # batched sibling and lets the shared is_supported_config() format check
        # drop the one that does not match the MoE layout; this stub mirrors it.
        own_format = (
            plugin.mk.FusedMoEActivationFormat.BatchedExperts
            if backend.name.startswith("BATCHED_")
            else plugin.mk.FusedMoEActivationFormat.Standard
        )
        if activation_format != own_format:
            raise ValueError(f"{backend.value} rejected: activation format")
        captured.update(
            backend=backend,
            weight_key=weight_key,
            activation_key=activation_key,
            activation_format=activation_format,
        )
        return backend, _FakeExperts

    monkeypatch.setattr(plugin.oracle.Mxfp4MoEKernelOracle, "return_or_raise", _capture)
    monkeypatch.setattr(
        plugin.oracle, "_requested_kernel_cls", lambda backend, config: [_FakeExperts]
    )
    monkeypatch.setattr(plugin.oracle, "_resolve_activation_key", lambda key: key)
    backend, kernel_cls = plugin.select_mxfp4_moe_backend(
        _select_config(oracles, backend="ppu_deep_gemm", batched=False)
    )
    assert backend is plugin.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4
    assert kernel_cls is _FakeExperts
    assert captured["backend"] is plugin.Mxfp4MoeBackend.PPU_DEEPGEMM_MXFP4
    assert captured["activation_key"] is plugin.oracle.kMxfp4Dynamic
    assert captured["activation_format"] is plugin.mk.FusedMoEActivationFormat.Standard


@pytest.mark.requires_runtime
def test_select_mxfp4_moe_backend_is_upstreams(oracles) -> None:
    """The plugin re-exports upstream's selector instead of copying its body."""
    plugin = oracles.mxfp4
    assert plugin.select_mxfp4_moe_backend is (
        plugin._upstream_select_mxfp4_moe_backend
    )


# ---------------------------------------------------------------------------
# is_supported_config coverage for every registered PPU expert kernel. The
# static predicate matrix below the upstream staticmethod is what the oracles
# consult, so each PPU-overridden gate gets an accept and a reject case.
# ---------------------------------------------------------------------------


@pytest.fixture()
def moe_config_factory():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="naive",
        enable_eplb=False,
    )

    def _make(activation=MoEActivation.SILU):
        import torch

        return FusedMoEConfig(
            num_experts=8,
            experts_per_token=2,
            hidden_dim=128,
            intermediate_size=256,
            num_local_experts=8,
            num_logical_experts=8,
            activation=activation,
            device="cpu",
            routing_method=RoutingMethodType.TopK,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
        )

    return _make


@pytest.mark.requires_runtime
def test_acext_experts_is_supported_config_matrix(
    monkeypatch: pytest.MonkeyPatch, oracles, moe_config_factory
) -> None:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

    _set_platform(monkeypatch, oracles, ppu=True)
    standard = mk.FusedMoEActivationFormat.Standard
    config = moe_config_factory()
    check = AcextExperts.is_supported_config

    assert check(AcextExperts, config, None, None, standard) == (True, None)
    assert check(
        AcextExperts,
        config,
        kInt8StaticChannelSym,
        kInt8DynamicTokenSym,
        standard,
    ) == (True, None)

    supported, reason = check(AcextExperts, config, kFp8StaticTensorSym, None, standard)
    assert supported is False and "quantization scheme" in reason
    supported, reason = check(
        AcextExperts, config, None, None, mk.FusedMoEActivationFormat.BatchedExperts
    )
    assert supported is False and "activation format" in reason
    supported, reason = check(
        AcextExperts, moe_config_factory(MoEActivation.GELU), None, None, standard
    )
    assert supported is False and "activation" in reason

    _set_platform(monkeypatch, oracles, ppu=False)
    supported, reason = check(AcextExperts, config, None, None, standard)
    assert supported is False and "current device" in reason


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("deep_gemm_moe", "PPUDeepGemmExperts"),
        ("deep_gemm_moe", "PPUDeepGemmExpertsMXFP4"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExperts"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExpertsMXFP4"),
    ],
)
@pytest.mark.requires_runtime
def test_deep_gemm_experts_support_gates(
    monkeypatch: pytest.MonkeyPatch, oracles, module_name, class_name
) -> None:
    pytest.importorskip("triton", reason="PPU DeepGEMM expert modules need Triton")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8DynamicTokenSym,
        kFp8Static128BlockSym,
        kFp8StaticChannelSym,
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    import vllm_sail.utils.deep_gemm as deep_gemm

    experts = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts." + module_name
    )
    kernel_cls = getattr(experts, class_name)

    # Device gate: needs the PPU DeepGEMM wrapper; MXFP4 variants additionally
    # refuse sm80. The experts modules bind the gate at import time, so patch
    # both the utils module and the bound reference.
    def _set_gate(value: bool) -> None:
        monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: value)
        monkeypatch.setattr(
            experts, "is_deep_gemm_supported", lambda: value, raising=False
        )

    _set_gate(False)
    assert kernel_cls._supports_current_device() is False
    _set_gate(True)
    _set_platform(monkeypatch, oracles, ppu=True)
    if class_name.endswith("MXFP4"):
        monkeypatch.setattr(
            oracles.current_platform,
            "is_device_capability",
            lambda cap: cap == (8, 0),
        )
        assert kernel_cls._supports_current_device() is False
        monkeypatch.setattr(
            oracles.current_platform, "is_device_capability", lambda cap: False
        )
    assert kernel_cls._supports_current_device() is True

    # Quant-scheme gate: every PPU DeepGEMM expert accepts bf16 and the fp8/int8
    # keys its oracle advertises, and rejects a mismatched pair.
    schemes = [
        (None, None),
        (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        (kFp8StaticChannelSym, kFp8DynamicTokenSym),
        (kInt8StaticChannelSym, kInt8DynamicTokenSym),
    ]
    if class_name.endswith("MXFP4"):
        schemes = [
            (
                oracles.mxfp4.oracle.kMxfp4Static,
                oracles.mxfp4.oracle.kMxfp4Dynamic,
            )
        ]
    for weight_key, activation_key in schemes:
        assert kernel_cls._supports_quant_scheme(weight_key, activation_key) is True
    assert kernel_cls._supports_quant_scheme(kFp8Static128BlockSym, None) is False

    # Activation gate: SILU always; the batched/class variants differ on the
    # SwiGLU variants, so only assert the common floor and one rejection.
    assert kernel_cls._supports_activation(MoEActivation.SILU) is True
    assert kernel_cls._supports_activation(MoEActivation.GELU) is False

    # No act-and_mul support anywhere in the PPU expert family.
    assert kernel_cls._supports_no_act_and_mul() is False


@pytest.mark.parametrize(
    ("module_name", "class_name", "expected_format"),
    [
        ("deep_gemm_moe", "PPUDeepGemmExperts", "Standard"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExperts", "BatchedExperts"),
        ("batched_deep_gemm_moe", "PPUBatchedDeepGemmExpertsMXFP4", "BatchedExperts"),
    ],
)
@pytest.mark.requires_runtime
def test_deep_gemm_experts_is_supported_config_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    oracles,
    moe_config_factory,
    module_name,
    class_name,
    expected_format,
) -> None:
    pytest.importorskip("triton", reason="PPU DeepGEMM expert modules need Triton")
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    import vllm_sail.utils.deep_gemm as deep_gemm

    experts = importlib.import_module(
        "vllm_sail.model_executor.layers.fused_moe.experts." + module_name
    )
    kernel_cls = getattr(experts, class_name)

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(experts, "is_deep_gemm_supported", lambda: True, raising=False)
    _set_platform(monkeypatch, oracles, ppu=True)

    fmt = getattr(mk.FusedMoEActivationFormat, expected_format)
    if class_name.endswith("MXFP4"):
        weight_key = oracles.mxfp4.oracle.kMxfp4Static
        activation_key = oracles.mxfp4.oracle.kMxfp4Dynamic
    else:
        weight_key = kInt8StaticChannelSym
        activation_key = kInt8DynamicTokenSym
    supported, reason = kernel_cls.is_supported_config(
        kernel_cls,
        moe_config_factory(),
        weight_key,
        activation_key,
        fmt,
    )
    assert supported is True, reason

    # Wrong activation format for the kernel must be rejected with a reason.
    wrong = mk.FusedMoEActivationFormat.BatchedExperts
    if expected_format == "BatchedExperts":
        wrong = mk.FusedMoEActivationFormat.Standard
    supported, reason = kernel_cls.is_supported_config(
        kernel_cls,
        moe_config_factory(),
        kInt8StaticChannelSym,
        kInt8DynamicTokenSym,
        wrong,
    )
    assert supported is False and reason
