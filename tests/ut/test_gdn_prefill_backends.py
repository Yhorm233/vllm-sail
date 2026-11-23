# SPDX-License-Identifier: Apache-2.0
"""Exercise copied GDN backend routing without importing device libraries."""

import ast
import inspect
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.support.source import ROOT, definition, function

PATCH = "vllm_sail/patch/enhancement/attention/gdn.py"


@pytest.mark.parametrize(
    "is_cuda,is_ppu,capability,active",
    [
        (True, True, 89, "pla"),
        (True, False, 90, "flashinfer"),
        (False, False, 0, "triton"),
    ],
    ids=["test-accelerator", "cuda-platform", "cpu-platform"],
)
def test_e2e_platform_double_preserves_resolver_routes(
    monkeypatch, is_cuda, is_ppu, capability, active
):
    module = NS(current_platform=None)
    stub = function(
        "tests/e2e/gdn/test_gdn_pla_prefill.py",
        "_stub_platform",
        {"gdn_mod": module, "SimpleNamespace": NS},
    )
    stub(monkeypatch, is_cuda=is_cuda, is_ppu=is_ppu, capability=capability)
    namespace = {
        "current_platform": module.current_platform,
        "_pla_prefill_supported": lambda _: True,
    }
    config = NS(
        additional_config={},
        model_config=NS(
            hf_text_config=NS(linear_key_head_dim=128, linear_value_head_dim=128)
        ),
    )
    assert function(PATCH, "_resolve_gdn_prefill_backend", namespace)(config) == (
        "auto",
        active,
    )
    assert module.current_platform.is_rocm() is False


@pytest.fixture
def resolve(monkeypatch):
    def run(
        device,
        backend="auto",
        *,
        head_k=128,
        head_v=128,
        dtype="bfloat16",
        available=True,
        xpu_available=True,
        pla=True,
        capability=89,
        cuda_runtime=13,
        mi3xx=False,
        gfx1151=False,
        hip_available=False,
        hip_helper=None,
        unsupported_reasons=(),
        helper=None,
        body=PATCH,
    ):
        warnings = []
        platform = NS(
            device_name=device,
            is_ppu=lambda: device == "ppu",
            is_cuda=lambda: device in ("ppu", "cuda"),
            is_rocm=lambda: device == "rocm",
            is_xpu=lambda: device == "xpu",
            is_device_capability=lambda value: capability == value,
            is_device_capability_family=lambda value: capability // 10 == value // 10,
            get_cuda_runtime_major=lambda: cuda_runtime,
        )

        def aiter_available():
            assert device == "rocm"
            return available

        config = NS(
            additional_config={"gdn_prefill_backend": backend},
            model_config=NS(
                hf_text_config=NS(
                    linear_key_head_dim=head_k, linear_value_head_dim=head_v
                ),
                dtype=dtype,
            ),
        )

        def hip_kernel_available():
            assert device == "rocm"
            return hip_available

        namespace = {
            "current_platform": platform,
            "rocm_aiter_ops": NS(is_gdn_flydsl_prefill_available=aiter_available),
            "torch": NS(bfloat16="bfloat16", float16="float16"),
            "logger": NS(warning_once=lambda *args: warnings.append(args)),
            "_pla_prefill_supported": lambda _: pla,
            "xpu_gdn_kernel_available": lambda: xpu_available,
            "is_hip_gdn_available": hip_kernel_available,
        }

        def reasons(candidate, key_dim, value_dim):
            assert candidate is config
            assert (key_dim, value_dim) == (head_k, head_v)
            if helper is not None:
                return helper(candidate, key_dim, value_dim)
            return list(unsupported_reasons)

        namespace["_aiter_flydsl_unsupported_reasons"] = reasons
        namespace["_hip_gdn_unsupported_reasons"] = hip_helper or (
            lambda *_: list(unsupported_reasons)
        )
        rocm = ModuleType("vllm.platforms.rocm")
        rocm.on_mi3xx = lambda: mi3xx
        rocm.on_gfx1151 = lambda: gfx1151
        monkeypatch.setitem(sys.modules, rocm.__name__, rocm)
        result = function(body, "_resolve_gdn_prefill_backend", namespace)(config)
        return result, warnings

    return run


def test_rocm_explicit_aiter_prefill_uses_supported_kernel(resolve):
    result, warnings = resolve("rocm", "  AITER_FLYDSL  ")
    assert result == ("aiter_flydsl", "aiter_flydsl")
    assert not warnings


def test_rocm_explicit_aiter_prefill_requires_importable_kernel(resolve):
    with pytest.raises(RuntimeError, match="not importable"):
        resolve("rocm", "aiter_flydsl", available=False)


@pytest.mark.parametrize(
    "shape", [{"head_k": 64}, {"head_v": 64}, {"dtype": "float16"}]
)
def test_rocm_unsupported_aiter_shapes_warn_and_fall_back(resolve, shape):
    result, warnings = resolve(
        "rocm",
        "aiter_flydsl",
        unsupported_reasons=["unsupported configuration"],
        **shape,
    )
    assert result == ("aiter_flydsl", "triton")
    assert len(warnings) == 1
    assert "Falling back to Triton/FLA" in warnings[0][0]


@pytest.mark.parametrize(
    "device", ["ppu", "cuda", "cpu"], ids=["fake-ppu", "fake-cuda", "fake-cpu"]
)
def test_explicit_aiter_prefill_rejects_non_rocm_devices(resolve, device):
    with pytest.raises(RuntimeError, match="ROCm backend"):
        resolve(device, "aiter_flydsl")


@pytest.mark.parametrize(
    "device,backend,options,active",
    [
        ("rocm", "auto", {}, "triton"),
        ("ppu", "auto", {}, "pla"),
        ("ppu", "auto", {"pla": False}, "triton"),
        ("ppu", "pla", {}, "pla"),
        ("ppu", "flashinfer", {"capability": 90}, "triton"),
        ("cuda", "auto", {"capability": 90}, "flashinfer"),
        ("cuda", "cutedsl", {"capability": 100}, "cutedsl"),
        ("cuda", "auto", {}, "triton"),
    ],
)
def test_existing_platform_routes_remain_available(
    resolve, device, backend, options, active
):
    result, warnings = resolve(device, backend, **options)
    assert result == (backend, active)
    assert not warnings


@pytest.mark.parametrize(
    "mi3xx,available,reasons,active",
    [
        (True, True, [], "aiter_flydsl"),
        (True, False, [], "triton"),
        (True, True, ["unsupported dtype"], "triton"),
        (True, True, ["unsupported head shape"], "triton"),
        (False, True, [], "triton"),
    ],
)
def test_rocm_auto_flydsl_requires_supported_hardware_model_and_kernel(
    resolve, mi3xx, available, reasons, active
):
    result, warnings = resolve(
        "rocm", mi3xx=mi3xx, available=available, unsupported_reasons=reasons
    )
    assert result == ("auto", active)
    assert not warnings


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize(
    "options,active",
    [
        ({}, "aiter_flydsl"),
        ({"head_k": 64}, "triton"),
        ({"head_v": 64}, "triton"),
        ({"dtype": "float16"}, "triton"),
        ({"head_k": 64, "dtype": "float16"}, "triton"),
        ({"available": False}, "triton"),
        ({"mi3xx": False}, "triton"),
    ],
)
def test_real_flydsl_reasons_drive_upstream_and_copied_auto_routing(
    upstream_source_root, resolve, copied, options, active
):
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    helper = function(
        path, "_aiter_flydsl_unsupported_reasons", {"torch": NS(bfloat16="bfloat16")}
    )
    result, warnings = resolve(
        "rocm",
        helper=helper,
        body=PATCH if copied else path,
        **({"mi3xx": True} | options),
    )
    assert result == ("auto", active)
    assert not warnings


@pytest.mark.upstream_source
@pytest.mark.parametrize("device", ["cuda", "ppu"])
@pytest.mark.parametrize("capability", [80, 89, 90, 100, 110, 120])
@pytest.mark.parametrize("head_k", [64, 128])
@pytest.mark.parametrize("cuda_runtime", [12, 13])
@pytest.mark.parametrize("backend", ["auto", "flashinfer", "cutedsl"])
def test_cuda_eligibility_and_ppu_short_circuit(
    upstream_source_root, resolve, device, capability, head_k, cuda_runtime, backend
):
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    options = dict(capability=capability, head_k=head_k, cuda_runtime=cuda_runtime)
    upstream, _ = resolve("cuda", backend, body=path, **options)
    copied, warnings = resolve(device, backend, **options)
    if device == "cuda":
        assert copied == upstream
    else:
        assert copied == (backend, "pla" if backend == "auto" else "triton")
    if capability == 110:
        expected = (
            "cutedsl"
            if backend == "cutedsl" and head_k == 128 and cuda_runtime >= 13
            else "triton"
        )
        assert upstream == (backend, expected)
    assert not warnings


@pytest.mark.upstream_source
def test_complete_resolver_retains_only_documented_ppu_differences(
    upstream_source_root,
):
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    copied = definition(ROOT / PATCH, "_resolve_gdn_prefill_backend")
    upstream = definition(path, "_resolve_gdn_prefill_backend")
    assert "pla" in {
        node.value
        for node in ast.walk(copied.returns)
        if isinstance(node, ast.Constant)
    }
    copied.returns = upstream.returns
    copied_doc = ast.get_docstring(copied, clean=False)
    upstream_doc = ast.get_docstring(upstream, clean=False)
    assert (
        copied_doc.split("    # PPU MODIFICATION: begin")[0].rstrip()
        == upstream_doc.rstrip()
    )
    copied.body[0] = upstream.body[0]
    ppu_branches = [
        node
        for node in copied.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "current_platform.is_ppu()"
    ]
    assert len(ppu_branches) == 1
    assert ast.unparse(ppu_branches[0].body[0].test) == (
        "backend in ('pla', 'auto') and _pla_prefill_supported(vllm_config)"
    )
    assert ast.unparse(ppu_branches[0].body[-1]) == "return (backend, 'triton')"
    copied.body.remove(ppu_branches[0])
    assert ast.dump(copied) == ast.dump(upstream)


def test_aiter_backend_has_a_log_name():
    messages = []
    config = NS(model_config=NS(hf_text_config=NS(linear_key_head_dim=128)))
    namespace = {
        "current_platform": NS(is_cpu=lambda: False, is_xpu=lambda: False),
        "logger": NS(info_once=lambda *args: messages.append(args)),
    }
    function(PATCH, "_log_gdn_backend_decision", namespace)(
        config, "aiter_flydsl", "aiter_flydsl"
    )
    assert messages[0][1:] == ("AITER FlyDSL", "aiter_flydsl", 128)


@pytest.mark.parametrize("active", ["aiter_flydsl", "hip", "pla"])
def test_constructor_selects_the_resolved_forward_method(active):
    class Parent:
        def __init__(self):
            self.parent_calls = 1

    class ChunkGatedDeltaRule(Parent):
        def forward_cuda(self):
            return "flashinfer"

        def forward_cutedsl(self):
            return "cutedsl"

        def forward_aiter_flydsl(self):
            return "aiter_flydsl"

        def forward_hip(self):
            return "hip"

        def forward_pla(self):
            return "pla"

        def forward_native(self):
            return "triton"

    namespace = {
        "ChunkGatedDeltaRule": ChunkGatedDeltaRule,
        "get_current_vllm_config": lambda: object(),
        "_resolve_gdn_prefill_backend": lambda _: (active, active),
        "_log_gdn_backend_decision": lambda *args: None,
    }
    ChunkGatedDeltaRule.__init__ = function(PATCH, "__init__", namespace)
    layer = ChunkGatedDeltaRule()
    assert layer.parent_calls == 1
    assert layer.gdn_prefill_backend == active
    assert layer._forward_method() == active


@pytest.mark.upstream_source
@pytest.mark.parametrize("caller", ["_warmup_prefill_kernels", "_forward_core"])
def test_pla_prefill_accepts_selected_upstream_call_keywords(
    upstream_source_root, caller
):
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    method = definition(path, "QwenGatedDeltaNetAttention." + caller)
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "chunk_gated_delta_rule"
    ]
    assert calls
    signature = inspect.signature(function(PATCH, "forward_pla", {}))
    for call in calls:
        assert not call.args
        assert all(keyword.arg is not None for keyword in call.keywords)
        signature.bind(object(), **{keyword.arg: object() for keyword in call.keywords})


@pytest.mark.parametrize(
    "device,backend,available,expected,error",
    [
        ("xpu", "auto", True, "sycl", None),
        ("xpu", "auto", False, "triton", None),
        ("xpu", "sycl", True, "sycl", None),
        ("xpu", "sycl", False, None, "not available"),
        ("ppu", "sycl", True, None, "only supported on XPU"),
        ("cuda", "sycl", True, None, "only supported on XPU"),
        ("rocm", "sycl", True, None, "only supported on XPU"),
        ("cpu", "sycl", True, None, "only supported on XPU"),
    ],
)
def test_sycl_routes_or_rejects_before_ppu_fallback(
    resolve, device, backend, available, expected, error
):
    if error:
        with pytest.raises(ValueError, match=error):
            resolve(device, backend, xpu_available=available)
    else:
        result, warnings = resolve(device, backend, xpu_available=available)
        assert result == (backend, expected)
        assert not warnings


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize(
    "device,backend,available,expected,error",
    [
        ("xpu", "auto", True, "sycl", None),
        ("xpu", "auto", False, "triton", None),
        ("xpu", "sycl", True, "sycl", None),
        ("xpu", "sycl", False, None, "not available"),
        ("ppu", "sycl", True, None, "only supported on XPU"),
    ],
)
def test_actual_sycl_provider_contract(
    upstream_source_root, resolve, copied, device, backend, available, expected, error
):
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    body = PATCH if copied else path
    if error:
        with pytest.raises(ValueError, match=error):
            resolve(device, backend, xpu_available=available, body=body)
    else:
        result, warnings = resolve(device, backend, xpu_available=available, body=body)
        assert result == (backend, expected)
        assert not warnings


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("backend", ["hip", "auto"])
@pytest.mark.parametrize(
    "options,active,error",
    [
        ({}, "hip", None),
        ({"dtype": "float16"}, "hip", None),
        ({"dtype": "float32"}, "triton", None),
        ({"head_k": 64}, "triton", None),
        ({"head_v": 64}, "triton", None),
        ({"hip_available": False}, "triton", "unavailable"),
        ({"gfx1151": False}, "triton", None),
    ],
)
def test_real_hip_reasons_drive_copied_and_upstream_routing(
    upstream_source_root, resolve, copied, backend, options, active, error
):
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    helper = function(
        path,
        "_hip_gdn_unsupported_reasons",
        {"torch": NS(bfloat16="bfloat16", float16="float16")},
    )
    kwargs = dict(gfx1151=True, hip_available=True) | options
    if error and backend == "hip":
        with pytest.raises(RuntimeError, match=error):
            resolve(
                "rocm",
                backend,
                hip_helper=helper,
                body=PATCH if copied else path,
                **kwargs,
            )
        return
    if backend == "hip" and options.get("gfx1151") is False:
        active = "hip"
    result, warnings = resolve(
        "rocm",
        backend,
        hip_helper=helper,
        body=PATCH if copied else path,
        **kwargs,
    )
    assert result == (backend, active)
    assert bool(warnings) == (backend == "hip" and active == "triton")
    if warnings:
        assert "Falling back to Triton/FLA" in warnings[0][0]


@pytest.mark.parametrize("device", ["ppu", "cuda", "cpu"])
def test_explicit_hip_rejects_non_rocm_before_backend_probes(resolve, device):
    with pytest.raises(RuntimeError, match="ROCm backend"):
        resolve(device, "hip", hip_available=True)


def test_hip_backend_has_a_log_name():
    messages = []
    config = NS(model_config=NS(hf_text_config=NS(linear_key_head_dim=128)))
    namespace = {
        "current_platform": NS(is_cpu=lambda: False, is_xpu=lambda: False),
        "logger": NS(info_once=lambda *args: messages.append(args)),
    }
    function(PATCH, "_log_gdn_backend_decision", namespace)(config, "hip", "hip")
    assert messages[0][1:] == ("HIP", "hip", 128)


@pytest.mark.upstream_source
def test_complete_constructor_and_logger_keep_only_ppu_extensions(upstream_source_root):
    path = (
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    copied = definition(ROOT / PATCH, "__init__")
    original = definition(path, "ChunkGatedDeltaRule.__init__")
    assert ast.unparse(copied.body[0]) == "super(ChunkGatedDeltaRule, self).__init__()"
    copied.body[0] = original.body[0]
    warning = next(node for node in copied.body if isinstance(node, ast.If))
    assert "pla" in ast.unparse(warning.test)
    warning.test = next(node for node in original.body if isinstance(node, ast.If)).test
    dispatch = next(
        node for node in copied.body if isinstance(node, ast.If) and node is not warning
    )
    tail = dispatch
    while tail.orelse and isinstance(tail.orelse[0], ast.If):
        previous, tail = tail, tail.orelse[0]
    assert ast.unparse(tail.test) == "active_backend == 'pla'"
    previous.orelse = tail.orelse
    assert ast.dump(copied) == ast.dump(original)
    copied_log = definition(ROOT / PATCH, "_log_gdn_backend_decision")
    upstream_log = definition(path, "_log_gdn_backend_decision")
    mapping = next(node for node in ast.walk(copied_log) if isinstance(node, ast.Dict))
    position = next(i for i, key in enumerate(mapping.keys) if key.value == "pla")
    assert mapping.values[position].value == "PLA FlashQLA"
    del mapping.keys[position]
    del mapping.values[position]
    assert ast.dump(copied_log) == ast.dump(upstream_log)


@pytest.mark.upstream_source
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("family", [False, True])
def test_real_hip_availability_requires_both_op_and_rdna35(
    modules, upstream_source_root, registered, family
):
    probes = []
    modules("vllm.platforms.rocm", on_gfx115x=lambda: probes.append("rdna35") or family)
    op_namespace = NS(rdna35_gdn_chunked=object()) if registered else NS()
    available = function(
        upstream_source_root / "vllm/model_executor/layers/mamba/ops/hip_gdn_chunk.py",
        "is_hip_gdn_available",
        {"torch": NS(ops=NS(_rocm_C=op_namespace))},
    )
    assert available() is (registered and family)
    assert probes == (["rdna35"] if registered else [])


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "case",
    [
        "batch",
        "dtype",
        "value_dtype",
        "key_dim",
        "value_dim",
        "heads",
        "bounds",
        "unavailable",
    ],
)
def test_real_hip_support_rejects_and_forward_preserves_fallback_arguments(
    upstream_source_root, case
):
    helper_path = (
        upstream_source_root / "vllm/model_executor/layers/mamba/ops/hip_gdn_chunk.py"
    )
    q, v = (
        NS(shape=(1, 8, 2, 128), dtype="bf16"),
        NS(shape=(1, 8, 4, 128), dtype="bf16"),
    )
    bounds = object()
    if case == "batch":
        q.shape = (2, 8, 2, 128)
    elif case == "dtype":
        q.dtype = v.dtype = "fp32"
    elif case == "value_dtype":
        v.dtype = "fp16"
    elif case == "key_dim":
        q.shape = (1, 8, 2, 64)
    elif case == "value_dim":
        v.shape = (1, 8, 4, 64)
    elif case == "heads":
        v.shape = (1, 8, 3, 128)
    elif case == "bounds":
        bounds = None
    supported = function(
        helper_path,
        "is_hip_gdn_supported",
        {
            "torch": NS(bfloat16="bf16", float16="fp16"),
            "is_hip_gdn_available": lambda: case != "unavailable",
        },
    )
    assert supported(q, v, bounds) is False
    calls, result = [], object()
    layer = NS(forward_native=lambda **kwargs: calls.append(kwargs) or result)
    forward = function(
        upstream_source_root
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "ChunkGatedDeltaRule.forward_hip",
        {"is_hip_gdn_supported": supported},
    )
    kwargs = dict(
        q=q,
        k=object(),
        v=v,
        g=object(),
        beta=object(),
        initial_state=object(),
        output_final_state=True,
        cu_seqlens=bounds,
        chunk_indices=object(),
        chunk_offsets=object(),
        use_qk_l2norm_in_kernel=False,
        core_attn_out=object(),
    )
    assert forward(layer, **kwargs) is result
    assert calls == [kwargs]


def test_sycl_backend_has_a_log_name():
    messages = []
    config = NS(model_config=NS(hf_text_config=NS(linear_key_head_dim=128)))
    namespace = {
        "current_platform": NS(is_cpu=lambda: False, is_xpu=lambda: False),
        "logger": NS(info_once=lambda *args: messages.append(args)),
    }
    function(PATCH, "_log_gdn_backend_decision", namespace)(config, "sycl", "sycl")
    assert messages[0][1:] == ("SYCL", "sycl", 128)
