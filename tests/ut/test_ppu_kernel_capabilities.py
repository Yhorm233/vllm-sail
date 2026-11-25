# SPDX-License-Identifier: Apache-2.0
"""Exercise imported providers and real compressor dispatch without device deps."""

from __future__ import annotations

import ast
import sys
import types
from abc import abstractmethod
from enum import Enum
from pathlib import Path

import pytest

from tests.support.modules import load_patch
from tests.support.source import definition, function

ROOT = Path(__file__).parents[2]


@pytest.mark.upstream_source
@pytest.mark.parametrize("name", ["__init__", "_compress_norm_rope_store"])
def test_compressor_complete_body_matches_provider_outside_ppu_regions(
    upstream_source_root,
    name,
):
    copied = definition(
        ROOT / "vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py",
        name,
    )
    original = definition(
        upstream_source_root / "vllm/models/deepseek_v4/compressor.py",
        "DeepseekCompressor." + name,
    )
    copied.decorator_list = original.decorator_list = []
    import inspect

    if name != "__init__":
        assert inspect.cleandoc(
            ast.get_docstring(copied, clean=False)
        ) == inspect.cleandoc(ast.get_docstring(original, clean=False))
    copied.body[0] = original.body[0]
    for node in ast.walk(copied):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
            node.test.values = [
                value
                for value in node.test.values
                if ast.unparse(value) != "not current_platform.is_ppu()"
            ]
    assert ast.dump(copied, include_attributes=False) == ast.dump(
        original,
        include_attributes=False,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("fail", [False, True])
def test_compressor_real_projection_preserves_weight_and_fp32_mm_contract(
    upstream_source_root,
    fail,
):
    """执行完整 projection；mm 边界保留原权重和 FP32 输出要求，不冒充数值验证。"""
    weight, transposed, inputs, output = object(), object(), object(), object()
    error = RuntimeError("FP32-output mm unavailable")
    calls = []

    def mm(lhs, rhs, *, out_dtype):
        assert lhs is inputs and rhs is transposed and out_dtype == "fp32"
        calls.append((lhs, rhs, out_dtype))
        if fail:
            raise error
        return output

    owner = types.SimpleNamespace(weight=types.SimpleNamespace(T=transposed))
    weight = owner.weight
    project = function(
        upstream_source_root / "vllm/models/deepseek_v4/compressor.py",
        "DeepseekV4CompressorKVScoreLinear.forward",
        {"torch": types.SimpleNamespace(mm=mm, float32="fp32")},
    )
    if fail:
        with pytest.raises(RuntimeError) as raised:
            project(owner, inputs)
        assert raised.value is error
    else:
        assert project(owner, inputs) is output
    assert calls == [(inputs, transposed, "fp32")]
    assert owner.weight is weight


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "patch_file,inventory_name,provider",
    [
        ("import_gates.py", "_CONSUMERS", "vllm.utils.import_utils"),
        (
            "attention/flashmla_ops.py",
            "_FLASHMLA_ALIAS_CONSUMERS",
            "vllm.v1.attention.ops.flashmla",
        ),
    ],
)
def test_optional_gate_consumer_inventory_matches_upstream(
    upstream_source_root, patch_file, inventory_name, provider
):
    tree = ast.parse((ROOT / "vllm_sail/patch/enhancement" / patch_file).read_text())
    inventory = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == inventory_name
            for target in node.targets
        )
    )
    discovered = {name: set() for name in inventory}
    source = upstream_source_root / "vllm"
    for path in source.rglob("*.py"):
        parts = path.relative_to(source).with_suffix("").parts
        module = ".".join(("vllm", *(parts[:-1] if parts[-1] == "__init__" else parts)))
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == provider:
                for alias in node.names:
                    if alias.name in discovered:
                        assert alias.asname is None, "renamed alias needs rebinding"
                        discovered[alias.name].add(module)
            pending.extend(ast.iter_child_nodes(node))
    assert discovered == {name: set(consumers) for name, consumers in inventory.items()}


@pytest.mark.upstream_source
@pytest.mark.parametrize("capability", [80, 89])
@pytest.mark.parametrize("quantized", [False, True])
def test_qwen35_real_budget_blocks_flashinfer_and_preserves_norm_fallback(
    modules, upstream_source_root, capability, quantized
):
    ns = types.SimpleNamespace
    source = upstream_source_root / "vllm"
    fusion = source / "compilation/passes/fusion/allreduce_rms_fusion.py"
    mapping = next(
        ast.literal_eval(node.value)
        for node in ast.parse(fusion.read_text()).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "FI_ALLREDUCE_FUSION_MAX_SIZE_MB"
    )
    platform = ns(
        is_cuda=lambda: True,
        get_device_capability=lambda: ns(to_int=lambda: capability),
    )
    modules("vllm.platforms", current_platform=platform)
    modules(
        "vllm.compilation.passes.fusion.allreduce_rms_fusion",
        FI_ALLREDUCE_FUSION_MAX_SIZE_MB=mapping,
    )
    budget = function(
        source / "config/compilation.py",
        "PassConfig.default_fi_allreduce_fusion_max_size_mb",
        {},
    )
    assert budget() == {}
    assert mapping[90][2] > 0
    pass_config = ns(default_fi_allreduce_fusion_max_size_mb=budget)
    modules("vllm.config.compilation", PassConfig=pass_config)

    def forbidden(*args, **kwargs):
        pytest.fail("空 capability budget 不得创建 workspace 或启动融合 kernel")

    hidden = ns(
        is_cuda=True,
        dim=lambda: 2,
        is_contiguous=lambda: True,
        dtype="bf16",
        shape=(1, 128),
        element_size=lambda: 2,
    )
    residual, weight, scale = object(), object(), object()
    reduced = ns(dtype="bf16", shape=hidden.shape)
    normed, new_residual, codes = object(), object(), object()
    calls = []

    def allreduce(value):
        assert value is hidden
        calls.append("allreduce")
        return reduced

    class Norm:
        reduce_inputs = True
        variance_epsilon = 1e-6

        def __call__(self, value, skip):
            assert value is reduced and skip is residual
            calls.append("norm")
            return normed, new_residual

    norm = Norm()
    norm.weight = weight
    torch = ns(
        bfloat16="bf16",
        float16="fp16",
        tensor=lambda *a, **kw: ns(element_size=lambda: 2),
    )
    if quantized:
        path = source / "models/qwen3_5/nvidia/ops/fused_allreduce_norm_quant.py"
        namespace = dict(
            torch=torch,
            PassConfig=pass_config,
            _workspace_unsupported=False,
            fi_ar=ns(fi_ar_available=True, get_fi_ar_quant_workspace=forbidden),
            get_tensor_model_parallel_world_size=lambda: 2,
        )
        namespace["_get_workspace"] = function(path, "_get_workspace", namespace)
        compute = function(path, "fused_allreduce_gemma_rms_norm_quant", namespace)
        assert compute(hidden, residual, weight, 1e-6, scale, "fp8") is None

        def norm_quant(value, skip, gamma, eps, actual_scale, nvfp4, write_y):
            assert value is reduced and skip is residual and gamma is weight
            assert eps == 1e-6 and actual_scale is scale
            assert nvfp4 == "fp8" and write_y
            calls.append("norm_quant")
            return codes, None, normed

        forward = function(
            source / "models/qwen3_5/nvidia/model.py",
            "Qwen3_5RMSNorm.forward_fused",
            dict(
                _norm_quant_key=lambda _: "fp8",
                kNvfp4Dynamic="nvfp4",
                kFp8DynamicTokenSym="dynamic_fp8",
                fused_allreduce_gemma_rms_norm_quant=compute,
                tensor_model_parallel_all_reduce=allreduce,
                gemma_add_rms_norm_quant=norm_quant,
                QuantizedActivation=ns,
            ),
        )
        qa, y, skip = forward(
            norm, hidden, residual, ns(input_scale=scale), write_y=True
        )
        assert qa.data is codes and qa.scale is None and y is normed
        assert qa.orig_dtype == reduced.dtype and qa.orig_shape == reduced.shape
        assert qa.quant_key == "fp8" and skip is residual
        assert calls == ["allreduce", "norm_quant"]
    else:
        path = source / "model_executor/layers/fused_allreduce_gemma_rms_norm.py"
        namespace = dict(
            torch=torch,
            envs=ns(VLLM_BATCH_INVARIANT=False),
            _FI_SUPPORTED_DTYPES=("bf16", "fp16"),
            MiB=1024 * 1024,
            flashinfer_trtllm_fused_allreduce_norm=forbidden,
            get_fi_ar_workspace=forbidden,
            _AR_RESIDUAL_RMS_NORM=object(),
            get_tensor_model_parallel_world_size=lambda: 2,
            dbo_enabled=lambda feature: False,
            _can_use_aiter_fused_ar_rms=lambda _: False,
            tensor_model_parallel_all_reduce=allreduce,
        )
        namespace["_max_token_num"] = function(path, "_max_token_num", namespace)
        namespace["_can_use_flashinfer"] = function(
            path, "_can_use_flashinfer", namespace
        )
        compute = function(path, "fused_allreduce_gemma_rms_norm", namespace)
        assert compute(hidden, residual, norm) == (normed, new_residual)
        assert calls == ["allreduce", "norm"]


@pytest.mark.upstream_source
@pytest.mark.parametrize("missing_shard", [None, "q", "index_g"])
def test_step5_real_parent_tracks_complete_and_partial_reload(
    upstream_source_root, missing_shard
):
    """执行完整类及真实父类 loader；存储拷贝由严格边界 double 记录。"""
    ns = types.SimpleNamespace
    linear = upstream_source_root / "vllm/model_executor/layers/linear.py"
    model = upstream_source_root / "vllm/models/step5/nvidia/model.py"
    namespace = dict(
        ColumnParallelLinear=type("ColumnParallelLinear", (), {}),
        BlockQuantScaleParameter=type("BlockQuantScaleParameter", (), {}),
    )
    classes = [
        definition(linear, "StackedColumnParallelLinear"),
        definition(linear, "QKVParallelLinear"),
        definition(model, "Step5FusedQKVIndexerLinear"),
    ]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *classes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(model), "exec"), namespace)
    cls = namespace["Step5FusedQKVIndexerLinear"]
    layer = object.__new__(cls)
    layer.loaded_shard_ids = {}
    layer.num_heads = layer.num_kv_heads = 2
    layer.head_size = layer.v_head_size = layer.proxy_dim = 1
    layer.num_kv_head_replicas = layer.index_kv_head_replicas = 1
    layer.local_index_q_heads, layer.local_index_kv_heads = 2, 1
    layer.local_index_groups = 1
    layer.local_gate_size = 2
    layer.index_group_start = layer.tp_rank = 0
    layer.qkv_size = 6
    calls = []

    class Storage:
        def narrow(self, dim, offset, size):
            assert dim == 0
            return ns(copy_=lambda source: calls.append((offset, size, source)))

    param = ns(
        data=Storage(),
        load_qkv_weight=lambda **kwargs: calls.append(kwargs),
    )
    layer.named_parameters = lambda: iter([("weight", param)])
    incoming = ns(narrow=lambda dim, offset, size: (dim, offset, size))

    def load(omit=None):
        calls.clear()
        for shard in cls.shard_ids:
            if shard != omit:
                layer.weight_loader_v2(param, incoming, shard)
        assert layer.loaded_shard_ids == {
            "weight": set(cls.shard_ids) - ({omit} if omit else set()),
        }

    with pytest.raises(NotImplementedError, match="pre-fused"):
        layer.weight_loader_v2(param, incoming, None)
    with pytest.raises(ValueError, match="Shard id"):
        layer.weight_loader_v2(param, incoming, "unknown")
    assert layer.loaded_shard_ids == {} and calls == []
    load()
    assert calls[:3] == [
        dict(
            loaded_weight=incoming,
            num_heads=1,
            shard_id=shard,
            shard_offset=offset,
            shard_size=2,
        )
        for shard, offset in [("q", 0), ("k", 2), ("v", 4)]
    ]
    assert calls[3:] == [
        (6, 2, (0, 0, 2)),
        (8, 1, (0, 0, 1)),
        (9, 1, (0, 0, 1)),
        (10, 2, (0, 0, 2)),
    ]
    layer.assert_fully_loaded("layers.0.qkv_indexer_proj.weight")
    assert layer.loaded_shard_ids == {}
    load(missing_shard)
    if missing_shard is None:
        layer.assert_fully_loaded("reload.weight")
    else:
        with pytest.raises(ValueError, match=missing_shard):
            layer.assert_fully_loaded("reload.weight")
    assert layer.loaded_shard_ids == {}
    layer.validate_shard_id("index_q", param)
    layer.reset_load_state()
    assert layer.loaded_shard_ids == {}
    load()
    layer.assert_fully_loaded("reload.weight")
    assert layer.loaded_shard_ids == {}


@pytest.mark.upstream_source
@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "ep,tp_size,backend,expected",
    [
        (True, 4, "deep_gemm_mega_moe", True),
        (False, 4, "deep_gemm_mega_moe", False),
        (True, 1, "deep_gemm_mega_moe", False),
        (True, 4, "auto", False),
    ],
)
def test_step5_sequence_parallel_keeps_ep_tp_backend_gate_with_pipeline(
    upstream_source_root, pp_size, ep, tp_size, backend, expected
):
    ns = types.SimpleNamespace
    resolve = function(
        upstream_source_root / "vllm/models/step5/nvidia/model.py",
        "_step5_use_sequence_parallel",
        {},
    )
    config = ns(
        parallel_config=ns(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
            enable_expert_parallel=ep,
        ),
        kernel_config=ns(moe_backend=backend),
    )
    assert resolve(config) is expected


def complete_warmup_contract(modules, provider, worker, platform, *, entered=None):
    """Strict external callbacks for the real copied warmup body."""
    import time

    ns = types.SimpleNamespace
    entered = entered or (lambda: None)
    platform.is_cuda = lambda: True
    platform.is_rocm = lambda: False
    platform.is_device_capability_family = lambda _: False
    worker.vllm_config.cache_config = ns(enable_prefix_caching=False)
    worker.use_v2_model_runner = True
    modules("vllm.distributed.parallel_state", get_pp_group=lambda: ns(world_size=1))
    worker.model_runner.attn_groups = []
    worker.vllm_config.model_config = ns()
    worker.vllm_config.use_flashinfer_replayssm = False
    worker.vllm_config.compilation_config = ns(cudagraph_capture_sizes=[])
    config = worker.vllm_config.kernel_config
    if not hasattr(config, "enable_cutedsl_warmup"):
        config.enable_cutedsl_warmup = False
    config.enable_flashinfer_autotune = False

    def registry_warmup():
        entered()
        for kernel, keys in list(
            worker.model_runner.jit_warmup_registry._registrations.items()
        ):
            kernel.compile_many(keys)

    worker.model_runner.jit_warmup_registry.warmup = registry_warmup

    def replayssm_lifecycle(runner):
        assert runner is worker.model_runner
        assert runner.vllm_config.use_flashinfer_replayssm is False

    worker.model_runner.vllm_config = worker.vllm_config
    callbacks = {
        "replayssm_lifecycle_warmup": replayssm_lifecycle,
        "qwen_triton_warmup": lambda runner, config: entered(),
        "qwen_vl_triton_warmup": lambda runner: None,
        "mamba_triton_warmup": lambda runner: None,
        "kimi_k3_triton_warmup": lambda worker: None,
        "watermark_sample_warmup": lambda worker: None,
        "qwen4_exp_qsa_triton_warmup": lambda worker: None,
        "_warmup_bf16x3_router_gemm": lambda model, tokens: None,
        "_warmup_ll_bf16_router_gemm": lambda model: None,
        "_warmup_gemm_rs_ar": lambda: None,
        "flashinfer_sparse_mla_decode_autotune_warmup": lambda worker: None,
        "deepseek_v4_sparse_mla_attention_warmup": lambda worker: None,
        "b12x_warmup": lambda worker, sizes: None,
        "mixed_batch_attention_warmup": lambda worker: None,
        "is_deep_gemm_supported": lambda: False,
        "has_flashinfer": lambda: False,
    }
    # 纯替身没有 upstream helper；真实源码 provider 必须保留其集体决策实现。
    provider.__dict__.setdefault("_warmup_flashinfer_attention", lambda worker: None)
    provider.__dict__.update(
        callbacks,
        current_platform=platform,
        time=time,
        envs=ns(
            VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=False, VLLM_DEEP_GEMM_WARMUP="skip"
        ),
        logger=ns(
            info=lambda *a: None, info_once=lambda *a: None, exception=lambda *a: None
        ),
    )
    modules(
        "vllm.model_executor.warmup.step5_dsa_warmup",
        step5_dsa_warmup=lambda worker: None,
    )


@pytest.mark.parametrize(
    "cute_consumer",
    [
        "vllm.models.deepseek_v4.common.ops.cache_utils",
        "vllm.models.glm5next.nvidia.sparse_indexer",
    ],
)
def test_optional_gates_rebind_preloaded_and_future_consumers(modules, cute_consumer):
    platform = types.SimpleNamespace(ppu=True, is_ppu=lambda: platform.ppu)
    modules("vllm.platforms", current_platform=platform)

    def probe():
        return True

    provider = modules("vllm.utils.import_utils", has_cutedsl=probe, has_humming=probe)
    cute = modules(cute_consumer, has_cutedsl=probe)
    humming = modules(
        "vllm.model_executor.layers.quantization.utils.humming.moe", has_humming=probe
    )
    patched = load_patch("vllm_sail/patch/enhancement/import_gates.py")
    assert cute.has_cutedsl is provider.has_cutedsl
    assert humming.has_humming is provider.has_humming
    assert not cute.has_cutedsl() and not humming.has_humming()
    future = modules("future", has_cutedsl=provider.has_cutedsl)
    assert not future.has_cutedsl()
    platform.ppu = False
    assert cute.has_cutedsl() and humming.has_humming()
    patched._rebind_loaded_consumers()
    humming.has_humming = lambda: "other override"
    patched._rebind_loaded_consumers()
    assert humming.has_humming() == "other override"


@pytest.mark.parametrize(
    "ppu,head,two_stage,expected",
    [
        (True, 512, False, "triton"),
        (True, 128, False, "triton"),
        (False, 512, False, "cute"),
        (False, 512, True, "two_stage"),
    ],
)
def test_compressor_uses_supported_kernel(modules, ppu, head, two_stage, expected):
    base = "vllm.models.deepseek_v4"
    calls = []
    platform = types.SimpleNamespace(
        is_ppu=lambda: ppu,
        is_cuda=lambda: not two_stage,
        is_rocm=lambda: two_stage,
        is_xpu=lambda: False,
    )

    def kernel(name):
        def call(**kwargs):
            calls.append((name, kwargs))

        return call

    meta = types.SimpleNamespace(
        token_to_req_indices=object(),
        slot_mapping=types.SimpleNamespace(shape=(3,)),
        block_table=object(),
        block_size=64,
        c128_boundary=True,
        num_decode_tokens=1,
    )
    context = types.SimpleNamespace(
        attn_metadata={"state": meta, "kv": object()},
        cudagraph_runtime_mode="none",
        capturing=False,
    )
    cache = types.SimpleNamespace(shape=(2, 64, 2048), dtype="u8")
    cls = type("DeepseekCompressor", (), {"_compress_norm_rope_store": lambda *a: None})
    module = modules(
        f"{base}.compressor",
        DeepseekCompressor=cls,
        current_platform=platform,
        get_forward_context=lambda: context,
        cast=lambda kind, value: value,
        CompressorMetadata=object,
        Any=object,
        CUDAGraphMode=types.SimpleNamespace(FULL="full"),
        torch=types.SimpleNamespace(
            uint8="u8",
            float8_e4m3fn="fp8",
            cuda=types.SimpleNamespace(
                is_current_stream_capturing=lambda: context.capturing
            ),
        ),
        _SAVE_PARTIAL_STATES_KERNEL=kernel("save"),
        compress_norm_rope_store_triton=kernel("triton"),
        compress_norm_rope_store_two_stage_triton=kernel("two_stage"),
    )
    original_forward = function(
        "tests/fixtures/deepseek_v4_compressor_forward.py", "forward", module.__dict__
    )
    cls.forward = original_forward
    if not ppu:
        modules(
            f"{base}.nvidia.ops.sparse_attn_compress_cutedsl",
            _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL=kernel("cute"),
        )
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py")
    obj = cls.__new__(cls)
    obj.__dict__.update(
        coff=2,
        head_dim=head,
        state_cache=types.SimpleNamespace(prefix="state", kv_cache=cache),
        ape=object(),
        compress_ratio=4,
        k_cache_prefix="kv",
        _static_forward_context={"kv": types.SimpleNamespace(kv_cache=cache)},
        overlap=False,
        eager_scratch_pool=None,
        _use_two_stage_fused_compressor=two_stage,
        _compress_scratch=object(),
        rope_head_dim=64,
        use_fp4_cache=False,
        norm=types.SimpleNamespace(weight=object()),
        rms_norm_eps=1e-6,
        _quant_block=64,
        _token_stride=576,
        _scale_dim=8,
    )
    scores = types.SimpleNamespace(split=lambda *args, **kwargs: (object(), object()))
    obj.forward(scores, object(), types.SimpleNamespace(cos_sin_cache=object()))
    assert [name for name, _ in calls] == ["save", expected]
    assert ("store_full_kv" in calls[-1][1]) == (expected == "cute")
    assert obj.forward.__func__.__globals__ is module.__dict__
    assert obj.forward.__func__ is original_forward
    assert obj._compress_norm_rope_store.__func__.__globals__ is module.__dict__
    # Non-boundary C128 steps still save partial state, except full graph capture.
    if not two_stage and head == 512:
        obj.compress_ratio = 128
        meta.c128_boundary = False
        calls.clear()
        obj.forward(scores, object(), types.SimpleNamespace(cos_sin_cache=object()))
        assert [name for name, _ in calls] == ["save"]
        context.cudagraph_runtime_mode = "full"
        context.capturing = True
        calls.clear()
        obj.forward(scores, object(), types.SimpleNamespace(cos_sin_cache=object()))
        assert [name for name, _ in calls] == ["save", expected]
    context.attn_metadata = None
    calls.clear()
    obj.forward(scores, object(), object())
    assert not calls


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("head,ratio", [(512, 4), (512, 128), (128, 4)])
@pytest.mark.parametrize("warmup", [True, False])
@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "bf16"])
def test_compressor_constructor_registers_supported_warmup(
    modules, ppu, head, ratio, warmup, cache_dtype
):
    base = "vllm.models.deepseek_v4"
    calls = []

    def kernel(name):
        return types.SimpleNamespace(register_warmup=lambda *a, **k: calls.append(name))

    def noop(*a, **k):
        return object()

    cls = type(
        "DeepseekCompressor", (), {"forward": noop, "_compress_norm_rope_store": noop}
    )
    provider = modules(
        f"{base}.compressor",
        DeepseekCompressor=cls,
        current_platform=types.SimpleNamespace(
            device_type="cuda",
            is_cuda=lambda: True,
            is_ppu=lambda: ppu,
            is_cpu=lambda: False,
        ),
        torch=types.SimpleNamespace(empty=noop, float32="fp32"),
        nn=types.SimpleNamespace(Parameter=noop),
        _prefer_two_stage_compressor=lambda: False,
        MergedColumnParallelLinear=lambda *a, **k: pytest.fail(
            "非 CPU 不应使用 BF16 输出 projection"
        ),
        DeepseekV4CompressorKVScoreLinear=noop,
        RMSNorm=noop,
        CompressorStateCache=noop,
        _SAVE_PARTIAL_STATES_KERNEL=kernel("save"),
    )
    modules(
        f"{base}.common.ops.fused_compress_quant_cache",
        _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL=kernel("triton"),
    )
    # On PPU the CuTeDSL module is deliberately unavailable: even importing it
    # during construction is a regression, regardless of the forward guard.
    if not ppu:
        modules(
            f"{base}.nvidia.ops.sparse_attn_compress_cutedsl",
            _SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL=kernel("cute128"),
            _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL=kernel("cute4"),
            _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL=kernel("cute4full"),
            _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL=kernel("cutestorefull"),
            _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL=kernel("cutestore"),
        )
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v4_compressor.py")
    ns = types.SimpleNamespace
    config = ns(
        model_config=ns(
            hf_config=ns(qk_rope_head_dim=64, rms_norm_eps=1e-6), max_model_len=1024
        ),
        scheduler_config=ns(max_num_seqs=4, max_num_batched_tokens=64),
        compilation_config=ns(static_forward_context={}),
        kernel_config=ns(enable_jit_warmup=warmup),
        cache_config=ns(cache_dtype=cache_dtype),
    )
    obj = cls(config, ratio, 256, head)
    assert obj.__init__.__func__.__globals__ is provider.__dict__
    assert obj.coff == (2 if ratio == 4 else 1)
    assert obj._quant_block == (64 if head == 512 else 128)
    assert obj._token_stride == (576 if head == 512 else 128)
    assert obj._scale_dim == (8 if head == 512 else 4)
    if not warmup:
        assert calls == []
    elif ppu or head != 512:
        assert calls == ["save", "triton"]
    elif ratio == 4:
        assert calls == [
            "save",
            "cute4" if cache_dtype == "fp8_ds_mla" else "cute4full",
        ]
    else:
        assert calls == [
            "save",
            "cute128",
            "cutestore" if cache_dtype == "fp8_ds_mla" else "cutestorefull",
        ]


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("process_local_only", [True, False])
@pytest.mark.parametrize("jit_enabled", [True, False])
def test_warmup_excludes_cutedsl_only_on_ppu(
    modules, ppu, process_local_only, jit_enabled
):
    ns = types.SimpleNamespace
    import contextlib

    calls = []
    modules(
        "vllm.platforms",
        current_platform=ns(is_ppu=lambda: ppu, has_device_capability=lambda _: True),
    )
    modules("vllm.logger", init_logger=lambda _: ns(warning=lambda *a: None))

    class VllmCuTeDSLJitKernel:
        def compile_many(self, keys):
            assert not ppu, "CuTeDSL compilation reached PPU"
            calls.append("cute")

    class CuTeKernel(VllmCuTeDSLJitKernel):
        pass

    class TritonKernel:
        def compile_many(self, keys):
            calls.append("triton")

    class SamplerKernel:
        def compile_many(self, keys):
            calls.append("sampler_compiled")

    modules(
        "vllm.model_executor.warmup.jit_warmup_cutedsl_helper",
        VllmCuTeDSLJitKernel=VllmCuTeDSLJitKernel,
    )
    cute, triton = CuTeKernel(), TritonKernel()
    registrations = {cute: [((), {})], triton: [((), {"head_dim": 128})]}
    retained = registrations[triton]

    def register_top_k_top_p_warmups():
        calls.append("sampler_registered")
        registrations[SamplerKernel()] = [((), {})]

    modules(
        "vllm.v1.sample.ops.topk_topp_sampler",
        register_top_k_top_p_warmups=register_top_k_top_p_warmups,
    )
    worker = ns(
        model_runner=ns(
            jit_warmup_registry=ns(
                _registrations=registrations, activate=lambda: contextlib.nullcontext()
            ),
            is_last_pp_rank=True,
            is_pooling_model=False,
        ),
        vllm_config=ns(
            kernel_config=ns(enable_jit_warmup=jit_enabled, enable_cutedsl_warmup=True)
        ),
        get_model=lambda: "model",
        scheduler_config=ns(max_num_batched_tokens=64),
    )

    def legacy():
        assert not ppu, "Legacy CuTeDSL warmup reached PPU"
        calls.append("legacy")

    legacy_module = modules(
        "vllm.model_executor.warmup.cutedsl_warmup", cutedsl_warmup=legacy
    )

    def upstream(actual_worker, *, process_local_only=False):
        assert actual_worker is worker
        calls.append(("upstream", process_local_only))
        if worker.vllm_config.kernel_config.enable_jit_warmup:
            for kernel, keys in registrations.items():
                kernel.compile_many(keys)
        provider.cutedsl_warmup()
        return "result"

    # The consumer imported the function by value before the patch was loaded.
    provider = modules(
        "vllm.model_executor.warmup.kernel_warmup",
        kernel_warmup=upstream,
        cutedsl_warmup=legacy,
    )
    entered = False

    def enter_body():
        nonlocal entered
        if not entered:
            calls.append(("upstream", process_local_only))
            entered = True

    complete_warmup_contract(
        modules,
        provider,
        worker,
        sys.modules["vllm.platforms"].current_platform,
        entered=enter_body,
    )
    modules(
        "vllm_sail.model_executor.warmup.deep_gemm_warmup",
        deep_gemm_warmup=lambda model, size: calls.append((model, size)),
    )
    patched = load_patch("vllm_sail/patch/enhancement/kernel_warmup.py")
    patched._ppu_deep_gemm_warmup_enabled = lambda: True
    assert provider.kernel_warmup(worker, process_local_only=process_local_only) == (
        None if ppu else "result"
    )
    expected = []
    if ppu:
        expected += ["sampler_registered"]
    expected += [("upstream", process_local_only)]
    if jit_enabled:
        expected += ["triton"] if ppu else ["cute", "triton"]
        if ppu:
            expected += ["sampler_compiled"]
    if not ppu:
        expected += ["legacy"]
    elif not process_local_only:
        expected += [("model", 64)]
    assert calls == expected
    assert registrations[triton] is retained
    assert (cute in registrations) is (not ppu)
    assert worker.vllm_config.kernel_config.enable_jit_warmup is jit_enabled
    assert worker.vllm_config.kernel_config.enable_cutedsl_warmup is True
    assert legacy_module.cutedsl_warmup is provider.cutedsl_warmup
    calls.clear()
    legacy_module.cutedsl_warmup()
    assert calls == ([] if ppu else ["legacy"])


class _TritonLang:
    """Nameable stand-in for ``triton.language`` inside never-executed fixtures."""

    float8e4nv = "fp8e4nv"
    bfloat16 = "bfloat16"


tl = _TritonLang()


def _fp8_direct(x):
    return x.to(tl.float8e4nv)


def _e4m3_helper(x):
    return x.to(tl.float8e4nv)


# ``@triton.jit`` helpers are inlined at trace time, so only their ``fn`` is read.
_INLINE_E4M3 = types.SimpleNamespace(fn=_e4m3_helper)


def _fp8_inlined(x):
    return _INLINE_E4M3(x)


def _bf16_only(x):
    return x.to(tl.bfloat16)


_FP8_COMPILE_ERROR = ValueError(
    "type fp8e4nv not supported in this architecture. "
    "The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')"
)


def _fixture_kernel(name, jit_fn, compiled, *, error=None):
    """Build a warmup registration shaped like a vLLM Triton kernel wrapper."""

    def compile_many(self, keys):
        compiled.append(name)
        if error is not None:
            raise error

    attrs = {"kernel": staticmethod(jit_fn), "compile_many": compile_many}
    return type(name, (), attrs)()


@pytest.mark.parametrize(
    "ppu,arch,fp8_skipped",
    [
        (True, (8, 0), True),
        (True, (8, 9), False),
        (False, (8, 0), False),
    ],
)
def test_warmup_excludes_fp8_kernels_without_native_fp8(
    modules, ppu, arch, fp8_skipped
):
    """Triton rejects tl.float8e4nv on sm80-class devices, so drop those warmups."""
    compiled = []
    direct = _fixture_kernel("FusedInvRopeFP8QuantKernel", _fp8_direct, compiled)
    inlined = _fixture_kernel(
        "FusedIndexerQRopeQuantTritonKernel", _fp8_inlined, compiled
    )
    clean = _fixture_kernel("PackSeqTritonKernel", _bf16_only, compiled)
    registrations = {item: [((), {})] for item in (direct, inlined, clean)}

    assert _run_ppu_warmup(modules, ppu, arch, registrations) == (
        None if ppu else "result"
    )

    skipped = fp8_skipped and ppu
    assert (direct in registrations) is not skipped
    assert (inlined in registrations) is not skipped
    assert clean in registrations
    assert len(compiled) == 3 - 2 * skipped


def test_warmup_does_not_suppress_unclassified_fp8_errors(modules):
    """An error without inspectable Triton inputs must still propagate."""
    compiled = []
    signature_only = _fixture_kernel(
        "SignatureFp8Kernel", _bf16_only, compiled, error=_FP8_COMPILE_ERROR
    )
    clean = _fixture_kernel("PackSeqTritonKernel", _bf16_only, compiled)

    registrations = {signature_only: [((), {})], clean: [((), {})]}
    with pytest.raises(ValueError, match="fp8e4nv"):
        _run_ppu_warmup(modules, True, (8, 0), registrations)
    assert compiled == ["SignatureFp8Kernel"]
    assert signature_only in registrations


def test_warmup_propagates_other_compile_errors(modules):
    """Only the unsupported-fp8 rejection is tolerated."""
    broken = _fixture_kernel(
        "BrokenKernel", _bf16_only, [], error=RuntimeError("fused build failed")
    )

    with pytest.raises(RuntimeError, match="fused build failed"):
        _run_ppu_warmup(modules, True, (8, 0), {broken: [((), {})]})


def test_warmup_keeps_a_read_only_owner_unchanged(modules):
    compiled = []

    class FrozenOwner:
        __slots__ = ()
        kernel = staticmethod(_bf16_only)

        def compile_many(self, keys):
            compiled.append("attempted")
            raise _FP8_COMPILE_ERROR

    owner = FrozenOwner()
    with pytest.raises(ValueError, match="fp8e4nv"):
        _run_ppu_warmup(modules, True, (8, 0), {owner: [((), {})]})
    assert compiled == ["attempted"]
    with pytest.raises(ValueError, match="fp8e4nv"):
        owner.compile_many([0])


def _run_ppu_warmup(modules, ppu, arch, registrations):
    """Install the PPU patch and run the warmup over ``registrations`` in order."""
    ns = types.SimpleNamespace
    import contextlib

    def has_device_capability(capability):
        wanted = (
            (capability // 10, capability % 10)
            if isinstance(capability, int)
            else capability
        )
        return arch >= wanted

    modules(
        "vllm.platforms",
        current_platform=ns(
            is_ppu=lambda: ppu, has_device_capability=has_device_capability
        ),
    )
    modules("vllm.logger", init_logger=lambda _: ns(warning=lambda *a: None))
    modules(
        "vllm.model_executor.warmup.jit_warmup_cutedsl_helper",
        VllmCuTeDSLJitKernel=type("VllmCuTeDSLJitKernel", (), {}),
    )
    modules(
        "vllm.v1.sample.ops.topk_topp_sampler",
        register_top_k_top_p_warmups=lambda: None,
    )
    worker = ns(
        model_runner=ns(
            jit_warmup_registry=ns(
                _registrations=registrations, activate=lambda: contextlib.nullcontext()
            ),
            is_last_pp_rank=True,
            is_pooling_model=False,
        ),
        vllm_config=ns(kernel_config=ns(enable_jit_warmup=True)),
        get_model=lambda: "model",
        scheduler_config=ns(max_num_batched_tokens=64),
    )

    def upstream(actual_worker, *, process_local_only=False):
        for actual_kernel, keys in list(registrations.items()):
            actual_kernel.compile_many(keys)
        return "result"

    provider = modules(
        "vllm.model_executor.warmup.kernel_warmup",
        kernel_warmup=upstream,
        cutedsl_warmup=lambda: None,
    )
    complete_warmup_contract(
        modules, provider, worker, sys.modules["vllm.platforms"].current_platform
    )
    modules("vllm.model_executor.warmup.cutedsl_warmup", cutedsl_warmup=lambda: None)
    patched = load_patch("vllm_sail/patch/enhancement/kernel_warmup.py")
    patched._ppu_deep_gemm_warmup_enabled = lambda: False
    return provider.kernel_warmup(worker)


@pytest.mark.parametrize(
    "arch,ppu,alibi,pinned,available,expected",
    [
        (80, True, False, None, (2, 3), 3),
        (89, True, False, None, (2, 3), 3),
        (89, True, True, None, (2, 3), 2),
        (89, True, False, 2, (2, 3), 2),
        (89, True, False, None, (2,), 2),
        (89, True, False, 3, (2,), None),
        (80, False, False, None, (2, 3), 2),
        (90, False, False, None, (2, 3), 3),
    ],
)
def test_fa_selection_and_fp8_contract(
    modules, arch, ppu, alibi, pinned, available, expected
):
    platform = types.SimpleNamespace(
        is_ppu=lambda: ppu,
        is_rocm=lambda: False,
        is_xpu=lambda: False,
        supports_fp8=lambda: arch >= 89,
        get_device_capability=lambda: types.SimpleNamespace(major=arch // 10),
        is_device_capability_family=lambda family: arch // 10 == family // 10,
    )

    def old(*a, **k):
        return "old"

    provider = modules(
        "vllm.v1.attention.backends.fa_utils",
        get_flash_attn_version=old,
        flash_attn_supports_kv_cache_dtype=old,
        current_platform=platform,
        envs=types.SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        logger=types.SimpleNamespace(
            warning_once=lambda *a: None, error=lambda *a: None
        ),
    )
    consumer = modules(
        "vllm.v1.attention.backends.flash_attn",
        get_flash_attn_version=old,
        flash_attn_supports_kv_cache_dtype=old,
    )
    modules(
        "vllm.config",
        get_current_vllm_config_or_none=lambda: types.SimpleNamespace(
            attention_config=types.SimpleNamespace(flash_attn_version=pinned)
        ),
    )
    modules(
        "vllm.vllm_flash_attn.flash_attn_interface",
        is_fa_version_supported=lambda version: version in available,
        fa_version_unsupported_reason=lambda version: "missing",
    )
    load_patch("vllm_sail/patch/enhancement/attention/fa_utils.py")
    assert consumer.get_flash_attn_version is provider.get_flash_attn_version
    assert consumer.get_flash_attn_version(requires_alibi=alibi) == expected
    assert consumer.flash_attn_supports_kv_cache_dtype(requires_alibi=alibi) == (
        expected == 3 and arch >= 89
    )
    assert not consumer.flash_attn_supports_kv_cache_dtype("fp8_e5m2")


def test_indexer_fields_survive_runner_replace_and_merge(modules):
    from dataclasses import asdict, dataclass, fields, replace

    @dataclass(frozen=True)
    class AttentionSpec:
        block_size: int
        indexes_kv_by_block_stride: bool = False

    @dataclass(frozen=True)
    class MLAAttentionSpec(AttentionSpec):
        alignment: int = 576

    @dataclass(frozen=True)
    class HiddenStateCacheSpec(MLAAttentionSpec):
        pass

    HiddenStateCacheSpec.__module__ = "vllm.v1.kv_cache_interface"
    modules(
        "vllm.v1.kv_cache_interface",
        MLAAttentionSpec=MLAAttentionSpec,
        HiddenStateCacheSpec=HiddenStateCacheSpec,
    )
    load_patch("vllm_sail/patch/enhancement/attention/kv_cache_interface.py")
    spec = MLAAttentionSpec(64, indexer_n_head=64, indexer_q_head_dim=128)
    rebuilt = replace(spec, indexes_kv_by_block_stride=True)
    (merged,) = tuple(dict.fromkeys([rebuilt, rebuilt]))
    assert (merged.indexer_n_head, merged.indexer_q_head_dim) == (64, 128)
    assert merged.indexes_kv_by_block_stride
    assert len(fields(AttentionSpec)) == 2  # inherited metadata remains untouched
    assert {"indexer_n_head", "indexer_q_head_dim"} <= {f.name for f in fields(spec)}
    assert asdict(merged)["indexer_n_head"] == 64
    assert spec == replace(spec) and hash(spec) == hash(replace(spec))
    assert len({spec, replace(spec, indexer_n_head=32)}) == 2
    hidden = HiddenStateCacheSpec(64, indexer_n_head=16, indexer_q_head_dim=128)
    assert isinstance(replace(hidden), HiddenStateCacheSpec)
    assert replace(hidden).indexer_n_head == 16
    assert hidden == replace(hidden) and hash(hidden) == hash(replace(hidden))


def test_body_binding_preserves_keyword_defaults_and_live_globals(modules):
    helper = sys.modules["vllm_sail.patch.bodies"].bind_body
    namespace = types.ModuleType("target")
    namespace.value = 3
    exec(
        "def body(*, offset: int = 2) -> int:\n    return value + offset",
        namespace.__dict__,
    )
    other = types.ModuleType("other")
    other.value = 10
    bound = helper(namespace.body, other)
    assert bound() == 12 and bound(offset=1) == 11
    other.value = 20
    assert bound() == 22
    assert bound.__annotations__ == namespace.body.__annotations__


def test_deepseek_config_preserves_dense_channelwise_patterns_for_mtp(modules):
    class Fp8Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.packed_modules_mapping = {}

        def apply_vllm_mapper(self, mapper):
            pass

    class LinearBase:
        pass

    modules("vllm.config", get_current_vllm_config=lambda: None)
    modules(
        "vllm.platforms", current_platform=types.SimpleNamespace(is_ppu=lambda: True)
    )
    modules(
        "vllm.model_executor.layers.fused_moe",
        RoutedExperts=type("RoutedExperts", (), {}),
        UnquantizedFusedMoEMethod=object,
    )
    modules("vllm.model_executor.layers.linear", LinearBase=LinearBase)
    modules("vllm.model_executor.layers.quantization", QuantizationMethods=str)
    modules("vllm.model_executor.layers.quantization.fp8", Fp8Config=Fp8Config)
    modules("vllm.model_executor.layers.quantization.mxfp4", Mxfp4MoEMethod=object)
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=lambda **kwargs: any(
            p in kwargs["prefix"] for p in kwargs["ignored_layers"]
        ),
    )
    cfg_module = load_patch("vllm_sail/models/deepseek_v4/quant_config.py")
    cls = cfg_module.DeepseekV4FP8Config
    checkpoint = {
        "quant_method": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wkv"],
    }
    mtp = types.SimpleNamespace(
        model_type="deepseek_mtp", architectures=["DeepSeekV4MTPModel"]
    )
    assert cls.override_quantization_method(checkpoint, None, mtp) == "deepseek_v4_fp8"
    config = cls.from_config(checkpoint)
    assert config.is_checkpoint_fp8_serialized
    config.apply_vllm_mapper(
        types.SimpleNamespace(apply_list=lambda values: ["model.layers.0.attn.wkv"])
    )
    assert config.fp8_channelwise_layers == ["model.layers.0.attn.wkv", "attn.wkv"]
    modules(
        "compressed_tensors.quantization",
        QuantizationArgs=lambda **kwargs: kwargs,
        QuantizationStrategy=types.SimpleNamespace(CHANNEL="channel"),
        QuantizationType=types.SimpleNamespace(FLOAT="float"),
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        CompressedTensorsLinearMethod=lambda cfg: ("channel-linear", cfg),
    )
    modules(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8",
        CompressedTensorsW8A8Fp8=lambda **kwargs: kwargs,
    )
    layer = LinearBase()
    assert config.get_quant_method(layer, "model.mtp.3.attn.wkv") == (
        "channel-linear",
        config,
    )
    assert layer.scheme["weight_quant"]["strategy"] == "channel"
    assert not layer.scheme["is_static_input_scheme"]


@pytest.mark.parametrize(
    "quant_dtype,block,expected",
    [
        ("fp8", None, (True, False, False)),
        ("fp8", [128, 128], (True, False, False)),
        ("fp8", [64, 64], (False, False, False)),
        ("int8", None, (False, True, False)),
        ("int8", [128, 128], (False, False, False)),
        ("mxfp4", None, (False, False, True)),
        (None, None, (False, False, False)),
    ],
)
@pytest.mark.parametrize("explicit_manager", [False, True])
@pytest.mark.parametrize(
    "consumer_name",
    [
        "vllm.model_executor.layers.fused_moe.oracle.base",
        "vllm.model_executor.layers.quantization.modelopt_block_quant",
    ],
)
def test_deepep_factory_uses_bound_globals_and_quantized_protocol(
    modules, quant_dtype, block, expected, explicit_manager, consumer_name
):
    """Execute the installed factory, including its ordinary non-EP branch."""
    ns = types.SimpleNamespace
    platform = ns(
        ppu=True,
        is_ppu=lambda: platform.ppu,
        is_xpu=lambda: False,
        fp8_dtype=lambda: "fp8",
    )
    modules("vllm.platforms", current_platform=platform)
    manager = ns(world_size=4, get_handle=lambda args: ("handle", args))

    def original(*args, **kwargs):
        return "original"

    def get_manager():
        assert not explicit_manager, "an explicitly supplied manager must be retained"
        return manager

    provider = modules(
        "vllm.model_executor.layers.fused_moe.all2all_utils",
        current_platform=platform,
        torch=ns(int8="int8"),
        maybe_make_prepare_finalize=original,
        maybe_roundup_layer_hidden_size=lambda *args: "upstream-roundup",
        make_moe_prepare_and_finalize_no_dp_ep=lambda flag: ("no-ep", flag),
        get_ep_all2all_manager=get_manager,
        DEEPEP_QUANT_BLOCK_SHAPE=[128, 128],
    )
    consumer = modules(
        consumer_name,
        maybe_make_prepare_finalize=original,
    )
    modules(
        "vllm_sail.model_executor.layers.fused_moe.prepare_finalize.deepep_ll",
        DeepEPLLPrepareAndFinalize=lambda *args, **kwargs: ns(args=args, **kwargs),
    )
    load_patch("vllm_sail/patch/enhancement/deepep.py")
    assert consumer.maybe_make_prepare_finalize is provider.maybe_make_prepare_finalize
    moe = ns(
        moe_parallel_config=ns(
            use_all2all_kernels=False, use_passthrough_all2all=False, dp_size=1
        )
    )
    assert provider.maybe_make_prepare_finalize(moe, None) is None
    assert provider.maybe_make_prepare_finalize(
        moe, None, allow_new_interface=True
    ) == ("no-ep", False)
    moe.moe_parallel_config.use_all2all_kernels = True
    moe.use_deepep_ht_kernels = False
    moe.use_deepep_ll_kernels = True
    moe.max_num_tokens = 32
    moe.hidden_dim = 3584
    moe.num_experts = 16
    result = consumer.maybe_make_prepare_finalize(
        moe,
        ns(quant_dtype=quant_dtype, block_shape=block),
        ("g2p", "p2g", "ids"),
        all2all_manager=manager if explicit_manager else None,
    )
    assert (
        result.use_fp8_dispatch,
        result.use_int8_dispatch,
        result.use_mxfp4_dispatch,
    ) == expected
    assert result.global_to_physical == "g2p"
    assert result.args[0][1]["token_hidden_size"] == 3584
    platform.ppu = False
    assert provider.maybe_make_prepare_finalize(moe, None) == "original"


def test_pla_prefill_resolution_remains_lazy_and_gate_is_live(modules):
    ns = types.SimpleNamespace
    platform = ns(ppu=True, is_ppu=lambda: platform.ppu)
    modules("vllm.platforms", current_platform=platform)
    modules(
        "vllm.logger",
        init_logger=lambda _: ns(
            info_once=lambda *a: None, warning_once=lambda *a: None
        ),
    )
    env = modules("vllm_sail.envs", VLLM_SAIL_USE_PLA=False)
    resolver = load_patch("vllm_sail/attention/pla_prefill.py")
    assert resolver.get_sail_cuda_pla_prefill_fwd() is None
    assert not resolver._resolved

    def kernel():
        return None

    modules("pla.prefill.flashqla", chunk_gated_delta_rule_fwd=kernel)
    modules("pla.prefill.flashqla.ops", SUPPORTED_HEAD_CONFIGS=frozenset({(8, 4)}))
    env.VLLM_SAIL_USE_PLA = True
    assert resolver.get_sail_cuda_pla_prefill_fwd() is kernel
    assert resolver.get_sail_cuda_pla_prefill_head_configs() == frozenset({(8, 4)})
    env.VLLM_SAIL_USE_PLA = False
    assert resolver.get_sail_cuda_pla_prefill_fwd() is None
    assert resolver.get_sail_cuda_pla_prefill_head_configs() == frozenset()
    env.VLLM_SAIL_USE_PLA = True
    platform.ppu = False
    assert resolver.get_sail_cuda_pla_prefill_fwd() is None


@pytest.mark.parametrize(
    "ppu,arch,version,fp8,expected",
    [
        (True, (8, 0), 3, False, None),
        (True, (8, 9), 3, True, None),
        (True, (8, 0), 2, False, "sink"),
        (False, (8, 9), 3, False, "sink"),
        (True, (8, 0), 3, True, "FP8"),
    ],
)
def test_fa_sink_combination_respects_ppu_fa3_and_fp8_gates(
    modules, ppu, arch, version, fp8, expected
):
    class Backend:
        head_size_v = None

        @classmethod
        def supports_combination(cls, *args, **kwargs):
            return "original"

    modules(
        "vllm.v1.attention.backends.flash_attn",
        FlashAttentionBackend=Backend,
        current_platform=types.SimpleNamespace(is_ppu=lambda: ppu),
        DeviceCapability=lambda major, minor: (major, minor),
        get_flash_attn_version=lambda **kw: version,
        is_quantized_kv_cache=lambda dtype: dtype == "fp8",
        flash_attn_supports_kv_cache_dtype=lambda *args, **kw: arch == (8, 9),
    )
    load_patch("vllm_sail/patch/enhancement/attention/flash_attn.py")
    result = Backend.supports_combination(
        128, "bf16", "fp8" if fp8 else "auto", 16, False, True, False, False, arch
    )
    if expected is None:
        assert result is None
    else:
        assert expected in result


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "name", ["supports_combination", "flash_attn_supports_kv_cache_dtype"]
)
def test_fa_complete_copied_body_matches_upstream_outside_ppu_regions(
    upstream_source_root, name
):
    """完整 AST 比较只归一化两处已声明的 PPU 差异，不刷新 drift baseline。"""
    combination = name == "supports_combination"
    module = "flash_attn" if combination else "fa_utils"
    copied = definition(
        ROOT / f"vllm_sail/patch/enhancement/attention/{module}.py", name
    )
    original = definition(
        upstream_source_root / f"vllm/v1/attention/backends/{module}.py",
        "FlashAttentionBackend." + name if combination else name,
    )
    copied.decorator_list = original.decorator_list = []
    if combination:
        # sinks 放宽和 PPU 错误说明之外，所有旧及新增准入条件都应一致。
        for test in (
            "has_sink and device_capability < DeviceCapability(9, 0)",
            "kv_cache_dtype is not None and is_quantized_kv_cache(kv_cache_dtype) and not flash_attn_supports_kv_cache_dtype(kv_cache_dtype, head_size=head_size, head_size_v=getattr(cls, 'head_size_v', None), has_sinks=has_sink)",
        ):
            local = [
                node
                for node in copied.body
                if isinstance(node, ast.If)
                and ast.dump(node.test) == ast.dump(ast.parse(test, mode="eval").body)
            ]
            provider = [
                node
                for node in original.body
                if isinstance(node, ast.If)
                and ast.dump(node.test) == ast.dump(ast.parse(test, mode="eval").body)
            ]
            assert len(local) == len(provider) == 1
            local[0].body = provider[0].body
    else:
        ppu = [
            node
            for node in copied.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "current_platform.is_ppu()"
        ]
        assert len(ppu) == 1 and not ppu[0].orelse
        assert (
            ast.unparse(ppu[0].body[0])
            == "return fa_version == 3 and current_platform.supports_fp8()"
        )
        copied.body.remove(ppu[0])
    assert ast.dump(copied, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "arch,ppu", [(90, False), (100, False), (89, True), (80, True)]
)
@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize(
    "head_size,head_size_v", [(512, 512), (512, 128), (128, 512), (None, None)]
)
def test_real_fa_fp8_helper_preserves_q_and_v_dimensions_and_ppu_gate(
    upstream_source_root, arch, ppu, version, head_size, head_size_v
):
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return version

    namespace = {
        "current_platform": types.SimpleNamespace(
            is_ppu=lambda: ppu,
            is_xpu=lambda: False,
            supports_fp8=lambda: arch == 89,
            is_device_capability_family=lambda family: arch == family,
        ),
        "get_flash_attn_version": selected,
    }
    original = function(
        upstream_source_root / "vllm/v1/attention/backends/fa_utils.py",
        "flash_attn_supports_kv_cache_dtype",
        dict(namespace),
    )
    copied = function(
        "vllm_sail/patch/enhancement/attention/fa_utils.py",
        "flash_attn_supports_kv_cache_dtype",
        dict(namespace),
    )
    kwargs = dict(head_size=head_size, head_size_v=head_size_v)
    expected = (
        version == 3 and arch == 89
        if ppu
        else version == 3 or (head_size in (None, 512) and head_size_v in (None, 512))
        if arch == 90
        else version == 4
    )
    assert copied(**kwargs) is expected
    assert calls[-1]["head_size"] == head_size
    assert calls[-1]["head_size_v"] == head_size_v
    if not ppu:
        assert copied(**kwargs) is original(**kwargs)
    count = len(calls)
    assert not copied("fp8_e5m2", **kwargs)
    assert len(calls) == count


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "arch,q,v,kv,block,sparse,mm_prefix,expected",
    [
        ((9, 0), 512, 512, "fp8", 16, False, False, "divisible by 64"),
        ((9, 0), 512, 512, "fp8", 64, True, False, "block sparsity"),
        ((9, 0), 512, 512, "fp8", 64, False, False, None),
        ((9, 0), 512, 128, "fp8", 64, False, False, "FP8"),
        ((9, 0), 128, 128, "fp8", 16, False, True, "requires Triton"),
        ((9, 0), 512, 512, "auto", 16, True, False, None),
        ((9, 0), 128, 128, "auto", 16, False, True, None),
        ((8, 9), 128, 64, "fp8", 16, False, False, None),
        ((8, 0), 128, 64, "fp8", 16, False, False, "FP8"),
        ((8, 9), 128, 64, "fp8", 16, False, True, "mm_prefix"),
        ((10, 0), 128, 128, "fp8", 16, True, False, None),
        ((9, 0), 512, 512, "fp8", None, False, False, None),
    ],
)
def test_bound_fa_combination_keeps_sm90_limits_and_real_v_dimension(
    modules, upstream_source_root, arch, q, v, kv, block, sparse, mm_prefix, expected
):
    """真实 helper 和 bind_body 的公共调用；版本选择为边界替身，不证明 native FA 支持。"""
    calls = []
    ppu = arch[0] == 8

    def selected(**kwargs):
        calls.append(kwargs)
        return 3 if ppu else 4

    platform = types.SimpleNamespace(
        is_ppu=lambda: ppu,
        is_xpu=lambda: False,
        supports_fp8=lambda: arch == (8, 9),
        is_device_capability_family=lambda family: arch[0] * 10 + arch[1] == family,
    )
    helper = function(
        "vllm_sail/patch/enhancement/attention/fa_utils.py",
        "flash_attn_supports_kv_cache_dtype",
        {"current_platform": platform, "get_flash_attn_version": selected},
    )

    class Backend:
        head_size_v = v

        @classmethod
        def supports_combination(cls, *args, **kwargs):
            return "original"

    provider = modules(
        "vllm.v1.attention.backends.flash_attn",
        FlashAttentionBackend=Backend,
        current_platform=platform,
        DeviceCapability=lambda major, minor: (major, minor),
        get_flash_attn_version=selected,
        is_quantized_kv_cache=lambda value: value == "fp8",
        flash_attn_supports_kv_cache_dtype=helper,
    )
    load_patch("vllm_sail/patch/enhancement/attention/flash_attn.py")
    result = Backend.supports_combination(
        q, "bf16", kv, block, False, False, sparse, mm_prefix, arch
    )
    if expected is None:
        assert result is None
    else:
        assert expected in result
    early = expected in ("divisible by 64", "block sparsity", "requires Triton")
    if early:
        assert calls == []
    elif kv == "fp8":
        assert calls[0]["head_size"] == q and calls[0]["head_size_v"] == v
    assert Backend.supports_combination.__func__.__globals__ is provider.__dict__


@pytest.fixture
def lut_b_source_contract(modules, upstream_source_root):
    """真实 LUT-B 类、准入与公共构造器；存储、Triton 计算和通信是显式边界。"""
    ns = types.SimpleNamespace
    source = upstream_source_root / "vllm"
    moe = source / "model_executor/layers/fused_moe"
    modular = moe / "modular_kernel.py"
    fmt = function(modular, "FusedMoEActivationFormat", {"Enum": Enum})
    calls = []
    platform = ns(
        capability=(8, 0),
        device_name="ppu",
        is_cuda=lambda: True,
        is_cuda_alike=lambda: True,
    )
    platform.get_device_capability = lambda **kw: platform.capability
    exact = function(
        source / "platforms/interface.py", "Platform.is_device_capability", {}
    )
    platform.is_device_capability = lambda value: exact(platform, value)

    def forbidden(*args, **kwargs):
        pytest.fail("此源码边界不得启动 SM107 compute stub 或 tile 重排")

    torch = ns(ops=ns(_moe_C=ns(lut_b_moe_gemm=forbidden)))
    ops = ns(lut_b_moe_gemm=forbidden)
    ops.lut_b_moe_gemm_supported = function(
        source / "_custom_ops.py",
        "lut_b_moe_gemm_supported",
        dict(torch=torch, current_platform=platform),
    )
    mk = ns(FusedMoEActivationFormat=fmt)
    namespace = dict(
        current_platform=platform,
        torch=torch,
        ops=ops,
        mk=mk,
        FusedMoEActivationFormat=fmt,
        envs=ns(VLLM_BATCH_INVARIANT=False),
        kLutBStatic="lut_b",
        kInt8StaticTensorSym="int8",
        kMxfp8Dynamic="mxfp8",
        LUT_B_SM107_TILE_K=128,
        LUT_B_SM107_TILE_N=128,
        apply_moe_activation_supported=lambda activation: activation == "silu",
        logger=ns(
            info_once=lambda *a: calls.append(("info", a)), warning_once=lambda *a: None
        ),
    )
    mk.FusedMoEExperts = ns(
        is_supported_config=function(
            modular,
            "FusedMoEExperts.is_supported_config",
            namespace,
        )
    )

    class TritonBoundary:
        def __init__(self, moe_config, quant_config):
            self.moe_config, self.quant_config = moe_config, quant_config

        def apply(self, **kwargs):
            calls.append(("triton", kwargs))

        def activation(self, activation, output, input, **kwargs):
            calls.append(("activation", activation, output, input, kwargs))

    for name in (
        "_supports_routing_method",
        "_supports_router_logits_dtype",
        "_supports_shape",
    ):
        setattr(
            TritonBoundary,
            name,
            staticmethod(
                function(
                    modular,
                    "FusedMoEExperts." + name,
                    namespace,
                )
            ),
        )
    TritonBoundary.executed_activation_key = staticmethod(
        function(modular, "FusedMoEExperts.executed_activation_key", namespace)
    )
    namespace["check_activation_quant_fallback"] = function(
        source / "model_executor/layers/quantization/utils/quant_utils.py",
        "check_activation_quant_fallback",
        {},
    )
    TritonBoundary.is_monolithic = staticmethod(
        function(
            modular,
            "FusedMoEExpertsModular.is_monolithic",
            namespace,
        )
    )
    for name in (
        "is_supported_config",
        "activation_format",
        "_supports_no_act_and_mul",
        "_supports_activation",
        "_supports_parallel_config",
        "_supports_batch_invariance",
    ):
        setattr(
            TritonBoundary,
            name,
            staticmethod(
                function(
                    moe / "experts/triton_moe.py",
                    "TritonExperts." + name,
                    namespace,
                )
            ),
        )
    namespace["TritonExperts"] = TritonBoundary
    path = moe / "experts/lut_b_moe.py"
    function(path, "mxfp8_qdq", namespace)
    for name in ("_LutBExpertsBase", "LutBEmulationTritonExperts", "LutBSm107Experts"):
        function(path, name, namespace)

    class PrepareBoundary:
        def __init__(self, activation_format):
            self.activation_format = activation_format

        def post_init_setup(self, experts):
            calls.append(("post_init", experts))

    class ModularImplBoundary:
        def __init__(self, prepare_finalize, fused_experts):
            self.prepare_finalize, self.fused_experts = prepare_finalize, fused_experts

    kernel_namespace = dict(
        FusedMoEPrepareAndFinalizeModular=PrepareBoundary,
        FusedMoEExpertsModular=TritonBoundary,
        FusedMoEKernelModularImpl=ModularImplBoundary,
        FusedMoEPrepareAndFinalizeMonolithic=type("MonolithicPrepareBoundary", (), {}),
        FusedMoEExpertsMonolithic=type("MonolithicExpertsBoundary", (), {}),
        FusedMoEKernelMonolithicImpl=type("MonolithicImplBoundary", (), {}),
    )
    mk.FusedMoEKernel = function(modular, "FusedMoEKernel", kernel_namespace)
    prepare = ns(format=fmt.Standard)

    def make_prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return PrepareBoundary(prepare.format)

    modules(
        "vllm.model_executor.layers.fused_moe.all2all_utils",
        maybe_make_prepare_finalize=make_prepare,
    )
    namespace["make_lut_b_moe_kernel"] = function(
        path, "make_lut_b_moe_kernel", namespace
    )

    class MethodBoundary:
        def __init__(self, moe):
            self.moe = moe

    online = source / "model_executor/layers/quantization/online"
    namespace.update(FusedMoEMethodBase=MethodBoundary, abstractmethod=abstractmethod)
    function(online / "requantization.py", "OnlineRequantizationMixin", namespace)
    function(online / "moe_base.py", "OnlineMoEMethodBase", namespace)
    namespace["FUSED_MOE_UNQUANTIZED_CONFIG"] = object()
    namespace["lut_b_to_sm107_tiles"] = forbidden
    method = function(online / "lut_b.py", "LutBOnlineMoEMethod", namespace)
    config = ns(
        has_bias=False,
        moe_backend="auto",
        hidden_dim=128,
        intermediate_size_per_partition=128,
        is_act_and_mul=True,
        activation="silu",
        has_hash_routing=False,
        require_decomposed_backend=False,
        routing_method="topk",
        router_logits_dtype=None,
        is_lora_enabled=False,
        num_experts=2,
        w13_num_shards=2,
        moe_parallel_config=ns(
            use_fi_nvl_two_sided_kernels=False, use_fi_nvl_one_sided_kernels=False
        ),
    )
    return ns(
        platform=platform,
        torch=torch,
        ops=ops,
        mk=mk,
        format=fmt,
        namespace=namespace,
        calls=calls,
        config=config,
        method=method,
        native=namespace["LutBSm107Experts"],
        emulation=namespace["LutBEmulationTritonExperts"],
        prepare=prepare,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("capability", [(8, 0), (8, 9)])
@pytest.mark.parametrize("backend", ["auto", "emulation"])
def test_lut_b_real_public_constructor_keeps_ppu_emulation_despite_schema(
    lut_b_source_contract, capability, backend
):
    f = lut_b_source_contract
    f.platform.capability = capability
    f.config.moe_backend = backend
    assert f.ops.lut_b_moe_gemm_supported() is True
    assert f.native._supports_current_device() is False
    method = f.method(moe=f.config)
    assert method.experts_cls is f.emulation
    assert (
        method.get_fused_moe_quant_config(None)
        is f.namespace["FUSED_MOE_UNQUANTIZED_CONFIG"]
    )
    assert f.emulation.activation_format() is f.format.PaddedStandard
    assert f.native.activation_format() is f.format.PaddedStandard


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "case,expected",
    [
        ("native", "native"),
        ("no_schema", "emulation"),
        ("wrong_sm", "emulation"),
        ("hidden_shape", "emulation"),
        ("intermediate_shape", "emulation"),
        ("explicit_emulation", "emulation"),
        ("lora", "emulation"),
        ("too_many_experts", "emulation"),
    ],
)
def test_lut_b_real_sm107_controls_preserve_all_public_selection_gates(
    lut_b_source_contract, case, expected
):
    f = lut_b_source_contract
    f.platform.capability = (10, 7)
    if case == "no_schema":
        del f.torch.ops._moe_C.lut_b_moe_gemm
    elif case == "wrong_sm":
        f.platform.capability = (10, 0)
    elif case == "hidden_shape":
        f.config.hidden_dim = 192
    elif case == "intermediate_shape":
        f.config.intermediate_size_per_partition = 192
    elif case == "explicit_emulation":
        f.config.moe_backend = "emulation"
    elif case == "lora":
        f.config.is_lora_enabled = True
    elif case == "too_many_experts":
        f.config.num_experts = 1024
    assert f.method(moe=f.config).experts_cls is getattr(f, expected)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "bias,backend,message",
    [
        (True, "auto", "does not support bias"),
        (False, "triton", "expected 'auto' or 'emulation'"),
    ],
)
def test_lut_b_real_public_constructor_rejects_unsupported_requests(
    lut_b_source_contract, bias, backend, message
):
    f = lut_b_source_contract
    f.config.has_bias, f.config.moe_backend = bias, backend
    with pytest.raises(ValueError, match=message):
        f.method(moe=f.config)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "prepare_format", ["Standard", "PaddedStandard", "BatchedExperts"]
)
def test_lut_b_real_kernel_factory_preserves_directional_activation_format(
    lut_b_source_contract, prepare_format
):
    f = lut_b_source_contract
    f.prepare.format = getattr(f.format, prepare_format)
    config = f.method(moe=f.config).get_fused_moe_quant_config(None)
    routing = (object(), object(), object())
    make = f.namespace["make_lut_b_moe_kernel"]
    if prepare_format == "BatchedExperts":
        with pytest.raises(AssertionError):
            make(f.emulation, f.config, config, routing)
    else:
        kernel = make(f.emulation, f.config, config, routing)
        assert isinstance(kernel, f.mk.FusedMoEKernel)
        assert isinstance(kernel.fused_experts, f.emulation)
        assert kernel.prepare_finalize.activation_format is f.prepare.format
        assert kernel.moe_config is f.config and not kernel.supports_lora()
        assert kernel.fused_experts.expects_unquantized_inputs
    assert f.calls[0] == (
        "info",
        (
            "LUT-B MoE falls back to emulation: %s",
            "kernel does not support current device ppu",
        ),
    )
    assert f.calls[1] == (
        "prepare",
        dict(
            moe=f.config,
            quant_config=config,
            routing_tables=routing,
            allow_new_interface=True,
            use_monolithic=False,
        ),
    )
    assert f.calls[2][0] == "post_init" and isinstance(f.calls[2][1], f.emulation)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "fail_at", [None, 0, 1], ids=["success", "first_weight", "second_weight"]
)
def test_lut_b_real_post_load_keeps_weight_release_and_idempotency_contract(
    lut_b_source_contract, fail_at
):
    """真实 get/zero_padding/release 链；量化、参数替换及 tensor 存储是记录边界。"""
    f = lut_b_source_contract
    ns = types.SimpleNamespace
    method = f.method(moe=f.config)
    f.calls.clear()
    weights = (
        ns(dtype="bf16", shape=(2, 256, 128)),
        ns(dtype="bf16", shape=(2, 128, 128)),
    )
    names = ("w13_weight", "w2_weight")
    routing = (object(), object(), object())
    layer = ns(
        w13_weight=weights[0],
        w2_weight=weights[1],
        moe_config=ns(
            hidden_dim_unpadded=128, intermediate_size_per_partition_unpadded=128
        ),
        _parameters=dict(zip(names, weights, strict=False)),
        _expert_routing_tables=lambda: routing,
    )
    method.requantization_source_parameters = dict(layer._parameters)
    converted = []
    error = RuntimeError("LUT-B conversion boundary failed")

    def quantize(weight):
        assert weight is weights[len(converted)]
        f.calls.append(("quantize", weight))
        return weight, object(), object()

    def dequantize(packed, codebooks, scales, dtype):
        assert packed is weights[len(converted)] and dtype == "bf16"
        if len(converted) == fail_at:
            raise error
        result = object()
        converted.append(result)
        return result

    def replace(actual_layer, name, value):
        assert actual_layer is layer and value is converted[-1]
        f.calls.append(("replace", name))
        setattr(layer, name, value)
        layer._parameters[name] = value

    f.namespace.update(
        quantize_lut_b=quantize, dequantize_lut_b=dequantize, replace_parameter=replace
    )
    if fail_at is not None:
        with pytest.raises(RuntimeError) as raised:
            method.process_weights_after_loading(layer)
        assert raised.value is error
        assert not hasattr(layer, "_already_called_process_weights_after_loading")
        assert not hasattr(method, "moe_kernel")
        # 上游逐笔替换；第二笔失败时保留第一笔结果，不伪造事务回滚。
        for index, name in enumerate(names):
            expected = converted[index] if index < fail_at else weights[index]
            assert layer._parameters[name] is expected
            assert getattr(layer, name) is expected
        assert all(
            method.requantization_source_parameters[name] is weight
            for name, weight in zip(names, weights, strict=False)
        )
        assert [call[0] for call in f.calls] == ["quantize", "replace"] * fail_at + [
            "quantize"
        ]
    else:
        method.process_weights_after_loading(layer)
        assert [call[0] for call in f.calls] == [
            "quantize",
            "replace",
            "quantize",
            "replace",
            "prepare",
            "post_init",
        ]
        assert all(
            layer._parameters[name] is value
            for name, value in zip(names, converted, strict=False)
        )
        assert method.requantization_source_parameters == {}
        assert isinstance(method.moe_kernel.fused_experts, f.emulation)
        assert layer._already_called_process_weights_after_loading
        before = list(f.calls)
        method.process_weights_after_loading(layer)
        assert f.calls == before


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "stored,fail_at",
    [
        ("unquantized", None),
        ("unquantized", "shared_shape"),
        ("mxfp8", None),
        ("mxfp8", "first_shared"),
        ("mxfp8", "second_shared"),
        ("mxfp8", "shared_shape"),
    ],
)
def test_lut_b_real_requantization_consumes_shared_staging_in_source_order(
    lut_b_source_contract, modules, upstream_source_root, stored, fail_at
):
    """真实 chunk/shared loader 生命周期；tensor 数值和量化 provider 为显式边界。"""
    f = lut_b_source_contract
    ns = types.SimpleNamespace
    events, writes = [], []
    error = RuntimeError("shared expert dequantization boundary failed")

    class TensorBoundary:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

        def to(self, dtype):
            assert dtype == "bf16"
            events.append(("cast", self.name))
            return self

        def new_empty(self, shape):
            assert shape == (3, *self.shape[1:])
            return TensorBoundary("output_" + self.name, shape)

        def __setitem__(self, experts, tensor):
            writes.append((self.name, experts.start, experts.stop, tensor))

    f.torch.Tensor = TensorBoundary
    routed = (
        TensorBoundary("routed_w13", (2, 256, 128)),
        TensorBoundary("routed_w2", (2, 128, 128)),
    )
    shared = (
        TensorBoundary("shared_w13", (1, 256, 128)),
        TensorBoundary("shared_w2", (1, 128, 128)),
    )
    staged = tuple(ns(data=tensor) for tensor in shared)
    scales = (ns(data=object()), ns(data=object()))
    names = ("online_w13_weight", "online_w2_weight")
    parameters = dict(zip(names, staged, strict=True))
    if stored == "mxfp8":
        parameters.update(
            online_w13_weight_scale=scales[0], online_w2_weight_scale=scales[1]
        )

    class ParameterModule(ns):
        # nn.Module.__delattr__ removes registered parameters as well as attributes.
        def __delattr__(self, name):
            super().__delattr__(name)
            self._parameters.pop(name, None)

    layer = ParameterModule(
        **parameters,
        _parameters=dict(parameters),
        local_num_experts=3,
        orig_dtype="bf16",
        expert_map_manager=ns(num_fused_shared_experts=1),
        moe_config=ns(
            hidden_dim_unpadded=128,
            intermediate_size_per_partition_unpadded=128,
            hidden_dim=128,
            intermediate_size_per_partition=128,
        ),
    )
    method = f.method(moe=f.config)
    source_method = ns()

    def source_chunks(actual_layer):
        assert actual_layer is layer
        events.append(("routed", source_method))
        yield slice(0, 2), *routed

    source_method.dequantize_weight = source_chunks
    method.set_requantization_source(source_method)
    method.requantization_source_parameters = {"source_weight": object()}
    source_parameters = dict(method.requantization_source_parameters)
    checkpoint_key = None if stored == "unquantized" else "mxfp8_static"

    def dequantize(weight, scale):
        index = 0 if weight is shared[0] else 1
        assert weight is shared[index] and scale is scales[index].data
        events.append(("dequantize", index))
        if fail_at == ("first_shared" if index == 0 else "second_shared"):
            raise error
        return weight

    modules(
        "vllm.model_executor.layers.quantization.online.base",
        ONLINE_SHARED_EXPERT_DEQUANTIZERS={"mxfp8_static": dequantize},
    )
    f.namespace.update(
        LinearBase=type("LinearBoundary", (), {}),
        get_checkpoint_quant_keys=lambda *args: (checkpoint_key, None),
        kMxfp8Static="mxfp8_static",
    )
    loader_class = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/online/shared_expert.py",
        "OnlineSharedExpertLoader",
        f.namespace,
    )
    loader = loader_class(
        quant_config=object(),
        online_config=ns(
            resolve_quant_method_cls=lambda *args: (
                None,
                None,
                None,
                ns(weight="lut_b"),
            )
        ),
        quant_method=method,
        shared_expert_prefix="model.shared_experts",
        num_logical_experts=2,
        num_fused_shared_experts=1,
    )
    loader.staging_parameters = dict(parameters)
    layer.shared_expert_online_loader = loader
    assert loader.routed_requantized and loader.stages_shared_experts
    assert loader.quantizer is None and method.uses_meta_device is True
    if fail_at == "shared_shape":
        shared[1].shape = (1, 128, 64)

    should_fail = fail_at == "shared_shape" or (
        stored == "mxfp8" and fail_at in ("first_shared", "second_shared")
    )
    if should_fail:
        expected = ValueError if fail_at == "shared_shape" else RuntimeError
        with pytest.raises(expected) as raised:
            method.get_weights_for_quantization(layer)
        if expected is RuntimeError:
            assert raised.value is error
        else:
            assert "unpadded source w2" in str(raised.value)
    else:
        output = method.get_weights_for_quantization(layer)
        assert tuple(tensor.shape for tensor in output) == (
            (3, 256, 128),
            (3, 128, 128),
        )
    expected_events = [("routed", source_method)]
    if stored == "mxfp8":
        expected_events.append(("dequantize", 0))
        if fail_at != "first_shared":
            expected_events.append(("dequantize", 1))
    if not should_fail or fail_at == "shared_shape":
        expected_events.extend(("cast", tensor.name) for tensor in shared)
    assert events == expected_events
    assert writes[:2] == [
        ("output_routed_w13", 0, 2, routed[0]),
        ("output_routed_w2", 0, 2, routed[1]),
    ]
    assert len(writes) == (2 if should_fail else 4)
    if not should_fail:
        assert writes[2:] == [
            ("output_routed_w13", 2, 3, shared[0]),
            ("output_routed_w2", 2, 3, shared[1]),
        ]
    released = not should_fail or fail_at == "shared_shape"
    assert bool(loader.staging_parameters) is not released
    for name, parameter in parameters.items():
        if released:
            assert name not in layer._parameters and not hasattr(layer, name)
        else:
            assert (
                layer._parameters[name] is parameter
                and getattr(layer, name) is parameter
            )
    if released:
        # 上游先释放 shared staging 再校验 padding；失败不能伪造回滚或允许重复加载。
        with pytest.raises(NotImplementedError, match="Reloading fused shared experts"):
            loader.load(layer, object(), object(), "w2_weight", "w2", 2, 2)
    assert method.requantization_source_parameters == source_parameters


@pytest.mark.upstream_source
@pytest.mark.parametrize("entry", ["apply", "activation"])
@pytest.mark.parametrize("fail_at", [None, "quantize", "dequantize", "cast"])
def test_lut_b_real_emulation_qdq_preserves_dtype_routing_and_failure_identity(
    lut_b_source_contract, entry, fail_at
):
    """执行真实 QDQ 与完整 emulation 方法；量化、存储和 Triton 计算为记录边界。"""
    f = lut_b_source_contract
    ns = types.SimpleNamespace
    experts = f.emulation(f.config, f.namespace["FUSED_MOE_UNQUANTIZED_CONFIG"])
    f.calls.clear()
    error = RuntimeError("LUT-B QDQ boundary failed")
    restored, codes, shaped_scales = object(), object(), object()

    def copy(value):
        assert value is restored
        f.calls.append(("copy", value))

    output = ns(shape=(2, 3, 128), dtype="fp16", copy_=copy)
    hidden = ns(shape=(2, 128), dtype="bf16")
    value = output if entry == "activation" else hidden

    def quantize(actual):
        assert actual is value
        f.calls.append(("quantize", actual))
        if fail_at == "quantize":
            raise error
        return codes, ns(view=view)

    def view(*shape):
        assert shape == (*value.shape[:-1], -1)
        f.calls.append(("view", shape))
        return shaped_scales

    def dequantize(actual_codes, actual_scales):
        assert actual_codes is codes and actual_scales is shaped_scales
        f.calls.append(("dequantize", actual_codes, actual_scales))
        if fail_at == "dequantize":
            raise error
        return ns(to=cast)

    def cast(dtype):
        assert dtype == value.dtype
        f.calls.append(("cast", dtype))
        if fail_at == "cast":
            raise error
        return restored

    f.namespace.update(mxfp8_e4m3_quantize=quantize, dequant_mxfp8_to_bf16=dequantize)
    if entry == "activation":
        input_tensor, options = object(), dict(boundary_option=object())
        arguments = dict(
            activation="silu", output=output, input=input_tensor, **options
        )
    else:
        arguments = {
            name: object()
            for name in (
                "w1",
                "w2",
                "topk_weights",
                "topk_ids",
                "expert_map",
                "a1q_scale",
                "a2_scale",
                "workspace13",
                "workspace2",
                "expert_tokens_meta",
            )
        }
        arguments.update(
            output=output,
            hidden_states=hidden,
            activation="silu",
            global_num_experts=2,
            apply_router_weight_on_input=True,
        )
    if fail_at is not None:
        with pytest.raises(RuntimeError) as raised:
            getattr(experts, entry)(**arguments)
        assert raised.value is error
    else:
        assert getattr(experts, entry)(**arguments) is None

    expected_events = ["quantize", "view", "dequantize", "cast"]
    if fail_at is not None:
        expected_events = expected_events[: expected_events.index(fail_at) + 1]
    else:
        expected_events.append("copy" if entry == "activation" else "triton")
    if entry == "activation":
        assert f.calls[0] == ("activation", "silu", output, input_tensor, options)
        expected_events.insert(0, "activation")
    elif fail_at is None:
        forwarded = f.calls[-1][1]
        assert forwarded.keys() == arguments.keys()
        for name, original in arguments.items():
            expected = (
                restored
                if name == "hidden_states"
                else None
                if name in ("a1q_scale", "a2_scale")
                else original
            )
            assert forwarded[name] is expected
    assert [call[0] for call in f.calls] == expected_events
    assert experts.apply.__func__.__globals__ is f.namespace
    assert experts.activation.__func__.__globals__ is f.namespace
    assert f.namespace["mxfp8_qdq"].__globals__ is f.namespace


@pytest.mark.upstream_source
@pytest.mark.parametrize("capability", [80, 89])
@pytest.mark.parametrize("rows", [1, 512, 1024])
def test_qwen_hc_real_admission_avoids_compute_stub_and_keeps_linear_fallback(
    upstream_source_root,
    capability,
    rows,
):
    ns = types.SimpleNamespace
    source = upstream_source_root / "vllm"
    family = function(
        source / "platforms/interface.py", "Platform.is_device_capability_family", {}
    )
    platform = ns(get_device_capability=lambda **kw: ns(to_int=lambda: capability))
    platform.is_device_capability_family = lambda value: family(platform, value)
    platform.is_device_capability = lambda value: capability == value

    def forbidden(*args, **kwargs):
        pytest.fail("PPU 80/89 不得调用 SM100 compute stub")

    torch_boundary = ns(
        bfloat16="bf16", ops=ns(_C=ns(qwen4_exp_hc_up_gate_mix=forbidden))
    )
    supported = function(
        source / "models/qwen4_exp/nvidia/ops/hc.py",
        "hc_up_gate_mix_supported",
        {"torch": torch_boundary, "current_platform": platform},
    )
    weight = ns(dtype="bf16", shape=(10240, 320))
    assert not supported(weight, 4)
    platform.get_device_capability = lambda **kw: ns(to_int=lambda: 100)
    assert supported(weight, 4)
    platform.get_device_capability = lambda **kw: ns(to_int=lambda: capability)
    lowrank, gate, result = object(), object(), object()
    xn = ns(shape=(rows, 10240))
    calls = []

    class Linear:
        def __init__(self):
            self.weight = weight

        def __call__(self, value):
            assert value is lowrank
            calls.append("linear")
            return gate

    def mix(value, actual_gate, count):
        assert value is xn and actual_gate is gate and count == 4
        calls.append("gate_mix")
        return result

    owner = ns(
        _use_hc_up_gate_mix=supported(weight, 4),
        input_mix_weight_up=Linear(),
        hc_count=4,
    )
    up = function(
        source / "models/qwen4_exp/nvidia/hyperconnection.py",
        "GatedResidual._up_gate_mix",
        {
            "MIN_FUSED_UP_M": 512,
            "ReplicatedLinear": Linear,
            "hc_up_gate_mix": forbidden,
            "hc_gate_mix": mix,
        },
    )
    assert up(owner, lowrank, xn) is result
    assert calls == ["linear", "gate_mix"]


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "capability,sleep,quantized,ubatching",
    [
        (80, False, False, False),
        (89, True, False, True),
        (80, True, True, True),
        (89, False, True, False),
        (100, True, False, True),
        (100, False, False, True),
    ],
)
def test_kimi_real_constructor_resolves_sleep_and_ppu_flags_before_tail_admission(
    upstream_source_root,
    capability,
    sleep,
    quantized,
    ubatching,
):
    ns = types.SimpleNamespace
    warnings, events, calls = [], [], []

    class Unquantized:
        pass

    transform = ns(up_proj=ns(quant_method=object() if quantized else Unquantized()))
    config = ns(
        model_config=ns(enable_sleep_mode=sleep),
        parallel_config=ns(use_ubatching=ubatching),
    )
    marker = object()

    class Base:
        def __init__(self, supplied, *, routed_output_transform):
            assert supplied is marker and routed_output_transform is transform
            calls.append("base")
            self.routed_output_transform = transform
            self.moe_config = ns(tp_size=8, is_sequence_parallel=False)
            self._shared_experts = object()
            self._fused_output_is_reduced = False

    def event():
        value = object()
        events.append(value)
        return value

    platform = ns(is_cuda=lambda: True, is_ppu=lambda: capability in (80, 89))
    family = function(
        upstream_source_root / "vllm/platforms/interface.py",
        "Platform.is_device_capability_family",
        {},
    )
    platform.get_device_capability = lambda **kw: ns(to_int=lambda: capability)
    platform.is_device_capability_family = lambda value: family(platform, value)
    platform.is_device_capability = lambda value: capability == value
    cls = function(
        upstream_source_root / "vllm/models/kimi_k3/nvidia/latent_moe_runner.py",
        "LatentMoERunner",
        {
            "MoERunner": Base,
            "torch": ns(cuda=ns(Event=event)),
            "current_platform": platform,
            "UnquantizedLinearMethod": Unquantized,
            "get_current_vllm_config": lambda: config,
            "get_symmetric_memory_topology": lambda group: "supported",
            "get_tp_group": lambda: ns(cpu_group=object()),
            "SymmetricMemoryTopology": ns(UNSUPPORTED="unsupported"),
            "logger": ns(warning_once=lambda *args: warnings.append(args)),
        },
    )
    initialize = function(
        "vllm_sail/patch/enhancement/models/kimi_k3.py",
        "runner_init",
        {
            "_upstream_runner_init": cls.__init__,
            "current_platform": platform,
            "UnquantizedLinearMethod": Unquantized,
        },
    )
    owner = object.__new__(cls)
    if capability == 100 and ubatching:
        with pytest.raises(ValueError, match="DBO or ubatching"):
            initialize(owner, marker, routed_output_transform=transform)
    else:
        initialize(owner, marker, routed_output_transform=transform)
        assert owner.enable_k3_latent_moe_tail_fusion is False
        if platform.is_ppu():
            assert owner._up_proj_is_quantized is quantized
            assert owner._use_fused_path() is (not quantized)
        else:
            assert not hasattr(owner, "_up_proj_is_quantized")
    assert calls == ["base"] and owner._shared_ar_events == tuple(events)
    assert len(events) == 2
    assert warnings == []


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("ppu", [False, True])
def test_glm_mega_admission_is_consistent_for_preloaded_model_classes(
    modules,
    upstream_source_root,
    preloaded,
    ppu,
):
    ns = types.SimpleNamespace
    platform = ns(is_ppu=lambda: ppu)
    modules("vllm.platforms", current_platform=platform)
    modules("torch")
    modules("vllm.logger", init_logger=lambda name: ns(warning=lambda *args: None))
    modules(
        "vllm.models.glm5next.nvidia.ops.kpool_compress", fwht128_quant_fp8=lambda q: q
    )
    original = function(
        upstream_source_root / "vllm/models/glm5next/common/model.py",
        "Glm5NextMoE._init_mega_moe_experts",
        {},
    )
    modules(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe",
        MegaMoEActivation=object,
    )
    modules("vllm.models.glm5next.nvidia.mega_moe", Glm5NextMegaMoEExperts=object)
    cls = type("Glm5NextMoE", (), {"_init_mega_moe_experts": original})
    model = modules("vllm.models.glm5next.common.model", Glm5NextMoE=cls)
    captured = model.Glm5NextMoE if preloaded else None
    patch_module = load_patch("vllm_sail/patch/enhancement/models/glm5_next.py")
    captured = captured if preloaded else model.Glm5NextMoE
    owner = captured()
    config = ns(parallel_config=ns(enable_expert_parallel=False))
    expected = "SAIL MegaMoE only supports MXFP4" if ppu else "requires expert parallel"
    with pytest.raises(NotImplementedError, match=expected):
        owner._init_mega_moe_experts(config, object(), object(), "model.layers.45.mlp")
    assert patch_module._upstream_init_mega_moe_experts is original
