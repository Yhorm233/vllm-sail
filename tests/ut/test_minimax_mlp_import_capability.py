# SPDX-License-Identifier: Apache-2.0
"""Run the whole upstream model's MLP across the NVIDIA import boundary."""

import ast
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import ROOT, definition, function


def load_model(modules, monkeypatch, source, ppu):
    """Configure only external imports; execute the complete real model module."""
    path = source / "vllm/models/minimax_m3/nvidia/model.py"

    def unavailable(self, *args, **kwargs):
        raise AssertionError("Unconfigured external dependency was invoked")

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith(
            ("vllm", "transformers")
        ):
            attrs = {
                alias.name: type(alias.name, (), {"__init__": unavailable})
                for alias in node.names
            }
            modules(node.module, **attrs)
    modules("vllm", _custom_ops=modules("vllm._custom_ops"))
    modules("vllm.envs")

    class Module:
        def __init__(self):
            self.initialized = True

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    tensor = type("Tensor", (), {})
    nn = modules("torch.nn", Module=Module, Parameter=tensor, ModuleList=list)
    modules(
        "torch",
        nn=nn,
        Tensor=tensor,
        dtype=type("dtype", (), {}),
        device=type("device", (), {}),
    )
    modules("vllm.logger", init_logger=lambda name: NS())
    modules(
        "vllm.multimodal",
        MULTIMODAL_REGISTRY=NS(
            register_processor=lambda processor, *, info, dummy_inputs: lambda cls: cls
        ),
    )
    modules(
        "vllm.model_executor.models.utils",
        WeightsMapper=lambda *, orig_to_new_prefix, orig_to_new_substr: NS(),
    )
    modules(
        "vllm.compilation.breakable_cudagraph",
        eager_break_during_capture=lambda fn: fn,
    )
    modules(
        "vllm.platforms", current_platform=NS(is_ppu=lambda: ppu, is_cuda=lambda: True)
    )
    name = "vllm.models.minimax_m3.nvidia.model"
    modules(name)
    spec = importlib.util.spec_from_file_location(name, path)
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    monkeypatch.setattr(sys.modules[name.rpartition(".")[0]], "model", provider)
    spec.loader.exec_module(provider)
    return provider


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("threshold", [16, 64])
@pytest.mark.parametrize("autotune", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("run_gemm_rs", [False, True])
def test_registered_mlp_keeps_constructor_and_forward_contract(
    modules,
    monkeypatch,
    upstream_source_root,
    ppu,
    fuse,
    threshold,
    autotune,
    sequence_parallel,
    run_gemm_rs,
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    calls = []
    quant = object()
    gate_up, output, reduced = object(), object(), object()
    activated = NS(shape=(64, 16))

    def all_gather(x):
        calls.append(("all-gather", x))
        return x

    def reduce_scatter(x):
        assert x is output
        calls.append(("reduce-scatter", x))
        return reduced

    provider.sp_all_gather = all_gather
    provider.sp_reduce_scatter = reduce_scatter
    gemm_rs = NS(can_run=lambda linear: True)
    modules(
        "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar",
        get_gemm_rs_ar=lambda: gemm_rs,
    )

    class Linear:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            calls.append(("linear", self))

        def __call__(self, x):
            calls.append(("linear-forward", self, x))
            return (
                gate_up
                if len(self.args) == 2 and isinstance(self.args[1], list)
                else output
            ), None

    class Activation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, x):
            assert x is gate_up
            calls.append(("act", x))
            return activated

    provider.MergedColumnParallelLinear = provider.RowParallelLinear = Linear
    provider.SiluAndMulWithClamp = Activation
    provider.envs = NS(VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=threshold)
    provider.is_flashinfer_autotuning = lambda: autotune
    config = NS(
        hidden_size=8,
        hidden_act="swigluoai",
        swiglu_limit=7,
        swiglu_alpha=1.702,
        swiglu_beta=1,
    )
    name = "vllm.models.minimax_m3.nvidia.ops"

    class Forbidden(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name or fullname.startswith(name + "."):
                raise ModuleNotFoundError("NVIDIA-only dense import", name="quack")
            return None

    def capable(*args):
        calls.append(("capability", args))
        return fuse

    def fused(x, linear, act):
        calls.append(("fused-forward", x, linear, act))
        return activated

    if ppu:
        monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
        with pytest.raises(ModuleNotFoundError, match="NVIDIA-only"):
            provider.MiniMaxM3MLP(config, 16)
        calls.clear()
    else:
        dense = modules(
            name + ".dense_gemm1_mxfp8",
            can_fuse_gemm1=capable,
            MIN_TOKENS=32,
            fused_gemm1=fused,
        )
        modules(name, dense_gemm1_mxfp8=dense)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    disable = threshold == 64
    mlp = provider.MiniMaxM3MLP(
        config,
        16,
        quant,
        not disable,
        disable,
        use_sequence_parallel=sequence_parallel,
        run_gemm_rs=run_gemm_rs,
        prefix="layers.0.mlp",
    )
    assert mlp.use_sequence_parallel is sequence_parallel
    assert mlp.gemm_rs is (
        gemm_rs if sequence_parallel and run_gemm_rs and not ppu else None
    )
    assert mlp.initialized
    assert mlp.gate_up_proj.args == (8, [16, 16])
    assert mlp.gate_up_proj.kwargs == dict(
        bias=False,
        quant_config=quant,
        disable_tp=disable,
        prefix="layers.0.mlp.gate_up_proj",
    )
    assert mlp.down_proj.args == (16, 8)
    assert mlp.down_proj.kwargs == dict(
        bias=False,
        quant_config=quant,
        reduce_results=not disable and not sequence_parallel,
        disable_tp=disable,
        prefix="layers.0.mlp.down_proj",
    )
    assert mlp.act_fn.kwargs == dict(swiglu_limit=7, alpha=1.702, beta=1)
    assert mlp.fused_gemm1_min_tokens == (
        None if ppu or not fuse else max(32, threshold)
    )
    assert len([c for c in calls if c[0] == "capability"]) == (0 if ppu else 1)
    calls.clear()
    x = NS(shape=(64, 8))
    assert mlp(x) is (reduced if sequence_parallel else output)
    expected = []
    if sequence_parallel:
        expected.append(("all-gather", x))
    if not ppu and fuse and threshold < 64 and not autotune:
        expected.extend(
            [
                ("fused-forward", x, mlp.gate_up_proj, mlp.act_fn),
                ("linear-forward", mlp.down_proj, activated),
            ]
        )
    else:
        expected.extend(
            [
                ("linear-forward", mlp.gate_up_proj, x),
                ("act", gate_up),
                ("linear-forward", mlp.down_proj, activated),
            ]
        )
    if sequence_parallel:
        expected.append(("reduce-scatter", output))
    assert calls == expected


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
def test_registered_mlp_retains_activation_validation(
    modules, monkeypatch, upstream_source_root, ppu
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    provider.MergedColumnParallelLinear = provider.RowParallelLinear = lambda *a, **k: (
        object()
    )
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    with pytest.raises(ValueError, match="Only swigluoai is supported"):
        provider.MiniMaxM3MLP(NS(hidden_size=8, hidden_act="unsupported"), 16)


@pytest.mark.upstream_source
def test_complete_mlp_constructor_preserves_upstream_body(upstream_source_root):
    local = definition(ROOT / "vllm_sail/models/minimax_m3_mlp.py", "init_mlp")
    upstream = definition(
        upstream_source_root / "vllm/models/minimax_m3/nvidia/model.py",
        "MiniMaxM3MLP.__init__",
    )
    local.name = upstream.name
    # 显式 super 是独立绑定 body 的要求，其余只还原两个 PPU 导入门控。
    assert ast.unparse(local.body[0]) == "super(MiniMaxM3MLP, self).__init__()"
    local.body[0] = ast.parse("super().__init__()").body[0]
    regions = [node for node in local.body if isinstance(node, ast.If)]
    assert len(regions) == 3
    import_guard, capability_guard = regions[1:]
    assert ast.unparse(import_guard.test) == "not current_platform.is_ppu()"
    assert len(import_guard.body) == 1 and not import_guard.orelse
    local.body[local.body.index(import_guard)] = import_guard.body[0]
    assert isinstance(capability_guard.test, ast.BoolOp)
    assert isinstance(capability_guard.test.op, ast.And)
    assert len(capability_guard.test.values) == 2
    assert (
        ast.unparse(capability_guard.test.values[0]) == "not current_platform.is_ppu()"
    )
    capability_guard.test = capability_guard.test.values[1]
    assert ast.dump(local) == ast.dump(upstream)


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tokens", [512, 2048])
def test_gemm_rs_initialization_keeps_collective_fallback(
    modules, monkeypatch, upstream_source_root, ppu, sequence_parallel, enabled, tokens
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    calls = []
    provider.envs = NS(VLLM_ENABLE_GEMM_RS=enabled)
    provider.logger = NS(info_once=lambda *args: None)
    config = NS(
        scheduler_config=NS(max_num_batched_tokens=tokens),
        model_config=NS(hf_text_config=NS(hidden_size=3584)),
    )

    def initialize(config, **kwargs):
        calls.append(kwargs)
        return True

    gemm_module = "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar"
    if ppu:
        modules(gemm_module.rpartition(".")[0])

        class Forbidden(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == gemm_module:
                    raise ModuleNotFoundError("NVIDIA-only GEMM-RS", name="cutlass")
                return None

        monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
        if sequence_parallel and enabled and tokens >= 1024:
            with pytest.raises(ModuleNotFoundError, match="NVIDIA-only GEMM-RS"):
                provider._maybe_init_gemm_rs(config, sequence_parallel)
    else:
        modules(gemm_module, maybe_init_gemm_rs_ar=initialize)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    eligible = not ppu and sequence_parallel and enabled and tokens >= 1024
    assert provider._maybe_init_gemm_rs(config, sequence_parallel) is eligible
    assert calls == ([dict(N=3584, all_reduce=False)] if eligible else [])


@pytest.mark.upstream_source
@pytest.mark.parametrize("residual_present", [False, True])
@pytest.mark.parametrize("dimensions", [2, 3])
def test_ppu_norm_retains_unquantized_identity_without_nvidia_import(
    modules, monkeypatch, upstream_source_root, residual_present, dimensions
):
    provider = load_model(modules, monkeypatch, upstream_source_root, True)
    calls = []
    x = NS(dim=lambda: dimensions)
    residual = NS(updated=False) if residual_present else None
    output, weight = object(), object()
    instance = object.__new__(provider.MiniMAXGemmaRMSNorm)
    instance.weight, instance.variance_epsilon = weight, 1e-6

    def plain(x_arg, w_arg, eps):
        assert (x_arg, w_arg, eps) == (x, weight, 1e-6)
        calls.append("plain")
        return output

    def fused(x_arg, residual_arg, w_arg, eps):
        assert x_arg is x and residual_arg is residual
        assert w_arg is weight and eps == 1e-6
        residual_arg.updated = True
        calls.append("fused")

    modules("flashinfer.norm", gemma_rmsnorm=plain, gemma_fused_add_rmsnorm=fused)

    class Forbidden(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("vllm.models.minimax_m3.nvidia.ops"):
                raise ModuleNotFoundError("NVIDIA-only norm import", name="quack")
            return None

    monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
    if dimensions == 2:
        with pytest.raises(ModuleNotFoundError, match="NVIDIA-only norm"):
            instance.forward(x, residual)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    result = instance.forward(x, residual)
    if residual_present:
        assert result[0] is x and result[1] is residual and residual.updated
    else:
        assert result is output
    assert calls == (["fused"] if residual_present else ["plain"])
    with pytest.raises(ValueError, match="128x4 MXFP8"):
        instance.forward(x, residual, mxfp8=True)
    assert calls == (["fused"] if residual_present else ["plain"])


@pytest.mark.upstream_source
def test_non_ppu_norm_retains_real_quantized_provider(
    modules, monkeypatch, upstream_source_root
):
    provider = load_model(modules, monkeypatch, upstream_source_root, False)
    instance = object.__new__(provider.MiniMAXGemmaRMSNorm)
    instance.weight, instance.variance_epsilon = object(), 1e-6
    x = NS(dim=lambda: 2, dtype=object(), shape=(3, 8))
    data, scale, residual = object(), object(), object()
    calls = []

    def norm(x_arg, w_arg, eps, **kwargs):
        calls.append((x_arg, w_arg, eps, kwargs))
        return data, scale

    modules("vllm.models.minimax_m3.nvidia.ops.norm_mxfp8", gemma_rmsnorm_mxfp8=norm)
    provider.QuantizedActivation = lambda **kwargs: NS(**kwargs)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    result, actual_residual = instance.forward(x, residual, mxfp8=True)
    assert actual_residual is residual
    assert result.data is data and result.scale is scale
    assert result.orig_shape == x.shape and result.orig_dtype is x.dtype
    assert result.quant_key is provider.kMxfp8Dynamic
    assert calls == [(x, instance.weight, 1e-6, dict(residual=residual, mxfp8=True))]


@pytest.mark.upstream_source
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("residual_present", [False, True])
def test_real_decoder_input_norm_checks_live_consumer_but_not_under_sp(
    modules, monkeypatch, upstream_source_root, sequence_parallel, residual_present
):
    provider = load_model(modules, monkeypatch, upstream_source_root, True)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    kernel_cls = type("FlashInferCutlassMxfp8LinearKernel", (), {})
    modules(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer",
        FlashInferCutlassMxfp8LinearKernel=kernel_cls,
        FlashInferCutedslMxfp8LinearKernel=type("OtherKernel", (), {}),
    )
    linear = NS(quant_method=NS(kernel=kernel_cls()), key=provider.kMxfp8Dynamic)
    probes, calls = [], []

    def input_key(actual):
        assert actual is linear
        probes.append(actual.key)
        return actual.key

    provider.get_input_quant_key = input_key
    hidden, residual, output, updated = object(), object(), object(), object()

    def norm(value, skip=None, *, mxfp8):
        assert value is hidden
        calls.append((skip, mxfp8))
        return output if skip is None else (output, updated)

    layer = object.__new__(provider.MiniMaxM3DecoderLayer)
    layer.use_sequence_parallel = sequence_parallel
    layer.fuse_input_allreduce = False
    layer.self_attn = NS(qkv_proj=linear)
    layer.input_layernorm = norm
    # 每次调用重新解析 consumer key；SP 时不得调用量化 consumer 探针。
    for key in (provider.kMxfp8Dynamic, None):
        linear.key = key
        skip = residual if residual_present else None
        result = layer.forward_input_norm(hidden, skip)
        assert result == (output, updated if residual_present else hidden)
        assert calls[-1] == (
            skip,
            not sequence_parallel and key is provider.kMxfp8Dynamic,
        )
    assert probes == ([] if sequence_parallel else [provider.kMxfp8Dynamic, None])


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("tp_size", [1, 4])
@pytest.mark.parametrize("replicate_index_q", [False, True])
def test_real_indexed_qkv_constructor_retains_storage_and_initializes_load_state(
    modules, monkeypatch, upstream_source_root, ppu, tp_size, replicate_index_q
):
    load_model(modules, monkeypatch, upstream_source_root, ppu)
    path = upstream_source_root / "vllm/model_executor/layers/linear.py"
    constructors, loads = [], []
    param = NS(load_qkv_weight=lambda **kwargs: loads.append(kwargs))

    class Column:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            self.tp_size, self.tp_rank = tp_size, 0
            self.weight = param

        def named_parameters(self):
            return iter([("weight", self.weight)])

    namespace = dict(
        ColumnParallelLinear=Column,
        get_tensor_model_parallel_world_size=lambda: tp_size,
        divide=lambda value, divisor: value // divisor,
        BlockQuantScaleParameter=type("BlockQuantScaleParameter", (), {}),
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *[
                definition(path, name)
                for name in (
                    "StackedColumnParallelLinear",
                    "QKVParallelLinear",
                    "MinimaxM3QKVParallelLinearWithIndexer",
                )
            ],
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    modules("vllm.model_executor.layers.linear", **namespace)
    cls = namespace["MinimaxM3QKVParallelLinearWithIndexer"]
    quant = object()
    kwargs = dict(
        hidden_size=16,
        head_size=4,
        total_num_heads=8,
        total_num_kv_heads=2,
        total_num_index_heads=2,
        index_head_size=4,
        bias=False,
        quant_config=quant,
        prefix="layers.0.qkv",
        replicate_index_q=replicate_index_q,
    )
    before = cls(**kwargs)
    assert before.loaded_shard_ids == {}
    incoming = object()
    before.weight_loader_v2(param, incoming, "index_k")
    assert before.loaded_shard_ids == {"weight": {"index_k"}}
    assert before.pop_missing_shards() == [
        "weight." + shard for shard in cls.shard_ids[:-1]
    ]
    loads.clear()
    real_init = cls.__init__
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    assert cls.__init__ is real_init
    after = cls(**kwargs)
    assert constructors[0] == constructors[1]
    assert constructors[1]["quant_config"] is quant
    assert after.weight is param
    assert after.__dict__ == before.__dict__
    assert after.loaded_shard_ids == {}
    for shard in cls.shard_ids:
        after.weight_loader_v2(param, incoming, shard)
    assert after.loaded_shard_ids == {"weight": set(cls.shard_ids)}
    assert after.pop_missing_shards() == [] and after.loaded_shard_ids == {}
    assert len(loads) == len(cls.shard_ids)
    for shard, load in zip(cls.shard_ids, loads, strict=False):
        replicated = shard == "index_k" or (shard == "index_q" and replicate_index_q)
        assert load == dict(
            loaded_weight=incoming,
            num_heads=tp_size if replicated else after.num_kv_head_replicas,
            shard_id=shard,
            shard_offset=after._get_shard_offset_mapping(shard),
            shard_size=after._get_shard_size_mapping(shard),
        )
    after.weight_loader_v2(param, incoming, "q")
    assert after.pop_missing_shards() == [
        "weight." + shard for shard in cls.shard_ids[1:]
    ]
    assert after.loaded_shard_ids == {}
    # 真实父类为新一轮初始化重置 bookkeeping，旧加载记录不应跨轮混入。
    after.validate_shard_id("index_q", param)
    existing = after.loaded_shard_ids
    cls.__init__(after, **kwargs)
    assert after.loaded_shard_ids == {} and after.loaded_shard_ids is not existing
    assert existing == {"weight": {"index_q"}}
    assert constructors[2] == constructors[0]
    after.weight_loader_v2(param, incoming, "v")
    assert after.pop_missing_shards() == [
        "weight." + shard for shard in cls.shard_ids if shard != "v"
    ]
    assert after.loaded_shard_ids == {}


@pytest.mark.upstream_source
@pytest.mark.parametrize("pp_size", [1, 3])
@pytest.mark.parametrize("tp_size", [1, 4])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("backend", ["deep_gemm_mega_moe", "triton"])
def test_real_minimax_sequence_parallel_admission_keeps_pp(
    upstream_source_root, pp_size, tp_size, ep, backend
):
    resolve = function(
        upstream_source_root / "vllm/models/minimax_m3/nvidia/model.py",
        "_use_sequence_parallel",
        {},
    )
    config = NS(
        kernel_config=NS(moe_backend=backend),
        parallel_config=NS(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
            enable_expert_parallel=ep,
        ),
    )
    assert resolve(config) is (backend == "deep_gemm_mega_moe" and ep and tp_size > 1)


@pytest.mark.upstream_source
@pytest.mark.parametrize("first_rank", [False, True])
@pytest.mark.parametrize("last_rank", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("with_aux", [False, True])
def test_real_minimax_pp_transport_restores_residual_aux_and_token_extent(
    modules,
    monkeypatch,
    upstream_source_root,
    first_rank,
    last_rank,
    sequence_parallel,
    with_aux,
):
    """执行完整真实 forward 和 Eagle 打包方法，外部 collective 用严格边界替身。"""
    provider = load_model(modules, monkeypatch, upstream_source_root, True)
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    events = []

    class Tensor:
        def __init__(self, rows):
            self.rows = [list(row) for row in rows]
            self.shape = (len(self.rows), len(self.rows[0]))

        def __getitem__(self, selection):
            assert isinstance(selection, slice)
            return Tensor(self.rows[selection])

        def clone(self):
            return Tensor(self.rows)

        def __add__(self, other):
            assert self.shape == other.shape
            return Tensor(
                [
                    [a + b for a, b in zip(x, y, strict=False)]
                    for x, y in zip(self.rows, other.rows, strict=False)
                ]
            )

        def split(self, width, *, dim):
            assert dim == -1 and self.shape[1] % width == 0
            return tuple(
                Tensor([row[start : start + width] for row in self.rows])
                for start in range(0, self.shape[1], width)
            )

    def shifted(value, increment):
        return Tensor([[cell + increment for cell in row] for row in value.rows])

    hidden = Tensor([[i, i + 1] for i in range(5)])
    residual = shifted(hidden, 100)
    positions = object()
    padding = Tensor([[i] for i in range(8)])
    context = NS(is_padding=padding)
    shard_mask = object()

    def shard(value):
        events.append(("shard", value))
        return Tensor(value.rows[:3])

    def gather(value):
        assert value.shape[0] == 3
        events.append(("gather", value))
        return Tensor(value.rows + value.rows)

    def concatenate(values, dim):
        assert dim == -1 and len({value.shape[0] for value in values}) == 1
        events.append(("cat", tuple(values)))
        return Tensor(
            [
                sum([value.rows[i] for value in values], [])
                for i in range(values[0].shape[0])
            ]
        )

    def mask(actual, value):
        assert actual.rows == padding.rows[: hidden.shape[0]] and value is hidden
        events.append(("mask", actual))
        return shard_mask

    def input_norm(value, skip):
        assert value.shape[0] == (3 if sequence_parallel else 5)
        if skip is not None:
            assert skip.shape == value.shape
        events.append(("input-norm", value, skip))
        return value, value if skip is None else skip

    def layer_forward(actual_positions, value, skip):
        assert actual_positions is positions
        assert context.is_padding is (shard_mask if sequence_parallel else padding)
        events.append(("layer", value, skip))
        return shifted(value, 10), shifted(skip, 20)

    class Intermediate(dict):
        @property
        def tensors(self):
            return self

    provider.torch.cat = concatenate
    provider.get_pp_group = lambda: NS(is_first_rank=first_rank, is_last_rank=last_rank)
    provider.sp_shard, provider.sp_all_gather = shard, gather
    provider.sp_padding_mask = mask
    provider.envs = NS(VLLM_MOE_SKIP_PADDING=True)
    provider.is_forward_context_available = lambda: True
    provider.get_forward_context = lambda: context
    provider.IntermediateTensors = Intermediate
    model = object.__new__(provider.MiniMaxM3Model)
    model.use_sequence_parallel = sequence_parallel
    model.start_layer, model.end_layer = 0, 1
    model.layers = [
        NS(forward_input_norm=input_norm, forward_after_input_norm=layer_forward)
    ]
    model.aux_hidden_state_layers = {0, 1} if with_aux else set()
    model.fuse_final_norm_allreduce = False
    model.norm = lambda value, skip: (shifted(value, 30), shifted(skip, 40))
    model.embed_input_ids = lambda actual: hidden
    model.AUX_HIDDEN_STATE_KEY = "aux_"
    model._aux_slot_base_cached = 0
    model._aux_upstream_total_cached = 0
    for name in (
        "_maybe_add_hidden_state",
        "pack_local_aux_hidden_states",
        "collect_remote_aux_hidden_states",
    ):
        method = function(
            upstream_source_root / "vllm/model_executor/models/interfaces.py",
            "EagleModelMixin." + name,
            {},
        )
        setattr(model, name, method.__get__(model))
    intermediate = (
        None
        if first_rank
        else Intermediate(
            hidden_states=hidden,
            residual=residual,
        )
    )
    result = model.forward(None, positions, intermediate)
    assert context.is_padding is padding
    assert [event[0] for event in events].count("shard") == (
        (1 if first_rank else 2) if sequence_parallel else 0
    )
    assert [event[0] for event in events].count("gather") == int(sequence_parallel)
    norm_event = next(event for event in events if event[0] == "input-norm")
    if not first_rank:
        assert norm_event[2].rows == (
            residual.rows[:3] if sequence_parallel else residual.rows
        )
    rows = [0, 1, 2, 0, 1] if sequence_parallel else list(range(5))
    expected_hidden = [
        [i + (40 if last_rank else 10), i + (41 if last_rank else 11)] for i in rows
    ]
    if not last_rank:
        assert result["hidden_states"].rows == expected_hidden
        residual_offset = 20 if first_rank else 120
        assert result["residual"].rows == [
            [i + residual_offset, i + residual_offset + 1] for i in rows
        ]
        aux_count = (2 if first_rank else 1) if with_aux else 0
        assert set(result) == {
            "hidden_states",
            "residual",
            *["aux_" + str(i) for i in range(aux_count)],
        }
        if with_aux:
            summed_offset = 30 if first_rank else 130
            assert result["aux_" + str(aux_count - 1)].rows == [
                [2 * i + summed_offset, 2 * i + summed_offset + 2] for i in rows
            ]
    else:
        if with_aux:
            output, aux = result
            assert len(aux) == (2 if first_rank else 1)
            skip_offset = 60 if first_rank else 160
            assert aux[-1].rows == [
                [i + skip_offset, i + skip_offset + 1] for i in rows
            ]
        else:
            output = result
        assert output.rows == expected_hidden
    if sequence_parallel:
        gather_event = next(event for event in events if event[0] == "gather")
        assert gather_event[1].shape[1] == 2 * (
            (1 if last_rank else 2) + ((2 if first_rank else 1) if with_aux else 0)
        )


@pytest.mark.upstream_source
def test_ppu_mxfp8_emulation_does_not_request_swizzled_norm(
    modules, monkeypatch, upstream_source_root
):
    provider = load_model(modules, monkeypatch, upstream_source_root, True)
    for path, name in [
        (
            ROOT / "vllm_sail/model_executor/kernels/linear/mxfp8.py",
            "PPUEmulationMxfp8LinearKernel",
        ),
        (
            upstream_source_root
            / "vllm/model_executor/kernels/linear/mxfp8/emulation.py",
            "EmulationMxfp8LinearKernel",
        ),
    ]:
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name == "input_quant_key"
            for node in definition(path, name).body
        )
    input_key = function(
        upstream_source_root
        / "vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py",
        "Mxfp8LinearKernel.input_quant_key",
        {},
    )
    linear = NS(quant_method=NS(kernel=object()))
    provider.get_input_quant_key = lambda layer: input_key(layer.quant_method.kernel)
    assert provider._consumes_mxfp8_128x4(linear) is False


@pytest.mark.upstream_source
def test_channelwise_qkv_skips_nvidia_provider_import(
    modules, monkeypatch, upstream_source_root
):
    provider = load_model(modules, monkeypatch, upstream_source_root, True)
    name = "vllm.model_executor.kernels.linear.mxfp8.flashinfer"

    class Forbidden(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name:
                raise ModuleNotFoundError("NVIDIA MXFP8 provider", name="flashinfer")

    modules("vllm.model_executor.kernels.linear.mxfp8")
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Forbidden(), *sys.meta_path])
    # The pin imports this provider before it looks at either the chip or weight.
    with pytest.raises(ModuleNotFoundError, match="NVIDIA MXFP8"):
        provider._supports_qkv_fused_epilogue(object())
    load_patch("vllm_sail/patch/enhancement/models/minimax_m3.py")
    assert provider._supports_qkv_fused_epilogue(object()) is False
