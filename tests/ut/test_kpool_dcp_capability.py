# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual registered INT8 guard and unchanged fp8 delegation."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch


@pytest.fixture
def kpool(modules):
    def load(*, fp8=False):
        calls = []
        modules(
            "torch",
            float32="fp32",
            ops=NS(
                vllm=NS(
                    ppu_sparse_attn_indexer_kpool=lambda *args: calls.append(args)
                    or "int8"
                )
            ),
        )
        modules(
            "vllm.platforms",
            current_platform=NS(is_ppu=lambda: True, supports_fp8=lambda: fp8),
        )
        modules("vllm.utils.torch_utils", _encode_layer_name=lambda name: name)
        modules("vllm_sail.ops")

        class Indexer:
            def forward_native(self, *args, **kwargs):
                raise AssertionError("INT8 redirect must reach the registered PPU path")

            def forward_cuda(
                self,
                hidden_states,
                q_quant,
                k,
                weights,
                *,
                gate_score=None,
                compress_ape=None,
                index_kpool=1,
                positions=None,
                topk_backend="auto",
            ):
                return "upstream-fp8", self.dcp_world_size, topk_backend

        modules(
            "vllm.models.glm5next.nvidia.sparse_indexer",
            SparseAttnIndexerKpool=Indexer,
            fp8_fp4_mqa_logits=lambda *args, **kwargs: "prefill",
            fp8_fp4_paged_mqa_logits=lambda *args, **kwargs: "decode",
        )
        for name in (
            "vllm.models.glm5next.sparse_indexer",
            "vllm.models.glm5next.common.attention",
        ):
            modules(name, SparseAttnIndexerKpool=Indexer)
        patch = load_patch(
            "vllm_sail/patch/enhancement/attention/sparse_attn_indexer_kpool.py"
        )
        assert Indexer.forward_ppu is patch.forward_ppu
        return Indexer, calls

    return load


@pytest.mark.parametrize("method", ["forward_ppu", "forward_native"])
def test_int8_dcp_fails_before_accessing_or_mutating_cache(kpool, method):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 2
    with pytest.raises(NotImplementedError, match="decode context parallel size 1"):
        getattr(layer, method)(*[object() for _ in range(4)])
    assert calls == []


@pytest.mark.parametrize("method", ["forward_ppu", "forward_native"])
def test_int8_pcp_fails_before_dtype_query_or_cache_access(kpool, method):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 1
    layer._parallel_config = NS(prefill_context_parallel_size=2)
    # 不提供 dtype、Q、cache 或 op 替身之外的属性，证明拒绝早于读取这些输入。
    with pytest.raises(NotImplementedError, match="prefill context parallel size 1"):
        getattr(layer, method)(*[object() for _ in range(4)])
    assert calls == []


@pytest.mark.parametrize("method", ["forward_ppu", "forward_native"])
@pytest.mark.parametrize("requested,producer", [("fp32", "bf16"), ("bf16", "fp32")])
def test_int8_dtype_mismatch_fails_before_cache_access(
    kpool, method, requested, producer
):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 1
    layer.logits_dtype = requested
    with pytest.raises(ValueError, match="must match the producer weights dtype"):
        getattr(layer, method)(object(), object(), object(), NS(dtype=producer))
    assert calls == []


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_int8_matching_dtype_preserves_inputs_and_tail_coordinates(kpool, dtype):
    cls, calls = kpool()
    layer = cls()
    layer.dcp_world_size = 1
    layer.logits_dtype = dtype
    layer.k_cache = NS(prefix="indexer", kv_cache=object())
    layer.tail_cache = NS(prefix="tail", kv_cache=object())
    for field in (
        "quant_block_size",
        "scale_fmt",
        "topk_tokens",
        "head_dim",
        "max_pool_len",
        "max_total_seq_len",
        "topk_indices_buffer",
        "skip_k_cache_insert",
        "use_fp4_cache",
    ):
        setattr(layer, field, object())
    hidden, q, scale, k, weights = (
        object(),
        object(),
        object(),
        object(),
        NS(dtype=dtype),
    )
    gate, ape, positions = object(), object(), object()
    assert (
        layer.forward_ppu(
            hidden,
            (q, scale),
            k,
            weights,
            gate_score=gate,
            compress_ape=ape,
            positions=positions,
            index_kpool=2,
        )
        == "int8"
    )
    assert calls == [
        (
            hidden,
            "indexer",
            layer.k_cache.kv_cache,
            q,
            scale,
            k,
            weights,
            layer.quant_block_size,
            layer.scale_fmt,
            layer.topk_tokens,
            layer.head_dim,
            layer.max_pool_len,
            layer.max_total_seq_len,
            layer.topk_indices_buffer,
            layer.skip_k_cache_insert,
            layer.use_fp4_cache,
            gate,
            ape,
            2,
            positions,
            layer.tail_cache.kv_cache,
            "tail",
        )
    ]


def test_fp8_dcp_keeps_upstream_dispatch(kpool):
    cls, calls = kpool(fp8=True)
    layer = cls()
    layer.dcp_world_size = 2
    assert layer.forward_ppu(*[object() for _ in range(4)], topk_backend="triton") == (
        "upstream-fp8",
        2,
        "triton",
    )
    assert calls == []


@pytest.mark.upstream_source
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("method", ["forward_ppu", "forward_native", "forward"])
@pytest.mark.parametrize("dcp_world", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_kpool_constructor_and_installer_preserve_dispatch_contract(
    modules,
    upstream_source_root,
    fp8,
    method,
    dcp_world,
    enabled,
):
    """真实 CustomOp、Kpool 类与安装器；nn.Module 和计算是边界，不证明分布式数值。"""
    import functools
    import inspect

    from tests.support.source import function

    calls, cache_reads = [], []
    platform = NS(
        is_ppu=lambda: True,
        is_cuda=lambda: True,
        supports_fp8=lambda: fp8,
        is_rocm=lambda: False,
        is_cpu=lambda: False,
        is_tpu=lambda: False,
        is_xpu=lambda: False,
        is_out_of_tree=lambda: False,
    )
    torch = modules(
        "torch",
        float32="fp32",
        ops=NS(
            vllm=NS(
                ppu_sparse_attn_indexer_kpool=lambda *args: calls.append(("int8", args))
                or "int8"
            )
        ),
    )
    modules("vllm.platforms", current_platform=platform)
    modules("vllm.utils.torch_utils", _encode_layer_name=lambda name: name)
    modules("vllm_sail.ops")
    config = NS(
        kernel_config=NS(sparse_indexer_topk_backend="triton"),
        attention_config=NS(resolve_indexer_logits_dtype=lambda default: default),
        parallel_config=NS(
            decode_context_parallel_size=dcp_world,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=8,
        ),
    )
    compilation = NS(
        custom_ops=["all" if enabled else "none"],
        enabled_custom_ops=set(),
        disabled_custom_ops=set(),
    )
    modules("vllm.config.compilation", CompilationMode=NS(NONE=0))
    namespace = dict(
        __name__=__name__,
        torch=torch,
        nn=NS(Module=type("ModuleBoundary", (), {})),
        functools=functools,
        inspect=inspect,
        op_registry={},
        op_registry_oot={},
        get_cached_compilation_config=lambda: compilation,
        current_platform=platform,
        has_deep_gemm=lambda: True,
        has_cutedsl=lambda: False,
        get_current_vllm_config_or_none=lambda: config,
        get_dcp_group=lambda: NS(rank_in_group=1),
        STR_DTYPE_TO_TORCH_DTYPE={"float32": "fp32"},
        sparse_attn_indexer_kpool=lambda *args: calls.append(("upstream", args))
        or "upstream",
    )
    base = function(
        upstream_source_root / "vllm/model_executor/custom_op.py",
        "CustomOp",
        namespace,
    )
    modules("vllm.model_executor.custom_op", CustomOp=base)
    dispatcher = load_patch("vllm_sail/patch/enhancement/custom_op_dispatch.py")
    assert base.dispatch_forward is dispatcher.dispatch_forward
    assert base.forward_ppu is dispatcher.forward_ppu
    cls = function(
        upstream_source_root / "vllm/models/glm5next/nvidia/sparse_indexer.py",
        "SparseAttnIndexerKpool",
        namespace,
    )
    # 源码抽取移除类的外层 decorator，显式应用真实 register，保留注册与分派语义。
    cls = base.register("sparse_attn_indexer_kpool")(cls)
    assert namespace["op_registry"][cls.name] is cls
    modules(
        "vllm.models.glm5next.nvidia.sparse_indexer",
        SparseAttnIndexerKpool=cls,
        fp8_fp4_mqa_logits=lambda *a, **kw: None,
        fp8_fp4_paged_mqa_logits=lambda *a, **kw: None,
    )
    for name in (
        "vllm.models.glm5next.sparse_indexer",
        "vllm.models.glm5next.common.attention",
    ):
        modules(name, SparseAttnIndexerKpool=cls)
    installed = load_patch(
        "vllm_sail/patch/enhancement/attention/sparse_attn_indexer_kpool.py"
    )
    assert cls.forward_ppu is installed.forward_ppu
    assert cls.forward_native is installed.forward_native

    class Cache:
        def __init__(self, prefix):
            self._prefix, self._cache = prefix, object()

        @property
        def prefix(self):
            cache_reads.append(self._prefix)
            return self._prefix

        @property
        def kv_cache(self):
            cache_reads.append(self._prefix)
            return self._cache

    cache, tail, output = Cache("indexer"), Cache("tail"), object()
    layer = cls(cache, 128, "ue8m0", 4, 128, 16, 32, output, tail_cache=tail)
    assert layer._forward_method == (
        layer.forward_ppu if enabled else layer.forward_native
    )
    assert compilation.enabled_custom_ops == ({cls.name} if enabled else set())
    assert compilation.disabled_custom_ops == (set() if enabled else {cls.name})
    assert layer._parallel_config is config.parallel_config
    assert layer.dcp_world_size == dcp_world
    assert layer.dcp_rank == (1 if dcp_world > 1 else 0)
    assert layer.cp_kv_cache_interleave_size == (8 if dcp_world > 1 else 1)
    assert layer.logits_dtype == "fp32" and layer.topk_backend == "triton"
    hidden, q, k, weights = object(), object(), object(), NS(dtype="fp32")
    gate, ape, positions = object(), object(), object()
    if not fp8 and dcp_world > 1:
        with pytest.raises(NotImplementedError, match="decode context parallel size 1"):
            getattr(layer, method)(hidden, (q, None), k, weights)
        assert calls == [] and cache_reads == []
    else:
        result = getattr(layer, method)(
            hidden,
            (q, None),
            k,
            weights,
            gate_score=gate,
            compress_ape=ape,
            positions=positions,
            index_kpool=2,
        )
        route = "upstream" if fp8 else "int8"
        assert result == route and len(calls) == 1
        actual_route, args = calls[0]
        assert actual_route == route
        common = (
            hidden,
            "indexer",
            cache._cache,
            q,
            None,
            k,
            weights,
            128,
            "ue8m0",
            4,
            128,
            16,
            32,
            output,
            False,
            False,
            gate,
            ape,
            2,
            positions,
            tail._cache,
            "tail",
        )
        extra = (
            (
                "triton",
                layer.dcp_rank,
                dcp_world,
                layer.cp_kv_cache_interleave_size,
                "fp32",
            )
            if fp8
            else ()
        )
        assert args == common + extra
    # 真实 constructor 保留 PCP+DCP 的上游拒绝边界。
    config.parallel_config.prefill_context_parallel_size = 2
    if dcp_world > 1:
        with pytest.raises(NotImplementedError, match=r"does not support PCP\+DCP"):
            cls(cache, 128, "ue8m0", 4, 128, 16, 32, output)
    else:
        # 单独 PCP 被真实 constructor 接受；拒绝必须来自实际安装的 INT8 分派。
        calls.clear()
        cache_reads.clear()
        pcp_layer = cls(cache, 128, "ue8m0", 4, 128, 16, 32, output, tail_cache=tail)
        assert pcp_layer._parallel_config is config.parallel_config
        assert pcp_layer.dcp_world_size == 1
        assert pcp_layer._forward_method == (
            pcp_layer.forward_ppu if enabled else pcp_layer.forward_native
        )
        if fp8:
            assert (
                getattr(pcp_layer, method)(
                    hidden,
                    (q, None),
                    k,
                    weights,
                    gate_score=gate,
                    compress_ape=ape,
                    positions=positions,
                    index_kpool=2,
                )
                == "upstream"
            )
            assert calls == [("upstream", common + ("triton", 0, 1, 1, "fp32"))]
        else:
            with pytest.raises(
                NotImplementedError, match="prefill context parallel size 1"
            ):
                getattr(pcp_layer, method)(hidden, object(), k, object())
            assert calls == [] and cache_reads == []
        assert cache._cache is pcp_layer.k_cache._cache
        assert tail._cache is pcp_layer.tail_cache._cache

        # 同一 PCP 配置在非 PPU 路径仍委托真实 upstream forward，不受 INT8 guard 影响。
        platform.is_ppu = lambda: False
        calls.clear()
        cache_reads.clear()
        other_layer = cls(cache, 128, "ue8m0", 4, 128, 16, 32, output, tail_cache=tail)
        assert (
            getattr(other_layer, method)(
                hidden,
                (q, None),
                k,
                weights,
                gate_score=gate,
                compress_ape=ape,
                positions=positions,
                index_kpool=2,
            )
            == "upstream"
        )
        assert calls == [("upstream", common + ("triton", 0, 1, 1, "fp32"))]


@pytest.mark.upstream_source
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("topk", [4, 512])
def test_real_dcp_merge_bypasses_single_rank_and_requires_cutedsl(
    upstream_source_root,
    world,
    dtype,
    topk,
):
    """真实 merge 的单 rank 与缺少 CuteDSL 边界；不提供分布式数值或设备证据。"""
    from tests.support.source import function

    capability_reads = []

    def cutedsl_available():
        capability_reads.append("cutedsl")
        return False

    namespace = dict(
        torch=NS(float32="fp32", int32="i32"),
        has_cutedsl=cutedsl_available,
    )
    path = upstream_source_root / "vllm/model_executor/layers/sparse_attn_indexer.py"
    function(path, "_assert_cutedsl_dcp_merge_supported", namespace)
    merge = function(path, "_merge_dcp_topk_global", namespace)
    logits = NS(dtype=dtype, device=NS(type="cuda"))
    indices = NS(dtype="i32")
    if world == 1:
        assert (
            merge(
                logits,
                indices,
                topk,
                0,
                world,
                1,
                logits_dtype=dtype,
                num_positions=16384,
            )
            is None
        )
        assert capability_reads == []
    else:
        # 未提供分配、kernel、group 或 collective 替身；缺能力时必须在触及它们之前拒绝。
        with pytest.raises(RuntimeError, match="requires CuteDSL"):
            merge(
                logits,
                indices,
                topk,
                0,
                world,
                1,
                logits_dtype=dtype,
                num_positions=16384,
            )
        assert capability_reads == ["cutedsl"]
