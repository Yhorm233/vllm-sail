# SPDX-License-Identifier: Apache-2.0
"""GLM's FP8 producers must not reach SAIL's MXFP4-only weighted API."""

import sys
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import assert_accepts_upstream_keywords, function

PATCH = "vllm_sail/patch/enhancement/attention/sparse_attn_indexer_kpool.py"
TARGET = "vllm.models.glm5next.nvidia.sparse_indexer"
CONSUMERS = (
    "vllm.models.glm5next.sparse_indexer",
    "vllm.models.glm5next.common.attention",
)


class Tensor:
    def __init__(self, shape, dtype="fp8"):
        self.shape, self.dtype = shape, dtype

    def reshape(self, *shape):
        self.shape = shape
        return self

    def contiguous(self):
        return self

    def stride(self, dim):
        return 1

    def dim(self):
        return len(self.shape)


@pytest.fixture(params=[False, True], ids=["cold", "preloaded"])
def kpool(modules, patch_utils_module, monkeypatch, request):
    if request.param:
        # Installed-vLLM CI can preload these consumers before this fixture
        # replaces the provider. Reproduce that state on bare CPU runners too.
        for name in CONSUMERS:
            monkeypatch.setitem(sys.modules, name, NS(SparseAttnIndexerKpool=object()))

    class Indexer:
        def forward_native(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

        def forward_cuda(self, *args, **kwargs):
            return "cuda"

    calls = []

    def spy(name):
        def record(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return record

    modules(
        "torch",
        float8_e4m3fn="fp8",
        float32="fp32",
        bfloat16="bf16",
        contiguous_format="contiguous",
    )
    platform = NS(ppu=True)
    platform.is_ppu = lambda: platform.ppu
    modules("vllm.platforms", current_platform=platform)
    modules("vllm.utils.torch_utils", _encode_layer_name=lambda name: name)
    modules("vllm_sail.ops")
    modules(
        "vllm_sail.utils.deep_gemm",
        fp8_mqa_logits=spy("fp8"),
        fp8_paged_mqa_logits=spy("paged_fp8"),
        get_num_sms=lambda: 7,
        get_paged_mqa_logits_metadata=spy("schedule"),
    )
    originals = (spy("cuda_prefill"), spy("cuda_decode"))
    provider = modules(
        TARGET,
        SparseAttnIndexerKpool=Indexer,
        fp8_fp4_mqa_logits=originals[0],
        fp8_fp4_paged_mqa_logits=originals[1],
    )
    # Isolate the complete provider/consumer graph, including modules that
    # real-vLLM test collection may already have imported.
    for name in CONSUMERS:
        modules(name, SparseAttnIndexerKpool=Indexer)
    installed = load_patch(PATCH)
    assert installed._EXPECTED_CONSUMERS == CONSUMERS
    for name in CONSUMERS:
        assert (
            sys.modules[name].SparseAttnIndexerKpool is provider.SparseAttnIndexerKpool
        )
    for name, original in zip(
        ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"), originals, strict=False
    ):
        replacement = getattr(provider, name)
        assert (
            patch_utils_module.original_of(replacement, TARGET + "." + name) is original
        )
        with pytest.raises(RuntimeError, match="already"):
            patch_utils_module.patch(
                TARGET,
                name,
                reason="Duplicate GLM logits override.",
                affected_versions=">=0.33.0,<0.34.0",
                remove_when="GLM has a platform logits registry.",
            )(replacement)
    return NS(provider=provider, patch=installed, platform=platform, calls=calls)


def test_glm_fp8_prefill_retains_relative_columns_and_fp32_weights(kpool):
    q, kv, weights, starts, ends = (
        Tensor((2, 32, 128)),
        object(),
        object(),
        object(),
        object(),
    )
    result = kpool.provider.fp8_fp4_mqa_logits((q, None), kv, weights, starts, ends, 65)
    assert result == "fp8"
    assert kpool.calls == [
        (
            "fp8",
            (q, kv, weights, starts, ends),
            {"clean_logits": False, "max_seqlen_k": 65},
        )
    ]


def test_glm_fp8_decode_rebuilds_schedule_without_mxfp4_request_ids(kpool):
    q, kv = Tensor((6, 1, 32, 128)), Tensor((9, 64, 1, 132))
    weights, contexts, tables, ids = object(), Tensor((6, 1)), Tensor((6, 2)), object()
    result = kpool.provider.fp8_fp4_paged_mqa_logits(
        (q, None),
        kv,
        weights,
        contexts,
        tables,
        "stale",
        65,
        indices=ids,
    )
    assert result == "paged_fp8"
    assert kpool.calls == [
        ("schedule", (contexts, 64, 7, (1, 32, 128, "fp8")), {}),
        (
            "paged_fp8",
            (q, kv, weights, contexts, tables, "schedule", 65),
            {"clean_logits": False},
        ),
    ]


def test_glm_decode_repairs_singleton_table_stride(kpool):
    repaired = Tensor((2, 1))

    class Table(Tensor):
        def stride(self, dim):
            return 8

        def clone(self, *, memory_format):
            assert memory_format == "contiguous"
            return repaired

    kpool.provider.fp8_fp4_paged_mqa_logits(
        (Tensor((2, 1, 32, 128)), None),
        Tensor((2, 64, 1, 132)),
        object(),
        Tensor((2, 1)),
        Table((2, 1)),
        "stale",
        64,
    )
    assert kpool.calls[-1][1][4] is repaired


@pytest.mark.parametrize("paged", [False, True])
def test_glm_wrappers_preserve_non_ppu_calls(kpool, paged):
    kpool.platform.ppu = False
    args = tuple(object() for _ in range(7 if paged else 6))
    if paged:
        ids = object()
        assert (
            kpool.provider.fp8_fp4_paged_mqa_logits(*args, indices=ids) == "cuda_decode"
        )
        assert kpool.calls == [
            ("cuda_decode", args, {"logits_dtype": "fp32", "indices": ids})
        ]
    else:
        assert kpool.provider.fp8_fp4_mqa_logits(*args) == "cuda_prefill"
        assert kpool.calls == [("cuda_prefill", args, {"logits_dtype": "fp32"})]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_glm_wrappers_forward_explicit_positional_dtype(kpool, paged, dtype):
    kpool.platform.ppu = False
    args = tuple(object() for _ in range(7 if paged else 6))
    if paged:
        ids = object()
        assert (
            kpool.provider.fp8_fp4_paged_mqa_logits(*args, dtype, ids) == "cuda_decode"
        )
        assert kpool.calls == [
            ("cuda_decode", args, {"logits_dtype": dtype, "indices": ids})
        ]
    else:
        assert kpool.provider.fp8_fp4_mqa_logits(*args, dtype) == "cuda_prefill"
        assert kpool.calls == [("cuda_prefill", args, {"logits_dtype": dtype})]


@pytest.mark.parametrize("paged", [False, True])
def test_glm_ppu_rejects_non_fp32_before_inspecting_or_scoring_inputs(kpool, paged):
    compute = (
        kpool.provider.fp8_fp4_paged_mqa_logits
        if paged
        else kpool.provider.fp8_fp4_mqa_logits
    )
    args = tuple(object() for _ in range(7 if paged else 6))
    with pytest.raises(ValueError, match="indexer_logits_dtype='float32'"):
        compute(*args, logits_dtype="bf16")
    assert kpool.calls == []


@pytest.mark.upstream_source
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_glm_non_ppu_reaches_real_unified_provider(
    kpool, upstream_source_root, paged, dtype
):
    name = "fp8_fp4_paged_mqa_logits" if paged else "fp8_fp4_mqa_logits"
    calls, output = [], object()

    def implementation(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    namespace = {
        "_lazy_init": lambda: None,
        "_missing": lambda: pytest.fail("real provider must reach implementation"),
        "_" + name + "_impl": implementation,
        "torch": NS(float32="fp32", contiguous_format="contiguous"),
    }
    original = function(
        upstream_source_root / "vllm/utils/deep_gemm.py", name, namespace
    )
    attribute = "_upstream_paged_mqa_logits" if paged else "_upstream_mqa_logits"
    setattr(kpool.patch, attribute, original)
    kpool.platform.ppu = False
    q, kv, weights, starts, ends = (
        object(),
        object(),
        object(),
        object(),
        Tensor((2, 3)),
    )
    args = (
        (q, kv, weights, starts, ends, object(), 65)
        if paged
        else (q, kv, weights, starts, ends, 65)
    )
    kwargs = {"logits_dtype": dtype}
    if paged:
        kwargs["indices"] = object()
    assert getattr(kpool.provider, name)(*args, **kwargs) is output
    assert len(calls) == 1
    passed, options = calls[0]
    assert all(
        actual is expected for actual, expected in zip(passed, args, strict=True)
    )
    assert options == kwargs


@pytest.mark.upstream_source
@pytest.mark.parametrize("paged", [False, True])
def test_glm_adapters_accept_pinned_upstream_keywords(upstream_source_root, paged):
    assert_accepts_upstream_keywords(
        "attention/sparse_attn_indexer_kpool",
        "fp8_paged_mqa_logits" if paged else "fp8_mqa_logits",
        upstream_source_root,
        "utils/deep_gemm",
        "fp8_fp4_paged_mqa_logits" if paged else "fp8_fp4_mqa_logits",
    )


def test_int8_paged_mtp_aligns_ragged_queries_contexts_and_tables():
    torch = pytest.importorskip("torch")
    q = (
        torch.arange(4 * 32 * 128, dtype=torch.int32)
        .to(torch.int8)
        .reshape(2, 2, 32, 128)
    )
    weights = torch.ones((4, 32))
    kv_cache = torch.empty((8, 64, 1, 132), dtype=torch.uint8)
    metadata = NS(
        seq_lens=torch.tensor([[4, 5], [9, 10]], dtype=torch.int32),
        block_table=torch.tensor([[1, 2], [5, 6]], dtype=torch.int32),
        decode_lens=torch.tensor([2, 1], dtype=torch.int32),
        per_req_decode_lens=torch.tensor([2, 1], dtype=torch.int32),
        indices=None,
        schedule_metadata="stale",
    )
    calls = []

    def schedule(contexts, block, sms, geometry):
        assert contexts.tolist() == [[4], [5], [10], [0]]
        assert (block, sms, geometry) == (64, 7, (1, 32, 128, torch.int8))
        calls.append(contexts)
        return "fresh"

    def compute(query, cache, passed_weights, contexts, tables, sched, **kwargs):
        assert query.shape == (4, 1, 32, 128)
        assert torch.equal(query.flatten(), q.flatten())
        assert cache is kv_cache and passed_weights is weights
        assert contexts is calls[0] and sched == "fresh"
        assert tables.tolist() == [[1, 2], [1, 2], [5, 6], [5, 6]]
        assert kwargs == {"max_model_len": 65, "clean_logits": False}
        return "logits"

    rows = function(
        "vllm_sail/attention/ops/mla_sparse.py", "_paged_decode_rows", {"torch": torch}
    )
    fn = function(
        "vllm_sail/attention/ops/mla_sparse_kpool.py",
        "_decode_logits_paged",
        {
            "torch": torch,
            "_paged_decode_rows": rows,
            "get_paged_mqa_logits_metadata": schedule,
            "get_num_sms": lambda: 7,
            "int8_paged_mqa_logits": compute,
        },
    )
    logits, contexts = fn(q, kv_cache, weights, metadata, 2, 65, 128)
    assert logits == "logits"
    assert contexts.tolist() == [[4, 5], [10, 0]]
