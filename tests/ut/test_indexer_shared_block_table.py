# SPDX-License-Identifier: Apache-2.0
"""Execute the delegating indexer build wrapper and its builder-aware helpers."""

from types import SimpleNamespace as NS
from unittest.mock import ANY

import pytest

from tests.support.source import definition, function

PATCH = "vllm_sail/patch/enhancement/attention/mla_indexer.py"


class Span:
    shape = (3, 1)
    ndim = 2
    device = "cuda"

    def reshape(self, *shape):
        return self

    def contiguous(self):
        return self

    def is_contiguous(self):
        return True

    @staticmethod
    def dim():
        return 2


def _paged_namespace(modules, *, ppu, builder=None):
    calls = []
    fallback = []

    def paged(context_lens, block_size, num_sms, metadata_extra=None, **kwargs):
        calls.append(
            (context_lens, block_size, num_sms, metadata_extra, kwargs.get("indices"))
        )
        return "ppu-metadata"

    def upstream(context_lens, block_size, num_sms, indices=None):
        fallback.append((context_lens, block_size, num_sms, indices))
        return "upstream-metadata"

    namespace = {
        "_current_builder": [builder] if builder is not None else [],
        "_upstream_paged_metadata": upstream,
        "_ppu_deep_gemm": NS(get_paged_mqa_logits_metadata=paged),
        "torch": NS(int8="int8", int32="int32"),
    }
    modules(
        "vllm.platforms",
        current_platform=NS(
            is_ppu=lambda: ppu,
            is_device_capability=lambda cap: cap == (8, 9),
            fp8_dtype=lambda: "fp8",
        ),
    )
    return namespace, calls, fallback


def test_wrapper_delegates_with_the_builder_visible(modules):
    seen = []
    metadata = NS(decode=NS(schedule_metadata="upstream"))
    builder = NS()
    namespace = {"_current_builder": []}

    def upstream(*args):
        seen.append(list(namespace["_current_builder"]))
        return metadata

    namespace["_upstream_build"] = upstream
    result = function(PATCH, "_build_wrapper", namespace)(builder, 0, None)

    assert result is metadata
    assert seen == [[builder]]
    assert namespace["_current_builder"] == []
    assert builder._sail_decode_request_ids is None


def test_wrapper_pops_the_builder_when_upstream_raises(modules):
    namespace = {"_current_builder": []}
    builder = NS()

    def upstream(*args):
        builder._sail_decode_request_ids = "partial-build-ids"
        raise RuntimeError("upstream build failed")

    namespace["_upstream_build"] = upstream

    with pytest.raises(RuntimeError, match="upstream build failed"):
        function(PATCH, "_build_wrapper", namespace)(builder, 0, None)

    assert namespace["_current_builder"] == []
    assert builder._sail_decode_request_ids is None


@pytest.mark.parametrize("use_flattening", [False, True])
def test_int8_builder_marks_only_graph_stable_flattened_schedules(
    modules, use_flattening
):
    builder = NS(use_flattening=use_flattening)
    namespace, calls, _ = _paged_namespace(modules, ppu=True)
    paged = function(PATCH, "_ppu_paged_metadata", namespace)
    geometry = (1, 64, 128, "int8")
    namespace["_upstream_build"] = lambda *args: NS(
        decode=NS(schedule_metadata=paged(Span(), 64, 7, geometry))
    )
    decode = function(PATCH, "_build_wrapper", namespace)(builder, 0, None).decode

    assert len(calls) == 1
    assert getattr(decode, "_sail_paged_schedule_key", None) == (
        (64, 7, geometry, False) if use_flattening else None
    )


@pytest.mark.parametrize(
    "page_size,num_sms,padding",
    [(64, 7, False), (32, 7, False), (64, 8, False), (64, 7, True)],
)
def test_int8_decode_layers_reuse_matching_schedule(page_size, num_sms, padding):
    contexts, tables, schedule = Span(), Span(), object()
    metadata = NS(
        block_table=tables,
        schedule_metadata=schedule,
        indices=None,
        requires_padding=padding,
        per_req_decode_lens=None,
        _sail_paged_schedule_key=(64, 7, (1, 64, 128, "int8"), False),
    )
    calls = []

    def rebuild(*args, indices):
        assert indices is None
        calls.append("rebuild")
        return "rebuilt"

    inputs = function(
        "vllm_sail/attention/ops/mla_sparse.py",
        "_paged_decode_inputs",
        {
            "torch": NS(int8="int8"),
            "current_platform": NS(
                is_ppu=lambda: True, is_device_capability=lambda cap: cap == (8, 0)
            ),
            "get_num_sms": lambda: num_sms,
            "_paged_decode_rows": lambda *args, **kwargs: (contexts, tables, None),
            "get_paged_mqa_logits_metadata": rebuild,
        },
    )
    reuse = (page_size, num_sms, padding) == (64, 7, False)
    for _ in range(2):
        assert inputs(
            metadata, contexts, NS(shape=(3,)), 1, page_size, 64, 128, "int8", False
        ) == (contexts, tables, None, schedule if reuse else "rebuilt")
    assert calls == ([] if reuse else ["rebuild"] * 2)


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_m890_builder_marks_only_native_varlen_or_flattened_schedules(
    modules, fp4, prepared
):
    builder = NS(
        indexer_uses_fp4=fp4,
        supports_varlen=fp4 and prepared,
        use_flattening=not fp4 and prepared,
    )
    namespace, calls, _ = _paged_namespace(modules, ppu=True)
    paged = function(PATCH, "_ppu_paged_metadata", namespace)
    geometry = (1, 64, 128, "int8" if fp4 else "fp8")
    indices = object() if fp4 and prepared else None
    fallback_ids = object()
    namespace["torch"].arange = lambda *args, **kw: NS(
        repeat_interleave=lambda next_n: fallback_ids
    )
    namespace["_upstream_build"] = lambda *args: NS(
        decode=NS(
            indices=indices,
            schedule_metadata=paged(Span(), 64, 7, geometry, indices=indices),
        )
    )
    decode = function(PATCH, "_build_wrapper", namespace)(builder, 0, None).decode

    assert len(calls) == 1
    assert calls[0][-1] is (fallback_ids if fp4 and not prepared else indices)
    assert getattr(decode, "_sail_paged_schedule_key", None) == (
        (64, 7, geometry, fp4) if prepared else None
    )
    assert decode.indices is indices


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("mismatch", [None, "page", "cu", "padding", "indices"])
def test_m890_decode_layers_reuse_matching_schedule_and_ids(fp4, mismatch):
    contexts, tables, schedule = Span(), Span(), object()
    indices = Span()
    indices.shape = (3,)
    metadata = NS(
        block_table=tables,
        schedule_metadata=schedule,
        indices=indices if fp4 and mismatch != "indices" else None,
        requires_padding=mismatch == "padding",
        per_req_decode_lens=None,
        _sail_paged_schedule_key=(64, 7, (1, 64, 128, "int8" if fp4 else "fp8"), fp4),
    )
    calls = []
    rebuilt_ids = object()

    def rebuild(*args, indices):
        assert indices is (rebuilt_ids if fp4 else None)
        calls.append("rebuild")
        return "rebuilt"

    def flatten(*args, need_request_ids):
        assert need_request_ids is fp4
        assert args[4] is metadata.indices
        return contexts, tables, rebuilt_ids if fp4 else None

    inputs = function(
        "vllm_sail/attention/ops/mla_sparse.py",
        "_paged_decode_inputs",
        {
            "torch": NS(int8="int8", float8_e4m3fn="fp8"),
            "current_platform": NS(
                is_ppu=lambda: True, is_device_capability=lambda cap: cap == 89
            ),
            "get_num_sms": lambda: 8 if mismatch == "cu" else 7,
            "_paged_decode_rows": flatten,
            "get_paged_mqa_logits_metadata": rebuild,
        },
    )
    rebuild_schedule = mismatch in ("page", "cu", "padding") or (
        fp4 and mismatch == "indices"
    )
    expected_ids = rebuilt_ids if rebuild_schedule else indices
    for _ in range(2):
        used = inputs(
            metadata,
            contexts,
            NS(shape=(3,)),
            1,
            32 if mismatch == "page" else 64,
            64,
            128,
            "fp8",
            fp4,
        )
        assert used == (
            contexts,
            tables,
            expected_ids if fp4 else None,
            "rebuilt" if rebuild_schedule else schedule,
        )
    assert calls == (["rebuild"] * 2 if rebuild_schedule else [])


def test_prebuilt_metadata_keeps_request_ids_after_upstream_flattens(modules):
    builder = NS(
        indexer_uses_fp4=True,
        indexer_n_head=32,
        indexer_q_head_dim=128,
        per_req_decode_lens_buffer=[3, 1, 2],
    )
    namespace, calls, _ = _paged_namespace(modules, ppu=True, builder=builder)
    namespace["_current_builder"] = []
    prepared = Span()
    prepared.shape = (6, 1)
    expected_ids = [0, 0, 0, 1, 2, 2]
    overwritten_lens = [3, 1, 2]

    def upstream_prepare(self, **kwargs):
        assert kwargs["num_decodes"] == 3
        kwargs["decode_lens"][:] = [1] * 6
        return prepared, object(), overwritten_lens, 6, False

    def request_ids(contexts, lengths, flattened):
        assert contexts is prepared and lengths == [3, 1, 2] and flattened
        return expected_ids

    namespace.update(
        _upstream_prepare_decode=upstream_prepare,
        _decode_request_ids=request_ids,
    )
    prepare = function(PATCH, "_prepare_decode_wrapper", namespace)
    paged = function(PATCH, "_ppu_paged_metadata", namespace)

    def upstream_build(self, *args):
        prepare(
            self,
            Span(),
            object(),
            overwritten_lens,
            object(),
            object(),
            3,
            6,
            False,
            3,
            3,
        )
        assert overwritten_lens == [1] * 6
        return paged(prepared, 64, 7)

    namespace["_upstream_build"] = upstream_build
    assert (
        function(PATCH, "_build_wrapper", namespace)(builder, 0, None) == "ppu-metadata"
    )
    assert calls == [(prepared, 64, 7, (1, 32, 128, "int8"), expected_ids)]
    assert builder._sail_decode_request_ids is None


@pytest.mark.parametrize(
    "shape,lengths,flattened,expected",
    [
        ((6, 1), [3, 1, 2], True, [0, 0, 0, 1, 2, 2]),
        ((4, 1), [2, 2], True, [0, 0, 1, 1]),
        ((8, 1), [3, 1, 2, 0], True, [0, 0, 0, 1, 2, 2, -7, -8]),
        ((3, 2), [2, 1, 0], False, [0, 0, 1, -4, -5, -6]),
    ],
)
def test_prebuilt_request_ids_follow_query_ownership(
    shape, lengths, flattened, expected
):
    torch = pytest.importorskip("torch")
    ids = function(PATCH, "_decode_request_ids", {"torch": torch})(
        torch.empty(shape, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        flattened,
    )
    assert ids.tolist() == expected
    assert ids.dtype == torch.int32 and ids.is_contiguous()


def test_draft_refresh_wrapper_keeps_the_builder_visible():
    seen = []
    builder = NS()
    metadata = NS(decode=None)
    namespace = {"_current_builder": []}

    def upstream(self, value):
        seen.append((self, value, list(namespace["_current_builder"])))
        return "refreshed"

    namespace["_upstream_update_draft_decode"] = upstream
    result = function(PATCH, "_update_draft_decode_wrapper", namespace)(
        builder, metadata
    )

    assert result == "refreshed"
    assert seen == [(builder, metadata, [builder])]
    assert namespace["_current_builder"] == []


def test_paged_metadata_derives_legacy_fp8_geometry_from_the_builder(modules):
    builder = NS(indexer_n_head=64, indexer_q_head_dim=128)
    namespace, calls, fallback = _paged_namespace(modules, ppu=True, builder=builder)

    result = function(PATCH, "_ppu_paged_metadata", namespace)(Span(), 64, 7)

    assert result == "ppu-metadata"
    assert calls == [(ANY, 64, 7, (1, 64, 128, "fp8"), None)]
    assert not fallback


@pytest.mark.parametrize("shape", [(630, 4), (313, 2)])
def test_upstream_copies_exact_schedule_shape_and_reuses_buffer(modules, shape):
    class Schedule:
        def __init__(self, shape):
            self.shape = shape
            self.copied = None

        def __getitem__(self, selection):
            return Schedule((min(selection.stop, self.shape[0]), self.shape[1]))

        def __setitem__(self, selection, source):
            assert self.shape == source.shape
            self.copied = source

    initial = Schedule((313, 2))
    builder = NS(
        indexer_n_head=64,
        indexer_q_head_dim=128,
        scheduler_metadata_buffer=initial,
    )
    namespace, _, _ = _paged_namespace(modules, ppu=True, builder=builder)
    metadata = Schedule(shape)
    namespace["_ppu_deep_gemm"].get_paged_mqa_logits_metadata = (
        lambda *args, **kwargs: (metadata)
    )
    metadata.new_empty = Schedule
    paged = function(PATCH, "_ppu_paged_metadata", namespace)
    for _ in range(2):
        produced = paged(Span(), 64, 39)
        view = builder.scheduler_metadata_buffer[: produced.shape[0]]
        view[:] = produced
        assert view.copied is metadata
        if shape == initial.shape:
            assert builder.scheduler_metadata_buffer is initial
        initial = builder.scheduler_metadata_buffer


def test_paged_metadata_rejects_missing_query_geometry(modules):
    namespace, calls, fallback = _paged_namespace(modules, ppu=True)
    namespace["_current_builder"] = [NS(indexer_n_head=0, indexer_q_head_dim=128)]

    with pytest.raises(ValueError, match="query head geometry"):
        function(PATCH, "_ppu_paged_metadata", namespace)(Span(), 64, 7)
    assert not calls
    assert not fallback


@pytest.mark.parametrize("fp4,tag", [(True, "int8"), (False, "fp8")])
def test_paged_metadata_groups_only_weighted_mxfp4_queries(modules, fp4, tag):
    builder = NS(indexer_n_head=64, indexer_q_head_dim=128, indexer_uses_fp4=fp4)
    namespace, calls, _ = _paged_namespace(modules, ppu=True, builder=builder)
    ids = Span()
    function(PATCH, "_ppu_paged_metadata", namespace)(Span(), 64, 7, indices=ids)
    assert calls == [(ANY, 64, 7, (1, 64, 128, tag), ids if fp4 else None)]


def test_paged_metadata_delegates_on_non_ppu(modules):
    builder = NS(indexer_n_head=64, indexer_q_head_dim=128)
    namespace, calls, fallback = _paged_namespace(modules, ppu=False, builder=builder)

    result = function(PATCH, "_ppu_paged_metadata", namespace)(
        Span(), 64, 7, indices=None
    )

    assert result == "upstream-metadata"
    assert not calls
    assert fallback == [(ANY, 64, 7, None)]


@pytest.mark.upstream_source
def test_selected_upstream_producer_preserves_the_same_table(
    upstream_source_root,
):
    import ast

    path = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    node = definition(path, "DeepseekV32IndexerMetadataBuilder.build")
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )

    class CpuValues(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            return CpuValues(value) if isinstance(index, slice) else value

        def max(self):
            return NS(item=lambda: max(self))

    block_table = object()
    common = NS(
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc=object(),
        query_start_loc_cpu=CpuValues([0, 1]),
        seq_lens=object(),
        slot_mapping=object(),
        block_table_tensor=block_table,
        dcp_local_seq_lens=None,
        max_seq_len=17,
        seq_lens_cpu_upper_bound=CpuValues([17]),
    )
    builder = NS(
        compress_ratio=1,
        decode_threshold=1,
        use_flattening=False,
        supports_varlen=False,
        use_pcp=False,
        use_shardq=False,
        fused_mqa_topk_eligible=False,
        max_prefill_buffer_size=128,
        indexer_uses_fp4=False,
        vllm_config=object(),
        dcp_world_size=1,
        tp_size=1,
        _prefill_split_seq_lens=lambda values: values,
        _split_indexer_prefill_chunks=lambda *args, **kwargs: [],
    )
    namespace = {
        "torch": NS(
            diff=lambda values: [values[1] - values[0]],
            bfloat16="bfloat16",
            float32="float32",
        ),
        "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
        "tp_prefill_row_sharding_supported": lambda config, dcp, pcp, tp: False,
        "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 1),
        "DeepseekV32IndexerMetadata": lambda **kwargs: NS(**kwargs),
        "DeepseekV32IndexerPrefillMetadata": lambda chunks, **kwargs: NS(
            chunks=chunks, **kwargs
        ),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    result = namespace["build"](builder, 0, common)
    assert result.block_table is block_table
