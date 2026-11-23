# SPDX-License-Identifier: Apache-2.0
"""Compressed DSA logits retain request-local masks and top-k coordinates."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function

INDEXER = "vllm_sail/attention/ops/mla_sparse.py"


@pytest.mark.parametrize("policy", ["mxfp4", "int8", "fp8"])
@pytest.mark.parametrize("model_bound,capacity", [(7, 7), (300, 205)])
def test_prefill_passes_window_bound_and_masks_relative_columns(
    policy, model_bound, capacity
):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("optional torch lacks FP8 dtype")
    fp4 = policy == "mxfp4"
    q_dtype = (
        torch.uint8
        if fp4
        else (torch.int8 if policy == "int8" else torch.float8_e4m3fn)
    )
    chunk = NS(
        total_seq_lens=205,
        skip_kv_gather=True,
        token_start=0,
        token_end=2,
        cu_seqlen_ks=torch.tensor([0, 100], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([3, 105], dtype=torch.int32),
    )
    metadata = NS(
        slot_mapping=torch.arange(2),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        prefill=NS(chunks=[chunk], shardq_gather_indices=None),
    )
    weights = torch.ones((2, 32), dtype=torch.bfloat16 if fp4 else torch.float32)
    calls = []
    masks = []

    def logits_api(q, kv, passed_weights, starts, ends, **kwargs):
        calls.append(kwargs)
        assert passed_weights.data_ptr() == weights.data_ptr()
        assert starts.tolist() == [0, 100] and ends.tolist() == [3, 105]
        logits = torch.arange(capacity).expand(2, -1).clone()
        return logits.to(torch.bfloat16 if fp4 else torch.float32)

    def topk(logits, starts, ends, out, *args):
        assert logits.shape == (2, capacity)
        assert starts.tolist() == [0, 0] and ends.tolist() == [3, 5]
        masked = logits.masked_fill(
            torch.arange(capacity)[None, :] >= ends[:, None], -float("inf")
        )
        out.copy_(masked.topk(2, dim=-1).indices)

    def filter_candidates(logits, starts, ends, *args):
        masks.append((starts.tolist(), ends.tolist()))

    class Torch:
        ops = NS(_C=NS(top_k_per_row_prefill=topk))

        def __getattr__(self, name):
            return getattr(torch, name)

    workspace = NS(
        get_simultaneous=lambda *specs: [
            torch.empty(shape, dtype=dtype) for shape, dtype in specs
        ]
    )
    namespace = {
        "torch": Torch(),
        "_resolve_layer_name": lambda name: name,
        "get_forward_context": lambda: NS(attn_metadata={"cache": metadata}),
        "DeepseekV32IndexerMetadata": NS,
        "current_workspace_manager": lambda: workspace,
        "is_deep_gemm_supported": lambda: True,
        "fp8_fp4_mqa_logits": logits_api,
        "int8_mqa_logits": logits_api,
        "fp8_mqa_logits": logits_api,
        "sail_ops": NS(top_k_per_row_prefill_bf16=topk),
        "_filter_candidates": filter_candidates,
        "_use_deep_select": lambda *args: False,
        "warn_deep_select_fallback": lambda: None,
    }
    namespace["_gather_workspace_shapes"] = function(
        INDEXER, "_gather_workspace_shapes", {"torch": torch, "MXFP4_BLOCK_SIZE": 32}
    )
    fn = function(INDEXER, "ppu_sparse_attn_indexer", namespace)
    out = torch.full((2, 2), -1, dtype=torch.int32)
    result = fn(
        torch.empty((2, 1)),
        "cache",
        None,
        torch.empty((2, 32, 64 if fp4 else 128), dtype=q_dtype),
        torch.empty((2, 32), dtype=torch.int32) if fp4 else None,
        None,
        weights,
        128,
        "ue8m0",
        2,
        128,
        model_bound,
        205,
        out,
        True,
        use_fp4_cache=fp4,
        candidate_blocks=torch.empty((2, 1)),
        candidate_block_size=64,
    )
    assert result is out
    assert calls[0]["max_seqlen_k"] == capacity
    assert masks == [([0, 0], [3, 5])]
    assert out.tolist() == [[2, 1], [4, 3]]


@pytest.mark.parametrize("policy", ["int8", "fp8", "mxfp4", "kpool"])
@pytest.mark.parametrize("shard", [False, True])
def test_compact_queries_preserve_cache_stream_and_chunk_output(policy, shard):
    """完整 PPU op 的查询行序；张量与 native kernel 仅作为严格边界替身。"""
    calls = []

    class Rows:
        def __init__(self, values, dtype="i8", *, root=None):
            self.values = list(values)
            self.dtype, self.device = dtype, "host-double"
            self.shape = (len(self.values), 2)
            self.root = self if root is None else root

        def __getitem__(self, index):
            index = index[0] if isinstance(index, tuple) else index
            values = (
                [self.values[i] for i in index.values]
                if isinstance(index, Rows)
                else self.values[index]
            )
            return Rows(values, self.dtype, root=self.root)

        def __setitem__(self, index, value):
            calls.append(("write", self.root, index, value))

        def view(self, dtype):
            return self

        def squeeze(self, dim):
            return self

        def to(self, dtype):
            return self

        def __sub__(self, other):
            return Rows(
                [a - b for a, b in zip(self.values, other.values, strict=False)]
            )

        def __add__(self, increment):
            return Rows([value + increment for value in self.values])

        def clamp_min(self, minimum):
            return Rows([max(value, minimum) for value in self.values])

        def stride(self, axis):
            return 2 if axis == 0 else 1

    fp4, kpool = policy == "mxfp4", policy == "kpool"
    indices = Rows([0, 2, 4, 6, 7]) if shard else None
    compact = indices.values if shard else list(range(8))
    query = Rows(range(8), "u8" if fp4 else ("fp8" if policy == "fp8" else "i8"))
    scales = Rows(range(20, 28)) if fp4 else None
    weights, positions, key = Rows(range(40, 48)), Rows(range(100, 108)), Rows(range(8))
    table, cu, cache = object(), object(), object()
    chunks = [
        NS(
            token_start=start,
            token_end=end,
            total_seq_lens=10,
            skip_kv_gather=start != 1,
            block_table=table,
            cu_seq_lens=cu,
            cu_seqlen_ks=Rows([0] * (end - start)),
            cu_seqlen_ke=Rows([8] * (end - start)),
        )
        for start, end in [(1, 3), (3, len(compact))]
    ]
    meta = NS(
        slot_mapping=Rows(range(8)),
        num_decode_tokens=1,
        num_prefills=1,
        num_decodes=0,
        prefill=NS(
            chunks=chunks, max_prefill_seq_len=108, shardq_gather_indices=indices
        ),
    )
    output = Rows(range(9))

    def insert(actual, actual_cache, slots, block):
        assert actual.values == list(range(8)) and actual.root is key
        assert actual_cache is cache and slots is meta.slot_mapping and block == 128
        calls.append(("insert",))

    def insert_fp8(actual, actual_cache, slots, block, scale_fmt):
        assert scale_fmt == "ue8m0"
        insert(actual, actual_cache, slots, block)

    def gather(actual_cache, values, used_scales, actual_table, actual_cu):
        assert actual_cache is cache and actual_table is table and actual_cu is cu
        calls.append(("gather",))

    def logits(q, kv, used_weights, starts, ends, **kwargs):
        q, sf = q if isinstance(q, tuple) else (q, None)
        row_ids = q.values
        assert used_weights.values == [40 + i for i in row_ids]
        if fp4:
            assert sf.values == [20 + i for i in row_ids]
        else:
            assert sf is None
        calls.append(("logits", row_ids))
        return Rows(row_ids, "bf16" if fp4 else "f32")

    def topk(scores, starts, ends, dst, rows, *args):
        assert rows == len(scores.values)
        if not kpool:
            assert dst.root is output
        calls.append(("topk", scores.values))

    def expand(pool_ids, seq_lens, pool_size, tokens):
        assert pool_size == 2 and tokens == 4
        calls.append(("expand", seq_lens.values))
        return Rows(seq_lens.values)

    torch = NS(
        int8="i8",
        uint8="u8",
        int32="i32",
        int64="i64",
        float32="f32",
        bfloat16="bf16",
        float8_e4m3fn="fp8",
        zeros_like=lambda value: Rows([0] * len(value.values)),
        empty=lambda shape, **kwargs: Rows(range(shape[0])),
        ops=NS(_C=NS(top_k_per_row_prefill=topk)),
    )
    namespace = {
        "torch": torch,
        "get_forward_context": lambda: NS(attn_metadata={"cache": meta}),
        "_resolve_layer_name": lambda name: name,
        "DeepseekV32IndexerMetadata": NS,
        "current_workspace_manager": lambda: NS(
            get_simultaneous=lambda *specs: [Rows(range(10)), Rows(range(10))]
        ),
        "_gather_workspace_shapes": lambda *args: (object(), object()),
        "indexer_k_quant_and_cache_int8": insert,
        "ops": NS(
            indexer_k_quant_and_cache=insert_fp8,
            cp_gather_indexer_k_quant_cache=gather,
        ),
        "is_deep_gemm_supported": lambda: True,
        "int8_mqa_logits": logits,
        "fp8_mqa_logits": logits,
        "fp8_fp4_mqa_logits": logits,
        "_use_deep_select": lambda *args: False,
        "warn_deep_select_fallback": lambda: None,
        "sail_ops": NS(top_k_per_row_prefill_bf16=topk),
        "_shared": lambda: NS(),
        "_expand_pools": expand,
    }
    path = "vllm_sail/attention/ops/mla_sparse_kpool.py" if kpool else INDEXER
    name = "ppu_sparse_attn_indexer_kpool" if kpool else "ppu_sparse_attn_indexer"
    extra = dict(index_kpool=2, positions=positions) if kpool else {}
    result = function(path, name, namespace)(
        Rows(range(9)),
        "cache",
        cache,
        query,
        scales,
        key,
        weights,
        128,
        "ue8m0",
        4,
        128,
        10,
        10,
        output,
        fp4,
        use_fp4_cache=fp4,
        **extra,
    )
    assert result is output
    assert [event for event in calls if event[0] == "insert"] == (
        [] if fp4 else [("insert",)]
    )
    assert [event for event in calls if event[0] == "gather"] == [("gather",)]
    assert [event[1] for event in calls if event[0] == "logits"] == [
        compact[1:3],
        compact[3:],
    ]
    assert [event[1] for event in calls if event[0] == "topk"] == [
        compact[1:3],
        compact[3:],
    ]
    writes = [event for event in calls if event[0] == "write"]
    assert writes[0] == ("write", output, slice(None, 9), -1)
    if kpool:
        assert [event[1] for event in calls if event[0] == "expand"] == [
            [101 + i for i in compact[1:3]],
            [101 + i for i in compact[3:]],
        ]
        assert [event[2][0] for event in writes[1:]] == [
            slice(1, 3),
            slice(3, len(compact)),
        ]
    else:
        assert len(writes) == 1


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize("pool_size", [1, 2])
def test_kpool_real_chunk_metadata_keeps_full_cache_and_compact_tail_coordinates(
    upstream_source_root,
    shard,
    pool_size,
):
    """真实 split、chunk builder、dataclass 与展开；缓存写入及 native 评分是边界。"""
    from dataclasses import dataclass
    from typing import NamedTuple

    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    query_start = torch.tensor([0, 8, 12], dtype=torch.int32)
    seq_lens = torch.tensor([17, 12], dtype=torch.int32)
    positions = torch.tensor([*range(9, 17), *range(8, 12)], dtype=torch.int32)
    common = NS(
        num_reqs=2, query_start_loc_cpu=query_start, seq_lens_cpu_upper_bound=seq_lens
    )
    config = NS(
        speculative_config=None,
        attention_config=NS(sparse_mla_shard_q_min_tokens=4),
        model_config=NS(hf_text_config=NS(index_topk=4)),
    )
    split_namespace = dict(
        __name__=__name__,
        np=np,
        NamedTuple=NamedTuple,
        get_tp_group=lambda: NS(world_size=2, rank_in_group=0),
        split_decodes_and_prefills=lambda *a, **kw: (0, 2, 0, 12),
    )
    shard_path = upstream_source_root / "vllm/v1/attention/ops/shardq.py"
    for name in ("zigzag_split", "ShardQSplit", "build_shardq_split"):
        function(shard_path, name, split_namespace)
    split = split_namespace["build_shardq_split"](common, config) if shard else None
    indices = (
        torch.tensor(split.local_token_ids, dtype=torch.int64)
        if split is not None
        else None
    )
    row_ids = indices if indices is not None else torch.arange(12)
    assert row_ids.tolist() == ([0, 1, 6, 7, 8, 11] if shard else list(range(12)))
    compact_positions = positions[row_ids]
    compact_query_start = torch.searchsorted(row_ids, query_start, out_int32=True)
    block_table = torch.tensor([[1], [2]], dtype=torch.int32)
    calls = []
    namespace = dict(__name__=__name__, torch=torch)
    indexer_path = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    for name in (
        "DeepseekV32IndexerPrefillChunkMetadata",
        "DeepseekV32IndexerPrefillMetadata",
        "DeepseekV32IndexerMetadata",
    ):
        namespace[name] = dataclass(function(indexer_path, name, namespace))
    starts = torch.tensor([0] * 8 + [17 // pool_size] * 4, dtype=torch.int32)[row_ids]
    ends = starts + (compact_positions + 1) // pool_size

    def metadata_kernel(
        qloc,
        lengths,
        cu,
        local_cu,
        token_to_seq,
        ks,
        ke,
        qs,
        qe,
        rank,
        world,
        interleave,
        **kwargs,
    ):
        # 严格 launcher 边界：只提供预定的输出，不执行或证明 Triton kernel 数值。
        assert qloc.tolist() == ([0, 4, 6] if shard else [0, 8, 12])
        assert lengths.tolist() == [17, 12]
        assert local_cu is cu and (rank, world, interleave) == (0, 1, 1)
        assert kwargs["num_reqs"] == 2 and kwargs["COMPRESS_RATIO"] == pool_size
        assert (qs, qe) == (0, len(row_ids))
        if shard:
            assert torch.equal(kwargs["positions"], compact_positions)
        else:
            assert kwargs["positions"] is None
        ks.copy_(starts)
        ke.copy_(ends)
        token_to_seq.zero_()
        calls.append("metadata")

    namespace["_BUILD_PREFILL_CHUNK_METADATA_KERNEL"] = metadata_kernel
    chunk = function(indexer_path, "build_prefill_chunk_metadata", namespace)(
        0,
        2,
        compact_query_start,
        compact_query_start,
        seq_lens,
        seq_lens // pool_size,
        seq_lens // pool_size,
        block_table,
        pool_size,
        positions=compact_positions if shard else None,
    )
    assert chunk.local_cu_seq_lens is chunk.cu_seq_lens
    assert (
        chunk.local_total_seq_lens
        == chunk.max_local_total_seq_lens
        == chunk.total_seq_lens
    )
    assert chunk.total_seq_lens == 17 // pool_size + 12 // pool_size
    assert (chunk.token_start, chunk.token_end) == (0, len(row_ids))
    slots = torch.arange(12, dtype=torch.int64)
    slots[(positions + 1) % pool_size != 0] = -1
    meta = namespace["DeepseekV32IndexerMetadata"](
        seq_lens,
        17,
        slots,
        0,
        0,
        2,
        12,
        prefill=namespace["DeepseekV32IndexerPrefillMetadata"]([chunk], 17, indices),
    )
    cache, tail = object(), object()
    key = torch.arange(12, dtype=torch.bfloat16)[:, None].expand(12, 128).contiguous()
    gate, ape = torch.ones_like(key), torch.zeros(pool_size, 128)
    query = (
        torch.arange(12, dtype=torch.int8)[:, None, None]
        .expand(12, 32, 128)
        .contiguous()
    )
    weights = (
        torch.arange(40, 52, dtype=torch.float32)[:, None].expand(12, 32).contiguous()
    )
    tail_slots = torch.arange(100, 112, dtype=torch.int64)
    tail_meta = namespace["DeepseekV32IndexerMetadata"](
        seq_lens, 17, tail_slots, 0, 0, 2, 12
    )

    def insert(actual_key, actual_cache, actual_slots, block):
        assert actual_key.data_ptr() == key.data_ptr() and actual_key.shape[0] == 12
        assert actual_cache is cache and actual_slots is slots and block == 128
        calls.append("insert")

    def pool_insert(
        actual_key, actual_gate, actual_ape, actual_cache, actual_slots, size, dim
    ):
        assert torch.equal(actual_key, key) and torch.equal(actual_gate, gate)
        assert (
            actual_ape is ape
            and actual_cache is cache
            and torch.equal(actual_slots, slots)
        )
        assert (size, dim) == (pool_size, 128)
        calls.append("pool-insert")

    def seed(actual_tail, actual_key, actual_gate, actual_slots, size, dim):
        assert (
            actual_tail is tail
            and torch.equal(actual_key, key)
            and torch.equal(actual_gate, gate)
        )
        assert torch.equal(actual_slots, tail_slots) and (size, dim) == (pool_size, 128)
        calls.append("tail-seed")

    def gather(actual_cache, values, scales, table, cu):
        assert actual_cache is cache and torch.equal(table, block_table)
        assert cu is chunk.cu_seq_lens and values.shape[0] == chunk.total_seq_lens
        scales.zero_()
        calls.append("gather")

    def logits(q, kv, used_weights, ks, ke, **kwargs):
        assert torch.equal(q, query[row_ids]) and torch.equal(
            used_weights, weights[row_ids]
        )
        assert ks is chunk.cu_seqlen_ks and ke is chunk.cu_seqlen_ke
        assert kwargs == {"clean_logits": False}
        calls.append("logits")
        return torch.zeros(len(row_ids), chunk.total_seq_lens)

    def topk(scores, ks, ke, dst, rows, *args):
        assert (
            torch.equal(ks, starts) and torch.equal(ke, ends) and rows == len(row_ids)
        )
        dst.copy_(torch.arange(4 // pool_size, dtype=torch.int32).expand(rows, -1))
        calls.append("topk")

    class Torch:
        ops = NS(_C=NS(top_k_per_row_prefill=topk))

        def __getattr__(self, name):
            return getattr(torch, name)

    kpool_path = (
        upstream_source_root / "vllm/models/glm5next/nvidia/ops/kpool_compress.py"
    )
    helpers = dict(torch=torch)
    for name in (
        "history_group_budget_for_topk",
        "expand_pools_to_tokens",
        "append_tail_to_topk",
    ):
        function(kpool_path, name, helpers)
    provider_path = "vllm_sail/attention/ops/mla_sparse_kpool.py"
    namespace.update(
        torch=Torch(),
        _shared=lambda: NS(),
        _resolve_layer_name=lambda name: name,
        get_forward_context=lambda: NS(
            attn_metadata={"cache": meta, "tail": tail_meta}
        ),
        current_workspace_manager=lambda: NS(
            get_simultaneous=lambda *specs: [
                torch.empty(shape, dtype=dtype) for shape, dtype in specs
            ]
        ),
        ops=NS(cp_gather_indexer_k_quant_cache=gather),
        int8_mqa_logits=logits,
        indexer_k_quant_and_cache_int8=insert,
        _kpool_compress_insert_int8=pool_insert,
        _kpool_seed_tail=seed,
        _kpool_ops=lambda: NS(**helpers),
    )
    for name in (
        "_gather_workspace_shapes",
        "_expand_pools",
        "ppu_sparse_attn_indexer_kpool",
    ):
        function(provider_path, name, namespace)
    output = torch.full((12, 4 + pool_size - 1), 99, dtype=torch.int32)
    result = namespace["ppu_sparse_attn_indexer_kpool"](
        torch.zeros(12, 1),
        "cache",
        cache,
        query,
        None,
        key,
        weights,
        128,
        "ue8m0",
        4,
        128,
        17 // pool_size,
        chunk.total_seq_lens,
        output,
        False,
        gate_score=gate,
        compress_ape=ape,
        index_kpool=pool_size,
        positions=positions,
        tail_kv_cache=tail,
        tail_prefix="tail",
    )
    assert result is output
    assert calls == [
        "metadata",
        *(["pool-insert", "tail-seed"] if pool_size > 1 else ["insert"]),
        "gather",
        "logits",
        "topk",
    ]
    expected = [
        [0, 1, 2, 3]
        + ([int(pos) if (int(pos) + 1) % 2 else -1] if pool_size > 1 else [])
        for pos in compact_positions
    ]
    assert output[: len(row_ids)].tolist() == expected
    assert torch.all(output[len(row_ids) :] == -1)


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("decode_tokens", [0, 1, 4])
@pytest.mark.parametrize("shard_decode", [False, True])
def test_real_pcp_slot_producer_and_cache_gather_align_rank_local_inputs(
    upstream_source_root, decode_tokens, shard_decode
):
    """真实 PCP slot producer 和输入 gather；collective 是严格边界，不证明 Kpool PCP 支持。"""
    torch = pytest.importorskip("torch")
    source = upstream_source_root / "vllm"
    local_ids = [0, 1, 6, 7]
    remote_ids = (
        [2, 3, 4, 5]
        if shard_decode or decode_tokens == 0
        else (local_ids if decode_tokens == 4 else [0, 2, 3, 4])
    )
    indices = torch.tensor(local_ids + remote_ids, dtype=torch.int64)
    write_mask = torch.ones(8, dtype=torch.bool)
    if not shard_decode:
        write_mask[4 : 4 + decode_tokens] = False
    producer = function(
        source / "v1/worker/gpu/pcp_manager.py",
        "PCPManager._convert_to_gathered_slot_mappings",
        {"torch": torch},
    )
    owner = NS(
        _padded_gather_idx=indices,
        _gathered_kv_write_mask=write_mask,
        _gathered_kv_slot_mappings=None,
        _pad_slot_id=torch.tensor(-1, dtype=torch.int64),
    )
    global_slots = torch.arange(100, 108, dtype=torch.int64).view(1, 8)
    before_global = global_slots.clone()
    slots = producer(owner, global_slots)[0]
    expected_slots = [100 + i for i in local_ids + remote_ids]
    if not shard_decode:
        expected_slots[4 : 4 + decode_tokens] = [-1] * decode_tokens
    assert slots.tolist() == expected_slots
    assert torch.equal(global_slots, before_global)

    local = torch.tensor(local_ids, dtype=torch.float32)[:, None].expand(4, 128).clone()
    remote = (
        torch.tensor(remote_ids, dtype=torch.float32)[:, None].expand(4, 128).clone()
    )
    before_key, before_slots = local.clone(), slots.clone()
    calls = []

    def gather(values, dim):
        assert dim == 0 and values.is_contiguous()
        start = 0 if shard_decode else decode_tokens
        assert torch.equal(values, local[start:])
        calls.append(start)
        return torch.cat((values, remote[start:]), dim=0)

    namespace = dict(
        torch=torch, get_pcp_group=lambda: NS(world_size=2, all_gather=gather)
    )
    path = source / "v1/attention/ops/pcp.py"
    function(path, "_gather_prefill_cache_inputs", namespace)
    cache_key, cache_slots = function(path, "maybe_gather_indexer_k", namespace)(
        local, slots, decode_tokens, True, pcp_shard_decode_requests=shard_decode
    )
    expected_ids = (
        local_ids + remote_ids
        if shard_decode
        else local_ids + remote_ids[decode_tokens:]
    )
    assert cache_key[:, 0].tolist() == expected_ids
    assert cache_key.shape == (len(expected_ids), 128)
    assert cache_slots.tolist() == [100 + i for i in expected_ids]
    assert calls == (
        []
        if not shard_decode and decode_tokens == 4
        else [0 if shard_decode else decode_tokens]
    )
    assert torch.equal(local, before_key) and torch.equal(slots, before_slots)
