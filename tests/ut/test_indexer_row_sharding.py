# SPDX-License-Identifier: Apache-2.0
"""真实 ShardQ producer 保留 cache 几何并重编号紧凑查询与 positions。"""

import ast
from types import SimpleNamespace as NS

import pytest


class Values:
    """CPU buffer 边界替身；numpy() 返回可写共享视图，不复制输出。"""

    def __init__(self, values):
        import numpy as np

        self.values = np.asarray(values)

    def __getitem__(self, index):
        if isinstance(index, Values):
            index = index.values
        value = self.values[index]
        return Values(value) if hasattr(value, "shape") and value.shape else value

    def __setitem__(self, index, value):
        self.values[index] = value.values if isinstance(value, Values) else value

    def __floordiv__(self, divisor):
        return Values(self.values // divisor)

    def __add__(self, other):
        return Values(
            self.values + (other.values if isinstance(other, Values) else other)
        )

    def __sub__(self, other):
        return Values(
            self.values - (other.values if isinstance(other, Values) else other)
        )

    def max(self):
        return self.values.max()

    def diff(self):
        import numpy as np

        return Values(np.diff(self.values))

    def numpy(self):
        return self.values


@pytest.fixture
def row_shard_case():
    def make(compress_ratio, shard, upstream_source_root):
        import numpy as np

        from tests.support.source import definition, function

        compressed_calls, shard_calls = [], []
        table, slot_mapping, compressed_slots = object(), object(), object()
        config = NS(
            attention_config=NS(sparse_mla_shard_q_min_tokens=4),
            speculative_config=None,
            model_config=NS(hf_text_config=NS(index_topk=8)),
        )
        shard_path = upstream_source_root / "vllm/v1/attention/ops/shardq.py"
        shard_namespace = {
            "np": np,
            "get_tp_group": lambda: NS(world_size=2, rank_in_group=0),
            "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 8),
        }
        shard_namespace["zigzag_split"] = function(
            shard_path, "zigzag_split", shard_namespace
        )
        local_ids = function(shard_path, "ShardQSplit.local_token_ids", shard_namespace)
        split_node = definition(shard_path, "ShardQSplit")
        assert [
            field.target.id
            for field in split_node.body
            if isinstance(field, ast.AnnAssign)
        ] == [
            "num_decode_tokens",
            "token_ids",
            "num_tokens_per_rank",
        ]

        class Split:
            def __init__(self, num_decode_tokens, token_ids, num_tokens_per_rank):
                self.num_decode_tokens = num_decode_tokens
                self.token_ids, self.num_tokens_per_rank = (
                    token_ids,
                    num_tokens_per_rank,
                )

            local_token_ids = property(local_ids)

        shard_namespace["ShardQSplit"] = Split
        build_split = function(shard_path, "build_shardq_split", shard_namespace)
        common = NS(
            num_reqs=1,
            num_actual_tokens=8,
            query_start_loc=Values([0, 8]),
            query_start_loc_cpu=Values([0, 8]),
            seq_lens=Values([17]),
            seq_lens_cpu_upper_bound=Values([17]),
            slot_mapping=slot_mapping,
            block_table_tensor=table,
            dcp_local_seq_lens=None,
            max_seq_len=17,
        )
        builder = NS(
            compress_ratio=compress_ratio,
            use_pcp=False,
            pcp_world_size=1,
            dcp_world_size=1,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
            compressed_cp_interleave_size=1,
            kv_cache_spec=NS(num_states=256),
            compressed_slot_mapping_buffer=compressed_slots,
            decode_threshold=1,
            use_flattening=False,
            supports_varlen=False,
            max_prefill_buffer_size=128,
            indexer_uses_fp4=True,
            vllm_config=config,
            tp_size=2,
            use_shardq=shard,
            fused_mqa_topk_eligible=False,
            shardq_gather_indices=NS(cpu=Values(np.zeros(8, dtype=np.int64))),
            shardq_query_start_loc=NS(cpu=Values(np.zeros(2, dtype=np.int32))),
            _prefill_split_seq_lens=lambda value: value,
        )
        builder.shardq_gather_indices.copy_to_gpu = (
            lambda count: builder.shardq_gather_indices.cpu[:count]
        )
        builder.shardq_query_start_loc.copy_to_gpu = (
            lambda count: builder.shardq_query_start_loc.cpu[:count]
        )

        def chunk_specs(seq_lens, query_lens, *args, **kwargs):
            assert seq_lens.numpy().tolist() == [17 // compress_ratio]
            assert query_lens.numpy().tolist() == [4 if shard else 8]
            assert kwargs == dict(request_offset=0, fused_mqa_topk_eligible=False)
            return [(slice(0, 1), slice(0, 4 if shard else 8))]

        builder._split_indexer_prefill_chunks = chunk_specs

        def compressed(*args, **kwargs):
            compressed_calls.append((args, kwargs))
            return compressed_slots

        def split(*args):
            shard_calls.append(args)
            assert args == (common, config)
            return build_split(*args)

        def searchsorted(sorted_values, values, right=False, out_int32=False, out=None):
            result = np.searchsorted(
                sorted_values.numpy(), values.numpy(), side="right" if right else "left"
            )
            if out is not None:
                out.values[:] = result
                return out
            return Values(result)

        def chunk_metadata(*args, **kwargs):
            assert args[2].numpy().tolist() == [0, 4 if shard else 8]
            assert args[3].numpy().tolist() == [0, 4 if shard else 8]
            assert args[5].numpy().tolist() == [17 // compress_ratio]
            assert args[7] is table
            positions = kwargs["positions"]
            assert (None if positions is None else positions.numpy().tolist()) == (
                [9, 10, 15, 16] if shard else None
            )
            return NS(block_table=args[7])

        namespace = {
            "get_compressed_slot_mapping": compressed,
            "torch": NS(
                diff=lambda value: value.diff(),
                searchsorted=searchsorted,
                bfloat16="bf16",
                float32="f32",
            ),
            "envs": NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1),
            "split_decodes_and_prefills": lambda *args, **kwargs: (0, 1, 0, 8),
            "np": np,
            "build_prefill_chunk_metadata": chunk_metadata,
            "build_shardq_split": split,
            "DeepseekV32IndexerMetadata": lambda **kwargs: NS(**kwargs),
            "DeepseekV32IndexerPrefillMetadata": lambda chunks, **kwargs: NS(
                chunks=chunks, **kwargs
            ),
        }
        return builder, common, namespace, compressed_calls, shard_calls

    return make


def check_result(body, row_shard_case, compress_ratio, shard, upstream_source_root):
    builder, common, namespace, compressed_calls, shard_calls = row_shard_case(
        compress_ratio, shard, upstream_source_root
    )
    assert not hasattr(builder, "kernel_block_size")
    result = body(namespace)(builder, 0, common)
    assert result.block_table is common.block_table_tensor
    assert result.prefill.chunks[0].block_table is common.block_table_tensor
    indices = result.prefill.shardq_gather_indices
    assert (indices.numpy().tolist() if indices is not None else None) == (
        [0, 1, 6, 7] if shard else None
    )
    assert result.num_prefill_tokens == 8
    assert result.prefill.max_prefill_seq_len == 17
    assert len(shard_calls) == int(shard)
    assert len(compressed_calls) == int(compress_ratio > 1)
    if compressed_calls:
        args, kwargs = compressed_calls[0]
        assert args[4] is common.block_table_tensor
        assert (
            result.slot_mapping
            is kwargs["out"]
            is builder.compressed_slot_mapping_buffer
        )
    else:
        assert result.slot_mapping is common.slot_mapping


@pytest.mark.upstream_source
@pytest.mark.parametrize("compress_ratio", [1, 8])
@pytest.mark.parametrize("shard", [False, True])
def test_actual_upstream_producer_has_same_geometry(
    upstream_source_root, row_shard_case, compress_ratio, shard
):
    import ast

    from tests.support.source import definition

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

    def load(namespace):
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        return namespace["build"]

    check_result(load, row_shard_case, compress_ratio, shard, upstream_source_root)
