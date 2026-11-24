# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 channelwise loading contracts against the selected upstream source.

These tests exercise upstream slicing with a row-only tensor double. They do
not establish FP8/INT8 kernel correctness or checkpoint inference quality.
"""

import ast

import pytest

from tests.support.source import definition, function


class Rows:
    """A two-dimensional tensor view with distinct values in every output row."""

    def __init__(self, rows, *, start=0, size=None):
        self.rows = rows
        self.start = start
        self.size = len(rows) if size is None else size
        self.shape = (self.size, 1)

    def narrow(self, dim, start, size):
        assert dim == 0 and 0 <= start <= start + size <= self.size
        return Rows(self.rows, start=self.start + start, size=size)

    def copy_(self, other):
        assert self.shape == other.shape
        self.rows[self.start : self.start + self.size] = other.rows[
            other.start : other.start + other.size
        ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
def test_qkv_and_channel_scales_share_sharding(upstream_source_root, tp_size):
    """Q shards, K/V/index-Q replicate in pairs at TP8, index-K always replicates.

    Both W8A8 formats use ModelWeightParameter and ChannelQuantScaleParameter,
    which inherit this same upstream column loader. Distinct scale values catch
    accidental reuse of a Q/K scale or the first rank's index-Q scale.
    """
    linear = upstream_source_root / "vllm/model_executor/layers/linear.py"
    parameter = upstream_source_root / "vllm/model_executor/parameter.py"
    load_column = function(
        parameter,
        "_ColumnvLLMParameter.load_qkv_weight",
        {
            "PackedColumnParameter": type("PackedColumn", (), {}),
            "PackedvLLMParameter": type("Packed", (), {}),
            "copy_weight_": lambda dst, src: dst.copy_(src),
        },
    )
    parameter_cls = type("ColumnParameter", (), {"load_qkv_weight": load_column})

    class Column:
        def __init__(self, **kwargs):
            self.tp_size = tp_size
            self.output_size_per_partition = kwargs["output_size"] // tp_size
            self.quant_config = kwargs["quant_config"]

        def validate_shard_id(self, shard_id, param):
            assert shard_id in self.shard_ids

    node = definition(linear, "MinimaxM3QKVParallelLinearWithIndexer")
    namespace = {
        "QKVParallelLinear": Column,
        "StackedColumnParallelLinear": Column,
        "get_tensor_model_parallel_world_size": lambda: tp_size,
        "divide": lambda a, b: a // b,
        "BlockQuantScaleParameter": type("BlockScale", (), {}),
    }
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(linear), "exec"), namespace)
    quant = object()
    layer = namespace[node.name](6144, 128, 64, 4, 4, 128, quant_config=quant)
    assert layer.quant_config is quant
    for rank in range(tp_size):
        for multiplier in (1, 0.001):  # weight rows and FP32 channel-scale rows
            param = parameter_cls()
            param.tp_rank, param.output_dim = rank, 0
            param.data = Rows([None] * layer.output_size_per_partition)
            expected = []
            for shard, total_rows in zip(
                layer.shard_ids, (8192, 512, 512, 512, 128), strict=False
            ):
                values = [
                    (10000 * (1 + layer.shard_ids.index(shard)) + i) * multiplier
                    for i in range(total_rows)
                ]
                layer.weight_loader_v2(param, Rows(values), shard)
                if shard == "q":
                    width, start = 8192 // tp_size, rank * (8192 // tp_size)
                elif shard == "index_k":
                    width, start = 128, 0
                else:
                    width = max(128, 512 // tp_size)
                    start = (rank // max(1, tp_size // 4)) * width
                expected.extend(values[start : start + width])
            assert param.data.rows == expected
