# SPDX-License-Identifier: Apache-2.0
"""Upstream KDA delegates in_proj padding to the Linear and preserves it."""

import ast
from types import SimpleNamespace as NS

import pytest

from tests.support.source import ROOT, definition

UPSTREAM = "vllm/models/kimi_k3/nvidia/kda.py"


class ProjectionReady(Exception):
    """Stop the real constructor immediately after its merged projection."""


def compile_class(node, namespace, path):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[node.name]


def check_constructor(path, name, heads):
    calls = []

    class Parent:
        def __init__(self, *args):
            self.cache_config = NS(use_kda_recoverssm=False, use_replayssm=False)
            self.num_spec, self.tp_size, self.hidden_size = 0, 8, 512
            self.quant_config = None

    def merged(*args, **kwargs):
        calls.append((args, kwargs))
        # No weight attribute: the caller must let the Linear own its padding.
        return NS()

    def following_projection(*args, **kwargs):
        raise ProjectionReady

    node = definition(ROOT / path, name)
    node.name, node.decorator_list = "__init__", []
    cls = ast.ClassDef(
        name="KimiK3DeltaAttention",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[node],
        decorator_list=[],
    )
    namespace = {
        "Parent": Parent,
        "divide": lambda n, d: n // d,
        "ModelOptMixedPrecisionConfig": type("Mixed", (), {}),
        "_KimiGDNMergedColumnParallelLinear": merged,
        "ColumnParallelLinear": following_projection,
    }
    attention = compile_class(cls, namespace, path)
    config = NS(
        linear_attn_config={
            "head_dim": 128,
            "num_heads": heads,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        }
    )
    layer = attention.__new__(attention)
    with pytest.raises(ProjectionReady):
        attention.__init__(layer, config, NS(), prefix="model.kda")
    assert len(calls) == 1
    args, kwargs = calls[0]
    padding = 8 if heads == 64 else 0
    assert layer.in_proj_padding == padding
    assert args[0] == 512
    assert args[1] == [128 * heads] * 4 + [128, heads]
    assert kwargs == {
        "replicated_shard_id": 4,
        "tp_size": 8,
        "padding_size": padding,
        "bias": False,
        "quant_config": None,
        "prefix": "model.kda.in_proj_qkvgfab",
    }


@pytest.mark.upstream_source
@pytest.mark.parametrize("heads", [64, 128])
def test_actual_constructor_padding_contract(upstream_source_root, heads):
    check_constructor(
        upstream_source_root / UPSTREAM, "KimiK3DeltaAttention.__init__", heads
    )


class Weight:
    def __init__(self, rows, columns=4, values=None):
        self.rows, self.columns = rows, columns
        self.values = [7] * rows if values is None else values
        self.start = 0
        self.tp_rank = 5
        self.data = self

    def __getitem__(self, index):
        start, stop, step = index.indices(self.rows)
        assert step == 1
        view = Weight(stop - start, self.columns, self.values)
        view.start = self.start + start
        return view

    def numel(self):
        return self.rows * self.columns

    def zero_(self):
        self.values[self.start : self.start + self.rows] = [0] * self.rows


def actual_linear(source_root, padding, failure=False):
    events = []

    class Stacked:
        def __init__(self, *, input_size, output_size, **kwargs):
            self.tp_rank, self.tp_size = 3, 8
            self.weight, self.bias = Weight(output_size // 8, input_size), None

    def attrs(weight, values):
        for key, value in values.items():
            setattr(weight, key, value)

    linear_path = source_root / "vllm/model_executor/layers/linear.py"
    node = definition(linear_path, "MergedColumnParallelLinear")
    node.decorator_list = []
    node.body = [
        n
        for n in node.body
        if getattr(n, "name", None)
        in ("__init__", "_zero_output_padding", "weight_loader", "weight_loader_v2")
    ]
    base = compile_class(
        node,
        dict(
            StackedColumnParallelLinear=Stacked,
            set_weight_attrs=attrs,
            get_tensor_model_parallel_world_size=lambda: 8,
            get_tensor_model_parallel_rank=lambda: 3,
        ),
        linear_path,
    )
    # Stop after the real loader has validated the shard and zeroed the padding.
    # The copied parameter here represents the native load operation.
    zero = base._zero_output_padding

    def load_at_boundary(owner, param):
        zero(owner, param)
        if hasattr(owner, "observed_shard"):
            loader, shard, loaded = owner.observed_shard
            events.append((loader, owner.tp_rank, param.tp_rank, loaded, shard))
            if failure:
                raise ValueError("load failed")
            param.values[:7] = [9] * 7
            raise ProjectionReady

    base._zero_output_padding = load_at_boundary
    base.validate_shard_id = lambda owner, shard, param: None
    namespace = {
        "MergedColumnParallelLinear": base,
        "BlockQuantScaleParameter": type("BlockScale", (), {}),
    }
    path = source_root / UPSTREAM
    cls = compile_class(
        definition(path, "_KimiGDNMergedColumnParallelLinear"), namespace, path
    )
    sizes = [8, 8, 8, 8, 2, 8]
    layer = cls(4, sizes, replicated_shard_id=4, tp_size=8, padding_size=padding)
    assert sizes == [8, 8, 8, 8, 2, 8]
    assert layer.output_sizes == [8, 8, 8, 8, 16, 8] + (
        [8 * padding] if padding else []
    )
    assert layer.weight.values == [7] * 7 + [0] * padding
    if padding:
        assert layer.weight.weight_loader_numel == 7 * 4
    else:
        assert not hasattr(layer.weight, "weight_loader_numel")
    return layer, events


@pytest.mark.upstream_source
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("loader", ["weight_loader", "weight_loader_v2"])
@pytest.mark.parametrize("shard", [0, 4])
def test_actual_padding_survives_both_weight_loaders(
    upstream_source_root, padding, loader, shard
):
    layer, events = actual_linear(upstream_source_root, padding)
    loaded = object()
    layer.observed_shard = ("v2" if loader.endswith("v2") else "v1", shard, loaded)
    with pytest.raises(ProjectionReady):
        getattr(layer, loader)(layer.weight, loaded, shard)
    assert layer.weight.values == [9] * 7 + [0] * padding
    rank = 0 if shard == 4 else 3
    param_rank = 0 if shard == 4 else 5
    assert events == [
        ("v2" if loader.endswith("v2") else "v1", rank, param_rank, loaded, shard)
    ]
    assert (layer.tp_rank, layer.weight.tp_rank) == (3, 5)


@pytest.mark.upstream_source
@pytest.mark.parametrize("loader", ["weight_loader", "weight_loader_v2"])
def test_actual_replicated_loader_restores_ranks_on_error(upstream_source_root, loader):
    layer, events = actual_linear(upstream_source_root, 2, failure=True)
    loaded = object()
    layer.observed_shard = ("v2" if loader.endswith("v2") else "v1", 4, loaded)
    with pytest.raises(ValueError, match="load failed"):
        getattr(layer, loader)(layer.weight, loaded, 4)
    assert events[0][1:3] == (0, 0)
    assert (layer.tp_rank, layer.weight.tp_rank) == (3, 5)
