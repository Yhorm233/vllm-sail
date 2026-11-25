# SPDX-License-Identifier: Apache-2.0
"""MLA profiling reserves workspace and honors the selected output quantizer."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.source import definition, function

PATCH = "vllm_sail/patch/enhancement/attention/mla_attention.py"
UPSTREAM = "vllm/model_executor/layers/attention/mla_attention.py"


class Tensor:
    def __init__(self, shape=(4, 8), *, dtype="bf16", root=None, events=None):
        self.shape = shape
        self.dtype = dtype
        self.device = "host-double"
        self.root = self if root is None else root
        self.events = events

    def __getitem__(self, index):
        first = index[0] if isinstance(index, tuple) else index
        start, stop, step = first.indices(self.shape[0])
        assert step == 1
        return Tensor(
            (stop - start, *self.shape[1:]),
            dtype=self.dtype,
            root=self.root,
            events=self.events,
        )

    def size(self, axis):
        return self.shape[axis]

    def fill_(self, value):
        self.events.append(("fill", self.root, value))
        return self

    def copy_(self, value):
        self.events.append(("copy", self.root, value))
        return self


def _make_case(ppu, quantized):
    events, allocated = [], []
    output = Tensor(dtype="fp8" if quantized else "bf16", events=events)

    def empty(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        result = Tensor(shape, dtype=kwargs["dtype"], events=events)
        allocated.append(result)
        events.append(("empty", result))
        return result

    namespace = {
        "torch": NS(empty=empty),
        "current_platform": NS(is_ppu=lambda: ppu),
        "_detect_output_quant_key": lambda *args: "static" if quantized else None,
    }
    layer = NS(
        num_heads=2,
        v_head_dim=4,
        qk_nope_head_dim=2,
        use_shardq=False,
        chunked_prefill_workspace_size=128,
        impl=NS(reserve_workspace=lambda: events.append(("reserve",))),
    )
    return layer, namespace, output, events, allocated


def check_profile(path, name, ppu, quantized, copied):
    layer, namespace, output, events, allocated = _make_case(ppu, quantized)
    tensor = Tensor()
    result = function(path, name, namespace)(
        layer, tensor, tensor, tensor, tensor, None, output
    )
    assert result is output
    assert [e[0] for e in events][-2:] == ["reserve", "fill"]
    assert events[-1] == ("fill", output, 0)
    assert sum(e[0] == "reserve" for e in events) == 1
    # Keep the existing PPU scratch-allocation workaround. The provider has no
    # such workaround, but both bodies must still reserve backend workspace.
    assert len(allocated) == int(quantized) + int(not (copied and ppu))
    if not (copied and ppu):
        assert allocated[-1].shape == (128, 2, 6)


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_profile_reserves_before_zero_fill(ppu, quantized):
    check_profile(PATCH, "_forward_impl_body", ppu, quantized, True)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_actual_profile_workspace_lifecycle(
    upstream_source_root, copied, ppu, quantized
):
    path = PATCH if copied else upstream_source_root / UPSTREAM
    name = "_forward_impl_body" if copied else "MLAAttention.forward_impl"
    check_profile(path, name, ppu, quantized, copied)


def check_quantization(path, name, aiter):
    layer, namespace, output, events, allocated = _make_case(False, True)
    scale, fp8_data = object(), object()

    def per_tensor(*args):
        events.append(("aiter", *args))

    def fallback(actual, used_scale):
        events.append(("fallback", actual, used_scale))
        return fp8_data, object()

    layer.impl = NS(
        is_sparse=False,
        dcp_world_size=1,
        forward_mha=lambda *args, **kwargs: events.append(("mha", kwargs["output"])),
    )
    layer.kv_cache_dtype = "auto"
    layer.use_pcp = False
    layer.prefill_backend = None
    layer._k_scale = object()
    layer._aiter_output_quant = aiter
    layer._quant_fp8_op = fallback
    namespace.update(
        torch=NS(
            empty=namespace["torch"].empty,
            ops=NS(vllm=NS(rocm_aiter_per_tensor_quant=per_tensor)),
        ),
        is_quantized_kv_cache=lambda dtype: False,
        kNvfp4Dynamic="nvfp4",
        kFp8Dynamic128Sym="fp8-128",
        kFp8Dynamic64Sym="fp8-64",
        kFp8StaticTensorSym="static",
    )
    metadata = NS(
        num_actual_tokens=2, num_decodes=0, num_prefills=1, num_decode_tokens=0
    )
    q = Tensor((4, 2, 4))
    result = function(path, name, namespace)(
        layer, q, q, q, q, metadata, output, output_scale=scale
    )
    assert result is output
    assert [e[0] for e in events] == (
        ["empty", "mha", "aiter"] if aiter else ["empty", "mha", "fallback", "copy"]
    )
    quant_event = events[2]
    actual = quant_event[2] if aiter else quant_event[1]
    assert actual.root is allocated[0] and actual.shape == (2, 8)
    if aiter:
        destination = quant_event[1]
        assert destination.root is output and destination.shape == (2, 8)
        assert quant_event[3:] == (scale, False)
    else:
        assert quant_event[2] is scale
        assert events[3] == ("copy", output, fp8_data)


@pytest.mark.parametrize("aiter", [False, True])
def test_selected_per_tensor_quantizer_receives_actual_rows(aiter):
    check_quantization(PATCH, "_forward_impl_body", aiter)


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("aiter", [False, True])
def test_actual_per_tensor_quantizer_contract(upstream_source_root, copied, aiter):
    path = PATCH if copied else upstream_source_root / UPSTREAM
    name = "_forward_impl_body" if copied else "MLAAttention.forward_impl"
    check_quantization(path, name, aiter)


@pytest.mark.upstream_source
def test_complete_forward_body_preserves_only_ppu_profile_region(upstream_source_root):
    copied = definition(
        Path(__file__).resolve().parents[2] / PATCH, "_forward_impl_body"
    )
    original = definition(upstream_source_root / UPSTREAM, "MLAAttention.forward_impl")
    copied.name = original.name
    copied.decorator_list = original.decorator_list = []
    profile = next(
        node
        for node in copied.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "attn_metadata is None"
    )
    assert isinstance(profile.body[0], ast.If)
    assert ast.unparse(profile.body[0].test) == "not current_platform.is_ppu()"
    profile.body[:1] = profile.body[0].body
    assert ast.dump(copied, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


@pytest.fixture
def sparse_forward_case(upstream_source_root):
    from types import MethodType

    torch = pytest.importorskip("torch")
    output = torch.full((5, 8), -99.0)
    q = torch.arange(4 * 2 * 4, dtype=torch.float32).reshape(4, 2, 4)
    layer = NS(
        num_heads=2,
        kv_b_proj=NS(),
        W_UV_dcp=None,
        kv_lora_rank=3,
        is_aiter_fused_decode_query_enabled=False,
        v_head_dim=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        q_pad_num_heads=None,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
        is_amx_bmm_enabled=False,
        W_UK_T=torch.ones(2, 2, 3),
        kv_cache_dtype="auto",
        use_shardq=False,
        use_pcp=False,
        prefill_backend=None,
        _use_sparse_mha=lambda meta: False,
    )
    namespace = {
        "torch": torch,
        "is_quantized_kv_cache": lambda dtype: False,
        "_detect_output_quant_key": lambda *args: None,
    }
    metadata = NS(
        num_actual_tokens=4,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        seq_lens=torch.tensor([10, 20]),
        query_start_loc=torch.tensor([0, 1, 4]),
        decode=NS(seq_lens=torch.tensor([10])),
    )
    projected = []

    def project(value, *, out):
        projected.append(value.clone())
        out.copy_(value.reshape(out.shape))

    for helper in (
        "_fused_decode_query_heads",
        "_prepare_mqa_query",
        "_apply_lora_projection",
    ):
        setattr(
            layer,
            helper,
            MethodType(
                function(
                    upstream_source_root / UPSTREAM, f"MLAAttention.{helper}", namespace
                ),
                layer,
            ),
        )
    layer._v_up_proj = project
    return torch, layer, namespace, metadata, q, output, projected


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("decode_rows", [0, 1])
def test_real_shardq_forward_routes_metadata_and_restores_output(
    upstream_source_root,
    sparse_forward_case,
    copied,
    decode_rows,
):
    torch, layer, namespace, metadata, q, output, projected = sparse_forward_case
    events = []
    local_metadata = object()
    shardq = NS(attn_metadata=local_metadata, gathered_context=None)
    metadata.shardq = shardq
    metadata.num_decode_tokens = decode_rows
    metadata.num_decodes = int(decode_rows > 0)
    decode_result = torch.full((decode_rows, 2, 4), 11.0)
    local_result = torch.full((2, 4, 4), 22.0)
    restored = torch.full((4, 2, 4), 22.0)
    restored[:decode_rows] = decode_result

    def shard(query, split):
        assert split is shardq and isinstance(query, tuple)
        assert query[0].shape == (4, 2, 3) and query[1].shape == (4, 2, 2)
        events.append("shard")
        return torch.empty(decode_rows, 2, 5), torch.empty(2, 4, 5)

    def attend(query, cache, used_meta, owner):
        assert cache is kv_cache and owner is layer
        events.append(used_meta)
        if used_meta is metadata:
            assert decode_rows > 0 and query.shape == (decode_rows, 2, 5)
            return decode_result, None
        assert used_meta is local_metadata and query.shape == (2, 4, 5)
        return local_result, None

    def unshard(value, split, decoded):
        assert value is local_result and split is shardq
        assert decoded is (decode_result if decode_rows else None)
        events.append("unshard")
        return restored

    layer.impl = NS(is_sparse=True, dcp_world_size=1, forward_mqa=attend)
    namespace.update(shard_q=shard, unshard_out=unshard)
    kv_cache = object()
    path = PATCH if copied else upstream_source_root / UPSTREAM
    name = "_forward_impl_body" if copied else "MLAAttention.forward_impl"
    result = function(path, name, namespace)(layer, q, q, q, kv_cache, metadata, output)
    assert result is output
    assert events == (
        ["shard", metadata, local_metadata, "unshard"]
        if decode_rows
        else ["shard", local_metadata, "unshard"]
    )
    assert torch.equal(projected[0], restored)
    assert torch.equal(output[:4], restored.reshape(4, 8))
    assert output[4].tolist() == [-99.0] * 8


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize(
    "pcp,merge_rows", [(False, 4), (True, 4), (True, 1), (True, 0)]
)
def test_real_dcp_merge_uses_only_lse_prefix_and_keeps_prefill_rows(
    upstream_source_root,
    sparse_forward_case,
    copied,
    pcp,
    merge_rows,
):
    torch, layer, namespace, metadata, q, output, projected = sparse_forward_case
    attended = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4)
    lse = torch.zeros(merge_rows, 2)
    events = []
    kv_cache = object()
    layer.use_pcp = pcp
    layer.impl = NS(is_sparse=True, dcp_world_size=2, pcp_world_size=2)

    def attend(query, cache, used_meta, owner):
        assert cache is kv_cache and used_meta is metadata and owner is layer
        events.append("attend")
        return attended.clone(), lse

    def gather(query):
        assert query.shape == (4, 2, 5)
        events.append("gather")
        return query

    def combine(value, used_lse, *, seq_lens, query_start_loc):
        assert used_lse is lse and value.shape[0] == merge_rows
        assert torch.equal(value, attended[:merge_rows])
        assert seq_lens is (metadata.decode.seq_lens if pcp else metadata.seq_lens)
        assert torch.equal(
            query_start_loc,
            metadata.query_start_loc[:2] if pcp else metadata.query_start_loc,
        )
        events.append("combine")
        return value + 100

    def finalize(value, heads):
        assert heads == 2
        events.append("finalize")
        return value + 1000

    layer.impl.forward_mqa = attend
    layer.dcp_manager = NS(query_gather=gather, combine=combine)
    namespace.update(cast=lambda kind, value: value, finalize_mla_pcp_decode=finalize)
    path = PATCH if copied else upstream_source_root / UPSTREAM
    name = "_forward_impl_body" if copied else "MLAAttention.forward_impl"
    result = function(path, name, namespace)(layer, q, q, q, kv_cache, metadata, output)
    expected = attended.clone()
    expected[:merge_rows] += 100
    if pcp:
        expected += 1000
    assert result is output and torch.equal(projected[0], expected)
    assert torch.equal(output[:4], expected.reshape(4, 8))
    assert output[4].tolist() == ([0.0] * 8 if pcp else [-99.0] * 8)
    assert events == ([] if pcp else ["gather"]) + ["attend"] + (
        ["combine"] if merge_rows else []
    ) + (["finalize"] if pcp else [])
