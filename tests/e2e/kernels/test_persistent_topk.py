# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent top-k overflow and graph regressions adapted from upstream.

The reference checks follow tests/kernels/test_top_k_per_row.py, including
distinct valid indices and exact selected values when scores contain ties.
"""

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def topk_runtime():
    import torch

    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    return torch


def assert_topk(torch, logits, indices, lengths):
    logits, indices, lengths = logits.cpu(), indices.cpu(), lengths.cpu().flatten()
    for row, length in enumerate(lengths.tolist()):
        count = min(max(length, 0), indices.shape[1])
        selected = indices[row, :count].long()
        assert bool(((selected >= 0) & (selected < length)).all())
        assert selected.unique().numel() == count
        assert bool((indices[row, count:] == -1).all())
        actual = logits[row, selected].sort().values
        expected = logits[row, : max(length, 0)].topk(count).values.sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("experts", [32, 128])
@pytest.mark.parametrize("renormalize", [False, True])
def test_softmax_topk_graph_replay_without_programmatic_dependency(
    topk_runtime, experts, renormalize
):
    """PPU SM80/89 的普通 graph launch 保持 top-k 数值和重放输入更新。"""
    torch = topk_runtime
    logits = torch.linspace(-2, 2, experts, device="cuda").repeat(5, 1)
    weights = torch.empty((5, 8), dtype=torch.float32, device="cuda")
    indices = torch.empty((5, 8), dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty_like(indices)

    def run():
        torch.ops._moe_C.topk_softmax(
            weights, indices, token_expert_indices, logits, renormalize, None, None
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for sign in (1, -1, 1):
        logits.copy_(sign * torch.linspace(-2, 2, experts, device="cuda"))
        graph.replay()
        expected_weights, expected_ids = logits.softmax(-1).topk(8, dim=-1)
        if renormalize:
            expected_weights /= expected_weights.sum(-1, keepdim=True)
        torch.testing.assert_close(indices.long(), expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("experts", [32, 128, 256])
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_grouped_topk_padding_preserves_unpadded_routing(
    topk_runtime, experts, groups, scoring_func
):
    """新增可选 mask 不改变有效 token 路由，padding 输出 -1 和零权重。"""
    torch = topk_runtime
    scores = torch.linspace(-1, 1, experts, device="cuda").repeat(4, 1)
    bias = torch.zeros(experts, device="cuda")
    args = (scores, groups, min(groups, 2), 8, True, 2.0, bias, scoring_func)
    original_weights, original_ids = torch.ops._moe_C.grouped_topk(*args)
    padding = torch.tensor([False, True, False, True], device="cuda")
    weights, ids = torch.ops._moe_C.grouped_topk(*args, padding)
    torch.testing.assert_close(
        weights[~padding], original_weights[~padding], rtol=0, atol=0
    )
    torch.testing.assert_close(ids[~padding], original_ids[~padding], rtol=0, atol=0)
    assert bool((weights[padding] == 0).all())
    assert bool((ids[padding] == -1).all())


@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("rows", [1, 64, 128])
@pytest.mark.parametrize(
    "scores", ["positive", "negative", "ties", "equal", "capacity"]
)
@pytest.mark.parametrize("length", [32768, 32769])
def test_overflow_preserves_exact_topk(topk_runtime, top_k, rows, scores, length):
    torch = topk_runtime
    width = 131072
    row = 1 + torch.arange(length, dtype=torch.float32) * (0.01 / length)
    if scores == "negative":
        row -= 2.01
    elif scores == "ties":
        row = row.bfloat16().float()
    elif scores == "equal":
        row.fill_(1)
    elif scores == "capacity":
        row.fill_(-2)
        capacity = max(top_k, 1024) + 1
        row[:capacity] = 1 + torch.arange(capacity) * (0.01 / capacity)
    logits = torch.full((rows, width), float("inf"), device="cuda")
    logits[:, :length] = row.to("cuda")
    lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    indices = torch.full((rows, top_k), -777, dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, top_k, width)
    torch.cuda.synchronize()
    assert_topk(torch, logits, indices, lengths)


@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("offset", [1, 2])
def test_aligned_stride_with_misaligned_base(topk_runtime, top_k, rows, offset):
    torch = topk_runtime
    width = 32768
    storage = torch.empty((rows, width + 8), device="cuda")
    logits = storage[:, offset : width + offset]
    assert logits.stride(0) % 4 == 0
    assert logits.data_ptr() % 16 == offset * 4
    logits.copy_(1 + torch.arange(width, device="cuda") * (0.01 / width))
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, top_k, width)
    torch.cuda.synchronize()
    assert_topk(torch, logits, indices, lengths)


@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("rows", [1, 64, 128])
def test_bf16_dtype_and_shifted_base_contract(topk_runtime, top_k, offset, rows):
    """BF16 使用真实两字节输入；共享内存不足时必须显式拒绝。"""
    torch = topk_runtime
    width = 32769
    storage = torch.empty((rows, width + 8), dtype=torch.bfloat16, device="cuda")
    logits = storage[:, offset : width + offset]
    values = torch.arange(width, dtype=torch.float32, device="cuda") / width
    logits.copy_(values)
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    props = torch.cuda.get_device_properties(logits.device)
    if props.shared_memory_per_block_optin < 128 * 1024:
        with pytest.raises(RuntimeError, match="requires 128 KiB shared memory"):
            torch.ops._C.persistent_topk(
                logits, lengths, indices, workspace, top_k, width
            )
        return
    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, top_k, width)
    torch.cuda.synchronize()
    assert_topk(torch, logits, indices, lengths)


def test_bf16_strided_columns_are_rejected(topk_runtime):
    torch = topk_runtime
    logits = torch.empty((1, 2048), dtype=torch.bfloat16, device="cuda")[:, ::2]
    lengths = torch.full((1,), 1024, dtype=torch.int32, device="cuda")
    indices = torch.empty((1, 512), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    with pytest.raises(
        RuntimeError, match="bfloat16 logits columns must be contiguous"
    ):
        torch.ops._C.persistent_topk(logits, lengths, indices, workspace, 512, 1024)


@pytest.mark.parametrize("top_k", [512, 2048])
def test_misaligned_mixed_lengths_and_changed_graph_replay(topk_runtime, top_k):
    torch = topk_runtime
    rows, width = 128, 65539
    storage = torch.empty((rows, width + 8), device="cuda")
    logits = storage[:, 1 : width + 1]
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    values = torch.arange(width, dtype=torch.float32, device="cuda")
    logits.copy_(values)

    def run():
        torch.ops._C.persistent_topk(
            logits, lengths.view(-1, 4), indices, workspace, top_k, width
        )

    run()
    torch.cuda.synchronize()
    assert_topk(torch, logits, indices, lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    bounds = torch.tensor(
        [0, 1, top_k - 1, top_k, top_k + 1, 32767, 32768, 32769, width],
        dtype=torch.int32,
        device="cuda",
    )
    for step in range(2):
        lengths.copy_(bounds[(torch.arange(rows, device="cuda") + step) % len(bounds)])
        logits.copy_(values if step == 0 else values.flip(0))
        logits.masked_fill_(values[None, :] >= lengths[:, None], float("inf"))
        indices.fill_(-777)
        graph.replay()
        torch.cuda.synchronize()
        assert_topk(torch, logits, indices, lengths)
