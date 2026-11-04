# SPDX-License-Identifier: Apache-2.0
"""Independent 810E V4.1 Q/K numerical and paged-layout tests without model weights."""

import importlib.util
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def kernels():
    if not torch.cuda.is_available():
        pytest.skip("An 810E device is required")
    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("This test matrix supports only 810E")
    current_platform.import_kernels()
    # DSV41_TEST_KERNEL_FILE overrides the installed kernels with a candidate file.
    path = os.environ.get("DSV41_TEST_KERNEL_FILE")
    if path:
        spec = importlib.util.spec_from_file_location("dsv41_candidate", Path(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from vllm_sail.models.deepseek_v41.ops import indexer_tilelang

    return indexer_tilelang


def cos_sin():
    pos = torch.arange(102401, device="cuda", dtype=torch.float32)
    freq = torch.arange(1, 33, device="cuda", dtype=torch.float32) / 1000
    angles = pos[:, None] * freq[None, :]
    return torch.cat((angles.cos(), angles.sin()), -1)


def rope(x, positions, cs):
    x = x.float().clone()
    c, s = cs[positions].float().chunk(2, -1)
    while c.ndim < x.ndim:
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    even, odd = x[..., 64::2].clone(), x[..., 65::2].clone()
    x[..., 64::2] = (even * c - odd * s).bfloat16().float()
    x[..., 65::2] = (odd * c + even * s).bfloat16().float()
    return x


def quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 127
    return (x / scale).clamp(-127, 127).to(torch.int8), scale


@pytest.mark.parametrize(
    "n,heads",
    [
        (0, 32),
        (1, 4),
        (1, 8),
        (1, 12),
        (1, 16),
        (1, 32),
        (1, 64),
        (16, 32),
        (33, 32),
        (256, 32),
        (8192, 32),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_q(kernels, n, heads, strided):
    torch.manual_seed(1017)
    backing = torch.randn(
        (max(n, 1), heads, 256 if strided else 128), device="cuda", dtype=torch.bfloat16
    )
    q = backing[:n, :, :128]
    w = torch.randn(
        (n, 2 * heads if strided else heads), device="cuda", dtype=torch.bfloat16
    )[:, :heads]
    positions = torch.arange(n, device="cuda", dtype=torch.int64) + 511
    cs = cos_sin()
    expected, scale = quant(rope(q, positions, cs))
    weights = w.float() * scale.squeeze(-1) * 0.125 * (heads**-0.5)
    result, actual_weights = kernels.fused_indexer_q_rope_quant_int8(
        positions, q, cs, w, 0.125, heads**-0.5
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=1)
    torch.testing.assert_close(actual_weights, weights, rtol=1e-2, atol=1e-8)
    repeated = kernels.fused_indexer_q_rope_quant_int8(
        positions, q, cs, w, 0.125, heads**-0.5
    )
    assert torch.equal(result, repeated[0]) and torch.equal(actual_weights, repeated[1])
    if n:
        error = (result.float() - expected.float()).abs() * scale
        assert torch.all(error <= scale * 1.01 + 1e-8)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("page_stride", [16 * 132, 16 * 132 + 576, 2**31 + 4096])
def test_k_paged(kernels, ratio, page_stride):
    torch.manual_seed(20261029)
    positions = torch.tensor(
        [0, 1, 2, 3, 511, 512, 1023, 1024, 102399], device="cuda", dtype=torch.int64
    )
    slots = torch.tensor(
        [0, 1, -1, 15, 16, 17, 18, 19, 20], device="cuda", dtype=torch.int64
    )
    k = torch.randn((positions.numel() + 3, 136), device="cuda", dtype=torch.bfloat16)[
        :, :128
    ]
    norm = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    cs = cos_sin()
    raw = torch.full((page_stride + 16 * 132,), 193, dtype=torch.uint8, device="cuda")
    cache = raw.as_strided((2, 16, 132), (page_stride, 132, 1))
    kernels.indexer_k_norm_rope_store_int8(
        k, positions, cs, norm, 1e-20, cache, slots, ratio, False
    )
    torch.cuda.synchronize()
    normalized = (
        k[: len(positions)].float()
        * torch.rsqrt(
            k[: len(positions)].float().square().mean(-1, keepdim=True) + 1e-20
        )
        * norm.float()
    ).bfloat16()
    expected, scales = quant(rope(normalized, positions // ratio * ratio, cs))
    active = {}
    for row, (pos, slot) in enumerate(
        zip(positions.cpu().tolist(), slots.cpu().tolist(), strict=True)
    ):
        if slot >= 0 and (pos + 1) % ratio == 0:
            active[slot] = row
    for slot in range(32):
        base = (slot // 16) * page_stride
        values = raw[base + (slot % 16) * 128 : base + (slot % 16 + 1) * 128]
        actual_scale = raw[
            base + 16 * 128 + (slot % 16) * 4 : base + 16 * 128 + (slot % 16 + 1) * 4
        ]
        if slot in active:
            row = active[slot]
            torch.testing.assert_close(
                values.view(torch.int8), expected[row], rtol=0, atol=1
            )
            torch.testing.assert_close(
                actual_scale.view(torch.float32), scales[row], rtol=1e-2, atol=1e-8
            )
        else:
            assert torch.all(values == 193) and torch.all(actual_scale == 193)
    if page_stride > 16 * 132:
        assert torch.all(raw[16 * 132 : 16 * 132 + 32] == 193)
        assert torch.all(raw[page_stride - 32 : page_stride] == 193)
    pages = torch.cat((raw[: 16 * 132], raw[page_stride:])).clone()
    kernels.indexer_k_norm_rope_store_int8(
        k, positions, cs, norm, 1e-20, cache, slots, ratio, False
    )
    assert torch.equal(pages, torch.cat((raw[: 16 * 132], raw[page_stride:])))


def test_new_buffers_stream_graph(kernels):
    torch.manual_seed(41)
    cs = cos_sin()
    pos = torch.arange(16, device="cuda", dtype=torch.int64)
    q = torch.randn((16, 32, 128), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((16, 32), device="cuda", dtype=torch.bfloat16)
    expected = kernels.fused_indexer_q_rope_quant_int8(pos, q, cs, w, 0.125, 32**-0.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = kernels.fused_indexer_q_rope_quant_int8(
            pos.clone(), q.clone(), cs, w.clone(), 0.125, 32**-0.5
        )
    torch.cuda.current_stream().wait_stream(stream)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    graph = torch.cuda.CUDAGraph()
    output = (torch.empty_like(expected[0]), torch.empty_like(expected[1]))
    with torch.cuda.graph(graph):
        kernels.fused_indexer_q_rope_quant_int8(
            pos, q, cs, w, 0.125, 32**-0.5, output_buffers=output
        )
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(output, expected, strict=True))
    q.zero_()
    graph.replay()
    refreshed = kernels.fused_indexer_q_rope_quant_int8(pos, q, cs, w, 0.125, 32**-0.5)
    assert all(torch.equal(a, b) for a, b in zip(output, refreshed, strict=True))


def assert_k_update(raw, before, cache, k, positions, cs, norm, slots, ratio):
    """Independent CPU reference with bytewise checks of unwritten slots, page padding, and shared layers."""
    actual = raw.cpu().flatten()
    touched = torch.zeros(actual.numel(), dtype=torch.bool)
    x = k[: slots.numel()].cpu().float()
    normalized = (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20) * norm.cpu().float()
    ).bfloat16()
    expected, scales = quant(
        rope(normalized, positions.cpu() // ratio * ratio, cs.cpu())
    )
    block_size = cache.shape[1]
    for row, (pos, slot) in enumerate(
        zip(positions.cpu().tolist(), slots.cpu().tolist(), strict=True)
    ):
        if slot < 0 or (pos + 1) % ratio:
            continue
        page, local = divmod(slot, block_size)
        base = cache.storage_offset() + page * cache.stride(0)
        value_slice = slice(base + local * 128, base + (local + 1) * 128)
        scale_slice = slice(
            base + block_size * 128 + local * 4,
            base + block_size * 128 + (local + 1) * 4,
        )
        values = actual[value_slice].view(torch.int8)
        scale = actual[scale_slice].view(torch.float32)
        torch.testing.assert_close(values, expected[row], rtol=0, atol=1)
        torch.testing.assert_close(scale, scales[row], rtol=1e-2, atol=1e-8)
        error = (values.float() * scale - expected[row].float() * scales[row]).abs()
        bound = scale + expected[row].float().abs() * (scale - scales[row]).abs()
        assert torch.all(error <= bound * 1.001 + 1e-8)
        touched[value_slice] = True
        touched[scale_slice] = True
    assert torch.equal(actual[~touched], before.flatten()[~touched])


@pytest.mark.parametrize("ratio", [1, 2])
def test_k_dynamic_buffers_stream_and_shared_layers(kernels, ratio):
    torch.manual_seed(41017)
    raw = torch.full((40, 3, 16, 132), 193, dtype=torch.uint8, device="cuda")
    cs = cos_sin()
    stream = torch.cuda.Stream()
    # Alternate shapes, input addresses, and shared layers; empty inputs must not change any bytes.
    for step, n in enumerate((33, 1, 16, 0, 3, 256)):
        cache = raw[:, 0 if step % 2 == 0 else 2]
        k = torch.randn(n + 2, 136, dtype=torch.bfloat16, device="cuda")[:, :128]
        norm = torch.randn(128, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(n, device="cuda", dtype=torch.int64) + 511
        slots = torch.arange(n, device="cuda", dtype=torch.int64) + step * 16
        if n:
            slots[-1] = -1
        before = raw.cpu().clone()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernels.indexer_k_norm_rope_store_int8(
                k, positions, cs, norm, 1e-20, cache, slots, ratio, False
            )
        torch.cuda.current_stream().wait_stream(stream)
        assert_k_update(raw, before, cache, k, positions, cs, norm, slots, ratio)
        first = raw.clone()
        kernels.indexer_k_norm_rope_store_int8(
            k, positions, cs, norm, 1e-20, cache, slots, ratio, False
        )
        assert torch.equal(raw, first)
    assert torch.all(raw[:, 1] == 193)


@pytest.mark.parametrize("ratio", [1, 2])
def test_k_graph_replays_changed_inputs_and_slots(kernels, ratio):
    torch.manual_seed(41018)
    raw = torch.full((4, 3, 16, 132), 193, dtype=torch.uint8, device="cuda")
    cache = raw[:, 1]
    k = torch.randn(4, 136, dtype=torch.bfloat16, device="cuda")[:, :128]
    norm = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([0, 1, 511, 102399], dtype=torch.int64, device="cuda")
    slots = torch.tensor([0, -1, 15, 16], dtype=torch.int64, device="cuda")
    cs = cos_sin()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernels.indexer_k_norm_rope_store_int8(
            k, positions, cs, norm, 1e-20, cache, slots, ratio, False
        )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernels.indexer_k_norm_rope_store_int8(
            k, positions, cs, norm, 1e-20, cache, slots, ratio, False
        )
    for changed in (False, True):
        if changed:
            k.neg_()
            norm.mul_(0.5)
            positions.copy_(torch.tensor([1, 2, 1023, 1024], device="cuda"))
            slots.copy_(torch.tensor([31, 32, -1, 63], device="cuda"))
        raw.fill_(193)
        before = raw.cpu().clone()
        graph.replay()
        assert_k_update(raw, before, cache, k, positions, cs, norm, slots, ratio)
        first = raw.clone()
        graph.replay()
        assert torch.equal(raw, first)


@pytest.mark.parametrize("ratio", [1, 2])
def test_k_reference_guard_cpu(monkeypatch, ratio):
    def forbidden(*args, **kwargs):
        raise AssertionError("Device initialization is forbidden in the CPU oracle")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", forbidden)
    raw = torch.full((2, 3, 16, 132), 193, dtype=torch.uint8)
    before = raw.clone()
    cache = raw[:, 1]
    k = torch.ones(3, 128, dtype=torch.bfloat16)
    norm = torch.ones(128, dtype=torch.bfloat16)
    positions = torch.tensor([1, 2, 3], dtype=torch.int64)
    slots = torch.tensor([15, -1, 16], dtype=torch.int64)
    cs = torch.cat((torch.ones(4, 32), torch.zeros(4, 32)), -1)
    # Analytical reference: RMSNorm and identity RoPE yield ones, quantized to 127.
    for slot in (15, 16):
        page, local = divmod(slot, 16)
        flat = cache[page].flatten()
        flat[local * 128 : (local + 1) * 128].fill_(127)
        flat[16 * 128 + local * 4 : 16 * 128 + (local + 1) * 4] = torch.tensor(
            [1 / 127], dtype=torch.float32
        ).view(torch.uint8)
    assert_k_update(raw, before, cache, k, positions, cs, norm, slots, ratio)
    raw[0, 0, 0, 0] = 0
    with pytest.raises(AssertionError):
        assert_k_update(raw, before, cache, k, positions, cs, norm, slots, ratio)
    raw[0, 0, 0, 0] = 193
    cache[0].flatten()[15 * 128] = 0
    with pytest.raises(AssertionError):
        assert_k_update(raw, before, cache, k, positions, cs, norm, slots, ratio)


def test_k_empty(kernels):
    kernels.indexer_k_norm_rope_store_int8(
        torch.empty((0, 128), device="cuda", dtype=torch.bfloat16),
        torch.empty(0, device="cuda", dtype=torch.int64),
        cos_sin(),
        torch.ones(128, device="cuda", dtype=torch.bfloat16),
        1e-20,
        torch.empty((0, 16, 132), device="cuda", dtype=torch.uint8),
        torch.empty(0, device="cuda", dtype=torch.int64),
        2,
        False,
    )
