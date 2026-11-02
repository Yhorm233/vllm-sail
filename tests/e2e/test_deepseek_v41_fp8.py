# SPDX-License-Identifier: Apache-2.0
"""Device numerical tests for the V4.1 main cache, native Q/SWA, and Engram."""

import types

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module", autouse=True)
def device():
    if not torch.cuda.is_available():
        pytest.skip("An 810E device is required")
    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("This test supports only 810E")
    current_platform.import_kernels()


def cpu_rope(x, positions, cs):
    out = x.float().clone()
    c, s = cs[positions].float().chunk(2, -1)
    while c.ndim < out.ndim:
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    a, b = out[..., 448::2].clone(), out[..., 449::2].clone()
    out[..., 448::2] = a * c - b * s
    out[..., 449::2] = b * c + a * s
    return out.bfloat16()


def cpu_record(latent, position, cs):
    x = latent.float()[:448].reshape(7, 64)
    exponent = torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / 448))
    scaled = (x * torch.exp2(-exponent[:, None])).clamp(-448, 448)
    packed = scaled.to(torch.float8_e4m3fn).view(torch.uint8).reshape(448)
    scales = torch.cat(((exponent + 127).byte(), torch.zeros(1, dtype=torch.uint8)))
    rotated = cpu_rope(latent[None], torch.tensor([position]), cs)[0]
    values = torch.cat((packed, rotated[448:].view(torch.uint8)))
    dequant = torch.cat(
        (
            (
                scaled.to(torch.float8_e4m3fn).float() * torch.exp2(exponent[:, None])
            ).flatten(),
            rotated[448:].float(),
        )
    ).bfloat16()
    return values, scales, dequant


def cpu_inputs(n, dim=512):
    generator = torch.Generator().manual_seed(1017)
    latent = torch.randn(n, dim, generator=generator).bfloat16()
    angles = (
        torch.arange(102401).float()[:, None]
        * torch.arange(1, 33).float()[None, :]
        / 1000
    )
    cs = torch.cat((angles.cos(), angles.sin()), -1)
    return latent, cs


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("stride", [16 * 584, 16 * 584 + 576, 2**31 + 4096])
def test_latent_publish_and_gather(ratio, stride):
    from vllm_sail.models.deepseek_v4.ops.cache import register_ops
    from vllm_sail.models.deepseek_v41.ops.latent_store import rope_quant_insert

    register_ops()
    positions = torch.tensor([0, 1, 2, 3, 511, 512, 1023, 1024, 102399])
    slots = torch.tensor([0, 1, -1, 15, 16, 17, 18, 19, 20])
    latent, cs = cpu_inputs(len(slots))
    raw = torch.full((stride + 16 * 584,), 193, dtype=torch.uint8, device="cuda")
    cache = raw.as_strided((2, 16, 584), (stride, 584, 1))
    args = (latent.cuda(), positions.cuda(), cs.cuda(), cache, slots.cuda(), ratio)
    rope_quant_insert(*args)
    torch.cuda.synchronize()
    pages = [raw[base : base + 16 * 584].cpu() for base in (0, stride)]
    active = {}
    for i, (pos, slot) in enumerate(
        zip(positions.tolist(), slots.tolist(), strict=True)
    ):
        if slot >= 0 and (pos + 1) % ratio == 0:
            active[slot] = cpu_record(latent[i], pos // ratio * ratio, cs)
    for slot in range(32):
        page, local = pages[slot // 16], slot % 16
        values = page[local * 576 : (local + 1) * 576]
        scales = page[16 * 576 + local * 8 : 16 * 576 + (local + 1) * 8]
        if slot in active:
            torch.testing.assert_close(values, active[slot][0], rtol=0, atol=0)
            torch.testing.assert_close(scales, active[slot][1], rtol=0, atol=0)
        else:
            assert torch.all(values == 193) and torch.all(scales == 193)
    if stride > 16 * 584:
        assert torch.all(raw[16 * 584 : 16 * 584 + 32] == 193)
        assert torch.all(raw[stride - 32 : stride] == 193)
    rope_quant_insert(*args)
    for base, page in zip((0, stride), pages, strict=True):
        assert torch.equal(raw[base : base + 16 * 584].cpu(), page)
    for slot, (_, _, expected) in active.items():
        out = torch.full((1, 2, 512), 77, dtype=torch.bfloat16, device="cuda")
        seq = torch.tensor([slot + 1], dtype=torch.int32, device="cuda")
        lens = torch.ones(1, dtype=torch.int32, device="cuda")
        table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        torch.ops.vllm.ppu_deepseek_v4_dequant_gather(
            out, cache, seq, lens, table, 16, 1
        )
        torch.testing.assert_close(out[0, 1].cpu(), expected, rtol=0, atol=0)
        assert torch.all(out[0, 0] == 77)


@pytest.mark.parametrize("heads", [8, 64])
def test_native_q_rope_swa_schema_and_values(heads):
    latent, cs = cpu_inputs(3)
    q = torch.randn(
        3, heads, 512, generator=torch.Generator().manual_seed(1017)
    ).bfloat16()
    positions = torch.tensor([0, 511, 102399])
    slots = torch.tensor([0, -1, 17])
    cache = torch.full((2, 16, 584), 193, dtype=torch.uint8, device="cuda")
    op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    schema = op._schemas[""]
    assert len(schema.arguments) == 13
    result = op(
        q.cuda(),
        latent.cuda(),
        cache.view(2, -1),
        slots.cuda(),
        positions.cuda(),
        cs.cuda(),
        heads,
        1e-20,
        16,
        False,
        False,
        True,
        False,
    )
    torch.testing.assert_close(
        result.cpu(), cpu_rope(q, positions, cs), rtol=1e-2, atol=1e-2
    )
    flat = cache.cpu().flatten()
    for i, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        values, scales, _ = cpu_record(latent[i], positions[i].item(), cs)
        page, local = slot // 16 * 16 * 584, slot % 16
        torch.testing.assert_close(
            flat[page + local * 576 : page + (local + 1) * 576], values, rtol=0, atol=0
        )
        torch.testing.assert_close(
            flat[page + 16 * 576 + local * 8 : page + 16 * 576 + (local + 1) * 8],
            scales,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("uva", [False, True])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("int8_table", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_engram_all_bytes_ownership_stream(uva, background, int8_table, ids_dtype):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    from vllm_sail.models.deepseek_v41.ops.engram_lookup import lookup

    data = (torch.arange(32 * 256) % 256).byte().reshape(32, 256)
    scales = (torch.arange(32 * 8) % 8 + 123).byte().reshape(32, 8)
    if int8_table:
        data = data.view(torch.int8)
        scales = (torch.arange(32).float() / 97 + 0.001).reshape(32, 1)
    dtype = torch.int8 if int8_table else torch.float8_e4m3fn
    if uva:
        host = data.pin_memory()
        host_scales = scales.pin_memory()
        assert host.is_pinned() and host_scales.is_pinned()
        weight = get_accelerator_view_from_cpu_tensor(host).view(dtype)
        device_scales = get_accelerator_view_from_cpu_tensor(host_scales)
    else:
        weight, device_scales = data.cuda().view(dtype), scales.cuda()
    # Own only valid hash heads 2 and 3; head 4 is padding.
    indices = torch.tensor(
        [[99, 100, 100, 131], [100, 100, 132, -1], [101, 102, 111, 112]]
    )
    indices_gpu = torch.empty(3, 8, dtype=ids_dtype, device="cuda")[:, ::2]
    indices_gpu.copy_(indices)
    owner = types.SimpleNamespace(
        part_n_hash_cols=3,
        head_start=2,
        n_hash_cols=4,
        vocab_start_idx=100,
        vocab_end_idx=132,
        dim=256,
        block_size=32,
        _num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        _storage=lambda: (weight, device_scales),
    )
    out = torch.full((3, 3, 256), 77, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lookup(owner, indices_gpu, out, background)
    stream.synchronize()
    expected = torch.zeros_like(out, device="cpu")
    decoded = data.view(dtype).float()
    for t in range(3):
        for h in range(2):
            row = indices[t, h + 2].item() - 100
            if 0 <= row < 32:
                row_scale = (
                    scales[row]
                    if int8_table
                    else torch.exp2(scales[row].float() - 127).repeat_interleave(32)
                )
                expected[t, h] = (decoded[row] * row_scale).bfloat16()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    if uva:
        host.zero_()
        lookup(owner, indices_gpu, out, background)
        assert torch.all(out == 0)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_int8_engram_large_row_address_and_empty(ids_dtype):
    from vllm_sail.models.deepseek_v41.ops.engram_lookup import lookup

    rows = 2**23 + 17
    weight = torch.empty(rows, 256, dtype=torch.int8, device="cuda")
    scales = torch.empty(rows, 1, dtype=torch.float32, device="cuda")
    selected = [0, 2**23, rows - 1]
    for i, row in enumerate(selected):
        weight[row].fill_(i * 53 - 127)
        scales[row].fill_(0.013 * (i + 1))
    vocab_start = 0 if ids_dtype == torch.int32 else 2**31
    owner = types.SimpleNamespace(
        part_n_hash_cols=1,
        head_start=0,
        n_hash_cols=1,
        vocab_start_idx=vocab_start,
        vocab_end_idx=vocab_start + rows,
        dim=256,
        block_size=32,
        _num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        _storage=lambda: (weight, scales),
    )
    indices = torch.tensor(
        [[row + vocab_start] for row in selected], dtype=ids_dtype, device="cuda"
    )
    out = torch.empty(3, 1, 256, dtype=torch.bfloat16, device="cuda")
    lookup(owner, indices, out)
    expected = torch.stack(
        [(weight[row].float() * scales[row]).bfloat16() for row in selected]
    )
    torch.testing.assert_close(out[:, 0], expected, rtol=0, atol=0)
    first = out.clone()
    lookup(owner, indices, out)
    assert torch.equal(first, out)
    lookup(owner, indices[:0], out[:0])


def test_upstream_ngram_hash_to_int8_lookup():
    """Pass per-layer INT32 views from the upstream hash kernel directly to lookup without casting."""
    from vllm.models.deepseek_v41.common.engram import (
        NgramHashState,
        compute_hash_multipliers,
    )

    from vllm_sail.models.deepseek_v41.ops.engram_lookup import lookup

    # Use a small layout without tokenizer/checkpoint loading; run upstream slot-cache and hash kernels.
    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.use_slot_cache = True
    state.block_size = 16
    state.pad_id = 0
    state._cache = torch.zeros(32, dtype=torch.int32, device="cuda")
    state.token_map = torch.arange(8, dtype=torch.int32, device="cuda")
    state.primes = torch.tensor(
        [[[11, 13], [17, 19]], [[23, 29], [31, 37]]], device="cuda"
    )
    flat = state.primes.flatten(1)
    state.offsets = flat.cumsum(-1) - flat
    state.multipliers = compute_hash_multipliers((1, 14), 3, 8).cuda()
    input_ids = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    positions = torch.tensor([0, 1, 2, 0, 1], device="cuda")
    starts = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    dead = torch.tensor([False, False, True, False, False], device="cuda")
    lookback = torch.full((2, 2), -1, device="cuda")
    slots = torch.tensor([0, 1, 2, 16, 17], device="cuda")
    blocks = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    hashes = state(
        input_ids,
        positions,
        starts,
        dead,
        lookback,
        torch.zeros_like(lookback, dtype=torch.bool),
        slots,
        blocks,
    )
    dummy, keep = state.dummy_hashes(input_ids)
    assert hashes.dtype == dummy.dtype == torch.int32
    assert hashes.shape == (5, 2, 4) and not keep.any()
    assert torch.all(dummy == -1)
    tokens_cpu, positions_cpu, dead_cpu = (
        tensor.cpu().tolist() for tensor in (input_ids, positions, dead)
    )
    for layer in range(2):
        rows = int(flat[layer].sum())
        weight = (
            (torch.arange(rows * 256) % 255 - 127).to(torch.int8).reshape(rows, 256)
        )
        scale = (torch.arange(rows).float() / 100 + 0.01).reshape(rows, 1)
        weight_gpu, scale_gpu = weight.cuda(), scale.cuda()
        owner = types.SimpleNamespace(
            part_n_hash_cols=4,
            head_start=0,
            n_hash_cols=4,
            vocab_start_idx=0,
            vocab_end_idx=rows,
            dim=256,
            block_size=32,
            _num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
            _storage=lambda w=weight_gpu, s=scale_gpu: (w, s),
        )
        ids = hashes[:, layer]
        assert ids.stride() == (8, 1) and not ids.is_contiguous()
        ids_cpu = ids.cpu().long()
        # Upstream hashes dead tokens using pad_id; only dummy_hashes returns -1.
        primes_cpu = flat[layer].cpu().tolist()
        offsets_cpu = state.offsets[layer].cpu().tolist()
        multipliers_cpu = state.multipliers[layer].cpu().tolist()
        expected_ids = torch.empty_like(ids_cpu)
        for token, position in enumerate(positions_cpu):
            rolling, blocked = 0, False
            for shift in range(3):
                blocked |= position < shift or dead_cpu[max(token - shift, 0)]
                value = state.pad_id if blocked else tokens_cpu[token - shift]
                rolling ^= value * multipliers_cpu[shift]
                if shift:
                    for head in range(2):
                        col = (shift - 1) * 2 + head
                        expected_ids[token, col] = (
                            rolling % primes_cpu[col] + offsets_cpu[col]
                        )
        torch.testing.assert_close(ids_cpu, expected_ids, rtol=0, atol=0)
        valid = (ids_cpu >= 0) & (ids_cpu < rows)
        expected = (
            weight[ids_cpu.clamp(0, rows - 1)].float()
            * scale[ids_cpu.clamp(0, rows - 1)]
        ).bfloat16()
        expected[~valid] = 0
        out = torch.full((5, 4, 256), 77, dtype=torch.bfloat16, device="cuda")
        lookup(owner, ids, out, background=True)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        lookup(owner, dummy[:, layer], out, background=True)
        assert torch.all(out == 0)
        for dtype in (torch.int16, torch.float32):
            with pytest.raises(ValueError, match="INT32/INT64"):
                lookup(owner, ids.to(dtype), out)


def test_int8_engram_shared_storage_layout_and_loader():
    from vllm_sail.models.deepseek_v41.engram import (
        Int8DPSharedEngramStorage,
        load_weight,
    )

    class Storage(Int8DPSharedEngramStorage):
        def _allocate(self, num_bytes):
            self.requested_bytes = num_bytes
            return torch.full((num_bytes,), 193, dtype=torch.uint8)

    storage = Storage(3, 256, 32, types.SimpleNamespace())
    assert storage.requested_bytes == 3 * 260
    assert storage.weight.dtype == torch.int8
    assert storage.weight_scale_inv.dtype == torch.float32
    assert storage.weight_scale_inv.data_ptr() - storage.weight.data_ptr() == 3 * 256
    for target in (storage.weight, storage.weight_scale_inv):
        target.engram_vocab_start = 2
        loaded = (
            torch.arange(7 * target.shape[1])
            .reshape(7, target.shape[1])
            .to(target.dtype)
        )
        load_weight(target, loaded)
        assert torch.equal(target, loaded[2:5])
        with pytest.raises(ValueError, match="implicit casting is forbidden"):
            load_weight(target, loaded.double())
