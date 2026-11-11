# SPDX-License-Identifier: Apache-2.0
"""Device oracles for the native ops used by DeepSeek V4 Flash.

Run against an installed native wheel with ``pytest`` (or an editable native
build). Missing ops must fail these tests, never silently skip or use a fallback.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    import torch

    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    return torch


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "shape", [(1, 256, 6), (17, 256, 6), (33, 384, 6), (17, 256, 8)]
)
@pytest.mark.parametrize("routing", ["bias", "hash32", "hash64"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softplus_sqrt(runtime, dtype_name, shape, routing, renormalize):
    torch = runtime
    tokens, experts, topk = shape
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(
        (tokens, experts),
        generator=generator,
        device="cuda",
        dtype=getattr(torch, dtype_name),
    )
    padding = torch.arange(tokens, device="cuda") % 4 == 3
    logits[padding] = float("nan")
    bias = input_ids = table = None
    index_dtype = torch.int64 if routing == "hash64" else torch.int32
    if routing == "bias":
        bias = torch.randn(experts, device="cuda", generator=generator)
    else:
        # Distinct experts per vocabulary row, with nontrivial token indirection.
        table = torch.stack(
            [
                torch.randperm(experts, device="cuda", generator=generator)[:topk]
                for _ in range(19)
            ]
        ).to(index_dtype)
        input_ids = (torch.arange(tokens, device="cuda") * 7 % 19).to(index_dtype)
    weights = torch.empty(tokens, topk, device="cuda", dtype=torch.float32)
    indices = torch.empty(tokens, topk, device="cuda", dtype=index_dtype)
    source_rows = torch.empty(tokens, topk, device="cuda", dtype=torch.int32)

    torch.ops._moe_C.topk_softplus_sqrt(
        weights,
        indices,
        source_rows,
        logits,
        renormalize,
        1.5,
        bias,
        input_ids,
        table,
        padding,
    )
    torch.cuda.synchronize()

    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    expected_ids = (
        table[input_ids.long()].long()
        if table is not None
        else (scores + bias).topk(topk, dim=-1).indices
    )
    expected = scores.gather(1, expected_ids)
    if renormalize:
        expected /= expected.sum(-1, keepdim=True)
    expected *= 1.5
    # Keep oracle masking independent of torch's device-side nonzero kernel.
    padding = padding.cpu()
    indices, weights = indices.cpu(), weights.cpu()
    expected_ids, expected = expected_ids.cpu(), expected.cpu()
    torch.testing.assert_close(indices[~padding].long(), expected_ids[~padding])
    torch.testing.assert_close(
        weights[~padding], expected[~padding], rtol=2e-5, atol=2e-6
    )
    assert bool((weights[padding] == 0).all())
    assert bool((indices[padding] == -1).all())


@pytest.mark.parametrize("hash_routing", [False, True])
def test_topk_nan_padding_without_mask(runtime, hash_routing):
    torch = runtime
    logits = torch.full((1, 256), float("nan"), device="cuda")
    weights = torch.empty(1, 6, device="cuda")
    indices = torch.empty(1, 6, device="cuda", dtype=torch.int32)
    rows = torch.empty_like(indices)
    ids = torch.zeros(1, device="cuda", dtype=torch.int32) if hash_routing else None
    table = (
        torch.arange(6, device="cuda", dtype=torch.int32).view(1, 6)
        if hash_routing
        else None
    )
    torch.ops._moe_C.topk_softplus_sqrt(
        weights, indices, rows, logits, True, 1.5, None, ids, table, None
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(weights).all())
    assert bool((weights == 0).all())


def _rope(torch, value, positions, cache):
    value = value.float()
    cs = cache[positions]
    cos, sin = cs[:, :32], cs[:, 32:]
    if value.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    even, odd = value[..., 448::2], value[..., 449::2]
    result = value.clone()
    result[..., 448::2] = torch.addcmul(-odd * sin, even, cos)
    result[..., 449::2] = torch.addcmul(odd * cos, even, sin)
    return result


def _assert_ulp(torch, actual, expected, max_ulp=1):
    """Allow only rounding-boundary differences, including values near zero."""
    assert actual.dtype == expected.dtype
    storage = torch.int8 if actual.element_size() == 1 else torch.int16
    sign = 1 << (actual.element_size() * 8 - 1)
    a = actual.detach().cpu().contiguous().view(storage).to(torch.int32)
    b = expected.detach().cpu().contiguous().view(storage).to(torch.int32)
    a = torch.where(a < 0, -sign - a, a)
    b = torch.where(b < 0, -sign - b, b)
    distance = (a - b).abs().max().item()
    assert distance <= max_ulp, f"maximum storage ULP distance: {distance}"


@pytest.mark.parametrize("mode", ["packed", "packed_no_qnorm", "bf16", "fp8"])
@pytest.mark.parametrize("tokens", [1, 17, 1025])
def test_qnorm_rope_cache_insert(runtime, mode, tokens):
    torch = runtime
    heads, padded_heads, block_size, eps = 16, 64, 16, 1e-6
    generator = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    kv = torch.randn(
        tokens, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    q_original, kv_original = q.clone(), kv.clone()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 3 % 4096
    frequency = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = torch.arange(4096, device="cuda").float()[:, None] * frequency
    rope_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    # Nonsequential slots cross block boundaries. Negative slots and a shorter
    # insertion list exercise DP padding without suppressing the Q output.
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64).flip(0) + 7
    slots[2::5] = -1
    if tokens > 1:
        slots = slots[:-1]
    blocks = (tokens + 7 + block_size - 1) // block_size + 1
    # Integer indices preserve the oracle rows without device-side nonzero.
    valid_rows = (slots.cpu() >= 0).nonzero(as_tuple=True)[0].to(slots.device)
    block_ids = slots[valid_rows] // block_size
    offsets = slots[valid_rows] % block_size
    q_float = q.float()
    q_norm = q_float * torch.rsqrt(q_float.square().mean(-1, keepdim=True) + eps)
    q_ref = _rope(
        torch, q_float if mode == "packed_no_qnorm" else q_norm, positions, rope_cache
    )
    kv_ref = _rope(torch, kv, positions, rope_cache)

    if mode.startswith("packed"):
        cache = torch.full(
            (blocks, block_size * 584), 93, device="cuda", dtype=torch.uint8
        )
        expected = cache.clone()
        args = (cache, slots, positions, rope_cache, padded_heads, eps, block_size)
        output = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, *args, apply_q_norm=mode != "packed_no_qnorm"
        )
        torch.cuda.synchronize()
        _assert_ulp(torch, output[:, :heads], q_ref.to(q.dtype))
        assert bool((output[:, heads:] == 0).all())
        torch.testing.assert_close(q, q_original, rtol=0, atol=0)

        # Decode the documented 576-byte token payload and separate 8-byte
        # scale tail, independently of vLLM's Triton cache helpers.
        rounded_kv = kv_ref.to(kv.dtype)[: slots.numel()][valid_rows]
        groups = rounded_kv[:, :448].float().reshape(-1, 7, 64)
        exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
        quantized = (groups * torch.exp2(-exponent[..., None])).clamp(-448, 448)
        payload = expected[:, : block_size * 576].view(blocks, block_size, 576)
        scales = expected[:, block_size * 576 :].view(blocks, block_size, 8)
        payload[block_ids, offsets, :448] = (
            quantized.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1, 448)
        )
        payload[block_ids, offsets, 448:] = (
            rounded_kv[:, 448:].contiguous().view(torch.uint8)
        )
        scales[block_ids, offsets, :7] = (exponent + 127).to(torch.uint8)
        scales[block_ids, offsets, 7] = 0
        actual_rope = cache[:, : block_size * 576].view(blocks, block_size, 576)[
            block_ids, offsets, 448:
        ]
        _assert_ulp(torch, actual_rope.contiguous().view(kv.dtype), rounded_kv[:, 448:])
        # Compare every other byte exactly, including untouched slots and scale
        # padding. The RoPE region was already checked within one BF16 ULP.
        payload[block_ids, offsets, 448:] = actual_rope
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    else:
        dtype = torch.bfloat16 if mode == "bf16" else torch.float8_e4m3fn
        cache = torch.zeros(blocks, block_size, 512, device="cuda", dtype=dtype)
        expected = torch.zeros(
            blocks, block_size, 512, device="cuda", dtype=torch.float32
        )
        if mode == "bf16":
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q, kv, cache, slots, positions, rope_cache, eps, block_size
            )
            output, output_ref = q, q_ref.to(dtype)
            kv_encoded = kv_ref.to(dtype)
        else:
            kv_scale = torch.tensor([0.25], device="cuda")
            q_scale_inv = torch.tensor([2.0], device="cuda")
            output = torch.empty_like(q, dtype=dtype)
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
                q,
                kv,
                output,
                cache,
                slots,
                positions,
                rope_cache,
                kv_scale,
                q_scale_inv,
                eps,
                block_size,
            )
            output_ref = (q_ref * q_scale_inv).clamp(-448, 448).to(dtype)
            # Full-cache FP8 keeps the KV rotation in fp32 until quantization.
            kv_encoded = (kv_ref / kv_scale).clamp(-448, 448).to(dtype)
            torch.testing.assert_close(q, q_original, rtol=0, atol=0)
        torch.cuda.synchronize()
        expected[block_ids, offsets] = kv_encoded.float()[: slots.numel()][valid_rows]
        _assert_ulp(torch, output, output_ref)
        _assert_ulp(torch, cache, expected.to(dtype))
        torch.testing.assert_close(
            cache[..., :448].float(), expected[..., :448], rtol=0, atol=0
        )
    torch.testing.assert_close(kv, kv_original, rtol=0, atol=0)


def _kv_only_expected(torch, backing, mode, kv, positions, slots, rope_cache, scale):
    """Build the byte-layout oracle on CPU, including untouched page padding."""
    expected = backing.cpu().clone()
    block_size = expected.shape[1] if mode in ("bf16", "fp8") else 16
    rotated = _rope(torch, kv.cpu(), positions.cpu(), rope_cache.cpu())
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        if mode == "bf16":
            expected[block, offset, :512] = rotated[row].bfloat16()
        elif mode == "fp8":
            encoded = (rotated[row] / scale.cpu()).clamp(-448, 448)
            expected[block, offset, :512] = encoded.to(torch.float8_e4m3fn)
        else:
            rounded = rotated[row].bfloat16()
            mxfp8 = mode == "mxfp8"
            quant_dims, group_size, scale_stride = (
                (512, 32, 16) if mxfp8 else (448, 64, 8)
            )
            payload_width = 512 if mxfp8 else 576
            groups = rounded[:quant_dims].float().reshape(-1, group_size)
            exponent = torch.ceil(
                torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)
            )
            encoded = (groups * torch.exp2(-exponent[:, None])).clamp(-448, 448)
            start = offset * payload_width
            expected[block, start : start + quant_dims] = (
                encoded.to(torch.float8_e4m3fn).view(torch.uint8).flatten()
            )
            if not mxfp8:
                expected[block, start + 448 : start + 576] = rounded[448:].view(
                    torch.uint8
                )
            start = block_size * payload_width + offset * scale_stride
            expected[block, start : start + exponent.numel()] = (exponent + 127).byte()
            if not mxfp8:
                expected[block, start + 7] = 0
    return expected


def _assert_kv_only_cache(torch, actual, expected, mode, slots):
    """Allow one rotation ULP while checking guards and all other bytes exactly."""
    actual = actual.cpu()
    expected = expected.clone()
    for slot in slots.cpu().tolist():
        if slot < 0:
            continue
        block, offset = divmod(slot, 16)
        if mode in ("bf16", "fp8"):
            region = (block, offset, slice(448, 512))
            _assert_ulp(torch, actual[region], expected[region])
        elif mode == "packed":
            region = (block, slice(offset * 576 + 448, (offset + 1) * 576))
            _assert_ulp(
                torch,
                actual[region].contiguous().view(torch.bfloat16),
                expected[region].contiguous().view(torch.bfloat16),
            )
        else:
            region = (block, slice(offset * 512 + 448, (offset + 1) * 512))
            _assert_ulp(
                torch,
                actual[region].contiguous().view(torch.float8_e4m3fn),
                expected[region].contiguous().view(torch.float8_e4m3fn),
            )
        expected[region] = actual[region]
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("mode", ["packed", "mxfp8", "bf16", "fp8"])
@pytest.mark.parametrize("tokens", [0, 1, 17, 1025])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_dspark_kv_only_rope_insert(runtime, mode, tokens, graph_replay):
    """Exercise the actual DSpark op, without allocating a dummy query tensor."""
    torch = runtime
    block_size = 16
    generator = torch.Generator().manual_seed(41)
    kv = torch.randn(tokens, 512, generator=generator).bfloat16().cuda()
    positions = (torch.arange(tokens, dtype=torch.int64) * 3 % 4096).cuda()
    angles = torch.arange(4096).float()[:, None] * (
        10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    )
    rope_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).cuda()
    slots = torch.arange(tokens, dtype=torch.int64).flip(0) + 7
    slots[2::5] = -1
    if tokens > 1:
        slots = slots[:-1]
    slots = slots.cuda()
    blocks = (tokens + 7 + block_size - 1) // block_size + 1
    packed = mode in ("packed", "mxfp8")
    width = (528 if mode == "mxfp8" else 584) if packed else 512
    dtype = (
        torch.uint8
        if packed
        else (torch.bfloat16 if mode == "bf16" else torch.float8_e4m3fn)
    )
    # A padded page stride detects writes into adjacent pages or guard regions.
    if packed:
        backing = torch.full((blocks, block_size * width + 64), 93, dtype=dtype).cuda()
        cache = backing[:, : block_size * width].view(blocks, block_size, width)
    else:
        backing = torch.full((blocks, block_size, width + 8), 0.5).to(dtype).cuda()
        cache = backing[..., :width]
    sentinel = backing.clone()
    scale = torch.tensor([0.25], device="cuda") if mode == "fp8" else None

    def insert():
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(
            kv, cache, slots, positions, rope_cache, block_size, scale, mode == "mxfp8"
        )

    original = kv.clone()
    insert()
    torch.cuda.synchronize()
    torch.testing.assert_close(kv, original, rtol=0, atol=0)
    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            insert()
        # Replay must read changed inputs, positions and per-layer slot mappings.
        for shift in (1, 2):
            kv.add_(0.25)
            positions.copy_(
                ((torch.arange(tokens, dtype=torch.int64) * 5 + shift) % 4096).cuda()
            )
            moved = slots.cpu()
            moved[moved >= 0] = (moved[moved >= 0] + 1) % (blocks * block_size)
            slots.copy_(moved.cuda())
            backing.copy_(sentinel)
            original = kv.clone()
            graph.replay()
            torch.cuda.synchronize()
            expected = _kv_only_expected(
                torch, sentinel, mode, kv, positions, slots, rope_cache, scale
            )
            _assert_kv_only_cache(torch, backing, expected, mode, slots)
            torch.testing.assert_close(kv, original, rtol=0, atol=0)
    else:
        expected = _kv_only_expected(
            torch, sentinel, mode, kv, positions, slots, rope_cache, scale
        )
        _assert_kv_only_cache(torch, backing, expected, mode, slots)


@pytest.mark.parametrize("invalid", ["kv", "slots", "positions", "cache", "scale"])
def test_dspark_kv_only_rejects_invalid_contract(runtime, invalid):
    torch = runtime
    kv = torch.zeros(2, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.zeros(1, device="cuda", dtype=torch.int64)
    positions = torch.zeros(2, device="cuda", dtype=torch.int64)
    cache = torch.zeros(1, 16, 512, device="cuda", dtype=torch.bfloat16)
    rope_cache = torch.zeros(1, 64, device="cuda")
    if invalid == "kv":
        kv = kv.float()
    elif invalid == "slots":
        slots = slots.int()
    elif invalid == "positions":
        positions = positions[:1]
    elif invalid == "cache":
        cache = cache[..., :511]
    else:
        cache = cache.to(torch.float8_e4m3fn)
    with pytest.raises(RuntimeError):
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(
            kv, cache, slots, positions, rope_cache, 16
        )
