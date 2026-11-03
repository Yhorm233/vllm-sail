# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish the 810E V4.1 main latent into a mixed-format FP8 cache.

Based on vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6:
models/deepseek_v41/common/ops/fused_compress_quant_cache.py.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_sail.models.deepseek_v4.ops.cache import _encode_e4m3fn


@triton.jit
def _rope_quant_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot % CACHE_BLOCK) * 576
    scales = page + CACHE_BLOCK * 576 + (slot % CACHE_BLOCK) * 8

    quant = tl.reshape(normed, (8, 64))
    amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
    scaled = quant * tl.reshape(tl.exp2(-exponent), (8, 1))
    packed = tl.reshape(_encode_e4m3fn(tl.clamp(scaled, -448.0, 448.0)), (512,))
    tl.store(values + d, packed, d < 448)
    s = tl.arange(0, 8)
    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), max_encoded)
    tl.store(scales + s, encoded.to(tl.uint8), s < 7)
    tl.store(scales + 7, tl.full((), 0, tl.uint8))

    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    if SANITIZE_CACHE_NANS:
        rotated = tl.where(rotated == rotated, rotated, 0.0)
    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + d - 448, rotated.to(tl.bfloat16), d >= 448)


def rope_quant_insert(
    latent,
    positions,
    cos_sin_cache,
    kv_cache,
    slot_mapping,
    compress_ratio,
    fp8_scale=None,
):
    """Publish BF16 latent after the main compressor RMSNorm."""
    if compress_ratio not in (1, 2):
        raise ValueError("V4.1 compress_ratio must be 1 or 2")
    if latent.ndim != 2 or latent.shape[1] != 512:
        raise ValueError("latent must have shape [tokens, 512]")
    if latent.dtype != torch.bfloat16 or not latent.is_contiguous():
        raise ValueError("latent must be contiguous BF16")
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 3 or kv_cache.shape[-1] != 584:
        raise ValueError("The 810E main cache requires a 584-byte uint8 layout")
    if kv_cache.stride(1) != 584 or kv_cache.stride(2) != 1:
        raise ValueError("Cache pages must be internally contiguous")
    if kv_cache.stride(0) < kv_cache.shape[1] * 584 or kv_cache.stride(0) % 2:
        raise ValueError("Invalid cache page stride or alignment")
    if positions.dtype != torch.int64 or slot_mapping.dtype != torch.int64:
        raise ValueError("positions and slot_mapping must be INT64")
    if not positions.is_contiguous() or not slot_mapping.is_contiguous():
        raise ValueError("positions and slot_mapping must be contiguous")
    n = slot_mapping.numel()
    if n > min(latent.shape[0], positions.numel()):
        raise ValueError("slot_mapping exceeds the valid input length")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError(
            "cos_sin_cache must have shape [positions, 64] with a contiguous last dimension"
        )
    tensors = (latent, positions, cos_sin_cache, kv_cache, slot_mapping)
    if any(not t.is_cuda or t.device != latent.device for t in tensors):
        raise ValueError("All inputs must be on the same device")
    if n:
        _rope_quant_insert_kernel[(n,)](
            latent,
            positions,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            COS_STRIDE=cos_sin_cache.stride(0),
            CACHE_STRIDE=kv_cache.stride(0),
            CACHE_BLOCK=kv_cache.shape[1],
            COMPRESS_RATIO=compress_ratio,
            SANITIZE_CACHE_NANS=False,
            num_warps=4,
        )
