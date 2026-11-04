# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 DSA INT8 Q and latent K1/K2 for 810E.

Mathematical contracts from vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6:
models/deepseek_v4/common/ops/fused_indexer_q.py and
models/deepseek_v41/common/ops/indexer_k_store.py.
K consumes the projected wk(latent), not the legacy V4 compressor state.
"""

from __future__ import annotations

import torch
from vllm.tilelang_utils import T, tilelang_jit

_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float16: "float16",
}


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    """Preserve storage offset and physical span without copying paged tensors into a contiguous layout."""
    if tensor.numel() == 0:
        return tensor.reshape(-1)
    span = 1 + sum(
        (n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.as_strided((span,), (1,))


@tilelang_jit
def _q_kernel(
    positions,
    q,
    cs,
    weights,
    out,
    out_weights,
    q_s0: int,
    q_s1: int,
    cs_s0: int,
    w_s0: int,
    softmax_scale: float,
    head_scale: float,
    heads: int,
    cs_dtype=T.float32,
    w_dtype=T.bfloat16,
):
    n = T.dynamic("n")
    positions: T.Tensor((n,), T.int64)
    q: T.Tensor((T.dynamic("q_span", "int64"),), T.bfloat16)
    cs: T.Tensor((T.dynamic("cs_span", "int64"),), cs_dtype)
    weights: T.Tensor((T.dynamic("w_span", "int64"),), w_dtype)
    out: T.Tensor((T.dynamic("out_span", "int64"),), T.int8)
    out_weights: T.Tensor((T.dynamic("ow_span", "int64"),), T.float32)
    with T.Kernel(n, heads // 4, threads=256) as (token, group):
        even = T.alloc_fragment((4, 64), T.float32)
        odd = T.alloc_fragment((4, 64), T.float32)
        absolute = T.alloc_fragment((4, 64), T.float32)
        maximum = T.alloc_fragment((4,), T.float32)
        for h, pair in T.Parallel(4, 64):
            base = T.cast(token, T.int64) * q_s0 + (group * 4 + h) * q_s1
            x = T.cast(q[base + pair * 2], T.float32)
            y = T.cast(q[base + pair * 2 + 1], T.float32)
            if pair >= 32:
                cbase = positions[token] * cs_s0 + pair - 32
                c = T.cast(cs[cbase], T.float32)
                s = T.cast(cs[cbase + 32], T.float32)
                even[h, pair] = T.cast(T.cast(x * c - y * s, T.bfloat16), T.float32)
                odd[h, pair] = T.cast(T.cast(y * c + x * s, T.bfloat16), T.float32)
            else:
                even[h, pair] = x
                odd[h, pair] = y
            absolute[h, pair] = T.max(T.abs(even[h, pair]), T.abs(odd[h, pair]))
        T.reduce_max(absolute, maximum, dim=1)
        for h, pair in T.Parallel(4, 64):
            head = group * 4 + h
            scale = T.max(maximum[h], 1e-4) / 127.0
            base = (T.cast(token, T.int64) * heads + head) * 128 + pair * 2
            out[base] = T.cast(
                T.max(-127.0, T.min(127.0, even[h, pair] / scale)), T.int8
            )
            out[base + 1] = T.cast(
                T.max(-127.0, T.min(127.0, odd[h, pair] / scale)), T.int8
            )
            if pair == 0:
                out_weights[T.cast(token, T.int64) * heads + head] = (
                    T.cast(weights[T.cast(token, T.int64) * w_s0 + head], T.float32)
                    * scale
                    * softmax_scale
                    * head_scale
                )


@tilelang_jit
def _k_kernel(
    k,
    positions,
    cs,
    norm,
    slots,
    cache,
    scales,
    k_s0: int,
    cs_s0: int,
    page_stride: int,
    block_size: int,
    blocks: int,
    ratio: int,
    eps: float,
    cs_dtype=T.float32,
    norm_dtype=T.bfloat16,
):
    n = T.dynamic("n")
    k: T.Tensor((T.dynamic("k_span", "int64"),), T.bfloat16)
    positions: T.Tensor((T.dynamic("pos_span", "int64"),), T.int64)
    cs: T.Tensor((T.dynamic("cs_span", "int64"),), cs_dtype)
    norm: T.Tensor((128,), norm_dtype)
    slots: T.Tensor((n,), T.int64)
    cache: T.Tensor((T.dynamic("cache_span", "int64"),), T.uint8)
    scales: T.Tensor((T.dynamic("scale_span", "int64"),), T.float32)
    with T.Kernel(n, threads=128) as token:
        slot = slots[token]
        position = positions[token]
        if slot >= 0 and slot < T.cast(blocks, T.int64) * block_size:
            if (position + 1) % ratio == 0:
                x = T.alloc_fragment((64,), T.float32)
                y = T.alloc_fragment((64,), T.float32)
                squares = T.alloc_fragment((64,), T.float32)
                variance = T.alloc_fragment((1,), T.float32)
                even = T.alloc_fragment((64,), T.float32)
                odd = T.alloc_fragment((64,), T.float32)
                absolute = T.alloc_fragment((64,), T.float32)
                maximum = T.alloc_fragment((1,), T.float32)
                for pair in T.Parallel(64):
                    base = T.cast(token, T.int64) * k_s0 + pair * 2
                    x[pair] = T.cast(k[base], T.float32)
                    y[pair] = T.cast(k[base + 1], T.float32)
                    squares[pair] = x[pair] * x[pair] + y[pair] * y[pair]
                T.reduce_sum(squares, variance, dim=0)
                for pair in T.Parallel(64):
                    inv = T.rsqrt(variance[0] / 128.0 + eps)
                    a = T.cast(
                        T.cast(
                            x[pair] * inv * T.cast(norm[pair * 2], T.float32),
                            T.bfloat16,
                        ),
                        T.float32,
                    )
                    b = T.cast(
                        T.cast(
                            y[pair] * inv * T.cast(norm[pair * 2 + 1], T.float32),
                            T.bfloat16,
                        ),
                        T.float32,
                    )
                    if pair >= 32:
                        base = (position // ratio) * ratio * cs_s0 + pair - 32
                        c = T.cast(cs[base], T.float32)
                        s = T.cast(cs[base + 32], T.float32)
                        even[pair] = T.cast(
                            T.cast(a * c - b * s, T.bfloat16), T.float32
                        )
                        odd[pair] = T.cast(T.cast(b * c + a * s, T.bfloat16), T.float32)
                    else:
                        even[pair] = a
                        odd[pair] = b
                    absolute[pair] = T.max(T.abs(even[pair]), T.abs(odd[pair]))
                T.reduce_max(absolute, maximum, dim=0)
                for pair in T.Parallel(64):
                    scale = T.max(maximum[0], 1e-4) / 127.0
                    base = (slot // block_size) * page_stride
                    offset = base + (slot % block_size) * 128 + pair * 2
                    cache[offset] = T.reinterpret(
                        T.cast(T.max(-127.0, T.min(127.0, even[pair] / scale)), T.int8),
                        T.uint8,
                    )
                    cache[offset + 1] = T.reinterpret(
                        T.cast(T.max(-127.0, T.min(127.0, odd[pair] / scale)), T.int8),
                        T.uint8,
                    )
                    if pair == 0:
                        # Convert element strides as Python constants to prevent the compiler
                        # from narrowing byte-address division to INT32; use INT64 for values and scales.
                        scale_base = T.cast(slot // block_size, T.int64) * (
                            page_stride // 4
                        )
                        scales[scale_base + block_size * 32 + slot % block_size] = scale


def fused_indexer_q_rope_quant_int8(
    positions,
    index_q,
    index_q_cos_sin_cache,
    index_weights,
    index_weights_softmax_scale,
    index_weights_head_scale,
    use_fp4=False,
    weights_out_dtype=torch.float32,
    *,
    output_buffers=None,
):
    """Fold V4.1 Q scales into FP32 weights, processing four heads per kernel group."""
    assert not use_fp4 and weights_out_dtype == torch.float32
    assert index_q.ndim == 3 and index_q.shape[2] == 128
    assert index_q.dtype == torch.bfloat16 and index_q.stride(2) == 1
    n, heads, _ = index_q.shape
    assert heads > 0 and heads % 4 == 0, "INT8 Q requires complete groups of 4 heads"
    assert positions.dtype == torch.int64 and positions.shape == (n,)
    assert positions.is_contiguous()
    assert index_weights.shape == (n, heads) and index_weights.stride(1) == 1
    assert index_weights.dtype in _DTYPES
    assert index_q_cos_sin_cache.ndim == 2 and index_q_cos_sin_cache.shape[1] == 64
    assert index_q_cos_sin_cache.stride(1) == 1
    assert index_q_cos_sin_cache.dtype in _DTYPES
    assert index_q.is_cuda
    assert all(
        t.device == index_q.device
        for t in (positions, index_weights, index_q_cos_sin_cache)
    )
    if output_buffers is None:
        out = torch.empty_like(
            index_q, dtype=torch.int8, memory_format=torch.contiguous_format
        )
        weights = torch.empty((n, heads), device=index_q.device, dtype=torch.float32)
    else:
        out, weights = output_buffers
        assert out.shape == index_q.shape and out.dtype == torch.int8
        assert weights.shape == (n, heads) and weights.dtype == torch.float32
        assert out.is_contiguous() and weights.is_contiguous()
        assert out.device == weights.device == index_q.device
    if n:
        _q_kernel(
            positions,
            _flat(index_q),
            _flat(index_q_cos_sin_cache),
            _flat(index_weights),
            out.view(-1),
            weights.view(-1),
            index_q.stride(0),
            index_q.stride(1),
            index_q_cos_sin_cache.stride(0),
            index_weights.stride(0),
            index_weights_softmax_scale,
            index_weights_head_scale,
            heads,
            _DTYPES[index_q_cos_sin_cache.dtype],
            _DTYPES[index_weights.dtype],
        )
    return out, weights


def indexer_k_norm_rope_store_int8(
    k_pre,
    positions,
    cos_sin_cache,
    rms_norm_weight,
    rms_norm_eps,
    k_cache,
    kv_slot_mapping,
    compress_ratio,
    use_fp4_cache,
):
    """Store V4.1 latent K1/K2 in place as paged INT8 values and FP32 scales."""
    assert not use_fp4_cache and compress_ratio in (1, 2)
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert kv_slot_mapping.ndim == 1 and kv_slot_mapping.dtype == torch.int64
    assert kv_slot_mapping.is_contiguous()
    n = kv_slot_mapping.numel()
    assert (
        positions.ndim == 1
        and positions.dtype == torch.int64
        and positions.is_contiguous()
    )
    assert n <= k_pre.shape[0] and n <= positions.numel()
    assert rms_norm_weight.shape == (128,) and rms_norm_weight.is_contiguous()
    assert rms_norm_weight.dtype in _DTYPES
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == 64
    assert cos_sin_cache.stride(1) == 1 and cos_sin_cache.dtype in _DTYPES
    assert k_cache.ndim == 3 and k_cache.dtype == torch.uint8
    assert k_cache.shape[1] > 0 and k_cache.shape[2] == 132
    assert k_cache.stride(1) == 132 and k_cache.stride(2) == 1
    assert k_cache.stride(0) >= k_cache.shape[1] * 132
    assert k_cache.stride(0) % 4 == 0 and k_cache.data_ptr() % 4 == 0
    assert k_pre.is_cuda
    assert all(
        t.device == k_pre.device
        for t in (positions, cos_sin_cache, rms_norm_weight, k_cache, kv_slot_mapping)
    )
    if not n:
        return
    flat_cache = _flat(k_cache)
    _k_kernel(
        _flat(k_pre),
        positions,
        _flat(cos_sin_cache),
        rms_norm_weight,
        kv_slot_mapping,
        flat_cache,
        flat_cache.view(torch.float32),
        k_pre.stride(0),
        cos_sin_cache.stride(0),
        k_cache.stride(0),
        k_cache.shape[1],
        k_cache.shape[0],
        compress_ratio,
        rms_norm_eps,
        _DTYPES[cos_sin_cache.dtype],
        _DTYPES[rms_norm_weight.dtype],
    )
