# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 DSA kernels that also run on PPU parts without hardware E4M3.

Triton on PPU 1.0 (ZW810E, SM 8.0) rejects ``fp8e4nv`` everywhere: casts,
bitcasts, FP8 ``tl.dot`` and even an FP8 tensor in a kernel signature. The
Step5 indexer keeps its E4M3 data (summary cache, indexer query) bit-identical
by storing it as bytes and converting in software on those parts:

* encode: round-to-nearest-even with saturation to +-448, as the hardware
  ``cvt.rn.satfinite`` path does;
* decode: exact E4M3 -> FP16, so the FP16 ``tl.dot`` with FP32 accumulation
  forms the same products as the FP8 dot; only summation order may differ.

Each kernel picks the native path at compile time when the target has E4M3
(``cuda_capability_geq(8, 9)``), so SM 8.9+ compiles the upstream math.

``VLLM_STEP5_INDEXER_FP8_AS_FP16=1`` keeps the E4M3 quantization but stores
FP16. Q/summary keep their activation and BF16 rounding; the FP16 dot with
FP32 accumulation is numerically equivalent to the E4M3 path (except for
summation order).

Kernels are copies of upstream's with ``# PPU MODIFICATION`` marking every
change, and are installed by
``vllm_sail/patch/enhancement/models/step5_soft_fp8.py`` and
``vllm_sail/patch/enhancement/models/step5_indexer_fp16.py``.
"""

from __future__ import annotations

import torch
from vllm.models.step5.common.ops.indexer_norm_rope import (
    _layer_norm_stats,
    _rms_norm_variance,
)
from vllm.models.step5.common.ops.summary_update import (
    _maximum,
    _update_decode_values,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_sail import envs

try:
    from triton.language.target_info import cuda_capability_geq
except ImportError:  # Triton-less test environments never compile kernels.
    cuda_capability_geq = None

libdevice = tl.extra.cuda.libdevice


def indexer_fp16() -> bool:
    """Mode 1 stores the E4M3-quantized values as FP16."""
    return envs.VLLM_STEP5_INDEXER_FP8_AS_FP16


def indexer_query_dtype() -> torch.dtype:
    """Storage dtype of the indexer query (and of the summary cache's values)."""
    return torch.float16 if indexer_fp16() else torch.float8_e4m3fn


@triton.jit
def e4m3_to_fp16(raw):
    """Decode E4M3FN bytes exactly into FP16 (0x7F/0xFF -> NaN)."""
    bits = raw.to(tl.int32)
    magnitude = bits & 0x7F
    half = (((bits & 0x80) << 8) | (magnitude << 7)).to(tl.int16)
    value = half.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return tl.where(magnitude == 0x7F, float("nan"), value).to(tl.float16)


@triton.jit
def float_to_e4m3(x):
    """Encode to E4M3FN bytes: RNE, saturating to +-448, NaN -> 0x7F."""
    x = x.to(tl.float32)
    sign = (x.to(tl.int32, bitcast=True) >> 24) & 0x80
    magnitude = tl.minimum(tl.abs(x), 448.0)
    bits = magnitude.to(tl.int32, bitcast=True)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - (120 << 3)
    subnormal = libdevice.rint(magnitude * 512.0).to(tl.int32)
    code = tl.where(magnitude < 0.015625, subnormal, normal) | sign
    return tl.where(x != x, 0x7F, code).to(tl.uint8)


@triton.jit
def encode_e4m3(x):
    """E4M3 bytes of ``x``; hardware conversion where the target has it."""
    if cuda_capability_geq(8, 9):
        code = x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        code = float_to_e4m3(x)
    return code


@triton.jit
def e4m3_dot(a, b):
    """Dot of E4M3-byte or FP16 tiles with FP32 accumulation."""
    if a.dtype == tl.float16:
        out = tl.dot(a, b, out_dtype=tl.float32)
    elif cuda_capability_geq(8, 9):
        out = tl.dot(
            a.to(tl.float8e4nv, bitcast=True),
            b.to(tl.float8e4nv, bitcast=True),
            out_dtype=tl.float32,
            max_num_imprecise_acc=0,
        )
    else:
        out = tl.dot(e4m3_to_fp16(a), e4m3_to_fp16(b), out_dtype=tl.float32)
    return out


@triton.jit
def _indexer_store_value(value, HALF: tl.constexpr):
    """Q and summary are both quantized to E4M3, then stored as bytes or FP16."""
    if HALF:
        stored = e4m3_to_fp16(encode_e4m3(value))
    else:
        stored = encode_e4m3(value)
    return stored


# Copied from vllm.models.step5.common.ops.summary_update._store_summary.
@triton.jit
def _store_summary(
    Mean,
    slot,
    dim,
    numerator,
    denominator,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write the summary of the region holding KV ``slot``; blocks of ``Mean``
    are ``MEAN_BLOCK`` elements apart."""
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    # PPU MODIFICATION: begin — E4M3-quantized values stored as bytes or FP16.
    stored = _indexer_store_value(summary, Mean.dtype.element_ty == tl.float16)
    # PPU MODIFICATION: end
    address = slot // BLOCK_SIZE * MEAN_BLOCK + slot % BLOCK_SIZE // 8 * 256
    tl.store(Mean + address + dim, stored)


# Copied from vllm.models.step5.common.ops.indexer_norm_rope._indexer_norm_rope.
@triton.jit(do_not_specialize=["CSA_ROWS", "TOKENS"])
def _indexer_norm_rope(
    Q,
    K,
    W,
    QWeight,
    KWeight,
    KBias,
    Rope,
    Positions,
    QOut,
    KOut,
    WOut,
    Z,
    Slots,
    Mean,
    Ring,
    RingBase,
    UPDATE_CSA: tl.constexpr,
    CSA_ROWS,
    Z_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    MEAN_BLOCK_SIZE: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    TOKENS,
    ROTARY: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_BIAS: tl.constexpr,
    NATIVE_ROPE: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1) // PAD_HEADS
    head = tl.program_id(1) % PAD_HEADS
    dim = tl.arange(0, 256)
    if (group < GROUPS) & (head < HEADS):
        x = tl.load(Q + token * Q_STRIDE + (group * HEADS + head) * 256 + dim).to(
            tl.float32
        )
        variance = _rms_norm_variance(x) if WEIGHT_BIAS == 1 else tl.sum(x * x, 0) / 256
        inv = tl.rsqrt(variance + EPS)
        weight = tl.load(QWeight + dim).to(tl.float32) + WEIGHT_BIAS
        normalized = x * inv
        if WEIGHT_BIAS == 0:
            normalized = normalized.to(QWeight.dtype.element_ty).to(tl.float32)
        x = normalized * weight
        # Preserve the activation-typed norm output before RoPE.
        x = x.to(Q.dtype.element_ty).to(tl.float32)
    elif group == GROUPS:
        x = tl.load(K + token * K_STRIDE + dim).to(tl.float32)
        mean, variance = _layer_norm_stats(x)
        centered = x - mean
        inv = tl.rsqrt(variance + EPS)
        weight = tl.load(KWeight + dim).to(tl.float32)
        bias = tl.load(KBias + dim).to(tl.float32)
        x = tl.fma(weight, centered * inv, bias)
        x = x.to(K.dtype.element_ty).to(tl.float32)
    else:
        x = tl.full((256,), 0, tl.float32)

    if (head < HEADS) | (group == GROUPS):
        position = tl.load(Positions + token)
        half = ROTARY // 2
        partner_dim = tl.where(
            dim < half, dim + half, tl.where(dim < ROTARY, dim - half, dim)
        )
        partner = tl.gather(x, partner_dim, 0)
        phase = dim % half
        cos = tl.load(Rope + position * ROPE_STRIDE + phase)
        sin = tl.load(Rope + position * ROPE_STRIDE + half + phase)
        first = x * cos.to(tl.float32)
        second = tl.where(dim < half, -partner, partner) * sin.to(tl.float32)
        if NATIVE_ROPE:
            first = first.to(Q.dtype.element_ty).to(tl.float32)
            second = second.to(Q.dtype.element_ty).to(tl.float32)
        rotated = first + second
        x = tl.where(dim < ROTARY, rotated, x)
    if group == GROUPS:
        tl.store(KOut + token * 256 + dim, x)
        if UPDATE_CSA:  # noqa: SIM102 - Guard absent cache pointers at compile time.
            if token < CSA_ROWS:
                slot = tl.load(Slots + token * SLOT_STRIDE)
                z = tl.load(Z + token * Z_STRIDE + dim).to(tl.float32)
                normalized_key = x.to(K.dtype.element_ty).to(tl.float32)
                _update_decode_values(
                    slot,
                    tl.load(Positions + token),
                    tl.load(RingBase + token),
                    dim,
                    normalized_key,
                    z,
                    Mean,
                    Ring,
                    RING_CAPACITY,
                    MEAN_BLOCK,
                    MEAN_BLOCK_SIZE,
                )
    else:
        # RoPE also materializes BF16/FP16 before the FP8 conversion.
        rounded = x.to(Q.dtype.element_ty)
        offset = (group * TOKENS + token) * PAD_HEADS + head
        # PPU MODIFICATION: begin — FP16 output keeps the E4M3-quantized values.
        if QOut.dtype.element_ty == tl.uint8:
            tl.store(QOut + offset * 256 + dim, _indexer_store_value(rounded, False))
        elif QOut.dtype.element_ty == tl.float16:
            tl.store(QOut + offset * 256 + dim, _indexer_store_value(rounded, True))
        else:
            tl.store(QOut + offset * 256 + dim, rounded)
        # PPU MODIFICATION: end
        w = tl.load(W + token * W_STRIDE + group * HEADS + head, head < HEADS, other=0)
        tl.store(WOut + offset, w.to(tl.float32) * HEADS**-0.5)


# Copied from vllm.models.step5.common.ops.sparse_indexer_scores._direct_scores.
@triton.jit
def _direct_scores(
    Q,
    W,
    Mean,
    Table,
    Slots,
    SeqLens,
    Scores,
    Active,
    Positions,
    Current,
    Lengths,
    TopKLengths,
    Q_TOKEN: tl.constexpr,
    Q_GROUP: tl.constexpr,
    W_TOKEN: tl.constexpr,
    W_GROUP: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    TILE: tl.constexpr,
    WORKERS: tl.constexpr,
    ROW_GROUPS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
):
    worker = tl.program_id(0)
    row = tl.program_id(1)
    # Rows are (token, KV group) pairs; each request has QUERY_LEN tokens.
    # A token at offset i of its request sees seq_len - (QUERY_LEN - 1 - i).
    token = row // ROW_GROUPS
    kv_group = row % ROW_GROUPS
    request = token // QUERY_LEN
    active = tl.load(Slots + token * SLOT_STRIDE) >= 0
    seq = tl.load(SeqLens + request * SEQ_STRIDE).to(tl.int64) - (
        QUERY_LEN - 1 - token % QUERY_LEN
    )
    position = tl.where(active, seq - 1, 0)
    current = position >> 3
    length = current + 1
    if worker == 0:
        tl.store(Active + row, active)
        tl.store(Positions + row, position)
        tl.store(Current + row, current)
        tl.store(Lengths + row, length)
        tl.store(TopKLengths + row, current)
    live_tiles = tl.cdiv(tl.minimum(tl.maximum(length, 0), CAPACITY).to(tl.int32), TILE)
    for tile in range(worker, live_tiles, WORKERS):
        region = tile * TILE + tl.arange(0, TILE)
        col = region // REGIONS_PER_BLOCK
        physical = tl.load(
            Table + request * TABLE_ROW + col, col < TABLE_COLS, other=-1
        )
        valid = (region < CAPACITY) & (physical >= 0) & (physical < CACHE_PAGES)
        d = tl.arange(0, 256)
        h = tl.arange(0, 16)
        raw = tl.load(
            Mean
            + (physical.to(tl.int64) * MEAN_BLOCK + region % REGIONS_PER_BLOCK * 256)[
                :, None
            ]
            + d[None, :],
            valid[:, None],
            other=0,
        )
        # PPU MODIFICATION: begin — an FP16 query is read as FP16.
        if Q.dtype.element_ty == tl.float16:
            query_ptr = Q
        else:
            query_ptr = Q.to(tl.pointer_type(tl.uint8))
        query = tl.load(
            query_ptr
            + token * Q_TOKEN
            + kv_group * Q_GROUP
            + h[None, :] * 256
            + d[:, None],
            h[None, :] < HEADS,
            other=0,
        )
        # PPU MODIFICATION: end
        # PPU MODIFICATION: begin — software E4M3 dot below SM 8.9.
        dot = e4m3_dot(raw, query)
        # PPU MODIFICATION: end
        dot = tl.maximum(dot, 0.0)
        w = tl.load(W + token * W_TOKEN + kv_group * W_GROUP + h, h < HEADS, other=0)
        # FI's FP32 epilogue has four independent FMA chains followed by
        # left-associated addition. Keep this order, not a tree reduction.
        sums = tl.full((TILE, 4), 0, tl.float32)
        lane = tl.arange(0, 4)
        for group in tl.static_range(HEADS // 4):
            index = group * 4 + lane
            a = tl.gather(dot, tl.broadcast_to(index[None, :], (TILE, 4)), 1)
            weights = tl.gather(w, index, 0)
            sums = tl.fma(a, weights[None, :], sums)
        a = tl.gather(sums, tl.full((TILE, 1), 0, tl.int32), 1).reshape((TILE,))
        b = tl.gather(sums, tl.full((TILE, 1), 1, tl.int32), 1).reshape((TILE,))
        c = tl.gather(sums, tl.full((TILE, 1), 2, tl.int32), 1).reshape((TILE,))
        e = tl.gather(sums, tl.full((TILE, 1), 3, tl.int32), 1).reshape((TILE,))
        result = ((a + b) + c) + e
        tl.store(Scores + row * SCORE_ROW + region, result, region < CAPACITY)


# Copied from vllm.models.step5.common.ops.sparse_indexer_scores._paged_scores_sm90.
@triton.jit(do_not_specialize=["TABLE_ROW", "REGIONS"])
def _paged_scores_sm90(
    Q,
    KV,
    W,
    Lengths,
    Table,
    Output,
    Q_BATCH: tl.constexpr,
    Q_NEXT: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    KV_PAGE: tl.constexpr,
    TABLE_ROW,
    TABLE_COL: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NEXT_N
    region = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    valid = (region < REGIONS) & (region < length)
    physical = tl.load(
        Table + batch * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
        valid,
        other=0,
    ).to(tl.int64)
    d = tl.arange(0, 256)
    h = tl.arange(0, 16)
    key = tl.load(
        KV
        + physical[:, None] * KV_PAGE
        + (region % PAGE_SIZE)[:, None] * 256
        + d[None, :],
        valid[:, None],
        other=0,
    )
    # PPU MODIFICATION: begin — an FP16 query is read as FP16.
    if Q.dtype.element_ty == tl.float16:
        query_ptr = Q
    else:
        query_ptr = Q.to(tl.pointer_type(tl.uint8))
    query = tl.load(
        query_ptr
        + batch * Q_BATCH
        + (row % NEXT_N) * Q_NEXT
        + h[None, :] * Q_HEAD
        + d[:, None],
        h[None, :] < HEADS,
        other=0,
    )
    # PPU MODIFICATION: end
    # PPU MODIFICATION: begin — software E4M3 dot below SM 8.9.
    dot = e4m3_dot(key, query)
    # PPU MODIFICATION: end
    dot = tl.maximum(dot, 0.0)
    weights = tl.load(W + row * W_ROW + h * W_HEAD, h < HEADS, other=0)
    # Match the four independent FP32 FMA chains used by the decode indexer.
    sums = tl.full((TILE, 4), 0, tl.float32)
    lane = tl.arange(0, 4)
    for group in tl.static_range(HEADS // 4):
        index = group * 4 + lane
        a = tl.gather(dot, tl.broadcast_to(index[None, :], (TILE, 4)), 1)
        w = tl.gather(weights, index, 0)
        sums = tl.fma(a, w[None, :], sums)
    a = tl.gather(sums, tl.full((TILE, 1), 0, tl.int32), 1).reshape((TILE,))
    b = tl.gather(sums, tl.full((TILE, 1), 1, tl.int32), 1).reshape((TILE,))
    c = tl.gather(sums, tl.full((TILE, 1), 2, tl.int32), 1).reshape((TILE,))
    e = tl.gather(sums, tl.full((TILE, 1), 3, tl.int32), 1).reshape((TILE,))
    # PPU MODIFICATION: begin — an FP32 scale spans two FP16 page elements.
    SCALE_STEP: tl.constexpr = 2 if KV.dtype.element_ty == tl.float16 else 4
    scales = tl.load(
        (
            KV
            + physical * KV_PAGE
            + PAGE_SIZE * 256
            + (region % PAGE_SIZE) * SCALE_STEP
        ).to(tl.pointer_type(tl.float32)),
        valid,
        other=0,
    )
    # PPU MODIFICATION: end
    result = (((a + b) + c) + e) * scales
    tl.store(
        Output + row * REGIONS + region,
        tl.where(valid, result, float("-inf")),
        region < REGIONS,
    )


def paged_mqa_logits_sm90(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    schedule_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sparse_indexer_scores.paged_mqa_logits_sm90`` that also accepts the
    FP16 storage of ``VLLM_STEP5_INDEXER_FP8_AS_FP16``.

    A packed page holds 32 keys followed by one FP32 scale per key: 260 bytes
    per key for E4M3 bytes, 258 FP16 elements per key for FP16.
    """
    # Copied from vllm.models.step5.common.ops.sparse_indexer_scores.
    batch, next_n, heads, dim = q.shape
    # PPU MODIFICATION: begin — E4M3 bytes or their exact FP16 values.
    assert (q.dtype, kv_fused.dtype) in (
        (torch.float8_e4m3fn, torch.uint8),
        (torch.float16, torch.float16),
    )
    assert dim == 256
    assert heads in (4, 8, 16) and q.stride(-1) == 1
    assert kv_fused.is_contiguous()
    assert kv_fused.shape[2:] == (1, 256 + 4 // kv_fused.element_size())
    if q.dtype == torch.float8_e4m3fn:
        q = q.view(torch.uint8)
    # PPU MODIFICATION: end
    assert weights.shape == (batch * next_n, heads)
    output = torch.empty(
        (batch * next_n, max_seq_len), dtype=torch.float32, device=q.device
    )
    if batch and max_seq_len:
        _paged_scores_sm90[(batch * next_n, triton.cdiv(max_seq_len, 32))](
            q,
            kv_fused,
            weights,
            seq_lens,
            block_tables,
            output,
            *q.stride()[:3],
            *weights.stride(),
            kv_fused.stride(0),
            *block_tables.stride(),
            seq_lens.stride(0),
            max_seq_len,
            next_n,
            heads,
            kv_fused.shape[1],
            32,
            num_warps=4,
        )
    return output


# Copied from vllm.models.step5.common.ops.sparse_indexer._copy_summary_page.
@triton.jit
def _copy_summary_page(
    Mean,
    Table,
    Packed,
    Scales,
    request,
    page,
    REGIONS,
    PAGES,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
):
    region = page * 32 + tl.arange(0, 32)
    col = region // REGIONS_PER_BLOCK
    physical = tl.load(
        Table + request * TABLE_STRIDE + col,
        col < TABLE_COLS,
        other=-1,
    )
    valid = (region < REGIONS) & (physical >= 0) & (physical < CACHE_PAGES)
    dim = tl.arange(0, 256)
    values = tl.load(
        Mean
        + physical[:, None].to(tl.int64) * MEAN_BLOCK
        + region[:, None] % REGIONS_PER_BLOCK * 256
        + dim[None, :],
        valid[:, None],
        other=0,
    )
    # Large graph batches can pack more than 2 GiB of summary bytes.
    output_page = request.to(tl.int64) * PAGES + page
    # PPU MODIFICATION: begin — a page is 32 keys plus 32 FP32 scales in
    # elements of the summary dtype (bytes, or FP16).
    ELEMENT_BYTES: tl.constexpr = 2 if Mean.dtype.element_ty == tl.float16 else 1
    PAGE_ELEMENTS: tl.constexpr = 32 * (256 + 4 // ELEMENT_BYTES)
    tl.store(
        Packed
        + output_page * PAGE_ELEMENTS
        + tl.arange(0, 32)[:, None] * 256
        + dim[None, :],
        values,
    )
    tl.store(
        Scales + output_page * (PAGE_ELEMENTS * ELEMENT_BYTES // 4) + tl.arange(0, 32),
        1.0,
    )
    # PPU MODIFICATION: end


# Copied from vllm.models.step5.common.ops.sparse_indexer._copy_summary_tile.
@triton.jit
def _copy_summary_tile(
    Mean,
    Table,
    Packed,
    Scales,
    request,
    page,
    REGIONS,
    PAGES,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TILE_PAGES: tl.constexpr,
):
    offset = tl.arange(0, TILE_PAGES * 32)
    region = page * 32 + offset
    packed_page = page + offset // 32
    col = region // REGIONS_PER_BLOCK
    physical = tl.load(Table + request * TABLE_STRIDE + col, col < TABLE_COLS, other=-1)
    valid = (region < REGIONS) & (physical >= 0) & (physical < CACHE_PAGES)
    # PPU MODIFICATION: begin — copy 32-bit words of 4 bytes or 2 FP16 values.
    ELEMENT_BYTES: tl.constexpr = 2 if Mean.dtype.element_ty == tl.float16 else 1
    WORD_ELEMENTS: tl.constexpr = 4 // ELEMENT_BYTES
    KEY_WORDS: tl.constexpr = 256 // WORD_ELEMENTS
    PAGE_WORDS: tl.constexpr = 32 * (KEY_WORDS + 1)
    dim = tl.arange(0, KEY_WORDS)
    values = tl.load(
        Mean.to(tl.pointer_type(tl.uint32))
        + physical[:, None].to(tl.int64) * (MEAN_BLOCK // WORD_ELEMENTS)
        + region[:, None] % REGIONS_PER_BLOCK * KEY_WORDS
        + dim[None, :],
        valid[:, None],
        other=0,
    )
    output_page = request.to(tl.int64) * PAGES + packed_page
    tl.store(
        Packed.to(tl.pointer_type(tl.uint32))
        + output_page[:, None] * PAGE_WORDS
        + (offset[:, None] % 32) * KEY_WORDS
        + dim[None, :],
        values,
        (packed_page < PAGES)[:, None],
    )
    tl.store(Scales + output_page * PAGE_WORDS + offset % 32, 1.0, packed_page < PAGES)
    # PPU MODIFICATION: end


class ByteFP8Launch:
    """Launch-site stand-in for an upstream kernel that takes E4M3 tensors.

    Upstream launchers call ``kernel[grid](*args)``. On PPU this forwards to
    ``ppu`` with the E4M3 tensors at positions ``byte_args`` viewed as
    ``uint8`` (PPU 1.0 Triton rejects FP8 in a kernel signature); elsewhere
    it launches the upstream kernel with unchanged arguments. FP16 tensors at
    ``byte_args`` (``VLLM_STEP5_INDEXER_FP8_AS_FP16``) always launch ``ppu``,
    as upstream kernels only read E4M3. Other FP8 arguments (e.g. an FP8 KV
    cache the kernel stores into by value) are left alone.
    """

    def __init__(self, upstream, ppu, byte_args: tuple[int, ...]):
        self.upstream = upstream
        self.ppu = ppu
        self.byte_args = byte_args
        self.__name__ = getattr(upstream, "__name__", type(self).__name__)
        self.__doc__ = getattr(upstream, "__doc__", None)

    def __getitem__(self, grid):
        byte_args = self.byte_args

        def run(*args, **kwargs):
            args = list(args)
            half = any(
                isinstance(args[index], torch.Tensor)
                and args[index].dtype == torch.float16
                for index in byte_args
            )
            if not (half or current_platform.is_ppu()):
                return self.upstream[grid](*args, **kwargs)
            for index in byte_args:
                arg = args[index]
                if isinstance(arg, torch.Tensor) and arg.dtype == torch.float8_e4m3fn:
                    args[index] = arg.view(torch.uint8)
            return self.ppu[grid](*args, **kwargs)

        return run

    def __getattr__(self, name):
        return getattr(self.upstream, name)
