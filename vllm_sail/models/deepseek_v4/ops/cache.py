# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU 1.0 packed MLA cache kernels without hardware FP8 casts.

Ported from the PPU fork 86f58e178b. The indexer has a separate INT8 format;
these kernels operate only on the head-512 MLA FP8/BF16/UE8M0 layout.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _decode_e4m3fn(u):
    """Decode an E4M3FN byte (uint8) to fp32 using only uint/int/fp ops.

    Triton on SM80 cannot compile `tl.float8e4nv`, so we never load the
    FP8 dtype directly — we load uint8 and decode in software here. The
    expansion is ~6 ops per element, dwarfed by the surrounding matmul.

    E4M3FN: 1 sign + 4 exp (bias 7) + 3 mantissa.  No infinities.
    Subnormal (exp=0): value = (-1)^s * (mant/8) * 2^(1 - 7)
    Normal           : value = (-1)^s * (1 + mant/8) * 2^(exp - 7)
    NaN at 0x7F/0xFF is decoded numerically as ±480 — sparse-MLA inputs
    never hit this so the loss of NaN propagation is acceptable.
    """
    sign = u >> 7
    exp_bits = ((u >> 3) & 0x0F).to(tl.int32)
    mant = (u & 0x07).to(tl.int32)
    is_normal = exp_bits != 0
    sign_f = tl.where(sign != 0, -1.0, 1.0)
    mant_f = tl.where(
        is_normal,
        (8 + mant).to(tl.float32) * 0.125,
        mant.to(tl.float32) * 0.125,
    )
    # Subnormals: real exponent = 1 - bias.
    eff_exp = tl.where(is_normal, exp_bits, 1)
    factor = tl.exp2((eff_exp - 7).to(tl.float32))
    return sign_f * mant_f * factor


@triton.jit
def _encode_e4m3fn(x):
    """Encode pre-clamped fp32 to FP8 E4M3FN as uint8 (SM80 compatible).

    Software replacement for ``x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)``
    that Triton refuses to compile on SM80. Pure integer bit manipulation, no
    fp8 hardware dependency.

    E4M3FN: 1 sign + 4 exp (bias 7) + 3 mantissa. No infinities; only NaN at
    0x7F/0xFF. Input must be pre-clamped to [-448, 448]; out-of-range values
    are NOT saturated here.

    Rounding mode: round-to-nearest-even (RNE), matching NVIDIA hardware
    ``tl.float8e4nv`` cast and PyTorch ``torch.float8_e4m3fn`` cast.

    Layout summary (positive value v != 0):
        unbiased_exp = floor(log2(v))
        fp32 stored exp = unbiased_exp + 127
        fp8  stored exp = unbiased_exp + 7   (= fp32_exp - 120)
        normal range :  fp8 stored exp in [1, 15]   <=> fp32_exp in [121, 135]
        subnormal    :  fp8 stored exp == 0        <=> fp32_exp in [117, 120]
        underflow    :  fp32_exp < 117 (-> 0)
    """
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 31) & 1
    abs_bits = bits & 0x7FFFFFFF

    fp32_exp = (abs_bits >> 23) & 0xFF  # 8-bit biased exponent
    fp32_mant = abs_bits & 0x7FFFFF  # 23-bit mantissa

    is_zero = abs_bits == 0
    # Target fp8 biased exponent for the normal path.
    fp8_exp_normal = fp32_exp - 120
    is_subnorm = (fp8_exp_normal <= 0) & (~is_zero)

    # ---- Normal path: take top 3 bits of mantissa with RNE rounding. ----
    # Truncated bits = fp32_mant[19:0]; halfway sentinel = 0x80000 (bit 19).
    truncated = fp32_mant & 0xFFFFF
    halfway = 0x80000
    lsb = (fp32_mant >> 20) & 1
    round_up_n = (truncated > halfway) | ((truncated == halfway) & (lsb == 1))
    normal_mant = (fp32_mant >> 20) + round_up_n.to(tl.int32)
    # Mantissa overflow (4 -> carry into exponent).
    mant_ovf = normal_mant >= 8
    normal_exp_out = fp8_exp_normal + mant_ovf.to(tl.int32)
    normal_mant_out = tl.where(mant_ovf, 0, normal_mant)

    # ---- Subnormal path: shift the implicit-1 mantissa to fit fp8 denormal. ----
    # full_mant = (1 << 23) | fp32_mant has 24 significant bits.
    # value = full_mant * 2^(fp32_exp - 127 - 23)
    # Subnormal fp8 represents value as fp8_mant * 2^(-9), so
    # fp8_mant = full_mant >> (141 - fp32_exp) with RNE rounding.
    full_mant = 0x800000 | fp32_mant
    sub_shift = 141 - fp32_exp  # in [21, ...]; <=20 means normal path
    # Clamp to a safe range to avoid undefined shift behavior; >=24 -> 0.
    sub_shift_safe = tl.minimum(tl.maximum(sub_shift, 1), 31)
    # Round bit position = sub_shift_safe - 1; sticky bits below it.
    one_i32 = tl.full((), 1, tl.int32)
    round_pos = sub_shift_safe - 1
    round_mask = one_i32 << round_pos
    sticky_mask = round_mask - 1
    sub_round_bit = (full_mant & round_mask) != 0
    sub_sticky = (full_mant & sticky_mask) != 0
    sub_truncated = full_mant >> sub_shift_safe
    sub_lsb = sub_truncated & 1
    sub_round_up = sub_round_bit & (sub_sticky | (sub_lsb == 1))
    sub_mant_out = sub_truncated + sub_round_up.to(tl.int32)
    # Subnormal rounding may carry into the (implicit) leading bit -> becomes
    # the smallest normal (exp=1, mant=0).
    sub_carry_to_normal = sub_mant_out >= 8
    sub_exp_final = tl.where(sub_carry_to_normal, 1, 0)
    sub_mant_final = tl.where(sub_carry_to_normal, 0, sub_mant_out & 0x7)

    # ---- Combine paths. ----
    out_exp = tl.where(is_subnorm, sub_exp_final, normal_exp_out)
    out_mant = tl.where(is_subnorm, sub_mant_final, normal_mant_out)
    out_exp = tl.where(is_zero, 0, out_exp)
    out_mant = tl.where(is_zero, 0, out_mant)

    # Saturate exponent into the 4-bit field (input is pre-clamped to <=448,
    # so this only protects against the rounding carry at the boundary).
    out_exp = tl.minimum(tl.maximum(out_exp, 0), 15)
    out_mant = out_mant & 0x7

    byte = (sign << 7) | (out_exp << 3) | out_mant
    return byte.to(tl.uint8)


@triton.jit
def _dequantize_and_gather_k_kernel_sm80(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    """SM80 variant of `_dequantize_and_gather_k_kernel`. Replaces the
    `tl.float8e4nv` bitcast with the software `_decode_e4m3fn` decoder
    from PR 38476 (uint8 → fp32 in ~6 ops/elt). Same launch shape and
    layout as the SM90+ kernel."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block
            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)
                # Software fp8e4nv → fp32 (no tl.float8e4nv reference).
                x_float = _decode_e4m3fn(x_uint8)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)
                x_dequant = x_float * scale
                tl.store(
                    output_row_ptr + offsets,
                    x_dequant.to(tl.bfloat16),
                    mask=mask,
                )

        bf16_output_offset = fp8_dim
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def _dequantize_and_gather_k_cache_triton(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """SM80 Triton dispatch — uses _decode_e4m3fn instead of the
    fp8e4nv bitcast that Triton refuses to compile on SM80."""
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel_sm80[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=FP8_MAX,
        n_quant_blocks=7,
    )


def _dequant_gather_sm80_op(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    _dequantize_and_gather_k_cache_triton(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )


def _dequant_gather_sm80_op_fake(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    return None


@triton.jit
def _fused_kv_compress_norm_rope_insert_sparse_attn_sm80(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,  # 448.0
    QUANT_BLOCK: tl.constexpr,  # 64 for DeepseekV4
    TOKEN_STRIDE: tl.constexpr,  # 576 for DeepseekV4
    SCALE_DIM: tl.constexpr,  # 8 for DeepseekV4 (7 real + 1 pad)
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """SM80 variant of the sparse-attn compressor kernel (head=512).

    Identical to _fused_kv_compress_norm_rope_insert_sparse_attn except
    the tl.float8e4nv cast is replaced by the software _encode_e4m3fn helper.
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    # ── Softmax + weighted sum ───────────────────────────────────────
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers ────────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # FP8 UE8M0 quant: cast fp32 → bf16 → fp32 before quant to match reference.
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # SM80: software fp32 → fp8e4m3fn → uint8 via _encode_e4m3fn.
    x_uint8 = _encode_e4m3fn(x_clamped)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    # Register-based GPT-J RoPE in fp32.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # [TRITON_BLOCK_SIZE] fp32

    # Store rotated rope portion as bf16 into the cache's bf16 area.
    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


def compress_mla_rope_store_fp8(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Compress into the packed FP8/BF16 main attention cache on PPU 1.0."""
    assert head_dim == 512 and rope_head_dim == 64 and not use_fp4_cache
    assert (quant_block, token_stride, scale_dim) == (64, 576, 8)
    kernel = _fused_kv_compress_norm_rope_insert_sparse_attn_sm80
    num_warps = 4

    kernel[(num_actual,)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # KV cache
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # paged KV cache block size (tokens per block)
        # constexprs
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **pdl_kwargs,
    )


def register_ops() -> None:
    """Register before tracing; module import alone does not mutate torch.ops."""
    global _registered
    if _registered:
        return
    direct_register_custom_op(
        op_name="ppu_deepseek_v4_dequant_gather",
        op_func=_dequant_gather_sm80_op,
        mutates_args=["out"],
        fake_impl=_dequant_gather_sm80_op_fake,
    )
    _registered = True


_registered = False
