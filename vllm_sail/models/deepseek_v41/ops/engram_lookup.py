# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""810E INT8/FP32 lookup and FP8/UE8M0 software compatibility for V4.1 Engram.

Based on vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6:
models/deepseek_v41/common/engram.py; preserve sharding, allocation, and sharing policies.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_sail.models.deepseek_v4.ops.cache import _decode_e4m3fn


@triton.jit
def _decode_engram_e4m3fn(value):
    decoded = _decode_e4m3fn(value)
    return tl.where((value & 0x7F) == 0x7F, float("nan"), decoded)


@triton.jit(
    do_not_specialize=[
        "vocab_start",
        "vocab_end",
        "num_rows",
        "ids_stride_t",
        "ids_stride_h",
        "GRID",
    ]
)
def _engram_lookup_kernel(
    weight,
    scales,
    ids,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    INT8_TABLE: tl.constexpr,
    FP32_SCALES: tl.constexpr,
    GRID,
):
    """Gather fp8 rows, apply their ue8m0 block scales, write bf16.

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head * ids_stride_h,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )
        if not INT8_TABLE:
            values = _decode_engram_e4m3fn(values)
        if FP32_SCALES:
            scale = tl.load(scales + local, mask=owned, other=0.0)[:, None]
        else:
            scale = tl.load(
                scales + local[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
                mask=owned[:, None],
                other=0,
            )
            # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
            scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(
            out + rows[:, None] * DIM + cols[None, :],
            (values.to(tl.float32) * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


def lookup(
    self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
) -> None:
    """Look up local heads of [T, heads] into [T, local_heads, dim] bf16.

    `background` limits the grid to leave SMs for concurrent work.
    """
    # Upstream NgramHashState returns INT32; promote to INT64 before row-address multiplication.
    if (
        indices.ndim != 2
        or indices.shape[1] != self.n_hash_cols
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Engram indices must be [tokens, hash_heads] INT32/INT64, "
            f"actual shape={tuple(indices.shape)}, dtype={indices.dtype}"
        )
    if (
        out.shape != (indices.shape[0], self.part_n_hash_cols, self.dim)
        or out.dtype != torch.bfloat16
    ):
        raise ValueError("Engram output shape or dtype does not match")
    if (
        not out.is_contiguous()
        or indices.device != out.device
        or out.device.type != "cuda"
    ):
        raise ValueError(
            "Engram indices and output must share a device, and output must be contiguous"
        )
    rows = indices.shape[0] * self.part_n_hash_cols
    if not rows:
        return
    weight, scales = self._storage()
    if weight.ndim != 2 or weight.shape != (
        self.vocab_end_idx - self.vocab_start_idx,
        self.dim,
    ):
        raise ValueError("Engram table shape does not match the current head shard")
    if weight.device != out.device or scales.device != out.device:
        raise ValueError(
            "Engram tables must be tensors or UVA views on the output device"
        )
    int8_table = weight.dtype == torch.int8
    if int8_table:
        if scales.dtype != torch.float32 or scales.shape != (weight.shape[0], 1):
            raise ValueError("1017 INT8 Engram requires FP32 per-row scales")
    elif weight.dtype == torch.float8_e4m3fn:
        weight = weight.view(torch.uint8)
        if scales.dtype == torch.float32:
            if scales.shape != (weight.shape[0], 1) or self.block_size != self.dim:
                raise ValueError(
                    "Channelwise FP8 Engram requires one FP32 scale per row"
                )
        elif scales.dtype == torch.uint8:
            if scales.shape != (weight.shape[0], self.dim // self.block_size):
                raise ValueError("FP8 Engram requires UE8M0 block scales")
        else:
            raise ValueError("Unsupported FP8 Engram scale dtype")
    else:
        raise ValueError("Unsupported Engram weight/scale dtype combination")
    if (
        not weight.is_contiguous()
        or not scales.is_contiguous()
        or not out.is_contiguous()
    ):
        raise ValueError("Engram tables, scales, and output must be contiguous")
    # The table dwarfs TLB reach, so a persistent grid near the SM count
    # beats one program per row; halve it to leave SMs for the main stream.
    tiles = triton.cdiv(rows, 16)
    grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
    _engram_lookup_kernel[(grid,)](
        weight,
        scales,
        indices,
        out,
        self.vocab_start_idx,
        self.vocab_end_idx,
        rows,
        indices.stride(0),
        indices.stride(1),
        HEAD_START=self.head_start,
        LOCAL_HEADS=self.part_n_hash_cols,
        TOTAL_HEADS=self.n_hash_cols,
        DIM=self.dim,
        QUANT_BLOCK=self.block_size,
        BLOCK_R=16,
        INT8_TABLE=int8_table,
        FP32_SCALES=scales.dtype == torch.float32,
        GRID=grid,
    )
