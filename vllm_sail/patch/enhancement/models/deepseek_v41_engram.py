# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve V4.1 Engram lookup semantics for PPU FP32 per-row scales."""

import re
from functools import cache

import torch
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_sail.models.deepseek_v41.config import int8_w8a8_enabled
from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.32.0,<0.33.0"
_FACTORY = "vllm.models.deepseek_v41.common.engram"


@patch(
    "vllm.models.deepseek_v41.common.engram",
    reason="PPU channelwise Engram exports carry FP32 row scales rather than E8M0 block scales.",
    affected_versions=_AFFECTED,
    remove_when="Upstream Engram lookup supports FP32 scales alongside E8M0 scales.",
)
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
    sorted_dst,
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
    GRID,
    # PPU MODIFICATION: begin
    SORTED: tl.constexpr,
    # PPU MODIFICATION: end
    PACKED: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    PEER_ROWS: tl.constexpr = 0,
    WORD: tl.constexpr = tl.uint8,
    PACKED_FP4: tl.constexpr = False,
    SOFTWARE_FP8: tl.constexpr = False,
):
    """Gather FP8 or MXFP4 rows as dequantized BF16, or FP8 rows as packed
    values/scales (copied WORD at a time).

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA. If SORTED,
    `ids` holds rows relative to vocab_start in table order and `sorted_dst`
    their output rows. SOFTWARE_FP8 (GPUs below SM89, which Triton gives no
    FP8 type): `weight` holds the E4M3 bytes as uint8 and they are decoded
    with integer operations, which is exact.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        if SORTED:
            local = tl.load(ids + rows, mask=valid, other=-1).to(tl.int64)
            dst = tl.load(sorted_dst + rows, mask=valid, other=0).to(tl.int32)
            owned = (local >= 0) & (local < vocab_end - vocab_start)
        else:
            head = HEAD_START + rows % LOCAL_HEADS
            token = (rows // LOCAL_HEADS).to(tl.int64)
            index = tl.load(
                ids + token * ids_stride_t + head * ids_stride_h,
                mask=valid & (head < TOTAL_HEADS),
                other=-1,
            ).to(tl.int64)
            dst = rows
            owned = valid & (head < TOTAL_HEADS)
            owned &= (index >= vocab_start) & (index < vocab_end)
            local = index - vocab_start
        local = tl.where(owned, local, 0)
        ROW_BYTES: tl.constexpr = DIM // 2 if PACKED_FP4 else DIM
        if PEER_ROWS:
            # `weight` holds each rank's buffer: PEER_ROWS rows, then scales.
            owner = local // PEER_ROWS
            peer = tl.load(weight + owner, mask=owned, other=0)
            local -= owner * PEER_ROWS
            if PACKED_FP4 or SOFTWARE_FP8:
                row_weight = peer.to(tl.pointer_type(tl.uint8))
            else:
                row_weight = peer.to(tl.pointer_type(tl.float8e4nv))
            row_scales = (peer + PEER_ROWS * ROW_BYTES).to(tl.pointer_type(tl.uint8))
        else:
            row_weight, row_scales = weight, scales
        row_weight += local * ROW_BYTES
        row_scales += local * (DIM // QUANT_BLOCK)
        if PACKED:
            # PPU MODIFICATION: begin
            tl.static_assert(row_scales.dtype.element_ty == tl.uint8, "Packed output requires E8M0 byte scales")
            # PPU MODIFICATION: end
            # Copy values and scales WORD at a time: one memory request a word.
            word_bytes: tl.constexpr = WORD.primitive_bitwidth // 8
            value_words = tl.arange(0, DIM // word_bytes)
            scale_words = tl.arange(0, DIM // QUANT_BLOCK // word_bytes)
            width: tl.constexpr = DIM + DIM // QUANT_BLOCK
            dst = dst.to(tl.int64)
            dst = dst // LOCAL_HEADS * OUT_STRIDE + dst % LOCAL_HEADS * width
            out_row = (out + dst).to(tl.pointer_type(WORD))
            # Load both first: a load after a store waits out its round trip.
            values = tl.load(
                row_weight.to(tl.pointer_type(WORD))[:, None] + value_words[None, :],
                mask=owned[:, None],
                other=0,
            )
            scale = tl.load(
                row_scales.to(tl.pointer_type(WORD))[:, None] + scale_words[None, :],
                mask=owned[:, None],
                other=0,
            )
            tl.store(
                out_row[:, None] + value_words[None, :], values, mask=valid[:, None]
            )
            out_row += DIM // word_bytes
            tl.store(
                out_row[:, None] + scale_words[None, :], scale, mask=valid[:, None]
            )
        else:
            if PACKED_FP4:
                # Even column in the low nibble. E2M1 magnitude with e = bits 2:1,
                # m = bit 0: m * 0.5 if e == 0, else (2 + m) * 2**(e - 2).
                byte = tl.load(
                    row_weight[:, None] + (cols // 2)[None, :],
                    mask=owned[:, None],
                    other=0,
                ).to(tl.int32)
                nib = tl.where(cols[None, :] % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
                e = (nib & 0x7) >> 1
                m = (nib & 0x1).to(tl.float32)
                pow2 = ((e - 2 + 127) << 23).to(tl.float32, bitcast=True)
                mag = tl.where(e == 0, m * 0.5, (2.0 + m) * pow2)
                values = tl.where((nib & 0x8) != 0, -mag, mag)
            else:
                values = tl.load(
                    row_weight[:, None] + cols[None, :], mask=owned[:, None], other=0.0
                )
            scale = tl.load(
                row_scales[:, None] + scale_cols[None, :], mask=owned[:, None], other=0
            )
            # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
            # PPU MODIFICATION: begin
            if row_scales.dtype.element_ty != tl.float32:
                scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            # PPU MODIFICATION: end
            if SOFTWARE_FP8:
                # E4M3: a normal code's exponent and mantissa shift into the
                # fp32 fields (bias 7 -> 127); a subnormal code is m * 2^-9.
                code = values.to(tl.int32)
                magnitude = code & 0x7F
                normal = ((magnitude << 20) + (120 << 23)).to(tl.float32, bitcast=True)
                subnormal = (magnitude & 7).to(tl.float32) * 0.001953125
                decoded = tl.where(magnitude < 8, subnormal, normal)
                decoded = tl.where((code & 0x80) != 0, -decoded, decoded)
            else:
                decoded = values.to(tl.float32)
            tl.store(
                out + dst[:, None] * DIM + cols[None, :],
                (decoded * scale).to(tl.bfloat16),
                mask=valid[:, None],
            )


@patch(
    _FACTORY,
    "Engram._create_embedding",
    reason="Select FP32 row-scale storage for channelwise Engram while reusing CUDA offload and prefetch.",
    affected_versions=_AFFECTED,
    remove_when="The Engram embedding factory exposes quantization-aware backend registration.",
)
def _create_embedding(self, layout, layer_hash_index):
    vllm_config = get_current_vllm_config()
    quant = (
        getattr(vllm_config.model_config.hf_config, "quantization_config", None) or {}
    )
    name = f"layers.{layout.layer_ids[layer_hash_index]}.engram.embed"
    if current_platform.is_ppu():
        # INT8 reuses the decision DeepseekV41ForCausalLM published before
        # building layers, unless the checkpoint kept this table in MXFP8;
        # FP8 channelwise tables are listed per layer.
        int8 = int8_w8a8_enabled() and _checkpoint_table_dtype(
            vllm_config.model_config, name
        ) in (None, "I8")
        if int8 or name in (quant.get("fp8_channelwise_layers") or []):
            from vllm_sail.models.deepseek_v41.engram import ChannelwiseEngramEmbedding

            config = vllm_config.engram_config
            # FP8 channelwise keeps the defaults: E4M3 values, no THP packing.
            int8_storage = (
                dict(use_thp=config.use_thp, weight_dtype=torch.int8) if int8 else {}
            )
            return ChannelwiseEngramEmbedding(
                layout.num_embeddings[layer_hash_index], layout.head_dim,
                tuple(size for order in layout.primes[layer_hash_index] for size in order),
                cpu_offload=config.cpu_offload, dp_shared_memory=bool(config.dp_shared_memory),
                **int8_storage,
            )
    original = getattr(_create_embedding, PATCH_MARKER)[f"{_FACTORY}.Engram._create_embedding"]
    return original(self, layout, layer_hash_index)


_ENGRAM_TABLE = re.compile(r"(layers\.\d+\.engram\.embed)\.weight$")


@cache
def _checkpoint_table_dtypes(model, revision):
    """Safetensors dtypes of the Engram tables, read from headers only."""
    from vllm.transformers_utils.config import get_safetensors_params_metadata

    metadata = get_safetensors_params_metadata(model, revision=revision)
    return {
        match[1]: info.get("dtype")
        for key, info in metadata.items()
        if (match := _ENGRAM_TABLE.search(key))
    }


def _checkpoint_table_dtype(model_config, name):
    """W8A8 exports may keep Engram in MXFP8; None when headers are unavailable."""
    return _checkpoint_table_dtypes(model_config.model, model_config.revision).get(name)


@patch(
    "vllm.models.deepseek_v41.common.engram", "ParallelEngramEmbedding.lookup",
    reason="PPU Engram uses INT8/FP32 lookup or software E4M3 decoding without changing upstream offload and prefetch.",
    affected_versions=_AFFECTED,
    remove_when="Upstream Engram lookup supports PPU software E4M3 and INT8 per-row scales.",
)
def lookup(self, indices, out, background=False):
    if current_platform.is_ppu():
        # INT8 tables exist only when the construction-time W8A8 factory chose
        # ChannelwiseEngramEmbedding(weight_dtype=int8); no runtime config query.
        if getattr(self, "weight_dtype", None) == torch.int8 or (
            self.weight.dtype != torch.int8
            and current_platform.is_device_capability((8, 0))
            and not getattr(self, "peer_rows", 0)
            and out.dtype == torch.bfloat16
        ):
            from vllm_sail.models.deepseek_v41.ops.engram_lookup import lookup as ppu_lookup

            return ppu_lookup(self, indices, out, background)
    original = getattr(lookup, PATCH_MARKER)[
        "vllm.models.deepseek_v41.common.engram.ParallelEngramEmbedding.lookup"
    ]
    return original(self, indices, out, background)
