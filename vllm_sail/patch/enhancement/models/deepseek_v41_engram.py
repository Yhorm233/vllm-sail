# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve V4.1 Engram lookup semantics for PPU FP32 per-row scales."""

import sys

import torch
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_sail.models.deepseek_v41.config import uses_int8_w8a8
from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.31.0,<0.32.0"
_FACTORY = "vllm.models.deepseek_v41.nvidia.engram"


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
        scale = tl.load(
            scales + local[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        # PPU MODIFICATION: begin
        if scales.dtype.element_ty == tl.float32:
            # PPU channelwise exports store real FP32 row scales.
            scale = scale.to(tl.float32)
        else:
            # PPU MODIFICATION: end
            # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
            scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(
            out + rows[:, None] * DIM + cols[None, :],
            (values.to(tl.float32) * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


@patch(
    _FACTORY, "Engram._create_embedding",
    reason="Select FP32 row-scale storage for channelwise Engram while reusing CUDA offload and prefetch.",
    affected_versions=_AFFECTED,
    remove_when="The Engram embedding factory exposes quantization-aware backend registration.",
)
def _create_embedding(self, layout, layer_hash_index):
    vllm_config = get_current_vllm_config()
    quant = getattr(vllm_config.model_config.hf_config, "quantization_config", None) or {}
    name = f"layers.{layout.layer_ids[layer_hash_index]}.engram.embed"
    if current_platform.is_ppu() and uses_int8_w8a8(vllm_config):
        from vllm_sail.models.deepseek_v41.engram import Int8ParallelEngramEmbedding

        config = vllm_config.engram_config
        return Int8ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index], layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=config.cpu_offload, dp_shared_memory=bool(config.dp_shared_memory),
            use_thp=config.use_thp,
        )
    if current_platform.is_ppu() and name in (quant.get("fp8_channelwise_layers") or []):
        from vllm_sail.models.deepseek_v41.engram import ChannelwiseEngramEmbedding

        config = vllm_config.engram_config
        return ChannelwiseEngramEmbedding(
            layout.num_embeddings[layer_hash_index], layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=config.cpu_offload, dp_shared_memory=config.dp_shared_memory,
        )
    original = getattr(_create_embedding, PATCH_MARKER)[f"{_FACTORY}.Engram._create_embedding"]
    return original(self, layout, layer_hash_index)


@patch(
    "vllm.models.deepseek_v41.common.engram", "ParallelEngramEmbedding.lookup",
    reason="PPU Engram uses INT8/FP32 lookup or software E4M3 decoding without changing upstream offload and prefetch.",
    affected_versions=_AFFECTED,
    remove_when="Upstream Engram lookup supports PPU software E4M3 and INT8 per-row scales.",
)
def lookup(self, indices, out, background=False):
    if current_platform.is_ppu():
        if self.weight.dtype != torch.int8 or uses_int8_w8a8(get_current_vllm_config()):
            from vllm_sail.models.deepseek_v41.ops.engram_lookup import lookup as ppu_lookup

            return ppu_lookup(self, indices, out, background)
    original = getattr(lookup, PATCH_MARKER)[
        "vllm.models.deepseek_v41.common.engram.ParallelEngramEmbedding.lookup"
    ]
    return original(self, indices, out, background)


@patch(
    _FACTORY, "can_share_engram_tables",
    reason="INT8 Engram shared storage has FP32 row scales; retain upstream per-rank fallback policy.",
    affected_versions=_AFFECTED,
    remove_when="Engram shared-memory capacity calculation follows the selected storage format.",
)
def can_share_engram_tables(layout, block_size=32):
    if current_platform.is_ppu() and uses_int8_w8a8(get_current_vllm_config()):
        from vllm_sail.models.deepseek_v41.engram import can_share_engram_tables as share

        return share(layout, block_size)
    original = getattr(can_share_engram_tables, PATCH_MARKER)[f"{_FACTORY}.can_share_engram_tables"]
    return original(layout, block_size)


def _rebind_share_aliases():
    original = getattr(can_share_engram_tables, PATCH_MARKER)[f"{_FACTORY}.can_share_engram_tables"]
    for name in ("vllm.models.deepseek_v41.nvidia.model",):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "can_share_engram_tables", None) is original:
            patch(name, "can_share_engram_tables",
                  reason="Preloaded V4.1 model must use quantization-aware Engram sharing capacity.",
                  affected_versions=_AFFECTED,
                  remove_when="V4.1 model resolves Engram sharing through the provider module.")(can_share_engram_tables)


_rebind_share_aliases()
