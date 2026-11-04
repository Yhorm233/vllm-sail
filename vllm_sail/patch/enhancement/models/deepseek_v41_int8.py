# SPDX-License-Identifier: Apache-2.0
"""Integrate PPU W8A8 operators into upstream V4.1 without replacing other quantization paths."""

import sys

import torch
from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v41.common import ops
from vllm.models.deepseek_v41.common.ops import (
    cache_utils,
    fused_compress_quant_cache,
    indexer_k_store,
)
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41.config import uses_int8_w8a8
from vllm_sail.patch.utils import patch

_PREFIX = "vllm.models.deepseek_v41"
REASON = "PPU V4.1 W8A8 uses INT8 DSA and software FP8 main-cache operators."
AFFECTED_VERSIONS = ">=0.31.0,<0.32.0"
REMOVE_WHEN = "V4.1 provides quantization-aware PPU operator registration for W8A8."
_q = ops.fused_indexer_q_rope_quant
_k = indexer_k_store.indexer_k_norm_rope_store
_store = fused_compress_quant_cache.rope_quant_insert
_gather = cache_utils.dequantize_and_gather_k_cache


def _uses_int8(vllm_config=None):
    if not current_platform.is_ppu():
        return False
    return uses_int8_w8a8(vllm_config or get_current_vllm_config())


@patch(
    _PREFIX + ".common.ops", "fused_indexer_q_rope_quant",
    reason=REASON, affected_versions=AFFECTED_VERSIONS, remove_when=REMOVE_WHEN,
)
def fused_indexer_q_rope_quant(
    positions, index_q, index_q_cos_sin_cache, index_weights,
    index_weights_softmax_scale, index_weights_head_scale, use_fp4=False,
    weights_out_dtype=torch.float32, *, output_buffers=None,
):
    if not _uses_int8() or use_fp4:
        if output_buffers is not None:
            raise ValueError("output_buffers is supported only on the PPU V4.1 INT8 path")
        return _q(
            positions, index_q, index_q_cos_sin_cache, index_weights,
            index_weights_softmax_scale, index_weights_head_scale,
            use_fp4=use_fp4, weights_out_dtype=weights_out_dtype,
        )
    from vllm_sail.models.deepseek_v41.ops.indexer_tilelang import fused_indexer_q_rope_quant_int8

    return fused_indexer_q_rope_quant_int8(
        positions, index_q, index_q_cos_sin_cache, index_weights,
        index_weights_softmax_scale, index_weights_head_scale,
        use_fp4=use_fp4, weights_out_dtype=weights_out_dtype, output_buffers=output_buffers,
    )


@patch(
    _PREFIX + ".common.ops.indexer_k_store", "indexer_k_norm_rope_store",
    reason=REASON, affected_versions=AFFECTED_VERSIONS, remove_when=REMOVE_WHEN,
)
def indexer_k_norm_rope_store(
    k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
    k_cache, kv_slot_mapping, compress_ratio, use_fp4_cache,
):
    implementation = _k
    if _uses_int8() and not use_fp4_cache:
        from vllm_sail.models.deepseek_v41.ops.indexer_tilelang import indexer_k_norm_rope_store_int8

        implementation = indexer_k_norm_rope_store_int8
    return implementation(
        k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps,
        k_cache, kv_slot_mapping, compress_ratio, use_fp4_cache,
    )


@patch(
    _PREFIX + ".common.ops.fused_compress_quant_cache", "rope_quant_insert",
    reason=REASON, affected_versions=AFFECTED_VERSIONS, remove_when=REMOVE_WHEN,
)
def rope_quant_insert(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
                      compress_ratio, fp8_scale=None):
    implementation = _store
    if _uses_int8():
        from vllm_sail.models.deepseek_v41.ops.latent_store import rope_quant_insert as ppu_store

        implementation = ppu_store
    return implementation(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
                          compress_ratio, fp8_scale)


@patch(
    _PREFIX + ".common.ops.cache_utils", "dequantize_and_gather_k_cache",
    reason=REASON, affected_versions=AFFECTED_VERSIONS, remove_when=REMOVE_WHEN,
)
def dequantize_and_gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table,
                                block_size, offset, use_fnuz=False):
    if _uses_int8():
        if use_fnuz or k_cache.dtype != torch.uint8 or k_cache.shape[-1] != 584:
            raise ValueError("PPU V4.1 INT8 gather supports only the 584-byte OCP main cache")
        return torch.ops.vllm.ppu_deepseek_v4_dequant_gather(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset,
        )
    # PPU MODIFICATION: begin
    # Decode PPU FP8/MXFP8 main caches in software without enabling INT8 Q/K.
    # Match existing layouts; delegate FNUZ, NVFP4 and non-PPU platforms.
    if (
        current_platform.is_ppu()
        and not use_fnuz
        and k_cache.dtype == torch.uint8
        and k_cache.shape[-1] in (528, 584)
    ):
        return torch.ops.vllm.ppu_deepseek_v4_dequant_gather(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset,
        )
    # PPU MODIFICATION: end
    return _gather(out, k_cache, seq_lens, gather_lens, block_table, block_size,
                   offset, use_fnuz=use_fnuz)


# Rebind only loaded V4.1 consumers; leave shared V4 providers untouched.
_ALIASES = (
    ("fused_indexer_q_rope_quant", _q, fused_indexer_q_rope_quant),
    ("indexer_k_norm_rope_store", _k, indexer_k_norm_rope_store),
    ("rope_quant_insert", _store, rope_quant_insert),
    ("dequantize_and_gather_k_cache", _gather, dequantize_and_gather_k_cache),
)


def _rebind_aliases():
    for module_name, module in tuple(sys.modules.items()):
        if module_name == _PREFIX or module_name.startswith(_PREFIX + "."):
            for name, original, replacement in _ALIASES:
                if getattr(module, name, None) is original:
                    patch(module_name, name, reason=REASON,
                          affected_versions=AFFECTED_VERSIONS,
                          remove_when=REMOVE_WHEN)(replacement)


_rebind_aliases()
