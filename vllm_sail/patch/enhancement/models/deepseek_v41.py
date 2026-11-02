# SPDX-License-Identifier: Apache-2.0
"""V4.1 810E routing without modifying shared V4 providers or non-PPU dispatch."""

import sys
from dataclasses import replace

import torch
from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v41 import attention
from vllm.models.deepseek_v41.common import engram, ops
from vllm.models.deepseek_v41.common.ops import (
    cache_utils,
    fused_compress_quant_cache,
    indexer_k_store,
)
from vllm.models.deepseek_v41.nvidia import engram as nvidia_engram
from vllm.models.deepseek_v41.nvidia import model
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend

from vllm_sail.patch.utils import patch

_PREFIX = "vllm.models.deepseek_v41"
REASON = "V4.1 on 810E requires INT8 DSA/Engram, actual head metadata, and software FP8 main-cache routing."
AFFECTED_VERSIONS = ">=0.31.0,<0.32.0"
REMOVE_WHEN = "Upstream V4.1 provides PPU operator registration and native 810E W8A8 and INT8/FP32 Engram support."
_METADATA = dict(
    reason=REASON, affected_versions=AFFECTED_VERSIONS, remove_when=REMOVE_WHEN
)
_select = model._select_dsv4_attn_cls
_q = ops.fused_indexer_q_rope_quant
_k = indexer_k_store.indexer_k_norm_rope_store
_store = fused_compress_quant_cache.rope_quant_insert
_gather = cache_utils.dequantize_and_gather_k_cache
_spec = attention.DeepseekV4IndexerCache.get_kv_cache_spec
_kernel_sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes
_lookup = engram.ParallelEngramEmbedding.lookup
_share = nvidia_engram.can_share_engram_tables
_create_embedding = nvidia_engram.Engram._create_embedding


def _is_810e():
    return current_platform.is_ppu() and current_platform.is_device_capability((8, 0))


@patch(
    _PREFIX + ".nvidia.model",
    "_select_dsv4_attn_cls",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def select_attention(vllm_config):
    selected = _select(vllm_config)
    if _is_810e() and selected is DeepseekV4FlashMLAAttention:
        from vllm_sail.models.deepseek_v41.flashmla import DeepseekV41FlashMLAAttention

        return DeepseekV41FlashMLAAttention
    return selected


@patch(
    _PREFIX + ".common.ops",
    "fused_indexer_q_rope_quant",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def fused_indexer_q_rope_quant(
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
    if not _is_810e():
        if output_buffers is not None:
            raise ValueError("output_buffers is supported only on the 810E INT8 path")
        return _q(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            use_fp4=use_fp4,
            weights_out_dtype=weights_out_dtype,
        )
    from vllm_sail.models.deepseek_v41.ops.indexer_tilelang import (
        fused_indexer_q_rope_quant_int8,
    )

    return fused_indexer_q_rope_quant_int8(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        use_fp4=use_fp4,
        weights_out_dtype=weights_out_dtype,
        output_buffers=output_buffers,
    )


@patch(
    _PREFIX + ".common.ops.indexer_k_store",
    "indexer_k_norm_rope_store",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def indexer_k_norm_rope_store(
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
    implementation = _k
    if _is_810e():
        from vllm_sail.models.deepseek_v41.ops.indexer_tilelang import (
            indexer_k_norm_rope_store_int8,
        )

        implementation = indexer_k_norm_rope_store_int8
    return implementation(
        k_pre,
        positions,
        cos_sin_cache,
        rms_norm_weight,
        rms_norm_eps,
        k_cache,
        kv_slot_mapping,
        compress_ratio,
        use_fp4_cache,
    )


@patch(
    _PREFIX + ".common.ops.fused_compress_quant_cache",
    "rope_quant_insert",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def rope_quant_insert(
    latent,
    positions,
    cos_sin_cache,
    kv_cache,
    slot_mapping,
    compress_ratio,
    fp8_scale=None,
):
    implementation = _store
    if _is_810e():
        from vllm_sail.models.deepseek_v41.ops.latent_store import (
            rope_quant_insert as ppu_store,
        )

        implementation = ppu_store
    return implementation(
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        compress_ratio,
        fp8_scale,
    )


@patch(
    _PREFIX + ".common.ops.cache_utils",
    "dequantize_and_gather_k_cache",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def dequantize_and_gather_k_cache(
    out,
    k_cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
    use_fnuz=False,
):
    if _is_810e():
        if use_fnuz or k_cache.dtype != torch.uint8 or k_cache.shape[-1] != 584:
            raise ValueError(
                "810E V4.1 gather supports only the 584-byte OCP main cache"
            )
        return torch.ops.vllm.ppu_deepseek_v4_dequant_gather(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset,
        )
    return _gather(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=use_fnuz,
    )


@patch(
    _PREFIX + ".attention",
    "DeepseekV4IndexerCache.get_kv_cache_spec",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def get_kv_cache_spec(self, vllm_config):
    spec = _spec(self, vllm_config)
    if _is_810e():
        config = vllm_config.model_config.hf_text_config
        # The SAIL INT8 SDK requires 64 stored rows per page.
        # C1/C2 use separate 64/128-token manager blocks with recomputed page alignment.
        return replace(
            spec,
            indexer_n_head=config.index_n_heads,
            indexer_q_head_dim=config.index_head_dim,
            block_size=64 * self.compress_ratio,
            page_size_padded=None,
        )
    return spec


@patch(
    "vllm.v1.attention.backends.mla.indexer",
    "DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
    reason="810E INT8 paged logits require 64 stored rows; C1/C2 use 64/128-token pages.",
    affected_versions=AFFECTED_VERSIONS,
    remove_when="The upstream V4.1 indexer backend selects page sizes according to PPU kernel contracts.",
)
def get_supported_kernel_block_sizes(kv_cache_spec=None):
    if _is_810e():
        return [64, 128]
    return _kernel_sizes(kv_cache_spec=kv_cache_spec)


@patch(
    _PREFIX + ".common.engram",
    "ParallelEngramEmbedding.lookup",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def lookup(self, indices, out, background=False):
    if _is_810e():
        from vllm_sail.models.deepseek_v41.ops.engram_lookup import lookup as ppu_lookup

        return ppu_lookup(self, indices, out, background)
    return _lookup(self, indices, out, background)


@patch(
    _PREFIX + ".nvidia.engram",
    "can_share_engram_tables",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def can_share_engram_tables(layout, block_size=32):
    if _uses_int8_engram():
        from vllm_sail.models.deepseek_v41.engram import (
            can_share_engram_tables as share,
        )

        return share(layout, block_size)
    return _share(layout, block_size)


def _uses_int8_engram():
    if not _is_810e():
        return False
    config = get_current_vllm_config()
    quant = config.quant_config
    if quant is None or quant.get_name() != "compressed-tensors":
        return False
    hf = config.model_config.hf_config
    raw = getattr(hf, "quantization_config", {}) or {}
    if getattr(hf, "model_type", None) != "deepseek_v41":
        return False
    if raw.get("expert_dtype") != "int8" or raw.get("format") != "int-quantized":
        return False
    groups = raw.get("config_groups", {}).values()
    for group in groups:
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if (
            "Linear" in group.get("targets", [])
            and weight.get("type") == activation.get("type") == "int"
            and weight.get("num_bits") == activation.get("num_bits") == 8
            and weight.get("strategy") == "channel"
            and activation.get("strategy") == "token"
            and weight.get("symmetric") is True
            and activation.get("symmetric") is True
            and weight.get("dynamic") is False
            and activation.get("dynamic") is True
        ):
            return True
    return False


@patch(
    _PREFIX + ".nvidia.engram",
    "Engram._create_embedding",
    reason=REASON,
    affected_versions=AFFECTED_VERSIONS,
    remove_when=REMOVE_WHEN,
)
def create_embedding(self, layout, layer_hash_index):
    if not _uses_int8_engram():
        return _create_embedding(self, layout, layer_hash_index)
    from vllm_sail.models.deepseek_v41.engram import Int8ParallelEngramEmbedding

    config = get_current_vllm_config().engram_config
    return Int8ParallelEngramEmbedding(
        layout.num_embeddings[layer_hash_index],
        layout.head_dim,
        tuple(size for order in layout.primes[layer_hash_index] for size in order),
        cpu_offload=config.cpu_offload,
        dp_shared_memory=bool(config.dp_shared_memory),
        use_thp=config.use_thp,
    )


# Rebind only loaded V4.1 consumers; later imports read the patched providers.
_ALIASES = (
    ("can_share_engram_tables", _share, can_share_engram_tables),
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
                    patch(
                        module_name,
                        name,
                        reason=REASON,
                        affected_versions=AFFECTED_VERSIONS,
                        remove_when=REMOVE_WHEN,
                    )(replacement)


_rebind_aliases()
