# SPDX-License-Identifier: Apache-2.0
"""SAIL MSA selection policy for MiniMax M3 sparse attention on PPU.

Default policy (mirrors ``sglang-for-sail``'s PPU hook):

* **890P** (hgcc ``ppu_15``): when the main KV cache is FP8, the default is an
  fp8 index cache + fp8 sparse attend + bf16 top-k logits. Without a
  ``--kv-cache-dtype fp8`` request the whole trio silently falls back to
  bf16 index cache + bf16 attend + fp32 logits -- fp8 sparse attend has no
  cheaper source of FP8 K/V, so the faster combination is only reachable with
  an fp8 cache.
* **810E** (``ppu_10``): bf16 index cache + bf16 attend + fp32 logits. The
  fp8 sparse variants only compile for ``ppu_15``.

Every decision is derived from the platform and the requested KV cache dtype;
only ``VLLM_SAIL_MINIMAX_M3_MSA=0`` changes the outcome (see
:mod:`vllm_sail.envs`). The policy is consulted only at backend selection and
metadata-build time, never in a hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_sail import envs
from vllm_sail.models.minimax_m3_msa.adapter import sail_msa_available

#: Sparse block size == index-K block == KV-cache page size.
PAGE_SIZE = 128
#: ``sparse_topk_select`` only implements this width.
SAIL_TOPK_BLOCKS = 16
#: Insertion-sort top-k limit of the installed SAIL kernel (tiles per row).
MAX_CONTEXT_TILES = 12288

_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")
_BF16_KV_DTYPES = ("bfloat16", "bf16", "auto")
#: ``minimax_m3_text`` / ``minimax_m3_vl`` / ``minimax_m3_mtp``.
_MINIMAX_M3_MODEL_TYPE_PREFIX = "minimax_m3"


@dataclass(frozen=True)
class SailMSASelection:
    """Resolved SAIL MSA configuration for one model instance."""

    enable: bool
    indexer_fp8: bool = False
    attend_fp8: bool = False
    score_dtype: torch.dtype = torch.float32
    reason: str | None = None


def _off(reason: str) -> SailMSASelection:
    return SailMSASelection(enable=False, reason=reason)


def ppu_arch() -> str | None:
    """``"ppu_15"`` (890P), ``"ppu_10"`` (810E), or ``None`` off PPU."""
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return None
    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return None
    return "ppu_15" if capability == (8, 9) else "ppu_10"


def kv_cache_is_fp8(cache_dtype: str | None) -> bool:
    return (cache_dtype or "").lower() in _FP8_KV_DTYPES


def is_minimax_m3(vllm_config) -> bool:
    """Whether ``vllm_config`` serves a MiniMax M3 target or draft model."""
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return False
    return any(
        str(getattr(config, "model_type", "")).startswith(_MINIMAX_M3_MODEL_TYPE_PREFIX)
        for config in (hf_config, getattr(hf_config, "text_config", None))
        if config is not None
    )


def _normalize_indexer_dtype(indexer_kv_dtype: str) -> str:
    value = (indexer_kv_dtype or "").lower()
    if value in _FP8_KV_DTYPES:
        return "fp8"
    if value in _BF16_KV_DTYPES:
        return "bf16"
    return value


def _shard_heads(total_heads: int, tp_size: int) -> int:
    if total_heads >= tp_size:
        return total_heads // tp_size
    return 1


def decide(
    *,
    topk_blocks: int,
    block_size: int,
    score_type: str,
    head_dim: int,
    idx_head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_idx_heads: int,
    kv_cache_dtype: str,
    indexer_kv_dtype: str,
    max_model_len: int,
) -> SailMSASelection:
    """Resolve the SAIL MSA path and its dtypes from concrete model shape."""
    if not envs.VLLM_SAIL_MINIMAX_M3_MSA:
        return _off("disabled by VLLM_SAIL_MINIMAX_M3_MSA=0")
    arch = ppu_arch()
    if arch is None:
        return _off("not running on PPU")
    if not sail_msa_available():
        return _off("msa_for_sail is not importable")

    unsupported: list[str] = []
    if block_size != PAGE_SIZE:
        unsupported.append(f"block_size={block_size} (SAIL needs 128)")
    if topk_blocks != SAIL_TOPK_BLOCKS:
        unsupported.append(f"sparse_topk_blocks={topk_blocks} (SAIL supports only 16)")
    if score_type != "max":
        unsupported.append(f"sparse_score_type={score_type!r} (SAIL needs 'max')")
    if head_dim != PAGE_SIZE:
        unsupported.append(f"head_dim={head_dim} (SAIL needs 128)")
    if idx_head_dim != PAGE_SIZE:
        unsupported.append(f"sparse_index_dim={idx_head_dim} (SAIL needs 128)")
    if num_q_heads % num_kv_heads != 0:
        unsupported.append(f"num_heads={num_q_heads} not a multiple of {num_kv_heads}")
    if num_idx_heads != num_kv_heads:
        unsupported.append(
            f"num_index_heads={num_idx_heads} != num_kv_heads={num_kv_heads} "
            "(the shared top-k buffer is one row per KV head)"
        )
    context_tiles = (max_model_len + PAGE_SIZE - 1) // PAGE_SIZE
    if context_tiles >= MAX_CONTEXT_TILES:
        unsupported.append(
            f"max_model_len={max_model_len} exceeds the SAIL top-k tile limit "
            f"({MAX_CONTEXT_TILES} tiles)"
        )
    if unsupported:
        return _off("; ".join(unsupported))

    kv_dtype = (kv_cache_dtype or "auto").lower()
    if kv_dtype not in _FP8_KV_DTYPES + _BF16_KV_DTYPES:
        return _off(
            f"kv_cache_dtype={kv_cache_dtype!r} is not supported by SAIL MSA "
            "(bfloat16 or fp8_e4m3)"
        )
    indexer_dtype = _normalize_indexer_dtype(indexer_kv_dtype)
    if indexer_dtype not in ("bf16", "fp8"):
        return _off(
            f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by SAIL MSA"
        )

    is_890p = arch == "ppu_15"
    kv_fp8 = kv_dtype in _FP8_KV_DTYPES
    indexer_fp8 = indexer_dtype == "fp8"
    if not is_890p:
        if indexer_fp8:
            return _off("an fp8 index cache requires 890P (ppu_15)")
        if kv_fp8:
            return _off("an fp8 KV cache requires 890P (ppu_15)")
    # fp8 sparse attend and the fp8 OnlyScore variant only compile for 890P, and
    # the attend needs fp8 KV pages (q and k must share a dtype), so the whole
    # trio rides on the explicitly requested fp8 KV cache.
    attend_fp8 = is_890p and kv_fp8
    # SAIL's fp8 OnlyScore has no fp32 score variant, so an fp8 index cache
    # always carries bf16 top-k logits.
    score_bf16 = is_890p and indexer_fp8

    return SailMSASelection(
        enable=True,
        indexer_fp8=indexer_fp8,
        attend_fp8=attend_fp8,
        score_dtype=torch.bfloat16 if score_bf16 else torch.float32,
    )


def _current_config():
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except Exception:
        return None


def decide_from_config(
    vllm_config,
    *,
    topk_blocks: int | None = None,
    kv_cache_dtype: str | None = None,
    indexer_kv_dtype: str | None = None,
) -> SailMSASelection:
    """Decision from an explicit ``VllmConfig`` (selectors and builders)."""
    if vllm_config is None:
        return _off("no active vllm config")
    text_config = vllm_config.model_config.hf_config
    text_config = getattr(text_config, "text_config", text_config)
    if not getattr(text_config, "index_topk_blocks", None):
        return _off("model has no sparse attention index config")

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    head_dim = int(getattr(text_config, "head_dim", 0) or 0)
    if not head_dim:
        head_dim = int(text_config.hidden_size) // int(text_config.num_attention_heads)
    if indexer_kv_dtype is None:
        resolved = vllm_config.attention_config.indexer_kv_dtype
        if resolved == "auto":
            resolved = default_indexer_kv_dtype("bf16", vllm_config) or "bf16"
        indexer_kv_dtype = resolved

    return decide(
        topk_blocks=int(topk_blocks or text_config.index_topk_blocks),
        block_size=int(vllm_config.cache_config.block_size or PAGE_SIZE),
        score_type="max",
        head_dim=head_dim,
        idx_head_dim=int(text_config.index_head_dim),
        num_q_heads=_shard_heads(int(text_config.num_attention_heads), tp_size),
        num_kv_heads=_shard_heads(int(text_config.num_key_value_heads), tp_size),
        num_idx_heads=_shard_heads(int(text_config.index_n_heads), tp_size),
        kv_cache_dtype=kv_cache_dtype or vllm_config.cache_config.cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype,
        max_model_len=int(vllm_config.model_config.max_model_len),
    )


def decide_from_current_config(**kwargs) -> SailMSASelection:
    return decide_from_config(_current_config(), **kwargs)


def default_indexer_kv_dtype(default: str = "bf16", vllm_config=None) -> str | None:
    """Architecture default for the indexer side cache when left on ``auto``.

    Returns ``None`` when the caller's ``default`` (usually ``"bf16"``) should
    stand. On 890P with an fp8 KV cache the index cache of a MiniMax M3 model
    follows the main cache to fp8, which is what enables the fp8/bf16-logits
    trio. Other models sharing ``indexer_kv_dtype`` keep their own default.
    """
    if default != "bf16":
        return None
    if not envs.VLLM_SAIL_MINIMAX_M3_MSA:
        return None
    if ppu_arch() != "ppu_15" or not sail_msa_available():
        return None
    config = vllm_config if vllm_config is not None else _current_config()
    if not is_minimax_m3(config):
        return None
    if not kv_cache_is_fp8(config.cache_config.cache_dtype):
        return None
    return "fp8"
