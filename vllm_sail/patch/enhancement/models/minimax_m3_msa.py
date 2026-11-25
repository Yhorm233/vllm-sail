# SPDX-License-Identifier: Apache-2.0
"""Select the SAIL (msa_for_sail) MiniMax M3 sparse attention on PPU.

The plugin's MiniMax M3 registrations subclass the upstream NVIDIA model, so
the upstream selectors are what the serving path reaches. Three selection
points are patched, all of them pure gates -- the upstream implementations
stay untouched and remain the fallback on every platform (and on PPU whenever
the SAIL policy declines):

* ``vllm.models.minimax_m3.common.indexer.select_indexer_impl_cls`` --
  indexer side cache scoring + top-k.
* ``vllm.models.minimax_m3.nvidia.model.select_main_backend_and_impl_cls`` --
  sparse attend over the selected pages (the model binds the selector by
  name, so the patch targets that binding).
* ``AttentionConfig.resolve_indexer_kv_dtype`` -- the index cache dtype that
  ``"auto"`` resolves to, so the fp8 index cache of a MiniMax M3 model follows
  ``--kv-cache-dtype fp8`` on 890P and stays bf16 everywhere else.

The policy itself lives in :mod:`vllm_sail.models.minimax_m3_msa.gating`.
"""

from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.config.attention import AttentionConfig
from vllm.logger import init_logger
from vllm.models.minimax_m3.common import indexer as _indexer_module
from vllm.models.minimax_m3.nvidia import model as _nvidia_model
from vllm.platforms import current_platform
from vllm_sail import envs

from vllm_sail.models.minimax_m3_msa.attention import (
    MiniMaxM3SparseSailMSABackend,
    MiniMaxM3SparseSailMSAImpl,
)
from vllm_sail.models.minimax_m3_msa.gating import (
    decide_from_current_config,
    default_indexer_kv_dtype,
)
from vllm_sail.models.minimax_m3_msa.indexer import MiniMaxM3IndexerSailMSAImpl
from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.32.0,<0.33.0"

_original_select_indexer_impl = _indexer_module.select_indexer_impl_cls
_original_select_main_backend_and_impl = (
    _nvidia_model.select_main_backend_and_impl_cls
)
_original_resolve_indexer_kv_dtype = AttentionConfig.resolve_indexer_kv_dtype


def _downgrade_cudagraph_mode() -> None:
    """Drop full cudagraphs once the SAIL MSA backends are selected.

    Both SAIL builders report ``AttentionCGSupport.NEVER``: the plan, the
    compacted page table and the indexer score buffer are rebuilt every step
    from the batch's lengths, so a FULL graph would replay capture-time
    pointers (illegal access / stale selections). Piecewise capture is fine --
    the model's ``@eager_break_during_capture`` attention keeps running eagerly
    between the captured segments. This runs at model construction, before the
    runner resolves the mode, and is a no-op once the mode is piecewise.
    """
    try:
        config = get_current_vllm_config()
    except Exception:
        return
    if config is None:
        return
    compilation = config.compilation_config
    mode = compilation.cudagraph_mode
    if mode is None or not mode.has_full_cudagraphs():
        return
    compilation.cudagraph_mode = CUDAGraphMode.PIECEWISE
    logger.warning_once(
        "MiniMax M3 SAIL MSA rebuilds its sparse plan every step, which full "
        "cudagraphs cannot replay; downgrading cudagraph_mode %s -> PIECEWISE.",
        mode.name,
    )


def _select_sail_indexer(indexer_kv_dtype: str, fallback: str) -> bool:
    """Whether the indexer runs on SAIL MSA; logs the outcome once."""
    if not current_platform.is_ppu() or envs.VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_TRITON:
        return False
    selection = decide_from_current_config(indexer_kv_dtype=indexer_kv_dtype)
    if not selection.enable:
        logger.info_once(
            "MiniMax M3 indexer: SAIL MSA declined (%s); using the %s.",
            selection.reason,
            fallback,
        )
        return False
    _downgrade_cudagraph_mode()
    logger.info_once(
        "MiniMax M3 indexer: selected SAIL MSA "
        "[indexer_kv_dtype=%s, score_dtype=%s]",
        indexer_kv_dtype,
        selection.score_dtype,
    )
    return True


def _select_sail_attend(topk_blocks: int, kv_cache_dtype: str, fallback: str) -> bool:
    """Whether the sparse attend runs on SAIL MSA; logs the outcome once."""
    if not current_platform.is_ppu() or envs.VLLM_SAIL_MINIMAX_M3_MSA_ATTEND_TRITON:
        return False
    selection = decide_from_current_config()
    if not selection.enable:
        logger.info_once(
            "MiniMax M3 sparse attention: SAIL MSA declined (%s); using the %s.",
            selection.reason,
            fallback,
        )
        return False
    _downgrade_cudagraph_mode()
    logger.info_once(
        "MiniMax M3 sparse attention: selected SAIL MSA "
        "[topk_blocks=%d, kv_cache_dtype=%s, attend=%s]",
        topk_blocks,
        kv_cache_dtype,
        "fp8" if selection.attend_fp8 else "bf16",
    )
    return True


@patch(
    "vllm.models.minimax_m3.common.indexer",
    "select_indexer_impl_cls",
    reason=(
        "MiniMax M3 indexer scoring/top-k is served by the SAIL MSA kernels on "
        "PPU, whose dtype defaults differ from the NVIDIA/ROCm selectors."
    ),
    affected_versions=_AFFECTED,
    remove_when="vLLM MiniMax M3 backend selection exposes a per-platform "
    "selector hook for PPU.",
)
def select_indexer_impl_cls(*, topk_blocks: int, indexer_kv_dtype="bf16", use_flashinfer_msa=False):
    if _select_sail_indexer(indexer_kv_dtype, "upstream indexer selector"):
        return MiniMaxM3IndexerSailMSAImpl
    return _original_select_indexer_impl(
        topk_blocks=topk_blocks, indexer_kv_dtype=indexer_kv_dtype, use_flashinfer_msa=use_flashinfer_msa
    )


@patch(
    "vllm.models.minimax_m3.nvidia.model",
    "select_main_backend_and_impl_cls",
    reason=(
        "MiniMax M3 block-sparse attention is served by the SAIL fmha_sail "
        "kernels on PPU instead of the Triton sparse attend."
    ),
    affected_versions=_AFFECTED,
    remove_when="vLLM MiniMax M3 backend selection exposes a per-platform "
    "selector hook for PPU.",
)
def select_main_backend_and_impl_cls(
    *,
    topk_blocks: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    emits_sparse_block_table: bool = False,
    use_flashinfer_msa: bool = False,
):
    if _select_sail_attend(topk_blocks, kv_cache_dtype, "upstream selector"):
        return MiniMaxM3SparseSailMSABackend, MiniMaxM3SparseSailMSAImpl
    return _original_select_main_backend_and_impl(
        topk_blocks=topk_blocks,
        kv_cache_dtype=kv_cache_dtype,
        num_kv_heads=num_kv_heads,
        emits_sparse_block_table=emits_sparse_block_table,
        use_flashinfer_msa=use_flashinfer_msa,
    )


@patch(
    "vllm.config.attention",
    "AttentionConfig.resolve_indexer_kv_dtype",
    reason=(
        "On 890P a MiniMax M3 fp8 KV cache should carry an fp8 index cache by "
        "default so the indexer runs its fp8 score path; other models and "
        "other PPUs keep the caller's default."
    ),
    affected_versions=_AFFECTED,
    remove_when="vLLM lets a platform default the indexer cache dtype.",
)
def resolve_indexer_kv_dtype(self, default):
    if self.indexer_kv_dtype != "auto":
        return self.indexer_kv_dtype
    platform_default = default_indexer_kv_dtype(default)
    if platform_default is None:
        return default
    return platform_default
