# SPDX-License-Identifier: Apache-2.0
"""Route upstream DeepSelect capability checks and calls to the SAIL wheel."""

from __future__ import annotations

import sys

import torch
from vllm.model_executor.kernels.attention.dsa import sparse_mqa_logits as _sparse
from vllm.model_executor.layers import indexer_topk as _topk
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch
from vllm_sail.utils import deep_select as _sail

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.model_executor.layers.indexer_topk"
_SPARSE_MODULE = "vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits"
_REASON = "PPU SM89 uses the SAIL BF16 DeepSelect wheel instead of vllm._deepselect_C."
_REMOVE_WHEN = "vLLM exposes a platform-selectable DeepSelect provider and capability hook."

_original_has = _sparse.has_deep_select
_original_stride = _topk.get_deep_select_stride_requirement
_original_supported = _topk.is_deep_select_supported
_original_topk = _topk.deep_select_topk
_original_resolve = _topk.SparseIndexerTopk.resolve_backend


@patch(
    _SPARSE_MODULE,
    "has_deep_select",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def has_deep_select() -> bool:
    if current_platform.is_ppu():
        return _sail.is_deep_select_available()
    return _original_has()


@patch(
    _MODULE,
    "get_deep_select_stride_requirement",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def get_deep_select_stride_requirement() -> tuple[int, int]:
    if current_platform.is_ppu():
        if not _sail.is_deep_select_available():
            raise RuntimeError("SAIL DeepSelect wheel is unavailable or disabled")
        return _sail._wheel().get_stride_requirement()
    return _original_stride()


@patch(
    _MODULE,
    "is_deep_select_supported",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def is_deep_select_supported(input: torch.Tensor, topk: int) -> bool:
    if current_platform.is_ppu():
        return _sail.can_use_deep_select(input, topk)
    return _original_supported(input, topk)


@patch(
    _MODULE,
    "deep_select_topk",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def deep_select_topk(
    input: torch.Tensor,
    topk: int,
    end: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    indices_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    if not current_platform.is_ppu():
        return _original_topk(input, topk, end, output_idx, indices_dtype)
    if output_idx is not None:
        if (
            output_idx.ndim != 2
            or output_idx.shape[0] < input.shape[0]
            or output_idx.shape[1] < topk
        ):
            raise ValueError("DeepSelect output is smaller than [rows, topk]")
        output = output_idx[: input.shape[0], :topk]
    else:
        output = None
    indices = _sail.deep_select_topk(input, topk, end, output, indices_dtype)
    return output_idx if output_idx is not None else indices


@patch(
    _MODULE,
    "SparseIndexerTopk.resolve_backend",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def resolve_backend(self, logits, topk_tokens, num_rows):
    if current_platform.is_ppu() and self._backend in ("auto", "deep_select"):
        # Resolve at call time, including for objects cached before patch install.
        # CUDA's cached SM100/_HAS_DEEP_SELECT_C flags do not describe SAIL.
        if _sail.can_use_deep_select(logits, topk_tokens):
            return "deep_select"
        if self._backend == "deep_select":
            raise RuntimeError(
                "SAIL DeepSelect requires an enabled wheel, PPU SM89+, BF16 "
                "logits and a supported shape; FP32 scores are never rounded"
            )
    return _original_resolve(self, logits, topk_tokens, num_rows)


def _rebind_consumers():
    # These consumers import functions by value. Later imports naturally bind
    # the patched provider; preloaded consumers need the same replacements.
    for names, original, replacement in (
        (
            (
                "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
                "vllm.model_executor.layers.sparse_mqa_indexer",
                "vllm.v1.attention.backends.mla.sparse_indexer",
            ),
            _original_has,
            has_deep_select,
        ),
        (
            (
                "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
                _SPARSE_MODULE,
            ),
            _original_stride,
            get_deep_select_stride_requirement,
        ),
        (
            (
                "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
                "vllm.models.step5.common.ops.sparse_indexer",
            ),
            _original_supported,
            is_deep_select_supported,
        ),
        (
            (
                "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
                _SPARSE_MODULE,
            ),
            _original_topk,
            deep_select_topk,
        ),
    ):
        for name in names:
            consumer = sys.modules.get(name)
            if (
                consumer is not None
                and getattr(consumer, replacement.__name__, None) is original
            ):
                patch(
                    name,
                    replacement.__name__,
                    reason=_REASON,
                    affected_versions=_AFFECTED,
                    remove_when=_REMOVE_WHEN,
                )(replacement)


_rebind_consumers()
