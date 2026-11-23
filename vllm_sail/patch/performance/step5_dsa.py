# SPDX-License-Identifier: Apache-2.0
"""Route Step5's DSA indexer scoring kernel to the PPU resident-block version.

Upstream's ``_paged_scores_sm90`` launches one block per ``(row, tile)``, so
every row reloads its query tile and gate ``cdiv(max_seq_len, TILE)`` times
(258 times at the production shape). The replacement lives in
``vllm_sail/models/step5/ops/sparse_attention_opt.py``.

The only consumer, ``vllm.models.step5.nvidia.flashinfer_sparse``, imports
``paged_mqa_logits_sm90`` lazily inside ``Step5FlashInferSparse.__init__``, so
replacing the module attribute is enough; no body needs to be copied.

Turning the switch off or removing this patch only costs performance, never
correctness, hence the performance category.
"""

from __future__ import annotations

import functools
import importlib
import sys

from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.step5.common.ops.sparse_indexer_scores"
_CONSUMERS = ("vllm.models.step5.nvidia.flashinfer_sparse",)
_NAME = "paged_mqa_logits_sm90"
_REASON = (
    "Step5's indexer scorer launches one block per (row, region tile), which "
    "reloads the row's query tile and gate once per tile -- 258 times on the "
    "production shape. The plugin keeps blocks resident per row instead "
    "(3.188 -> 2.070 ms on ZW-M890P). Opt-in via VLLM_SAIL_STEP5_DSA_SCORES."
)
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "Step5's _paged_scores_sm90 itself strides over region tiles from a "
    "resident grid, making this replacement redundant."
)
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    try:
        module = importlib.import_module(_MODULE)
    except ModuleNotFoundError as exc:
        # A vLLM build without the Step5 tree is not an error.
        if exc.name and (_MODULE == exc.name or _MODULE.startswith(exc.name + ".")):
            return
        raise

    original = getattr(module, _NAME)
    optimized = None

    @functools.wraps(original)
    def paged_mqa_logits_sm90(*args, **kwargs):
        # Read the switch per call to keep envs lazy; non-PPU stays upstream.
        # The resident kernel needs hardware E4M3; PPU 1.0 keeps upstream's
        # launcher, which reaches the software-E4M3 scorer.
        if not (
            current_platform.is_ppu()
            and envs.VLLM_SAIL_STEP5_DSA_SCORES
            and current_platform.has_device_capability(89)
        ):
            return original(*args, **kwargs)
        nonlocal optimized
        if optimized is None:
            # Deferred to first use so installing the patch never pulls in the
            # triton compilation stack.
            from vllm_sail.models.step5.ops.sparse_attention_opt import (
                paged_mqa_logits_sm90 as optimized_impl,
            )

            optimized = optimized_impl
        return optimized(*args, **kwargs)

    patch(
        _MODULE, _NAME, reason=_REASON,
        affected_versions=_AFFECTED, remove_when=_REMOVE_WHEN,
    )(paged_mqa_logits_sm90)
    # The consumer imports lazily inside __init__ and normally holds no copy.
    # If it already does, replace that too, or its instances keep calling
    # upstream's kernel.
    for consumer in _CONSUMERS:
        loaded = sys.modules.get(consumer)
        if loaded is not None and getattr(loaded, _NAME, None) is original:
            patch(
                consumer, _NAME, reason=_REASON,
                affected_versions=_AFFECTED, remove_when=_REMOVE_WHEN,
            )(paged_mqa_logits_sm90)
    _installed = True


install()
