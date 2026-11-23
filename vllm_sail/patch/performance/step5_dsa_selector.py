# SPDX-License-Identifier: Apache-2.0
"""Replace the Step5 region selector only when explicitly enabled.

The indexer, MTP and attention are unchanged.
"""
from __future__ import annotations

import functools
import importlib
import sys

from vllm.platforms import current_platform
from vllm_sail import envs
from vllm_sail.patch.utils import patch

_TARGET = "vllm.models.step5.common.ops.sparse_indexer"
_REASON = "Step5 selector_v2 exact FP32 radix candidate reusing stabilization; off by default, chosen by UT performance."
_REMOVE = "Upstream per_row provides an equivalent or faster streaming radix selector."
_installed = False


def install():
    global _installed
    if _installed:
        return
    try:
        module = importlib.import_module(_TARGET)
    except ModuleNotFoundError as exc:
        if exc.name and (_TARGET == exc.name or _TARGET.startswith(exc.name + ".")):
            return
        raise
    original = module.select_regions

    @functools.wraps(original)
    def select(scores, current, topk, backend="auto", **kwargs):
        if (not envs.VLLM_SAIL_STEP5_DSA_SELECTOR_PPU or not current_platform.is_ppu()
                or backend not in ("auto", "per_row") or not 0 < topk <= 1024
                or not scores.is_contiguous()
                or str(scores.dtype) != "torch.float32"):
            return original(scores, current, topk, backend, **kwargs)
        from vllm_sail.models.step5.ops.selector_v2 import select_regions
        return select_regions(scores, current, topk, backend, **kwargs)

    for name in (_TARGET, "vllm.models.step5.nvidia.flashinfer_sparse"):
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "select_regions", None) is original:
            patch(name, "select_regions", reason=_REASON,
                  affected_versions=">=0.32.0,<0.33.0", remove_when=_REMOVE)(select)
    _installed = True


install()
