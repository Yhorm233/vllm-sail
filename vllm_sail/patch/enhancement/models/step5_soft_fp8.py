# SPDX-License-Identifier: Apache-2.0
"""Run Step5's E4M3 indexer kernels on PPU 1.0 (SM 8.0) in software.

PPU 1.0 Triton rejects ``fp8e4nv`` outright: casts, bitcasts, FP8 ``tl.dot``,
and an FP8 tensor anywhere in a kernel signature. Step5 DSA needs E4M3 in three
places: the summary-cache store, the indexer query store, and the two indexer
scorers. Each is replaced with the byte-based copy from
``vllm_sail/models/step5/ops/soft_fp8.py``, which compiles the upstream FP8
math when the target has E4M3 (SM 8.9+) and a bit-exact software conversion
otherwise.

* ``_store_summary`` / ``qkv_indexer_norm_rope._indexer_norm_rope`` are device
  helpers that other upstream kernels resolve from their module globals at
  compile time, so the copies are installed for every platform. Their native
  branch is upstream's code verbatim.
* ``indexer_norm_rope._indexer_norm_rope``, ``_joint_prepare``,
  ``_direct_scores`` and ``_paged_scores_sm90`` are launched by upstream
  Python launchers; they are replaced by ``ByteFP8Launch`` stand-ins that pass
  the E4M3 query as ``uint8`` on PPU and launch upstream unchanged elsewhere.
  The launchers still allocate and return ``torch.float8_e4m3fn`` tensors, so
  every dtype check downstream holds.

Mode 0 keeps E4M3 byte storage; mode 1 keeps the quantization and stores FP16.
Q and summary both keep the E4M3 quantization; the main attention helpers are
unchanged.

Patches install before any kernel compiles (plugin load), which is what makes
replacing a module-global ``@triton.jit`` helper effective.
"""

from __future__ import annotations

from vllm.models.step5.common.ops import (
    indexer_norm_rope as _indexer,
    sparse_indexer_scores as _scores,
    summary_update as _summary,
)
from vllm.models.step5.nvidia.ops import qkv_indexer_norm_rope as _joint

from vllm_sail.models.step5.ops import soft_fp8
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_REASON = (
    "PPU 1.0 (SM 8.0) Triton lacks FP8 instructions and arguments; Step5 indexer "
    "mode 0 stores E4M3 as bytes with software conversion, and mode 1 stores "
    "E4M3-quantized values as FP16."
)
_REMOVE_WHEN = (
    "Step5 DSA ships a software-E4M3 path for targets without hardware FP8, "
    "or PPU 1.0 Triton supports fp8e4nv."
)


def _install(module: str, name: str, replacement) -> None:
    patch(
        module,
        name,
        reason=_REASON,
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )(replacement)


# Device helpers: resolved by upstream kernels through module globals.
_install(
    "vllm.models.step5.common.ops.summary_update",
    "_store_summary",
    soft_fp8._store_summary,
)
_install(
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope",
    "_indexer_norm_rope",
    soft_fp8._indexer_norm_rope,
)

# Launch sites: E4M3 query arguments are passed as bytes on PPU.
_install(
    "vllm.models.step5.common.ops.indexer_norm_rope",
    "_indexer_norm_rope",
    soft_fp8.ByteFP8Launch(
        _indexer._indexer_norm_rope,
        soft_fp8._indexer_norm_rope,
        byte_args=(8,),
    ),
)
# _joint_prepare itself has no FP8 op; its indexer half is the helper above.
# Only QOut is converted: its FP8_CACHE store relies on the cache's FP8 type.
_install(
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope",
    "_joint_prepare",
    soft_fp8.ByteFP8Launch(
        _joint._joint_prepare,
        _joint._joint_prepare,
        byte_args=(8,),
    ),
)
_install(
    "vllm.models.step5.common.ops.sparse_indexer_scores",
    "_direct_scores",
    soft_fp8.ByteFP8Launch(
        _scores._direct_scores, soft_fp8._direct_scores, byte_args=(0,)
    ),
)
_install(
    "vllm.models.step5.common.ops.sparse_indexer_scores",
    "_paged_scores_sm90",
    soft_fp8.ByteFP8Launch(
        _scores._paged_scores_sm90, soft_fp8._paged_scores_sm90, byte_args=(0,)
    ),
)
