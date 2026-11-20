# SPDX-License-Identifier: Apache-2.0
"""Advertise 810E-safe paged-KV block sizes to the GLM-5.3-Flash kpool indexer.

Zhenwu 810E cannot address ``page_kv`` values that are not a positive multiple of
64 in the SAIL DeepGEMM paged-MQA kernel. The GLM-5.3-Flash kpool indexer derives
``page_kv`` from the cache block size and ``index_kpool``
(``page_kv = block_size // index_kpool``), so the block sizes the indexer backend
declares as supported must keep ``page_kv`` on a 64-token boundary. Upstream only
exempts the SM120 family (``get_paged_mqa_page_sizes`` returns ``(64,)`` there and
``(32, 64)`` elsewhere), so an 810E launch can select a 32-token page and abort
during CUDA-graph warm-up.
"""

from __future__ import annotations

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import Glm5NextIndexerBackend

from vllm_sail.patch.utils import patch

_kernel_sizes = Glm5NextIndexerBackend.get_supported_kernel_block_sizes
_AFFECTED = ">=0.32.0,<0.33.0"


def _resolve_index_kpool(kv_cache_spec) -> int:
    """Read ``index_kpool`` the same way upstream does, never hard-coding it.

    ``get_kv_cache_spec`` runs with ``tokens_per_state`` already set; the
    platform block-size selection runs before a spec exists, under the current
    config. Mirror both paths so the page-size contract tracks the real geometry.
    """
    if kv_cache_spec is not None:
        return int(kv_cache_spec.tokens_per_state)
    return get_current_vllm_config().model_config.hf_text_config.index_kpool


@patch(
    "vllm.v1.attention.backends.mla.indexer",
    "Glm5NextIndexerBackend.get_supported_kernel_block_sizes",
    reason=(
        "Zhenwu 810E DeepGEMM paged-MQA requires page_kv (block_size // "
        "index_kpool) to be a positive multiple of 64; upstream exempts only "
        "SM120. Advertise block sizes that are multiples of index_kpool * 64 on "
        "810E so the launched page_kv always satisfies the kernel contract, while "
        "leaving the common-block selector free to pick a block compatible with "
        "other cache groups (for example the 640-token mamba pages)."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "The indexer backend obtains the platform page-size contract, including "
        "the 810E 64-token multiple, from SAIL DeepGEMM."
    ),
)
def get_supported_kernel_block_sizes(
    kv_cache_spec=None,
) -> list[int | MultipleOf]:
    if current_platform.is_ppu() and current_platform.is_device_capability((8, 0)):
        index_kpool = _resolve_index_kpool(kv_cache_spec)
        return [MultipleOf(index_kpool * 64)]
    # Match the exact 810E capability (8, 0), mirroring this repo's ``is_ppu_15``
    # idiom. ``is_device_capability_family(80)`` is NOT used here: it matches by
    # major, so 890P (8, 9) would also be True. 890P (SM89), other platforms and
    # CUDA/ROCm keep the upstream default: 890P still supports 32-token pages and
    # must remain byte-for-byte unchanged.
    return _kernel_sizes(kv_cache_spec=kv_cache_spec)
