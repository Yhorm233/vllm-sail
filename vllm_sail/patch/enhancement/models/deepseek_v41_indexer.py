# SPDX-License-Identifier: Apache-2.0
"""Give each V4.1 indexer cache the physical page size required by SAIL."""

from dataclasses import replace

from vllm.models.deepseek_v41 import attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend

from vllm_sail.models.deepseek_v41.config import uses_int8_w8a8
from vllm_sail.patch.utils import patch

_cache_spec = attention.DeepseekV4IndexerCache.get_kv_cache_spec
_kernel_sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes
_AFFECTED = ">=0.31.0,<0.32.0"


@patch(
    "vllm.models.deepseek_v41.attention",
    "DeepseekV4IndexerCache.get_kv_cache_spec",
    reason=(
        "SAIL paged indexer logits require 64 stored rows per page. V4.1 C1 and "
        "C2 indexers therefore need independent 64- and 128-token cache blocks; "
        "splitting the main cache's shared padded pages cannot express this."
    ),
    affected_versions=_AFFECTED,
    remove_when="V4.1 indexer cache sizing delegates to a platform-specific kernel contract.",
)
def get_kv_cache_spec(self, vllm_config):
    spec = _cache_spec(self, vllm_config)
    if current_platform.is_ppu():
        # Size the indexer's own manager blocks before cache grouping. Virtual
        # splitting cannot represent the shared, padded BLHNC allocation.
        # Reset the old padding so MLAAttentionSpec recomputes its alignment.
        spec = replace(
            spec, block_size=64 * self.compress_ratio, page_size_padded=None
        )
        if uses_int8_w8a8(vllm_config):
            config = vllm_config.model_config.hf_text_config
            spec = replace(spec, indexer_n_head=config.index_n_heads,
                           indexer_q_head_dim=config.index_head_dim)
    return spec


@patch(
    "vllm.v1.attention.backends.mla.indexer",
    "DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
    reason=(
        "PPU V4.1 indexer cache specs select 64/128 uncompressed tokens for "
        "C1/C2, respectively, so both retain 64 physical rows without virtual "
        "splitting of interleaved, padded cache pages."
    ),
    affected_versions=_AFFECTED,
    remove_when="The indexer backend obtains supported page sizes from the PPU kernel contract.",
)
def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
    if current_platform.is_ppu():
        return [64, 128]
    return _kernel_sizes(kv_cache_spec=kv_cache_spec)
