# SPDX-License-Identifier: Apache-2.0
"""Carry V4 query dimensions from the indexer into its metadata builder."""

from __future__ import annotations

from dataclasses import replace

from vllm.models.deepseek_v4 import attention
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.deepseek_v4.attention"
_METADATA = dict(
    reason="PPU paged logits metadata needs the indexer's unpadded query head count and dimension.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="DeepseekV4Indexer propagates query dimensions into its MLA cache spec.",
)
_cache_init = attention.DeepseekV4IndexerCache.__init__
_indexer_init = attention.DeepseekV4Indexer.__init__
_cache_spec = attention.DeepseekV4IndexerCache.get_kv_cache_config


@patch(
    _MODULE,
    "DeepseekV4IndexerCache.__init__",
    reason="PPU paged logits metadata needs the indexer query head count and dimension.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="DeepseekV4Indexer propagates query dimensions into its MLA cache spec.",
)
def cache_init(self, *args, n_head=None, q_head_dim=None, **kwargs):
    _cache_init(self, *args, **kwargs)
    self.n_head = n_head
    self.q_head_dim = q_head_dim


@patch(
    _MODULE,
    "DeepseekV4Indexer.__init__",
    reason="PPU paged logits metadata needs the indexer query head count and dimension.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="DeepseekV4Indexer propagates query dimensions into its MLA cache spec.",
)
def indexer_init(self, *args, **kwargs):
    _indexer_init(self, *args, **kwargs)
    if current_platform.is_ppu():
        self.k_cache.n_head = self.n_head
        self.k_cache.q_head_dim = self.head_dim


@patch(
    _MODULE,
    "DeepseekV4IndexerCache.get_kv_cache_config",
    reason="PPU paged logits metadata needs the indexer query head count and dimension.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="DeepseekV4Indexer propagates query dimensions into its MLA cache spec.",
)
def get_kv_cache_config(self, vllm_config):
    layer = _cache_spec(self, vllm_config)
    spec = layer.spec
    if current_platform.is_ppu():
        spec = replace(
            spec, indexer_n_head=self.n_head, indexer_q_head_dim=self.q_head_dim
        )
        return replace(layer, spec=spec)
    return layer
