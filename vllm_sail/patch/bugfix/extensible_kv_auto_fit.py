# SPDX-License-Identifier: Apache-2.0
"""Reconcile automatic context length with post-warmup PPU cache capacity."""

from __future__ import annotations

import functools

from vllm_sail.patch.utils import patch

_installed = False


def _uses_auto_fit(config):
    from vllm.platforms import current_platform

    return (
        current_platform.is_ppu()
        and config.cache_config.enable_extensible_kv_cache
        and config.model_config.original_max_model_len == -1
    )


def _wrap_finalize(original):
    @functools.wraps(original)
    def finalize(
        vllm_config, kv_cache_configs, scheduler_kv_cache_config, compilation_times
    ):
        config = vllm_config
        worker_configs = kv_cache_configs
        scheduler_config = scheduler_kv_cache_config
        if _uses_auto_fit(config):
            from vllm.v1.core import kv_cache_planning as planning
            from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder

            measured = [
                times.num_kv_blocks
                for times in compilation_times
                if times.num_kv_blocks is not None
            ]
            num_blocks = min([scheduler_config.num_blocks, *measured])
            builder = get_kv_cache_config_builder(config)
            groups = [
                cache.kv_cache_groups for cache in [*worker_configs, scheduler_config]
            ]
            memory = [
                (num_blocks - 1)
                * planning._pool_bytes_per_block_from_builder(builder, config, group)
                if group
                else 0
                for group in groups
            ]
            # Use the same builder, cross-rank minimum and null-block reservation
            # as upstream admission. Explicit context lengths still fail there.
            planning._auto_fit_max_model_len(builder, config, groups, memory)
        return original(config, worker_configs, scheduler_config, compilation_times)

    return finalize


def _wrap_initialize(original):
    @functools.wraps(original)
    def initialize(self, vllm_config):
        config = vllm_config
        before = config.model_config.max_model_len
        result = original(self, config)
        after = config.model_config.max_model_len
        if _uses_auto_fit(config) and after != before:
            # The first upstream update precedes warmup. Refresh cached worker
            # limits again after the final measured fit, before serving starts.
            self.collective_rpc("update_max_model_len", args=(after,))
        return result

    return initialize


def install():
    global _installed
    if _installed:
        return
    from vllm.v1.core import kv_cache_planning
    from vllm.v1.engine import core

    metadata = dict(
        reason=(
            "PPU warmup can shrink extensible KV capacity below the initial "
            "auto-fit estimate. Refit automatic lengths to measured capacity "
            "and synchronize workers before admitting requests."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when=(
            "Upstream extensible KV finalization repeats auto-fit and updates "
            "worker context limits after measured sizing."
        ),
    )
    original = kv_cache_planning.finalize_extensible_kv_cache
    replacement = _wrap_finalize(original)
    patch("vllm.v1.core.kv_cache_planning", "finalize_extensible_kv_cache", **metadata)(
        replacement
    )
    # EngineCore caches this function by value, including when imported early.
    if core.finalize_extensible_kv_cache is original:
        patch("vllm.v1.engine.core", "finalize_extensible_kv_cache", **metadata)(
            replacement
        )
    patch("vllm.v1.engine.core", "EngineCore._initialize_kv_caches", **metadata)(
        _wrap_initialize(core.EngineCore._initialize_kv_caches)
    )
    _installed = True
