# SPDX-License-Identifier: Apache-2.0
"""Real PPU planning reconciles automatic lengths after measured KV shrink."""

from types import SimpleNamespace as NS

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("auto", [True, False])
def test_measured_cache_refits_only_automatic_context(auto):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.config import VllmConfig
    from vllm.platforms import current_platform
    from vllm.v1.core import kv_cache_planning as planning
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
    from vllm.v1.engine import core
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    if not current_platform.is_ppu():
        pytest.skip("Requires the PPU planning adapter")
    config = VllmConfig()
    config.model_config = NS(
        max_model_len=1000, original_max_model_len=-1 if auto else 1000
    )
    config.cache_config.enable_extensible_kv_cache = True
    config.cache_config.kv_cache_layout = "LBNHC"
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float32
    )
    groups = [KVCacheGroupSpec(["layer.0", "layer.1"], spec)]
    cache = planning.get_kv_cache_config_from_groups(
        config, groups, 2 * spec.page_size_bytes * 100
    )
    scheduler = generate_scheduler_kv_cache_config([cache])
    assert core.finalize_extensible_kv_cache is planning.finalize_extensible_kv_cache
    measured = [NS(num_kv_blocks=8), NS(num_kv_blocks=6)]
    if auto:
        blocks = core.finalize_extensible_kv_cache(
            vllm_config=config,
            kv_cache_configs=[cache],
            scheduler_kv_cache_config=scheduler,
            compilation_times=measured,
        )
        assert blocks == cache.num_blocks == scheduler.num_blocks == 6
        assert config.model_config.max_model_len == 80
    else:
        with pytest.raises(ValueError, match="max seq len"):
            core.finalize_extensible_kv_cache(config, [cache], scheduler, measured)
        assert config.model_config.max_model_len == 1000
