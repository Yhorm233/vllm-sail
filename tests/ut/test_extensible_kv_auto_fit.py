# SPDX-License-Identifier: Apache-2.0
"""The pinned planner now refits measured KV capacity without a PPU patch."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


@pytest.mark.upstream_source
@pytest.mark.parametrize("auto", [True, False])
@pytest.mark.parametrize("measured,expected", [([9, 7], 7), ([12, None], 10), ([], 10)])
def test_final_capacity_refits_auto_context(
    upstream_source_root, auto, measured, expected
):
    calls = []
    workers = [NS(kv_cache_groups=[16]), NS(kv_cache_groups=[])]
    scheduler = NS(num_blocks=10, kv_cache_groups=[32])
    config = NS(
        cache_config=NS(), model_config=NS(original_max_model_len=-1 if auto else 1000)
    )
    finalize = function(
        upstream_source_root / "vllm/v1/core/kv_cache_planning.py",
        "finalize_extensible_kv_cache",
        {
            "get_kv_cache_config_builder": lambda cfg: "builder",
            "_pool_bytes_per_block_from_builder": lambda b, cfg, groups: groups[0],
            "_auto_fit_max_model_len": lambda *a, **kw: calls.append((a, kw)),
            "shrink_kv_cache_configs": lambda *a: calls.append(a),
            "update_kv_cache_capacity": lambda *a: calls.append(a),
        },
    )
    assert (
        finalize(config, workers, scheduler, [NS(num_kv_blocks=v) for v in measured])
        == expected
    )
    if auto:
        assert calls.pop(0) == (
            ("builder", config, [[16], []], [(expected - 1) * 16, 0]),
            {"after_warmup": True},
        )
    assert calls == [(config, [*workers, scheduler], expected), (config, scheduler)]
    assert config.cache_config.num_gpu_blocks == expected
