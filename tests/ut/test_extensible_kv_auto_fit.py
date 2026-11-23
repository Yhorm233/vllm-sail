# SPDX-License-Identifier: Apache-2.0
"""Automatic context lengths follow final measured capacity, explicit ones do not."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


@pytest.fixture
def adapter(modules):
    path = (
        Path(__file__).parents[2] / "vllm_sail/patch/bugfix/extensible_kv_auto_fit.py"
    )
    spec = importlib.util.spec_from_file_location("_auto_fit_leaf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    platform = NS(is_ppu=lambda: True)
    modules("vllm.platforms", current_platform=platform)
    calls = []
    builder = object()
    modules(
        "vllm.v1.core.kv_cache_config_builder",
        get_kv_cache_config_builder=lambda config: builder,
    )
    planning = modules(
        "vllm.v1.core.kv_cache_planning",
        _pool_bytes_per_block_from_builder=lambda b, config, group: group[0],
        _auto_fit_max_model_len=lambda *args: calls.append(args),
    )
    config = NS(
        cache_config=NS(enable_extensible_kv_cache=True),
        model_config=NS(original_max_model_len=-1, max_model_len=1000),
    )
    return NS(
        module=module,
        platform=platform,
        planning=planning,
        builder=builder,
        calls=calls,
        config=config,
    )


@pytest.mark.parametrize("measured,expected", [([9, 7], 7), ([12, None], 10), ([], 10)])
def test_measured_minimum_reserves_null_block(adapter, measured, expected):
    workers = [NS(kv_cache_groups=[16]), NS(kv_cache_groups=[])]
    scheduler = NS(num_blocks=10, kv_cache_groups=[32])
    times = [NS(num_kv_blocks=value) for value in measured]
    token = object()

    def original(*args):
        assert args == (adapter.config, workers, scheduler, times)
        assert adapter.calls == [
            (
                adapter.builder,
                adapter.config,
                [[16], [], [32]],
                [(expected - 1) * 16, 0, (expected - 1) * 32],
            )
        ]
        return token

    assert (
        adapter.module._wrap_finalize(original)(
            adapter.config, workers, scheduler, times
        )
        is token
    )


@pytest.mark.parametrize(
    "ppu,auto,extensible",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_other_paths_keep_original_failure(adapter, ppu, auto, extensible):
    adapter.platform.is_ppu = lambda: ppu
    adapter.config.model_config.original_max_model_len = -1 if auto else 1000
    adapter.config.cache_config.enable_extensible_kv_cache = extensible
    error = ValueError("explicit context does not fit")

    def original(*args):
        raise error

    with pytest.raises(ValueError) as raised:
        adapter.module._wrap_finalize(original)(adapter.config, [], None, [])
    assert raised.value is error and not adapter.calls


@pytest.mark.parametrize("final", [1000, 700])
def test_workers_receive_final_context_before_return(adapter, final):
    calls = []
    engine = NS(collective_rpc=lambda *args, **kwargs: calls.append((args, kwargs)))
    token = object()

    def original(self, config):
        config.model_config.max_model_len = final
        return token

    assert adapter.module._wrap_initialize(original)(engine, adapter.config) is token
    assert calls == (
        [(("update_max_model_len",), {"args": (700,)})] if final == 700 else []
    )


def test_install_updates_preloaded_consumer_once(adapter, modules, patch_utils_module):
    def original(*args):
        return None

    class Engine:
        def _initialize_kv_caches(self, vllm_config):
            return None

    adapter.planning.finalize_extensible_kv_cache = original
    core = modules(
        "vllm.v1.engine.core", finalize_extensible_kv_cache=original, EngineCore=Engine
    )
    before = len(patch_utils_module.PATCH_REGISTRY)
    adapter.module.install()
    assert (
        core.finalize_extensible_kv_cache
        is adapter.planning.finalize_extensible_kv_cache
    )
    assert core.finalize_extensible_kv_cache is not original
    assert len(patch_utils_module.PATCH_REGISTRY) == before + 3
    adapter.module.install()
    assert len(patch_utils_module.PATCH_REGISTRY) == before + 3


def test_initializer_failure_is_preserved(adapter):
    error = RuntimeError("warmup failed")

    def original(*args):
        raise error

    with pytest.raises(RuntimeError) as raised:
        adapter.module._wrap_initialize(original)(object(), adapter.config)
    assert raised.value is error


@pytest.mark.upstream_source
@pytest.mark.parametrize("blocks", [1, 6, 15])
def test_real_binary_search_fits_final_capacity(adapter, upstream_source_root, blocks):
    path = upstream_source_root / "vllm/v1/core/kv_cache_planning.py"
    config = adapter.config
    config.attention_config = NS(hisparse_config=None)
    namespace = {
        "_max_memory_usage_bytes_from_groups": lambda b, c, g: (
            ((c.model_config.max_model_len + 15) // 16) * g[0]
        ),
        "logger": NS(info_once=lambda *args: None),
        "format_gib": str,
        "_kv_cache_device_name": lambda c: "PPU",
        "_more_kv_cache_memory_hint": lambda c: "insufficient capacity",
    }
    function(path, "_estimate_max_model_len_from_groups", namespace)
    adapter.planning._auto_fit_max_model_len = function(
        path, "_auto_fit_max_model_len", namespace
    )
    workers = [NS(kv_cache_groups=[16])]
    scheduler = NS(num_blocks=20, kv_cache_groups=[32])

    def admission(c, *args):
        assert 0 < c.model_config.max_model_len <= (blocks - 1) * 16
        return c.model_config.max_model_len

    call = adapter.module._wrap_finalize(admission)
    if blocks == 1:
        with pytest.raises(ValueError, match="single token"):
            call(config, workers, scheduler, [NS(num_kv_blocks=blocks)])
    else:
        assert (
            call(config, workers, scheduler, [NS(num_kv_blocks=blocks)])
            == (blocks - 1) * 16
        )
