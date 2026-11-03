# SPDX-License-Identifier: Apache-2.0
"""Exercise V4.1 cache sizing and upstream kernel selection without torch."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.support.source import function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu,capability", [(True, 89), (False, 90), (False, 100)])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("manager_block", [128, 256])
def test_indexer_cache_pages(
    modules, upstream_source_root, ppu, capability, ratio, manager_block
):
    @dataclass(frozen=True)
    class Spec:
        block_size: int
        num_kv_heads: int
        head_size: int
        dtype: object
        tokens_per_state: int
        alignment: int
        block_stride_alignment: int | None = None
        page_size_padded: int | None = None

        def __post_init__(self):
            size = self.block_size // self.tokens_per_state * self.head_size
            if self.page_size_padded is None:
                object.__setattr__(
                    self,
                    "page_size_padded",
                    -(-size // self.alignment) * self.alignment,
                )

    platform = SimpleNamespace(
        is_ppu=lambda: ppu,
        is_device_capability_family=lambda value: capability == value,
        is_device_capability=lambda value: capability == value[0] * 10 + value[1],
    )
    modules("vllm.platforms", current_platform=platform)
    modules("vllm_sail.models.deepseek_v41.config", uses_int8_w8a8=lambda config: False)
    indexer_source = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    kernel_sizes = function(
        indexer_source,
        "DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
        {"current_platform": platform},
    )
    backend = type(
        "Backend", (), {"get_supported_kernel_block_sizes": staticmethod(kernel_sizes)}
    )
    modules("vllm.v1.attention.backends.mla.indexer", DeepseekV41IndexerBackend=backend)
    cache_spec = function(
        upstream_source_root / "vllm/models/deepseek_v41/attention.py",
        "DeepseekV4IndexerCache.get_kv_cache_spec",
        {
            "MLAAttentionSpec": Spec,
            "_use_v41_mxfp8_kv_record": lambda: capability == 100,
        },
    )
    cache_cls = type("Cache", (), {"get_kv_cache_spec": cache_spec})
    modules("vllm.models.deepseek_v41.attention", DeepseekV4IndexerCache=cache_cls)
    cache = cache_cls()
    cache.cache_config = SimpleNamespace(block_size=manager_block)
    cache.compress_ratio, cache.head_dim, cache.dtype = ratio, 132, "uint8"
    cache.sparse_logits = False
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"))
    before = cache.get_kv_cache_spec(config)
    harness.load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py")
    spec = cache.get_kv_cache_spec(config)
    select = function(
        upstream_source_root / "vllm/v1/worker/utils.py",
        "select_common_block_size",
        {"MultipleOf": type("MultipleOf", (), {})},
    )
    selected = select(spec.block_size, [backend])
    assert backend.get_supported_kernel_block_sizes(kv_cache_spec=spec) == (
        [64, 128] if ppu else [64 if capability == 90 else 128]
    )
    if ppu:
        assert spec.block_size == selected == 64 * ratio
        assert spec.block_size // spec.tokens_per_state == 64
        # Recompute alignment after changing the block size; never retain the
        # old 128/256-row page padding. No virtual splitting is needed.
        assert spec.page_size_padded == 8640
    else:
        assert spec == before
        assert selected == (64 if capability == 90 else 128)
    assert cache.cache_config.block_size == manager_block
