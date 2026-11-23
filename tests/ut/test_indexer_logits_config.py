# SPDX-License-Identifier: Apache-2.0
"""Model config resolution must agree with the PPU indexer's native reduction."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import function


@pytest.fixture
def policy(modules, patch_utils_module):
    calls = []
    platform = NS(ppu=True)
    platform.is_ppu = lambda: platform.ppu
    modules("vllm.platforms", current_platform=platform)

    def original(config):
        calls.append(config)
        return "upstream"

    provider = modules(
        "vllm.model_executor.models.config",
        _set_deepseek_indexer_logits_dtype=original,
    )
    modules(
        "vllm.v1.attention.backends.mla.indexer",
        dsa_indexer_uses_fp4=lambda config: config.fp4,
    )
    load_patch("vllm_sail/patch/enhancement/attention/indexer_logits_config.py")
    return NS(provider=provider, platform=platform, calls=calls)


def make_config(requested, fp4):
    attention = NS(indexer_logits_dtype=requested, indexer_sparse_logits=True)
    attention.resolve_indexer_logits_dtype = lambda default: (
        default
        if attention.indexer_logits_dtype == "auto"
        else attention.indexer_logits_dtype
    )
    return NS(attention_config=attention, fp4=fp4)


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("requested", ["auto", "float32", "bfloat16"])
def test_auto_default_and_explicit_requests(policy, ppu, fp4, requested):
    policy.platform.ppu = ppu
    config = make_config(requested, fp4)
    result = policy.provider._set_deepseek_indexer_logits_dtype(config)
    if not ppu:
        assert result == "upstream" and policy.calls == [config]
        assert config.attention_config.indexer_logits_dtype == requested
    else:
        expected = (
            ("bfloat16" if fp4 else "float32") if requested == "auto" else requested
        )
        assert config.attention_config.indexer_logits_dtype == expected
        assert policy.calls == []
    assert config.attention_config.indexer_sparse_logits is True


@pytest.mark.upstream_source
@pytest.mark.parametrize("requested", ["auto", "float32", "bfloat16"])
def test_real_model_config_hook_preserves_native_default(
    policy, modules, upstream_source_root, requested
):
    path = upstream_source_root / "vllm/model_executor/models/config.py"
    modules("vllm.utils.deep_gemm", mqa_logits_use_mx_scales=lambda: False)
    original = function(path, "_set_deepseek_indexer_logits_dtype", {})
    before = make_config(requested, True)
    original(before)
    assert before.attention_config.indexer_logits_dtype == (
        "float32" if requested == "auto" else requested
    )
    # Exercise the actual upstream hook, not only the layer's late resolver.
    hook = function(
        path,
        "DeepseekV4ForCausalLMConfig.verify_and_update_config",
        {
            "_set_deepseek_indexer_logits_dtype": policy.provider._set_deepseek_indexer_logits_dtype,
            "_indexer_sparse_logits_supported": lambda config: False,
        },
    )
    config = make_config(requested, True)
    hook(config)
    assert config.attention_config.indexer_logits_dtype == (
        "bfloat16" if requested == "auto" else requested
    )
    assert config.attention_config.indexer_sparse_logits is True
