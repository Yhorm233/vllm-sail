# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiniMax M3 SAIL-MSA gating policy.

``gating`` needs only a handful of names from ``torch``, ``vllm.logger`` and
``vllm.platforms``, so the CPU-only tier stubs exactly those instead of
skipping — the dtype matrix is pure policy and must be covered everywhere.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from collections.abc import Iterator

import pytest

import vllm_sail.models.minimax_m3_msa  # noqa: F401  (marks the parent package)

_MODULE = "vllm_sail.models.minimax_m3_msa.gating"

#: Shapes of the M3.1 checkpoint at TP8 (one KV head, one index head per rank).
_BASE = dict(
    topk_blocks=16,
    block_size=128,
    score_type="max",
    head_dim=128,
    idx_head_dim=128,
    num_q_heads=8,
    num_kv_heads=1,
    num_idx_heads=1,
    kv_cache_dtype="fp8",
    indexer_kv_dtype="fp8",
    max_model_len=100_000,
)


class _OnceLogger:
    """Stand-in for vLLM's ``init_logger`` result (``*_once`` methods)."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def __getattr__(self, name: str):
        return getattr(self._logger, name)

    def info_once(self, *args, **kwargs) -> None:
        del args, kwargs

    def warning_once(self, *args, **kwargs) -> None:
        del args, kwargs


class _Harness:
    """Loaded gating module with the fakes its decisions depend on."""

    def __init__(self, module: types.ModuleType, capability: tuple[int, int]) -> None:
        self.module = module
        self.capability = capability

    def decide(self, **overrides):
        return self.module.decide(**{**_BASE, **overrides})

    def set_sail_available(self, available: bool) -> None:
        self.module.sail_msa_available = lambda: available


@pytest.fixture(params=[(8, 9), (8, 0)], ids=["890p", "810e"])
def harness(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_Harness]:
    capability = request.param
    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []
        monkeypatch.setitem(sys.modules, "vllm", vllm)

    logger_mod = sys.modules.get("vllm.logger")
    if logger_mod is None or not hasattr(logger_mod, "init_logger"):
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = _OnceLogger
        monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)

    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(is_ppu=lambda: True)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    torch = types.ModuleType("torch")
    torch.float32 = "torch.float32"
    torch.bfloat16 = "torch.bfloat16"
    torch.dtype = object
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda *a, **k: capability)
    monkeypatch.setitem(sys.modules, "torch", torch)

    monkeypatch.delenv("VLLM_SAIL_MINIMAX_M3_MSA", raising=False)
    monkeypatch.delenv("VLLM_PPU_MINIMAX_M3_MSA", raising=False)

    monkeypatch.delitem(sys.modules, _MODULE, raising=False)
    module = importlib.import_module(_MODULE)
    module.sail_msa_available = lambda: True
    yield _Harness(module, capability)
    sys.modules.pop(_MODULE, None)


def test_890p_fp8_defaults_select_fp8_trio(harness: _Harness) -> None:
    if harness.capability != (8, 9):
        pytest.skip("890P-only default")
    selection = harness.decide()
    assert selection.enable
    assert selection.indexer_fp8 and selection.attend_fp8
    assert selection.score_dtype == "torch.bfloat16"


def test_bf16_kv_selects_bf16_trio(harness: _Harness) -> None:
    selection = harness.decide(kv_cache_dtype="bfloat16", indexer_kv_dtype="bf16")
    assert selection.enable
    assert not selection.indexer_fp8
    assert not selection.attend_fp8
    assert selection.score_dtype == "torch.float32"


def test_890p_explicit_bf16_indexer_keeps_fp8_attend(harness: _Harness) -> None:
    if harness.capability != (8, 9):
        pytest.skip("890P-only combination")
    selection = harness.decide(indexer_kv_dtype="bf16")
    assert selection.enable
    assert not selection.indexer_fp8
    assert selection.attend_fp8
    assert selection.score_dtype == "torch.float32"


def test_890p_bf16_kv_with_fp8_indexer_is_served(harness: _Harness) -> None:
    if harness.capability != (8, 9):
        pytest.skip("890P-only combination")
    selection = harness.decide(kv_cache_dtype="bfloat16", indexer_kv_dtype="fp8")
    assert selection.enable
    assert selection.indexer_fp8 and not selection.attend_fp8
    # fp8 OnlyScore only has a bf16 logits variant.
    assert selection.score_dtype == "torch.bfloat16"


def test_810e_rejects_fp8_paths(harness: _Harness) -> None:
    if harness.capability == (8, 9):
        pytest.skip("810E-only policy")
    selection = harness.decide()
    assert not selection.enable
    assert "890P" in selection.reason
    selection = harness.decide(indexer_kv_dtype="bf16")
    assert not selection.enable
    assert "890P" in selection.reason


def test_disable_switch_and_missing_library(harness: _Harness, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SAIL_MINIMAX_M3_MSA", "0")
    assert not harness.decide().enable
    monkeypatch.delenv("VLLM_SAIL_MINIMAX_M3_MSA")
    harness.set_sail_available(False)
    selection = harness.decide()
    assert not selection.enable
    assert "msa_for_sail" in selection.reason


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"topk_blocks": 32}, "sparse_topk_blocks"),
        ({"block_size": 64}, "block_size"),
        ({"score_type": "sum"}, "score_type"),
        ({"head_dim": 64}, "head_dim"),
        ({"idx_head_dim": 64}, "sparse_index_dim"),
        ({"num_idx_heads": 2}, "num_index_heads"),
        ({"max_model_len": 12_288 * 128}, "tile limit"),
        ({"kv_cache_dtype": "nvfp4"}, "kv_cache_dtype"),
        ({"indexer_kv_dtype": "nvfp4"}, "indexer_kv_dtype"),
    ],
)
def test_unsupported_shapes_disable_the_path(
    harness: _Harness, overrides: dict, needle: str
) -> None:
    selection = harness.decide(**overrides)
    assert not selection.enable
    assert needle in selection.reason


def test_default_indexer_dtype_follows_890p_fp8_cache(harness: _Harness) -> None:
    config = _config(cache_dtype="fp8")
    default = harness.module.default_indexer_kv_dtype("bf16", config)
    if harness.capability == (8, 9):
        assert default == "fp8"
    else:
        assert default is None

    assert harness.module.default_indexer_kv_dtype("bf16", _config("bfloat16")) is None
    assert harness.module.default_indexer_kv_dtype("fp8", config) is None


@pytest.mark.parametrize("model_type", ["qwen4_exp", None])
def test_default_indexer_dtype_ignores_other_models(
    harness: _Harness, model_type: str | None
) -> None:
    # Other sparse-indexer models resolve the same AttentionConfig field.
    config = _config(cache_dtype="fp8", model_type=model_type)
    assert harness.module.default_indexer_kv_dtype("bf16", config) is None
    assert not harness.module.is_minimax_m3(None)


def test_multimodal_wrapper_is_recognized_by_its_text_config(
    harness: _Harness,
) -> None:
    config = _config(cache_dtype="fp8")
    text_config = config.model_config.hf_config
    config.model_config.hf_config = types.SimpleNamespace(
        model_type="wrapper", text_config=text_config
    )
    assert harness.module.is_minimax_m3(config)


def test_decide_from_config_resolves_auto_index_dtype(harness: _Harness) -> None:
    selection = harness.module.decide_from_config(_config(cache_dtype="fp8"))
    if harness.capability == (8, 9):
        assert selection.enable
        assert selection.indexer_fp8 and selection.attend_fp8
        assert selection.score_dtype == "torch.bfloat16"
    else:
        assert not selection.enable

    selection = harness.module.decide_from_config(_config(cache_dtype="bfloat16"))
    assert selection.enable
    assert not selection.indexer_fp8 and not selection.attend_fp8
    assert selection.score_dtype == "torch.float32"


def _config(
    cache_dtype: str, model_type: str | None = "minimax_m3_text"
) -> types.SimpleNamespace:
    hf_config = types.SimpleNamespace(
        model_type=model_type,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        index_topk_blocks=16,
        index_head_dim=128,
        index_n_heads=4,
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config, max_model_len=100_000),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
        cache_config=types.SimpleNamespace(block_size=128, cache_dtype=cache_dtype),
        attention_config=types.SimpleNamespace(indexer_kv_dtype="auto"),
    )
