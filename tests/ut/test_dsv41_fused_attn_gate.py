# SPDX-License-Identifier: Apache-2.0
"""Offline UT for the PR#229 fused norm-rope gate wiring.

CPU-only tier: no torch, no vLLM, no device. Covers the three pieces of
PR#229 that are decidable without a kernel:

* ``vllm_sail.envs`` parsing of the three new switches
  (``VLLM_SAIL_FUSED_NORM_ROPE_ATTN`` / ``VLLM_SAIL_PPU_KV_MXFP8`` /
  ``VLLM_SAIL_PPU_KV_NVFP4``) including their auto-derived ``VLLM_PPU_*``
  aliases and SAIL precedence;
* ``fused_gate`` in ``vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py``:
  every gate branch (env, kv-record, platform, capability, the opt-in NVFP4
  capability/top-k checks, the PR#59 per-token-scale capability check, wheel
  probe, topology) and the fully-open report;
* the PR#59 output scale arms: ``VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE`` parsing
  and alias precedence, ``_output_scale_args`` for both arms (read live, so an
  A/B run flips only the environment), the ``_call_fused`` wheel-requirement
  error, and static wiring of ``_alloc_attn_out`` / ``_o_proj`` / both kernel
  call sites;
* the ``dsv41_kv_record`` patch module: non-PPU delegation, once-per-process
  env caching, the forensic log line, and the patch metadata (target,
  affected_versions against the repo's SUPPORTED_RANGE);
* the ``_select_dsv4_attn_cls`` selector patch: which attention class is
  returned for each (platform, selected, gate) combination.

Torch-dependent bodies (``_o_proj``, ``_dequant_fused_output``,
``_forward_{decode,prefill}_mega``) are intentionally out of scope here;
they are covered by the FlashMLA kernel UTs (PR#54, plus PR#59's
``--output-scale {block32,per-token}`` cases) and on-device smoke.

The one exception is the optional numerical tier at the bottom of this file:
``_wo_a_scale_mode`` and ``_dequant_fp8_per_channel`` are pure tensor helpers
with no kernel and no vLLM dependency, so they are executed against real torch
when it is installed and skipped explicitly when it is not. The CPU-only merge
gate therefore still passes without torch, vLLM or a device.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.upstream import SUPPORTED_RANGE
from vllm_sail import envs

_REAL_FIND_SPEC = importlib.util.find_spec

_SWITCH_NAMES = (
    "VLLM_SAIL_FUSED_NORM_ROPE_ATTN",
    "VLLM_PPU_FUSED_NORM_ROPE_ATTN",
    "VLLM_SAIL_PPU_KV_MXFP8",
    "VLLM_PPU_PPU_KV_MXFP8",
    "VLLM_SAIL_PPU_KV_NVFP4",
    "VLLM_PPU_PPU_KV_NVFP4",
    "VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE",
    "VLLM_PPU_FUSED_ATTN_PER_TOKEN_SCALE",
)


@pytest.fixture(autouse=True)
def _clean_switch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SWITCH_NAMES:
        monkeypatch.delenv(name, raising=False)


def _config(
    heads: int = 64,
    groups: int = 8,
    tp: int = 8,
    *,
    nested: bool = False,
    sliding_window: int | None = None,
):
    hf = SimpleNamespace(num_attention_heads=heads, o_groups=groups)
    if sliding_window is not None:
        hf.sliding_window = sliding_window
    if nested:
        hf = SimpleNamespace(text_config=hf, num_attention_heads=None, o_groups=None)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        lora_config=None,
    )


# ---------------------------------------------------------------------------
# env switch parsing
# ---------------------------------------------------------------------------


def test_new_switches_default_off() -> None:
    """Opt-in contract: unconfigured processes keep the baseline path."""
    assert envs.VLLM_SAIL_FUSED_NORM_ROPE_ATTN is False
    assert envs.VLLM_SAIL_PPU_KV_MXFP8 is False
    assert envs.VLLM_SAIL_PPU_KV_NVFP4 is False
    snapshot = envs.snapshot()
    assert snapshot["VLLM_SAIL_FUSED_NORM_ROPE_ATTN"] is False
    assert snapshot["VLLM_SAIL_PPU_KV_MXFP8"] is False
    assert snapshot["VLLM_SAIL_PPU_KV_NVFP4"] is False


@pytest.mark.parametrize(
    ("sail", "ppu"),
    [
        ("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "VLLM_PPU_FUSED_NORM_ROPE_ATTN"),
        ("VLLM_SAIL_PPU_KV_MXFP8", "VLLM_PPU_PPU_KV_MXFP8"),
        ("VLLM_SAIL_PPU_KV_NVFP4", "VLLM_PPU_PPU_KV_NVFP4"),
        (
            "VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE",
            "VLLM_PPU_FUSED_ATTN_PER_TOKEN_SCALE",
        ),
    ],
)
def test_switches_turn_on_via_either_name(
    monkeypatch: pytest.MonkeyPatch, sail: str, ppu: str
) -> None:
    for name in (sail, ppu):
        monkeypatch.setenv(name, "1")
        assert getattr(envs, sail) is True
        assert getattr(envs, ppu) is True
        monkeypatch.delenv(name)
        assert getattr(envs, sail) is False


@pytest.mark.parametrize(
    ("sail", "ppu"),
    [
        ("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "VLLM_PPU_FUSED_NORM_ROPE_ATTN"),
        ("VLLM_SAIL_PPU_KV_MXFP8", "VLLM_PPU_PPU_KV_MXFP8"),
        ("VLLM_SAIL_PPU_KV_NVFP4", "VLLM_PPU_PPU_KV_NVFP4"),
        (
            "VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE",
            "VLLM_PPU_FUSED_ATTN_PER_TOKEN_SCALE",
        ),
    ],
)
def test_sail_name_takes_precedence_over_ppu_alias(
    monkeypatch: pytest.MonkeyPatch, sail: str, ppu: str
) -> None:
    monkeypatch.setenv(ppu, "0")
    monkeypatch.setenv(sail, "1")
    assert getattr(envs, sail) is True
    monkeypatch.setenv(sail, "0")
    monkeypatch.setenv(ppu, "1")
    assert getattr(envs, ppu) is False


# ---------------------------------------------------------------------------
# fused_gate
# ---------------------------------------------------------------------------


@pytest.fixture
def gate(modules, monkeypatch):
    logs = []
    warnings = []
    logger = SimpleNamespace(
        info_once=lambda msg, *a: logs.append((msg, a)),
        warning_once=lambda msg, *a: warnings.append((msg, a)),
    )
    modules("torch")
    modules("vllm.logger", init_logger=lambda name: logger)
    modules(
        "vllm.model_executor.layers.fusion.quant_activation",
        QuantizedActivation=type("QuantizedActivation", (), {}),
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda weight, scale: weight,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        kFp8DynamicTokenSym="kFp8DynamicTokenSym",
    )
    modules(
        "vllm.models.deepseek_v41.common.ops",
        combine_topk_swa_indices=lambda *a, **k: None,
        dequantize_and_gather_k_cache=lambda *a, **k: None,
    )
    modules("vllm.models.deepseek_v41.common.ops.fused_layout", WV_GROUP_SIZE=8)
    modules(
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
        _HEAD_DIM_V=512,
        _QUANT_GROUP=32,
        _ROPE_DIM=64,
        _token_slice=lambda out, start, end: out,
        alloc_mega_attn_output=lambda num_tokens, n_wv_group, device: SimpleNamespace(
            num_tokens=num_tokens, n_wv_group=n_wv_group, device=device
        ),
        DeepseekV4MegaAttnAttention=type("DeepseekV4MegaAttnAttention", (), {}),
    )
    modules(
        "vllm.models.deepseek_v41.sparse_mla",
        DeepseekV4FlashMLABackend=type("DeepseekV4FlashMLABackend", (), {}),
    )
    modules("vllm.utils.math_utils", round_up=lambda x, y: -(-x // y) * y)
    modules("vllm.v1.worker.workspace", current_workspace_manager=lambda: None)
    platform = SimpleNamespace(
        is_ppu=lambda: True,
        get_device_capability=lambda device_id=0: SimpleNamespace(major=8, minor=9),
    )
    modules("vllm.platforms", current_platform=platform)
    module = load_patch("vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py")
    # Opt-in contract: gate tests explicitly enable both switches; the
    # default-off behaviour is asserted separately below.
    monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "1")
    monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "1")

    def set_capability(value):
        monkeypatch.setattr(
            platform, "get_device_capability", lambda device_id=0: value
        )

    def set_wheel(state: str) -> None:
        """state: missing | api | no-api | no-decode | import-error"""
        if state == "missing":
            monkeypatch.setattr(
                importlib.util,
                "find_spec",
                lambda name, *a, **k: (
                    None if name == "flash_mla" else _REAL_FIND_SPEC(name, *a, **k)
                ),
            )
            return
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: (
                SimpleNamespace(name=name)
                if name == "flash_mla"
                else _REAL_FIND_SPEC(name, *a, **k)
            ),
        )
        if state == "api":
            modules(
                "flash_mla",
                fused_norm_rope_attn_rope_cast=SimpleNamespace(
                    prefill=object(), decode=object()
                ),
            )
        elif state == "no-api":
            modules("flash_mla")
        elif state == "no-decode":
            modules(
                "flash_mla",
                fused_norm_rope_attn_rope_cast=SimpleNamespace(prefill=object()),
            )
        elif state == "import-error":
            monkeypatch.setitem(
                __import__("sys").modules, "flash_mla", None
            )  # import of flash_mla halts -> ImportError
        else:  # pragma: no cover - guard against typos in tests
            raise AssertionError(f"unknown wheel state {state!r}")

    return SimpleNamespace(
        module=module,
        platform=platform,
        logs=logs,
        warnings=warnings,
        set_capability=set_capability,
        set_wheel=set_wheel,
        monkeypatch=monkeypatch,
    )


def test_gate_closed_by_default_without_env_opt_in(gate) -> None:
    """Requirement: the fused path never runs on an unconfigured process."""
    gate.monkeypatch.delenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN")
    gate.monkeypatch.delenv("VLLM_SAIL_PPU_KV_MXFP8")
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-env:")
    assert (
        gate.module.PPUDeepseekV4MegaAttnAttention.is_available_for(_config()) is False
    )


def test_gate_env_switch_closes_first(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "0")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-env:")


def test_gate_kv_record_switch_closes(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "0")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-kvrecord:")


def test_gate_non_ppu_platform_closes(gate) -> None:
    gate.platform.is_ppu = lambda: False
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-platform:")


@pytest.mark.parametrize("cap", [None, (9, 0), (8, 6), (10, 0)])
def test_gate_capability_outside_ppu_sm80_sm89_closes(gate, cap) -> None:
    value = None if cap is None else SimpleNamespace(major=cap[0], minor=cap[1])
    gate.set_capability(value)
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-cap:")


@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
def test_gate_accepts_both_ppu_capabilities(gate, cap) -> None:
    gate.set_capability(SimpleNamespace(major=cap[0], minor=cap[1]))
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert ok
    assert report.startswith(f"all gates open: cap={cap[0]}.{cap[1]} ")


def test_lora_closes_fused_gate_before_the_wheel_probe(gate, monkeypatch) -> None:
    config = _config()
    config.lora_config = object()

    def forbidden(*args, **kwargs):
        raise AssertionError("LoRA 配置不应探测 fused wheel")

    monkeypatch.setattr(importlib.util, "find_spec", forbidden)
    ok, report = gate.module.fused_gate(config)
    assert not ok and report.startswith("gate-lora:")
    assert gate.module.PPUDeepseekV4MegaAttnAttention.is_available_for(config) is False


def test_gate_missing_wheel_closes(gate) -> None:
    gate.set_wheel("missing")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report == "gate-wheel: flash_mla not installed"


@pytest.mark.parametrize("state", ["no-api", "no-decode"])
def test_gate_baseline_wheel_without_fused_api_closes(gate, state) -> None:
    gate.set_wheel(state)
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith(
        "gate-wheel: flash_mla lacks fused_norm_rope_attn_rope_cast"
    )


def test_gate_wheel_import_error_closes_with_repr(gate) -> None:
    gate.set_wheel("import-error")
    ok, report = gate.module.fused_gate(_config())
    assert not ok
    assert report.startswith("gate-wheel: flash_mla import failed:")


@pytest.mark.parametrize(
    ("heads", "groups", "tp"),
    [
        (60, 8, 8),  # heads not divisible by tp
        (64, 4, 8),  # o_groups not divisible by tp
        (64, 8, 16),  # local_heads=4 < WV_GROUP_SIZE
        (64, 16, 8),  # 4 local heads per local group, need exactly 8
        (64, 0, 8),  # no o_groups at all
    ],
)
def test_gate_topology_mismatch_closes(gate, heads, groups, tp) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(heads, groups, tp))
    assert not ok
    assert report.startswith("gate-topology:")


def test_gate_fully_open_tp8_reports_padded_heads(gate) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert ok
    assert "tp=8 local_heads=8 padded_heads=64" in report


def test_gate_reads_nested_text_config(gate) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(nested=True))
    assert ok
    assert "local_heads=8" in report


# ---------------------------------------------------------------------------
# opt-in NVFP4 compressed record (VLLM_SAIL_PPU_KV_NVFP4)
# ---------------------------------------------------------------------------


def test_gate_nvfp4_off_by_default_keeps_mxfp8_record(gate) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(sliding_window=128))
    assert ok
    assert "compressed_kv=mxfp8_528B" in report
    assert gate.warnings == []


@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
def test_gate_nvfp4_requires_capability_89(gate, cap) -> None:
    """The 288B record is written with the e2m1 PTX conversion (8.9 only)."""
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    gate.set_capability(SimpleNamespace(major=cap[0], minor=cap[1]))
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(sliding_window=128))
    if cap == (8, 9):
        assert ok
        assert "compressed_kv=nvfp4_288B" in report
    else:
        assert not ok
        assert report.startswith("gate-nvfp4-cap:")
        assert "8.9" in report


def test_gate_nvfp4_still_requires_the_mxfp8_swa_record(gate) -> None:
    """gate-kvrecord closes first: the main cache must stay 528B."""
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "0")
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(sliding_window=128))
    assert not ok
    assert report.startswith("gate-kvrecord:")


@pytest.mark.parametrize("window", [8, 64, 128, 2048])
def test_gate_nvfp4_accepts_topk_multiple_of_8(gate, window) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(sliding_window=window))
    assert ok
    assert "compressed_kv=nvfp4_288B" in report


@pytest.mark.parametrize("window", [1, 12, 130, 2050])
def test_gate_nvfp4_rejects_topk_not_multiple_of_8(gate, window) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(sliding_window=window))
    assert not ok
    assert report.startswith("gate-nvfp4-topk:")
    assert f"sliding_window={window}" in report


def test_gate_nvfp4_topk_check_reads_nested_text_config(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(nested=True, sliding_window=130))
    assert not ok
    assert report.startswith("gate-nvfp4-topk:")


def test_gate_nvfp4_absent_window_does_not_close(gate) -> None:
    """Configs without sliding_window keep the pre-existing gate behaviour."""
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    gate.set_wheel("api")
    ok, _ = gate.module.fused_gate(_config())
    assert ok


def test_gate_nvfp4_topk_check_is_opt_in(gate) -> None:
    """Without the switch an odd window must not close the fused path."""
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config(sliding_window=130))
    assert ok
    assert "compressed_kv=mxfp8_528B" in report


def test_packed_kv_cache_dtype_follows_nvfp4_switch(gate) -> None:
    cls = gate.module.PPUDeepseekV4MegaAttnAttention
    instance = cls()
    assert instance.packed_kv_cache_dtype == "fp8_ds_mla"
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    assert instance.packed_kv_cache_dtype == "nvfp4_ds_mla"
    gate.monkeypatch.delenv("VLLM_SAIL_PPU_KV_NVFP4")
    gate.monkeypatch.setenv("VLLM_PPU_PPU_KV_NVFP4", "1")
    assert instance.packed_kv_cache_dtype == "nvfp4_ds_mla"
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "0")
    assert instance.packed_kv_cache_dtype == "fp8_ds_mla"


def test_nvfp4_opt_in_warns_about_accuracy_and_block_stride(gate) -> None:
    cls = gate.module.PPUDeepseekV4MegaAttnAttention
    cls()
    assert gate.warnings == []
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    cls()
    assert len(gate.warnings) == 1
    msg = gate.warnings[0][0]
    assert "288B" in msg and "528B" in msg
    assert "lcm(528,288)=3168" in msg
    assert "accuracy" in msg


def test_backend_cls_unchanged_by_nvfp4_switch(gate) -> None:
    """The record switch must not swap the backend (metadata/KV geometry)."""
    cls = gate.module.PPUDeepseekV4MegaAttnAttention
    assert cls.backend_cls is gate.module.DeepseekV4FlashMLABackend
    gate.monkeypatch.setenv("VLLM_SAIL_PPU_KV_NVFP4", "1")
    assert cls.backend_cls is gate.module.DeepseekV4FlashMLABackend


def test_decode_forensics_and_record_guard_are_wired() -> None:
    """Source-level regression: the ledger must show the live extra record.

    The decode body needs torch, so assert the wiring statically instead of
    executing it (same pattern as the int32 cast regression below).
    """
    import ast

    from tests.support.source import ROOT

    source = (ROOT / "vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "PPUDeepseekV4MegaAttnAttention"
    )
    decode = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_decode_mega"
    )
    body = ast.get_source_segment(source, decode) or ""
    assert "extra_kv_bytes=%d" in body, (
        "decode lost the extra_kv_bytes forensic field; the run ledger can no "
        "longer prove whether the 288B NVFP4 or the 528B MXFP8 record was live"
    )
    assert "VLLM_SAIL_PPU_KV_NVFP4" in body, (
        "decode lost the compressed-record consistency guard"
    )
    prop = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "packed_kv_cache_dtype"
    )
    prop_body = ast.get_source_segment(source, prop) or ""
    assert '"nvfp4_ds_mla"' in prop_body and '"fp8_ds_mla"' in prop_body
    assert "VLLM_SAIL_PPU_KV_NVFP4" in prop_body


def test_ppu_class_contract(gate) -> None:
    cls = gate.module.PPUDeepseekV4MegaAttnAttention
    # backend stays FLASHMLA_SPARSE_DSV41, KV record stays fp8_ds_mla.
    assert cls.backend_cls is gate.module.DeepseekV4FlashMLABackend
    instance = cls()
    assert instance.packed_kv_cache_dtype == "fp8_ds_mla"
    # v1.1 pins: per-32 packed UE8M0 recipe, no TMA-aligned scales.
    assert instance._einsum_recipe == (1, 1, 32)
    assert instance._tma_aligned_scales is False
    assert gate.module._PPU_EINSUM_RECIPE == (1, 1, 32)
    # is_available_for mirrors fused_gate.
    gate.set_wheel("api")
    assert cls.is_available_for(_config()) is True
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "0")
    assert cls.is_available_for(_config()) is False


# ---------------------------------------------------------------------------
# dsv41_kv_record patch module
# ---------------------------------------------------------------------------


@pytest.fixture
def kv_record(modules, patch_utils_module, monkeypatch):
    logs = []
    logger = SimpleNamespace(info=lambda msg, *a: logs.append((msg, a)))
    modules("vllm.logger", init_logger=lambda name: logger)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return "upstream-record"

    attention_stub = modules(
        "vllm.models.deepseek_v41.attention",
        _use_v41_mxfp8_kv_record=upstream,
    )
    platform = SimpleNamespace(is_ppu=lambda: True)
    modules("vllm.platforms", current_platform=platform)
    module = load_patch("vllm_sail/patch/enhancement/attention/dsv41_kv_record.py")
    return SimpleNamespace(
        module=module,
        patched=module._use_v41_mxfp8_kv_record,
        attention_stub=attention_stub,
        platform=platform,
        logs=logs,
        upstream_calls=upstream_calls,
        registry=patch_utils_module.PATCH_REGISTRY,
        marker=patch_utils_module.PATCH_MARKER,
        monkeypatch=monkeypatch,
    )


def test_kv_record_non_ppu_delegates_upstream(kv_record) -> None:
    kv_record.platform.is_ppu = lambda: False
    assert kv_record.patched() == "upstream-record"
    assert kv_record.upstream_calls == [1]
    assert kv_record.logs == []


def test_kv_record_ppu_default_false_with_forensic_log(kv_record) -> None:
    assert kv_record.patched() is False
    assert kv_record.upstream_calls == []
    assert kv_record.logs == [
        ("[VLLM_SAIL] kv_mxfp8=%s kv_bytes_per_token=%d", (False, 584))
    ]
    # Cached: a second call neither re-reads nor re-logs.
    assert kv_record.patched() is False
    assert len(kv_record.logs) == 1


def test_kv_record_opt_in_selects_528(kv_record) -> None:
    kv_record.monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "1")
    assert kv_record.patched() is True
    assert kv_record.logs == [
        ("[VLLM_SAIL] kv_mxfp8=%s kv_bytes_per_token=%d", (True, 528))
    ]


def test_kv_record_env_read_once_per_process(kv_record) -> None:
    assert kv_record.patched() is False
    kv_record.monkeypatch.setenv("VLLM_SAIL_PPU_KV_MXFP8", "1")
    assert kv_record.patched() is False


def test_kv_record_patch_metadata(kv_record) -> None:
    target = "vllm.models.deepseek_v41.attention._use_v41_mxfp8_kv_record"
    # The module attribute was replaced on the (stubbed) upstream module.
    assert kv_record.attention_stub._use_v41_mxfp8_kv_record is kv_record.patched
    # The replacement remembers the original for drift tooling.
    originals = getattr(kv_record.patched, kv_record.marker, None)
    assert originals is not None and target in originals
    assert callable(originals[target])
    records = [r for r in kv_record.registry if r.target == target]
    assert records, "dsv41_kv_record did not register a PatchRecord"
    record = records[-1]
    assert record.kind == "module"
    assert record.affected_versions == SUPPORTED_RANGE, (
        "dsv41_kv_record must declare the repo's supported vLLM range "
        f"({SUPPORTED_RANGE}), got {record.affected_versions!r}"
    )
    assert record.remove_when.strip()
    assert record.reason.strip()


# ---------------------------------------------------------------------------
# _select_dsv4_attn_cls selector patch
# ---------------------------------------------------------------------------


@pytest.fixture
def selector(modules, patch_utils_module):
    logs = []
    logger = SimpleNamespace(info_once=lambda msg, *a: logs.append((msg, a)))
    modules("vllm.logger", init_logger=lambda name: logger)
    upstream_flashmla = type("UpstreamFlashMLAAttention", (), {})
    modules(
        "vllm.models.deepseek_v41.nvidia.flashmla",
        DeepseekV4FlashMLAAttention=upstream_flashmla,
    )
    state = {"selected": upstream_flashmla, "calls": []}

    def original(vllm_config):
        state["calls"].append(vllm_config)
        return state["selected"]

    def original_scale_name(vllm_config, expert_dtype):
        return "upstream-scale-name"

    model_stub = modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _select_dsv4_attn_cls=original,
        _linear_scale_param_name=original_scale_name,
    )
    platform = SimpleNamespace(is_ppu=lambda: True)
    modules("vllm.platforms", current_platform=platform)
    baseline = type("DeepseekV41FlashMLAAttention", (), {})
    modules(
        "vllm_sail.models.deepseek_v41.flashmla",
        DeepseekV41FlashMLAAttention=baseline,
    )
    upstream_mega = type("DeepseekV4MegaAttnAttention", (), {})
    modules(
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
        DeepseekV4MegaAttnAttention=upstream_mega,
    )
    ppu_fused = type("PPUDeepseekV4MegaAttnAttention", (), {})
    gate_state = {"result": (True, "all gates open: test"), "calls": []}

    def fake_gate(vllm_config):
        gate_state["calls"].append(vllm_config)
        return gate_state["result"]

    modules(
        "vllm_sail.models.deepseek_v41.flash_mla_mega_attn",
        PPUDeepseekV4MegaAttnAttention=ppu_fused,
        fused_gate=fake_gate,
    )
    module = load_patch("vllm_sail/patch/enhancement/models/deepseek_v41.py")
    return SimpleNamespace(
        select=module._select_dsv4_attn_cls,
        model_stub=model_stub,
        platform=platform,
        logs=logs,
        state=state,
        gate_state=gate_state,
        registry=patch_utils_module.PATCH_REGISTRY,
        marker=patch_utils_module.PATCH_MARKER,
        classes=SimpleNamespace(
            upstream_flashmla=upstream_flashmla,
            upstream_mega=upstream_mega,
            baseline=baseline,
            ppu_fused=ppu_fused,
        ),
    )


def test_selector_non_ppu_passthrough(selector) -> None:
    selector.platform.is_ppu = lambda: False
    other = type("Other", (), {})
    selector.state["selected"] = other
    config = object()
    assert selector.select(config) is other
    assert selector.state["calls"] == [config]
    assert selector.gate_state["calls"] == []


def test_selector_ppu_gate_open_returns_fused_class(selector) -> None:
    config = object()
    assert selector.select(config) is selector.classes.ppu_fused
    assert selector.gate_state["calls"] == [config]
    assert any("PPUDeepseekV4MegaAttnAttention" in msg for msg, _ in selector.logs)


def test_selector_ppu_gate_closed_falls_back_to_baseline(selector) -> None:
    selector.gate_state["result"] = (False, "gate-cap: capability (9,0)")
    assert selector.select(object()) is selector.classes.baseline
    assert any("gate closed" in msg for msg, _ in selector.logs)


def test_selector_upstream_mega_selection_also_gated(selector) -> None:
    selector.state["selected"] = selector.classes.upstream_mega
    assert selector.select(object()) is selector.classes.ppu_fused
    selector.gate_state["result"] = (False, "gate-wheel: flash_mla not installed")
    assert selector.select(object()) is selector.classes.baseline


def test_selector_baseline_selection_kept_when_gate_closed(selector) -> None:
    selector.state["selected"] = selector.classes.baseline
    selector.gate_state["result"] = (False, "gate-env: off")
    assert selector.select(object()) is selector.classes.baseline
    assert len(selector.gate_state["calls"]) == 1


def test_selector_foreign_class_never_gated(selector) -> None:
    other = type("FlashInferAttention", (), {})
    selector.state["selected"] = other
    assert selector.select(object()) is other
    assert selector.gate_state["calls"] == []


@pytest.mark.upstream_source
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize(
    "backend", [None, "FLASHMLA_MEGA_ATTN_DSV41", "FLASHMLA_SPARSE"]
)
def test_actual_selector_lora_fallback_is_not_promoted_back_to_fused(
    gate, selector, monkeypatch, upstream_source_root, lora, backend
):
    from tests.support.source import function

    names = (
        "FLASHINFER_MLA_SPARSE",
        "FLASHINFER_MLA_SPARSE_SM120",
        "FLASHINFER_MLA_SPARSE_DSV4",
        "FLASHINFER_MLA_SPARSE_DSV41",
        "FLASHMLA_MEGA_ATTN_DSV41",
        "FLASHMLA_SPARSE",
        "FLASHMLA_SPARSE_DSV4",
        "FLASHMLA_SPARSE_DSV41",
    )
    enum = SimpleNamespace(**{name: object() for name in names})
    monkeypatch.setattr(
        selector.classes.upstream_mega,
        "is_available_for",
        lambda _: True,
        raising=False,
    )
    original = function(
        upstream_source_root / "vllm/models/deepseek_v41/nvidia/model.py",
        "_select_dsv4_attn_cls",
        {
            "AttentionBackendEnum": enum,
            "current_platform": gate.platform,
            "DeepseekV4FlashMLAAttention": selector.classes.upstream_flashmla,
            "DeepseekV4MegaAttnAttention": selector.classes.upstream_mega,
        },
    )
    target = "vllm.models.deepseek_v41.nvidia.model._select_dsv4_attn_cls"
    monkeypatch.setitem(getattr(selector.select, selector.marker), target, original)
    provider = sys.modules["vllm_sail.models.deepseek_v41.flash_mla_mega_attn"]
    monkeypatch.setattr(provider, "fused_gate", gate.module.fused_gate)
    gate.set_wheel("api")
    config = _config()
    config.lora_config = object() if lora else None
    config.attention_config = SimpleNamespace(
        backend=getattr(enum, backend) if backend else None
    )
    config.parallel_config.prefill_context_parallel_size = 1
    if lora:
        assert original(config) is selector.classes.upstream_flashmla
    assert selector.select(config) is (
        selector.classes.baseline if lora else selector.classes.ppu_fused
    )
    if lora:
        assert any("gate-lora:" in args[0] for _, args in selector.logs)


def test_selector_patch_metadata(selector) -> None:
    target = "vllm.models.deepseek_v41.nvidia.model._select_dsv4_attn_cls"
    assert selector.model_stub._select_dsv4_attn_cls is selector.select
    originals = getattr(selector.select, selector.marker, None)
    assert originals is not None and target in originals
    records = [r for r in selector.registry if r.target == target]
    assert records, "selector patch did not register a PatchRecord"
    assert records[-1].affected_versions == SUPPORTED_RANGE


# ---------------------------------------------------------------------------
# int32 positions cast regression (runtime bug found on lingjun-08, 2026-11-12)
# ---------------------------------------------------------------------------


def test_forward_mega_methods_cast_positions_to_int32() -> None:
    """vllm main (8b10bda) forwards the runner's int64 positions unchanged.

    The PPU fused kernel enforces int32 (``KU_CHECK_DTYPE(token_positions,
    torch::kInt32)``), so both PPU forward methods must defensively cast
    before the kernel call; without it the server dies in warmup with
    ``RuntimeError: token_positions must have dtype torch::kInt32``.
    """
    import ast

    from tests.support.source import ROOT

    source = (ROOT / "vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "PPUDeepseekV4MegaAttnAttention"
    )
    for name in ("_forward_decode_mega", "_forward_prefill_mega"):
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        body_source = ast.get_source_segment(source, method) or ""
        assert "positions_int32.to(torch.int32)" in body_source, (
            f"{name} lost the defensive int32 cast for token_positions; the "
            "PPU fused kernel rejects int64 positions (seen on lingjun-08)."
        )


# ---------------------------------------------------------------------------
# PR#59 per-token FP32 output scale arm (FlashMLA-for-sail 7bce34f)
# ---------------------------------------------------------------------------


def _method_source(name: str) -> str:
    """Source of a PPU mega-attention method, for torch-free wiring checks."""
    import ast

    from tests.support.source import ROOT

    source = (ROOT / "vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "PPUDeepseekV4MegaAttnAttention"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, method) or ""


def test_per_token_scale_switch_defaults_off() -> None:
    """Opt-in contract: the default arm stays byte-identical to PR#54."""
    assert envs.VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE is False
    assert envs.snapshot()["VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE"] is False


def test_per_token_scale_group_is_the_wv_row_width(gate) -> None:
    """PR#59 selects the per-token format with num_per_channels == 8 * 512."""
    assert gate.module._PER_TOKEN_SCALE_GROUP == 8 * 512 == 4096


def test_output_scale_args_default_arm_is_the_pr54_contract(gate) -> None:
    assert gate.module._output_scale_args() == {
        "num_per_channels": 32,
        "use_tma_aligned_col_major_sf": True,
        "round_sf": True,
        "use_packed_ue8m0": True,
    }


def test_output_scale_args_per_token_arm_clears_every_flag(gate) -> None:
    """The kernel rejects mixed combinations, so all three flags must flip."""
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", "1")
    assert gate.module._output_scale_args() == {
        "num_per_channels": 4096,
        "use_tma_aligned_col_major_sf": False,
        "round_sf": False,
        "use_packed_ue8m0": False,
    }


def test_output_scale_args_are_read_live_not_cached(gate) -> None:
    """A/B flips only the env var: one code tree, one wheel, both arms."""
    assert gate.module._output_scale_args()["num_per_channels"] == 32
    gate.monkeypatch.setenv("VLLM_PPU_FUSED_ATTN_PER_TOKEN_SCALE", "1")
    assert gate.module._output_scale_args()["num_per_channels"] == 4096
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", "0")
    assert gate.module._output_scale_args()["num_per_channels"] == 32


def test_gate_closes_per_token_arm_on_sm80(gate) -> None:
    gate.set_wheel("api")
    gate.set_capability(SimpleNamespace(major=8, minor=0))
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", "1")
    ok, report = gate.module.fused_gate(_config())
    assert ok is False
    assert report.startswith("gate-per-token-cap:")
    assert "8.9" in report


def test_gate_keeps_per_token_arm_open_on_sm89(gate) -> None:
    gate.set_wheel("api")
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", "1")
    ok, report = gate.module.fused_gate(_config())
    assert ok is True
    assert report.startswith("all gates open: cap=8.9 ")
    assert "output_scale=per_token_fp32" in report


def test_gate_open_report_names_the_default_arm(gate) -> None:
    gate.set_wheel("api")
    ok, report = gate.module.fused_gate(_config())
    assert ok is True
    assert "output_scale=per32_ue8m0" in report


def test_per_token_switch_alone_does_not_enable_the_fused_path(gate) -> None:
    """The master switch still governs; the scale switch only picks the arm."""
    gate.set_wheel("api")
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", "0")
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", "1")
    ok, report = gate.module.fused_gate(_config())
    assert ok is False
    assert report.startswith("gate-env:")


def test_alloc_attn_out_is_arm_aware() -> None:
    """The per-token arm needs its own float32 buffer, not upstream's int32."""
    body = _method_source("_alloc_attn_out")
    for needle in (
        "alloc_mega_attn_output",
        "_per_token_scale()",
        "torch.float32",
        "kFp8DynamicTokenSym",
        "replace(",
    ):
        assert needle in body, f"_alloc_attn_out lost {needle!r}"


def test_o_proj_asserts_the_selected_scale_format() -> None:
    body = _method_source("_o_proj")
    for needle in ("_per_token_scale()", "torch.float32", "torch.int32"):
        assert needle in body, f"_o_proj lost the arm-aware check {needle!r}"


def test_both_call_sites_share_the_output_scale_args() -> None:
    """No call site may hardcode one arm's flags independently."""
    for name in ("_forward_decode_mega", "_forward_prefill_mega"):
        body = _method_source(name)
        assert "**_output_scale_args()" in body, (
            f"{name} no longer takes the shared PR#59 output-format arguments"
        )
        assert "num_per_channels=" not in body, (
            f"{name} hardcodes num_per_channels; the two arms must stay "
            "selectable from one place"
        )
        assert "_call_fused(" in body, f"{name} bypasses the wheel-error wrapper"


def test_call_fused_names_the_wheel_requirement_on_format_rejection(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", "1")

    def reject(**kwargs):
        raise RuntimeError("num_per_channels == 32")

    gate.monkeypatch.setattr(gate.module, "_API", SimpleNamespace(decode=reject))
    with pytest.raises(RuntimeError, match="PR#59"):
        gate.module._call_fused("decode", num_per_channels=4096)


def test_call_fused_does_not_relabel_default_arm_errors(gate) -> None:
    def reject(**kwargs):
        raise RuntimeError("num_per_channels == 32")

    gate.monkeypatch.setattr(gate.module, "_API", SimpleNamespace(decode=reject))
    with pytest.raises(RuntimeError) as exc:
        gate.module._call_fused("decode", num_per_channels=32)
    assert "PR#59" not in str(exc.value)


def test_call_fused_reraises_unrelated_errors_unchanged(gate) -> None:
    gate.monkeypatch.setenv("VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", "1")

    def boom(**kwargs):
        raise RuntimeError("token_positions must have dtype torch::kInt32")

    gate.monkeypatch.setattr(gate.module, "_API", SimpleNamespace(decode=boom))
    with pytest.raises(RuntimeError, match="token_positions"):
        gate.module._call_fused("decode")


# ---------------------------------------------------------------------------
# per-channel wo_a scales: shape-aware load permutation + per-channel dequant
# ---------------------------------------------------------------------------
#
# An FP8 weight-per-channel checkpoint (``fp8_channelwise_layers``) stores one
# scale per output row, so ``wo_a.weight_scale`` has a size-1 K axis while
# upstream unconditionally registers the 128-entry per-32 chunk permutation on
# it; loading aborted with ``output with shape [rows, 1] doesn't match the
# broadcast shape [rows, 128]``. The patch makes ``permute_on_load`` skip the
# permutation for that geometry and keeps every other geometry delegated.
#
# The dequantization recipe itself needs torch, so it is pinned statically
# here (same pattern as the int32-cast regression above) and verified on device.


@pytest.fixture
def permute_patch(modules, patch_utils_module):
    """Load the shape-aware ``permute_on_load`` patch against a stub target."""
    logs = []
    logger = SimpleNamespace(info_once=lambda msg, *a: logs.append((msg, a)))
    modules("vllm.logger", init_logger=lambda name: logger)
    calls = []

    def permute_on_load(param, perm, dim):
        calls.append((param, perm, dim))

    modules(
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
        permute_on_load=permute_on_load,
    )
    module = load_patch("vllm_sail/patch/enhancement/deepseek_v41_fused_layout.py")
    return SimpleNamespace(
        module=module,
        patched=module.permute_on_load,
        original=permute_on_load,
        calls=calls,
        logs=logs,
        registry=patch_utils_module.PATCH_REGISTRY,
        marker=patch_utils_module.PATCH_MARKER,
    )


@pytest.mark.parametrize(("axis", "perm_length"), [(4096, 4096), (128, 128), (1, 1)])
def test_scale_permute_action_applies_to_one_entry_per_element(
    permute_patch, axis, perm_length
) -> None:
    """Canonical geometry is what the permutation was written for."""
    assert permute_patch.module.scale_permute_action(axis, perm_length) == "apply"


@pytest.mark.parametrize(("axis", "perm_length"), [(1, 128), (1, 4096)])
def test_scale_permute_action_skips_per_channel_scale(
    permute_patch, axis, perm_length
) -> None:
    """A per-row scale is constant along K, so the K-chunk move is identity."""
    assert permute_patch.module.scale_permute_action(axis, perm_length) == "skip"


@pytest.mark.parametrize(("axis", "perm_length"), [(128, 4096), (4096, 128), (0, 128)])
def test_scale_permute_action_rejects_unknown_geometry(
    permute_patch, axis, perm_length
) -> None:
    """Anything else means caller and permutation disagree: never guess."""
    with pytest.raises(AssertionError, match="geometry mismatch"):
        permute_patch.module.scale_permute_action(axis, perm_length)


def test_permute_on_load_skips_per_channel_scale(permute_patch) -> None:
    param = SimpleNamespace(shape=(1024, 1))
    perm = SimpleNamespace(numel=lambda: 128)
    permute_patch.patched(param, perm, 1)
    assert permute_patch.calls == []
    assert permute_patch.logs, "the skip must leave one forensic line"
    assert "skipping the 32-chunk K permutation" in permute_patch.logs[-1][0]


def test_permute_on_load_delegates_canonical_geometry(permute_patch) -> None:
    param = SimpleNamespace(shape=(1024, 4096))
    perm = SimpleNamespace(numel=lambda: 4096)
    permute_patch.patched(param, perm, 1)
    assert permute_patch.calls == [(param, perm, 1)]
    assert permute_patch.logs == []


def test_permute_on_load_rejects_geometry_mismatch(permute_patch) -> None:
    param = SimpleNamespace(shape=(1024, 64))
    perm = SimpleNamespace(numel=lambda: 128)
    with pytest.raises(AssertionError, match="geometry mismatch"):
        permute_patch.patched(param, perm, 1)
    assert permute_patch.calls == []


def test_fused_layout_patch_metadata(permute_patch) -> None:
    import sys

    target = "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn.permute_on_load"
    stub = sys.modules["vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn"]
    # The replacement is installed on the module upstream reads at call time.
    assert stub.permute_on_load is permute_patch.patched
    originals = getattr(permute_patch.patched, permute_patch.marker, None)
    assert originals is not None and target in originals
    assert originals[target] is permute_patch.original
    records = [r for r in permute_patch.registry if r.target == target]
    assert records, "the loader patch did not register a PatchRecord"
    record = records[-1]
    assert record.kind == "module"
    assert record.affected_versions == SUPPORTED_RANGE
    assert record.remove_when.strip() and record.reason.strip()


def test_perchannel_loader_fix_source_is_pinned() -> None:
    """Source-level guards for the pieces the offline tier cannot execute."""
    from tests.support.source import ROOT

    patch_source = (
        ROOT / "vllm_sail/patch/enhancement/deepseek_v41_fused_layout.py"
    ).read_text()
    # Patch the *consumer* module: upstream holds a by-value alias.
    assert "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn" in patch_source
    assert '"permute_on_load"' in patch_source
    assert 'return "skip"' in patch_source
    # The new module is installed with the rest of the enhancement category.
    init_source = (ROOT / "vllm_sail/patch/enhancement/__init__.py").read_text()
    assert "deepseek_v41_fused_layout" in init_source

    model_source = (
        ROOT / "vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py"
    ).read_text()
    # Definition + the single call site in the 1-byte branch.
    assert model_source.count("_dequant_fp8_per_channel(") == 2
    assert "_SCALE_PER_CHANNEL" in model_source
    assert "_SCALE_MXFP8_BLOCK" in model_source
    assert "wo_a_scale_mode=%s scale_permute=%s" in model_source
    # The block recipe stays reachable and is still the delegated default.
    assert "dequant_mxfp8_to_bf16(weight, weight_scale)" in model_source


def test_o_proj_selects_the_dequant_recipe_by_scale_layout() -> None:
    """The 1-byte wo_a branch must dispatch on scale_mode, not on dtype alone.

    Broadcasting the MXFP8 block helper over a per-channel ``[rows, 1]`` FP32
    scale produces finite-looking garbage instead of failing, so the dispatch
    and its loud else-branch are pinned statically.
    """
    import ast

    from tests.support.source import ROOT

    source = (ROOT / "vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "PPUDeepseekV4MegaAttnAttention"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_o_proj"
    )
    body = ast.get_source_segment(source, method) or ""
    assert "scale_mode = (" in body, "_o_proj must classify the scale layout"
    assert "_wo_a_scale_mode(weight, weight_scale)" in body
    assert "if scale_mode == _SCALE_PER_CHANNEL:" in body
    assert "elif scale_mode == _SCALE_MXFP8_BLOCK:" in body
    assert "needs a quantized scale layout" in body, (
        "an unrecognised scale layout must raise, not fall through to a "
        "broadcasting helper"
    )
    # The classifier itself must reject unknown layouts rather than guess.
    classifier = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_wo_a_scale_mode"
    )
    classifier_body = ast.get_source_segment(source, classifier) or ""
    assert "unrecognised wo_a weight_scale layout" in classifier_body
    assert "scale.numel() == rows" in classifier_body


# ---------------------------------------------------------------------------
# Optional numerical tier: real torch, still no vLLM and no device.
# ---------------------------------------------------------------------------
#
# Everything above pins wiring statically. The two helpers that actually
# compute numbers are executed here against real torch when it is installed,
# and skipped explicitly when it is not -- same contract as
# tests/ut/test_portable_weight_packing.py, so the CPU-only merge gate keeps
# passing without torch.
#
# Exactness note: e4m3 carries 3 mantissa bits, so the integers 1..16 are
# exact in it, and bf16 carries 8, so those integers times a power of two are
# exact in bf16. Every expected value below is therefore representable and the
# comparisons use torch.equal rather than a tolerance.


@pytest.fixture
def numerics(modules, monkeypatch):
    """Load the real model module with real torch and stubbed vLLM imports."""
    torch = pytest.importorskip("torch", reason="numerical tier requires torch")
    logs: list = []
    logger = SimpleNamespace(
        info_once=lambda msg, *a: logs.append((msg, a)),
        warning_once=lambda msg, *a: logs.append((msg, a)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    modules("vllm.logger", init_logger=lambda name: logger)
    modules(
        "vllm.model_executor.layers.fusion.quant_activation",
        QuantizedActivation=type("QuantizedActivation", (), {}),
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        dequant_mxfp8_to_bf16=lambda weight, scale: weight,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        kFp8DynamicTokenSym="kFp8DynamicTokenSym",
    )
    modules(
        "vllm.models.deepseek_v41.common.ops",
        combine_topk_swa_indices=lambda *a, **k: None,
        dequantize_and_gather_k_cache=lambda *a, **k: None,
    )
    modules("vllm.models.deepseek_v41.common.ops.fused_layout", WV_GROUP_SIZE=8)
    modules(
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
        _HEAD_DIM_V=512,
        _QUANT_GROUP=32,
        _ROPE_DIM=64,
        _token_slice=lambda out, start, end: out,
        alloc_mega_attn_output=lambda num_tokens, n_wv_group, device: SimpleNamespace(
            num_tokens=num_tokens, n_wv_group=n_wv_group, device=device
        ),
        DeepseekV4MegaAttnAttention=type("DeepseekV4MegaAttnAttention", (), {}),
    )
    modules(
        "vllm.models.deepseek_v41.sparse_mla",
        DeepseekV4FlashMLABackend=type("DeepseekV4FlashMLABackend", (), {}),
    )
    modules("vllm.utils.math_utils", round_up=lambda x, y: -(-x // y) * y)
    modules("vllm.v1.worker.workspace", current_workspace_manager=lambda: None)
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: True,
            get_device_capability=lambda device_id=0: SimpleNamespace(major=8, minor=9),
        ),
    )
    module = load_patch("vllm_sail/models/deepseek_v41/flash_mla_mega_attn.py")
    return SimpleNamespace(module=module, torch=torch, logs=logs)


def _fp8_weight(torch, rows: int, cols: int):
    """Integers 1..16 in e4m3: every value is exactly representable."""
    values = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return (values % 16 + 1).to(torch.float8_e4m3fn)


def _row_scale(torch, rows: int, spell: str):
    """Distinct powers of two per row, as ``[rows]`` or ``[rows, 1]``."""
    scale = torch.tensor(
        [2.0 ** ((i % 5) - 2) for i in range(rows)], dtype=torch.float32
    )
    return scale if spell == "flat" else scale.reshape(rows, 1)


def _row_loop_reference(torch, weight, scale):
    """Independent reference: one Python-level row at a time, no broadcasting."""
    flat = scale.reshape(-1)
    return torch.stack(
        [
            (weight[i].to(torch.float32) * float(flat[i])).to(torch.bfloat16)
            for i in range(weight.shape[0])
        ]
    )


@pytest.mark.requires_runtime
def test_per_channel_dequant_scales_each_row_by_its_own_multiplier(numerics) -> None:
    torch = numerics.torch
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    out = numerics.module._dequant_fp8_per_channel(weight, scale)
    assert out.dtype == torch.bfloat16
    assert out.shape == weight.shape
    expected = torch.tensor([[0.5, 1.0], [6.0, 8.0]], dtype=torch.bfloat16)
    assert torch.equal(out, expected)


@pytest.mark.requires_runtime
@pytest.mark.parametrize("spell", ["flat", "column"])
@pytest.mark.parametrize(("rows", "cols"), [(4, 128), (256, 512), (64, 4096)])
def test_per_channel_dequant_matches_an_independent_row_loop(
    numerics, spell, rows, cols
) -> None:
    """The reshape-and-broadcast recipe must equal a per-row Python loop."""
    torch = numerics.torch
    weight = _fp8_weight(torch, rows, cols)
    scale = _row_scale(torch, rows, spell)
    out = numerics.module._dequant_fp8_per_channel(weight, scale)
    assert out.dtype == torch.bfloat16
    assert out.shape == (rows, cols)
    assert torch.equal(out, _row_loop_reference(torch, weight, scale))


@pytest.mark.requires_runtime
def test_flat_and_column_scale_spellings_dequantize_identically(numerics) -> None:
    torch = numerics.torch
    weight = _fp8_weight(torch, 8, 128)
    flat = numerics.module._dequant_fp8_per_channel(
        weight, _row_scale(torch, 8, "flat")
    )
    column = numerics.module._dequant_fp8_per_channel(
        weight, _row_scale(torch, 8, "column")
    )
    assert torch.equal(flat, column)


@pytest.mark.requires_runtime
def test_per_channel_dequant_does_not_reuse_the_first_row_scale(numerics) -> None:
    """Guards the exact broadcasting bug the block helper would have produced."""
    torch = numerics.torch
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    out = numerics.module._dequant_fp8_per_channel(weight, scale)
    first_row_only = (weight.to(torch.float32) * scale[0]).to(torch.bfloat16)
    assert not torch.equal(out, first_row_only)


@pytest.mark.requires_runtime
def test_per_channel_dequant_handles_zero_and_negative_scales(numerics) -> None:
    torch = numerics.torch
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.0], [-1.0], [2.0]], dtype=torch.float32)
    out = numerics.module._dequant_fp8_per_channel(weight, scale)
    expected = torch.tensor(
        [[0.0, 0.0], [-3.0, -4.0], [10.0, 12.0]], dtype=torch.bfloat16
    )
    assert torch.equal(out, expected)


@pytest.mark.requires_runtime
def test_per_channel_dequant_fails_loudly_on_a_row_count_mismatch(numerics) -> None:
    """A scale that cannot be one-per-row must raise, not silently broadcast."""
    torch = numerics.torch
    weight = _fp8_weight(torch, 4, 128)
    with pytest.raises(RuntimeError):
        numerics.module._dequant_fp8_per_channel(
            weight, torch.ones(8, dtype=torch.float32)
        )


@pytest.mark.requires_runtime
def test_scale_mode_is_none_without_a_scale(numerics) -> None:
    torch = numerics.torch
    weight = torch.ones((4, 128), dtype=torch.float8_e4m3fn)
    assert numerics.module._wo_a_scale_mode(weight, None) == (
        numerics.module._SCALE_NONE
    )


@pytest.mark.requires_runtime
@pytest.mark.parametrize("spell", ["flat", "column"])
def test_scale_mode_recognises_per_channel_scales(numerics, spell) -> None:
    torch = numerics.torch
    weight = torch.ones((4, 128), dtype=torch.float8_e4m3fn)
    scale = _row_scale(torch, 4, spell)
    assert numerics.module._wo_a_scale_mode(weight, scale) == (
        numerics.module._SCALE_PER_CHANNEL
    )


@pytest.mark.requires_runtime
def test_scale_mode_recognises_the_mxfp8_block_layout(numerics) -> None:
    torch = numerics.torch
    weight = torch.ones((4, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones((4, 4), dtype=torch.uint8)
    assert numerics.module._wo_a_scale_mode(weight, scale) == (
        numerics.module._SCALE_MXFP8_BLOCK
    )


@pytest.mark.requires_runtime
@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        ("float32", (4, 2)),
        ("float32", (8,)),
        ("float32", (4, 4)),
        ("int32", (4, 1)),
        ("uint8", (4, 5)),
        ("bfloat16", (4, 1)),
    ],
)
def test_scale_mode_rejects_unknown_layouts(numerics, dtype, shape) -> None:
    torch = numerics.torch
    weight = torch.ones((4, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones(shape, dtype=getattr(torch, dtype))
    with pytest.raises(AssertionError, match="unrecognised wo_a weight_scale layout"):
        numerics.module._wo_a_scale_mode(weight, scale)


@pytest.mark.requires_runtime
def test_unknown_layout_error_carries_the_forensic_fields(numerics) -> None:
    """The acceptance runbook asks for these three fields verbatim on failure."""
    torch = numerics.torch
    weight = torch.ones((4, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones((4, 2), dtype=torch.float32)
    with pytest.raises(AssertionError) as excinfo:
        numerics.module._wo_a_scale_mode(weight, scale)
    message = str(excinfo.value)
    assert "shape=(4, 2)" in message
    assert "dtype=torch.float32" in message
    assert "weight.shape=(4, 128)" in message


@pytest.mark.requires_runtime
def test_classifier_and_recipe_compose_on_a_per_channel_shard(numerics) -> None:
    """End to end for the two helpers: classify, then dequantize by row."""
    torch = numerics.torch
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    module = numerics.module
    assert module._wo_a_scale_mode(weight, scale) == module._SCALE_PER_CHANNEL
    out = module._dequant_fp8_per_channel(weight, scale)
    expected = torch.tensor([[0.5, 1.0], [6.0, 8.0]], dtype=torch.bfloat16)
    assert torch.equal(out, expected)
