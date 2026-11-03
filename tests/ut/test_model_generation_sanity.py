# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the accuracy gate functions in ``_accuracy_gate``.

Schema v2 replaces per-token logprob goldens with a single accuracy score
from evalscope GSM8K evaluation.  This module tests every gate branch:

1. :func:`check_score_threshold` — primary ``score >= threshold`` gate.
2. :func:`check_accuracy_golden` — three-stage golden regression gate
   (completeness, hard floor, regression lower bound).
3. :func:`build_accuracy_golden` — golden record construction.
4. Naming helpers (``golden_filename``, ``pending_golden_filename``).
5. I/O round-trip (``write_golden``, ``read_golden``).

The module under test imports neither ``torch`` nor ``vllm``, so everything
here runs on a plain CPU interpreter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.e2e.models import _accuracy_gate as gate

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------


def test_default_threshold_is_0_9() -> None:
    assert gate.DEFAULT_THRESHOLD == 0.9


def test_golden_schema_version_is_2() -> None:
    assert gate.GOLDEN_SCHEMA_VERSION == 2


def test_benchmark_constant() -> None:
    assert gate.BENCHMARK_GSM8K == "gsm8k"


def test_gsm8k_full_size() -> None:
    assert gate.GSM8K_FULL_SIZE == 1319


def test_default_tolerance_has_required_keys() -> None:
    t = gate.DEFAULT_TOLERANCE
    assert "mode" in t
    assert "rel" in t
    assert "abs" in t
    assert "hard_floor" in t
    assert t["hard_floor"] == gate.DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# File naming helpers.
# ---------------------------------------------------------------------------


def test_golden_filename_format() -> None:
    assert (
        gate.golden_filename("deepseek_v4_int8", "810e")
        == "accuracy_deepseek_v4_int8_810e.json"
    )


def test_pending_golden_filename_format() -> None:
    name = gate.pending_golden_filename("deepseek_v4_mxfp4", "890p")
    assert name.startswith("accuracy_deepseek_v4_mxfp4_890p")
    assert name.endswith(".pending-review.json")


# ---------------------------------------------------------------------------
# check_score_threshold.
# ---------------------------------------------------------------------------


def test_score_above_threshold_passes() -> None:
    ok, err = gate.check_score_threshold(0.95)
    assert ok is True
    assert err is None


def test_score_exactly_at_threshold_passes() -> None:
    ok, err = gate.check_score_threshold(0.9)
    assert ok is True
    assert err is None


def test_score_below_threshold_fails() -> None:
    ok, err = gate.check_score_threshold(0.89)
    assert ok is False
    assert err is not None
    assert "below threshold" in err


def test_custom_threshold() -> None:
    ok, _ = gate.check_score_threshold(0.8, threshold=0.7)
    assert ok is True
    ok2, err2 = gate.check_score_threshold(0.6, threshold=0.7)
    assert ok2 is False
    assert "0.7" in err2


def test_invalid_scores_are_rejected() -> None:
    for score in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1):
        ok, err = gate.check_score_threshold(score)
        assert ok is False
        assert err == f"invalid accuracy score: {score!r}"


# ---------------------------------------------------------------------------
# build_accuracy_golden — schema v2 record construction.
# ---------------------------------------------------------------------------


def _make_golden(
    score: float = 0.92,
    score_num: int = 1319,
    **overrides: Any,
) -> dict[str, Any]:
    """Helper: build a golden record with sensible defaults."""
    kwargs: dict[str, Any] = {
        "model_key": "demo_model",
        "chip": "890p",
        "checkpoint": "/ckpt/v4/Model-W8A8",
        "dataset_cfg": {
            "dataset_id": "/data/gsm8k",
            "few_shot_num": 4,
            "few_shot_random": False,
            "subset_list": ["main"],
            "num_expected": 1319,
        },
        "serving_cfg": {
            "tensor_parallel_size": 8,
            "max_model_len": 4096,
            "dtype": "bfloat16",
        },
        "score": score,
        "score_num": score_num,
    }
    kwargs.update(overrides)
    return gate.build_accuracy_golden(**kwargs)


def test_build_golden_has_required_fields() -> None:
    record = _make_golden()
    assert record["schema_version"] == 2
    assert record["model"] == "demo_model"
    assert record["chip"] == "890p"
    assert record["benchmark"] == "gsm8k"
    assert record["score"] == 0.92
    assert record["score_num"] == 1319
    assert "measured_at" in record
    assert "tolerance" in record
    assert record["checkpoint_tail"] is not None


def test_build_golden_custom_tolerance() -> None:
    custom = {"mode": "lower_bound", "rel": 0.95, "abs": 0.05, "hard_floor": 0.8}
    record = _make_golden(tolerance=custom)
    assert record["tolerance"] == custom


# ---------------------------------------------------------------------------
# check_accuracy_golden — completeness, hard floor, regression.
# ---------------------------------------------------------------------------


def test_completeness_gate_passes_with_matching_count() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.91, 1319, golden)
    assert ok is True
    assert err is None


def test_completeness_gate_fails_on_count_mismatch() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.92, 1000, golden)
    assert ok is False
    assert "completeness" in err
    assert "1319" in err


def test_completeness_gate_skipped_when_num_expected_absent() -> None:
    """If num_expected is not in the golden dataset, completeness is skipped."""
    golden = _make_golden(score=0.92, score_num=1319)
    golden["dataset"].pop("num_expected", None)
    ok, err = gate.check_accuracy_golden(0.92, 500, golden)
    assert ok is True


def test_hard_floor_gate_fails_below_floor() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.85, 1319, golden)
    assert ok is False
    assert "hard floor" in err


def test_hard_floor_gate_passes_at_floor() -> None:
    golden = _make_golden(score=0.92, score_num=1319)
    ok, err = gate.check_accuracy_golden(0.9, 1319, golden)
    assert ok is True


def test_regression_gate_fails_on_large_drop() -> None:
    golden = _make_golden(score=0.95, score_num=1319)
    # regression_delta = max(0.02, (1-0.99)*0.95) = max(0.02, 0.0095) = 0.02
    # lower_bound = 0.95 - 0.02 = 0.93
    ok, err = gate.check_accuracy_golden(0.92, 1319, golden)
    assert ok is False
    assert "regression" in err


def test_regression_gate_passes_within_tolerance() -> None:
    golden = _make_golden(score=0.95, score_num=1319)
    # lower_bound = 0.95 - 0.02 = 0.93
    ok, err = gate.check_accuracy_golden(0.935, 1319, golden)
    assert ok is True
    assert err is None


def test_regression_gate_uses_custom_tolerance() -> None:
    golden = _make_golden(score=0.95, score_num=1319)
    golden["tolerance"] = {
        "rel": 0.99,
        "abs": 0.01,
        "hard_floor": 0.8,
    }
    # regression_delta = max(0.01, 0.01*0.95) = max(0.01, 0.0095) = 0.01
    # lower_bound = 0.95 - 0.01 = 0.94
    ok, err = gate.check_accuracy_golden(0.935, 1319, golden)
    assert ok is False
    assert "regression" in err


def test_gates_are_evaluated_in_order() -> None:
    """Completeness gate fires before hard floor and regression."""
    golden = _make_golden(score=0.95, score_num=1319)
    # score=0.5 would fail all three; completeness should be reported first.
    ok, err = gate.check_accuracy_golden(0.5, 100, golden)
    assert ok is False
    assert "completeness" in err


# ---------------------------------------------------------------------------
# I/O round-trip: write_golden → read_golden.
# ---------------------------------------------------------------------------


def test_write_and_read_golden_round_trip(tmp_path: Path) -> None:
    record = _make_golden()
    path = tmp_path / "golden" / "test.json"
    gate.write_golden(record, path)
    assert path.exists()
    loaded = gate.read_golden(path)
    assert loaded["schema_version"] == record["schema_version"]
    assert loaded["score"] == record["score"]
    assert loaded["model"] == record["model"]


def test_write_golden_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "dir" / "golden.json"
    gate.write_golden({"test": 1}, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# Integration: the accuracy test module exists and has the right marks.
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[1] / "e2e" / "models"
ACCURACY_TEST = MODELS_DIR / "test_model_accuracy.py"


def test_accuracy_test_file_exists() -> None:
    assert ACCURACY_TEST.exists()


def test_accuracy_test_uses_correct_marks() -> None:
    text = ACCURACY_TEST.read_text()
    assert "pytest.mark.ppu" in text
    assert "pytest.mark.model_e2e" in text
    assert text.count("@pytest.mark.cap89") == 2


def test_accuracy_test_requires_eight_devices() -> None:
    text = ACCURACY_TEST.read_text()
    assert "PPU_CI_MIN_DEVICES = 8" in text


def test_accuracy_test_guards_against_no_device() -> None:
    text = ACCURACY_TEST.read_text()
    assert "importorskip" in text
    assert "is_ppu()" in text


def test_accuracy_test_references_eval_script() -> None:
    text = ACCURACY_TEST.read_text()
    assert "ppu_gsm8k_eval" in text
    assert "run_accuracy_eval" in text


def test_accuracy_test_uses_current_model_keys() -> None:
    text = ACCURACY_TEST.read_text()
    for model_key in (
        "minimax-m3-fp8",
        "qwen3.8-flash-next",
    ):
        assert model_key in text
    for old_key in ("minimax_m3", "kimi_k3", "qwen3_8_27b"):
        assert f'"{old_key}"' not in text


def test_no_other_model_test_files_remain() -> None:
    """Old per-model test files should have been deleted."""
    old_files = [
        "test_model_deepseek_v4.py",
        "test_model_kimi_k3.py",
        "test_model_minimax_m3.py",
        "test_model_qwen3_next.py",
    ]
    for name in old_files:
        assert not (MODELS_DIR / name).exists(), f"old model test {name} still exists"
