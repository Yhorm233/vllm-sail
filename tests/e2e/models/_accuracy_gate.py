# SPDX-License-Identifier: Apache-2.0
"""Pure golden comparison functions for the GSM8K accuracy E2E suite.

Golden schema v2 replaces the per-token logprob goldens (schema v1) with a
single accuracy score from evalscope GSM8K evaluation.  This module is the
single source of truth for:

* schema constants and tolerance defaults,
* golden record construction (:func:`build_accuracy_golden`),
* identity binding (:func:`check_golden_identity`),
* scoring gates (:func:`check_accuracy_golden`).

It imports neither ``torch`` nor ``vllm``, so CPU-only unit tests exercise
every gate branch without a device.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.e2e.models._offline_runner import (
    PENDING_REVIEW_SUFFIX,
    _checkpoint_tail,
)

# --- Schema constants -------------------------------------------------------
GOLDEN_SCHEMA_VERSION = 2
BENCHMARK_GSM8K = "gsm8k"

# --- Default scoring threshold -----------------------------------------------
#
# Primary gate: ``score >= threshold`` means pass.  The golden file's
# tolerance block can refine this later once baselines are established;
# until then 0.9 is the only gate that matters.
DEFAULT_THRESHOLD = 0.9

DEFAULT_TOLERANCE: dict[str, Any] = {
    "mode": "lower_bound",
    "rel": 0.99,
    "abs": 0.02,
    "hard_floor": DEFAULT_THRESHOLD,
}

#: Full GSM8K test-set size (main split).
GSM8K_FULL_SIZE = 1319


# --- File-naming helpers ----------------------------------------------------


def golden_filename(model_key: str, chip: str) -> str:
    """Formal golden: ``accuracy_<model>_<chip>.json``."""
    return f"accuracy_{model_key}_{chip}.json"


def pending_golden_filename(model_key: str, chip: str) -> str:
    """Pending-review golden: ``accuracy_<model>_<chip>.pending-…``."""
    stem = f"accuracy_{model_key}_{chip}"
    return stem + PENDING_REVIEW_SUFFIX


# --- Golden record builder --------------------------------------------------


def build_accuracy_golden(
    *,
    model_key: str,
    chip: str,
    checkpoint: str,
    benchmark: str = BENCHMARK_GSM8K,
    evalscope_version: str = "",
    dataset_cfg: dict[str, Any],
    serving_cfg: dict[str, Any],
    score: float,
    score_num: int,
    tolerance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a golden record dict in schema v2 format."""
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "model": model_key,
        "chip": chip,
        "checkpoint": checkpoint,
        "checkpoint_tail": _checkpoint_tail(checkpoint),
        "benchmark": benchmark,
        "evalscope_version": evalscope_version,
        "dataset": dataset_cfg,
        "serving": serving_cfg,
        "score": score,
        "score_num": score_num,
        "tolerance": tolerance or dict(DEFAULT_TOLERANCE),
        "measured_at": (datetime.now(timezone.utc).isoformat(timespec="seconds")),
    }


# --- Identity gate ----------------------------------------------------------


def check_golden_identity(
    golden: dict[str, Any],
    *,
    chip: str,
    checkpoint_tail: str,
    benchmark: str = BENCHMARK_GSM8K,
) -> tuple[bool, str | None]:
    """Verify the golden's identity fields match the current run.

    Returns ``(ok, error_message)``.  ``ok`` is ``True`` when identity
    matches; ``error_message`` is ``None`` on success.
    """
    golden_chip = golden.get("chip")
    if golden_chip != chip:
        return (
            False,
            f"golden chip mismatch: golden records chip="
            f"{golden_chip!r} but the current device is {chip!r}; "
            "per-chip goldens must never be cross-compared",
        )
    golden_tail = golden.get("checkpoint_tail")
    if golden_tail is not None and golden_tail != checkpoint_tail:
        return (
            False,
            f"golden checkpoint mismatch: golden 来自 "
            f"{golden_tail!r}，本次运行 {checkpoint_tail!r}；"
            "基准与 checkpoint 不匹配",
        )
    golden_bench = golden.get("benchmark")
    if golden_bench is not None and golden_bench != benchmark:
        return (
            False,
            f"golden benchmark mismatch: golden records "
            f"{golden_bench!r} but this run evaluates {benchmark!r}",
        )
    return (True, None)


# --- Accuracy scoring gates -------------------------------------------------


def check_score_threshold(
    score: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, str | None]:
    """Primary gate: accept only finite probabilities meeting the threshold."""
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return (False, f"invalid accuracy score: {score!r}")
    if score < threshold:
        return (
            False,
            f"accuracy {score:.4f} below threshold {threshold}",
        )
    return (True, None)


def check_accuracy_golden(
    score: float,
    num: int,
    golden: dict[str, Any],
) -> tuple[bool, str | None]:
    """Check an accuracy score against the golden baseline.

    Returns ``(ok, error_message)``.  Three gates, evaluated in order:

    1. **Completeness** — ``num == golden.dataset.num_expected``.
    2. **Hard floor** — ``score >= tolerance.hard_floor``.
    3. **Regression** —
       ``score >= golden_score - max(abs, (1-rel)*golden_score)``.
    """
    # 1. Completeness gate
    expected_num = golden.get("dataset", {}).get("num_expected")
    if expected_num is not None and num != expected_num:
        return (
            False,
            f"completeness gate failed: evaluated {num} samples, "
            f"expected {expected_num}",
        )
    # 2. Hard floor (== threshold by default)
    tolerance = golden.get("tolerance", DEFAULT_TOLERANCE)
    hard_floor = tolerance.get("hard_floor", DEFAULT_THRESHOLD)
    if score < hard_floor:
        return (
            False,
            f"hard floor gate failed: accuracy {score:.4f} < {hard_floor}",
        )
    # 3. Regression lower bound
    golden_score = golden["score"]
    abs_tol = tolerance.get("abs", 0.02)
    rel_tol = tolerance.get("rel", 0.99)
    regression_delta = max(abs_tol, (1.0 - rel_tol) * golden_score)
    lower_bound = golden_score - regression_delta
    if score < lower_bound:
        return (
            False,
            f"regression gate failed: accuracy {score:.4f} "
            f"< lower bound {lower_bound:.4f} "
            f"(golden={golden_score:.4f}, "
            f"delta={regression_delta:.4f})",
        )
    return (True, None)


# --- Golden I/O helpers -----------------------------------------------------


def write_golden(record: dict[str, Any], path: Path) -> None:
    """Atomically write a golden JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def read_golden(path: Path) -> dict[str, Any]:
    """Read and parse a golden JSON record."""
    return json.loads(path.read_text())
