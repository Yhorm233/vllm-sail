# SPDX-License-Identifier: Apache-2.0
"""Join vLLM SAIL Nightly job conclusions to their checked public summaries."""

import json
import os
from functools import partial

from ppu_notify_summary import MODEL, json_file, model_result, public_context, run_jobs

STARTUP = (
    "server_startup_failed",
    "server_readiness_timeout",
    "chat_probe_failed",
    "stream_probe_failed",
    "stream_http_error",
    "stream_transport_error",
    "stream_response_too_large",
    "stream_timeout",
    "invalid_stream_chunk",
    "truncated_stream",
    "missing_done",
    "empty_stream_content",
)
ACCURACY = (
    "gsm8k_eval_failed",
    "gsm8k_gate_failed",
    "completeness_failed",
    "golden_identity_mismatch",
    "golden_config_mismatch",
)
SETUP = (
    "runtime_setup_failed",
    "wheel_verification_failed",
    "recipe_validation_failed",
    "runtime_dependencies_failed",
    "platform_check_failed",
)


def evidence(root, model, prefix, run_id, attempt):
    report = json_file(root / f"{prefix}-{model}-{run_id}-{attempt}" / "summary.json")
    if prefix == "ppu-recipe-model_sanity" and report.get("model_key") != model:
        raise ValueError("Recipe model mismatch")
    status, reason = report["status"], report["failure_reason"]
    gsm8k = report["gsm8k"]
    if status == "pass" and (
        gsm8k.get("gate") != "pass"
        or (prefix == "ppu-recipe-model_sanity" and report.get("micro_e2e") != "pass")
    ):
        raise ValueError("Missing passing gate evidence")

    phase = "unknown"
    if reason in STARTUP:
        phase = "startup"
    elif reason in ACCURACY:
        phase = "gsm8k_score vs golden"
    elif reason == "micro_e2e_failed":
        phase = "micro_e2e"
    elif reason == "model_cache_check_failed":
        phase = "model_cache_check"
    elif reason in SETUP:
        phase = "runtime_setup"
    elif report.get("retention") == "failed":
        phase = "private_retention"
    return status, phase, gsm8k.get("score"), gsm8k.get("golden_score")


def keys(variable, field):
    matrix = json.loads(os.environ.get(variable) or '{"include":[]}')
    values = [item[field] for item in matrix["include"]]
    if len(values) != len(set(values)) or any(
        not MODEL.fullmatch(key) for key in values
    ):
        raise ValueError("Invalid model selection")
    return values


def failure(model, phase):
    return {"model": model, "status": "fail", "phase": phase}


def collect(root):
    _, _, _, run_id = public_context()
    legacy = keys("NOTIFY_LEGACY_MATRIX", "model")
    recipe = keys("NOTIFY_RECIPE_MATRIX", "key")
    legacy_jobs = {f"Accuracy {key} / Accuracy {key}": key for key in legacy}
    recipe_jobs = {
        f"Recipe model sanity & micro-E2E / Model sanity & micro-E2E {key}": key
        for key in recipe
    }
    jobs, elapsed = run_jobs(set(legacy_jobs) | set(recipe_jobs))
    build = os.environ["NOTIFY_BUILD_RESULT"]
    skip_build = os.environ.get("NOTIFY_SKIP_BUILD") == "true"
    if os.environ["NOTIFY_SELECT_RESULT"] != "success":
        results = [failure("select", "selection")]
    elif build != "success" and not (skip_build and build == "skipped"):
        results = [failure(key, "build") for key in legacy + recipe] or [
            failure("build", "build")
        ]
    elif os.environ.get("NOTIFY_IMAGE_RESULT", "skipped") not in ("success", "skipped"):
        results = [failure(key, "runtime_setup") for key in legacy + recipe] or [
            failure("resolve-image", "runtime_setup")
        ]
    else:
        results = []
        for selected, prefix in (
            (legacy_jobs, "ppu-nightly-legacy"),
            (recipe_jobs, "ppu-recipe-model_sanity"),
        ):
            for name, key in selected.items():
                results.append(
                    model_result(
                        key,
                        jobs.get(name),
                        partial(evidence, root, key, prefix, run_id),
                    )
                )
        if not results:
            results.append(failure("model-matrix", "selection"))
        if all(row["status"] == "pass" for row in results) and (
            os.environ["NOTIFY_LEGACY_RESULT"] == "failure"
            or os.environ["NOTIFY_RECIPE_RESULT"] == "failure"
        ):
            results.append(failure("model-matrix", "matrix"))
    return {"elapsed_s": elapsed, "results": results}
