import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/_ppu-accuracy-tests.yaml"
SIDECAR = "legacy-accuracy-result.json"
PIN, HEAD = "a" * 40, "b" * 40


@pytest.fixture
def evaluator(monkeypatch):
    script = ROOT / "scripts/eval/ppu_gsm8k_eval.py"
    spec = importlib.util.spec_from_file_location("nightly_gsm8k_evaluator", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("GSM8K_DATASET_PATH", "/unused/dataset")
    spec.loader.exec_module(module)
    return module


def workflow_step(job, name):
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"][job]["steps"]
    return next(step for step in steps if step["name"] == name)


def run_step(job, name, env):
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", workflow_step(job, name)["run"]],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=10,
    )


def summarize(tmp_path, outcome, *, wheels="success"):
    step = run_step(
        "accuracy",
        "Generate public legacy accuracy summary",
        {
            "PRIVATE_DIR": str(tmp_path),
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "ACCURACY_OUTCOME": outcome,
            "WHEELS_OUTCOME": wheels,
        },
    )
    assert step.returncode == 0, step.stderr
    path = Path((tmp_path / "output").read_text().strip().removeprefix("summary="))
    assert {entry.name for entry in path.parent.iterdir()} == {"summary.json"}
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    ("outcome", "status", "reason", "gate"),
    [
        ("success", "pass", "not_applicable", "pass"),
        ("success", "created", "not_applicable", "pass"),
        ("failure", "fail_threshold", "gsm8k_gate_failed", "fail"),
        ("failure", "fail_accuracy", "gsm8k_gate_failed", "fail"),
        ("failure", "fail_completeness", "completeness_failed", "fail"),
        ("failure", "fail_config", "golden_config_mismatch", "fail"),
        ("failure", "fail_identity", "golden_identity_mismatch", "fail"),
        ("failure", "runtime_setup_failed", "runtime_setup_failed", "not_available"),
        ("failure", "server_startup_failed", "server_startup_failed", "not_available"),
        ("failure", "gsm8k_eval_failed", "gsm8k_eval_failed", "not_available"),
        ("failure", "pass", "unknown", "not_available"),
        ("success", "fail_accuracy", "unknown", "not_available"),
        ("cancelled", "fail_accuracy", "unknown", "not_available"),
        ("failure", "/private/path", "unknown", "not_available"),
    ],
)
def test_summary_projects_only_allowlisted_result(
    tmp_path, outcome, status, reason, gate
):
    (tmp_path / SIDECAR).write_text(
        json.dumps(
            {"status": status, "score": 0.91, "num": 1319, "error": "/private/log"}
        )
    )
    has_gate = gate != "not_available"
    expected_status = {"success": "pass", "failure": "fail", "cancelled": "cancelled"}[
        outcome
    ]
    assert summarize(tmp_path, outcome) == {
        "status": expected_status,
        "failure_reason": reason,
        "gsm8k": {
            "score": 0.91 if has_gate else None,
            "evaluated_samples": 1319 if has_gate else None,
            "gate": gate,
        },
    }


@pytest.mark.parametrize("source_type", ["missing", "invalid", "large", "link", "fifo"])
def test_unusable_evidence_is_unknown_without_inventing_a_gate(tmp_path, source_type):
    sidecar = tmp_path / SIDECAR
    if source_type == "invalid":
        sidecar.write_text("{::error::/private/log}")
    elif source_type == "large":
        sidecar.write_text('{"status":"pass"}' + " " * 4096)
    elif source_type == "link":
        target = tmp_path / "untrusted.json"
        target.write_text('{"status":"pass","score":1,"num":1319}')
        sidecar.symlink_to(target)
    elif source_type == "fifo":
        os.mkfifo(sidecar)
    assert summarize(tmp_path, "success") == {
        "status": "pass",
        "failure_reason": "unknown",
        "gsm8k": {"score": None, "evaluated_samples": None, "gate": "not_available"},
    }


@pytest.mark.parametrize(
    ("score", "samples"),
    [(True, False), (float("nan"), 1.5), (1.01, -1), (-0.1, 100_001)],
)
def test_out_of_range_or_non_numeric_fields_are_omitted(tmp_path, score, samples):
    (tmp_path / SIDECAR).write_text(
        json.dumps({"status": "fail_threshold", "score": score, "num": samples})
    )
    assert summarize(tmp_path, "failure")["gsm8k"] == {
        "score": None,
        "evaluated_samples": None,
        "gate": "fail",
    }


def test_failed_wheel_download_has_setup_evidence(tmp_path):
    summary = summarize(tmp_path, "skipped", wheels="failure")
    assert summary["status"] == "fail"
    assert summary["failure_reason"] == "runtime_setup_failed"


@pytest.mark.parametrize(
    ("head", "wheel", "run", "case", "passed"),
    [
        (HEAD, HEAD, "10", "deepseek-v4-mxfp4", True),
        (HEAD, HEAD, "10", "one,two", True),
        ("main", "main", "10", "deepseek-v4-mxfp4", False),
        (HEAD, PIN, "10", "deepseek-v4-mxfp4", False),
        (HEAD, HEAD, "11", "deepseek-v4-mxfp4", False),
        (HEAD, HEAD, "10", "all", False),
        (HEAD, HEAD, "10", "one,all", False),
        (HEAD, HEAD, "10", "one,one", False),
        (HEAD, HEAD, "10", "one,", False),
        (HEAD, HEAD, "10", ",one", False),
        (HEAD, HEAD, "10", "", False),
    ],
)
def test_pr_input_guard(head, wheel, run, case, passed):
    step = run_step(
        "prepare",
        "Validate PR Nightly inputs",
        {
            "PR_HEAD_SHA": head,
            "WHEEL_SOURCE_SHA": wheel,
            "GITHUB_RUN_ID": "10",
            "BUILD_RUN_ID": run,
            "CASE_ID": case,
        },
    )
    assert (step.returncode == 0) == passed


@pytest.mark.parametrize(
    ("models", "expected"),
    [
        ("case-a", [{"case_id": "case-a"}]),
        ("case-a,case-b", [{"case_id": "case-a"}, {"case_id": "case-b"}]),
        ("case-a,case-a", None),
        ("case-a,smoke-only", None),
        ("case-a,unknown", None),
    ],
)
def test_pr_select_expands_multiple_legacy_cases_and_rejects_invalid(
    tmp_path, models, expected
):
    git = tmp_path / "git"
    git.write_text(
        "#!/bin/bash\n"
        'echo \'[{"key":"case-a","category":"accuracy"},'
        '{"key":"case-b","category":"accuracy"},'
        '{"key":"smoke-only","category":"smoke"}]\'\n'
    )
    git.chmod(0o700)
    output = tmp_path / "github-output"
    step = run_step(
        "prepare",
        "Select accuracy models",
        {
            "PATH": os.pathsep.join(
                (str(tmp_path), str(Path(sys.executable).parent), os.environ["PATH"])
            ),
            "PR_MODE": "true",
            "MODELS": models,
            "GITHUB_OUTPUT": str(output),
        },
    )
    if expected is None:
        assert step.returncode != 0
        assert not output.exists() or output.read_text() == ""
        return
    assert step.returncode == 0, step.stderr
    line = output.read_text().strip()
    assert json.loads(line.removeprefix("accuracy_matrix=")) == {"include": expected}


@pytest.mark.parametrize("commits", [(PIN, HEAD), (HEAD, HEAD), (PIN, PIN)])
def test_launcher_checks_both_commits_and_keeps_historical_output_private(
    tmp_path, commits
):
    source = tmp_path / "source"
    wrapper = source / "scripts/ci/ppu_runtime_container.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        'test -z "${GITHUB_TOKEN+x}${ACTIONS_RUNTIME_TOKEN+x}${GITHUB_OUTPUT+x}" || exit 8\n'
        'test "$PPU_CI_REQUIRE_PR_ARTIFACT" = true || exit 9\n'
        'printf "%s\n" "private model output at $PPU_CI_ARTIFACT_DIR"\n'
        'printf "%s\n" "private stderr" >&2\n'
    )
    git = tmp_path / "git"
    git.write_text(
        f"#!/bin/bash\ncase $4 in\nHEAD:.github/*) echo {PIN};;\n"
        'HEAD:scripts/*) echo \'[{"key":"case","category":"accuracy",'
        '"model_id":"test/model","model_key":"fixture","ppu_count":1}]\';;\nesac\n'
    )
    git.chmod(0o700)
    (tmp_path / "wheel-manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "commits": dict(zip(("vllm", "vllm-sail"), commits, strict=False)),
            }
        )
    )
    step = run_step(
        "accuracy",
        "Run PR Nightly accuracy with private logging",
        {
            "PATH": os.pathsep.join(
                (str(tmp_path), str(Path(sys.executable).parent), os.environ["PATH"])
            ),
            "GITHUB_TOKEN": "test-token",
            "ACTIONS_RUNTIME_TOKEN": "test-runtime-token",
            "GITHUB_OUTPUT": str(tmp_path / "not-for-candidate"),
            "GITHUB_RUN_ID": "10",
            "GITHUB_RUN_ATTEMPT": "1",
            "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "PPU_CI_RUNTIME_IMAGE": "test-image",
            "PPU_CI_SOURCE_DIR": str(source),
            "PPU_CI_SOURCE_SHA": HEAD,
            "PPU_CI_WHEEL_SOURCE_SHA": HEAD,
            "PPU_CI_WHEELS_DIR": str(tmp_path),
            "PPU_CI_ARTIFACT_DIR": str(tmp_path),
            "PPU_CI_HF_HOME": str(tmp_path),
            "PPU_CI_DG_CACHE_DIR": str(tmp_path),
            "GSM8K_DATASET_PATH": str(tmp_path),
            "CASE_ID": "case",
        },
    )
    assert not step.stdout and not step.stderr
    private_log = (tmp_path / "launcher.log").read_text()
    if commits == (PIN, HEAD):
        assert step.returncode == 0
        assert f"private model output at {tmp_path}" in private_log
        assert "private stderr" in private_log
        assert not (tmp_path / SIDECAR).exists()
        assert summarize(tmp_path, "success")["gsm8k"]["gate"] == "not_available"
    else:
        assert step.returncode != 0
        assert "private model output" not in private_log
        assert (
            summarize(tmp_path, "failure")["failure_reason"] == "runtime_setup_failed"
        )


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("start_vllm_server", "server_startup_failed"),
        ("wait_server_ready", "server_startup_failed"),
        ("run_evalscope", "gsm8k_eval_failed"),
        ("extract_score", "gsm8k_eval_failed"),
        ("build_serve_argv", None),
    ],
)
def test_evaluator_records_failure_at_the_point_of_failure(
    tmp_path, monkeypatch, evaluator, stage, expected
):
    def fail(*args, **kwargs):
        raise RuntimeError("private model output")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 8)),
    )
    monkeypatch.setattr(evaluator, "resolve_gsm8k_dataset", lambda: "/private/dataset")
    monkeypatch.setattr(
        evaluator, "start_vllm_server", lambda *args, **kwargs: (object(), None)
    )
    monkeypatch.setattr(evaluator, "wait_server_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "SERVER_POST_READY_SLEEP", 0)
    monkeypatch.setattr(
        evaluator, "_ensure_launched_server_alive", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(evaluator, "run_evalscope", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator, "stop_server", lambda *args: None)
    monkeypatch.setattr(evaluator, stage, fail)
    result = tmp_path / SIDECAR
    with pytest.raises(RuntimeError, match="private model output"):
        evaluator.run_accuracy_eval(
            "deepseek_v4_mxfp4", chip="890p", work_dir=tmp_path, ci_result_file=result
        )
    if expected is None:
        assert not result.exists()
    else:
        assert json.loads(result.read_text()) == {"status": expected}
