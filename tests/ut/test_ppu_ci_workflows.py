# SPDX-License-Identifier: Apache-2.0
import hashlib
import itertools
import json
import os
import re
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
DEFAULT_BRANCH = "${{ github.event.repository.default_branch }}"
EXPECTED_SHA = "${{ inputs.pr_head_sha || github.sha }}"
DEFAULT_REF_GUARD = (
    "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
)
BUILD_IMAGE = (
    "download.t-head.cn/docker_release/pytorch@sha256:"
    "a33a67832f7ec442c481421e94255518dd67e464406f8e610c15d2b594b21f80"
)


def workflow(name):
    path = WORKFLOWS / name
    assert path.is_file(), f"Missing workflow: {name}"
    return yaml.safe_load(path.read_text())


def checkouts(job):
    return [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]


@pytest.mark.parametrize(
    "name,job_id,ref",
    [
        ("run-ci-command.yml", "command", DEFAULT_BRANCH),
        ("notify-ci-authorized.yml", "notify", DEFAULT_BRANCH),
        ("ppu-pr-ci.yaml", "validate", "${{ github.sha }}"),
        ("ppu-pr-ci.yaml", "report", "${{ github.sha }}"),
    ],
)
def test_control_jobs_only_execute_trusted_code(name, job_id, ref):
    job = workflow(name)["jobs"][job_id]
    assert job["runs-on"] == "ubuntu-latest"
    assert len(checkouts(job)) == 1
    checkout = checkouts(job)[0]
    assert checkout["with"]["ref"] == ref
    assert checkout["with"]["persist-credentials"] is False
    if name == "ppu-pr-ci.yaml":
        assert DEFAULT_REF_GUARD in job.get("if", checkout.get("if", "")) or (
            job_id == "report"
            and all(DEFAULT_REF_GUARD in step.get("if", "") for step in job["steps"])
        )


def test_ppu_jobs_use_hosted_build_and_self_hosted_runtime_runners():
    assert workflow("_ppu-build.yaml")["jobs"]["build"]["runs-on"] == "ubuntu-24.04"
    assert (
        workflow("_ppu-model-tests.yaml")["jobs"]["prepare"]["runs-on"]
        == "ubuntu-latest"
    )
    for filename, job_id in (
        ("_ppu-pin-ut.yaml", "pin-unit-tests"),
        ("_ppu-model-tests.yaml", "e2e"),
        ("_ppu-ops-tests.yaml", "ops-tests"),
        ("_ppu-upstream-reuse.yaml", "upstream-reuse"),
    ):
        assert workflow(filename)["jobs"][job_id]["runs-on"] == "8-ppu-m890p"


def test_controller_filters_comments_serializes_all_commands_and_reports_failure():
    config = workflow("run-ci-command.yml")
    assert config[True] == {"issue_comment": {"types": ["created"]}}
    job = config["jobs"]["command"]
    assert "github.event.issue.pull_request" in job["if"]
    assert "github.event.comment.user.type == 'User'" in job["if"]
    assert "contains(github.event.comment.body, '/ci')" in job["if"]
    assert job["concurrency"] == {
        "group": "ppu-ci-command-${{ github.event.issue.number }}",
        "cancel-in-progress": False,
        "queue": "max",
    }
    fallback = next(step for step in job["steps"] if step.get("if") == "failure()")
    assert fallback["uses"].startswith("actions/github-script@")
    assert "ppu-ci-command:" in fallback["with"]["script"]
    assert "context.runId" in fallback["with"]["script"]


def test_notify_only_receives_label_events_with_minimum_permissions():
    config = workflow("notify-ci-authorized.yml")
    assert config[True] == {"pull_request_target": {"types": ["labeled"]}}
    job = config["jobs"]["notify"]
    assert "ready-run-all-tests" in job["if"]


# GITHUB_TOKEN needs pull-requests: write to comment or react on a PR;
# issues: write alone returns 403 "Resource not accessible by integration".
@pytest.mark.parametrize(
    "name,job_id,expected",
    [
        (
            "run-ci-command.yml",
            "command",
            {
                "contents": "read",
                "actions": "write",
                "issues": "write",
                "pull-requests": "write",
                "statuses": "write",
            },
        ),
        (
            "notify-ci-authorized.yml",
            "notify",
            {"contents": "read", "issues": "write", "pull-requests": "write"},
        ),
        (
            "ppu-pr-ci.yaml",
            "report",
            {
                "contents": "read",
                "actions": "read",
                "statuses": "write",
                "issues": "write",
                "pull-requests": "write",
            },
        ),
    ],
)
def test_pr_commenting_jobs_can_write_pull_request_comments(name, job_id, expected):
    assert workflow(name)["jobs"][job_id]["permissions"] == expected


def test_executor_uses_only_dispatch_with_a_stable_request_identity():
    config = workflow("ppu-pr-ci.yaml")
    assert set(config[True]) == {"workflow_dispatch"}
    inputs = config[True]["workflow_dispatch"]["inputs"]
    required_inputs = {"pr_number", "head_sha", "requester", "comment_id"}
    assert set(inputs) == required_inputs
    for name in required_inputs:
        assert inputs[name]["type"] == "string" and inputs[name]["required"]
    assert config["run-name"] == (
        "PPU PR #${{ inputs.pr_number }} @ ${{ inputs.head_sha }} "
        "by ${{ inputs.requester }} (comment ${{ inputs.comment_id }})"
    )
    assert config["permissions"] == {}
    assert config["cache-mode"] == "none"
    assert config["concurrency"] == {
        "group": "ppu-pr-ci-${{ inputs.pr_number }}",
        "cancel-in-progress": False,
        "queue": "max",
    }
    assert set(config["jobs"]) == {"validate", "cpu", "ppu", "report"}


def test_executor_separates_read_only_pr_execution_and_always_reporting():
    jobs = workflow("ppu-pr-ci.yaml")["jobs"]
    cpu = jobs["cpu"]
    assert cpu["needs"] == "validate"
    assert cpu["uses"] == "./.github/workflows/ut.yaml"
    assert cpu["permissions"] == {"contents": "read"}
    assert cpu["with"] == {
        "pr_head_sha": "${{ inputs.head_sha }}",
        "pr_number": "${{ inputs.pr_number }}",
        "pr_head_ref": "${{ needs.validate.outputs.pr_head_ref }}",
    }
    assert "if" not in cpu
    assert "secrets" not in cpu
    ppu = jobs["ppu"]
    assert ppu["needs"] == "validate"
    assert ppu["uses"] == "./.github/workflows/ppu-build-and-test.yaml"
    assert ppu["permissions"] == {"contents": "read", "actions": "read"}
    assert "secrets" not in ppu
    assert ppu["with"] == {
        "pr_head_sha": "${{ inputs.head_sha }}",
        "skip_ppu": "${{ needs.validate.outputs.skip_ppu == 'true' }}",
        "existing_build_run_id": "${{ needs.validate.outputs.existing_build_run_id }}",
        "native_reuse_run_id": "${{ needs.validate.outputs.native_reuse_run_id }}",
        "wheel_source_sha": "${{ needs.validate.outputs.wheel_source_sha }}",
    }
    validate = jobs["validate"]
    assert validate["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "read",
        "statuses": "write",
    }
    step = next(step for step in validate["steps"] if "run" in step)
    assert step["run"] == "python -m scripts.ci.ppu_ci_command validate"
    assert "continue-on-error" not in step
    assert validate["outputs"] == {
        key: "${{ steps." + step["id"] + ".outputs." + key + " }}"
        for key in (
            "pr_head_ref",
            "skip_ppu",
            "existing_build_run_id",
            "native_reuse_run_id",
            "wheel_source_sha",
        )
    }
    report = jobs["report"]
    assert report["if"] == "always()"
    assert report["needs"] == ["validate", "cpu", "ppu"]
    assert all("always()" in step["if"] for step in report["steps"])
    step = next(step for step in report["steps"] if "run" in step)
    assert step["env"]["NEEDS"] == "${{ toJSON(needs) }}"
    assert "continue-on-error" not in step
    assert step["run"] == "python -m scripts.ci.ppu_ci_command report"


@pytest.mark.parametrize(
    "name",
    [
        "ppu-build-and-test.yaml",
        "_ppu-build.yaml",
        "_ppu-pin-ut.yaml",
        "_ppu-model-tests.yaml",
        "_ppu-kernel-perf.yaml",
        "_ppu-upstream-reuse.yaml",
        "ppu-ops-tests.yaml",
        "ppu-debug.yaml",
        "kernel-perf-label.yaml",
        "perf-regression.yaml",
    ],
)
def test_untrusted_workflows_have_no_write_permissions_or_inherited_secrets(name):
    config = workflow(name)
    assert config["cache-mode"] == "none"
    assert config["permissions"] == {"contents": "read", "actions": "read"}
    for job in config["jobs"].values():
        assert "secrets" not in job
        assert all(value != "write" for value in job.get("permissions", {}).values())
        for step in checkouts(job) if "steps" in job else []:
            assert step["with"]["persist-credentials"] is False


@pytest.mark.parametrize(
    "name,job_ids",
    [
        ("_ppu-build.yaml", ["build"]),
        ("_ppu-pin-ut.yaml", ["pin-unit-tests"]),
        ("_ppu-model-tests.yaml", ["prepare", "e2e"]),
        ("_ppu-kernel-perf.yaml", ["kernel-perf"]),
        ("_ppu-upstream-reuse.yaml", ["upstream-reuse"]),
    ],
)
def test_every_pr_checkout_verifies_the_requested_sha_before_execution(name, job_ids):
    config = workflow(name)
    inputs = config[True]["workflow_call"]["inputs"]
    if name in ("_ppu-build.yaml",):
        assert "pr_mode" not in inputs
    else:
        assert inputs["pr_mode"]["type"] == "boolean"
        assert inputs["pr_mode"]["default"] is False
    assert inputs["pr_head_sha"]["type"] == "string"
    assert inputs["pr_head_sha"]["default"] == ""
    for job_id in job_ids:
        job = config["jobs"][job_id]
        checkout = checkouts(job)[0]
        assert checkout["with"]["ref"] == EXPECTED_SHA
        index = job["steps"].index(checkout)
        verify = job["steps"][index + 1]
        assert verify["env"]["EXPECTED_SHA"] == EXPECTED_SHA
        if "SOURCE_DIR" in verify["env"]:
            assert checkout["with"]["path"] == "${{ steps.runtime.outputs.source_rel }}"
            assert verify["env"]["SOURCE_DIR"] == "${{ steps.runtime.outputs.source }}"
            assert (
                'test "$(git -c safe.directory="$SOURCE_DIR" -C "$SOURCE_DIR" rev-parse HEAD)" = "$EXPECTED_SHA"'
                == verify["run"]
            )
        else:
            assert (
                'test "$(git -c safe.directory="$PWD" rev-parse HEAD)" = "$EXPECTED_SHA"'
                == verify["run"]
            )
        guard = next(
            step
            for step in job["steps"]
            if step.get("name") == "Require disabled cache access"
        )
        assert job["steps"].index(guard) < index
        assert "process.env.ACTIONS_CACHE_MODE !== 'none'" in guard["with"]["script"]


@pytest.mark.parametrize(
    "name,job_id",
    [
        ("_ppu-build.yaml", "build"),
        ("_ppu-pin-ut.yaml", "pin-unit-tests"),
        ("_ppu-model-tests.yaml", "prepare"),
        ("_ppu-model-tests.yaml", "e2e"),
        ("_ppu-kernel-perf.yaml", "kernel-perf"),
        ("_ppu-upstream-reuse.yaml", "upstream-reuse"),
    ],
)
def test_sha_verification_allows_different_owner_but_rejects_wrong_sha(
    name, job_id, tmp_path
):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    sha = "a" * 40
    (tmp_path / ".git" / "HEAD").write_text(sha + "\n")
    env = {
        **os.environ,
        "LC_ALL": "C",
        "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "",
        "EXPECTED_SHA": sha,
    }
    untrusted = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert untrusted.returncode != 0
    assert "dubious ownership" in untrusted.stderr
    job = workflow(name)["jobs"][job_id]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Verify checked-out revision"
    )
    for expected, returncode in ((sha, 0), ("b" * 40, 1)):
        result = subprocess.run(
            ["sh", "-e", "-c", command],
            cwd=tmp_path,
            env={**env, "EXPECTED_SHA": expected, "SOURCE_DIR": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == returncode, result.stderr
        assert not result.stderr


def test_pr_pipeline_structure():
    config = workflow("ppu-build-and-test.yaml")
    assert set(config[True]) == {"workflow_call", "pull_request"}
    inputs = config[True]["workflow_call"]["inputs"]
    assert set(inputs) == {
        "pr_head_sha",
        "skip_ppu",
        "existing_build_run_id",
        "native_reuse_run_id",
        "skip_device_tests",
        "wheel_source_sha",
    }
    assert inputs["pr_head_sha"]["type"] == "string"
    assert inputs["pr_head_sha"]["required"] is True
    assert inputs["skip_ppu"] == {
        "description": "Report success without allocating PPU runners for non-impacting changes",
        "type": "boolean",
        "default": False,
    }
    assert inputs["skip_device_tests"] == {
        "description": "Build wheels only, without pin-UT, model checks or upstream-reuse lanes",
        "type": "boolean",
        "default": False,
    }
    for name in (
        "existing_build_run_id",
        "native_reuse_run_id",
        "wheel_source_sha",
    ):
        assert inputs[name]["type"] == "string"
        assert inputs[name]["default"] == ""
        assert not inputs[name].get("required", False)
    jobs = config["jobs"]
    assert set(jobs) == {
        "runtime-image",
        "preview-donor",
        "build",
        "pin-unit-tests",
        "model-tests",
        "recipe-sanity",
        "upstream-reuse",
        "gate",
    }
    runtime = jobs["runtime-image"]
    assert runtime["runs-on"] == "8-ppu-m890p"
    assert "!inputs.skip_ppu" in runtime["if"]
    assert runtime["outputs"] == {
        "upstream_reuse": "${{ steps.reuse.outputs.supported }}",
        "recipe_sanity": "${{ steps.recipe.outputs.supported }}",
    }
    resolve = next(
        step
        for step in runtime["steps"]
        if step.get("name") == "Resolve the runner-configured CUDA-compatible runtime"
    )
    assert "ppu_runtime_image.sh" in resolve["run"]
    assert "PPU_CI_RUNTIME_IMAGE" not in resolve.get("env", {})
    build = jobs["build"]
    assert "ci-preview" in build["if"]
    assert "inputs.existing_build_run_id || ''" in build["if"]
    assert build["uses"] == "./.github/workflows/_ppu-build.yaml"
    assert build["with"] == {
        "pr_head_sha": "${{ inputs.pr_head_sha || github.sha }}",
        "native_reuse_run_id": (
            "${{ inputs.native_reuse_run_id || needs.preview-donor.outputs.run_id || '' }}"
        ),
    }
    assert build["needs"] == ["runtime-image", "preview-donor"]
    assert jobs["gate"]["needs"] == [
        "runtime-image",
        "build",
        "pin-unit-tests",
        "model-tests",
        "recipe-sanity",
        "upstream-reuse",
    ]
    assert "always()" in jobs["gate"]["if"]
    assert "ci-preview" in jobs["gate"]["if"]
    gate = jobs["gate"]["steps"][0]
    assert gate["env"] == {
        "JOB_RESULTS": "${{ toJSON(needs) }}",
        "SKIP_PPU": "${{ inputs.skip_ppu }}",
        "SKIP_DEVICE_TESTS": "${{ inputs.skip_device_tests }}",
        "EXISTING_BUILD_RUN_ID": "${{ inputs.existing_build_run_id || '' }}",
        "WHEEL_SOURCE_SHA": "${{ inputs.wheel_source_sha || '' }}",
    }
    assert "PR_MODE" not in gate["run"]
    assert "smoke" not in gate["run"]
    for job_id, name, target in (
        ("pin-unit-tests", "PPU pin UT", "_ppu-pin-ut.yaml"),
        ("model-tests", "PPU model tests", "_ppu-model-tests.yaml"),
        ("recipe-sanity", "PPU recipe fast sanity", "_ppu-recipe-fast-sanity.yaml"),
        ("upstream-reuse", "PPU upstream reuse", "_ppu-upstream-reuse.yaml"),
    ):
        job = jobs[job_id]
        assert job["name"] == name
        assert job["needs"] == ["runtime-image", "build"]
        assert job["uses"] == f"./.github/workflows/{target}"
        assert "!cancelled()" in job["if"]
        assert "ci-preview" in job["if"]
        assert "needs.build.result == 'success'" in job["if"]
        assert job["with"] == {
            "pr_mode": True,
            "pr_head_sha": "${{ inputs.pr_head_sha || github.sha }}",
            "wheel_source_sha": "${{ inputs.wheel_source_sha || '' }}",
            "build_run_id": "${{ inputs.existing_build_run_id || github.run_id }}",
        }
    # kernel-perf is no longer in the gate; it lives in kernel-perf-label.yaml
    assert "kernel-perf" not in jobs


def test_docs_only_call_uses_hosted_gate_without_ppu_jobs():
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    context = {
        "github.event_name": "workflow_dispatch",
        "inputs.skip_ppu": True,
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    for job_id in (
        "runtime-image",
        "build",
        "pin-unit-tests",
        "model-tests",
        "recipe-sanity",
        "upstream-reuse",
    ):
        assert not expression(jobs[job_id]["if"], context)
    assert expression(jobs["gate"]["if"], context)
    gate = jobs["gate"]["steps"][0]["run"].replace(
        "python - <<'PYTHON'", f"{sys.executable} - <<'PYTHON'"
    )
    result = subprocess.run(
        ["bash", "-e", "-c", gate],
        env={
            **os.environ,
            "JOB_RESULTS": json.dumps(
                {
                    job: {"result": "skipped"}
                    for job in (
                        "runtime-image",
                        "build",
                        "pin-unit-tests",
                        "model-tests",
                        "recipe-sanity",
                        "upstream-reuse",
                    )
                }
            ),
            "SKIP_PPU": "true",
            "SKIP_DEVICE_TESTS": "false",
            "EXISTING_BUILD_RUN_ID": "",
            "WHEEL_SOURCE_SHA": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PPU CI skipped: no PPU-impacting changes" in result.stdout


def test_skip_device_tests_guards_the_device_lanes():
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    context = {
        "github.event_name": "workflow_dispatch",
        "inputs.skip_ppu": False,
        "inputs.skip_device_tests": True,
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "success",
        "needs.build.result": "success",
    }
    # Building wheels for a lane that tests elsewhere must not also allocate
    # the pin-UT and Fast Sanity hosts.
    for job_id in ("pin-unit-tests", "model-tests", "recipe-sanity", "upstream-reuse"):
        assert not expression(jobs[job_id]["if"], context)
    assert expression(jobs["build"]["if"], context)
    assert expression(jobs["gate"]["if"], context)


@pytest.mark.parametrize("skip_device_tests,returncode", [("true", 0), ("false", 1)])
def test_gate_honours_skip_device_tests(skip_device_tests, returncode):
    step = workflow("ppu-build-and-test.yaml")["jobs"]["gate"]["steps"][0]
    script = step["run"].replace(
        "python - <<'PYTHON'", f"{sys.executable} - <<'PYTHON'"
    )
    env = {
        **os.environ,
        "JOB_RESULTS": json.dumps(
            {
                "runtime-image": {"result": "success"},
                "build": {"result": "success"},
                "pin-unit-tests": {"result": "skipped"},
                "model-tests": {"result": "skipped"},
                "recipe-sanity": {"result": "skipped"},
                "upstream-reuse": {"result": "skipped"},
            }
        ),
        "SKIP_PPU": "false",
        "SKIP_DEVICE_TESTS": skip_device_tests,
        "EXISTING_BUILD_RUN_ID": "",
        "WHEEL_SOURCE_SHA": "",
    }
    result = subprocess.run(
        ["bash", "-e", "-c", script], env=env, capture_output=True, text=True
    )
    # Only the explicit switch tolerates the skipped lanes; the default stays
    # fail-closed so a broken device lane cannot hide behind it.
    assert result.returncode == returncode, result.stderr


def test_ops_manual_entry_structure():
    config = workflow("ppu-ops-tests.yaml")
    assert config["name"] == "PPU ops tests (manual)"
    assert set(config[True]) == {"workflow_dispatch"}
    assert config[True]["workflow_dispatch"]["inputs"] == {
        "build_image": {
            "description": "Digest-pinned SDK image; CI installs the checksum-pinned SAIL Torch",
            "required": False,
            "default": BUILD_IMAGE,
            "type": "string",
        },
        "ops_image": {
            "description": "Digest-pinned operator runtime image (required; value is not stored in the repository)",
            "required": True,
            "type": "string",
        },
        "ops_test_scope": {
            "description": "Initial manual runs always use full; auto remains disabled until qualification",
            "default": "full",
            "type": "choice",
            "options": ["auto", "full"],
        },
        "vllm_ref": {
            "description": "vLLM ref (blank uses .github/vllm-main-verified.commit)",
            "required": False,
            "default": "",
            "type": "string",
        },
        "pytorch_sail_arch": {
            "description": "Semicolon-separated HG architectures",
            "required": True,
            "default": "ppu_15;ppu_10",
            "type": "string",
        },
    }
    assert config["concurrency"] == {
        "group": "ppu-ops-tests-${{ github.ref }}",
        "cancel-in-progress": True,
    }
    jobs = config["jobs"]
    assert set(jobs) == {
        "detect-ops-changes",
        "ops-build",
        "select-ops-tests",
        "ops-tests",
    }
    assert "if" not in jobs["detect-ops-changes"]
    assert jobs["ops-build"]["needs"] == "detect-ops-changes"
    assert "if" not in jobs["ops-build"]
    assert jobs["select-ops-tests"]["needs"] == ["detect-ops-changes", "ops-build"]
    assert "if" not in jobs["select-ops-tests"]
    assert jobs["ops-tests"]["needs"] == ["detect-ops-changes", "select-ops-tests"]
    assert (
        jobs["ops-tests"]["if"] == "needs.select-ops-tests.outputs.has_tests == 'true'"
    )
    assert jobs["ops-tests"]["uses"] == "./.github/workflows/_ppu-ops-tests.yaml"


def test_debug_structure():
    config = workflow("ppu-debug.yaml")
    assert config["name"] == "PPU debug"
    assert set(config[True]) == {"workflow_dispatch"}
    inputs = config[True]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "build_run_id",
        "wheel_source_sha",
        "target",
        "models",
        "accuracy_models",
    }
    assert inputs["build_run_id"]["type"] == "string"
    assert inputs["build_run_id"]["required"] is True
    assert inputs["wheel_source_sha"]["type"] == "string"
    assert inputs["wheel_source_sha"]["required"] is True
    assert inputs["target"]["type"] == "choice"
    assert inputs["target"]["options"] == [
        "both",
        "pin-ut",
        "model-tests",
        "accuracy-tests",
        "kernel-perf",
    ]
    assert inputs["target"]["default"] == "both"
    assert inputs["models"]["type"] == "string"
    assert inputs["models"]["default"] == "qwen3.8-27b-m890p"
    assert inputs["accuracy_models"]["type"] == "string"
    assert inputs["accuracy_models"]["default"] == "deepseek-v4-mxfp4"
    assert "concurrency" not in config
    jobs = config["jobs"]
    expected_jobs = {
        "pin-unit-tests": (
            "inputs.target == 'both' || inputs.target == 'pin-ut'",
            "_ppu-pin-ut.yaml",
            {},
        ),
        "model-tests": (
            "inputs.target == 'both' || inputs.target == 'model-tests'",
            "_ppu-model-tests.yaml",
            {"models": "${{ inputs.models }}"},
        ),
        "accuracy-tests": (
            "inputs.target == 'accuracy-tests'",
            "_ppu-accuracy-tests.yaml",
            {"models": "${{ inputs.accuracy_models }}"},
        ),
        "kernel-perf": (
            "inputs.target == 'both' || inputs.target == 'kernel-perf'",
            "_ppu-kernel-perf.yaml",
            {},
        ),
    }
    assert set(jobs) == set(expected_jobs)
    for job_id, (condition, target, extra_inputs) in expected_jobs.items():
        job = jobs[job_id]
        assert job["if"] == condition
        assert "needs" not in job
        assert job["uses"] == f"./.github/workflows/{target}"
        assert job["with"] == {
            "pr_mode": False,
            "build_run_id": "${{ inputs.build_run_id }}",
            "wheel_source_sha": "${{ inputs.wheel_source_sha }}",
            **extra_inputs,
        }


def test_build_leaf_preserves_retry_identity_and_image_output():
    config = workflow("_ppu-build.yaml")
    assert set(config[True]) == {"workflow_call"}
    call = config[True]["workflow_call"]
    assert set(call["inputs"]) == {
        "pr_head_sha",
        "build_image",
        "vllm_ref",
        "pytorch_sail_arch",
        "native_reuse_run_id",
    }
    for value in call["inputs"].values():
        assert value["type"] == "string"
        assert value["default"] == ""
        assert not value.get("required", False)
    assert set(call["outputs"]) == {"image"}
    assert call["outputs"]["image"]["value"] == "${{ jobs.build.outputs.image }}"
    assert set(config["jobs"]) == {"build"}
    build = config["jobs"]["build"]
    # The controller identifies successful wheel builds by the leaf job name on retry.
    assert build["name"] == "PPU wheel build"
    assert build["outputs"] == {"image": "${{ steps.image.outputs.image }}"}
    assert "if" not in build
    assert "needs" not in build
    caller = workflow("ppu-build-and-test.yaml")["jobs"]["build"]
    assert caller["uses"] == "./.github/workflows/_ppu-build.yaml"
    download = next(
        step
        for step in build["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download["if"] == "inputs.native_reuse_run_id != ''"
    assert download["continue-on-error"] is True
    assert download["with"] == {
        "name": "ppu-wheels-${{ inputs.native_reuse_run_id }}",
        "path": "artifacts/native-reuse",
        "github-token": "${{ github.token }}",
        "run-id": "${{ inputs.native_reuse_run_id }}",
    }
    package = next(
        step
        for step in build["steps"]
        if step.get("name") == "Build or repackage wheels"
    )
    assert "VLLM_SAIL_NATIVE_REUSE_DIR" in package["env"]


def test_workflow_filenames_distinguish_reusable_calls_from_entrypoints():
    for path in WORKFLOWS.glob("*.y*ml"):
        triggers = workflow(path.name)[True]
        if path.name == "ppu-build-and-test.yaml":
            assert set(triggers) == {"workflow_call", "pull_request"}
        elif path.name == "ut.yaml":
            assert set(triggers) == {"workflow_call", "push", "workflow_dispatch"}
        elif path.name.startswith("_"):
            assert set(triggers) == {"workflow_call"}, path.name
        else:
            assert "workflow_call" not in triggers, path.name


@pytest.mark.parametrize(
    "name", ["build-ppu-wheels.yaml", "ppu-pin-ut.yaml", "e2e-ppu.yaml"]
)
def test_obsolete_workflow_paths_are_absent(name):
    assert not (WORKFLOWS / name).exists(), name


def test_pr_pipeline_is_the_only_ppu_pull_request_route():
    for path in WORKFLOWS.glob("*.y*ml"):
        triggers = workflow(path.name)[True]
        if path.name == "ppu-build-and-test.yaml":
            assert set(triggers) == {"workflow_call", "pull_request"}
        elif path.name == "kernel-perf-label.yaml":
            assert set(triggers) == {"pull_request"}
        elif path.name.startswith(("_ppu-", "ppu-")):
            assert "pull_request" not in triggers, path.name


def test_preview_route_requires_the_ci_preview_label():
    config = workflow("ppu-build-and-test.yaml")
    assert config["name"] == "PPU build and test"
    assert config["run-name"] == (
        "${{ github.event_name == 'pull_request' && "
        "format('PPU preview PR #{0} @ {1}', github.event.pull_request.number, "
        "github.sha) || 'PPU build and test' }}"
    )
    assert config[True]["pull_request"] == {
        "branches": ["main"],
        "types": [
            "opened",
            "synchronize",
            "reopened",
            "ready_for_review",
            "labeled",
            "unlabeled",
        ],
        "paths": [".github/workflows/**", "scripts/ci/**"],
    }
    assert not (WORKFLOWS / "ppu-pr-preview.yaml").exists()
    for name, job in config["jobs"].items():
        if name == "preview-donor":
            assert job["needs"] == "runtime-image"
            continue  # Inherits preview admission from runtime-image.
        guard = job["if"]
        assert "github.event_name != 'pull_request'" in guard
        assert "!github.event.pull_request.draft" in guard
        assert "ci-preview" in guard
        assert "github.event.label.name == 'ci-preview'" in guard


@pytest.mark.parametrize(
    "event,lookup,donor,formal,expected",
    [
        ("pull_request", "success", "100", "", "100"),
        ("pull_request", "success", "", "", ""),
        ("pull_request", "failure", "", "", ""),
        ("workflow_dispatch", "skipped", "", "101", "101"),
    ],
)
def test_preview_reuses_native_donor_without_skipping_candidate_build(
    event, lookup, donor, formal, expected
):
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    selection = jobs["preview-donor"]
    context = {
        "github.event_name": event,
        "github.event.action": "synchronize",
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["ci-preview"],
        "github.sha": "a" * 40,
        "github.run_id": "200",
        "needs.runtime-image.result": "success",
        "needs.preview-donor.result": lookup,
        "needs.preview-donor.outputs.run_id": donor,
        "inputs.native_reuse_run_id": formal,
    }
    assert bool(expression(selection["if"], context)) == (event == "pull_request")
    assert selection["runs-on"] == "ubuntu-24.04"
    assert selection["continue-on-error"] is True
    assert selection["permissions"] == {"contents": "read", "actions": "read"}
    assert selection["outputs"] == {"run_id": "${{ steps.select.outputs.run_id }}"}
    checkout = checkouts(selection)[0]
    assert expression(checkout["with"]["ref"], context) == context["github.sha"]
    assert checkout["with"]["persist-credentials"] is False
    select = next(step for step in selection["steps"] if step.get("id") == "select")
    assert select["run"] == "python3 -m scripts.ci.ppu_preview_wheel_donor"
    assert expression(jobs["build"]["if"], context)
    assert expression(jobs["build"]["with"]["native_reuse_run_id"], context) == expected
    assert expression(jobs["recipe-sanity"]["with"]["build_run_id"], context) == "200"


def test_preview_label_is_distinct_from_formal_ready_authorization():
    text = (WORKFLOWS / "ppu-build-and-test.yaml").read_text()
    assert "ci-preview" in text
    assert "ready and ready-run-all-tests continue to authorize only" in text
    for label in ("ready", "ready-run-all-tests"):
        assert label not in config_preview_guards()


def config_preview_guards():
    return "\n".join(
        job["if"] for job in workflow("ppu-build-and-test.yaml")["jobs"].values()
    )


def test_preview_cancellation_is_per_pr_and_formal_calls_are_isolated():
    concurrency = workflow("ppu-build-and-test.yaml")["concurrency"]
    cancel = concurrency["cancel-in-progress"]
    assert "github.event.label.name == 'ci-preview'" in cancel

    groups = []
    for number in (49, 50):
        context = {
            "github.event_name": "pull_request",
            "github.event.action": "synchronize",
            "github.event.label.name": None,
            "github.event.pull_request.number": number,
            "github.run_id": "900",
        }
        group = re.sub(
            r"\$\{\{.*?\}\}",
            lambda match, context=context: str(expression(match[0], context)),
            concurrency["group"],
        )
        assert group == f"ppu-build-and-test-{number}"
        assert expression(cancel, context) is True
        groups.append(group)
    assert len(set(groups)) == 2

    formal = {
        "github.event_name": "workflow_dispatch",
        "github.event.pull_request.number": 0,
        "github.run_id": "901",
    }
    group = re.sub(
        r"\$\{\{.*?\}\}",
        lambda match: str(expression(match[0], formal)),
        concurrency["group"],
    )
    assert group == "ppu-build-and-test-901"
    assert expression(cancel, formal) is False


def test_unrelated_label_events_do_not_cancel_or_start_preview():
    config = workflow("ppu-build-and-test.yaml")
    concurrency = config["concurrency"]
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "labeled",
        "github.event.label.name": "ready",
        "github.event.pull_request.number": 49,
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["ci-preview", "ready"],
        "github.run_id": "902",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    group = re.sub(
        r"\$\{\{.*?\}\}",
        lambda match: str(expression(match[0], context)),
        concurrency["group"],
    )
    assert group == "ppu-build-and-test-902"
    assert expression(concurrency["cancel-in-progress"], context) is False
    assert all(not expression(job["if"], context) for job in config["jobs"].values())


def test_removing_ci_preview_cancels_active_preview_without_starting_jobs():
    config = workflow("ppu-build-and-test.yaml")
    concurrency = config["concurrency"]
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "unlabeled",
        "github.event.label.name": "ci-preview",
        "github.event.pull_request.number": 49,
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": [],
        "github.run_id": "903",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    group = re.sub(
        r"\$\{\{.*?\}\}",
        lambda match: str(expression(match[0], context)),
        concurrency["group"],
    )
    assert group == "ppu-build-and-test-49"
    assert expression(concurrency["cancel-in-progress"], context) is True
    assert all(not expression(job["if"], context) for job in config["jobs"].values())


@pytest.mark.parametrize(
    "caller_id,leaf_name,job_ids",
    [
        ("build", "_ppu-build.yaml", ["build"]),
        ("pin-unit-tests", "_ppu-pin-ut.yaml", ["pin-unit-tests"]),
        ("model-tests", "_ppu-model-tests.yaml", ["prepare", "e2e"]),
        ("recipe-sanity", "_ppu-recipe-fast-sanity.yaml", ["prepare", "sanity"]),
        ("upstream-reuse", "_ppu-upstream-reuse.yaml", ["upstream-reuse"]),
    ],
)
def test_labeled_preview_propagates_merge_sha_to_build_and_test_checkouts(
    caller_id, leaf_name, job_ids
):
    merge_sha, head_sha = "a" * 40, "b" * 40
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "synchronize",
        "github.event.label.name": None,
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["ci-preview"],
        "github.sha": merge_sha,
        "github.event.pull_request.head.sha": head_sha,
        "github.run_id": "200",
        "inputs.pr_head_sha": "",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "success",
        "needs.build.result": "success",
    }
    pipeline = workflow("ppu-build-and-test.yaml")["jobs"]
    assert expression(pipeline["build"]["if"], context)
    caller = pipeline[caller_id]
    assert caller["uses"] == f"./.github/workflows/{leaf_name}"
    context["inputs.pr_head_sha"] = expression(caller["with"]["pr_head_sha"], context)
    assert context["inputs.pr_head_sha"] == merge_sha
    if caller_id != "build":
        assert caller["with"]["pr_mode"] is True
        assert expression(caller["with"]["wheel_source_sha"], context) == ""
    for job_id in job_ids:
        job = workflow(leaf_name)["jobs"][job_id]
        checkout = checkouts(job)[0]
        assert expression(checkout["with"]["ref"], context) == merge_sha
        verify = job["steps"][job["steps"].index(checkout) + 1]
        assert expression(verify["env"]["EXPECTED_SHA"], context) == merge_sha


@pytest.mark.parametrize(
    "labels,draft", [([], False), (["ready"], False), (["ci-preview"], True)]
)
def test_preview_jobs_skip_without_an_active_ci_preview_label(labels, draft):
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    context = {
        "github.event_name": "pull_request",
        "github.event.action": "synchronize",
        "github.event.label.name": None,
        "github.event.pull_request.draft": draft,
        "github.event.pull_request.labels.*.name": labels,
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "skipped",
        "needs.build.result": "skipped",
    }
    assert all(not expression(job["if"], context) for job in jobs.values())


@pytest.mark.parametrize(
    "target,expected",
    [
        ("both", (True, True, True)),
        ("pin-ut", (True, False, False)),
        ("model-tests", (False, True, False)),
        ("kernel-perf", (False, False, True)),
    ],
)
def test_debug_targets_select_requested_consumers(target, expected):
    jobs = workflow("ppu-debug.yaml")["jobs"]
    context = {"inputs.target": target}
    assert (
        tuple(
            bool(expression(jobs[job_id]["if"], context))
            for job_id in ("pin-unit-tests", "model-tests", "kernel-perf")
        )
        == expected
    )


def test_pr_wheel_verification_runs_inside_the_runtime_container():
    source = "${{ inputs.wheel_source_sha || inputs.pr_head_sha || github.sha }}"
    for workflow_name, job_id in (
        ("_ppu-pin-ut.yaml", "pin-unit-tests"),
        ("_ppu-model-tests.yaml", "e2e"),
    ):
        job = workflow(workflow_name)["jobs"][job_id]
        assert not any(
            "ppu_wheel_manifest.py" in step.get("run", "") for step in job["steps"]
        )
        runtime = next(
            step
            for step in job["steps"]
            if "ppu_runtime_container.sh" in step.get("run", "")
        )
        assert runtime["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
        assert runtime["env"]["PPU_CI_WHEEL_SOURCE_SHA"] == source

    pin_script = (ROOT / "scripts/ci/ppu_pin_ut.sh").read_text()
    model_script = (ROOT / "scripts/ci/ppu_e2e_smoke.sh").read_text()
    for script in (pin_script, model_script):
        assert '--vllm-commit "${EXPECTED_VLLM_COMMIT}"' in script
        assert '--sail-commit "${EXPECTED_SAIL_COMMIT}"' in script
    assert "PPU PIN UT SKIP" in pin_script

    # perf_regression.sh also verifies commits when the env vars are set
    perf_script = (ROOT / "scripts/ci/perf_regression.sh").read_text()
    assert "EXPECTED_VLLM_COMMIT" in perf_script
    assert "EXPECTED_SAIL_COMMIT" in perf_script
    assert "--vllm-commit" in perf_script
    assert "--sail-commit" in perf_script


def test_retry_reuses_build_artifacts_but_keeps_pin_diagnostics_per_attempt():
    for name, job_id in (
        ("_ppu-build.yaml", "build"),
        ("ppu-ops-tests.yaml", "ops-build"),
    ):
        uploads = {
            step["with"]["name"]: step["with"]
            for step in workflow(name)["jobs"][job_id]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        }
        assert uploads["ppu-wheels-${{ github.run_id }}"]["overwrite"] is True
        assert (
            uploads["ppu-build-diagnostics-${{ github.run_id }}"]["overwrite"] is True
        )
    pin = workflow("_ppu-pin-ut.yaml")["jobs"]["pin-unit-tests"]
    upload = next(
        step
        for step in pin["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert (
        upload["with"]["name"]
        == "ppu-pin-main-verified-${{ github.run_id }}-${{ github.run_attempt }}"
    )


def test_bare_cpu_matrix_is_on_demand_and_uses_the_exact_pr_head():
    config = workflow("ut.yaml")
    assert config["name"] == "CPU unit tests"
    assert set(config[True]) == {"workflow_call", "push", "workflow_dispatch"}
    assert config[True]["push"]["branches"] == ["main", "master"]
    assert set(config[True]["workflow_call"]["inputs"]) == {
        "pr_head_sha",
        "pr_number",
        "pr_head_ref",
    }
    assert config["permissions"] == {"contents": "read"}
    assert config["concurrency"] == {
        "group": "ut-${{ inputs.pr_head_sha && github.run_id || github.ref }}",
        "cancel-in-progress": True,
    }
    assert set(config["jobs"]) == {"cpu-unit-tests"}
    cpu = config["jobs"]["cpu-unit-tests"]
    assert cpu["runs-on"] == "ubuntu-latest"
    assert cpu["strategy"]["matrix"]["python-version"] == [
        "3.10",
        "3.11",
        "3.12",
        "3.13",
    ]
    assert "secrets" not in cpu
    expected_ref = "${{ inputs.pr_head_sha || github.sha }}"
    assert checkouts(cpu)[0]["with"] == {
        "ref": expected_ref,
        "persist-credentials": False,
    }
    for step in cpu["steps"]:
        assert "cache" not in step.get("with", {})
        if "pip install" in step.get("run", ""):
            assert step["run"] == (
                "python -m pip install --no-cache-dir -r requirements/dev.txt"
            )
    coverage = next(
        step for step in cpu["steps"] if step.get("uses") == "codecov/codecov-action@v5"
    )
    assert coverage["with"]["override_commit"] == expected_ref
    assert coverage["with"]["override_pr"] == "${{ inputs.pr_number }}"
    assert coverage["with"]["override_branch"] == (
        "${{ inputs.pr_head_ref || github.ref_name }}"
    )


def test_ops_partial_retry_downloads_diagnostics_from_successful_previous_attempt():
    build = workflow("ppu-ops-tests.yaml")["jobs"]
    producer = next(
        step["with"]["name"]
        for step in build["ops-build"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
        and step["with"]["name"].startswith("ppu-build-diagnostics-")
    )
    uploaded = producer.replace("${{ github.run_id }}", "100").replace(
        "${{ github.run_attempt }}", "1"
    )
    consumers = [
        build["select-ops-tests"],
        *workflow("_ppu-ops-tests.yaml")["jobs"].values(),
    ]
    for job in consumers:
        name = next(
            step["with"]["name"]
            for step in job["steps"]
            if step.get("uses") == "actions/download-artifact@v4"
            and step["with"].get("name", "").startswith("ppu-build-diagnostics-")
        )
        downloaded = (
            name.replace("${{ github.run_id }}", "100")
            .replace("${{ inputs.build_run_id }}", "100")
            .replace("${{ github.run_attempt }}", "2")
        )
        assert downloaded == uploaded


@pytest.mark.parametrize(
    "name", ["ppu-build-and-test.yaml", "_ppu-pin-ut.yaml", "_ppu-model-tests.yaml"]
)
def test_wheel_source_is_optional_internal_input_only(name):
    triggers = workflow(name)[True]
    source = triggers["workflow_call"]["inputs"].get("wheel_source_sha")
    assert source is not None
    assert source["type"] == "string"
    assert source["default"] == ""
    assert not source.get("required", False)
    assert "wheel_source_sha" not in triggers.get("workflow_dispatch", {}).get(
        "inputs", {}
    )


def test_all_local_reusable_calls_use_declared_inputs():
    for path in WORKFLOWS.iterdir():
        if path.suffix not in {".yaml", ".yml"}:
            continue
        for job_id, job in workflow(path.name)["jobs"].items():
            target = job.get("uses", "")
            if not target.startswith("./.github/workflows/"):
                continue
            declared = workflow(Path(target).name)[True]["workflow_call"].get(
                "inputs", {}
            )
            supplied = job.get("with", {})
            assert supplied.keys() <= declared.keys(), (path.name, job_id)
            required = {key for key, value in declared.items() if value.get("required")}
            assert required <= supplied.keys(), (path.name, job_id)


def expression(value, context):
    """Evaluate only the Boolean/context subset used by these workflow guards."""
    value = value.removeprefix("${{").removesuffix("}}").strip()
    value = re.sub(
        r"\b(?:inputs|github|needs|steps|vars)(?:\.[\w*-]+)+",
        lambda match: repr(context.get(match[0])),
        value,
    )
    value = value.replace("always()", "True").replace("cancelled()", "False")
    value = value.replace("contains(", "contains_value(")
    value = value.replace("&&", " and ").replace("||", " or ")
    value = re.sub(r"!(?!=)", "not ", value)
    value = re.sub(r"(?<!['\"])\btrue\b(?!['\"])", "True", value)
    value = re.sub(r"(?<!['\"])\bfalse\b(?!['\"])", "False", value)

    def contains_value(values, item):
        return item in values

    return eval(
        value.strip(),
        {"__builtins__": {}, "contains_value": contains_value},
        {},
    )


@pytest.mark.parametrize(
    "pr_mode,event",
    [
        (True, "workflow_dispatch"),
        (False, "workflow_dispatch"),
    ],
)
def test_e2e_concurrency_isolates_reuse_consumers_only_in_pr_mode(pr_mode, event):
    concurrency = workflow("_ppu-model-tests.yaml")["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert "queue" not in concurrency
    refs = ("refs/heads/main", "refs/heads/release")
    groups_by_ref = []
    for ref in refs:
        groups_by_attempt = []
        for attempt in (1, 2):
            groups = {}
            for run_id in ("901", "902", "903"):
                context = {
                    "inputs.pr_mode": pr_mode,
                    "inputs.build_run_id": "555",
                    "github.ref": ref,
                    "github.event_name": event,
                    "github.run_id": run_id,
                    "github.run_attempt": attempt,
                }
                groups[run_id] = re.sub(
                    r"\$\{\{.*?\}\}",
                    lambda match, context=context: str(expression(match[0], context)),
                    concurrency["group"],
                )
            # One running and two pending consumers must not share a PR-mode key.
            assert len(set(groups.values())) == (3 if pr_mode else 1)
            assert groups == {
                run_id: f"e2e-ppu-{ref}-{run_id if pr_mode else '555'}"
                for run_id in groups
            }
            groups_by_attempt.append(groups)
        assert groups_by_attempt[0] == groups_by_attempt[1]
        groups_by_ref.append(set(groups_by_attempt[0].values()))
    assert groups_by_ref[0].isdisjoint(groups_by_ref[1])


@pytest.fixture(scope="module")
def build_jobs():
    return workflow("ppu-build-and-test.yaml")["jobs"]


@pytest.mark.parametrize("build", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize(
    "run_id,source", [("", ""), ("123", "a" * 40), ("999", ""), ("", "a" * 40)]
)
def test_consumers_require_build_or_valid_reuse(build_jobs, build, run_id, source):
    context = {
        "github.event_name": "workflow_dispatch",
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": [],
        "inputs.skip_ppu": False,
        "inputs.existing_build_run_id": run_id,
        "inputs.wheel_source_sha": source,
        "needs.runtime-image.result": "success",
        "needs.runtime-image.outputs.upstream_reuse": "true",
        "needs.runtime-image.outputs.recipe_sanity": "true",
        "needs.build.result": build,
        "vars.PPU_RECIPE_PUBLIC_ENABLED": "true",
    }
    reuse = bool(run_id) and bool(source)
    eligible = build == "success" or (build == "skipped" and reuse)
    for job_id in ("pin-unit-tests", "recipe-sanity", "upstream-reuse"):
        assert bool(expression(build_jobs[job_id]["if"], context)) == eligible
    assert bool(expression(build_jobs["build"]["if"], context)) == (not run_id)
    # Gate must execute even after failed/cancelled/skipped dependencies.
    assert bool(expression(build_jobs["gate"]["if"], context)) is True


@pytest.mark.parametrize(
    "states",
    itertools.product(["success", "failure", "cancelled", "skipped"], repeat=4),
)
@pytest.mark.parametrize(
    "run_id,source", [("", ""), ("123", "a" * 40), ("999", ""), ("", "a" * 40)]
)
def test_gate_executes_fail_closed_truth_table(
    build_jobs, monkeypatch, states, run_id, source
):
    step = build_jobs["gate"]["steps"][0]
    context = {
        "inputs.skip_ppu": False,
        "inputs.skip_device_tests": False,
        "inputs.existing_build_run_id": run_id,
        "inputs.wheel_source_sha": source,
        "vars.PPU_RECIPE_PUBLIC_ENABLED": "true",
    }
    for key, value in step["env"].items():
        if key != "JOB_RESULTS":
            resolved = expression(value, context)
            monkeypatch.setenv(
                key, str(resolved).lower() if isinstance(resolved, bool) else resolved
            )
    results = {
        "runtime-image": {"result": "success"},
        "model-tests": {"result": "skipped"},
    }
    results.update(
        zip(
            (
                "build",
                "pin-unit-tests",
                "recipe-sanity",
                "upstream-reuse",
            ),
            ({"result": state} for state in states),
            strict=True,
        )
    )
    monkeypatch.setenv("JOB_RESULTS", json.dumps(results))
    script = step["run"].removeprefix("python - <<'PYTHON'\n").removesuffix("PYTHON\n")
    try:
        exec(compile(script, "ppu-ci-gate", "exec"), {})
        passed = True
    except SystemExit as error:
        assert error.code
        passed = False
    expected = (
        states[0] == "success"
        or (states[0] == "skipped" and bool(run_id) and bool(source))
    ) and states[1:] == ("success", "success", "success")
    assert passed is expected


@pytest.mark.parametrize("present", [True, False])
def test_runtime_preflight_detects_the_upstream_reuse_entry_script(
    build_jobs, tmp_path, present
):
    step = next(
        step
        for step in build_jobs["runtime-image"]["steps"]
        if step.get("id") == "reuse"
    )
    assert step["env"] == {"SOURCE_DIR": "${{ steps.workspace.outputs.source }}"}
    if present:
        (tmp_path / "scripts/ci").mkdir(parents=True)
        (tmp_path / "scripts/ci/ppu_upstream_reuse.sh").write_text("")
    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={**os.environ, "SOURCE_DIR": str(tmp_path), "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"supported={str(present).lower()}\n"


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "scripts/ci/ppu_e2e_recipe_sanity.sh",
        "scripts/ci/ppu_recipe.py",
        "scripts/ci/ppu_recipe_runner.py",
        "scripts/ci/ppu_recipe_retention.py",
        "scripts/ci/ppu_e2e_select.py",
        "tests/ppu/recipes",
        "tests/ppu/policy/models",
        "selector dispatch",
        "runtime dispatch",
    ],
)
def test_recipe_capability_probes_candidate_checkout(build_jobs, tmp_path, missing):
    step = next(
        step
        for step in build_jobs["runtime-image"]["steps"]
        if step.get("id") == "recipe"
    )
    assert step["env"] == {"SOURCE_DIR": "${{ steps.workspace.outputs.source }}"}
    for directory in ("scripts/ci", "tests/ppu/recipes", "tests/ppu/policy/models"):
        if directory != missing:
            (tmp_path / directory).mkdir(parents=True)
    for script in (
        "ppu_e2e_recipe_sanity.sh",
        "ppu_recipe.py",
        "ppu_recipe_runner.py",
        "ppu_recipe_retention.py",
        "ppu_e2e_select.py",
        "ppu_runtime_container.sh",
    ):
        relative = Path("scripts/ci") / script
        if str(relative) == missing:
            continue
        candidate = (ROOT / relative).read_text()
        if script == "ppu_e2e_select.py" and missing == "selector dispatch":
            candidate = '# A future version may support "--recipes"\n'
        if script == "ppu_runtime_container.sh" and missing == "runtime dispatch":
            candidate = "# An older runtime with no recipe-sanity mode.\n"
        (tmp_path / relative).write_text(candidate)

    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={**os.environ, "SOURCE_DIR": str(tmp_path), "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    supported = str(missing is None).lower()
    assert output.read_text() == f"supported={supported}\n"
    context = {
        "vars.PPU_RECIPE_PUBLIC_ENABLED": "true",
        "github.event_name": "workflow_dispatch",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "success",
        "needs.runtime-image.outputs.recipe_sanity": supported,
        "needs.build.result": "success",
    }
    assert bool(expression(build_jobs["recipe-sanity"]["if"], context)) == (
        missing is None
    )
    assert bool(expression(build_jobs["model-tests"]["if"], context)) == (
        missing is not None
    )


@pytest.mark.parametrize(
    "outputs,outcome,legacy_outcome,expected_error",
    [
        ({"recipe_sanity": "false"}, "skipped", "success", None),
        ({"recipe_sanity": "false"}, "skipped", "skipped", "model-tests"),
        ({"recipe_sanity": "false"}, "skipped", "failure", "model-tests"),
        ({"recipe_sanity": "false"}, "failure", "success", "recipe-sanity"),
        ({"recipe_sanity": "true"}, "success", "skipped", None),
        ({"recipe_sanity": "true"}, "skipped", "skipped", "recipe-sanity"),
        ({"recipe_sanity": "true"}, "failure", "skipped", "recipe-sanity"),
        ({"recipe_sanity": ""}, "skipped", "skipped", "recipe-sanity"),
        ({}, "skipped", "skipped", "recipe-sanity"),
    ],
)
def test_recipe_gate_accepts_skip_only_for_unsupported_source(
    build_jobs, outputs, outcome, legacy_outcome, expected_error
):
    script = build_jobs["gate"]["steps"][0]["run"].replace(
        "python - <<'PYTHON'", f"{sys.executable} - <<'PYTHON'"
    )
    results = {key: {"result": "success"} for key in build_jobs["gate"]["needs"]}
    results["runtime-image"]["outputs"] = outputs
    results["recipe-sanity"]["result"] = outcome
    results["model-tests"]["result"] = legacy_outcome
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env={
            **os.environ,
            "JOB_RESULTS": json.dumps(results),
            "SKIP_PPU": "false",
            "SKIP_DEVICE_TESTS": "false",
            "RECIPE_PUBLIC_ENABLED": "true",
            "EXISTING_BUILD_RUN_ID": "",
            "WHEEL_SOURCE_SHA": "",
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (expected_error is None), result.stderr
    if expected_error is None and outcome == "skipped":
        assert "source does not support Recipe Fast Sanity" in result.stdout
    if expected_error:
        assert f"{expected_error} did not succeed" in result.stderr


@pytest.mark.parametrize("supported", ["true", "false", None])
def test_upstream_reuse_lane_requires_the_entry_script(build_jobs, supported):
    context = {
        "github.event_name": "workflow_dispatch",
        "inputs.skip_ppu": False,
        "inputs.skip_device_tests": False,
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "success",
        "needs.runtime-image.outputs.upstream_reuse": supported,
        "needs.build.result": "success",
    }
    assert bool(expression(build_jobs["upstream-reuse"]["if"], context)) == (
        supported == "true"
    )
    # Only the reuse lane depends on the entry script.
    context["vars.PPU_RECIPE_PUBLIC_ENABLED"] = "true"
    context["needs.runtime-image.outputs.recipe_sanity"] = "true"
    for job_id in ("pin-unit-tests", "recipe-sanity"):
        assert expression(build_jobs[job_id]["if"], context)


@pytest.mark.parametrize(
    "outputs,lanes,passed",
    [
        ({"upstream_reuse": "false"}, ("success", "skipped"), True),
        ({"upstream_reuse": "false"}, ("success", "failure"), False),
        ({"upstream_reuse": "false"}, ("skipped", "skipped"), False),
        ({"upstream_reuse": "true"}, ("success", "skipped"), False),
        ({"upstream_reuse": ""}, ("success", "skipped"), False),
        ({}, ("success", "skipped"), False),
    ],
)
def test_gate_accepts_skipped_reuse_only_without_the_entry_script(
    build_jobs, outputs, lanes, passed
):
    script = build_jobs["gate"]["steps"][0]["run"].replace(
        "python - <<'PYTHON'", f"{sys.executable} - <<'PYTHON'"
    )
    results = {
        "runtime-image": {"result": "success", "outputs": outputs},
        "build": {"result": "success"},
        "model-tests": {"result": "skipped"},
        "recipe-sanity": {"result": "success"},
    }
    results.update(
        (job, {"result": state})
        for job, state in zip(("pin-unit-tests", "upstream-reuse"), lanes, strict=True)
    )
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env={
            **os.environ,
            "JOB_RESULTS": json.dumps(results),
            "SKIP_PPU": "false",
            "SKIP_DEVICE_TESTS": "false",
            "EXISTING_BUILD_RUN_ID": "",
            "WHEEL_SOURCE_SHA": "",
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is passed, result.stderr


@pytest.mark.parametrize("enabled", ["", "false", "true", "typo"])
def test_recipe_public_flag_controls_callers_and_gate(build_jobs, enabled):
    context = {
        "vars.PPU_RECIPE_PUBLIC_ENABLED": enabled,
        "github.event_name": "workflow_dispatch",
        "inputs.existing_build_run_id": "",
        "inputs.wheel_source_sha": "",
        "needs.runtime-image.result": "success",
        "needs.runtime-image.outputs.recipe_sanity": "true",
        "needs.build.result": "success",
        "needs.select.result": "success",
        "needs.select.outputs.recipe_models": "glm-5.3",
    }
    fast_manual = workflow("ppu-recipe-fast-sanity.yaml")["jobs"]
    fast_leaf = workflow("_ppu-recipe-fast-sanity.yaml")["jobs"]
    model_leaf = workflow("_ppu-recipe-model-sanity.yaml")["jobs"]
    accuracy_nightly = workflow("ppu-accuracy-nightly.yaml")["jobs"]
    for job in (
        fast_manual["build"],
        fast_manual["recipe-fast-sanity"],
        model_leaf["prepare"],
        accuracy_nightly["recipe-model-sanity"],
    ):
        assert bool(expression(job["if"], context)) == (enabled == "true")
    assert bool(expression(fast_leaf["prepare"]["if"], context)) == (
        enabled in ("", "true")
    )
    for labels in ([], ["ci-preview"]):
        preview = {
            **context,
            "github.event_name": "pull_request",
            "github.event.action": "synchronize",
            "github.event.pull_request.draft": False,
            "github.event.pull_request.labels.*.name": labels,
        }
        assert bool(expression(build_jobs["recipe-sanity"]["if"], preview)) == (
            "ci-preview" in labels and enabled in ("", "true")
        )
    assert (
        fast_manual["recipe-fast-sanity"]["with"]["wheel_source_sha"]
        == "${{ github.sha }}"
    )
    assert (
        fast_manual["recipe-fast-sanity"]["with"]["build_run_id"]
        == "${{ github.run_id }}"
    )
    assert (
        accuracy_nightly["recipe-model-sanity"]["with"]["wheel_source_sha"]
        == "${{ github.sha }}"
    )
    assert (
        accuracy_nightly["recipe-model-sanity"]["with"]["build_run_id"]
        == "${{ github.run_id }}"
    )
    gate = build_jobs["gate"]["steps"][0]["run"].replace(
        "python - <<'PYTHON'", f"{sys.executable} - <<'PYTHON'"
    )
    results = {key: {"result": "success"} for key in build_jobs["gate"]["needs"]}
    env = {
        **os.environ,
        "SKIP_PPU": "false",
        "SKIP_DEVICE_TESTS": "false",
        "EXISTING_BUILD_RUN_ID": "",
        "WHEEL_SOURCE_SHA": "",
    }
    for outcome in ("skipped", "success", "failure"):
        results["recipe-sanity"]["result"] = outcome
        result = subprocess.run(
            ["bash", "-e", "-c", gate],
            env={**env, "JOB_RESULTS": json.dumps(results)},
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == (outcome == "success")


def test_recipe_model_sanity_has_one_scheduled_entry():
    callers = []
    for path in WORKFLOWS.iterdir():
        if path.suffix not in {".yaml", ".yml"}:
            continue
        config = workflow(path.name)
        if "schedule" not in config[True]:
            continue
        if any(
            job.get("uses") == "./.github/workflows/_ppu-recipe-model-sanity.yaml"
            for job in config["jobs"].values()
        ):
            callers.append(path.name)
    assert callers == ["ppu-accuracy-nightly.yaml"]


@pytest.mark.parametrize(
    "event,models,enabled,legacy_cases,recipe_slugs,error",
    [
        ("schedule", "", "true", None, "", ""),
        ("schedule", "", "TRUE", None, "", ""),
        ("schedule", "", "false", None, None, ""),
        ("schedule", "", "", None, None, ""),
        ("workflow_dispatch", "", "true", None, None, ""),
        (
            "workflow_dispatch",
            "glm-5.3-mxfp4-tp4",
            "true",
            ["glm-5.3-mxfp4-tp4"],
            None,
            "",
        ),
        (
            "workflow_dispatch",
            "recipe:glm-5.3, deepseek-v4-flash",
            "true",
            [],
            "glm-5.3,deepseek-v4-flash",
            "",
        ),
        (
            "workflow_dispatch",
            "recipe:glm-5.3",
            "",
            [],
            None,
            "Recipe accuracy is disabled",
        ),
        (
            "workflow_dispatch",
            "recipe:glm-5.3",
            "TRUE",
            [],
            None,
            "Recipe accuracy is disabled",
        ),
        (
            "workflow_dispatch",
            "recipe: ,",
            "true",
            [],
            None,
            "nonempty Recipe policy slugs",
        ),
        ("workflow_dispatch", "recipe:glm-5.3,glm-5.3", "true", [], None, "distinct"),
        (
            "workflow_dispatch",
            "recipe:typo",
            "true",
            [],
            None,
            "unknown recipe policy key",
        ),
    ],
)
def test_accuracy_nightly_routes_selected_models(
    tmp_path, event, models, enabled, legacy_cases, recipe_slugs, error
):
    jobs = workflow("ppu-accuracy-nightly.yaml")["jobs"]
    selector = next(
        step for step in jobs["select"]["steps"] if step.get("id") == "select"
    )
    output = tmp_path / "github-output"
    assert selector["env"]["SCHEDULED_RECIPE_ENABLED"] == (
        "${{ github.event_name == 'schedule' && vars.PPU_RECIPE_PUBLIC_ENABLED == 'true' }}"
    )
    assert (
        "vars.PPU_RECIPE_PUBLIC_ENABLED == 'true'" in jobs["recipe-model-sanity"]["if"]
    )
    recipe_enabled = enabled.lower() == "true"
    script = selector["run"].replace("python3 -", f'"{sys.executable}" -', 1)
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "MODELS": models,
            "RECIPE_PUBLIC_ENABLED": enabled,
            "SCHEDULED_RECIPE_ENABLED": str(
                event == "schedule" and recipe_enabled
            ).lower(),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert jobs["build"]["needs"] == "select"
    if error:
        assert result.returncode != 0 and error in result.stderr
        assert not output.exists()
        return
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    legacy_matrix = values.get("matrix", "")
    context = {
        "github.event_name": event,
        "vars.PPU_RECIPE_PUBLIC_ENABLED": str(recipe_enabled).lower(),
        "needs.build.result": "success",
        "needs.select.result": "success",
        "needs.select.outputs.recipe_models": values.get("recipe_models", ""),
        "needs.select.outputs.matrix": legacy_matrix,
    }
    assert expression(jobs["accuracy"]["if"], context) == (legacy_cases != [])
    if legacy_cases is None:
        catalog = json.loads((ROOT / "scripts/ci/ppu_e2e_models.json").read_text())
        expected = [
            entry["key"] for entry in catalog if entry["category"] == "accuracy"
        ]
        if event == "schedule" and recipe_enabled:
            claimed = {
                case_id
                for path in (ROOT / "tests/ppu/policy/models").glob("*.yaml")
                for case_id in yaml.safe_load(path.read_text()).get(
                    "replaces_legacy_cases", []
                )
            }
            expected = [case_id for case_id in expected if case_id not in claimed]
        assert [
            entry["model"] for entry in json.loads(legacy_matrix)["include"]
        ] == expected
    elif legacy_cases:
        assert [
            entry["model"] for entry in json.loads(legacy_matrix)["include"]
        ] == legacy_cases
    recipe = jobs["recipe-model-sanity"]
    assert "accuracy" in recipe["needs"]
    assert bool(expression(recipe["if"], context)) == (recipe_slugs is not None)
    assert recipe["with"]["models"] == "${{ needs.select.outputs.recipe_models }}"
    assert values.get("recipe_models", "") == (recipe_slugs or "")


@pytest.mark.parametrize(
    "leaf_name,pr_mode,tier,build_run,allowed",
    [
        ("_ppu-recipe-model-sanity.yaml", "false", "model_sanity", "101", True),
        ("_ppu-recipe-model-sanity.yaml", "false", "model_sanity", "100", False),
        ("_ppu-recipe-fast-sanity.yaml", "false", "fast_sanity", "100", False),
        ("_ppu-recipe-fast-sanity.yaml", "true", "fast_sanity", "100", True),
        ("_ppu-recipe-model-sanity.yaml", "true", "model_sanity", "101", False),
    ],
)
def test_recipe_workflow_refuses_arbitrary_wheel_run(
    leaf_name, pr_mode, tier, build_run, allowed
):
    prepare = workflow(leaf_name)["jobs"]["prepare"]
    select = next(step for step in prepare["steps"] if step.get("id") == "select")
    script = select["run"].partition("python3 scripts/ci/ppu_e2e_select.py")[0]
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env={
            **os.environ,
            "TIER": tier,
            "PR_MODE": pr_mode,
            "BUILD_RUN_ID": build_run,
            "GITHUB_RUN_ID": "101",
        },
    )
    assert (result.returncode == 0) is allowed


@pytest.mark.parametrize(
    "leaf_name,tier",
    [
        ("_ppu-recipe-fast-sanity.yaml", "fast_sanity"),
        ("_ppu-recipe-model-sanity.yaml", "model_sanity"),
    ],
)
@pytest.mark.parametrize("exit_code,status", [(0, "pass"), (3, "fail")])
@pytest.mark.parametrize(
    "retention_case",
    [
        "configured",
        "shared_root",
        "writable_root",
        "public_subtree",
        "unset",
        "empty",
        "copy_error",
        "retry",
        "unsafe_source",
        "ephemeral",
    ],
)
def test_recipe_retention_and_publication(
    tmp_path, leaf_name, tier, exit_code, status, retention_case
):
    steps = workflow(leaf_name)["jobs"]["sanity"]["steps"]
    runtime = next(step for step in steps if step.get("id") == "recipe")
    retain = next(step for step in steps if step.get("id") == "retain")
    publish = next(step for step in steps if step.get("id") == "publish")
    upload = next(
        step for step in steps if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert upload["with"]["path"] == "${{ steps.runtime.outputs.public }}/summary.json"
    assert steps.index(runtime) < steps.index(retain) < steps.index(publish)
    assert retain["if"] == "always() && steps.runtime.outcome == 'success'"
    assert retain["env"]["TIER"] == tier
    assert all(
        "PPU_CI_PRIVATE_RESULTS_ROOT" not in step.get("env", {}) for step in steps
    )
    assert (
        not {"PPU_CI_HF_HOME", "GSM8K_DATASET_PATH", "PPU_CI_DG_CACHE_DIR"}
        & runtime["env"].keys()
    )

    launcher_path = tmp_path / "scripts/ci/ppu_runtime_container.sh"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text(
        'echo "synthetic-private-stdout"\necho "synthetic-private-stderr" >&2\n'
        'mkdir "$PPU_CI_ARTIFACT_DIR/results"\n'
        'printf "%s\\n" "$FAKE_PRIVATE_SUMMARY" > "$PPU_CI_ARTIFACT_DIR/results/summary.json"\n'
        f"exit {exit_code}\n"
    )
    private_dir, public_dir = tmp_path / "private", tmp_path / "public"
    private_dir.mkdir()
    public_dir.mkdir()
    runner_temp = tmp_path / "runner-temp"
    retention_root = (
        runner_temp if retention_case == "ephemeral" else tmp_path
    ) / "mock-intranet-retention"
    retention_root.mkdir(parents=True)
    retention_root.chmod(
        {"shared_root": 0o755, "writable_root": 0o777}.get(retention_case, 0o700)
    )
    if retention_case in {"shared_root", "public_subtree"}:
        subtree = retention_root / "vllm-sail"
        subtree.mkdir()
        subtree.chmod(0o700 if retention_case == "shared_root" else 0o755)
    attempt_dir = retention_root / "vllm-sail" / "101-2" / tier
    retained_dir = attempt_dir / "glm-5.3"
    if retention_case == "retry":
        for directory in (
            retention_root / "vllm-sail",
            attempt_dir.parent,
            attempt_dir,
            retained_dir,
        ):
            directory.mkdir(mode=0o700)
        (retained_dir / "index.json").write_text("original index")
    job_summary = tmp_path / "job-summary.md"
    private_summary = {
        "path": str(retention_root),
        "accuracy": {
            "score": 0.96,
            "num": 1319,
            "expected_num": 1319,
            "golden_score": 0.97,
            "minimum_score": 0.95,
            "maximum_score": 0.99,
            "within_band": True,
            "error": "synthetic-private-error",
            "per_question": {
                "status": "baseline_established",
                "source_file": "synthetic-private-review.jsonl",
            },
        },
        "micro_e2e_passed": exit_code == 0,
        "bench": {"c1": {"status": "synthetic-private-bench"}},
    }
    launch_result = subprocess.run(
        ["bash", "-e", "-c", runtime["run"]],
        env={
            **os.environ,
            "PPU_CI_SOURCE_DIR": str(tmp_path),
            "PPU_CI_ARTIFACT_DIR": str(private_dir),
            "FAKE_PRIVATE_SUMMARY": json.dumps(private_summary),
        },
        capture_output=True,
        text=True,
    )
    assert launch_result.returncode == exit_code
    assert not launch_result.stdout and not launch_result.stderr
    assert (private_dir / "launcher.log").read_text().splitlines() == [
        "synthetic-private-stdout",
        "synthetic-private-stderr",
    ]
    results_dir = private_dir / "results"
    if retention_case == "unsafe_source":
        (private_dir / "provenance.json").symlink_to(results_dir / "summary.json")
    elif retention_case != "copy_error":
        (private_dir / "provenance.json").write_text("private provenance")
    retained_sources = {
        "summary.json": results_dir / "summary.json",
        "provenance.json": private_dir / "provenance.json",
        "launcher.log": private_dir / "launcher.log",
    }
    if tier == "model_sanity":
        per_question = results_dir / "gsm8k_per_question.jsonl"
        per_question.write_text("private scores")
        retained_sources[per_question.name] = per_question
    (private_dir / "summary.json").write_text('{"accuracy":{"score":0.01}}')
    runtime_log = private_dir / "runtime.log"
    runtime_log.write_text(f"{retention_root}\n")
    retained_sources[runtime_log.name] = runtime_log
    phase_file = private_dir / "recipe-phase.txt"
    phase_file.write_text("micro_e2e\n" if tier == "model_sanity" else "chat_probe\n")
    retained_sources[phase_file.name] = phase_file
    (private_dir / "unlisted.log").write_text("not on the retention allowlist")
    outputs = tmp_path / "retention-output"
    retention_env = os.environ.copy()
    if retention_case == "unset":
        retention_env.pop("PPU_CI_PRIVATE_RESULTS_ROOT", None)
    else:
        retention_env["PPU_CI_PRIVATE_RESULTS_ROOT"] = (
            "" if retention_case == "empty" else str(retention_root)
        )
    retain_result = subprocess.run(
        ["bash", "-e", "-c", retain["run"]],
        env={
            **retention_env,
            "PPU_CI_SOURCE_DIR": str(ROOT),
            "PPU_CI_SOURCE_SHA": "a" * 40,
            "PPU_CI_ARTIFACT_DIR": str(private_dir),
            "TIER": tier,
            "MODEL_KEY": "glm-5.3",
            "RECIPE_OUTCOME": "failure" if exit_code else "success",
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_WORKSPACE": str(tmp_path / "workspace"),
            "GITHUB_RUN_ID": "101",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_OUTPUT": str(outputs),
        },
        capture_output=True,
        text=True,
    )
    expected_retention = {
        "configured": "retained",
        "shared_root": "retained",
        "unset": "not_configured",
        "empty": "not_configured",
    }.get(
        retention_case,
        "retained" if retention_case == "copy_error" and exit_code else "failed",
    )
    retention = outputs.read_text().removeprefix("retention=").strip()
    assert retention == expected_retention
    expected_error = expected_retention == "failed"
    assert retain_result.returncode == int(expected_error)
    assert retain_result.stderr == ""
    if expected_retention == "retained":
        if retention_case == "copy_error":
            del retained_sources["provenance.json"]
        assert {
            file.name for file in retained_dir.iterdir()
        } == retained_sources.keys() | {"index.json"}
        assert json.loads((retained_dir / "index.json").read_text()) == {
            "checkout_sha": "a" * 40,
            "tier": tier,
            "model_key": "glm-5.3",
            "run_id": "101",
            "attempt": "2",
            "files": {
                name: hashlib.sha256(source.read_bytes()).hexdigest()
                for name, source in retained_sources.items()
            },
        }
        for entry in retention_root.rglob("*"):
            assert stat.S_IMODE(entry.stat().st_mode) == (
                0o700 if entry.is_dir() else 0o600
            )
        for name, source in retained_sources.items():
            assert (retained_dir / name).read_bytes() == source.read_bytes()
    elif retention_case == "retry":
        assert (retained_dir / "index.json").read_text() == "original index"
    else:
        assert not retained_dir.exists()
    assert not list(retention_root.rglob(".staging-*"))
    if expected_retention == "not_configured":
        assert (
            retain_result.stdout
            == "::warning::private results retention not configured\n"
        )
    else:
        assert retain_result.stdout == ""

    publish_result = subprocess.run(
        ["bash", "-e", "-c", publish["run"]],
        env={
            **os.environ,
            "RECIPE_OUTCOME": "failure" if exit_code else "success",
            "RETENTION_OUTCOME": "failure" if retain_result.returncode else "success",
            "RETENTION": retention,
            "MODEL_KEY": "glm-5.3",
            "TIER": tier,
            "PPU_CI_SOURCE_DIR": str(ROOT),
            "PRIVATE_DIR": str(private_dir),
            "PUBLIC_DIR": str(public_dir),
            "GITHUB_STEP_SUMMARY": str(job_summary),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    if expected_error:
        status = "fail"
    public_summary = (public_dir / "summary.json").read_text()
    reason = (
        {"model_sanity": "micro_e2e_failed", "fast_sanity": "chat_probe_failed"}[tier]
        if exit_code
        else "not_applicable"
    )
    assert publish_result.stdout == (
        f"Recipe sanity: {status}\nretention: {retention}\nfailure_reason: {reason}\n"
    )
    assert not publish_result.stderr
    expected_public = {
        "status": status,
        "retention": retention,
        "failure_reason": reason,
    }
    if tier == "model_sanity":
        micro_e2e = "fail" if exit_code else "pass"
        expected_public.update(
            model_key="glm-5.3",
            gsm8k={
                "score": 0.96,
                "evaluated_samples": 1319,
                "expected_samples": 1319,
                "golden_score": 0.97,
                "minimum_score": 0.95,
                "maximum_score": 0.99,
                "gate": "pass",
            },
            baseline_completeness="baseline_established",
            micro_e2e=micro_e2e,
        )
        assert f"**Result:** {status}" in job_summary.read_text()
        assert f"**Private retention:** {retention}" in job_summary.read_text()
        assert f"**Recipe failure reason:** {reason}" in job_summary.read_text()
        assert "96.00% (golden: 97.00%)" in job_summary.read_text()
        assert "1319 / 1319" in job_summary.read_text()
        assert "min: 95.00%, max: 99.00%" in job_summary.read_text()
        assert f"**Micro-E2E:** {micro_e2e}" in job_summary.read_text()
    else:
        assert job_summary.read_text() == publish_result.stdout
    assert json.loads(public_summary) == expected_public
    for public_output in (
        retain_result.stdout + retain_result.stderr + publish_result.stdout,
        job_summary.read_text(),
        public_summary,
    ):
        assert str(retention_root) not in public_output
        assert "mock-intranet-retention" not in public_output
        assert "synthetic-private-" not in public_output


@pytest.mark.parametrize(
    "tier,server_label,error,reason",
    [
        ("fast_sanity", None, None, "runtime_setup_failed"),
        (
            "model_sanity",
            "model_sanity",
            "The size of tensor a (2) must match the size of tensor b (4)"
            " at non-singleton dimension 1",
            "mla_metadata_shape_mismatch",
        ),
        (
            "model_sanity",
            "model_sanity",
            "The expanded size of the tensor (2) must match the existing size (4)"
            " at non-singleton dimension 1. Target sizes: [313, 2]. Tensor sizes: [630, 4]",
            "mla_metadata_shape_mismatch",
        ),
    ],
)
def test_failed_recipe_retains_logs_without_summary(
    tmp_path, tier, server_label, error, reason
):
    leaf = f"_ppu-recipe-{tier.replace('_', '-')}.yaml"
    steps = workflow(leaf)["jobs"]["sanity"]["steps"]
    retain = next(step for step in steps if step.get("id") == "retain")
    publish = next(step for step in steps if step.get("id") == "publish")
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    retention_root = tmp_path / "synthetic-private-retention"
    retention_root.mkdir(mode=0o700)
    logs = {"launcher.log": f"synthetic-private-launcher {retention_root}\n"}
    if server_label:
        (private_dir / "results").mkdir()
        logs["runtime.log"] = f"synthetic-private-runtime {retention_root}\n"
        logs["recipe-phase.txt"] = "server_startup\n"
        logs[f"results/vllm_serve_{server_label}.log"] = (
            f'File "{retention_root}/mla_indexer.py", line 315\n'
            "    self.scheduler_metadata_buffer.copy_(schedule_metadata)\n"
            f"RuntimeError: {error}\n"
        )
    for name, content in logs.items():
        (private_dir / name).write_text(content)
    outputs = tmp_path / "retention-output"
    summary = tmp_path / "job-summary.md"
    env = {
        **os.environ,
        "PPU_CI_SOURCE_DIR": str(ROOT),
        "PPU_CI_SOURCE_SHA": "a" * 40,
        "PPU_CI_ARTIFACT_DIR": str(private_dir),
        "PPU_CI_PRIVATE_RESULTS_ROOT": str(retention_root),
        "TIER": tier,
        "MODEL_KEY": "test-model",
        "RUNNER_TEMP": str(tmp_path / "runner-temp"),
        "GITHUB_WORKSPACE": str(tmp_path / "workspace"),
        "GITHUB_RUN_ID": "101",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(outputs),
        "RECIPE_OUTCOME": "failure",
    }
    retained = subprocess.run(
        ["bash", "-e", "-c", retain["run"]],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert outputs.read_text() == "retention=retained\n"
    destination = retention_root / "vllm-sail" / "101-2" / tier / "test-model"
    index = json.loads((destination / "index.json").read_text())
    assert index["files"] == {
        Path(name).name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in logs.items()
    }
    assert not (destination / "summary.json").exists()
    for name, value in logs.items():
        assert (destination / Path(name).name).read_text() == value
    published = subprocess.run(
        ["bash", "-e", "-c", publish["run"]],
        env={
            **env,
            "RETENTION_OUTCOME": "success",
            "RETENTION": "retained",
            "PRIVATE_DIR": str(private_dir),
            "PUBLIC_DIR": str(tmp_path),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    public_json = (tmp_path / "summary.json").read_text()
    assert json.loads(public_json)["failure_reason"] == reason
    assert json.loads(public_json)["status"] == "fail"
    assert f"failure_reason: {reason}" in published.stdout
    assert reason in summary.read_text()
    assert "synthetic-private-" not in (
        retained.stdout
        + retained.stderr
        + published.stdout
        + published.stderr
        + public_json
        + summary.read_text()
    )


@pytest.mark.parametrize(
    "private_summary",
    [
        None,
        b'{"token":"synthetic-private-value"',
        b"\xffsynthetic-private-value",
        b'{"accuracy":{"score":NaN,"num":true,"expected_num":1319.5,'
        b'"within_band":"pass","golden_score":"synthetic-private-value",'
        b'"per_question":{"status":"synthetic-private-value"}},'
        b'"micro_e2e_passed":"pass"}',
    ],
)
def test_nightly_public_summary_drops_missing_and_unsafe_metrics(
    tmp_path, private_summary
):
    steps = workflow("_ppu-recipe-model-sanity.yaml")["jobs"]["sanity"]["steps"]
    publish = next(step for step in steps if step.get("id") == "publish")
    private_dir = tmp_path / "private"
    results_dir = private_dir / "results"
    results_dir.mkdir(parents=True)
    if private_summary is not None:
        (results_dir / "summary.json").write_bytes(private_summary)
    job_summary = tmp_path / "job-summary.md"
    result = subprocess.run(
        ["bash", "-e", "-c", publish["run"]],
        env={
            **os.environ,
            "RECIPE_OUTCOME": "failure",
            "RETENTION_OUTCOME": "failure",
            "RETENTION": "failed",
            "MODEL_KEY": "deepseek-v4-flash",
            "TIER": "model_sanity",
            "PPU_CI_SOURCE_DIR": str(ROOT),
            "PRIVATE_DIR": str(private_dir),
            "PUBLIC_DIR": str(tmp_path),
            "GITHUB_STEP_SUMMARY": str(job_summary),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    public_summary = (tmp_path / "summary.json").read_text()
    public = json.loads(public_summary)
    assert public["status"] == "fail"
    assert public["model_key"] == "deepseek-v4-flash"
    assert public["micro_e2e"] == public["baseline_completeness"] == "not_available"
    assert public["gsm8k"].pop("gate") == "not_available"
    assert all(value is None for value in public["gsm8k"].values())
    assert "synthetic-private-value" not in (
        result.stdout + result.stderr + job_summary.read_text() + public_summary
    )


def test_runtime_workflows_use_the_checked_in_docker_wrapper():
    pin = workflow("_ppu-pin-ut.yaml")["jobs"]["pin-unit-tests"]
    model = workflow("_ppu-model-tests.yaml")["jobs"]["e2e"]
    pin_step = next(
        step
        for step in pin["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    model_step = next(
        step
        for step in model["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    assert (
        pin_step["run"]
        == 'bash "${PPU_CI_SOURCE_DIR}/scripts/ci/ppu_runtime_container.sh" pin-ut'
    )
    assert pin_step["env"]["PPU_CI_DEVICE_COUNT"] == "1"
    assert "PPU_RUNTIME_IMAGE" not in pin_step["env"]
    assert pin_step["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
    assert (
        model_step["run"]
        == 'bash "${PPU_CI_SOURCE_DIR}/scripts/ci/ppu_runtime_container.sh" model-test'
    )
    assert model_step["env"]["PPU_CI_DEVICE_COUNT"] == "${{ matrix.model.ppu_count }}"
    # Smoke cases keep their parallelism inside one process, so the tensor
    # parallel size is the card count they asked for.
    assert model_step["env"]["TP_SIZE"] == "${{ matrix.model.ppu_count }}"
    assert model_step["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
    assert pin_step["env"]["PPU_CI_SOURCE_DIR"] == "${{ steps.runtime.outputs.source }}"
    assert (
        model_step["env"]["PPU_CI_SOURCE_DIR"] == "${{ steps.runtime.outputs.source }}"
    )
    assert model_step["env"]["MODEL_ID"] == "${{ matrix.model.model_id }}"
    assert "MODEL_PATH" not in model_step["env"]
    assert "ppu-scheduler-action" not in (WORKFLOWS / "_ppu-pin-ut.yaml").read_text()
    assert (
        "ppu-scheduler-action" not in (WORKFLOWS / "_ppu-model-tests.yaml").read_text()
    )
    kperf = workflow("_ppu-kernel-perf.yaml")["jobs"]["kernel-perf"]
    kperf_step = next(
        step
        for step in kperf["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    assert (
        kperf_step["run"]
        == 'bash "${PPU_CI_SOURCE_DIR}/scripts/ci/ppu_runtime_container.sh" kernel-perf'
    )
    assert kperf_step["env"]["PPU_CI_DEVICE_COUNT"] == "8"
    assert kperf_step["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
    assert kperf_step["env"]["PPU_CI_WHEEL_SOURCE_SHA"] == (
        "${{ inputs.wheel_source_sha }}"
    )
    assert kperf_step["env"]["SUITE"] == "kernel"
    assert (
        kperf_step["env"]["PPU_CI_SOURCE_DIR"] == "${{ steps.runtime.outputs.source }}"
    )
    assert (
        "ppu-scheduler-action" not in (WORKFLOWS / "_ppu-kernel-perf.yaml").read_text()
    )


@pytest.mark.parametrize(
    "name,job_id,mode",
    [
        ("_ppu-pin-ut.yaml", "pin-unit-tests", "pin-ut"),
        ("_ppu-upstream-reuse.yaml", "upstream-reuse", "upstream-reuse"),
    ],
)
def test_ut_and_reuse_leaves_prepare_verified_upstream_source(name, job_id, mode):
    job = workflow(name)["jobs"][job_id]
    source = checkouts(job)[1]
    assert source["with"] == {
        "repository": "vllm-project/vllm",
        "ref": "${{ steps.upstream.outputs.ref }}",
        "path": "${{ steps.runtime.outputs.upstream_rel }}",
        "persist-credentials": False,
    }
    verify = job["steps"][job["steps"].index(source) + 1]
    assert verify["env"]["EXPECTED_SHA"] == source["with"]["ref"]
    assert verify["env"]["UPSTREAM_DIR"] == "${{ steps.runtime.outputs.upstream }}"
    assert 'rev-parse HEAD)" = "$EXPECTED_SHA"' in verify["run"]
    runtime = next(
        step
        for step in job["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    assert (
        runtime["env"]["PPU_CI_UPSTREAM_DIR"] == "${{ steps.runtime.outputs.upstream }}"
    )
    assert runtime["run"].endswith(f'" {mode}')


def test_upstream_reuse_only_runs_in_the_authorized_pipeline():
    texts = {path.name: path.read_text() for path in WORKFLOWS.glob("*.y*ml")}
    callers = {
        name for name, text in texts.items() if "_ppu-upstream-reuse.yaml" in text
    }
    assert callers == {"ppu-build-and-test.yaml"}
    runners = {
        name
        for name, text in texts.items()
        if "tests/upstream_reuse" in text
        or 'ppu_runtime_container.sh" upstream-reuse' in text
    }
    assert runners == {"_ppu-upstream-reuse.yaml"}


def test_upstream_reuse_leaf_uses_exact_pin_and_dual_source_identity():
    config = workflow("_ppu-upstream-reuse.yaml")
    assert set(config[True]) == {"workflow_call"}
    assert set(config[True]["workflow_call"]["inputs"]) == {
        "pr_mode",
        "pr_head_sha",
        "build_run_id",
        "wheel_source_sha",
    }
    assert set(config["jobs"]) == {"upstream-reuse"}
    job = config["jobs"]["upstream-reuse"]
    assert "container" not in job and "strategy" not in job
    assert job["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "mktemp" in job["steps"][0]["run"]
    upstream = checkouts(job)[1]
    assert upstream["with"] == {
        "repository": "vllm-project/vllm",
        "ref": "${{ steps.upstream.outputs.ref }}",
        "path": "${{ steps.runtime.outputs.upstream_rel }}",
        "persist-credentials": False,
    }
    verify = job["steps"][job["steps"].index(upstream) + 1]
    assert verify["env"]["EXPECTED_SHA"] == upstream["with"]["ref"]
    assert "rev-parse HEAD" in verify["run"]
    runtime = next(
        s for s in job["steps"] if "ppu_runtime_container.sh" in s.get("run", "")
    )
    assert runtime["run"].endswith('" upstream-reuse')
    assert runtime["env"]["HF_HOME"] == "/nas_aisw/datasets/hf_cache"
    assert runtime["env"]["PPU_CI_REUSE_CHECKPOINT_ROOT"] == (
        "/nas_aisw/datasets/checkpoints/LLM"
    )
    assert runtime["env"]["PPU_CI_DEVICE_COUNT"] == "1"
    assert (
        runtime["env"]["PPU_CI_UPSTREAM_DIR"] == "${{ steps.runtime.outputs.upstream }}"
    )
    assert runtime["env"]["PPU_CI_SOURCE_SHA"] == EXPECTED_SHA
    assert runtime["env"]["PPU_CI_WHEEL_SOURCE_SHA"] == (
        "${{ inputs.wheel_source_sha || inputs.pr_head_sha || github.sha }}"
    )
    assert runtime["env"]["PPU_CI_REQUIRE_PR_ARTIFACT"] == "${{ inputs.pr_mode }}"
    assert not any("TOKEN" in key or "IMAGE" in key for key in runtime["env"])
    download = next(
        s for s in job["steps"] if s.get("uses") == "actions/download-artifact@v4"
    )
    assert download["with"]["run-id"] == "${{ inputs.build_run_id }}"
    assert download["with"]["name"] == "ppu-wheels-${{ inputs.build_run_id }}"
    upload = job["steps"][-1]
    assert upload["if"] == "always()"
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert "github.run_attempt" in upload["with"]["name"]
    jobs = workflow("ppu-build-and-test.yaml")["jobs"]
    runtime_ok = "needs.runtime-image.result == 'success' &&"
    assert jobs["upstream-reuse"]["if"] == jobs["pin-unit-tests"]["if"].replace(
        runtime_ok,
        runtime_ok + " needs.runtime-image.outputs.upstream_reuse == 'true' &&",
    )


@pytest.mark.parametrize(
    "upstream_sha,count,missing_cache,accepted",
    [
        ("b" * 40, "1", None, True),
        ("c" * 40, "1", None, False),
        ("b" * 40, "2", None, False),
        ("b" * 40, "0", None, False),
        ("b" * 40, "1", "hub", False),
        ("b" * 40, "1", "checkpoints", False),
    ],
)
def test_upstream_reuse_container_mounts_pin_readonly_without_host_sdk_or_token(
    tmp_path, upstream_sha, count, missing_cache, accepted
):
    source, upstream, wheels, artifacts, binaries = [
        tmp_path / name for name in ("source", "upstream", "wheels", "artifacts", "bin")
    ]
    for directory in (source, upstream, wheels, artifacts, binaries):
        directory.mkdir()
    (source / ".github").mkdir()
    (source / ".github/vllm-main-verified.commit").write_text("b" * 40 + "\n")
    (source / "scripts/ci").mkdir(parents=True)
    (source / "scripts/ci/ppu_runtime_image.sh").write_text(
        (ROOT / "scripts/ci/ppu_runtime_image.sh").read_text()
    )
    (upstream / "vllm").mkdir()
    (upstream / "vllm/__init__.py").write_text("")
    hf_home = tmp_path / "hf_cache"
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "opt-125m"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"opt"}')
    snapshots = hf_home / "hub/models--facebook--opt-125m/snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "main").symlink_to(
        os.path.relpath(checkpoint, snapshots), target_is_directory=True
    )
    commands = {
        "git": (
            "import os,sys\n"
            "print(os.environ['UPSTREAM_SHA'] if sys.argv[sys.argv.index('-C')+1].endswith('/upstream') "
            "else 'a'*40)\n"
        ),
        "flock": "pass\n",
        "docker": (
            "import json,os,sys\n"
            "with open(os.environ['DOCKER_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "if sys.argv[1:3] == ['image','inspect']: print('sha256:'+'d'*64)\n"
        ),
    }
    for name, body in commands.items():
        path = binaries / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)
    log = tmp_path / "docker.jsonl"
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "PPU_CI_SOURCE_DIR": str(source),
        "PPU_CI_SOURCE_SHA": "a" * 40,
        "PPU_CI_UPSTREAM_DIR": str(upstream),
        "PPU_CI_WHEELS_DIR": str(wheels),
        "PPU_CI_ARTIFACT_DIR": str(artifacts),
        "PPU_CI_DEVICE_COUNT": count,
        "PPU_CI_REQUIRE_PR_ARTIFACT": "true",
        "PPU_CI_WHEEL_SOURCE_SHA": "e" * 40,
        "PPU_CI_RUNTIME_IMAGE": "runner-configured",
        "HF_HOME": str(hf_home / "missing" if missing_cache == "hub" else hf_home),
        "PPU_CI_REUSE_CHECKPOINT_ROOT": str(
            checkpoint_root / "missing"
            if missing_cache == "checkpoints"
            else checkpoint_root
        ),
        "GITHUB_TOKEN": "never-pass-this-token",
        "UPSTREAM_SHA": upstream_sha,
        "DOCKER_LOG": str(log),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_container.sh"), "upstream-reuse"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is accepted, result.stderr
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    creates = [call for call in calls if call[0] == "create"]
    if not accepted:
        assert not creates
        return
    args = creates[0]
    assert f"type=bind,source={upstream},target=/upstream,readonly" in args
    assert f"type=bind,source={source},target=/source,readonly" in args
    assert f"type=bind,source={wheels},target=/wheels,readonly" in args
    assert f"type=bind,source={artifacts},target=/artifacts" in args
    assert f"type=bind,source={hf_home},target={hf_home},readonly" in args
    assert (
        f"type=bind,source={checkpoint_root},target={checkpoint_root},readonly" in args
    )
    assert not Path(os.readlink(snapshots / "main")).is_absolute()
    assert (snapshots / "main/config.json").is_file()
    assert (snapshots / "main").resolve() == checkpoint
    assert "HF_HOME=" + str(hf_home) in args
    assert "HF_HUB_CACHE=" + str(hf_home / "hub") in args
    assert "HUGGINGFACE_HUB_CACHE=" + str(hf_home / "hub") in args
    assert "TRANSFORMERS_CACHE=" + str(hf_home / "hub") in args
    assert "HF_MODULES_CACHE=/tmp/hf-modules" in args
    assert "HF_HUB_OFFLINE=1" in args
    assert "TRANSFORMERS_OFFLINE=1" in args
    assert "REUSE_CHECKPOINT_ROOT=" + str(checkpoint_root) in args
    assert "TESTED_SAIL_COMMIT=" + "a" * 40 in args
    assert "EXPECTED_SAIL_COMMIT=" + "e" * 40 in args
    assert "EXPECTED_VLLM_COMMIT=" + "b" * 40 in args
    assert "VLLM_SOURCE_ROOT=/upstream" in args
    assert args[-1] == "/source/scripts/ci/ppu_upstream_reuse.sh"
    assert "--ipc=host" in args
    assert sum(arg.startswith("--device=/dev/alixpu_ppu") for arg in args) == 1
    assert not any(
        "TOKEN" in arg or "never-pass" in arg or "PPU_SDK" in arg for arg in args
    )


def test_upstream_reuse_script_protects_image_and_wheel_imports_and_is_fail_closed():
    text = (ROOT / "scripts/ci/ppu_upstream_reuse.sh").read_text()
    assert '--sail-commit "${EXPECTED_SAIL_COMMIT}"' in text
    assert '--vllm-commit "${EXPECTED_VLLM_COMMIT}"' in text
    assert "PPU PIN UT SKIP" not in text
    assert "810E skipped: runner unavailable" in text
    assert "unset VLLM_VERSION PYTHONPATH GH_TOKEN GITHUB_TOKEN" in text
    assert "--no-index --no-deps --force-reinstall" in text
    assert 'PIP_CONSTRAINT="${REUSE_RESULTS_DIR}/image-constraints.txt"' in text
    assert 'ppu_image_stack.py" capture' in text
    assert 'ppu_image_stack.py" verify' in text
    assert "trap cleanup EXIT" in text
    assert "requirements/upstream-reuse.txt" in text
    assert "requirements/dev.txt" not in text
    assert "current_platform.is_ppu()" in text
    assert "export VLLM_PLUGINS=ppu" in text
    assert "VLLM_TARGET_DEVICE" not in text
    assert "import torchvision" in text
    assert "torch.cuda.is_available()" in text
    assert "assert count == 1" in text
    assert "capability == (8, 9)" in text
    assert 'assert "890" in name.upper()' in text
    assert 'cp -a "${ROOT_DIR}/tests" "${WORK_DIR}/tests"' in text
    assert 'cp "${ROOT_DIR}/pyproject.toml" "${WORK_DIR}/pyproject.toml"' in text
    assert 'cd "${WORK_DIR}"' in text
    assert "cp -a /upstream" not in text and "PYTHONPATH=" not in text
    for arg in (
        "tests/upstream_reuse -q -rs -m upstream_source",
        "--require-upstream-source --strict-upstream-reuse",
        "--upstream-reuse-target=890p",
        "--upstream-reuse-report-dir=/artifacts/reuse",
    ):
        assert arg in text


@pytest.mark.parametrize(
    "change,reason",
    [
        ({}, None),
        ({"is_ppu": False}, "namespace"),
        ({"available": False}, "real PPU"),
        ({"count": 0}, "visible card"),
        ({"count": 2}, "visible card"),
        ({"capability": (8, 0)}, "non-M890"),
        ({"platform_capability": (8, 0)}, "8, 0"),
        ({"name": "NVIDIA RTX 4090"}, "non-M890"),
        ({"vllm_path": "/upstream/vllm/__init__.py"}, "vllm"),
        ({"sail_path": "/source/vllm_sail/__init__.py"}, "vllm_sail"),
    ],
)
def test_reuse_preflight_rejects_wrong_device_and_checkout_imports(
    tmp_path, monkeypatch, change, reason
):
    values = {
        "is_ppu": True,
        "available": True,
        "count": 1,
        "capability": (8, 9),
        "platform_capability": (8, 9),
        "name": "ZW-M890P",
        "vllm_path": "/site-packages/vllm/__init__.py",
        "sail_path": "/site-packages/vllm_sail/__init__.py",
        **change,
    }
    platform = types.SimpleNamespace(
        is_ppu=lambda: values["is_ppu"],
        get_device_capability=lambda _: values["platform_capability"],
    )
    torch = types.SimpleNamespace(
        __file__="/site-packages/torch/__init__.py",
        __version__="2.13+sail",
        cuda=types.SimpleNamespace(
            is_available=lambda: values["available"],
            device_count=lambda: values["count"],
            get_device_name=lambda _: values["name"],
            get_device_capability=lambda _: values["capability"],
        ),
    )
    modules = {
        "numba": types.SimpleNamespace(__version__="0.65.0"),
        "tblib": types.SimpleNamespace(__version__="3.1.0"),
        "torchvision": types.SimpleNamespace(__version__="0.28.0"),
        "torch": torch,
        "vllm": types.SimpleNamespace(__file__=values["vllm_path"]),
        "vllm_sail": types.SimpleNamespace(__file__=values["sail_path"]),
        "vllm.platforms": types.SimpleNamespace(current_platform=platform),
        "vllm_sail.compat": types.SimpleNamespace(
            check_vllm_compatibility=lambda **_: None
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for key, value in {
        "TESTED_SAIL_COMMIT": "a" * 40,
        "EXPECTED_SAIL_COMMIT": "b" * 40,
        "EXPECTED_VLLM_COMMIT": "c" * 40,
        "HF_HOME": "/nas_aisw/datasets/hf_cache",
        "HF_HUB_CACHE": "/nas_aisw/datasets/hf_cache/hub",
        "REUSE_CHECKPOINT_ROOT": "/nas_aisw/datasets/checkpoints/LLM",
        "REUSE_RESULTS_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    text = (ROOT / "scripts/ci/ppu_upstream_reuse.sh").read_text()
    code = text.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    if reason is not None:
        with pytest.raises(AssertionError, match=reason):
            exec(compile(code, "reuse-preflight", "exec"), {})
        assert not (tmp_path / "runtime-identity.json").exists()
    else:
        exec(compile(code, "reuse-preflight", "exec"), {})
        identity = json.loads((tmp_path / "runtime-identity.json").read_text())
        assert identity["tested_sha"] == "a" * 40
        assert identity["wheel_source_sha"] == "b" * 40
        assert identity["is_ppu"] is True
        assert identity["device_count"] == 1
        assert identity["capability"] == [8, 9]
        assert identity["torchvision_version"] == "0.28.0"
        assert identity["hf_home"] == "/nas_aisw/datasets/hf_cache"
        assert identity["hf_hub_cache"] == "/nas_aisw/datasets/hf_cache/hub"
        assert identity["checkpoint_root"] == "/nas_aisw/datasets/checkpoints/LLM"
        assert identity["810e"] == "skipped: runner unavailable"


# --- kernel-perf label workflow ------------------------------------------------


def test_kernel_perf_label_workflow_triggers_only_on_correct_label():
    config = workflow("kernel-perf-label.yaml")
    assert config["name"] == "Kernel Perf Gate"
    assert set(config[True]) == {"pull_request"}
    pr = config[True]["pull_request"]
    assert set(pr["types"]) == {"labeled", "synchronize", "reopened"}
    assert pr["branches"] == ["main"]

    # Top-level permissions include actions:read for artifact access
    assert config["permissions"] == {"contents": "read", "actions": "read"}
    assert config["cache-mode"] == "none"

    # The build job carries the label / draft / action guard
    assert set(config["jobs"]) == {"build", "kernel-perf"}
    build = config["jobs"]["build"]
    guard = build["if"]

    # Build calls the reusable _ppu-build.yaml with the PR head SHA
    assert build["uses"] == "./.github/workflows/_ppu-build.yaml"
    assert build["with"] == {
        "pr_head_sha": "${{ github.event.pull_request.head.sha }}",
    }

    # kernel-perf calls _ppu-kernel-perf.yaml with run_id (same workflow)
    kp = config["jobs"]["kernel-perf"]
    assert kp["needs"] == "build"
    assert kp["uses"] == "./.github/workflows/_ppu-kernel-perf.yaml"
    assert kp["with"] == {
        "pr_mode": True,
        "pr_head_sha": "${{ github.event.pull_request.head.sha }}",
        "build_run_id": "${{ github.run_id }}",
        "wheel_source_sha": "${{ github.event.pull_request.head.sha }}",
    }

    # labeled event for a *different* label must NOT trigger
    ctx_other_label = {
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["kernel-perf", "ready"],
        "github.event.action": "labeled",
        "github.event.label.name": "ready",
    }
    assert not expression(guard, ctx_other_label)

    # labeled event for kernel-perf label triggers
    ctx_kp_label = {
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["kernel-perf"],
        "github.event.action": "labeled",
        "github.event.label.name": "kernel-perf",
    }
    assert expression(guard, ctx_kp_label)

    # synchronize with label present triggers
    ctx_sync = {
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["kernel-perf"],
        "github.event.action": "synchronize",
        "github.event.label.name": None,
    }
    assert expression(guard, ctx_sync)

    # reopened with label present triggers
    ctx_reopen = {
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": ["kernel-perf"],
        "github.event.action": "reopened",
        "github.event.label.name": None,
    }
    assert expression(guard, ctx_reopen)

    # draft PRs do NOT trigger
    ctx_draft = {
        "github.event.pull_request.draft": True,
        "github.event.pull_request.labels.*.name": ["kernel-perf"],
        "github.event.action": "labeled",
        "github.event.label.name": "kernel-perf",
    }
    assert not expression(guard, ctx_draft)

    # synchronize without label does NOT trigger
    ctx_no_label = {
        "github.event.pull_request.draft": False,
        "github.event.pull_request.labels.*.name": [],
        "github.event.action": "synchronize",
        "github.event.label.name": None,
    }
    assert not expression(guard, ctx_no_label)


def test_kernel_perf_label_is_independent_of_ci_gate():
    """kernel-perf-label.yaml is not referenced by ppu-build-and-test gate."""
    gate = workflow("ppu-build-and-test.yaml")
    assert "kernel-perf" not in gate["jobs"]
    gate_text = (WORKFLOWS / "ppu-build-and-test.yaml").read_text()
    assert "kernel-perf-label" not in gate_text
    assert "_ppu-kernel-perf.yaml" not in gate_text


# --- perf-regression schedule --------------------------------------------------


def test_perf_regression_is_weekly_and_has_no_update_baseline():
    config = workflow("perf-regression.yaml")
    assert config[True]["schedule"] == [{"cron": "0 20 * * 1"}]
    assert "workflow_dispatch" in config[True]
    dispatch_inputs = config[True]["workflow_dispatch"]["inputs"]
    assert "update_baseline" not in dispatch_inputs
    # The extra_env should not contain UPDATE_BASELINE
    perf_text = (WORKFLOWS / "perf-regression.yaml").read_text()
    assert "UPDATE_BASELINE" not in perf_text
    assert "update-baseline" not in perf_text.lower().replace("update_baseline", "")


def test_perf_regression_script_has_no_update_baseline():
    text = (ROOT / "scripts" / "ci" / "perf_regression.sh").read_text()
    assert "UPDATE_BASELINE" not in text
    assert "--update-baseline" not in text


def test_perf_regression_avoids_source_shadowing():
    """perf_regression.sh must keep the source tree off pytest's sys.path.

    The script copies tests/ and pyproject.toml into a clean directory that
    does NOT contain vllm_sail/.  pyproject.toml's pythonpath=["."] is safe
    in that context because "." only exposes tests/, so the installed wheel
    is never shadowed by the uncompiled source tree.
    """
    text = (ROOT / "scripts" / "ci" / "perf_regression.sh").read_text()
    # Isolated run directory: tests/ copied, vllm_sail/ excluded.
    assert 'PERF_RUN_DIR="/tmp/ppu-perf-run"' in text
    assert 'cp -r "${ROOT_DIR}/tests" "${PERF_RUN_DIR}/tests"' in text
    assert 'cp "${ROOT_DIR}/pyproject.toml" "${PERF_RUN_DIR}/pyproject.toml"' in text
    assert 'cd "${PERF_RUN_DIR}"' in text
    # vllm_sail must NOT be copied into the run directory.
    assert (
        "cp" not in text.split('cd "${PERF_RUN_DIR}"')[1].split("pytest_args")[0]
        or "vllm_sail"
        not in text.split('cd "${PERF_RUN_DIR}"')[1].split("pytest_args")[0]
    )
    # unset PYTHONPATH still guards against external environment leaks.
    assert "unset PYTHONPATH" in text
    # No legacy three-layer workarounds remain.
    assert '-o "pythonpath="' not in text
    assert '-c "${ROOT_DIR}/pyproject.toml"' not in text
    assert '--rootdir="${ROOT_DIR}"' not in text
    # TEST_PATH is relative (no ROOT_DIR prefix) since CWD is the run dir.
    assert '"${ROOT_DIR}/${TEST_PATH}"' not in text
    assert '"${TEST_PATH}"' in text


# --- debug workflow still includes kernel-perf ---------------------------------


def test_debug_workflow_retains_kernel_perf_target():
    config = workflow("ppu-debug.yaml")
    assert "kernel-perf" in config["jobs"]
    kp = config["jobs"]["kernel-perf"]
    assert kp["uses"] == "./.github/workflows/_ppu-kernel-perf.yaml"
    assert (
        "kernel-perf"
        in config[True]["workflow_dispatch"]["inputs"]["target"]["options"]
    )
