# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for selection and trusted PR Nightly workflow routing."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.ci.ppu_ci_command import Rejected
from scripts.ci.ppu_e2e_select import build_recipe_matrix
from scripts.ci.ppu_pr_nightly import resolve_cases

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case", ["glm-5.3", "deepseek-v4-flash"])
def test_resolves_a_built_in_nightly_recipe(case):
    assert resolve_cases(ROOT, case, True) == (
        [case],
        build_recipe_matrix(case, "model_sanity"),
        [],
    )


def test_resolves_an_exact_legacy_case_even_with_recipes_disabled():
    assert resolve_cases(ROOT, "deepseek-v4-int8-dp-ep", False) == (
        [],
        None,
        ["deepseek-v4-int8-dp-ep"],
    )


def test_resolves_multiple_mixed_recipe_and_legacy_cases():
    recipe_cases, recipe_matrix, legacy_cases = resolve_cases(
        ROOT, "glm-5.3, deepseek-v4-int8, deepseek-v4-flash, minimax-m3-fp8", True
    )
    assert recipe_cases == ["glm-5.3", "deepseek-v4-flash"]
    assert recipe_matrix == build_recipe_matrix(
        "glm-5.3,deepseek-v4-flash", "model_sanity"
    )
    assert legacy_cases == ["deepseek-v4-int8", "minimax-m3-fp8"]


def test_resolves_all_cases_as_recipe_only_like_scheduled_nightly():
    recipe_cases, recipe_matrix, legacy_cases = resolve_cases(ROOT, "all", True)
    expected_recipe = build_recipe_matrix("", "model_sanity")
    assert recipe_matrix == expected_recipe
    assert recipe_cases == [row["key"] for row in expected_recipe["include"]]
    assert legacy_cases == []


def test_resolves_all_cases_rejected_when_recipes_disabled():
    with pytest.raises(Rejected, match="disabled"):
        resolve_cases(ROOT, "all", False)


@pytest.mark.parametrize(
    "case,enabled,error",
    [
        ("glm-5.3", False, "disabled"),
        ("qwen3.8-27b", True, "selection failed"),
        ("unknown-case", True, "Unknown"),
        ("glm-5.3,unknown-case", True, "Unknown"),
        ("all,glm-5.3", True, "Cannot combine"),
        ("glm-5.3,glm-5.3", True, "Duplicate"),
        ("glm-5.3,", True, "Use `/ci run nightly"),
    ],
)
def test_invalid_or_disabled_case_never_reaches_build(case, enabled, error):
    with pytest.raises(Rejected, match=error):
        resolve_cases(ROOT, case, enabled)


def test_case_resolution_reads_the_candidate_instead_of_the_controller_checkout(
    tmp_path,
):
    selector_dir = tmp_path / "scripts/ci"
    selector_dir.mkdir(parents=True)
    for filename in ("ppu_e2e_select.py", "ppu_recipe.py"):
        shutil.copy(ROOT / "scripts/ci" / filename, selector_dir)
    catalog = json.loads((ROOT / "scripts/ci/ppu_e2e_models.json").read_text())
    case = next(entry for entry in catalog if entry["key"] == "deepseek-v4-int8")
    (selector_dir / "ppu_e2e_models.json").write_text(
        json.dumps([{**case, "key": "new-pr-case"}])
    )
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    policy_dir = tmp_path / "tests/ppu/policy/models"
    shutil.copy(policy_dir / "glm-5.3.yaml", policy_dir / "new-pr-policy.yaml")
    assert resolve_cases(tmp_path, "new-pr-case", True) == ([], None, ["new-pr-case"])
    recipe_cases, matrix, legacy_cases = resolve_cases(
        tmp_path, "new-pr-policy,new-pr-case", True
    )
    assert recipe_cases == ["new-pr-policy"]
    assert legacy_cases == ["new-pr-case"]
    assert [row["key"] for row in matrix["include"]] == ["new-pr-policy"]
    shutil.copy(policy_dir / "glm-5.3.yaml", policy_dir / "new-pr-case.yaml")
    with pytest.raises(Rejected, match="ambiguous"):
        resolve_cases(tmp_path, "new-pr-case", True)


@pytest.mark.parametrize(
    "scope",
    ["single", "multiple", "extra", "wrong_order", "duplicate", "scheduled"],
)
def test_device_matrix_uses_the_exact_selected_cases(tmp_path, scope):
    two_models = "glm-5.3,deepseek-v4-flash"
    two_cases = build_recipe_matrix(two_models, "model_sanity")
    _, single_case, _ = resolve_cases(ROOT, "glm-5.3", True)
    swapped = build_recipe_matrix("deepseek-v4-flash,glm-5.3", "model_sanity")
    values = {
        "single": ("glm-5.3", json.dumps(single_case), single_case),
        "multiple": (two_models, json.dumps(two_cases), two_cases),
        "extra": ("glm-5.3", json.dumps(two_cases), None),
        "wrong_order": (two_models, json.dumps(swapped), None),
        "duplicate": ("glm-5.3,glm-5.3", json.dumps(single_case), None),
        "scheduled": ("", "", two_cases),
    }
    requested_models, selected_matrix, expected = values[scope]
    output = tmp_path / "github-output"
    selector = tmp_path / "scripts/ci/ppu_e2e_select.py"
    selector.parent.mkdir(parents=True)
    selector.write_text(f"print({('recipe_matrix=' + json.dumps(two_cases))!r})\n")
    recipe = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-recipe-model-sanity.yaml").read_text()
    )
    step = next(
        step
        for step in recipe["jobs"]["prepare"]["steps"]
        if step.get("id") == "select"
    )
    env = {
        **os.environ,
        "MODELS": requested_models,
        "TIER": "model_sanity",
        "PR_MODE": "false",
        "GITHUB_RUN_ID": "10",
        "BUILD_RUN_ID": "10",
        "GITHUB_OUTPUT": str(output),
        "SELECTED_MATRIX": selected_matrix,
    }
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    if expected is None:
        assert result.returncode != 0
        assert output.read_text() == ""
        return
    assert result.returncode == 0, result.stderr
    line = output.read_text().splitlines()[0]
    assert json.loads(line.removeprefix("recipe_matrix=")) == expected


def test_pr_nightly_workflow_pins_same_head_serializes_lanes_and_keeps_ci_status_separate():
    workflows = ROOT / ".github/workflows"
    workflow = yaml.safe_load((workflows / "ppu-pr-nightly.yaml").read_text())
    assert set(workflow[True]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {}
    assert workflow["concurrency"]["queue"] == "max"
    jobs = workflow["jobs"]
    assert jobs["validate"]["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    assert "default_branch" in jobs["validate"]["if"]
    assert jobs["select"]["needs"] == "validate"
    assert jobs["select"]["permissions"] == {"contents": "read"}
    assert jobs["select"]["outputs"] == {
        "recipe_models": "${{ steps.select.outputs.recipe_models }}",
        "legacy_models": "${{ steps.select.outputs.legacy_models }}",
        "recipe_matrix": "${{ steps.select.outputs.recipe_matrix }}",
    }
    assert jobs["select"]["steps"][1]["with"] == {
        "ref": "${{ inputs.head_sha }}",
        "path": "candidate",
        "persist-credentials": False,
    }
    assert jobs["build"]["needs"] == "select"
    assert jobs["build"]["with"]["pr_head_sha"] == "${{ inputs.head_sha }}"
    assert jobs["legacy"]["needs"] == ["select", "build"]
    assert jobs["legacy"]["if"] == "needs.select.outputs.legacy_models != ''"
    assert jobs["legacy"]["with"] == {
        "pr_mode": True,
        "pr_head_sha": "${{ inputs.head_sha }}",
        "wheel_source_sha": "${{ inputs.head_sha }}",
        "build_run_id": "${{ github.run_id }}",
        "models": "${{ needs.select.outputs.legacy_models }}",
    }
    assert jobs["recipe"]["needs"] == ["select", "build", "legacy"]
    assert "!cancelled()" in jobs["recipe"]["if"]
    assert "needs.select.result == 'success'" in jobs["recipe"]["if"]
    assert "needs.build.result == 'success'" in jobs["recipe"]["if"]
    assert "needs.select.outputs.recipe_models != ''" in jobs["recipe"]["if"]
    assert "needs.select.outputs.recipe_matrix != ''" in jobs["recipe"]["if"]
    assert jobs["recipe"]["with"] == {
        "pr_mode": False,
        "pr_head_sha": "${{ inputs.head_sha }}",
        "wheel_source_sha": "${{ inputs.head_sha }}",
        "build_run_id": "${{ github.run_id }}",
        "models": "${{ needs.select.outputs.recipe_models }}",
        "selected_matrix": "${{ needs.select.outputs.recipe_matrix }}",
    }
    assert "needs.validate.result == 'success'" in jobs["report"]["if"]
    assert jobs["report"]["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    downloads = [
        step
        for step in jobs["report"]["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    ]
    assert {step["with"]["pattern"] for step in downloads} == {
        "ppu-recipe-model_sanity-*-${{ github.run_id }}-*",
        "ppu-pr-nightly-legacy-*-${{ github.run_id }}-*",
    }
    assert all("statuses" not in job.get("permissions", {}) for job in jobs.values())
    assert all("secrets" not in job for job in jobs.values())

    legacy_leaf = yaml.safe_load((workflows / "_ppu-accuracy-tests.yaml").read_text())
    assert legacy_leaf["jobs"]["accuracy"]["strategy"]["max-parallel"] == 1

    command = yaml.safe_load((workflows / "run-ci-command.yml").read_text())
    assert any(
        step.get("run") == "python3 -m scripts.ci.ppu_pr_nightly command"
        for step in command["jobs"]["command"]["steps"]
    )


def test_recipe_sanity_isolates_actions_env_and_sanitizes_diagnostics(tmp_path):
    recipe = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-recipe-model-sanity.yaml").read_text()
    )
    steps = recipe["jobs"]["sanity"]["steps"]
    checkouts = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2
    assert checkouts[1]["with"]["ref"] == "${{ github.sha }}"
    assert checkouts[1]["with"]["sparse-checkout"] == "scripts/ci"
    assert checkouts[1]["with"]["persist-credentials"] is False

    runtime_step = next(step for step in steps if step.get("id") == "recipe")
    retain_step = next(step for step in steps if step.get("id") == "retain")
    publish_step = next(step for step in steps if step.get("id") == "publish")
    for step in (runtime_step, retain_step, publish_step):
        assert step["env"]["TRUSTED_CI_DIR"] == "${{ steps.runtime.outputs.trusted }}"

    # 1. Verify Actions control env is stripped before executing the wrapper,
    #    while legitimate runtime env vars pass through.
    trusted_dir = tmp_path / "trusted"
    candidate_dir = tmp_path / "candidate"
    (trusted_dir / "scripts/ci").mkdir(parents=True)
    (candidate_dir / "scripts/ci").mkdir(parents=True)
    (candidate_dir / "scripts/ci/ppu_runtime_container.sh").write_text(
        "echo CANDIDATE_WRAPPER_EXECUTED >&2; exit 99\n"
    )
    probe_out = tmp_path / "env-probe.json"
    (trusted_dir / "scripts/ci/ppu_runtime_container.sh").write_text(
        "python3 - <<'PY'\n"
        "import json, os\n"
        f"open({str(probe_out)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
        "PY\n"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", runtime_step["run"]],
        env={
            **os.environ,
            "TRUSTED_CI_DIR": str(trusted_dir),
            "PPU_CI_SOURCE_DIR": str(candidate_dir),
            "PPU_CI_ARTIFACT_DIR": str(artifacts),
            "MODEL_ID": "test/model",
            "GITHUB_RUN_ID": "900",
            "GITHUB_RUN_ATTEMPT": "2",
            "ACTIONS_RUNTIME_TOKEN": "synthetic-actions-token",
            "GITHUB_TOKEN": "synthetic-gh-token",
            "GITHUB_OUTPUT": str(tmp_path / "gh-output"),
            "GITHUB_ENV": str(tmp_path / "gh-env"),
            "GITHUB_PATH": str(tmp_path / "gh-path"),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "gh-summary"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    seen_env = json.loads(probe_out.read_text())
    for forbidden in (
        "ACTIONS_RUNTIME_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_OUTPUT",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_STEP_SUMMARY",
    ):
        assert forbidden not in seen_env
    assert seen_env["MODEL_ID"] == "test/model"
    assert seen_env["GITHUB_RUN_ID"] == "900"
    assert seen_env["GITHUB_RUN_ATTEMPT"] == "2"

    # 2. Verify publish step allowlists diagnostics output before stdout,
    #    summary.json, and GITHUB_STEP_SUMMARY (adversarial + normal control).
    for raw_diag, expected_reason in (
        ("SYNTHETIC_TOKEN /private/model.log", "unknown"),
        ("gsm8k_gate_failed", "gsm8k_gate_failed"),
    ):
        diag_dir = tmp_path / f"diag-{expected_reason}"
        (diag_dir / "scripts/ci").mkdir(parents=True)
        (diag_dir / "scripts/ci/ppu_recipe_diagnostics.py").write_text(
            f"print({raw_diag!r})\n"
        )
        public_dir = tmp_path / f"public-{expected_reason}"
        public_dir.mkdir()
        step_summary = tmp_path / f"summary-{expected_reason}.md"
        pub = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", publish_step["run"]],
            env={
                **os.environ,
                "RECIPE_OUTCOME": "failure",
                "RETENTION_OUTCOME": "success",
                "RETENTION": "retained",
                "TRUSTED_CI_DIR": str(diag_dir),
                "PPU_CI_SOURCE_DIR": str(candidate_dir),
                "MODEL_KEY": "glm-5.3",
                "PRIVATE_DIR": str(artifacts),
                "PUBLIC_DIR": str(public_dir),
                "TIER": "model_sanity",
                "GITHUB_STEP_SUMMARY": str(step_summary),
            },
            capture_output=True,
            text=True,
        )
        assert pub.returncode == 0, pub.stderr
        public_json = (public_dir / "summary.json").read_text()
        summary_md = step_summary.read_text()
        assert json.loads(public_json)["failure_reason"] == expected_reason
        assert f"failure_reason: {expected_reason}" in pub.stdout
        assert f"**Recipe failure reason:** {expected_reason}" in summary_md
        assert "SYNTHETIC_TOKEN" not in (
            pub.stdout + pub.stderr + public_json + summary_md
        )
        assert "/private/model.log" not in (
            pub.stdout + pub.stderr + public_json + summary_md
        )


def test_trusted_runtime_container_never_executes_candidate_runtime_image_on_host(
    tmp_path,
):
    recipe = yaml.safe_load(
        (ROOT / ".github/workflows/_ppu-recipe-model-sanity.yaml").read_text()
    )
    runtime_step = next(
        step for step in recipe["jobs"]["sanity"]["steps"] if step.get("id") == "recipe"
    )

    # Set up trusted checkout with the real ppu_runtime_container.sh and a
    # recording trusted ppu_runtime_image.sh.
    trusted_dir = tmp_path / "trusted"
    (trusted_dir / "scripts/ci").mkdir(parents=True)
    shutil.copy(
        ROOT / "scripts/ci/ppu_runtime_container.sh",
        trusted_dir / "scripts/ci/ppu_runtime_container.sh",
    )
    trusted_marker = tmp_path / "trusted-image-called"
    image_id = "sha256:" + "1" * 64
    (trusted_dir / "scripts/ci/ppu_runtime_image.sh").write_text(
        f"#!/usr/bin/env bash\n"
        f"echo trusted > {str(trusted_marker)!r}\n"
        f"printf '%s\\n' {image_id!r}\n"
    )

    # Set up a separate candidate git repo with a hostile ppu_runtime_image.sh.
    candidate_dir = tmp_path / "candidate"
    (candidate_dir / ".github").mkdir(parents=True)
    (candidate_dir / "scripts/ci").mkdir(parents=True)
    (candidate_dir / ".github/vllm-main-verified.commit").write_text("a" * 40 + "\n")
    candidate_marker = tmp_path / "candidate-image-executed"
    (candidate_dir / "scripts/ci/ppu_runtime_image.sh").write_text(
        f"#!/usr/bin/env bash\necho hostile > {str(candidate_marker)!r}\nexit 99\n"
    )
    subprocess.run(["git", "init", str(candidate_dir)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(candidate_dir), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(candidate_dir), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(candidate_dir), "add", "."], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(candidate_dir), "commit", "-m", "candidate"],
        check=True,
        capture_output=True,
    )
    candidate_sha = subprocess.check_output(
        ["git", "-C", str(candidate_dir), "rev-parse", "HEAD"], text=True
    ).strip()

    # Set up minimal offline cache, wheels, artifacts, and stub flock/docker.
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    hf_home = tmp_path / "hf_home"
    snapshot = hf_home / "hub/models--test--model/snapshots/rev1"
    snapshot.mkdir(parents=True)
    refs_dir = hf_home / "hub/models--test--model/refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("rev1\n")
    dg_cache = tmp_path / "dg_cache"
    dg_cache.mkdir()
    gsm8k = tmp_path / "gsm8k"
    gsm8k.mkdir()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "flock").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "flock").chmod(0o755)
    docker_log = tmp_path / "docker.log"
    (fake_bin / "docker").write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {str(docker_log)!r}\nexit 0\n"
    )
    (fake_bin / "docker").chmod(0o755)

    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", runtime_step["run"]],
        env={
            **os.environ,
            "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"])),
            "TRUSTED_CI_DIR": str(trusted_dir),
            "PPU_CI_SOURCE_DIR": str(candidate_dir),
            "PPU_CI_SOURCE_SHA": candidate_sha,
            "PPU_CI_WHEEL_SOURCE_SHA": candidate_sha,
            "PPU_CI_WHEELS_DIR": str(wheels_dir),
            "PPU_CI_ARTIFACT_DIR": str(artifacts_dir),
            "PPU_CI_DEVICE_COUNT": "1",
            "PPU_CI_REQUIRE_PR_ARTIFACT": "false",
            "PPU_CI_HF_HOME": str(hf_home),
            "PPU_CI_DG_CACHE_DIR": str(dg_cache),
            "GSM8K_DATASET_PATH": str(gsm8k),
            "MODEL_ID": "test/model",
            "RECIPE_PATH": "tests/ppu/recipes/test.yaml",
            "POLICY_PATH": "tests/ppu/policy/models/test.yaml",
            "TIER": "model_sanity",
            "PRESET": "",
            "PPU_CI_QUEUE_S": "0",
            "GITHUB_RUN_ID": "900",
            "GITHUB_RUN_ATTEMPT": "1",
            "ACTIONS_RUNTIME_TOKEN": "synthetic-actions-token",
            "GITHUB_OUTPUT": str(tmp_path / "gh-output"),
        },
        capture_output=True,
        text=True,
    )
    assert not candidate_marker.exists(), (
        "Candidate ppu_runtime_image.sh must never execute on the host"
    )
    assert result.returncode == 0, (artifacts_dir / "launcher.log").read_text()
    assert trusted_marker.is_file()
    assert (artifacts_dir / "runtime-image-id.txt").read_text().strip() == image_id
    assert f"source={candidate_dir.resolve()},target=/source,readonly" in (
        docker_log.read_text()
    )
