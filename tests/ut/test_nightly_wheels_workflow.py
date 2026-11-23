# SPDX-License-Identifier: Apache-2.0
"""Nightly entry-point contracts and real Git branch resolution."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/nightly-wheels.yaml").read_text())


def test_nightly_reuses_the_builder_and_publishes_its_wheel_pair():
    jobs = WORKFLOW["jobs"]
    assert WORKFLOW[True]["schedule"] == [{"cron": "0 13 * * *"}]
    assert set(WORKFLOW[True]["workflow_dispatch"]["inputs"]) == {"branch"}
    assert jobs["prepare"]["if"] == (
        "github.repository == 'vllm-project/vllm-sail' && github.ref == 'refs/heads/main'"
    )
    assert WORKFLOW["permissions"] == {"contents": "read", "actions": "read"}
    assert WORKFLOW["cache-mode"] == "none"
    assert jobs["build"] == {
        "needs": "prepare",
        "uses": "./.github/workflows/_ppu-build.yaml",
        "with": {"pr_head_sha": "${{ needs.prepare.outputs.sha }}"},
    }
    publish = jobs["publish"]
    assert publish["needs"] == "build"
    assert publish["uses"] == (
        "t-head/ci-infra/.github/workflows/push-tmp-wheel.yml@main"
    )
    builder = yaml.safe_load((ROOT / ".github/workflows/_ppu-build.yaml").read_text())
    upload = next(
        step["with"]
        for step in builder["jobs"]["build"]["steps"]
        if step.get("name") == "Upload wheels"
    )
    assert publish["with"] == {
        "artifact_name": upload["name"],
        "repo": "${{ needs.build.outputs.repo }}",
        "commit": "${{ needs.build.outputs.commit }}",
        "branch": "${{ needs.build.outputs.branch }}",
        "tag": "${{ needs.build.outputs.tag }}",
        "template_repository": "t-head/ci-infra",
        "template_ref": "main",
    }
    assert publish["secrets"] == {
        "release_username": "${{ secrets.RELEASE_USERNAME }}",
        "release_token": "${{ secrets.RELEASE_TOKEN }}",
        "template_token": "${{ secrets.TEMPLATE_TOKEN }}",
    }
    assert jobs["downloads"]["needs"] == ["prepare", "build", "publish"]
    summary_env = jobs["downloads"]["steps"][0]["env"]
    assert summary_env["SOURCE_BRANCH"] == publish["with"]["branch"]
    assert summary_env["SOURCE_SHA"] == publish["with"]["commit"]
    build = builder["jobs"]["build"]
    checkout = next(
        step for step in build["steps"] if step.get("name") == "Check out vLLM SAIL"
    )
    source = next(step for step in build["steps"] if step.get("id") == "source")
    assert (
        checkout["with"]["repository"]
        == source["env"]["SOURCE_REPO"]
        == ("${{ github.repository }}")
    )
    assert checkout["with"]["ref"] == "${{ inputs.pr_head_sha || github.sha }}"
    assert checkout["with"]["fetch-depth"] == 0
    checkout_index = build["steps"].index(checkout)
    assert build["steps"][checkout_index + 1]["name"] == "Verify checked-out revision"
    compile_step = next(
        step
        for step in build["steps"]
        if step.get("name") == "Build or repackage wheels"
    )
    assert (
        checkout_index + 1
        < build["steps"].index(source)
        < build["steps"].index(compile_step)
    )
    for field in ("repo", "commit", "branch", "tag"):
        assert build["outputs"][field] == "${{ steps.source.outputs." + field + " }}"
        assert builder[True]["workflow_call"]["outputs"][field]["value"] == (
            "${{ jobs.build.outputs." + field + " }}"
        )


@pytest.mark.parametrize(
    "event,branch,success",
    [
        ("schedule", "main", True),
        ("workflow_dispatch", "main", True),
        ("workflow_dispatch", "feature/nightly", True),
        ("workflow_dispatch", "missing", False),
        ("workflow_dispatch", "main\nsha=injected", False),
    ],
)
def test_source_resolution_uses_real_git_and_rejects_invalid_branches(
    tmp_path, monkeypatch, event, branch, success
):
    config = tmp_path / "gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    remote = tmp_path / "remote"
    subprocess.run(
        ["git", "init", "-b", "main", str(remote)], check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(remote),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(remote), "branch", "feature/nightly"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(remote),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "advance main",
        ],
        check=True,
        capture_output=True,
    )
    expected_ref = "feature/nightly" if branch == "feature/nightly" else "main"
    head = subprocess.check_output(
        ["git", "-C", str(remote), "rev-parse", f"refs/heads/{expected_ref}"], text=True
    ).strip()
    # Redirect only the Git network boundary to a real local repository.
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(config),
            f"url.{remote.as_uri()}.insteadOf",
            "https://github.com/example/project.git",
        ],
        check=True,
    )
    output = tmp_path / "outputs"
    trigger = "a" * 40
    script = WORKFLOW["jobs"]["prepare"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "SOURCE_EVENT": event,
            "SOURCE_BRANCH": branch,
            "TRIGGER_SHA": trigger,
            "GITHUB_REPOSITORY": "example/project",
            "GITHUB_OUTPUT": str(output),
            "GIT_CONFIG_GLOBAL": str(config),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    if success:
        assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
            "sha": trigger if event == "schedule" else head,
            "branch": branch,
        }
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    "detached,tags",
    [(False, ()), (False, ("v1", "v2")), (True, ()), (True, ("v1", "v2"))],
)
def test_build_source_metadata_uses_checkout_without_branch_fallback(
    tmp_path, detached, tags
):
    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                *args,
            ],
            cwd=tmp_path,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    git("init", "-b", "feature/build")
    git("commit", "--allow-empty", "-m", "old commit")
    git("tag", "old-tag")
    git("commit", "--allow-empty", "-m", "build commit")
    commit = git("rev-parse", "HEAD")
    for tag in tags:
        git("tag", "-a", tag, "-m", tag)
    if detached:
        git("checkout", "--detach", commit)
    builder = yaml.safe_load((ROOT / ".github/workflows/_ppu-build.yaml").read_text())
    source = next(
        step for step in builder["jobs"]["build"]["steps"] if step.get("id") == "source"
    )
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-c", source["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "SOURCE_REPO": "example/build-source",
            "SOURCE_BRANCH": "main",
            "GITHUB_REPOSITORY": "example/workflow-source",
            "GITHUB_REF_NAME": "main",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_OUTPUT": str(output),
            "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
        "repo": "example/build-source",
        "commit": commit,
        "branch": "" if detached else "feature/build",
        "tag": ",".join(tags),
    }
