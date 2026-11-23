from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ut.test_ppu_wheel_manifest import make_wheels, manifest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = "scripts/ci/ppu_e2e_recipe_sanity.sh"
RECIPES = "tests/ppu/recipes"
RECIPE = f"{RECIPES}/zai-org/GLM-5.3.yaml"
POLICY = "tests/ppu/policy/models/glm-5.3.yaml"
CHECKOUT_SHA = "a" * 40
VLLM_SHA = "b" * 40
OLD_SHA = "c" * 40


@pytest.fixture
def run_recipe(tmp_path):
    wheels = tmp_path / "wheel bundle"
    wheels.mkdir()
    make_wheels(wheels)
    verified_wheels = manifest.wheel_records(wheels)
    source = tmp_path / "source checkout"
    shutil.copytree(ROOT / "tests/ppu", source / "tests/ppu")
    scripts = source / "scripts/ci"
    scripts.mkdir(parents=True)
    for name in (
        Path(ENTRYPOINT).name,
        "ppu_wheel_manifest.py",
        "ppu_recipe.py",
        "ppu_recipe_runner.py",
    ):
        shutil.copyfile(ROOT / "scripts/ci" / name, scripts / name)
    (source / "scripts/eval").mkdir()
    shutil.copyfile(
        ROOT / "scripts/eval/ppu_gsm8k_eval.py",
        source / "scripts/eval/ppu_gsm8k_eval.py",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        """import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args[:2] == ['-IB', '-']:
    os.execv(sys.executable, [sys.executable, *args])
with open(os.environ['CALL_TRACE'], 'a', encoding='utf-8') as trace:
    trace.write(json.dumps(args) + '\\n')
if args[:2] == ['-m', 'pip']:
    assert (Path(os.environ['RESULTS_DIR']) / 'provenance.json').is_file()
if args and args[0].endswith('/ppu_recipe_runner.py'):
    results = Path(args[args.index('--results-dir') + 1])
    results.mkdir(parents=True, exist_ok=True)
    (results / 'model-server.log').write_text('raw log with /private/mount/path')
    (results / 'summary.json').write_text('{"score": 0.75}')
    raise SystemExit(int(os.environ.get('RUNNER_EXIT', '0')))
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    results = tmp_path / "private results"
    trace = tmp_path / "calls.jsonl"

    def run(*, vllm=VLLM_SHA, sail=CHECKOUT_SHA, **overrides):
        wheel_manifest = {
            "schema": 1,
            "commits": {"vllm": vllm, "vllm-sail": sail},
            "wheels": verified_wheels,
        }
        (wheels / manifest.MANIFEST).write_text(json.dumps(wheel_manifest))
        env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CALL_TRACE": str(trace),
            "HF_HOME": str(tmp_path / "hf cache"),
            "WHEELS_DIR": str(wheels),
            "RESULTS_DIR": str(results),
            "MODEL_ID": "zai-org/GLM-5.3",
            "RECIPE_PATH": RECIPE,
            "POLICY_PATH": POLICY,
            "PPU_RECIPE_SOURCE_SHA": CHECKOUT_SHA,
            "EXPECTED_VLLM_COMMIT": VLLM_SHA,
            "EXPECTED_SAIL_COMMIT": CHECKOUT_SHA,
            "REQUIRE_PR_ARTIFACT": "false",
            "TIER": "fast_sanity",
            "RUNNER_EXIT": "0",
            **overrides,
        }
        trace.unlink(missing_ok=True)
        proc = subprocess.run(
            ["bash", str(source / ENTRYPOINT)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        calls = (
            [json.loads(line) for line in trace.read_text().splitlines()]
            if trace.exists()
            else []
        )
        return proc, calls, wheel_manifest

    return run, source, wheels, results


@pytest.mark.parametrize(
    "wrong_revision",
    [
        {"vllm": OLD_SHA},
        {"sail": OLD_SHA},
        {"EXPECTED_SAIL_COMMIT": OLD_SHA, "REQUIRE_PR_ARTIFACT": "true"},
    ],
)
def test_wrong_manifest_commit_stops_before_pip(run_recipe, wrong_revision):
    run, _, _, results = run_recipe
    results.mkdir()
    output = results / "provenance.json"
    output.write_text("stale")

    proc, calls, _ = run(**wrong_revision)

    assert proc.returncode != 0
    assert "Wheel manifest verification failed" in proc.stderr
    assert calls == []
    assert not output.exists()


def test_provenance_rejects_nonrelative_recipe_path_before_pip(run_recipe):
    run, source, _, results = run_recipe
    proc, calls, _ = run(RECIPE_PATH=str(source / RECIPE))

    assert proc.returncode != 0
    assert "Recipe provenance recording failed" in proc.stderr
    assert str(source) not in proc.stderr
    assert calls == []
    assert not (results / "provenance.json").exists()


@pytest.mark.parametrize(
    ("tier", "require_pr", "allowed"),
    [
        ("fast_sanity", "false", False),
        ("model_sanity", "true", False),
        ("fast_sanity", "true", True),
    ],
)
def test_mixed_wheel_is_only_allowed_in_pr_fast_sanity(
    run_recipe, tier, require_pr, allowed
):
    run, _, _, results = run_recipe
    proc, calls, wheel_manifest = run(
        sail=OLD_SHA,
        EXPECTED_SAIL_COMMIT=OLD_SHA,
        REQUIRE_PR_ARTIFACT=require_pr,
        TIER=tier,
    )

    output = results / "provenance.json"
    if allowed:
        assert proc.returncode == 0, proc.stderr
        assert any(call[:2] == ["-m", "pip"] for call in calls)
        evidence = json.loads(output.read_text())
        assert evidence["exact_source_match"] is False
        assert evidence["checkout_sha"] == CHECKOUT_SHA
        assert evidence["commits"] == wheel_manifest["commits"]
    else:
        assert proc.returncode != 0
        assert "SAIL wheel revision must match the recipe checkout" in proc.stderr
        assert calls == []
        assert not output.exists()


def test_exact_wheels_and_private_results(run_recipe):
    run, source, wheels, results = run_recipe
    proc, calls, wheel_manifest = run(TIER="model_sanity", RUNNER_EXIT="23")

    assert proc.returncode == 23, proc.stderr
    assert proc.stdout == ""
    evidence = json.loads((results / "provenance.json").read_text())
    assert evidence["checkout_sha"] == CHECKOUT_SHA
    assert evidence["commits"] == wheel_manifest["commits"]
    assert evidence["wheels"] == wheel_manifest["wheels"]
    assert evidence["exact_source_match"] is True
    assert evidence["tier"] == "model_sanity"
    for label, path in (("recipe", RECIPE), ("policy", POLICY)):
        assert evidence[label] == {
            "path": path,
            "sha256": hashlib.sha256((source / path).read_bytes()).hexdigest(),
        }
    assert {entry["path"] for entry in evidence["runner_sources"]} == {
        ENTRYPOINT,
        "scripts/ci/ppu_recipe.py",
        "scripts/ci/ppu_recipe_runner.py",
        "scripts/eval/ppu_gsm8k_eval.py",
    }
    for entry in evidence["runner_sources"]:
        assert (
            entry["sha256"]
            == hashlib.sha256((source / entry["path"]).read_bytes()).hexdigest()
        )

    expected_installs = [
        str(wheels.resolve() / wheel_manifest["wheels"][package]["file"])
        for package in ("vllm", "vllm-sail")
    ]
    installs = [call[-1] for call in calls if "--force-reinstall" in call]
    assert installs == expected_installs
    runner = calls[-1]
    assert runner[0].endswith("/ppu_recipe_runner.py")
    for flag, expected in (("--recipe", RECIPE), ("--policy", POLICY)):
        assert (
            Path(runner[runner.index(flag) + 1]).resolve()
            == (source / expected).resolve()
        )
    assert (results / "results/model-server.log").read_text() == (
        "raw log with /private/mount/path"
    )
    assert json.loads((results / "results/summary.json").read_text()) == {"score": 0.75}
    assert not (results / "summary.json").exists()


def test_recipes_tree_digest_is_stable_and_covers_unselected_recipes(run_recipe):
    run, source, _, results = run_recipe

    def digest():
        proc, _, _ = run()
        assert proc.returncode == 0, proc.stderr
        evidence = json.loads((results / "provenance.json").read_text())
        return evidence["recipes_tree_sha256"]

    original = digest()
    assert len(original) == 64
    assert digest() == original
    other_recipe = source / RECIPES / "deepseek-ai/DeepSeek-V4-Flash.yaml"
    other_recipe.write_bytes(other_recipe.read_bytes() + b"\n")
    assert digest() != original


def test_prebuilt_image_mode_skips_wheel_install_and_records_digest(run_recipe):
    run, _, _, results = run_recipe
    image_digest = "@sha256:" + "d" * 64
    image_id = "sha256:" + "e" * 64
    proc, calls, _ = run(
        TIER="model_sanity",
        WHEELS_DIR="",
        PPU_CI_SKIP_WHEEL_INSTALL="true",
        PPU_CI_RUNTIME_IMAGE_DIGEST=image_digest,
        PPU_CI_RUNTIME_IMAGE_ID=image_id,
    )
    assert proc.returncode == 0, proc.stderr
    evidence = json.loads((results / "provenance.json").read_text())
    assert evidence["install_mode"] == "prebuilt_image"
    assert evidence["runtime_image_digest"] == image_digest
    assert evidence["runtime_image_id"] == image_id
    assert "wheels" not in evidence
    assert not any("--force-reinstall" in call for call in calls)


def test_trusted_entrypoint_runs_prebuilt_mode_against_older_source_tree(
    tmp_path, run_recipe
):
    _, source, _, results = run_recipe
    # Simulate an older target_ref checkout whose own scripts/ci entrypoint
    # cannot handle prebuilt-image mode; the trusted /trusted-ci entrypoint
    # runs with PPU_CI_SOURCE_ROOT pointing at the target checkout.
    (source / "scripts/ci/ppu_e2e_recipe_sanity.sh").write_text(
        "#!/usr/bin/env bash\necho 'legacy entrypoint should not run' >&2\nexit 99\n",
        encoding="utf-8",
    )
    trusted_ci = tmp_path / "trusted-ci"
    trusted_ci.mkdir()
    for name in (
        "ppu_e2e_recipe_sanity.sh",
        "ppu_wheel_manifest.py",
        "ppu_image_stack.py",
    ):
        shutil.copyfile(ROOT / "scripts/ci" / name, trusted_ci / name)
    bin_dir = tmp_path / "bin"
    trace = tmp_path / "trusted-calls.jsonl"
    image_digest = "@sha256:" + "d" * 64
    proc = subprocess.run(
        ["bash", str(trusted_ci / "ppu_e2e_recipe_sanity.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CALL_TRACE": str(trace),
            "PPU_CI_SOURCE_ROOT": str(source),
            "HF_HOME": str(tmp_path / "hf cache"),
            "WHEELS_DIR": "",
            "RESULTS_DIR": str(results),
            "MODEL_ID": "zai-org/GLM-5.3",
            "RECIPE_PATH": RECIPE,
            "POLICY_PATH": POLICY,
            "PPU_RECIPE_SOURCE_SHA": CHECKOUT_SHA,
            "PPU_CI_SKIP_WHEEL_INSTALL": "true",
            "PPU_CI_RUNTIME_IMAGE_DIGEST": image_digest,
            "REQUIRE_PR_ARTIFACT": "false",
            "TIER": "model_sanity",
            "RUNNER_EXIT": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    evidence = json.loads((results / "provenance.json").read_text())
    assert evidence["install_mode"] == "prebuilt_image"
    assert evidence["runtime_image_digest"] == image_digest
    recorded = {entry["path"]: entry["sha256"] for entry in evidence["runner_sources"]}
    trusted_sha = hashlib.sha256(
        (trusted_ci / "ppu_e2e_recipe_sanity.sh").read_bytes()
    ).hexdigest()
    stub_sha = hashlib.sha256(
        (source / "scripts/ci/ppu_e2e_recipe_sanity.sh").read_bytes()
    ).hexdigest()
    assert trusted_sha != stub_sha
    assert recorded["scripts/ci/ppu_e2e_recipe_sanity.sh"] == trusted_sha
    for rel in (
        "scripts/ci/ppu_recipe.py",
        "scripts/ci/ppu_recipe_runner.py",
        "scripts/eval/ppu_gsm8k_eval.py",
    ):
        assert recorded[rel] == hashlib.sha256((source / rel).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "bad_overrides,expected_msg",
    [
        (
            {
                "WHEELS_DIR": "",
                "PPU_CI_SKIP_WHEEL_INSTALL": "true",
                "PPU_CI_RUNTIME_IMAGE_DIGEST": "",
            },
            "verified runtime image digest",
        ),
        (
            {
                "WHEELS_DIR": "",
                "PPU_CI_SKIP_WHEEL_INSTALL": "true",
                "PPU_CI_RUNTIME_IMAGE_DIGEST": "private.internal/repo@sha256:"
                + "d" * 64,
            },
            "Invalid runtime image digest",
        ),
        (
            {
                "WHEELS_DIR": "",
                "PPU_CI_SKIP_WHEEL_INSTALL": "true",
                "REQUIRE_PR_ARTIFACT": "true",
                "PPU_CI_RUNTIME_IMAGE_DIGEST": "@sha256:" + "d" * 64,
            },
            "not allowed in PR mode",
        ),
        (
            {
                "PPU_CI_SKIP_WHEEL_INSTALL": "true",
                "PPU_CI_RUNTIME_IMAGE_DIGEST": "@sha256:" + "d" * 64,
            },
            "WHEELS_DIR must be empty",
        ),
    ],
)
def test_prebuilt_image_mode_rejects_unsafe_inputs(
    run_recipe, bad_overrides, expected_msg
):
    run, _, _, results = run_recipe
    proc, calls, _ = run(TIER="model_sanity", **bad_overrides)
    assert proc.returncode != 0
    assert expected_msg in proc.stderr
    assert calls == []
    assert not (results / "provenance.json").exists()
