# SPDX-License-Identifier: Apache-2.0
"""PPU operator CI regressions for isolation, node evidence, and fail-closed behavior using fake CPU-installed packages."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "tests/support/ppu_ci.py"
RUNNER = ROOT / "scripts/ci/ppu_ops_runner.py"


@pytest.fixture
def runner():
    assert RUNNER.is_file(), "Missing device runner"
    return load(RUNNER)


def test_runtime_preserves_strict_single_device_check(tmp_path):
    result, _ = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    torch_file = tmp_path / "site-packages/torch/__init__.py"
    torch_file.write_text(
        torch_file.read_text().replace(
            "device_count=lambda:1", "device_count=lambda:16"
        )
    )
    command = f"""
import sys
sys.path.insert(0, {str(tmp_path / "site-packages")!r})
from tests.support.ppu_ci import inspect_runtime
inspect_runtime({str(tmp_path / "isolated")!r}, {str(ROOT)!r}, [])
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path / "isolated",
        env={**os.environ, "VLLM_SAIL_USE_PLA": "1"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "ValueError:" in completed.stderr
    assert "inspect_runtime" in completed.stderr


def test_runner_entrypoint():
    assert RUNNER.is_file(), "Missing device runner"
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/ci/ppu_ops_test.sh")], capture_output=True
    )
    assert result.returncode == 0


def test_run_cli_cannot_succeed_without_inputs():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CI_RUN", "DEVICE_", "EXPECTED_", "SELECTION_"))
    }
    result = subprocess.run(
        [sys.executable, str(RUNNER), "run"], env=env, capture_output=True
    )
    assert result.returncode != 0


def test_observe_environment_rejects_unidentified_driver(runner):
    runtime = {"capability": [8, 0], "runtime_env": {"VLLM_SAIL_USE_PLA": True}}
    versions = {"torch": "2", "triton": "3", "pla": "1"}
    profile = runner.runtime_profile(
        runtime, versions, "HGCC 2.2.0", "Driver Version: 1.2.3"
    )
    assert profile["driver_identity"] == "1.2.3"
    with pytest.raises(ValueError):
        runner.runtime_profile(runtime, versions, "HGCC 2.2.0", "no driver identity")
    with pytest.raises(ValueError):
        runner.runtime_profile(
            runtime,
            {**versions, "pla": "unknown"},
            "HGCC 2.2.0",
            "Driver Version: 1.2.3",
        )


def test_isolated_tree_and_environment(runner, tmp_path):
    isolated = tmp_path / "isolated"
    runner.copy_test_tree(ROOT, isolated)
    assert (isolated / "tests/support/ppu_ci.py").is_file()
    assert (isolated / "tests/__init__.py").is_file()
    assert (isolated / "pyproject.toml").read_bytes() == (
        ROOT / "pyproject.toml"
    ).read_bytes()
    assert not (isolated / "tests/ut").exists()
    assert not (isolated / "vllm_sail").exists()
    env = runner.clean_environment(
        {
            "PYTHONPATH": "bad",
            "PYTEST_ADDOPTS": "-k bad",
            "VLLM_VERSION": "bad",
            "VLLM_PPU_USE_PLA": "0",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_SAIL": "bad",
        },
        ROOT,
        isolated,
    )
    for name in (
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "VLLM_VERSION",
        "VLLM_PPU_USE_PLA",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_SAIL",
    ):
        assert name not in env
    assert (
        env["VLLM_SAIL_USE_PLA"] == "1" and env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    )


@pytest.mark.parametrize(
    "run,attempt,group",
    [
        ("bad", "1", "ppu10-0"),
        ("123", "0", "ppu10-0"),
        ("123", "1", "../x"),
        ("123", "1", "ppu10-1"),
    ],
)
def test_nas_identity_rejects_invalid_values(runner, run, attempt, group):
    with pytest.raises(ValueError):
        runner.result_path(run, attempt, group, pod=True)


def test_nas_prefix_and_missing_mount(runner, monkeypatch):
    assert (
        str(runner.result_path("123", "2", "ppu10-0", pod=True))
        == "/mnt/wl_nas/devops/123-2/ops-tests/ppu10-0"
    )
    assert (
        str(runner.result_path("123", "2", "ppu10-0", pod=False))
        == "/wl_nas/devops/123-2/ops-tests/ppu10-0"
    )
    monkeypatch.setattr(os.path, "ismount", lambda p: False)
    with pytest.raises(ValueError, match="NAS"):
        runner.require_nas(Path("/mnt/wl_nas"))


def test_runtime_expansion_preserves_other_board(runner):
    config = json.loads((ROOT / "scripts/ci/ppu_ops_tests.json").read_bytes())
    selector = load(ROOT / "scripts/ci/ppu_ops_select.py")
    selector.discover_catalog(config, ROOT)
    original = {
        "mode": "subset",
        "test_ids": ["kda"],
        "files": ["tests/e2e/gdn/test_kda.py"],
        **selector.route_tests(config, ["kda"]),
    }
    effective = runner.effective_selection(config, original, b"original", "ppu10", True)
    assert effective["expanded_boards"] == ["ppu10"]
    assert effective["groups"][0]["files"] == sorted(
        t["test"]
        for t in config["test_catalog"]
        if "ppu10" in selector.runnable_boards(config, t)
    )
    assert "tests/e2e/test_mxfp8_emulation.py" in effective["groups"][0]["files"]
    assert effective["groups"][1] == original["groups"][1]
    assert original["test_ids"] == ["kda"]


@pytest.mark.parametrize(
    "damage", ["xml", "json", "empty", "wrong-board", "call-missing", "xml-skip"]
)
def test_incomplete_file_evidence_rejected(runner, tmp_path, damage):
    result, report = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    output = tmp_path / "results"
    for kind in ("logs", "nodes", "junit"):
        (output / kind).mkdir(parents=True)
    (output / "logs/sample.log").write_text(result.stdout)
    shutil.copyfile(tmp_path / "junit.xml", output / "junit/sample.xml")
    node_path = output / "nodes/sample.json"
    node_path.write_text(json.dumps(report))
    entry = json.loads((tmp_path / "config.json").read_bytes())["test_catalog"][0]
    runner.validate_file_evidence(output, entry, "ppu10")
    if damage == "xml":
        (output / "junit/sample.xml").rename(output / "junit/other.xml")
    elif damage == "json":
        node_path.rename(output / "nodes/other.json")
    elif damage == "empty":
        node_path.write_text("{}")
    elif damage == "xml-skip":
        path = output / "junit/sample.xml"
        path.write_text(
            path.read_text().replace("</testcase>", "<skipped/></testcase>")
            if "</testcase>" in path.read_text()
            else path.read_text().replace(" />", "><skipped/></testcase>", 1)
        )
    else:
        if damage == "wrong-board":
            report["board"] = "ppu15"
        else:
            report["nodes"][report["applicable"][0]].pop("call")
        node_path.write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError, OSError)):
        runner.validate_file_evidence(output, entry, "ppu10")


def test_file_failure_does_not_stop_remaining_files(runner, tmp_path):
    result, _ = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    isolated = tmp_path / "isolated"
    (isolated / "tests/e2e/test_first.py").write_text("def test_fail(): assert False\n")
    (isolated / "tests/e2e/test_second.py").write_text("def test_pass(): pass\n")
    entries = [
        {
            "id": n,
            "test": f"tests/e2e/test_{n}.py",
            "boards": ["ppu10"],
            "requires": ["torch", "vllm", "vllm_sail"],
            "board_overrides": {},
        }
        for n in ("first", "second")
    ]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"test_catalog": entries}))
    env = runner.clean_environment(os.environ, ROOT, isolated)
    # Only the fixture supplies fake installed packages.
    env["PYTHONPATH"] = str(tmp_path / "site-packages")
    ticks = iter(range(20))
    output = tmp_path / "results"
    output.mkdir()
    reports = runner.run_files(
        entries, "ppu10", config, isolated, output, env, clock=lambda: next(ticks)
    )
    assert [r["outcome"] for r in reports] == ["failure", "success"]
    assert all(r["duration_seconds"] == 1 for r in reports)
    assert (output / "nodes/second.json").is_file()


@pytest.fixture
def valid_inputs(runner, tmp_path):
    from tests.ut.test_ppu_wheel_manifest import make_wheels

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    make_wheels(wheels)
    manifest_tool = runner.select.load_sibling("ppu_wheel_manifest")
    manifest = {
        "schema": 1,
        "commits": {"vllm": "c" * 40, "vllm-sail": "d" * 40},
        "wheels": manifest_tool.wheel_records(wheels),
    }
    runner.select.write_json(wheels / "wheel-manifest.json", manifest)
    config_path = tmp_path / "ppu_ops_tests.json"
    env_path = tmp_path / "ppu_ops_environment.json"
    shutil.copyfile(ROOT / "scripts/ci/ppu_ops_tests.json", config_path)
    shutil.copyfile(ROOT / "scripts/ci/ppu_ops_environment.json", env_path)
    image = "registry/image@sha256:" + "a" * 64
    diagnostics = dict(
        sdk_sha256="a" * 64,
        torch_sha256="b" * 64,
        pytorch_sail_arch="ppu_15;ppu_10",
        build_image=image,
        vllm_commit="c" * 40,
        vllm_sail_commit="d" * 40,
        status="succeeded",
    )
    build = tmp_path / "build-manifest.txt"
    build.write_text("".join(f"{k}={v}\n" for k, v in diagnostics.items()))
    changes = runner.select.collect_changes(ROOT, None, None, "d" * 40)
    selection = runner.select.make_selection(
        config_bytes=config_path.read_bytes(),
        environment_bytes=env_path.read_bytes(),
        changes=changes,
        diagnostics=diagnostics,
        wheel_manifest=manifest,
        build_run_id="123",
        build_image=image,
        runtime_image=image,
        expected_vllm_commit="c" * 40,
        force_full=True,
    )
    selection_path = tmp_path / "selection.json"
    runner.select.write_json(selection_path, selection)
    return dict(
        WHEELS_DIR=str(wheels),
        BUILD_MANIFEST_FILE=str(build),
        SELECTION_FILE=str(selection_path),
        DEVICE_BOARD="ppu10",
        DEVICE_GROUP_ID="ppu10-0",
        CI_RUN_ID="123",
        CI_RUN_ATTEMPT="1",
        DEVICE_RESULTS_DIR=str(runner.result_path("123", "1", "ppu10-0", pod=True)),
        EXPECTED_SAIL_COMMIT="d" * 40,
        EXPECTED_VLLM_COMMIT="c" * 40,
        EXPECTED_BUILD_RUN_ID="123",
        EXPECTED_BUILD_IMAGE=image,
        RUNTIME_IMAGE=image,
        DISPATCH_MATRIX=json.dumps(selection["matrix"]),
    )


@pytest.mark.parametrize("board", ["ppu10", "ppu15"])
def test_full_inputs_need_no_confirmation(runner, valid_inputs, board):
    assert "QUALIFICATION_MODE" not in valid_inputs
    valid_inputs["DEVICE_BOARD"] = board
    valid_inputs["DEVICE_GROUP_ID"] = board + "-0"
    valid_inputs["DEVICE_RESULTS_DIR"] = str(
        runner.result_path("123", "1", board + "-0", pod=True)
    )
    environment_path = Path(valid_inputs["SELECTION_FILE"]).with_name(
        "ppu_ops_environment.json"
    )
    original_environment = environment_path.read_bytes()
    raw, _, config, environment, manifest = runner.read_inputs(valid_inputs)
    selection = json.loads(raw)
    assert selection["mode"] == "full"
    assert selection["files"] == sorted(
        t["test"]
        for t in config["test_catalog"]
        if runner.select.runnable_boards(config, t)
    )
    assert len(manifest["wheels"]) == 2
    assert environment["qualified"] is False
    assert environment_path.read_bytes() == original_environment


@pytest.mark.parametrize("mode", ["subset", "none"])
def test_unapproved_inputs_reject_partial_selection(runner, valid_inputs, mode):
    path = Path(valid_inputs["SELECTION_FILE"])
    selection = json.loads(path.read_bytes())
    config = json.loads(path.with_name("ppu_ops_tests.json").read_bytes())
    ids = ["kda"] if mode == "subset" else []
    selection.update(
        mode=mode,
        test_ids=ids,
        files=sorted(t["test"] for t in config["test_catalog"] if t["id"] in ids),
        **runner.select.route_tests(config, ids),
    )
    runner.select.write_json(path, selection)
    valid_inputs["DISPATCH_MATRIX"] = json.dumps(selection["matrix"])
    with pytest.raises(
        ValueError, match="Non-full selection requires a qualified environment"
    ):
        runner.read_inputs(valid_inputs)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("sdk_sha256", "", "hash"),
        ("sdk_sha256", "a" * 63, "hash"),
        ("torch_sha256", "", "hash"),
        ("torch_sha256", "unknown", "hash"),
        ("pytorch_sail_arch", "", "architectures"),
        ("pytorch_sail_arch", "ppu_10", "architectures"),
        ("pytorch_sail_arch", "ppu_15", "architectures"),
    ],
)
def test_full_inputs_reject_incomplete_fingerprint(
    runner, valid_inputs, field, value, message
):
    path = Path(valid_inputs["SELECTION_FILE"])
    selection = json.loads(path.read_bytes())
    build_path = Path(valid_inputs["BUILD_MANIFEST_FILE"])
    diagnostics = runner.select.read_diagnostics(build_path)
    diagnostics[field] = value
    manifest = json.loads(
        (Path(valid_inputs["WHEELS_DIR"]) / "wheel-manifest.json").read_bytes()
    )
    selection["build_fingerprint"] = runner.select.build_fingerprint(
        diagnostics,
        manifest,
        tested_sha=valid_inputs["EXPECTED_SAIL_COMMIT"],
        expected_vllm_commit=valid_inputs["EXPECTED_VLLM_COMMIT"],
        build_image=valid_inputs["EXPECTED_BUILD_IMAGE"],
    )
    runner.select.write_json(path, selection)
    build_path.write_text("".join(f"{k}={v}\n" for k, v in diagnostics.items()))
    with pytest.raises(ValueError, match=message):
        runner.read_inputs(valid_inputs)


@pytest.mark.parametrize(
    "matrix",
    [
        {"include": []},
        {"include": [{"group_id": "ppu10-0", "board": "ppu10", "nproc_per_node": 2}]},
    ],
)
def test_dispatch_matrix_must_equal_selection(runner, valid_inputs, matrix):
    valid_inputs["DISPATCH_MATRIX"] = json.dumps(matrix)
    with pytest.raises(ValueError, match="matrix"):
        runner.read_inputs(valid_inputs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("CI_RUN_ATTEMPT", "0"),
        ("EXPECTED_BUILD_RUN_ID", "122"),
        ("DEVICE_GROUP_ID", "ppu15-0"),
        ("DEVICE_RESULTS_DIR", "/tmp/results"),
        ("RUNTIME_IMAGE", "registry/image@sha256:" + "b" * 64),
        ("EXPECTED_SAIL_COMMIT", "a" * 40),
    ],
)
def test_invalid_execution_inputs_fail_before_install(
    runner, valid_inputs, name, value
):
    valid_inputs[name] = value
    with pytest.raises(ValueError):
        runner.read_inputs(valid_inputs)


def test_changed_wheel_is_rejected_before_install(runner, valid_inputs):
    wheel = next(Path(valid_inputs["WHEELS_DIR"]).rglob("vllm-*.whl"))
    with wheel.open("ab") as stream:
        stream.write(b"changed bytes")
    with pytest.raises(ValueError, match="hashes"):
        runner.read_inputs(valid_inputs)


def test_selection_cli_with_verified_wheels(runner, valid_inputs, tmp_path):
    changes_path = tmp_path / "changes.json"
    runner.select.write_json(
        changes_path, runner.select.collect_changes(ROOT, None, None, "d" * 40)
    )
    output, github_output = tmp_path / "selected.json", tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci/ppu_ops_select.py"),
            "select",
            "--repo",
            str(ROOT),
            "--config",
            str(tmp_path / "ppu_ops_tests.json"),
            "--environment",
            str(tmp_path / "ppu_ops_environment.json"),
            "--changes",
            str(changes_path),
            "--build-manifest",
            valid_inputs["BUILD_MANIFEST_FILE"],
            "--wheel-manifest",
            str(Path(valid_inputs["WHEELS_DIR"]) / "wheel-manifest.json"),
            "--build-run-id",
            "123",
            "--build-image",
            valid_inputs["EXPECTED_BUILD_IMAGE"],
            "--runtime-image",
            valid_inputs["RUNTIME_IMAGE"],
            "--expected-vllm-commit",
            "c" * 40,
            "--scope",
            "full",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_OUTPUT": str(github_output)},
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "ppu_ops_tests.json").read_bytes())
    runner.select.discover_catalog(config, ROOT)
    assert json.loads(output.read_bytes())["files"] == sorted(
        t["test"]
        for t in config["test_catalog"]
        if runner.select.runnable_boards(config, t)
    )
    assert "has_tests=true" in github_output.read_text()


@pytest.mark.parametrize("pip_index", [None, "", "https://packages.example/simple/"])
@pytest.mark.parametrize(
    "failure",
    [None, "torch", "stack", "install", "pytest", "preflight", "missing-report"],
)
def test_execute_only_publishes_success_after_complete_evidence(
    runner, valid_inputs, tmp_path, monkeypatch, failure, pip_index
):
    # Replace only device/NAS/install boundaries; run real pytest subprocesses and all evidence checks.
    case = tmp_path / "case"
    case.mkdir()
    result, _ = run_case(case, "def test_ok(): pass\n")
    assert result.returncode == 0
    repo = case / "isolated"
    (repo / "tests/e2e/test_first.py").write_text(
        "def test_first(): "
        + ("assert False" if failure == "pytest" else "pass")
        + "\n"
    )
    (repo / "tests/e2e/test_second.py").write_text("def test_second(): pass\n")
    entries = [
        {
            "id": n,
            "test": f"tests/e2e/test_{n}.py",
            "boards": ["ppu10"],
            "requires": ["torch", "triton", "vllm", "vllm_sail"],
            "board_overrides": {},
        }
        for n in ("first", "second")
    ]
    raw, config_path, config, baseline, manifest = runner.read_inputs(valid_inputs)
    (repo / "tests/e2e/test_multi.py").write_text(
        "PPU_CI_MIN_DEVICES = 2\ndef test_multi(): assert False\n"
    )
    config["test_catalog"] = [
        *entries,
        {
            "id": "multi",
            "test": "tests/e2e/test_multi.py",
            "boards": ["ppu10"],
            "requires": [
                "missing_multi_device_dependency",
                "torch",
                "triton",
                "vllm",
                "vllm_sail",
            ],
            "board_overrides": {},
            "min_devices": 2,
        },
    ]
    config_path.write_text(json.dumps(config))
    original = json.loads(raw)
    original.update(
        test_ids=[t["id"] for t in entries],
        files=[t["test"] for t in entries],
        **runner.select.route_tests(config, [t["id"] for t in entries]),
    )
    raw = json.dumps(original).encode()
    output = tmp_path / "nas"
    output.mkdir()
    probe = dict(run_id="123", run_attempt="1", group_id="ppu10-0", nonce="test")
    runner.select.write_json(output / "cpu-probe.json", probe)
    env = {
        **os.environ,
        **valid_inputs,
        "DEVICE_RESULTS_DIR": str(output),
        "PIP_INDEX_URL": "https://unreachable.example/simple/",
    }
    env.pop("SAIL_PIP_INDEX_URL", None)
    if pip_index is not None:
        env["SAIL_PIP_INDEX_URL"] = pip_index
    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "result_path", lambda *a, **kw: output)
    monkeypatch.setattr(runner, "require_nas", lambda mount: None)
    monkeypatch.setattr(
        runner,
        "read_inputs",
        lambda env: (raw, config_path, config, baseline, manifest),
    )
    original_clean = runner.clean_environment

    def child_environment(env, repo, isolated):
        child = original_clean(env, repo, isolated)
        child["PYTHONPATH"] = str(case / "site-packages")
        return child

    monkeypatch.setattr(runner, "clean_environment", child_environment)
    original_run = subprocess.run
    installed = []
    stack_commands = []

    def device_boundary(command, **kwargs):
        if command[1].endswith("ppu_image_stack.py"):
            stack_commands.append(command[2])
            assert command[3:7] == [
                "--sdk",
                "/usr/local/PPU_SDK",
                "--output",
                str(output / "image-stack.json"),
            ]
            if command[2] == "capture":
                assert not installed
                assert command[7:] == ["--constraints", kwargs["env"]["PIP_CONSTRAINT"]]
                if failure == "torch":
                    raise subprocess.CalledProcessError(1, command)
            elif command[2] == "verify":
                assert len(installed) == 2
                if failure == "stack":
                    raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        if command[1:3] == ["-m", "pip"]:
            assert stack_commands == ["capture"]
            assert kwargs["env"]["PIP_CONSTRAINT"] == str(
                output / "image-constraints.txt"
            )
            installed.append(command)
            if failure == "install" and "-r" in command:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        if command[1] == "-c":
            assert "missing_multi_device_dependency" not in json.loads(
                kwargs["env"]["PPU_CI_REQUIRES"]
            )
            if failure == "preflight":
                raise subprocess.CalledProcessError(1, command)
            observed = dict(
                runtime={
                    "capability": [8, 0],
                    "runtime_env": {"VLLM_SAIL_USE_PLA": True},
                },
                packages=dict(torch="2", triton="3", pla="1"),
                sdk_output="HGCC 2.2",
                smi_output="Driver Version: 1.2.3",
            )
            runner.select.write_json(
                Path(kwargs["env"]["PPU_CI_OBSERVATION"]), observed
            )
            return subprocess.CompletedProcess(command, 0)
        if command[1:3] == ["-m", "pytest"]:
            effective_config = Path(command[command.index("--ppu-ci-config") + 1])
            assert effective_config.parent == Path(kwargs["cwd"])
            assert (
                json.loads(effective_config.read_bytes())["test_catalog"]
                == config["test_catalog"]
            )
            assert "tests/e2e/test_multi.py" not in command
        completed = original_run(command, **kwargs)
        if failure == "missing-report" and "tests/e2e/test_first.py" in command:
            (output / "nodes/first.json").rename(output / "nodes/unexpected.json")
        return completed

    monkeypatch.setattr(subprocess, "run", device_boundary)
    code = runner.execute(env)
    summary = json.loads((output / "summary.json").read_bytes())
    assert code == (0 if failure is None else 1)
    assert summary["status"] == ("success" if failure is None else "failure")
    assert stack_commands == (
        ["capture"] if failure in {"torch", "install"} else ["capture", "verify"]
    )
    if failure == "torch":
        assert not installed
        assert summary["failure_stage"] == "install"
        assert not (output / "preflight.log").exists()
        return
    assert "--no-deps" in installed[0] and "--force-reinstall" in installed[0]
    assert len(installed[0][installed[0].index("--force-reinstall") + 1 :]) == 2
    assert installed[1] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--index-url",
        pip_index or "https://mirrors.aliyun.com/pypi/simple/",
        "-r",
        str(repo / "requirements/dev.txt"),
    ]
    assert len(installed) == 2
    if failure not in {"preflight", "install", "stack"}:
        assert (
            json.loads((output / "nodes/second.json").read_bytes())["status"]
            == "success"
        )
    else:
        assert summary["failure_stage"] == (
            "install" if failure == "stack" else failure
        )
        assert not (output / "nodes").exists()
        if failure in {"install", "stack"}:
            assert not (output / "preflight.log").exists()


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_install(site):
    contents = {
        "torch/__init__.py": "from types import SimpleNamespace\n__version__='1'\ncuda=SimpleNamespace(is_available=lambda:True, device_count=lambda:1, get_device_capability=lambda i:(8,0))\n",
        "vllm/__init__.py": "",
        "vllm/platforms/__init__.py": "from types import SimpleNamespace\ncurrent_platform=SimpleNamespace(is_ppu=lambda:True)\n",
        "vllm_sail/__init__.py": "",
        "vllm_sail/envs.py": "import os\nVLLM_SAIL_USE_PLA=VLLM_PPU_USE_PLA=os.environ.get('VLLM_SAIL_USE_PLA')=='1'\n",
        "vllm_sail/native/__init__.py": "",
        "vllm_sail/native/extensions.py": "def import_kernels(*, strict=False):\n    assert strict\n    return ('fake-provider',)\n",
        "triton/__init__.py": "__version__='1'\n",
        "pla/__init__.py": "__version__='1'\n",
    }
    for name in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C", "_cumem_allocator"):
        contents[f"vllm_sail/{name}.py"] = ""
    for name, text in contents.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def run_case(
    tmp_path,
    text,
    *,
    overrides=None,
    board="ppu10",
    real_device=True,
    pla="1",
    shadow=False,
    extra_args=(),
):
    assert PLUGIN.is_file(), "Missing explicit pytest CI plugin"
    isolated, site = tmp_path / "isolated", tmp_path / "site-packages"
    for name in ("tests", "tests/support", "tests/e2e"):
        path = isolated / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("")
    shutil.copyfile(PLUGIN, isolated / "tests/support/ppu_ci.py")
    (isolated / "tests/conftest.py").write_text(f"HAS_REAL_DEVICE = {real_device!r}\n")
    file = "tests/e2e/test_sample.py"
    (isolated / file).write_text(text)
    (isolated / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath=["."]\n'
    )
    fake_install(isolated if shadow else site)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "test_catalog": [
                    {
                        "id": "sample",
                        "test": file,
                        "boards": ["ppu10", "ppu15"],
                        "board_overrides": overrides or {},
                        "requires": ["torch", "triton", "vllm", "vllm_sail"],
                    }
                ]
            }
        )
    )
    report = tmp_path / "nodes.json"
    args = [
        "-c",
        "pyproject.toml",
        "-p",
        "tests.support.ppu_ci",
        file,
        "--ppu-ci-config",
        str(config),
        "--ppu-ci-board",
        board,
        "--ppu-ci-report",
        str(report),
        "--junitxml",
        str(tmp_path / "junit.xml"),
        *extra_args,
    ]
    command = f"import sys; sys.path.insert(0, {str(site)!r}); import pytest; sys.exit(pytest.main({args!r}))"
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "VLLM_SAIL_USE_PLA": pla,
        "PPU_CI_CHECKOUT": str(ROOT),
        "PPU_CI_ISOLATED_ROOT": str(isolated),
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=isolated,
        env=env,
        capture_output=True,
        text=True,
    )
    assert report.is_file(), result.stdout + result.stderr
    return result, json.loads(report.read_text())


@pytest.mark.parametrize(
    "text,success",
    [
        ("def test_ok(): pass\n", True),
        ("def test_bad(): assert False\n", False),
        (
            "import pytest\ndef test_mock(): pass\ndef test_numeric(): pytest.skip('missing SDK')\n",
            False,
        ),
        ("import pytest\npytest.skip('module', allow_module_level=True)\n", False),
        ("import pytest\npytest.importorskip('not_a_real_sdk_package')\n", False),
        ("import pytest\n@pytest.mark.xfail\ndef test_bad(): assert False\n", False),
        ("import pytest\n@pytest.mark.xfail\ndef test_ok(): pass\n", False),
        (
            "import pytest\n@pytest.fixture\ndef bad(): assert False\ndef test_bad(bad): pass\n",
            False,
        ),
        (
            "import pytest\n@pytest.fixture\ndef bad():\n yield\n assert False\ndef test_bad(bad): pass\n",
            False,
        ),
        ("", False),
        ("raise RuntimeError('collection')\n", False),
    ],
)
def test_pytest_node_guard(tmp_path, text, success):
    result, report = run_case(tmp_path, text)
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert (report["status"] == "success") is success
    if success:
        assert report["applicable"] == ["tests/e2e/test_sample.py::test_ok"]
        assert report["nodes"][report["applicable"][0]]["call"]["outcome"] == "passed"
    if "missing SDK" in text:
        assert "missing SDK" in json.dumps(report)


@pytest.mark.parametrize("damage", ["missing", "source-path", "conftest-path", "pla"])
def test_node_evidence_rechecks_installation_paths(tmp_path, damage):
    result, report = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    runtime = report["runtime"]
    if damage == "missing":
        runtime["paths"].pop("vllm_sail._C")
    elif damage == "source-path":
        runtime["paths"]["vllm"] = str(ROOT / "vllm/__init__.py")
    elif damage == "conftest-path":
        runtime["paths"]["conftest"] = "/other/tests/conftest.py"
    else:
        runtime["runtime_env"]["VLLM_SAIL_USE_PLA"] = False
    entry = json.loads((tmp_path / "config.json").read_bytes())["test_catalog"][0]
    with pytest.raises(ValueError):
        load(PLUGIN).validate_node_report(report, entry, "ppu10")


def test_hardware_exclusion_is_not_pass(tmp_path):
    result, report = run_case(
        tmp_path,
        "def test_general(): pass\ndef test_fp8(): assert False\n",
        overrides={"test_fp8": ["ppu15"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(report["collected"]) == 2
    assert len(report["hardware_excluded"]) == 1
    assert len(report["applicable"]) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"overrides": {"test_typo": ["ppu15"]}},
        {"real_device": False},
        {"pla": "0"},
        {"shadow": True},
        {"extra_args": ("-k", "general")},
    ],
)
def test_runtime_or_collection_violation(tmp_path, options):
    result, report = run_case(
        tmp_path, "def test_general(): pass\ndef test_other(): pass\n", **options
    )
    assert result.returncode != 0
    assert report["status"] == "failure"
    assert report["errors"]


def test_test_local_pla_monkeypatch_does_not_invalidate_startup(tmp_path):
    result, report = run_case(
        tmp_path,
        "def test_toggle(monkeypatch):\n import vllm_sail.envs as envs\n monkeypatch.setattr(envs, 'VLLM_SAIL_USE_PLA', False)\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["status"] == "success"
