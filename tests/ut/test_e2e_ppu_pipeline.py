# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "ci" / "ppu_e2e_smoke.sh"
ACCURACY = ROOT / "scripts" / "ci" / "ppu_e2e_accuracy.sh"
RUNTIME = ROOT / "scripts" / "ci" / "ppu_runtime_container.sh"
SELECT = ROOT / "scripts" / "ci" / "ppu_e2e_select.py"
MODELS = ROOT / "scripts" / "ci" / "ppu_e2e_models.json"
MODEL_WORKFLOW = ROOT / ".github" / "workflows" / "_ppu-model-tests.yaml"
ACCURACY_WORKFLOW = ROOT / ".github" / "workflows" / "_ppu-accuracy-tests.yaml"
NIGHTLY_WORKFLOW = ROOT / ".github" / "workflows" / "ppu-accuracy-nightly.yaml"
KERNEL_PERF_WORKFLOW = ROOT / ".github" / "workflows" / "_ppu-kernel-perf.yaml"
KERNEL_PERF_LABEL_WORKFLOW = ROOT / ".github" / "workflows" / "kernel-perf-label.yaml"
PERF_REGRESSION_WORKFLOW = ROOT / ".github" / "workflows" / "perf-regression.yaml"
PERF_REGRESSION_SCRIPT = ROOT / "scripts" / "ci" / "perf_regression.sh"
MODEL_PERF_TEST = ROOT / "tests" / "e2e" / "perf" / "test_model_perf.py"
DEBUG_WORKFLOW = ROOT / ".github" / "workflows" / "ppu-debug.yaml"


def _load_select():
    spec = importlib.util.spec_from_file_location("ppu_e2e_select", SELECT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _catalog() -> list[dict]:
    return json.loads(MODELS.read_text())


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# --- model catalog -----------------------------------------------------------


def test_models_catalog_is_partitioned_by_category_and_contract() -> None:
    catalog = _catalog()
    assert isinstance(catalog, list) and len(catalog) == 5
    common = {"key", "served_name", "board", "tp"}
    for entry in catalog:
        assert common <= set(entry), entry
        category = entry.get("category", "smoke")
        assert category in {"smoke", "accuracy"}
        assert "model_id" in entry and "checkpoint" not in entry
    keys = [entry["key"] for entry in catalog]
    assert len(keys) == len(set(keys)), "model keys must be unique"


def test_catalog_keeps_smoke_and_accuracy_subsets_exact() -> None:
    by_category = {
        category: {
            entry["key"]
            for entry in _catalog()
            if entry.get("category", "smoke") == category
        }
        for category in ("smoke", "accuracy")
    }
    assert by_category["smoke"] == {"qwen3.8-27b", "qwen3.8-27b-m890p"}
    assert by_category["accuracy"] == {
        "deepseek-v4-int8",
        "deepseek-v4-mxfp4",
        "minimax-m3-fp8",
    }


@pytest.mark.parametrize(
    ("key", "board"),
    [
        ("qwen3.8-27b", "OAM-810E"),
        ("qwen3.8-27b-m890p", "ZW-M890P"),
    ],
)
def test_qwen_smoke_catalog_targets_both_boards(key: str, board: str) -> None:
    qwen = {entry["key"]: entry for entry in _catalog()}[key]
    assert qwen["model_id"] == "Qwen/Qwen3.8-27B"
    assert qwen["board"] == board
    selected = _load_select().build_matrix(_catalog(), key)["include"][0]
    assert selected["checkpoint"] == selected["model_id"]
    assert int(qwen["tp"]) == 2
    assert qwen["served_name"] == "qwen3.8-27b"
    assert qwen["extra_serve_flags"] == "--trust-remote-code"


def test_accuracy_catalog_keeps_model_ids_and_per_model_tp() -> None:
    accuracy = {
        entry["key"]: entry for entry in _catalog() if entry["category"] == "accuracy"
    }
    assert accuracy["deepseek-v4-int8"]["model_id"] == (
        "T-HEAD/DeepSeek-V4-Flash-0731-W8A8-INT8"
    )
    assert accuracy["deepseek-v4-int8"]["board"] == "OAM-810E"
    assert accuracy["deepseek-v4-mxfp4"]["board"] == "ZW-M890P"
    assert {int(entry["tp"]) for entry in accuracy.values()} == {8}


# --- matrix selector ---------------------------------------------------------


def test_selector_defaults_to_smoke_and_legacy_entries_are_smoke() -> None:
    select = _load_select()
    matrix = select.build_matrix(_catalog(), "")
    assert [entry["key"] for entry in matrix["include"]] == [
        "qwen3.8-27b",
        "qwen3.8-27b-m890p",
    ]
    legacy = [{"key": "legacy", "checkpoint": "/tmp/model"}]
    assert select.build_matrix(legacy, "")["include"] == legacy


def test_selector_filters_smoke_without_leaking_accuracy() -> None:
    select = _load_select()
    matrix = select.build_matrix(_catalog(), "deepseek-v4-int8,qwen3.8-27b-m890p")
    assert [entry["key"] for entry in matrix["include"]] == ["qwen3.8-27b-m890p"]


def test_accuracy_matrix_uses_only_accuracy_and_eval_keys() -> None:
    select = _load_select()
    matrix = select.build_accuracy_matrix(
        _catalog(), "deepseek-v4-mxfp4,minimax-m3-fp8"
    )
    assert matrix == {
        "include": [
            {
                "model_key": "deepseek_v4_mxfp4",
                "model_id": (
                    "T-HEAD/DeepSeek-V4-Flash-0731-MoE-Quant-W-MXFP4-A-MXFP4-"
                    "Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken"
                ),
                "board": "ZW-M890P",
                "ppu_count": 8,
            },
            {
                "model_key": "minimax-m3-fp8",
                "model_id": ("T-HEAD/MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"),
                "board": "ZW-M890P",
                "ppu_count": 8,
            },
        ]
    }


def test_accuracy_matrix_default_contains_every_accuracy_model() -> None:
    select = _load_select()
    matrix = select.build_accuracy_matrix(_catalog(), "")
    assert {entry["model_key"] for entry in matrix["include"]} == {
        "deepseek_v4_int8",
        "deepseek_v4_mxfp4",
        "minimax-m3-fp8",
    }


def test_selector_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown model key"):
        _load_select().build_matrix(_catalog(), "does-not-exist")


def test_selector_cli_emits_separate_smoke_and_accuracy_outputs() -> None:
    result = subprocess.run(
        ["python3", str(SELECT), ""],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in result.stdout.splitlines())
    smoke = json.loads(outputs["matrix"])
    accuracy = json.loads(outputs["accuracy_matrix"])
    assert len(smoke["include"]) == 2
    assert len(accuracy["include"]) == 3


# --- scripts -----------------------------------------------------------------


@pytest.mark.parametrize("script", [SMOKE, ACCURACY, RUNTIME])
def test_e2e_shell_scripts_have_valid_bash_syntax(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_script_preserves_required_e2e_contract() -> None:
    text = SMOKE.read_text()
    for marker in (
        "set -Eeuo pipefail",
        "--no-deps",
        "--force-reinstall",
        "from vllm_sail import collect_env",
        "vllm serve",
        "/v1/chat/completions",
        "Application startup complete",
        "kill -0",
        "ppu-smi",
        "check_vllm_compatibility(force=True)",
        "ppu_wheel_manifest.py",
    ):
        assert marker in text, marker


def test_smoke_resolves_model_id_only_from_offline_hf_cache() -> None:
    text = SMOKE.read_text()
    assert 'MODEL_ID="${MODEL_ID:?MODEL_ID is required}"' in text
    assert 'HF_HOME="${HF_HOME:?HF_HOME is required}"' in text
    assert "local_files_only=True" in text
    assert "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in text
    assert "snapshot_download(repo_id=sys.argv[1]" in text


def test_accuracy_script_uses_writable_results_with_read_only_source() -> None:
    text = ACCURACY.read_text()
    assert '--golden-dir "${RESULTS_DIR}/golden"' in text
    assert "cd /tmp" in text
    assert "unset PYTHONPATH" in text
    assert 'cp -fv "${wheel_so[@]}" "${ROOT_DIR}/vllm_sail/"' not in text
    assert 'python "${ROOT_DIR}/scripts/eval/ppu_gsm8k_eval.py"' in text
    # A repo missing from the offline cache must fail before the server boots
    # instead of twelve minutes into engine startup.
    probe = "snapshot_download(repo_id=sys.argv[1]"
    assert probe in text
    assert "local_files_only=True" in text
    assert text.index(probe) < text.index(
        'python "${ROOT_DIR}/scripts/eval/ppu_gsm8k_eval.py"'
    )


def test_accuracy_script_seeds_reviewed_goldens_into_the_writable_tree() -> None:
    """Without the seed every run reports "created" and never compares."""
    text = ACCURACY.read_text()
    seed = 'GOLDEN_SEED_DIR="${GOLDEN_SEED_DIR:-/source/tests/e2e/models/_golden}"'
    assert seed in text
    # The checked-out source is mounted read-only, so the copy must land in
    # RESULTS_DIR and the evaluator must read the copy.
    assert 'cp -f "${golden}" "${golden_dir}/"' in text
    assert text.index(seed) < text.index('--golden-dir "${RESULTS_DIR}/golden"')
    assert "threshold-only run" in text


def test_runtime_wrapper_exposes_accuracy_mode_and_environment() -> None:
    text = RUNTIME.read_text()
    assert "pin-ut|model-test|model-accuracy|kernel-perf" in text
    assert "/source/scripts/ci/ppu_e2e_accuracy.sh" in text
    for marker in (
        "HF_HOME",
        "HF_HUB_OFFLINE=1",
        "GSM8K_DATASET_PATH",
        "MODEL_KEY",
        "RESULTS_DIR=/artifacts",
    ):
        assert marker in text


def test_evaluator_requires_dataset_environment_without_hardcoded_paths() -> None:
    text = (ROOT / "scripts" / "eval" / "ppu_gsm8k_eval.py").read_text()
    assert "GSM8K_DATASET_PATH environment variable is required" in text
    assert "/ppusw/" not in text
    assert "/nas_aisw/" not in text


# --- workflows ---------------------------------------------------------------


def test_model_workflow_is_call_only_and_uses_docker_runner() -> None:
    workflow = _workflow(MODEL_WORKFLOW)
    assert set(workflow[True]) == {"workflow_call"}
    assert workflow["jobs"]["prepare"]["runs-on"] == "ubuntu-latest"
    assert workflow["jobs"]["e2e"]["runs-on"] == "8-ppu-m890p"
    assert "container" not in workflow["jobs"]["e2e"]
    text = MODEL_WORKFLOW.read_text()
    assert 'ppu_runtime_container.sh" model-test' in text
    assert "ppu-scheduler-action" not in text


@pytest.mark.parametrize(
    ("filename", "event", "expected"),
    [
        ("_ppu-model-tests.yaml", "workflow_call", "qwen3.8-27b-m890p"),
        ("ppu-debug.yaml", "workflow_dispatch", "qwen3.8-27b-m890p"),
    ],
)
def test_smoke_workflow_defaults_target_available_m890_runner(
    filename: str, event: str, expected: str
) -> None:
    config = _workflow(MODEL_WORKFLOW.parent / filename)
    models = config[True][event]["inputs"]["models"]["default"]
    assert models == expected
    matrix = _load_select().build_matrix(_catalog(), models)
    assert [(entry["key"], entry["board"]) for entry in matrix["include"]] == [
        ("qwen3.8-27b-m890p", "ZW-M890P")
    ]


def test_accuracy_workflow_is_call_only_and_uses_docker_runner() -> None:
    workflow = _workflow(ACCURACY_WORKFLOW)
    assert workflow["name"] == "PPU accuracy tests"
    assert set(workflow[True]) == {"workflow_call"}
    assert set(workflow["jobs"]) == {"prepare", "accuracy"}
    assert workflow["jobs"]["prepare"]["runs-on"] == "ubuntu-latest"
    accuracy = workflow["jobs"]["accuracy"]
    assert accuracy["runs-on"] == "8-ppu-m890p"
    assert "container" not in accuracy
    text = ACCURACY_WORKFLOW.read_text()
    assert 'ppu_runtime_container.sh" model-accuracy' in text
    assert "ppu-scheduler-action" not in text
    assert text.count("actions/download-artifact@v4") == 1


def test_accuracy_entrypoints_are_the_manual_debug_and_the_nightly_runs() -> None:
    debug = _workflow(DEBUG_WORKFLOW)
    assert set(debug[True]) == {"workflow_dispatch"}
    inputs = debug[True]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "build_run_id",
        "wheel_source_sha",
        "target",
        "models",
        "accuracy_models",
    }
    assert "accuracy-tests" in inputs["target"]["options"]
    assert "kernel-perf" in inputs["target"]["options"]
    assert debug["jobs"]["accuracy-tests"]["uses"] == (
        "./.github/workflows/_ppu-accuracy-tests.yaml"
    )

    nightly = _workflow(NIGHTLY_WORKFLOW)
    assert set(nightly[True]) == {"workflow_dispatch", "schedule"}
    assert nightly[True]["schedule"] == [{"cron": "0 12 * * *"}]
    assert set(nightly["jobs"]["accuracy"]["with"]) >= {
        "build_run_id",
        "models",
        "pr_head_sha",
        "wheel_source_sha",
    }
    assert nightly["jobs"]["accuracy"]["uses"] == (
        "./.github/workflows/_ppu-accuracy-tests.yaml"
    )

    assert debug["jobs"]["kernel-perf"]["uses"] == (
        "./.github/workflows/_ppu-kernel-perf.yaml"
    )

    # kernel-perf is no longer part of the per-PR gate.
    automatic = (ROOT / ".github/workflows/ppu-build-and-test.yaml").read_text()
    assert "_ppu-accuracy-tests.yaml" not in automatic
    assert "model-accuracy" not in automatic
    assert "_ppu-kernel-perf.yaml" not in automatic


def test_accuracy_nightly_is_serial_and_wheel_sourced_from_the_build_commit() -> None:
    nightly = _workflow(NIGHTLY_WORKFLOW)
    accuracy = nightly["jobs"]["accuracy"]
    # Every lane needs all eight devices and the runtime wrapper holds an
    # exclusive lock, so a parallel matrix would abort instead of queueing.
    assert accuracy["strategy"]["max-parallel"] == 1
    assert accuracy["strategy"]["fail-fast"] is False
    assert accuracy["strategy"]["matrix"] == (
        "${{ fromJson(needs.select.outputs.matrix) }}"
    )
    # The nightly builds its own wheels, so github.sha is both the source
    # checkout and the commit whose artifact names carry this run.
    assert nightly["jobs"]["build"]["uses"] == (
        "./.github/workflows/ppu-build-and-test.yaml"
    )
    assert nightly["jobs"]["build"]["with"] == {
        "pr_head_sha": "${{ github.sha }}",
        "skip_device_tests": True,
    }
    assert accuracy["needs"] == ["build", "select"]
    # Checkout and wheels must describe the same SAIL tree.
    assert accuracy["with"]["pr_head_sha"] == "${{ github.sha }}"
    assert accuracy["with"]["wheel_source_sha"] == "${{ github.sha }}"
    assert accuracy["with"]["build_run_id"] == "${{ github.run_id }}"
    assert accuracy["with"]["pr_mode"] is False
    # The accuracy work runs inside the reusable call, not on a hosted shell.
    assert "runs-on" not in accuracy
    text = NIGHTLY_WORKFLOW.read_text()
    # Each model must reach the reusable workflow with one catalog key.
    matrix = _load_select().build_accuracy_matrix(_catalog(), "")
    assert len(matrix["include"]) == 3
    assert "models: ${{ matrix.model }}" in text


def test_workflows_pull_wheels_by_run_id_and_keep_python_frontend() -> None:
    for path in (MODEL_WORKFLOW, ACCURACY_WORKFLOW):
        text = path.read_text()
        assert "build_run_id:" in text
        assert "actions/download-artifact@v4" in text
        assert "run-id: ${{ inputs.build_run_id }}" in text
        assert "ppu-wheels-${{ inputs.build_run_id }}" in text
        assert "VLLM_VERSION=" not in text
    smoke_text = SMOKE.read_text()
    assert "VLLM_USE_RUST_FRONTEND=0" in smoke_text
    assert "VLLM_USE_RUST_BENCH=0" in smoke_text


def test_kernel_perf_label_workflow_structure() -> None:
    """kernel-perf-label.yaml triggers independently via the kernel-perf label."""
    workflow = _workflow(KERNEL_PERF_LABEL_WORKFLOW)
    assert workflow["name"] == "Kernel Perf Gate"
    assert set(workflow[True]) == {"pull_request"}
    pr = workflow[True]["pull_request"]
    assert set(pr["types"]) == {"labeled", "synchronize", "reopened"}

    # Fresh-build design: build + kernel-perf (no resolve-build)
    assert set(workflow["jobs"]) == {"build", "kernel-perf"}

    build = workflow["jobs"]["build"]
    # Build job is a reusable workflow call to _ppu-build.yaml
    assert build["uses"] == "./.github/workflows/_ppu-build.yaml"
    # Guard: label + not draft + correct label on labeled event
    assert "kernel-perf" in build["if"]
    assert "!github.event.pull_request.draft" in build["if"]
    assert "github.event.label.name == 'kernel-perf'" in build["if"]
    # Build checks out and packages the PR head
    assert build["with"]["pr_head_sha"] == ("${{ github.event.pull_request.head.sha }}")

    kp = workflow["jobs"]["kernel-perf"]
    assert kp["needs"] == "build"
    assert kp["uses"] == "./.github/workflows/_ppu-kernel-perf.yaml"
    assert kp["with"]["pr_mode"] is True
    assert kp["with"]["pr_head_sha"] == ("${{ github.event.pull_request.head.sha }}")
    # build_run_id uses the caller's own run_id (fresh build, not external)
    assert kp["with"]["build_run_id"] == "${{ github.run_id }}"
    assert kp["with"]["wheel_source_sha"] == (
        "${{ github.event.pull_request.head.sha }}"
    )

    # This workflow is NOT part of ppu-build-and-test gate
    gate_text = (ROOT / ".github/workflows/ppu-build-and-test.yaml").read_text()
    assert "kernel-perf-label" not in gate_text


def test_weekly_perf_builds_current_commit_and_never_resolves_old_runs() -> None:
    workflow = _workflow(PERF_REGRESSION_WORKFLOW)
    assert set(workflow[True]) == {"workflow_dispatch", "schedule"}
    assert workflow[True]["schedule"] == [{"cron": "0 20 * * 1"}]
    assert "build_run_id" not in workflow[True]["workflow_dispatch"]["inputs"]
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["cache-mode"] == "none"
    assert set(workflow["jobs"]) == {"build", "perf"}

    build = workflow["jobs"]["build"]
    assert build["uses"] == "./.github/workflows/_ppu-build.yaml"
    assert build["with"] == {"pr_head_sha": "${{ github.sha }}"}

    perf = workflow["jobs"]["perf"]
    assert perf["needs"] == "build"
    assert {entry["chip"] for entry in perf["strategy"]["matrix"]["include"]} == {
        "810e",
        "890p",
    }
    download = next(
        step
        for step in perf["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download["with"]["name"] == "ppu-wheels-${{ github.run_id }}"
    assert download["with"]["run-id"] == "${{ github.run_id }}"
    verify = next(
        step
        for step in perf["steps"]
        if step.get("name") == "Verify wheel manifest against current sources"
    )
    assert verify["env"]["EXPECTED_SAIL_COMMIT"] == "${{ github.sha }}"
    assert ".github/vllm-main-verified.commit" in verify["run"]
    assert "--vllm-commit" in verify["run"]
    assert "--sail-commit" in verify["run"]

    text = PERF_REGRESSION_WORKFLOW.read_text()
    assert "ppu-build-and-test.yaml/runs" not in text
    assert "api.github.com" not in text
    assert "latest successful" not in text
    assert "EXPECTED_SAIL_COMMIT=${{ github.sha }}" in text


def test_kernel_perf_workflow_is_call_only_and_uses_docker_runner() -> None:
    workflow = _workflow(KERNEL_PERF_WORKFLOW)
    assert workflow["name"] == "PPU kernel performance"
    assert set(workflow[True]) == {"workflow_call"}
    assert set(workflow["jobs"]) == {"kernel-perf"}
    kp = workflow["jobs"]["kernel-perf"]
    assert kp["runs-on"] == "8-ppu-m890p"
    assert kp["timeout-minutes"] == 60
    assert "container" not in kp
    inputs = workflow[True]["workflow_call"]["inputs"]
    assert inputs["wheel_source_sha"]["required"] is True
    assert inputs["wheel_source_sha"]["type"] == "string"
    runtime = next(
        step
        for step in kp["steps"]
        if "ppu_runtime_container.sh" in step.get("run", "")
    )
    assert runtime["env"]["PPU_CI_WHEEL_SOURCE_SHA"] == "${{ inputs.wheel_source_sha }}"
    text = KERNEL_PERF_WORKFLOW.read_text()
    assert 'ppu_runtime_container.sh" kernel-perf' in text
    assert "ppu-scheduler-action" not in text


def test_runtime_container_kernel_perf_mode() -> None:
    text = RUNTIME.read_text()
    assert "kernel-perf)" in text
    assert "/source/scripts/ci/perf_regression.sh" in text
    # kernel-perf mode injects SUITE and does not require MODEL_PATH/HF_HOME
    lines = text.split("\n")
    in_kernel_block = False
    kernel_block = []
    for line in lines:
        if line.strip() == "kernel-perf)":
            in_kernel_block = True
        elif in_kernel_block and line.strip() == ";;":
            break
        elif in_kernel_block:
            kernel_block.append(line)
    kernel_text = "\n".join(kernel_block)
    assert "SUITE" in kernel_text
    assert "TEST_PATH=tests/e2e/perf/test_kernel_perf.py" in kernel_text
    assert "RESULTS_DIR=/artifacts" in kernel_text
    assert "BASELINE_ROOT=/source/tests/e2e/perf/baselines" in kernel_text
    assert "BASELINE_OUTPUT_ROOT=/artifacts/baselines" in kernel_text
    assert "perf_regression.sh" in kernel_text
    assert "MODEL_PATH" not in kernel_text
    assert "HF_HOME" not in kernel_text
    assert "GSM8K_DATASET_PATH" not in kernel_text


# --- perf regression (model) -------------------------------------------------


def test_perf_regression_workflow_no_hardcoded_hf_paths() -> None:
    """The perf workflow must inherit its cache instead of hardcoding paths."""
    text = PERF_REGRESSION_WORKFLOW.read_text()
    workflow = _workflow(PERF_REGRESSION_WORKFLOW)
    matrix = workflow["jobs"]["perf"]["strategy"]["matrix"]
    for entry in matrix["include"]:
        assert "hf_home" not in entry, f"hf_home leaked into matrix entry: {entry}"
        assert "hf_cache" not in entry, f"hf_cache leaked into matrix entry: {entry}"
    assert "/nas_aisw/datasets/hf_home" not in text
    assert "/ppusw/datasets/hf_home" not in text


def test_perf_regression_workflow_model_lanes_have_offline_protection() -> None:
    """Model lanes must prevent Hugging Face and transformers network access."""
    text = PERF_REGRESSION_WORKFLOW.read_text()
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text


def test_perf_regression_script_exports_transformers_offline() -> None:
    """The model perf runner must export both offline controls."""
    text = PERF_REGRESSION_SCRIPT.read_text()
    assert "TRANSFORMERS_OFFLINE" in text
    assert "export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE" in text


def test_perf_regression_script_does_not_hardcode_hf_paths() -> None:
    """The model perf runner must not contain hardcoded cache paths."""
    text = PERF_REGRESSION_SCRIPT.read_text()
    assert "/nas_aisw/datasets/hf_home" not in text
    assert "/ppusw/datasets/hf_home" not in text


def test_model_perf_uses_model_id_not_checkpoint() -> None:
    """PerfModel must use the PR 88 model_id loading contract."""
    import ast

    text = MODEL_PERF_TEST.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PerfModel":
            field_names = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            assert "model_id" in field_names, "PerfModel must have a model_id field"
            assert "checkpoint" not in field_names, (
                "PerfModel must not have a checkpoint field (use model_id)"
            )
            break
    else:
        raise AssertionError("PerfModel class not found in test_model_perf.py")
