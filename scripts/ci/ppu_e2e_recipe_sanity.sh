#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Container entrypoint for recipe fast_sanity / model_sanity verification.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${HF_HOME:?HF_HOME must be set}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"

unset VLLM_VERSION 2>/dev/null || true
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
RECIPE_PATH="${RECIPE_PATH:?RECIPE_PATH is required}"
POLICY_PATH="${POLICY_PATH:?POLICY_PATH is required}"
TIER="${TIER:-fast_sanity}"
PRESET="${PRESET:-}"
LAYER_MODE="${LAYER_MODE:-auto}"
VLLM_PORT="${VLLM_PORT:-9985}"
PPU_CI_QUEUE_S="${PPU_CI_QUEUE_S:-0}"

mkdir -p "${RESULTS_DIR}"
RESULTS_DIR="$(cd -- "${RESULTS_DIR}" && pwd)"
rm -f -- "${RESULTS_DIR}/provenance.json"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

export PPU_RECIPE_PHASE_FILE="${RESULTS_DIR}/recipe-phase.txt"
record_phase() {
    printf '%s\n' "$1" >"${PPU_RECIPE_PHASE_FILE}"
}

record_phase wheel_verification
WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
verified_wheels="$(python -IB - "${ROOT_DIR}" "${WHEELS_DIR}" "${RESULTS_DIR}" \
    "${RECIPE_PATH}" "${POLICY_PATH}" "${TIER}" "${REQUIRE_PR_ARTIFACT:-false}" \
    "${PPU_RECIPE_SOURCE_SHA:?PPU_RECIPE_SOURCE_SHA is required}" \
    "${EXPECTED_VLLM_COMMIT:?EXPECTED_VLLM_COMMIT is required}" \
    "${EXPECTED_SAIL_COMMIT:?EXPECTED_SAIL_COMMIT is required}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

(root, wheels, results, recipe, policy, tier, require_pr, checkout, vllm, sail) = sys.argv[1:]
if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (checkout, vllm, sail)):
    raise SystemExit("Recipe provenance requires exact checkout and wheel commit SHAs")
if tier not in {"fast_sanity", "model_sanity"} or require_pr not in {"true", "false"}:
    raise SystemExit("Invalid recipe tier or PR artifact mode")
exact_source_match = checkout == sail
if not exact_source_match and (require_pr != "true" or tier != "fast_sanity"):
    raise SystemExit("SAIL wheel revision must match the recipe checkout outside PR fast_sanity")

root = Path(root).resolve()
wheels = Path(wheels)
sys.path.insert(0, str(root / "scripts/ci"))
from ppu_wheel_manifest import verify

try:
    manifest = verify(wheels, vllm_commit=vllm, sail_commit=sail)
except Exception as error:
    raise SystemExit(f"Wheel manifest verification failed ({type(error).__name__})") from None


def source_record(relative):
    if Path(relative).is_absolute():
        raise ValueError("source paths must be checkout-relative")
    path = (root / relative).resolve(strict=True)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


try:
    recipes = root / "tests/ppu/recipes"
    if not recipes.is_dir() or recipes.is_symlink():
        raise ValueError("invalid recipes tree")
    tree_digest = hashlib.sha256()
    for path in sorted(recipes.rglob("*"), key=lambda path: path.relative_to(recipes).as_posix()):
        if path.is_symlink():
            raise ValueError("symlink in recipes tree")
        if path.is_file():
            tree_digest.update(path.relative_to(recipes).as_posix().encode("utf-8") + b"\0")
            tree_digest.update(hashlib.sha256(path.read_bytes()).digest())

    provenance = {
        "schema": 1,
        "checkout_sha": checkout,
        "commits": {name: manifest["commits"][name] for name in ("vllm", "vllm-sail")},
        "wheels": manifest["wheels"],
        "tier": tier,
        "exact_source_match": exact_source_match,
        "recipe": source_record(recipe),
        "recipes_tree_sha256": tree_digest.hexdigest(),
        "policy": source_record(policy),
        "runner_sources": [
            source_record(path) for path in (
                "scripts/ci/ppu_e2e_recipe_sanity.sh",
                "scripts/ci/ppu_recipe.py",
                "scripts/ci/ppu_recipe_runner.py",
                "scripts/eval/ppu_gsm8k_eval.py",
            )
        ],
    }
    wheel_paths = [str(wheels / manifest["wheels"][name]["file"]) for name in ("vllm", "vllm-sail")]
    if any("\n" in path for path in wheel_paths):
        raise ValueError("newline in wheel path")
    (Path(results) / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
except (OSError, ValueError) as error:
    raise SystemExit(f"Recipe provenance recording failed ({type(error).__name__})") from None

print(*wheel_paths, sep="\n")
PY
)"
vllm_wheel="${verified_wheels%%$'\n'*}"
sail_wheel="${verified_wheels#*$'\n'}"

record_phase runtime_dependencies
export PIP_CONSTRAINT="/tmp/ppu-recipe-image-constraints.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output "/tmp/ppu-recipe-image-stack.json" \
    --constraints "${PIP_CONSTRAINT}"
if [[ "${TIER}" == "model_sanity" ]]; then
    numpy_version="$(python -c 'from importlib.metadata import version; print(version("numpy"))')"
    printf 'numpy===%s\npyarrow<26\n' "${numpy_version}" >> "${PIP_CONSTRAINT}"
fi

python -m pip install --no-deps --force-reinstall "${vllm_wheel}"
python -m pip install --no-deps --force-reinstall "${sail_wheel}"
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    python -m pip install -r "${ROOT_DIR}/requirements/ci-runtime.txt"
if [[ "${TIER}" == "model_sanity" ]]; then
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
        python -m pip install "evalscope==1.14.1"
    python -c 'from evalscope import TaskConfig, run_task'
fi

python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "/tmp/ppu-recipe-image-stack.json"

cd /tmp
unset PYTHONPATH

record_phase platform_check
python - <<'PY'
import torch
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), f"expected PPU platform, got {current_platform}"
assert torch.cuda.is_available(), "PPU device required"
PY

record_phase model_cache_check
python - "${MODEL_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], local_files_only=True)
PY

record_phase recipe_validation
python "${ROOT_DIR}/scripts/ci/ppu_recipe.py" validate --all --require-vllm-parser

runner_args=(
    --recipe "${ROOT_DIR}/${RECIPE_PATH}"
    --policy "${ROOT_DIR}/${POLICY_PATH}"
    --tier "${TIER}"
    --results-dir "${RESULTS_DIR}/results"
    --port "${VLLM_PORT}"
    --layer-mode "${LAYER_MODE}"
    --vllm-parser required
    --queue-s "${PPU_CI_QUEUE_S}"
)
if [[ -n "${PRESET}" ]]; then
    runner_args+=(--preset "${PRESET}")
fi

python "${ROOT_DIR}/scripts/ci/ppu_recipe_runner.py" "${runner_args[@]}"
