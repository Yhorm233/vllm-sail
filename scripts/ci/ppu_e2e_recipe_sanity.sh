#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Container entrypoint for recipe fast_sanity / model_sanity verification.
set -Eeuo pipefail

CI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PPU_CI_SOURCE_ROOT:-$(cd -- "${CI_SCRIPT_DIR}/../.." && pwd)}"
export HF_HOME="${HF_HOME:?HF_HOME must be set}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"

unset VLLM_VERSION 2>/dev/null || true
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

SKIP_WHEEL_INSTALL="${PPU_CI_SKIP_WHEEL_INSTALL:-false}"
[[ "${SKIP_WHEEL_INSTALL}" =~ ^(true|false)$ ]] || {
    echo "PPU_CI_SKIP_WHEEL_INSTALL must be true or false" >&2
    exit 2
}
if [[ "${SKIP_WHEEL_INSTALL}" == "false" ]]; then
    WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
else
    WHEELS_DIR="${WHEELS_DIR:-}"
fi
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
rm -f -- "${RESULTS_DIR}/provenance.json" "${RESULTS_DIR}/runtime-fingerprint.json"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

export PPU_RECIPE_PHASE_FILE="${RESULTS_DIR}/recipe-phase.txt"
record_phase() {
    printf '%s\n' "$1" >"${PPU_RECIPE_PHASE_FILE}"
}

record_phase wheel_verification
vllm_wheel=""
sail_wheel=""
if [[ "${SKIP_WHEEL_INSTALL}" == "false" ]]; then
    WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
fi
verified_wheels="$(python -IB - "${ROOT_DIR}" "${CI_SCRIPT_DIR}" "${WHEELS_DIR}" "${RESULTS_DIR}" \
    "${RECIPE_PATH}" "${POLICY_PATH}" "${TIER}" "${REQUIRE_PR_ARTIFACT:-false}" \
    "${PPU_RECIPE_SOURCE_SHA:?PPU_RECIPE_SOURCE_SHA is required}" \
    "${EXPECTED_VLLM_COMMIT:-}" \
    "${EXPECTED_SAIL_COMMIT:-}" \
    "${SKIP_WHEEL_INSTALL}" \
    "${PPU_CI_RUNTIME_IMAGE_DIGEST:-}" \
    "${PPU_CI_RUNTIME_IMAGE_ID:-}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

(
    root,
    ci_script_dir,
    wheels,
    results,
    recipe,
    policy,
    tier,
    require_pr,
    checkout,
    vllm,
    sail,
    skip_wheel_install,
    image_digest,
    image_id,
) = sys.argv[1:]
if not re.fullmatch(r"[0-9a-f]{40}", checkout):
    raise SystemExit("Recipe provenance requires exact checkout commit SHA")
if tier not in {"fast_sanity", "model_sanity"} or require_pr not in {"true", "false"}:
    raise SystemExit("Invalid recipe tier or PR artifact mode")
if skip_wheel_install not in {"true", "false"}:
    raise SystemExit("Invalid skip wheel install mode")
if image_digest and not re.fullmatch(r"@sha256:[0-9a-f]{64}", image_digest):
    raise SystemExit("Invalid runtime image digest")
if image_id and not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
    raise SystemExit("Invalid runtime image ID")

root = Path(root).resolve()
ci_script_path = Path(ci_script_dir).resolve()
manifest = None
exact_source_match = False
wheel_paths = []

if skip_wheel_install == "true":
    if require_pr == "true":
        raise SystemExit("Prebuilt image mode is not allowed in PR mode")
    if wheels:
        raise SystemExit("WHEELS_DIR must be empty in prebuilt image mode")
    if not image_digest:
        raise SystemExit("Prebuilt image mode requires a verified runtime image digest")
else:
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (vllm, sail)):
        raise SystemExit("Recipe provenance requires exact checkout and wheel commit SHAs")
    exact_source_match = checkout == sail
    if not exact_source_match and (require_pr != "true" or tier != "fast_sanity"):
        raise SystemExit("SAIL wheel revision must match the recipe checkout outside PR fast_sanity")

    wheels_path = Path(wheels)
    sys.path.insert(0, str(ci_script_path))
    from ppu_wheel_manifest import verify

    try:
        manifest = verify(wheels_path, vllm_commit=vllm, sail_commit=sail)
    except Exception as error:
        raise SystemExit(f"Wheel manifest verification failed ({type(error).__name__})") from None
    wheel_paths = [str(wheels_path / manifest["wheels"][name]["file"]) for name in ("vllm", "vllm-sail")]
    if any("\n" in path for path in wheel_paths):
        raise SystemExit("Recipe provenance recording failed (ValueError)")


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

    entrypoint_path = (ci_script_path / "ppu_e2e_recipe_sanity.sh").resolve(strict=True)
    provenance = {
        "schema": 1,
        "install_mode": "prebuilt_image" if skip_wheel_install == "true" else "wheel_overlay",
        "checkout_sha": checkout,
        "tier": tier,
        "exact_source_match": exact_source_match,
        "recipe": source_record(recipe),
        "recipes_tree_sha256": tree_digest.hexdigest(),
        "policy": source_record(policy),
        "runner_sources": [
            {
                "path": "scripts/ci/ppu_e2e_recipe_sanity.sh",
                "sha256": hashlib.sha256(entrypoint_path.read_bytes()).hexdigest(),
            },
            *(
                source_record(path)
                for path in (
                    "scripts/ci/ppu_recipe.py",
                    "scripts/ci/ppu_recipe_runner.py",
                    "scripts/eval/ppu_gsm8k_eval.py",
                )
            ),
        ],
    }
    if manifest is not None:
        provenance["commits"] = {name: manifest["commits"][name] for name in ("vllm", "vllm-sail")}
        provenance["wheels"] = manifest["wheels"]
    if image_digest:
        provenance["runtime_image_digest"] = image_digest
    if image_id:
        provenance["runtime_image_id"] = image_id
    (Path(results) / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
except (OSError, ValueError) as error:
    raise SystemExit(f"Recipe provenance recording failed ({type(error).__name__})") from None

if wheel_paths:
    print(*wheel_paths, sep="\n")
PY
)"
if [[ "${SKIP_WHEEL_INSTALL}" == "false" ]]; then
    vllm_wheel="${verified_wheels%%$'\n'*}"
    sail_wheel="${verified_wheels#*$'\n'}"
fi

record_phase runtime_dependencies
export PIP_CONSTRAINT="/tmp/ppu-recipe-image-constraints.txt"
python "${CI_SCRIPT_DIR}/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output "/tmp/ppu-recipe-image-stack.json" \
    --constraints "${PIP_CONSTRAINT}"
if [[ "${TIER}" == "model_sanity" ]]; then
    numpy_version="$(python -c 'from importlib.metadata import version; print(version("numpy"))')"
    printf 'numpy===%s\npyarrow<26\n' "${numpy_version}" >> "${PIP_CONSTRAINT}"
fi

if [[ "${SKIP_WHEEL_INSTALL}" == "false" ]]; then
    python -m pip install --no-deps --force-reinstall "${vllm_wheel}"
    python -m pip install --no-deps --force-reinstall "${sail_wheel}"
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
        python -m pip install -r "${ROOT_DIR}/requirements/ci-runtime.txt"
    if [[ "${TIER}" == "model_sanity" ]]; then
        PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
            python -m pip install "evalscope==1.14.1"
        python -c 'from evalscope import TaskConfig, run_task'
    fi
elif [[ "${TIER}" == "model_sanity" ]]; then
    if ! python -c 'from evalscope import TaskConfig, run_task' >/dev/null 2>&1; then
        PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
            python -m pip install "evalscope==1.14.1"
        python -c 'from evalscope import TaskConfig, run_task'
    fi
fi

python "${CI_SCRIPT_DIR}/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "/tmp/ppu-recipe-image-stack.json"

cd /tmp
unset PYTHONPATH

record_phase platform_check
python - "${RESULTS_DIR}" "${SKIP_WHEEL_INSTALL}" "${PPU_CI_RUNTIME_IMAGE_DIGEST:-}" "${PPU_CI_RUNTIME_IMAGE_ID:-}" <<'PY'
import json
import re
import sys
from importlib import metadata
from pathlib import Path

import torch
import vllm
import vllm_sail
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), f"expected PPU platform, got {current_platform}"
assert torch.cuda.is_available(), "PPU device required"

results_dir, skip_wheel_install, image_digest, image_id = sys.argv[1:5]
SAFE_VERSION_RE = re.compile(
    r"\d+(?:\.\d+){1,3}(?:(?:a|b|rc|\.post|\.dev)\d+){0,2}(?:\+[A-Za-z0-9_-]{1,24})?"
)
BASE_VERSION_RE = re.compile(
    r"\d+(?:\.\d+){1,3}(?:(?:a|b|rc|\.post|\.dev)\d+){0,2}"
)


def safe_version(dist_name: str, fallback: str | None = None) -> str | None:
    try:
        value = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        value = fallback
    if not isinstance(value, str):
        return None
    if SAFE_VERSION_RE.fullmatch(value):
        return value
    if "+" in value:
        base = value.split("+", 1)[0]
        if BASE_VERSION_RE.fullmatch(base):
            return base
    return None


versions = {
    "vllm": safe_version("vllm", getattr(vllm, "__version__", None)),
    "vllm_sail": safe_version("vllm-sail", getattr(vllm_sail, "__version__", None)),
    "torch": safe_version("torch", getattr(torch, "__version__", None)),
}
if not versions["vllm"] or not versions["vllm_sail"]:
    raise SystemExit("Unable to determine installed vllm/vllm_sail versions")

fingerprint = {
    "schema": 1,
    "install_mode": "prebuilt_image" if skip_wheel_install == "true" else "wheel_overlay",
    "runtime_image_digest": image_digest if re.fullmatch(r"@sha256:[0-9a-f]{64}", image_digest) else None,
    "runtime_image_id": image_id if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) else None,
    "versions": {k: v for k, v in versions.items() if v is not None},
}
results_path = Path(results_dir)
(results_path / "runtime-fingerprint.json").write_text(
    json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
provenance_path = results_path / "provenance.json"
if provenance_path.is_file() and not provenance_path.is_symlink():
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_versions"] = fingerprint["versions"]
    if fingerprint["runtime_image_digest"]:
        provenance["runtime_image_digest"] = fingerprint["runtime_image_digest"]
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
