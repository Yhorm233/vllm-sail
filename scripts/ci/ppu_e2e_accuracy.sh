#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# PPU GSM8K accuracy E2E runner (server-client mode via evalscope).
# Executed inside the read-only-source Docker runtime created by the CI wrapper.
#
# Starts ``vllm serve`` + runs evalscope GSM8K accuracy evaluation.
# Results (junit XML, golden JSON, vllm serve logs) land in RESULTS_DIR.
#
# Environment contract (injected by ppu_runtime_container.sh):
#   MODEL_KEY          evaluator model-config key; also keys the golden file
#                      (e.g. deepseek_v4_int8)
#   CASE_ID            catalog case label; several cases may share one
#                      MODEL_KEY and therefore one golden (default: MODEL_KEY)
#   SERVER_CMD         JSON array of the full ``vllm serve`` argv for this
#                      case, with {model}/{port}/{served_name} placeholders.
#                      Empty or "[]" uses the evaluator's built-in argv.
#   SERVER_ENV         JSON object of extra environment variables for this
#                      case's server process (e.g. VLLM_ENGINE_READY_TIMEOUT_S).
#                      Empty or "{}" adds nothing.
#   MODEL_ID           Hugging Face model id this key resolves to; checked
#                      against the offline cache before the server starts
#   WHEELS_DIR         directory holding the downloaded *.whl artifacts
#   RESULTS_DIR        absolute output dir for xml/logs/golden
#   UPDATE_GOLDEN      "1" to pass --update-golden (default: "0")
#   THRESHOLD          accuracy threshold (default: 0.9)
#   VLLM_PORT          dynamically allocated vLLM API port (default: 9985)
#   DG_JIT_CACHE_DIR   host-backed DeepGEMM kernel cache; without it the JIT
#                      pass is repeated from scratch on every container
#   GOLDEN_SEED_DIR    reviewed goldens to seed from; the checkout is read-only
#                      (default: /source/tests/e2e/models/_golden)
set -Eeuo pipefail

CI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${PPU_CI_SOURCE_ROOT:-$(cd -- "${CI_SCRIPT_DIR}/../.." && pwd)}"

RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
ci_result_file="${RESULTS_DIR}/legacy-accuracy-result.json"
record_legacy_accuracy="${REQUIRE_PR_ARTIFACT:-false}"
# The trusted launcher can also be used against older evaluator revisions.
if [[ "${RECORD_LEGACY_ACCURACY:-false}" == true ]] &&
    grep -q -- '--ci-result-file' "${ROOT_DIR}/scripts/eval/ppu_gsm8k_eval.py"; then
    record_legacy_accuracy=true
fi
if [[ "$record_legacy_accuracy" == true ]]; then
    umask 077
    mkdir -p "${RESULTS_DIR}"
    record_setup_failure() {
        if (( $1 != 0 )); then
            printf '%s\n' '{"status":"runtime_setup_failed"}' > "${ci_result_file}"
        fi
    }
    trap 'record_setup_failure "$?"' EXIT
fi

# HuggingFace Hub: resolve model_id to local cache; never contact the hub.
export HF_HOME="${HF_HOME:?HF_HOME must be set to the Hugging Face cache directory}"
export HF_HUB_OFFLINE=1

# ``datasets`` builds an Arrow cache under ``${HF_HOME}/datasets``, but HF_HOME is
# a read-only view of the NAS cache here, so point it at writable scratch space.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"

# Prevent image defaults from masking the installed version or changing frontend.
unset VLLM_VERSION 2>/dev/null || true
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

export GSM8K_DATASET_PATH="${GSM8K_DATASET_PATH:?GSM8K_DATASET_PATH must be set}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
CASE_ID="${CASE_ID:-${MODEL_KEY}}"
SERVER_CMD="${SERVER_CMD:-}"
SERVER_ENV="${SERVER_ENV:-}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
SKIP_WHEEL_INSTALL="${PPU_CI_SKIP_WHEEL_INSTALL:-false}"
[[ "${SKIP_WHEEL_INSTALL}" =~ ^(true|false)$ ]] || {
    echo "PPU_CI_SKIP_WHEEL_INSTALL must be true or false" >&2
    exit 2
}
if [[ "${SKIP_WHEEL_INSTALL}" == "false" ]]; then
    WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
else
    [[ "${REQUIRE_PR_ARTIFACT:-false}" != true ]] || {
        echo "PPU_CI_SKIP_WHEEL_INSTALL is not allowed in PR mode" >&2
        exit 2
    }
    [[ "${PPU_CI_RUNTIME_IMAGE_DIGEST:-}" =~ ^@sha256:[0-9a-f]{64}$ ]] || {
        echo "Prebuilt image mode requires a verified runtime image digest" >&2
        exit 2
    }
    WHEELS_DIR="${WHEELS_DIR:-}"
fi
UPDATE_GOLDEN="${UPDATE_GOLDEN:-0}"
THRESHOLD="${THRESHOLD:-0.9}"
VLLM_PORT="${VLLM_PORT:-9985}"

mkdir -p "${RESULTS_DIR}"
exec > >(tee "${RESULTS_DIR}/accuracy.log") 2>&1

# The internal HTTP(S) proxies do not route to NAS or loopback.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

log() { printf '\n=== %s ===\n' "$1"; }

# ------------------------------------------------------------------
log "environment"
echo "MODEL_KEY=${MODEL_KEY}"
echo "CASE_ID=${CASE_ID}"
echo "SERVER_CMD=${SERVER_CMD:-<built-in>}"
echo "SERVER_ENV=${SERVER_ENV:-<none>}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "UPDATE_GOLDEN=${UPDATE_GOLDEN}"
echo "THRESHOLD=${THRESHOLD}"
echo "VLLM_PORT=${VLLM_PORT}"
echo "DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-<unset: JIT recompiles this case>}"
echo "GOLDEN_SEED_DIR=${GOLDEN_SEED_DIR:-/source/tests/e2e/models/_golden}"
python -c 'import sys; print("python", sys.version)'

# ------------------------------------------------------------------
# `capture` records the stack the image already ships. `prepare` would fetch a
# torch tarball from the internal artifactory and swap it in, but that endpoint
# answers 401 from the runners and the image torch is what the smoke lane
# serves models with, so there is nothing to replace.
log "capture SAIL PyTorch stack"
if [[ "${SKIP_WHEEL_INSTALL}" == "false" ]]; then
    WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
    verify_args=(verify "${WHEELS_DIR}")
    if [[ "${REQUIRE_PR_ARTIFACT:-false}" == true ]]; then
        verify_args+=(
            --vllm-commit "${EXPECTED_VLLM_COMMIT:?EXPECTED_VLLM_COMMIT is required}"
            --sail-commit "${EXPECTED_SAIL_COMMIT:?EXPECTED_SAIL_COMMIT is required}"
        )
    fi
    python "${CI_SCRIPT_DIR}/ppu_wheel_manifest.py" "${verify_args[@]}"
fi

RAW_IMAGE_STACK="/tmp/ppu-accuracy-image-stack.json"
export PIP_CONSTRAINT="/tmp/ppu-accuracy-image-constraints.txt"
if ! python "${CI_SCRIPT_DIR}/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output "${RAW_IMAGE_STACK}" \
    --constraints "${PIP_CONSTRAINT}" >/tmp/ppu-accuracy-image-stack-capture.log 2>&1; then
    echo "ERROR: SAIL PyTorch image stack capture failed" >&2
    exit 1
fi
python - "${RAW_IMAGE_STACK}" "${RESULTS_DIR}/image-stack.json" <<'PY'
import json
import re
import sys
from pathlib import Path

raw_path, public_path = Path(sys.argv[1]), Path(sys.argv[2])
raw = json.loads(raw_path.read_text(encoding="utf-8"))
SAFE_VERSION_RE = re.compile(
    r"\d+(?:\.\d+){1,3}(?:(?:a|b|rc|\.post|\.dev)\d+){0,2}(?:\+[A-Za-z0-9_-]{1,24})?"
)
BASE_VERSION_RE = re.compile(
    r"\d+(?:\.\d+){1,3}(?:(?:a|b|rc|\.post|\.dev)\d+){0,2}"
)


def sanitize_version(value):
    if not isinstance(value, str):
        return None
    if SAFE_VERSION_RE.fullmatch(value):
        return value
    if "+" in value:
        base = value.split("+", 1)[0]
        if BASE_VERSION_RE.fullmatch(base):
            return base
    return None


packages = {}
raw_packages = raw.get("packages")
if isinstance(raw_packages, dict):
    for name, version in raw_packages.items():
        if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9._-]+", name):
            cleaned = sanitize_version(version)
            if cleaned is not None:
                packages[name] = cleaned

sanitized = {
    "stack_source": raw.get("stack_source") if raw.get("stack_source") in {"image", "image+torch"} else "image",
    "fingerprint_kind": raw.get("fingerprint_kind") if raw.get("fingerprint_kind") == "installed-tree-sha256" else "installed-tree-sha256",
    "sdk_sha256": raw.get("sdk_sha256") if isinstance(raw.get("sdk_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", raw["sdk_sha256"]) else None,
    "torch_sha256": raw.get("torch_sha256") if isinstance(raw.get("torch_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", raw["torch_sha256"]) else None,
    "torch_version": sanitize_version(raw.get("torch_version")),
    "packages": packages,
}
public_path.write_text(json.dumps(sanitized, sort_keys=True, indent=2) + "\n", encoding="utf-8")
for key in ("stack_source", "fingerprint_kind", "sdk_sha256", "torch_sha256", "torch_version"):
    if sanitized[key] is not None:
        print(f"{key}={sanitized[key]}")
PY

# ------------------------------------------------------------------
if [[ "${SKIP_WHEEL_INSTALL}" == "false" ]]; then
    log "install wheels"
    mapfile -t vllm_wheels < <(
        find "${WHEELS_DIR}" -type f -name 'vllm-*.whl' -print | sort
    )
    mapfile -t sail_wheels < <(
        find "${WHEELS_DIR}" -type f -name 'vllm_sail-*.whl' -print | sort
    )
    ((${#vllm_wheels[@]} == 1)) || { echo "expected one vLLM wheel under ${WHEELS_DIR}, got ${#vllm_wheels[@]}"; exit 1; }
    ((${#sail_wheels[@]} == 1)) || { echo "expected one vLLM SAIL wheel under ${WHEELS_DIR}, got ${#sail_wheels[@]}"; exit 1; }
    python -m pip install --no-deps --force-reinstall "${vllm_wheels[0]}"
    python -m pip install --no-deps --force-reinstall "${sail_wheels[0]}"

    # Install test dependencies (pytest, etc.)
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
        python -m pip install -r "${ROOT_DIR}/requirements/dev.txt"

    # Install accuracy evaluation dependencies (not in dev.txt, needs a real device)
    if ! PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
        python -m pip install "evalscope==1.14.1" >/tmp/ppu-accuracy-evalscope-pip.log 2>&1; then
        echo "ERROR: failed to install evalscope==1.14.1" >&2
        exit 1
    fi
else
    log "prebuilt image mode (skip wheel install)"
    if ! python -c "from evalscope import TaskConfig, run_task" >/dev/null 2>&1; then
        if ! PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
            python -m pip install "evalscope==1.14.1" >/tmp/ppu-accuracy-evalscope-pip.log 2>&1; then
            echo "ERROR: failed to install evalscope==1.14.1" >&2
            exit 1
        fi
    fi
fi

# Verify evalscope is importable
python -c "from evalscope import TaskConfig, run_task; print('evalscope OK')"

if ! python "${CI_SCRIPT_DIR}/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "${RAW_IMAGE_STACK}" >/tmp/ppu-accuracy-image-stack-verify.log 2>&1; then
    echo "ERROR: SAIL PyTorch image stack verification failed" >&2
    exit 1
fi
echo "stack_unchanged=true"

# ------------------------------------------------------------------
log "platform self-check"
cd /tmp
unset PYTHONPATH
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
try:
    check_vllm_compatibility(force=True)
except Exception as exc:
    raise SystemExit(
        f"vLLM compatibility check failed ({type(exc).__name__}): "
        f"vllm={versions['vllm'] or 'unknown'}, "
        f"vllm_sail={versions['vllm_sail'] or 'unknown'}"
    ) from None
assert current_platform.is_ppu(), f"expected PPU platform, got {current_platform}"
assert torch.cuda.is_available(), "accuracy E2E pod requires a PPU device"
if not versions["vllm"] or not versions["vllm_sail"]:
    raise SystemExit("Unable to determine installed vllm/vllm_sail versions")
print("Device:", current_platform.get_device_name(0))
print("Capability:", current_platform.get_device_capability())
print("vLLM:", versions["vllm"])
print("vLLM SAIL:", versions["vllm_sail"])
fingerprint = {
    "schema": 1,
    "install_mode": "prebuilt_image" if skip_wheel_install == "true" else "wheel_overlay",
    "runtime_image_digest": image_digest if re.fullmatch(r"@sha256:[0-9a-f]{64}", image_digest) else None,
    "runtime_image_id": image_id if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) else None,
    "versions": {k: v for k, v in versions.items() if v is not None},
}
(Path(results_dir) / "runtime-fingerprint.json").write_text(
    json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

# ------------------------------------------------------------------
# The offline cache is a view tree over a separate checkpoint export. Verify
# the repo is there before vLLM starts: a missing snapshot otherwise surfaces
# as a bare LocalEntryNotFoundError twelve minutes into engine startup.
log "resolve model path"
MODEL_PATH=$(python - "${MODEL_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], local_files_only=True))
PY
) || { echo "ERROR: Cannot resolve model_id '${MODEL_ID}' from HF cache"; exit 1; }
echo "Resolved model_id '${MODEL_ID}' -> '${MODEL_PATH}'"

# ------------------------------------------------------------------
log "run accuracy eval: ${CASE_ID}"
# Keep the read-only checkout off cwd/PYTHONPATH so the installed wheels win.
cd /tmp
unset PYTHONPATH

# Reviewed goldens ship inside the read-only checkout, so seed them into the
# writable results tree. Without this seed every run starts with an empty
# golden directory, reports "created", and only the score threshold is
# enforced; the regression gates would never fire. Pending output also lands
# here and is uploaded as an artifact for a human to promote.
GOLDEN_SEED_DIR="${GOLDEN_SEED_DIR:-/source/tests/e2e/models/_golden}"
golden_dir="${RESULTS_DIR}/golden"
mkdir -p "${golden_dir}"
if [[ -d "${GOLDEN_SEED_DIR}" ]]; then
    seeded=0
    for golden in "${GOLDEN_SEED_DIR}"/accuracy_*.json; do
        [[ -f "${golden}" ]] || continue
        cp -f "${golden}" "${golden_dir}/"
        seeded=$((seeded + 1))
    done
    echo "seeded ${seeded} reviewed golden(s) from ${GOLDEN_SEED_DIR}"
else
    echo "no reviewed golden directory at ${GOLDEN_SEED_DIR}; threshold-only run"
fi

eval_args=(
    --model-key "${MODEL_KEY}"
    --case-id "${CASE_ID}"
    --threshold "${THRESHOLD}"
    --work-dir "${RESULTS_DIR}"
    --golden-dir "${RESULTS_DIR}/golden"
    --port "${VLLM_PORT}"
)
if [[ -n "${SERVER_CMD}" && "${SERVER_CMD}" != "[]" ]]; then
    eval_args+=(--server-cmd "${SERVER_CMD}")
fi
if [[ -n "${SERVER_ENV}" && "${SERVER_ENV}" != "{}" && "${SERVER_ENV}" != "null" ]]; then
    eval_args+=(--server-env "${SERVER_ENV}")
fi
if [[ "${UPDATE_GOLDEN}" == "1" ]]; then
    eval_args+=(--update-golden)
fi
if [[ "$record_legacy_accuracy" == true ]]; then
    eval_args+=(--ci-result-file "${ci_result_file}")
    trap - EXIT
fi

set +e
python "${ROOT_DIR}/scripts/eval/ppu_gsm8k_eval.py" "${eval_args[@]}"
rc=$?
set -e

# ------------------------------------------------------------------
log "collect evidence"
ppu-smi > "${RESULTS_DIR}/ppu-smi.log" 2>&1 || true

# ------------------------------------------------------------------
if (( rc != 0 )); then
    log "ACCURACY ${CASE_ID} FAILED (exit ${rc})"
    exit "${rc}"
fi
log "ACCURACY ${CASE_ID} PASS"
