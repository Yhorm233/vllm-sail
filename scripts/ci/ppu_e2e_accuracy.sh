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
#   MODEL_KEY          model config key (e.g. deepseek_v4_int8)
#   WHEELS_DIR         directory holding the downloaded *.whl artifacts
#   RESULTS_DIR        absolute output dir for xml/logs/golden
#   UPDATE_GOLDEN      "1" to pass --update-golden (default: "0")
#   THRESHOLD          accuracy threshold (default: 0.9)
#   VLLM_PORT          dynamically allocated vLLM API port (default: 9985)
#   GOLDEN_SEED_DIR    reviewed goldens to seed from; the checkout is read-only
#                      (default: /source/tests/e2e/models/_golden)
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# HuggingFace Hub: resolve model_id to local cache; never contact the hub.
export HF_HOME="${HF_HOME:?HF_HOME must be set to the Hugging Face cache directory}"
export HF_HUB_OFFLINE=1

# Prevent image defaults from masking the installed version or changing frontend.
unset VLLM_VERSION 2>/dev/null || true
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

export GSM8K_DATASET_PATH="${GSM8K_DATASET_PATH:?GSM8K_DATASET_PATH must be set}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
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
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "UPDATE_GOLDEN=${UPDATE_GOLDEN}"
echo "THRESHOLD=${THRESHOLD}"
echo "VLLM_PORT=${VLLM_PORT}"
echo "GOLDEN_SEED_DIR=${GOLDEN_SEED_DIR:-/source/tests/e2e/models/_golden}"
python -c 'import sys; print("python", sys.version)'

# ------------------------------------------------------------------
log "prepare SAIL PyTorch stack"
WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}"

export PIP_CONSTRAINT="${RESULTS_DIR}/image-constraints.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" prepare \
    --sdk /usr/local/PPU_SDK --output "${RESULTS_DIR}/image-stack.json" \
    --constraints "${PIP_CONSTRAINT}"

# ------------------------------------------------------------------
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
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    python -m pip install "evalscope>=0.6,<1.0"

# Verify evalscope is importable
python -c "from evalscope import TaskConfig, run_task; print('evalscope OK')"

python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "${RESULTS_DIR}/image-stack.json"

# ------------------------------------------------------------------
log "platform self-check"
cd /tmp
unset PYTHONPATH
python - <<'PY'
import torch, vllm, vllm_sail
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), f"expected PPU platform, got {current_platform}"
assert torch.cuda.is_available(), "accuracy E2E pod requires a PPU device"
print("Device:", current_platform.get_device_name(0))
print("Capability:", current_platform.get_device_capability())
print("vLLM:", vllm.__version__)
print("vLLM SAIL:", vllm_sail.__version__)
PY

# ------------------------------------------------------------------
log "run accuracy eval: ${MODEL_KEY}"
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
    --threshold "${THRESHOLD}"
    --work-dir "${RESULTS_DIR}"
    --golden-dir "${RESULTS_DIR}/golden"
    --port "${VLLM_PORT}"
)
if [[ "${UPDATE_GOLDEN}" == "1" ]]; then
    eval_args+=(--update-golden)
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
    log "ACCURACY ${MODEL_KEY} FAILED (exit ${rc})"
    exit "${rc}"
fi
log "ACCURACY ${MODEL_KEY} PASS"
