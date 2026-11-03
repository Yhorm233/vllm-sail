#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# PPU end-to-end smoke test, executed inside the CUDA-compatible runtime
# container created on a self-hosted PPU runner. It installs the CI-built vLLM
# and vLLM SAIL wheels, self-checks the SAIL platform, serves a model from the
# read-only NAS checkpoint mount, and verifies one chat completion end to end.
#
# Environment contract (injected by ppu_runtime_container.sh):
#   MODEL_ID            Hugging Face repository ID resolved from the offline cache
#   SERVED_NAME         served-model-name exposed by vLLM
#   TP_SIZE             tensor-parallel size (matches the pod's PPU card count)
#   WHEELS_DIR          directory holding the downloaded *.whl artifacts
#   EXTRA_SERVE_FLAGS   optional extra flags forwarded to `vllm serve`
#   GPU_MEM_UTIL        optional gpu-memory-utilization (default 0.9)
#   MAX_MODEL_LEN       optional max-model-len (default 8192)
#   SERVE_PORT          optional serve port (default 9985)
#   READY_TIMEOUT_SECS  optional readiness budget in seconds (default 1200)
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Prevent image defaults from masking the installed version or changing frontend.
unset VLLM_VERSION
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
HF_HOME="${HF_HOME:?HF_HOME is required}"
WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
SERVED_NAME="${SERVED_NAME:-model}"
TP_SIZE="${TP_SIZE:-1}"
EXTRA_SERVE_FLAGS="${EXTRA_SERVE_FLAGS:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVE_PORT="${SERVE_PORT:-9985}"
SERVE_HOST="127.0.0.1"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-1200}"
SERVE_LOG="/tmp/vllm_serve.log"

# The internal HTTP(S) proxies do not route to the loopback serve endpoint.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

log() { printf '\n=== %s ===\n' "$1"; }

log "environment"
echo "MODEL_ID=${MODEL_ID}"
echo "HF_HOME=${HF_HOME}"
echo "SERVED_NAME=${SERVED_NAME} TP_SIZE=${TP_SIZE} PORT=${SERVE_PORT}"
python -c 'import sys; print("python", sys.version)'

log "install wheels"
WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
REQUIRE_PR_ARTIFACT="${REQUIRE_PR_ARTIFACT:-false}"
EXPECTED_VLLM_COMMIT="${EXPECTED_VLLM_COMMIT:?EXPECTED_VLLM_COMMIT is required}"
EXPECTED_SAIL_COMMIT="${EXPECTED_SAIL_COMMIT:?EXPECTED_SAIL_COMMIT is required}"
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}"
if [[ "${REQUIRE_PR_ARTIFACT}" == "true" ]]; then
    python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}" \
        --vllm-commit "${EXPECTED_VLLM_COMMIT}" \
        --sail-commit "${EXPECTED_SAIL_COMMIT}"
fi
# Match the vLLM wheel first, excluding the vllm_sail wheel (underscore prefix).
mapfile -t vllm_wheels < <(
    find "${WHEELS_DIR}" -type f -name 'vllm-*.whl' -print | sort
)
mapfile -t sail_wheels < <(
    find "${WHEELS_DIR}" -type f -name 'vllm_sail-*.whl' -print | sort
)
((${#vllm_wheels[@]} == 1)) || { echo "expected one vLLM wheel under ${WHEELS_DIR}"; exit 1; }
((${#sail_wheels[@]} == 1)) || { echo "expected one vLLM SAIL wheel under ${WHEELS_DIR}"; exit 1; }
echo "vllm wheel: ${vllm_wheels[0]}"
echo "sail wheel: ${sail_wheels[0]}"
export PIP_CONSTRAINT="/tmp/ppu-smoke-image-constraints.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output /tmp/ppu-smoke-image-stack.json \
    --constraints "${PIP_CONSTRAINT}"
python -m pip install --no-deps --force-reinstall "${vllm_wheels[0]}"
python -m pip install --no-deps --force-reinstall "${sail_wheels[0]}"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output /tmp/ppu-smoke-image-stack.json

# Test the installed wheel, not the staged source checkout.
cd /tmp
unset PYTHONPATH
python - <<'PY'
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
PY

log "collect_env"
python -c "from vllm_sail import collect_env; collect_env()"

log "resolve model path"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL_PATH=$(python - "${MODEL_ID}" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], local_files_only=True))
PY
) || { echo "ERROR: Cannot resolve model_id '${MODEL_ID}' from HF cache"; exit 1; }
echo "Resolved model_id '${MODEL_ID}' -> '${MODEL_PATH}'"
export MODEL_PATH

log "checkpoint"
test -d "${MODEL_PATH}" || { echo "checkpoint dir missing: ${MODEL_PATH}"; exit 1; }
test -f "${MODEL_PATH}/config.json" || { echo "config.json missing in checkpoint"; exit 1; }
find "${MODEL_PATH}" -maxdepth 1 -type f -print | sed -n '1,20p'

log "serve"
# shellcheck disable=SC2086
nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --host "${SERVE_HOST}" --port "${SERVE_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${EXTRA_SERVE_FLAGS} \
    >"${SERVE_LOG}" 2>&1 &
SERVE_PID=$!

cleanup() {
    kill "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

log "wait for readiness"
ready=0
deadline=$((SECONDS + READY_TIMEOUT_SECS))
# Only genuine fatal markers; benign lines such as the stubbed
# "compute ops raise NotImplementedError" must not trip the failure detector.
fatal_re='Traceback \(most recent|EngineCore failed|Engine core initialization failed|core dumped|Failed to infer device type|Error: model architectures'
while ((SECONDS < deadline)); do
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "serve process exited before becoming ready"
        tail -n 200 "${SERVE_LOG}" || true
        exit 1
    fi
    if grep -qE 'Application startup complete' "${SERVE_LOG}"; then
        ready=1
        break
    fi
    if grep -qE "${fatal_re}" "${SERVE_LOG}"; then
        echo "fatal error detected in serve log"
        tail -n 200 "${SERVE_LOG}" || true
        exit 1
    fi
    sleep 5
done
if ((ready == 0)); then
    echo "server not ready within ${READY_TIMEOUT_SECS}s"
    tail -n 200 "${SERVE_LOG}" || true
    exit 1
fi

log "chat completion request"
request_body=$(cat <<JSON
{"model": "${SERVED_NAME}",
 "messages": [{"role": "user", "content": "简单说一下什么是投机采样"}],
 "max_completion_tokens": 100,
 "top_k": 1}
JSON
)
response=$(curl -sS -X POST "http://${SERVE_HOST}:${SERVE_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "${request_body}")
echo "${response}"
echo "${response}" | python -c '
import json, sys
data = json.load(sys.stdin)
content = data["choices"][0]["message"]["content"]
assert content and content.strip(), "empty completion content"
print(f"completion OK, {len(content)} chars")
'

log "ppu-smi occupancy evidence"
ppu-smi || true

log "E2E SMOKE PASS"
