#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Consume CI wheels in a fresh single-card SAIL runtime, without rebuilding.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
UT_RESULTS_DIR="${UT_RESULTS_DIR:?UT_RESULTS_DIR is required}"
mkdir -p "${UT_RESULTS_DIR}"
exec > >(tee "${UT_RESULTS_DIR}/environment.log") 2>&1

unset VLLM_VERSION PYTHONPATH
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0
REQUIRE_PR_ARTIFACT="${REQUIRE_PR_ARTIFACT:-false}"
EXPECTED_VLLM_COMMIT="${EXPECTED_VLLM_COMMIT:?EXPECTED_VLLM_COMMIT is required}"
EXPECTED_SAIL_COMMIT="${EXPECTED_SAIL_COMMIT:?EXPECTED_SAIL_COMMIT is required}"
: "${VLLM_SOURCE_ROOT:?VLLM_SOURCE_ROOT is required}"
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}"
if [[ "${REQUIRE_PR_ARTIFACT}" == "true" ]]; then
    python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}" \
        --vllm-commit "${EXPECTED_VLLM_COMMIT}" \
        --sail-commit "${EXPECTED_SAIL_COMMIT}"
elif ! python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}" \
    --vllm-commit "${EXPECTED_VLLM_COMMIT}"; then
    echo "PPU PIN UT SKIP: build does not contain the current main-verified vLLM pin"
    exit 0
fi

# Copy the read-only wheel mount outside the checkout so packaging UT does not
# copy the large wheels into its temporary source trees.
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cp -a "${WHEELS_DIR}" "${WORK_DIR}/wheels"
WHEELS_DIR="${WORK_DIR}/wheels"
export PIP_CONSTRAINT="${UT_RESULTS_DIR}/image-constraints.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk /usr/local/PPU_SDK --output "${UT_RESULTS_DIR}/image-stack.json" \
    --constraints "${PIP_CONSTRAINT}"
python -m pip install --no-deps --force-reinstall \
    "${WHEELS_DIR}"/vllm/*.whl "${WHEELS_DIR}"/vllm-sail/*.whl
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    python -m pip install -r "${ROOT_DIR}/requirements/dev.txt" \
    -r "${ROOT_DIR}/requirements/profiling.txt"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk /usr/local/PPU_SDK --output "${UT_RESULTS_DIR}/image-stack.json"

# Check the installed packages outside the source checkout first.
cd /tmp
python - <<'PY'
import torch
import vllm
import vllm_sail
# Platform discovery can succeed even if the wheel has unresolved native symbols.
import vllm_sail._upstream_C
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), current_platform
assert torch.cuda.is_available(), "the pin UT pod requires a PPU driver/device"
print("SAIL PyTorch:", torch.__version__, torch.__file__)
print("vLLM:", vllm.__file__)
print("vLLM SAIL:", vllm_sail.__file__)
print("Platform:", type(current_platform).__name__)
print("Device capability:", current_platform.get_device_capability())
PY

cd "${ROOT_DIR}"
git config --global --add safe.directory "${ROOT_DIR}"
python -m pytest tests/ut -q -rs \
    -m "upstream_source or requires_runtime or ppu or cap80 or cap89" \
    --require-upstream-source \
    --junitxml="${UT_RESULTS_DIR}/pin-ut.xml" \
    2>&1 | tee "${UT_RESULTS_DIR}/pin-ut.log"
echo "PPU PIN UT PASS"
