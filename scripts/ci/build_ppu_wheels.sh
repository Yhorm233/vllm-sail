#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# The public entry point uses an already activated SDK/Python environment.
# With no arguments, retain the pinned CI image setup.
PREPARED_ENV=0
if [[ "${1:-}" == "--prepared-env" ]]; then
    PREPARED_ENV=1
    shift
fi
if (($#)); then
    printf 'unknown build argument: %s\n' "$1" >&2
    exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts}"
mkdir -p "${ARTIFACT_DIR}"
ARTIFACT_DIR="$(cd -- "${ARTIFACT_DIR}" && pwd)"
VLLM_WHEEL_DIR="${ARTIFACT_DIR}/wheels/vllm"
SAIL_WHEEL_DIR="${ARTIFACT_DIR}/wheels/vllm-sail"
LOG_DIR="${ARTIFACT_DIR}/logs"
MANIFEST="${ARTIFACT_DIR}/build-manifest.txt"

VLLM_REPOSITORY="${VLLM_REPOSITORY:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-$(tr -d '[:space:]' < "${ROOT_DIR}/.github/vllm-main-verified.commit")}"
PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_15;ppu_10}"
# Building the native PPU kernels is memory-heavy (~2 GiB per compile job). The
# The CI build environment exposes many cores but a smaller memory limit, so an unbounded
# MAX_JOBS makes the OOM killer terminate the pod (exit 137). Derive a safe
# default from the cgroup memory limit (falling back to MemTotal) and apply a
# conservative ceiling. Override explicitly with BUILD_JOBS.
default_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
mem_bytes=0
if [[ -r /sys/fs/cgroup/memory.max ]]; then
    read -r mem_bytes </sys/fs/cgroup/memory.max || mem_bytes=0
    [[ "${mem_bytes}" == "max" ]] && mem_bytes=0
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    read -r mem_bytes </sys/fs/cgroup/memory/memory.limit_in_bytes || mem_bytes=0
fi
if [[ ! "${mem_bytes}" =~ ^[0-9]+$ ]] || ((mem_bytes <= 0 || mem_bytes > 1099511627776)); then
    mem_bytes="$(awk '/MemTotal/{print $2*1024}' /proc/meminfo 2>/dev/null || printf '0')"
fi
if [[ "${mem_bytes}" =~ ^[0-9]+$ ]] && ((mem_bytes > 0)); then
    mem_jobs=$((mem_bytes / (2 * 1024 * 1024 * 1024)))
    ((mem_jobs < 1)) && mem_jobs=1
    ((default_jobs > mem_jobs)) && default_jobs="${mem_jobs}"
fi
((default_jobs > 16)) && default_jobs=16
BUILD_JOBS="${BUILD_JOBS:-${default_jobs}}"
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'BUILD_JOBS must be a positive integer\n' >&2
    exit 2
}

mkdir -p "${VLLM_WHEEL_DIR}" "${SAIL_WHEEL_DIR}" "${LOG_DIR}"
: >"${MANIFEST}"
exec > >(tee -a "${LOG_DIR}/build.log") 2>&1

WORK_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "${WORK_DIR}"
}
on_error() {
    local status=$?
    trap - ERR
    printf 'status=failed\nexit_code=%s\n' "${status}" >>"${MANIFEST}"
    exit "${status}"
}
trap cleanup EXIT
trap on_error ERR

fail() {
    printf '%s\n' "$1" >&2
    return 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

[[ "$(uname -s)" == "Linux" ]] || fail "native wheel builds require Linux"
for command_name in git python tee awk find; do
    require_command "${command_name}"
done
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ((PREPARED_ENV)); then
    [[ "${PYTHON_VERSION}" =~ ^3\.(10|11|12|13)$ ]] || fail "use Python 3.10-3.13 with matching SAIL PyTorch"
else
    [[ "${PYTHON_VERSION}" == "3.12" ]] || fail "the selected image requires Python 3.12"
fi
{
    printf 'status=running\n'
    printf 'vllm_ref=%s\n' "${VLLM_REF}"
    printf 'pytorch_sail_arch=%s\n' "${PYTORCH_SAIL_ARCH}"
    printf 'build_image=%s\n' "${VLLM_SAIL_BUILD_IMAGE:-}"
} >>"${MANIFEST}"

if ((PREPARED_ENV)); then
    export PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
    [[ -d "${PPU_SDK}" ]] || fail "activate the SAIL SDK and set PPU_SDK to its directory"
    PPU_SDK="$(cd -- "${PPU_SDK}" && pwd)"
    export HGCC="${HGCC:-${PPU_SDK}/bin/hgcc}"
    [[ -x "${HGCC}" ]] || fail "SAIL compiler not executable: ${HGCC}"
    HGCC="$(cd -- "$(dirname -- "${HGCC}")" && pwd)/$(basename -- "${HGCC}")"
else
    [[ -r /usr/local/PPU_SDK/envsetup.sh ]] || fail "the image must contain the SAIL SDK"
    # shellcheck disable=SC1091
    source /usr/local/PPU_SDK/envsetup.sh
    export PPU_SDK=/usr/local/PPU_SDK
    export HGCC="${PPU_SDK}/bin/hgcc"
fi
export PYTORCH_SAIL_ARCH
unset TORCH_CUDA_ARCH_LIST
unset VLLM_SAIL_HG_ARCH
unset VLLM_SAIL_SKIP_EXT
unset VLLM_SAIL_PREBUILT_EXT_DIR
# Derive package versions from the selected source, never the image's overrides.
unset VLLM_VERSION VLLM_VERSION_OVERRIDE
unset SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM
unset VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST
unset VLLM_PRECOMPILED_WHEEL_LOCATION
# envsetup.sh repoints pip at an internal mirror (art.eng.t-head.cn) that is
# unreachable from the CI build environment; force the public Aliyun PyPI mirror after the
# SDK env is sourced. Override with SAIL_PIP_INDEX_URL.
if ((PREPARED_ENV == 0)); then
    export PIP_INDEX_URL="${SAIL_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
    export PIP_EXTRA_INDEX_URL=""
fi
export PIP_CONSTRAINT="${WORK_DIR}/image-constraints.txt"

if ((PREPARED_ENV == 0)); then
    command -v hgcc
fi
"${HGCC}" --version 2>&1 | tee "${LOG_DIR}/hgcc-version.log"
python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" capture \
    --sdk "${PPU_SDK}" --output "${LOG_DIR}/image-stack.json" \
    --constraints "${PIP_CONSTRAINT}" | tee -a "${MANIFEST}"
python - <<'PY' 2>&1 | tee "${LOG_DIR}/torch-env.log"
import torch

print(torch.__version__)
print(torch.__file__)
PY

VLLM_SRC="${WORK_DIR}/vllm"
PREBUILT_EXT_DIR="${WORK_DIR}/prebuilt-extensions"
NATIVE_REUSE_DIR="${VLLM_SAIL_NATIVE_REUSE_DIR:-}"
NATIVE_REUSE=0

# A native-reuse artifact is only a hint from the trusted controller. Every
# donor operation is guarded so malformed or legacy artifacts fall back to a
# full compile instead of failing the CI run.
fingerprint_args=(
    --build-image "${VLLM_SAIL_BUILD_IMAGE:-}"
    --pytorch-sail-arch "${PYTORCH_SAIL_ARCH}"
    --cmake-build-type "${CMAKE_BUILD_TYPE:-Release}"
    --cmake-hgcc-dir "${CMAKE_HGCC_DIR:-}"
    --vllm-sail-hg-flags "${VLLM_SAIL_HG_FLAGS:-}"
    --vllm-sail-hg-std "${VLLM_SAIL_HG_STD:-20}"
)
prepare_native_reuse() {
    python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify \
        "${NATIVE_REUSE_DIR}" --vllm-commit "${VLLM_REF}" >/dev/null || return 1
    local current_fingerprint donor_fingerprint
    current_fingerprint="$(python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" \
        fingerprint "${ROOT_DIR}" "${fingerprint_args[@]}")" || return 1
    donor_fingerprint="$(python - "${NATIVE_REUSE_DIR}/wheel-manifest.json" <<'PY'
import json
import sys

native = json.load(open(sys.argv[1])).get("native_reuse")
if not isinstance(native, dict):
    raise SystemExit(1)
print(native.get("fingerprint", ""))
PY
    )" || return 1
    [[ -n "${donor_fingerprint}" && "${donor_fingerprint}" == "${current_fingerprint}" ]] || return 1
    rm -rf "${PREBUILT_EXT_DIR}"
    python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" extract-native \
        "${NATIVE_REUSE_DIR}" --output "${PREBUILT_EXT_DIR}" >/dev/null || return 1
    mapfile -t donor_vllm_wheels < <(
        find "${NATIVE_REUSE_DIR}" -type f -name 'vllm-*.whl' -print
    )
    ((${#donor_vllm_wheels[@]} == 1)) || return 1
    cp "${donor_vllm_wheels[0]}" "${VLLM_WHEEL_DIR}/" || return 1
}
if [[ -n "${NATIVE_REUSE_DIR}" && -d "${NATIVE_REUSE_DIR}" && "${VLLM_REF}" =~ ^[0-9a-f]{40}$ ]]; then
    if prepare_native_reuse; then
        NATIVE_REUSE=1
        printf 'reuse_mode=native\nreuse_source=%s\n' \
            "${NATIVE_REUSE_DIR}" >>"${MANIFEST}"
        echo "Native fingerprint matched; reusing vLLM wheel and SAIL extensions"
    else
        rm -rf "${PREBUILT_EXT_DIR}" "${VLLM_WHEEL_DIR:?}/"*
        echo "Native reuse unavailable or invalid; compiling"
    fi
fi

if ((NATIVE_REUSE == 0)); then
    git init "${VLLM_SRC}"
    git -C "${VLLM_SRC}" remote add origin "${VLLM_REPOSITORY}"
    # setuptools-scm needs history and release tags, including for commit/branch refs.
    # A depth-1 checkout can otherwise produce an unverifiable 0.1.dev1 version.
    for attempt in 1 2 3; do
        if git -C "${VLLM_SRC}" fetch --tags origin "${VLLM_REF}"; then
            break
        fi
        if ((attempt == 3)); then
            fail "failed to fetch vLLM after ${attempt} attempts"
        fi
        sleep "$((2 ** (attempt - 1)))"
    done
    git -C "${VLLM_SRC}" checkout --detach FETCH_HEAD
    if ((PREPARED_ENV)); then
        [[ "$(git -C "${VLLM_SRC}" rev-parse HEAD)" == "${VLLM_REF}" ]] || fail "fetched vLLM commit differs from the checkout's pin"
    fi
    (
        cd "${VLLM_SRC}"
        export RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://rsproxy.cn}"
        export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://rsproxy.cn/rustup}"
        if ! command -v rustup >/dev/null 2>&1; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
                | sh -s -- -y --profile minimal --default-toolchain none
        fi
        if [[ -r "${CARGO_HOME:-$HOME/.cargo}/env" ]]; then
            # shellcheck disable=SC1091
            source "${CARGO_HOME:-$HOME/.cargo}/env"
        fi
        # Fetch Cargo Git dependencies through the Git CLI to honor job-local Git configuration.
        export CARGO_NET_GIT_FETCH_WITH_CLI=true
        # Follow the Rust toolchain pinned by the checked-out vLLM source.
        # Install the TOML fallback for Python 3.10 before reading the toolchain.
        python -m pip install -r "${ROOT_DIR}/requirements/build.txt"
        RUSTUP_TOOLCHAIN="$(python - <<'PY'
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

print(tomllib.loads(Path("rust-toolchain.toml").read_text())["toolchain"]["channel"])
PY
        )"
        export RUSTUP_TOOLCHAIN
        rustup toolchain install "${RUSTUP_TOOLCHAIN}" --profile minimal
        rustc --version | tee "${LOG_DIR}/rustc-version.log"
        cargo --version | tee "${LOG_DIR}/cargo-version.log"
        python tools/use_existing_torch.py
        python -m pip install -r requirements/build/cuda.txt
        export VLLM_TARGET_DEVICE=empty VLLM_REQUIRE_RUST_FRONTEND=1
        export MAX_JOBS="${BUILD_JOBS}" CARGO_BUILD_JOBS="${BUILD_JOBS}"
        # Compile explicitly: pip hides successful backend output by default, and
        # a wheel's presence alone does not demonstrate a local Cargo build.
        python setup.py build_rust --release --inplace \
            2>&1 | tee "${LOG_DIR}/vllm-rust-build.log"
        python - <<'PY'
from pathlib import Path

binary = Path("vllm/vllm-rs")
extensions = list(Path("vllm").glob("_rust_tool_parser.*.so"))
if not binary.is_file() or binary.stat().st_size == 0 or len(extensions) != 1:
    raise SystemExit("required local Rust artifacts were not built")
for artifact in (binary, *extensions):
    if artifact.stat().st_size == 0:
        raise SystemExit(f"empty Rust artifact: {artifact}")
    print(f"Local Rust artifact: {artifact} ({artifact.stat().st_size} bytes)")
PY
        # Upstream setup.py packages these locally built artifacts without rebuilding.
        python -m pip wheel --verbose --no-build-isolation --no-deps \
            --wheel-dir "${VLLM_WHEEL_DIR}" .
    ) 2>&1 | tee "${LOG_DIR}/vllm-build.log"
fi

mapfile -t vllm_wheels < <(
    find "${VLLM_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print
)
((${#vllm_wheels[@]} == 1)) || \
    fail "expected exactly one vLLM wheel, found ${#vllm_wheels[@]}"
python -m pip install --force-reinstall --no-deps "${vllm_wheels[0]}"

# The CUDA-free build image intentionally does not carry runtime-only PPU
# packages such as tokenspeed-mla. They are protected by the image constraints
# and supplied by the separate CUDA-compatible runtime image, so install only
# the dependencies needed to compile or repackage the plugin here.
python -m pip install -r "${ROOT_DIR}/requirements/build.txt"
if ((NATIVE_REUSE == 0)); then
    (
        cd "${ROOT_DIR}"
        python - <<'PY'
from vllm_sail.native.toolchain import decide, probe
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility()
decision = decide(probe())
print(decision.reason)
if not decision.build:
    raise SystemExit(1)
PY
    ) 2>&1 | tee "${LOG_DIR}/native-preflight.log"
else
    (
        cd "${ROOT_DIR}"
        python - <<'PY'
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility()
print("native extensions supplied by verified donor wheel")
PY
    ) 2>&1 | tee "${LOG_DIR}/native-preflight.log"
    export VLLM_SAIL_PREBUILT_EXT_DIR="${PREBUILT_EXT_DIR}"
fi
(
    cd "${ROOT_DIR}"
    VERBOSE=1 MAX_JOBS="${BUILD_JOBS}" \
        python setup.py bdist_wheel --dist-dir "${SAIL_WHEEL_DIR}"
) 2>&1 | tee "${LOG_DIR}/vllm-sail-build.log"

VLLM_COMMIT="${VLLM_REF}"
manifest_vllm_args=(--vllm-source "${VLLM_SRC}")
if ((NATIVE_REUSE == 1)); then
    manifest_vllm_args=(--vllm-commit "${VLLM_COMMIT}")
else
    VLLM_COMMIT="$(git -C "${VLLM_SRC}" rev-parse HEAD)"
fi
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" create \
    "${ARTIFACT_DIR}/wheels" "${manifest_vllm_args[@]}" \
    --sail-source "${ROOT_DIR}" "${fingerprint_args[@]}"

mapfile -t sail_wheels < <(
    find "${SAIL_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print
)
((${#sail_wheels[@]} == 1)) || \
    fail "expected exactly one vLLM SAIL wheel, found ${#sail_wheels[@]}"
[[ "$(basename "${sail_wheels[0]}")" != *-none-any.whl ]] || \
    fail "native build unexpectedly produced a pure-Python wheel"
[[ -s "${vllm_wheels[0]}" ]] || fail "vLLM wheel is empty"
[[ -s "${sail_wheels[0]}" ]] || fail "vLLM SAIL wheel is empty"

python "${ROOT_DIR}/scripts/ci/ppu_image_stack.py" verify \
    --sdk "${PPU_SDK}" --output "${LOG_DIR}/image-stack.json" | tee -a "${MANIFEST}"

{
    printf 'status=succeeded\n'
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'hgcc=%s\n' "$("${HGCC}" --version 2>&1 | head -n 1)"
    printf 'reuse_mode=%s\n' "$([[ ${NATIVE_REUSE} == 1 ]] && printf native || printf full)"
    if ((NATIVE_REUSE == 0)); then
        printf 'rustc=%s\n' "$(cat "${LOG_DIR}/rustc-version.log")"
        printf 'cargo=%s\n' "$(cat "${LOG_DIR}/cargo-version.log")"
    fi
    printf 'vllm_commit=%s\n' "${VLLM_COMMIT}"
    printf 'vllm_wheel=%s\n' "$(basename "${vllm_wheels[0]}")"
    printf 'vllm_sail_commit=%s\n' "$(git -c "safe.directory=${ROOT_DIR}" -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'vllm_sail_wheel=%s\n' "$(basename "${sail_wheels[0]}")"
} >>"${MANIFEST}"
printf 'Build succeeded.\nvLLM: %s\nvLLM SAIL: %s\nManifest: %s\n' \
    "${vllm_wheels[0]}" "${sail_wheels[0]}" "${MANIFEST}"
