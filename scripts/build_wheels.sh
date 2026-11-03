#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/build_wheels.sh [--output-dir DIRECTORY]

Build the vLLM commit pinned by this checkout with VLLM_TARGET_DEVICE=empty,
compile its Rust frontend and parser locally, then build the vllm-sail wheel.
Run inside an activated Linux SAIL SDK + SAIL PyTorch environment (Python
3.10-3.13, Bash 4+). No device is needed to compile.

Options:
  --output-dir DIRECTORY  Empty output directory (default: artifacts/local-<run>)
  -h, --help              Show this help

Environment:
  PPU_SDK            SDK directory (default: /usr/local/PPU_SDK)
  HGCC               Compiler path (default: $PPU_SDK/bin/hgcc)
  PYTORCH_SAIL_ARCH  Targets (default: ppu_15;ppu_10)
  BUILD_JOBS         Parallel jobs (default: memory-aware, at most 16)
  VLLM_REPOSITORY    Git URL or local mirror for upstream vLLM
  ARTIFACT_DIR       Output directory, overridden by --output-dir
  CMAKE_HGCC_DIR     Optional local cmake-hgcc checkout

The script installs build requirements and the newly built vLLM wheel into
the active Python environment, preserving SDK package versions. It does not
install runtime dependencies or the resulting vllm-sail wheel. Pip uses your
current index configuration. Rustup is bootstrapped if missing; its toolchain
comes from vLLM's rust-toolchain.toml. RUSTUP_DIST_SERVER/RUSTUP_UPDATE_ROOT
can override the build pipeline's default rsproxy.cn mirrors.

Outputs: wheels/vllm/*.whl, wheels/vllm-sail/*.whl, wheels/wheel-manifest.json,
build-manifest.txt and logs/ (including vllm-rust-build.log).
EOF
}

while (($#)); do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --output-dir)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf '%s\n' '--output-dir requires a directory' >&2
                exit 2
            fi
            ARTIFACT_DIR="$2"
            shift 2
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

# Always use this checkout's verified upstream pin, even if CI overrides leaked
# into the caller's environment. A local invocation must compile both projects.
VLLM_REF="$(tr -d '[:space:]' < "${ROOT_DIR}/.github/vllm-main-verified.commit")"
[[ "${VLLM_REF}" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'expected a full vLLM commit in .github/vllm-main-verified.commit\n' >&2
    exit 2
}
export VLLM_REF
unset VLLM_SAIL_NATIVE_REUSE_DIR

export ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts/local-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "${ARTIFACT_DIR}"
ARTIFACT_DIR="$(cd -- "${ARTIFACT_DIR}" && pwd)"
[[ -z "$(find "${ARTIFACT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    printf 'output directory must be empty: %s\n' "${ARTIFACT_DIR}" >&2
    exit 2
}
printf 'vLLM commit: %s\nOutput: %s\n' "${VLLM_REF}" "${ARTIFACT_DIR}"
exec bash "${ROOT_DIR}/scripts/ci/build_ppu_wheels.sh" --prepared-env
