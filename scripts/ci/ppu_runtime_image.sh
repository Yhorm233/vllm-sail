#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resolve the runner-configured CUDA-compatible PPU runtime to an immutable ID.
set -Eeuo pipefail

case "${PPU_CI_RUNTIME_VERSION:-}" in
    ""|main)
        : "${PPU_CI_RUNTIME_IMAGE:?Runner must configure PPU_CI_RUNTIME_IMAGE}"
        runtime_image="${PPU_CI_RUNTIME_IMAGE}"
        override_repo="${PPU_CI_RUNTIME_IMAGE_REPO:-}"
        ;;
    v0.34.0)
        if [[ -z "${PPU_CI_RUNTIME_IMAGE_V034:-}" ]]; then
            echo "Runner must configure PPU_CI_RUNTIME_IMAGE_V034" >&2
            exit 2
        fi
        runtime_image="${PPU_CI_RUNTIME_IMAGE_V034}"
        override_repo="${PPU_CI_RUNTIME_IMAGE_REPO_V034:-}"
        ;;
    *)
        echo "Unsupported PPU runtime version" >&2
        exit 2
        ;;
esac

image_tag="${PPU_CI_RUNTIME_IMAGE_TAG:-}"
image_digest="${PPU_CI_RUNTIME_IMAGE_DIGEST:-}"
digest_out="${PPU_CI_RUNTIME_DIGEST_OUT:-}"

strip_image_repo() {
    local ref=$1
    local no_digest=${ref%%@*}
    local last_segment=${no_digest##*/}
    if [[ "${last_segment}" == *:* ]]; then
        printf '%s\n' "${no_digest%:*}"
    else
        printf '%s\n' "${no_digest}"
    fi
}

validate_repo_name() {
    local repo=$1
    [[ -n "${repo}" ]] || return 1
    [[ "${repo}" != *".."* && "${repo}" != *"//"* && "${repo}" != *"@"* ]] || return 1
    local last_segment=${repo##*/}
    [[ "${last_segment}" != *:* ]] || return 1
    [[ "${repo}" =~ ^[a-z0-9]([a-z0-9._:/-]*[a-z0-9])?$ ]] || return 1
}

repo_is_allowed() {
    local candidate=$1
    local default_repo=$2
    local allowlist=${PPU_CI_RUNTIME_IMAGE_ALLOWED_REPOS:-}
    if [[ -z "${allowlist}" ]]; then
        [[ "${candidate}" == "${default_repo}" ]]
        return
    fi
    local entry trimmed
    IFS=',' read -r -a entries <<< "${allowlist}"
    for entry in "${entries[@]}"; do
        trimmed=$(printf '%s' "${entry}" | tr -d '[:space:]')
        [[ -n "${trimmed}" ]] || continue
        if [[ "${trimmed}" == */ ]]; then
            local prefix_no_slash=${trimmed%/}
            validate_repo_name "${prefix_no_slash}" || continue
            if [[ "${candidate}" == "${trimmed}"* && "${candidate}" != "${trimmed}" ]]; then
                return 0
            fi
        else
            validate_repo_name "${trimmed}" || continue
            if [[ "${candidate}" == "${trimmed}" ]]; then
                return 0
            fi
        fi
    done
    return 1
}

resolve_repo_digest() {
    local ref=$1
    local expected_repo=$2
    local pull_log=${3:-}
    local requested_digest=${4:-}
    local strict_override=${5:-false}
    local line pulled_digest=""
    local repo_digests=()

    while IFS= read -r line; do
        if [[ "${line}" == "${expected_repo}@"* ]]; then
            local candidate=${line#"${expected_repo}@"}
            if [[ "${candidate}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
                repo_digests+=("${candidate}")
            fi
        fi
    done < <(docker image inspect "${ref}" --format '{{range .RepoDigests}}{{println .}}{{end}}' 2>/dev/null || true)

    if [[ -n "${pull_log}" && -f "${pull_log}" ]]; then
        local pulled
        pulled=$(awk '$1 == "Digest:" {print $2}' "${pull_log}" | tail -1 || true)
        if [[ "${pulled}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            pulled_digest="${pulled}"
        fi
    fi

    digest_in_list() {
        local needle=$1 item
        for item in "${repo_digests[@]}"; do
            if [[ "${item}" == "${needle}" ]]; then
                return 0
            fi
        done
        return 1
    }

    if [[ -n "${requested_digest}" ]]; then
        if [[ -n "${pulled_digest}" && "${pulled_digest}" != "${requested_digest}" ]]; then
            return 1
        fi
        if (( ${#repo_digests[@]} > 0 )); then
            digest_in_list "${requested_digest}" || return 1
        elif [[ "${pulled_digest}" != "${requested_digest}" ]]; then
            return 1
        fi
        printf '%s\n' "${requested_digest}"
        return 0
    fi

    if [[ -n "${pulled_digest}" ]]; then
        if (( ${#repo_digests[@]} > 0 )) && ! digest_in_list "${pulled_digest}"; then
            return 1
        fi
        printf '%s\n' "${pulled_digest}"
        return 0
    fi

    if (( ${#repo_digests[@]} == 1 )); then
        printf '%s\n' "${repo_digests[0]}"
        return 0
    fi

    if [[ "${strict_override}" == "true" ]]; then
        return 1
    fi
    printf '\n'
    return 0
}

default_repo=$(strip_image_repo "${runtime_image}")
validate_repo_name "${default_repo}" || {
    echo "Runner PPU_CI_RUNTIME_IMAGE repository is invalid" >&2
    exit 2
}

resolved_digest=""
if [[ -z "${image_tag}" && -z "${image_digest}" ]]; then
    pull_log=""
    if ! image_id=$(docker image inspect "${runtime_image}" --format '{{.Id}}' 2>/dev/null); then
        private_dir=$(mktemp -d "${HOME:?}/.vllm-sail-ppu-pull.XXXXXX")
        chmod 0700 "${private_dir}"
        pull_log="${private_dir}/pull.log"
        if ! docker pull "${runtime_image}" >"${pull_log}" 2>&1; then
            echo "Failed to pull the runner-configured PPU runtime image" >&2
            exit 2
        fi
        image_id=$(docker image inspect "${runtime_image}" --format '{{.Id}}' 2>>"${pull_log}") || {
            echo "Failed to inspect the runner-configured PPU runtime image" >&2
            exit 2
        }
    fi
    resolved_digest=$(resolve_repo_digest "${runtime_image}" "${default_repo}" "${pull_log}" "" "false" || true)
else
    if [[ -n "${image_tag}" ]]; then
        if [[ "${image_tag}" == *".."* || ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
            echo "PPU_CI_RUNTIME_IMAGE_TAG must be a valid tag without registry or repository path" >&2
            exit 2
        fi
    fi
    if [[ -n "${image_digest}" ]]; then
        if [[ ! "${image_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            echo "PPU_CI_RUNTIME_IMAGE_DIGEST must be a valid sha256 digest" >&2
            exit 2
        fi
    fi
    target_repo="${override_repo:-${default_repo}}"
    validate_repo_name "${target_repo}" || {
        echo "Runner PPU runtime image repository configuration is invalid" >&2
        exit 2
    }
    repo_is_allowed "${target_repo}" "${default_repo}" || {
        echo "PPU runtime image repository is not in the runner allowlist" >&2
        exit 2
    }
    if [[ -n "${image_digest}" ]]; then
        pull_ref="${target_repo}@${image_digest}"
    else
        pull_ref="${target_repo}:${image_tag}"
    fi
    private_dir=$(mktemp -d "${HOME:?}/.vllm-sail-ppu-pull.XXXXXX")
    chmod 0700 "${private_dir}"
    pull_log="${private_dir}/pull.log"
    if ! docker pull "${pull_ref}" >"${pull_log}" 2>&1; then
        echo "Failed to pull the requested PPU runtime image" >&2
        exit 2
    fi
    image_id=$(docker image inspect "${pull_ref}" --format '{{.Id}}' 2>>"${pull_log}") || {
        echo "Failed to inspect the requested PPU runtime image" >&2
        exit 2
    }
    if ! resolved_digest=$(resolve_repo_digest "${pull_ref}" "${target_repo}" "${pull_log}" "${image_digest}" "true"); then
        echo "Docker did not resolve a verified repository digest for the requested PPU runtime image" >&2
        exit 2
    fi
    [[ "${resolved_digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
        echo "Docker did not resolve a verified repository digest for the requested PPU runtime image" >&2
        exit 2
    }
fi

[[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo "Docker did not resolve an immutable PPU runtime image ID" >&2
    exit 2
}
if [[ -n "${digest_out}" ]]; then
    if [[ -n "${resolved_digest}" ]]; then
        printf '@%s\n' "${resolved_digest}" >"${digest_out}"
    else
        : >"${digest_out}"
    fi
fi
printf '%s\n' "${image_id}"
