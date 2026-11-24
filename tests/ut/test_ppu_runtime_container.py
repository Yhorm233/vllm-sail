# SPDX-License-Identifier: Apache-2.0
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/ppu_runtime_container.sh"


def test_runtime_wrapper_uses_preloaded_image_and_explicit_ppu_devices():
    text = SCRIPT.read_text()
    assert 'bash "${script_dir}/ppu_runtime_image.sh" 9>&-' in text
    assert "${source_dir}/scripts/ci/ppu_runtime_image.sh" not in text
    resolver = (ROOT / "scripts/ci/ppu_runtime_image.sh").read_text()
    assert "Runner must configure PPU_CI_RUNTIME_IMAGE" in resolver
    assert 'docker image inspect "${runtime_image}"' in resolver
    assert 'docker pull "${runtime_image}" >"${pull_log}" 2>&1' in resolver
    assert "docker image ls" not in resolver
    assert "@sha256:" not in resolver
    assert "docker_release/llm" not in resolver
    assert "known_image_id" not in resolver
    assert "runtime-image.txt" not in text
    assert "--device=/dev/alixpu" in text
    assert "--device=/dev/alixpu_ctl" in text
    assert "--device=/dev/alixpu_sep" in text
    assert "/dev/alixpu_ppu${index}" in text
    assert "--env NVIDIA_VISIBLE_DEVICES=void" in text
    assert '--env CUDA_VISIBLE_DEVICES="${visible_csv}"' in text
    assert "--env PYTHONPYCACHEPREFIX=/tmp/pycache" in text
    assert "--privileged" not in text


def test_runtime_wrapper_mounts_inputs_read_only_and_artifacts_writable():
    text = SCRIPT.read_text()
    assert "target=/source,readonly" in text
    assert "target=/wheels,readonly" in text
    assert "source=${hf_home},target=${hf_home},readonly" in text
    # The view tree usually points outside HF_HOME, so both device lanes have to
    # mount the resolved snapshot explicitly.
    assert "mount_snapshot_if_external" in text
    assert "source=$1,target=$1,readonly" in text
    assert r"models--${model_id//\//--}" in text
    assert "--env HF_HUB_OFFLINE=1" in text
    assert "--env TRANSFORMERS_OFFLINE=1" in text
    assert 'target=/artifacts"' in text
    assert "--env BASELINE_ROOT=/source/tests/e2e/perf/baselines" in text
    assert "--env BASELINE_OUTPUT_ROOT=/artifacts/baselines" in text
    assert "/source/scripts/ci/ppu_pin_ut.sh" in text
    assert "/source/scripts/ci/ppu_e2e_smoke.sh" in text


@pytest.mark.parametrize("branch", ["upstream-reuse", "model-test", "model-accuracy"])
def test_runtime_wrapper_keeps_dynamic_module_cache_writable(branch):
    text = SCRIPT.read_text()
    assert "hf_modules_cache=/tmp/hf-modules" in text
    branch_body = text.split(f"\n    {branch})\n", 1)[1].split("\n        ;;", 1)[0]
    assert '--env HF_MODULES_CACHE="${hf_modules_cache}"' in branch_body


def test_runtime_wrapper_keeps_deep_gemm_cache_outside_the_container():
    text = SCRIPT.read_text()
    # Left to its default the cache lands under VLLM_CACHE_ROOT, i.e. in the
    # container's writable layer, which docker rm discards -- every accuracy
    # case then repeats the DeepGEMM JIT pass.
    assert ': "${PPU_CI_DG_CACHE_DIR:?PPU_CI_DG_CACHE_DIR is required' in text
    assert 'dg_cache_root=$(realpath "${PPU_CI_DG_CACHE_DIR}")' in text
    assert '[[ -d "${dg_cache_root}" ]]' in text
    assert '--mount "type=bind,source=${dg_cache_root},target=${dg_cache_root}"' in text
    # deep_gemm reads DG_JIT_CACHE_DIR; the SDK builds also honour the older
    # DG_CACHE_DIR, so both names point at the same directory.
    assert '--env DG_JIT_CACHE_DIR="${dg_cache_root}"' in text
    assert '--env DG_CACHE_DIR="${dg_cache_root}"' in text
    # One root serves every case: deep_gemm keys the cache by the toolchain
    # that produced the binaries, so the wrapper must not invent a second level
    # of its own on top of that.
    assert "${image_id#sha256:}" not in text
    assert "chmod 0777" not in text


def test_accuracy_workflow_supplies_the_deep_gemm_cache_root():
    workflow = (ROOT / ".github/workflows/_ppu-accuracy-tests.yaml").read_text()
    # The wrapper fails closed on a missing root, so the caller has to pass one;
    # the default keeps the nightly and the manual debug lane on one value.
    assert "PPU_CI_DG_CACHE_DIR: ${{ inputs.dg_cache_dir }}" in workflow
    assert "dg_cache_dir:" in workflow
    assert "default: /nas_aisw/vllmci_dgcache" in workflow


@pytest.mark.parametrize("mode", ["pin-ut", "upstream-reuse"])
def test_ut_and_reuse_require_exact_read_only_upstream_source(mode):
    text = SCRIPT.read_text()
    branch = text.split(f"\n    {mode})\n", 1)[1].split("\n        ;;", 1)[0]
    assert "PPU_CI_UPSTREAM_DIR is required" in branch
    assert "source=${upstream_dir},target=/upstream,readonly" in branch
    assert "--env VLLM_SOURCE_ROOT=/upstream" in branch
    assert 'rev-parse HEAD)" == "${expected_vllm_commit}"' in branch
    assert "exit 2" in branch


def test_runtime_wrapper_serializes_jobs_and_verifies_source_identity():
    text = SCRIPT.read_text()
    assert "flock -n 9" in text
    assert "rev-parse HEAD" in text
    assert "PPU_CI_SOURCE_SHA" in text
    assert ".github/vllm-main-verified.commit" in text
    assert '--env EXPECTED_VLLM_COMMIT="${expected_vllm_commit}"' in text
    assert '--env EXPECTED_SAIL_COMMIT="${expected_sail_commit}"' in text
    assert "docker stop --time=2" in text
    assert "docker rm --force" in text


@pytest.mark.parametrize(
    "bad_env",
    [
        {"PPU_CI_RUNTIME_IMAGE_TAG": "evil.example/vllm-sail:latest"},
        {"PPU_CI_RUNTIME_IMAGE_TAG": "team/vllm-sail:nightly"},
        {"PPU_CI_RUNTIME_IMAGE_TAG": "nightly@sha256:" + "a" * 64},
        {"PPU_CI_RUNTIME_IMAGE_TAG": "../escape"},
        {"PPU_CI_RUNTIME_IMAGE_TAG": "-flag"},
        {"PPU_CI_RUNTIME_IMAGE_TAG": ".hidden"},
        {"PPU_CI_RUNTIME_IMAGE_TAG": "has space"},
        {"PPU_CI_RUNTIME_IMAGE_TAG": "t" * 129},
        {"PPU_CI_RUNTIME_IMAGE_DIGEST": "sha256:short"},
        {"PPU_CI_RUNTIME_IMAGE_DIGEST": "sha256:" + "A" * 64},
        {"PPU_CI_RUNTIME_IMAGE_DIGEST": "evil.example/repo@sha256:" + "a" * 64},
        {
            "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122",
            "PPU_CI_RUNTIME_IMAGE_REPO": "evil.example/other",
            "PPU_CI_RUNTIME_IMAGE_ALLOWED_REPOS": "registry.internal/team/",
        },
        {
            "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122",
            "PPU_CI_RUNTIME_IMAGE_REPO": "registry.internal/other/vllm",
        },
        {"PPU_CI_RUNTIME_VERSION": "v0.34.0", "PPU_CI_RUNTIME_IMAGE_V034": ""},
    ],
)
def test_runtime_image_rejects_illegal_tag_digest_and_cross_repo(tmp_path, bad_env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = bin_dir / "docker"
    docker.write_text(
        f'#!/usr/bin/env bash\necho "$*" >> {docker_log}\nexit 0\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "PPU_CI_RUNTIME_IMAGE": "registry.internal/team/vllm:base",
            **bad_env,
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert not docker_log.exists()
    assert "registry.internal" not in proc.stdout + proc.stderr


@pytest.mark.parametrize("version", ["main", "v0.34.0"])
def test_runtime_image_forces_pull_and_verifies_digest_without_leaking_repo(
    tmp_path, version
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    expected_id = "sha256:" + "1" * 64
    expected_digest = "sha256:" + "2" * 64
    wrong_digest = "sha256:" + "3" * 64
    image_name = "vllm" if version == "main" else "vllm-v034"
    default_ref = f"registry.internal/team/{image_name}-base:base"
    override_repo = f"registry.internal/team/{image_name}-nightly"
    docker = bin_dir / "docker"
    docker.write_text(
        f"""#!/usr/bin/env bash
echo "$*" >> "{docker_log}"
if [[ "$1" == "pull" ]]; then
    echo "Digest: {expected_digest}"
    exit 0
fi
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
    if [[ "$*" == *"{{.Id}}"* ]]; then
        echo "{expected_id}"
        exit 0
    fi
    if [[ "$*" == *"RepoDigests"* ]]; then
        echo "${{FAKE_REPO_DIGEST:-{override_repo}@{expected_digest}}}"
        exit 0
    fi
fi
exit 1
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    digest_out = tmp_path / "digest.txt"

    base_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PPU_CI_RUNTIME_IMAGE": "registry.internal/team/vllm-base:base",
        "PPU_CI_RUNTIME_IMAGE_REPO": "registry.internal/team/vllm-nightly",
        "PPU_CI_RUNTIME_VERSION": version,
        "PPU_CI_RUNTIME_IMAGE_V034": "registry.internal/team/vllm-v034-base:base",
        "PPU_CI_RUNTIME_IMAGE_REPO_V034": "registry.internal/team/vllm-v034-nightly",
        "PPU_CI_RUNTIME_IMAGE_ALLOWED_REPOS": "registry.internal/team/",
        "PPU_CI_RUNTIME_DIGEST_OUT": str(digest_out),
    }

    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == expected_id
    assert f"image inspect {default_ref} " in docker_log.read_text()
    docker_log.unlink()

    # Tag-only override forces docker pull and records redacted @sha256 digest.
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env={**base_env, "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected_id
    assert digest_out.read_text().strip() == f"@{expected_digest}"
    assert "registry.internal" not in proc.stdout + proc.stderr + digest_out.read_text()
    assert f"pull {override_repo}:nightly-20261122" in docker_log.read_text()

    # Tag + matching digest pulls by immutable @sha256 and succeeds.
    docker_log.unlink()
    proc2 = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env={
            **base_env,
            "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122",
            "PPU_CI_RUNTIME_IMAGE_DIGEST": expected_digest,
        },
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, proc2.stderr
    assert f"pull {override_repo}@{expected_digest}" in docker_log.read_text()

    # Mismatched digest fails closed.
    proc3 = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env={
            **base_env,
            "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122",
            "PPU_CI_RUNTIME_IMAGE_DIGEST": wrong_digest,
        },
        capture_output=True,
        text=True,
    )
    assert proc3.returncode == 2
    assert "registry.internal" not in proc3.stdout + proc3.stderr


def test_runtime_image_handles_multiple_repo_digests_without_stale_first_match(
    tmp_path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    expected_id = "sha256:" + "1" * 64
    older_digest = "sha256:" + "3" * 64
    pulled_digest = "sha256:" + "2" * 64
    docker = bin_dir / "docker"
    docker.write_text(
        f"""#!/usr/bin/env bash
if [[ "$1" == "pull" ]]; then
    if [[ "${{OMIT_PULL_DIGEST:-0}}" != "1" ]]; then
        echo "Digest: {pulled_digest}"
    fi
    exit 0
fi
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
    if [[ "$*" == *"{{.Id}}"* ]]; then
        echo "{expected_id}"
        exit 0
    fi
    if [[ "$*" == *"RepoDigests"* ]]; then
        printf '%s\\n%s\\n' \\
            "registry.internal/team/vllm-nightly@{older_digest}" \\
            "registry.internal/team/vllm-nightly@{pulled_digest}"
        exit 0
    fi
fi
exit 1
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    digest_out = tmp_path / "digest.txt"
    base_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PPU_CI_RUNTIME_IMAGE": "registry.internal/team/vllm-nightly:base",
        "PPU_CI_RUNTIME_DIGEST_OUT": str(digest_out),
    }

    # Tag override resolves the freshly pulled digest, not the older first RepoDigest.
    proc_tag = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env={**base_env, "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122"},
        capture_output=True,
        text=True,
    )
    assert proc_tag.returncode == 0, proc_tag.stderr
    assert digest_out.read_text().strip() == f"@{pulled_digest}"

    # Pinned digest matching the second RepoDigests entry succeeds.
    proc_pin = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env={
            **base_env,
            "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122",
            "PPU_CI_RUNTIME_IMAGE_DIGEST": pulled_digest,
        },
        capture_output=True,
        text=True,
    )
    assert proc_pin.returncode == 0, proc_pin.stderr
    assert digest_out.read_text().strip() == f"@{pulled_digest}"

    # Ambiguous multi-RepoDigests without a pull Digest line fails closed on tag override.
    proc_ambig = subprocess.run(
        ["bash", str(ROOT / "scripts/ci/ppu_runtime_image.sh")],
        env={
            **base_env,
            "OMIT_PULL_DIGEST": "1",
            "PPU_CI_RUNTIME_IMAGE_TAG": "nightly-20261122",
        },
        capture_output=True,
        text=True,
    )
    assert proc_ambig.returncode == 2


def test_runtime_container_mounts_trusted_ci_entrypoint_for_separate_target_source(
    tmp_path,
):
    text = SCRIPT.read_text()
    assert 'container_ci_dir="/trusted-ci"' in text
    assert "target=/trusted-ci,readonly" in text
    assert "--env PPU_CI_SOURCE_ROOT=/source" in text
    assert '"${container_ci_dir}/ppu_e2e_recipe_sanity.sh"' in text
    assert '"${container_ci_dir}/ppu_e2e_accuracy.sh"' in text


def test_accuracy_self_check_redacts_raw_version_on_compat_failure(tmp_path):
    script_text = (ROOT / "scripts/ci/ppu_e2e_accuracy.sh").read_text(encoding="utf-8")
    marker = 'python - "${RESULTS_DIR}" "${SKIP_WHEEL_INSTALL}" "${PPU_CI_RUNTIME_IMAGE_DIGEST:-}" "${PPU_CI_RUNTIME_IMAGE_ID:-}" <<\'PY\'\n'
    block = script_text.split(marker, 1)[1].split("\nPY\n", 1)[0]

    stubs = tmp_path / "stubs"
    (stubs / "vllm").mkdir(parents=True)
    (stubs / "vllm_sail").mkdir(parents=True)
    (stubs / "torch.py").write_text(
        '__version__ = "2.13.0+10.1.2.3"\n'
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return True\n",
        encoding="utf-8",
    )
    (stubs / "vllm/__init__.py").write_text(
        '__version__ = "0.10.2+registry.internal"\n', encoding="utf-8"
    )
    (stubs / "vllm/platforms.py").write_text(
        "class _P:\n"
        "    def is_ppu(self): return True\n"
        '    def get_device_name(self, idx): return "ZW-M890P"\n'
        "    def get_device_capability(self): return (9, 0)\n"
        "current_platform = _P()\n",
        encoding="utf-8",
    )
    (stubs / "vllm_sail/__init__.py").write_text(
        '__version__ = "0.34.0.dev1+internal.host"\n', encoding="utf-8"
    )
    (stubs / "vllm_sail/compat.py").write_text(
        "def check_vllm_compatibility(force=True):\n"
        "    raise RuntimeError(\"installed version is '0.10.2+registry.internal'\")\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            block,
            str(tmp_path),
            "true",
            "@sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(stubs)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "vLLM compatibility check failed (RuntimeError)" in combined
    assert "vllm=0.10.2" in combined
    assert "vllm_sail=0.34.0.dev1" in combined
    assert "registry.internal" not in combined
    assert "internal.host" not in combined
    assert "10.1.2.3" not in combined


def test_runtime_container_prebuilt_mode_guards_and_relaxed_wheels():
    text = SCRIPT.read_text()
    assert 'skip_wheel_install="${PPU_CI_SKIP_WHEEL_INSTALL:-false}"' in text
    assert (
        "PPU_CI_SKIP_WHEEL_INSTALL is only supported for recipe-sanity and model-accuracy"
        in text
    )
    assert (
        "PPU_CI_SKIP_WHEEL_INSTALL is not allowed when PPU_CI_REQUIRE_PR_ARTIFACT is true"
        in text
    )
    assert (
        "PPU_CI_WHEELS_DIR must be unset when PPU_CI_SKIP_WHEEL_INSTALL is true" in text
    )
    assert "Prebuilt image mode requires a verified runtime image digest" in text
    assert '--env PPU_CI_SKIP_WHEEL_INSTALL="${skip_wheel_install}"' in text
    assert '--env PPU_CI_RUNTIME_IMAGE_DIGEST="${runtime_image_digest}"' in text
    assert '--env PPU_CI_RUNTIME_IMAGE_ID="${image_id}"' in text


def test_accuracy_script_redacts_raw_image_stack_in_stdout_and_artifacts(tmp_path):
    ci_dir = tmp_path / "trusted-ci"
    ci_dir.mkdir()
    shutil.copyfile(
        ROOT / "scripts/ci/ppu_e2e_accuracy.sh", ci_dir / "ppu_e2e_accuracy.sh"
    )
    # Stub ppu_image_stack.py to reproduce a raw dotted local version in capture
    # output and in the captured JSON, plus verify failure behavior.
    (ci_dir / "ppu_image_stack.py").write_text(
        "import argparse, json, os, sys\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('cmd')\n"
        "p.add_argument('--sdk')\n"
        "p.add_argument('--output', type=Path)\n"
        "p.add_argument('--constraints', type=Path)\n"
        "a = p.parse_args()\n"
        "if os.environ.get('FAIL_STACK') == a.cmd:\n"
        "    print('torch_version=2.13.0+example.invalid', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "if a.cmd == 'capture':\n"
        "    doc = {\n"
        "        'stack_source': 'image',\n"
        "        'fingerprint_kind': 'installed-tree-sha256',\n"
        "        'sdk_path': '/usr/local/PPU_SDK',\n"
        "        'sdk_sha256': 'a' * 64,\n"
        "        'torch_path': '/opt/torch',\n"
        "        'torch_sha256': 'b' * 64,\n"
        "        'torch_version': '2.13.0+example.invalid',\n"
        "        'packages': {'torch': '2.13.0+example.invalid', 'triton': '3.1.0+ok_1'},\n"
        "    }\n"
        "    a.output.write_text(json.dumps(doc), encoding='utf-8')\n"
        "    a.constraints.write_text('torch===2.13.0+example.invalid\\n', encoding='utf-8')\n"
        "    print('torch_version=2.13.0+example.invalid')\n"
        "elif a.cmd == 'verify':\n"
        "    print('stack_unchanged=true')\n",
        encoding="utf-8",
    )

    source_dir = tmp_path / "source"
    (source_dir / "scripts/eval").mkdir(parents=True)
    (source_dir / "scripts/eval/ppu_gsm8k_eval.py").write_text(
        "print('gsm8k eval ok')\n", encoding="utf-8"
    )

    stubs = tmp_path / "stubs"
    (stubs / "vllm").mkdir(parents=True)
    (stubs / "vllm_sail").mkdir(parents=True)
    (stubs / "evalscope").mkdir(parents=True)
    (stubs / "huggingface_hub").mkdir(parents=True)
    (stubs / "torch.py").write_text(
        '__version__ = "2.13.0+example.invalid"\n'
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available(): return True\n",
        encoding="utf-8",
    )
    (stubs / "vllm/__init__.py").write_text(
        '__version__ = "0.10.2+example.invalid"\n', encoding="utf-8"
    )
    (stubs / "vllm/platforms.py").write_text(
        "class _P:\n"
        "    def is_ppu(self): return True\n"
        '    def get_device_name(self, idx): return "ZW-M890P"\n'
        "    def get_device_capability(self): return (9, 0)\n"
        "current_platform = _P()\n",
        encoding="utf-8",
    )
    (stubs / "vllm_sail/__init__.py").write_text(
        '__version__ = "0.34.0.dev1+example.invalid"\n', encoding="utf-8"
    )
    (stubs / "vllm_sail/compat.py").write_text(
        "def check_vllm_compatibility(force=True): return None\n", encoding="utf-8"
    )
    (stubs / "evalscope/__init__.py").write_text(
        "TaskConfig = object\ndef run_task(*a, **k): pass\n", encoding="utf-8"
    )
    (stubs / "huggingface_hub/__init__.py").write_text(
        f'def snapshot_download(**k): return "{tmp_path / "snapshot"}"\n',
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Wrapper python that keeps stubs on sys.path even after script unsets PYTHONPATH.
    (bin_dir / "python").write_text(
        f"#!/usr/bin/env bash\n"
        f'export PYTHONPATH="{stubs}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    (bin_dir / "python").chmod(0o755)
    (bin_dir / "ppu-smi").write_text("#!/usr/bin/env bash\necho ppu-smi-ok\n")
    (bin_dir / "ppu-smi").chmod(0o755)

    results_dir = tmp_path / "results"
    base_env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PPU_CI_SOURCE_ROOT": str(source_dir),
        "RESULTS_DIR": str(results_dir),
        "HF_HOME": str(tmp_path / "hf"),
        "GSM8K_DATASET_PATH": str(tmp_path / "gsm8k"),
        "MODEL_KEY": "qwen3_8_27b",
        "MODEL_ID": "Qwen/Qwen3.8-27B",
        "PPU_CI_SKIP_WHEEL_INSTALL": "true",
        "PPU_CI_RUNTIME_IMAGE_DIGEST": "@sha256:" + "a" * 64,
        "PPU_CI_RUNTIME_IMAGE_ID": "sha256:" + "b" * 64,
        "GOLDEN_SEED_DIR": str(tmp_path / "no-golden"),
    }
    proc = subprocess.run(
        ["bash", str(ci_dir / "ppu_e2e_accuracy.sh")],
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "example.invalid" not in proc.stdout
    assert "example.invalid" not in proc.stderr
    assert not (results_dir / "image-constraints.txt").exists()
    for artifact in results_dir.rglob("*"):
        if artifact.is_file():
            assert "example.invalid" not in artifact.read_text(encoding="utf-8"), (
                artifact
            )
    stack_doc = json.loads((results_dir / "image-stack.json").read_text())
    assert stack_doc["torch_version"] == "2.13.0"
    assert stack_doc["packages"] == {"torch": "2.13.0", "triton": "3.1.0+ok_1"}
    assert "sdk_path" not in stack_doc and "torch_path" not in stack_doc
    fp = json.loads((results_dir / "runtime-fingerprint.json").read_text())
    assert fp["versions"] == {
        "torch": "2.13.0",
        "vllm": "0.10.2",
        "vllm_sail": "0.34.0.dev1",
    }

    # Capture failure path must also avoid leaking raw versions.
    fail_results = tmp_path / "results-fail"
    fail_proc = subprocess.run(
        ["bash", str(ci_dir / "ppu_e2e_accuracy.sh")],
        env={**base_env, "RESULTS_DIR": str(fail_results), "FAIL_STACK": "capture"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert fail_proc.returncode != 0
    assert "example.invalid" not in (fail_proc.stdout + fail_proc.stderr)
    assert "example.invalid" not in (fail_results / "accuracy.log").read_text()
