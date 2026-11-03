# SPDX-License-Identifier: Apache-2.0
"""Exercise artifact handoff without torch, vLLM, Rust or a device."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci/ppu_wheel_manifest.py"
spec = importlib.util.spec_from_file_location("ppu_wheel_manifest", SCRIPT)
assert spec is not None and spec.loader is not None
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def make_wheels(directory: Path, *, rust: bool = True) -> Path:
    for name, version in (("vllm", "0.30.0+empty"), ("vllm_sail", "0.1.dev1")):
        target = directory / name
        target.mkdir(exist_ok=True)
        wheel = target / f"{name}-{version}-cp312-cp312-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"{name}-{version}.dist-info/METADATA",
                f"Name: {name}\nVersion: {version}\n",
            )
            if name == "vllm" and rust:
                archive.writestr("vllm/vllm-rs", b"test executable")
                archive.writestr("vllm/_rust_tool_parser.abi3.so", b"test extension")
            elif name == "vllm_sail":
                for module in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
                    archive.writestr(f"vllm_sail/{module}.abi3.so", b"test extension")
    return next(directory.rglob("vllm-*.whl"))


def record(directory: Path) -> dict:
    data = {"schema": 1, "wheels": manifest.wheel_records(directory)}
    (directory / manifest.MANIFEST).write_text(json.dumps(data))
    return data


def test_preserves_real_wheel_versions_and_verifies_hashes(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    assert data["wheels"]["vllm"]["version"] == "0.30.0+empty"
    assert manifest.verify(tmp_path) == data


def test_rejects_missing_rust_even_when_wheel_exists(tmp_path: Path) -> None:
    make_wheels(tmp_path, rust=False)
    with pytest.raises(ValueError, match="required Rust artifacts"):
        manifest.wheel_records(tmp_path)


def test_rejects_ambiguous_wheel_set(tmp_path: Path) -> None:
    wheel = make_wheels(tmp_path)
    (wheel.parent / "vllm-extra.whl").write_bytes(wheel.read_bytes())
    with pytest.raises(ValueError, match="exactly one vllm wheel"):
        manifest.wheel_records(tmp_path)


def test_rejects_changed_wheel_bytes(tmp_path: Path) -> None:
    wheel = make_wheels(tmp_path)
    record(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("vllm/changed.py", "# substituted wheel\n")
    with pytest.raises(ValueError, match="differ from build manifest"):
        manifest.verify(tmp_path)


def test_rejects_manifest_version_override(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    data["wheels"]["vllm"]["version"] = "0.27.1"
    (tmp_path / manifest.MANIFEST).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differ from build manifest"):
        manifest.verify(tmp_path)


def test_legacy_artifact_without_manifest_fails_closed(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    with pytest.raises(FileNotFoundError):
        manifest.verify(tmp_path)


@pytest.mark.parametrize("stale_package", [None, "vllm", "vllm-sail", "missing"])
def test_cli_requires_the_pr_and_upstream_commits(
    tmp_path: Path, stale_package: str | None
) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    commits = {"vllm": "a" * 40, "vllm-sail": "b" * 40}
    data["commits"] = commits.copy()
    if stale_package == "missing":
        del data["commits"]
    elif stale_package:
        data["commits"][stale_package] = "c" * 40
    (tmp_path / manifest.MANIFEST).write_text(json.dumps(data))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            str(tmp_path),
            "--vllm-commit",
            commits["vllm"],
            "--sail-commit",
            commits["vllm-sail"],
        ],
        capture_output=True,
        text=True,
    )
    if stale_package:
        assert result.returncode != 0
        assert "source commit differs" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("different_owner", [False, True])
def test_cli_records_source_commits(tmp_path: Path, different_owner: bool) -> None:
    make_wheels(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "create",
            str(tmp_path),
            "--vllm-source",
            str(ROOT),
            "--sail-source",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_TEST_ASSUME_DIFFERENT_OWNER": str(int(different_owner)),
        },
    )
    assert result.returncode == 0, result.stderr
    data = manifest.verify(tmp_path)
    assert (
        data["commits"]["vllm"]
        == subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    )


def test_native_fingerprint_only_changes_for_native_inputs(tmp_path: Path) -> None:
    import shutil

    source = tmp_path / "source"
    for relative in manifest.NATIVE_INPUTS:
        original = ROOT / relative
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_dir():
            shutil.copytree(original, target)
        else:
            shutil.copy2(original, target)
    first = manifest.native_fingerprint(
        source, build_image="image@sha256:one", pytorch_sail_arch="ppu_15;ppu_10"
    )
    (source / "vllm_sail/runtime.py").parent.mkdir(parents=True, exist_ok=True)
    (source / "vllm_sail/runtime.py").write_text("python-only change\n")
    assert (
        manifest.native_fingerprint(
            source,
            build_image="image@sha256:one",
            pytorch_sail_arch="ppu_15;ppu_10",
        )
        == first
    )
    assert (
        manifest.native_fingerprint(
            source,
            build_image="image@sha256:one",
            pytorch_sail_arch="ppu_15;ppu_10",
            vllm_sail_hg_flags="-O3",
        )
        != first
    )
    native_file = next((source / "csrc").rglob("*.hg"))
    native_file.write_text(native_file.read_text() + "\n// changed\n")
    assert (
        manifest.native_fingerprint(
            source,
            build_image="image@sha256:one",
            pytorch_sail_arch="ppu_15;ppu_10",
        )
        != first
    )
    assert (
        manifest.native_fingerprint(
            source,
            build_image="image@sha256:two",
            pytorch_sail_arch="ppu_15;ppu_10",
        )
        != first
    )


def test_extract_native_requires_verified_reuse_metadata(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    with pytest.raises(ValueError, match="predates native reuse metadata"):
        manifest.extract_native(tmp_path, tmp_path / "out")

    sail_wheel = tmp_path / data["wheels"]["vllm-sail"]["file"]
    data["native_reuse"] = {
        "fingerprint": "a" * 64,
        "extensions": manifest.native_extension_records(sail_wheel),
    }
    (tmp_path / manifest.MANIFEST).write_text(json.dumps(data))
    extracted = tmp_path / "out"
    assert manifest.extract_native(tmp_path, extracted) == data["native_reuse"]
    assert {path.name for path in extracted.iterdir()} == {
        f"{module}.abi3.so" for module in manifest.NATIVE_MODULES
    }


def test_create_accepts_verified_upstream_commit_for_native_repack(
    tmp_path: Path,
) -> None:
    make_wheels(tmp_path)
    commit = "a" * 40
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "create",
            str(tmp_path),
            "--vllm-commit",
            commit,
            "--sail-source",
            str(ROOT),
            "--build-image",
            "image@sha256:one",
            "--pytorch-sail-arch",
            "ppu_15;ppu_10",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = manifest.verify(tmp_path, vllm_commit=commit)
    assert data["native_reuse"]["extensions"]
    assert len(data["native_reuse"]["fingerprint"]) == 64


def test_verify_rejects_null_native_metadata_and_empty_extensions(
    tmp_path: Path,
) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    data["native_reuse"] = None
    (tmp_path / manifest.MANIFEST).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata must be an object"):
        manifest.verify(tmp_path)

    sail = next(tmp_path.rglob("vllm_sail-*.whl"))
    replacement = tmp_path / "replacement.whl"
    with zipfile.ZipFile(sail) as source, zipfile.ZipFile(replacement, "w") as target:
        for name in source.namelist():
            target.writestr(
                name, b"" if name == "vllm_sail/_C.abi3.so" else source.read(name)
            )
    replacement.replace(sail)
    with pytest.raises(ValueError, match="empty native extension"):
        manifest.wheel_records(tmp_path)
