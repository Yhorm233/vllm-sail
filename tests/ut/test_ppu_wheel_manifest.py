# SPDX-License-Identifier: Apache-2.0
"""Exercise artifact handoff without torch, vLLM, Rust or a device."""

from __future__ import annotations

import hashlib
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


def make_wheels(
    directory: Path, *, rust: bool = True, parser_bridge: bool = True
) -> Path:
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
                if parser_bridge:
                    archive.writestr(
                        "vllm/_rust_tool_parser.abi3.so", b"test extension"
                    )
            elif name == "vllm_sail":
                for module in manifest.NATIVE_MODULES:
                    archive.writestr(f"vllm_sail/{module}.abi3.so", b"test extension")
    return next(directory.rglob("vllm-*.whl"))


def record(directory: Path) -> dict:
    data = {"schema": 1, "wheels": manifest.wheel_records(directory)}
    (directory / manifest.MANIFEST).write_text(json.dumps(data))
    return data


def make_donor(
    directory: Path, *, vllm_commit: str, sail_commit: str, fingerprint: str
) -> dict:
    directory.mkdir(parents=True)
    make_wheels(directory)
    data = record(directory)
    data["commits"] = {"vllm": vllm_commit, "vllm-sail": sail_commit}
    sail_wheel = directory / data["wheels"]["vllm-sail"]["file"]
    data["native_reuse"] = {
        "fingerprint": fingerprint,
        "extensions": manifest.native_extension_records(sail_wheel),
    }
    (directory / manifest.MANIFEST).write_text(json.dumps(data))
    return data


def make_native_source(directory: Path) -> Path:
    for relative in manifest.NATIVE_INPUTS:
        relative = {
            "cmake": "cmake/fixture.cmake",
            "csrc": "csrc/fixture.hg",
        }.get(relative, relative)
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"old {relative}\n".encode())
    (directory / manifest.FINGERPRINT_COMPAT).write_text("{}\n")
    return directory


def test_preserves_real_wheel_versions_and_verifies_hashes(tmp_path: Path) -> None:
    make_wheels(tmp_path)
    data = record(tmp_path)
    assert data["wheels"]["vllm"]["version"] == "0.30.0+empty"
    assert manifest.verify(tmp_path) == data


def test_rejects_missing_rust_even_when_wheel_exists(tmp_path: Path) -> None:
    make_wheels(tmp_path, rust=False)
    with pytest.raises(ValueError, match="required Rust artifacts"):
        manifest.wheel_records(tmp_path)


def test_accepts_upstream_frontend_without_removed_parser_bridge(
    tmp_path: Path,
) -> None:
    make_wheels(tmp_path, parser_bridge=False)
    data = record(tmp_path)
    assert manifest.verify(tmp_path) == data


@pytest.mark.parametrize("binary", [None, b""])
def test_parser_bridge_does_not_replace_required_frontend(
    tmp_path: Path, binary: bytes | None
) -> None:
    wheel = make_wheels(tmp_path, rust=False)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("vllm/_rust_tool_parser.abi3.so", b"test extension")
        if binary is not None:
            archive.writestr("vllm/vllm-rs", binary)
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


@pytest.mark.parametrize(
    "change",
    [
        None,
        *manifest.FINGERPRINT_COMPAT_PATHS,
        "csrc/fixture.hg",
        "settings",
        "unknown_fingerprint",
        "extra_path",
        "missing_path",
        "incorrect_previous",
        "malformed_map",
    ],
)
def test_native_reuse_bridges_only_exact_v2_inputs(tmp_path: Path, change) -> None:
    source = make_native_source(tmp_path / "source")
    settings = {"build_image": "image@sha256:one", "vllm_sail_hg_flags": "-O3"}
    # Produced independently by the unmodified 714079e fingerprint function.
    previous = "a5ec5996d87cec44e15e912a1b2fae9982ef009376632b54f0db9bd9e4ad15d4"
    versions = {}
    for relative in manifest.FINGERPRINT_COMPAT_PATHS:
        path = source / relative
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_text(f"new {relative}\n")
        versions[relative] = {
            "current": hashlib.sha256(path.read_bytes()).hexdigest(),
            "previous": before,
        }
    compat = source / manifest.FINGERPRINT_COMPAT
    unconfigured_fingerprint = manifest.native_fingerprint(source, **settings)
    if change == "extra_path":
        versions["csrc/fixture.hg"] = versions[manifest.FINGERPRINT_COMPAT_PATHS[0]]
    elif change == "missing_path":
        del versions[manifest.FINGERPRINT_COMPAT_PATHS[0]]
    elif change == "incorrect_previous":
        versions[manifest.FINGERPRINT_COMPAT_PATHS[0]]["previous"] = "0" * 64
    compat.write_text("{" if change == "malformed_map" else json.dumps(versions))
    current = manifest.native_fingerprint(source, **settings)
    assert current not in {previous, unconfigured_fingerprint}

    if change in (*manifest.FINGERPRINT_COMPAT_PATHS, "csrc/fixture.hg"):
        (source / change).write_text("changed again\n")
    elif change == "settings":
        settings["vllm_sail_hg_flags"] = "-O2"
    elif change == "unknown_fingerprint":
        previous = "0" * 64

    donor = tmp_path / "donor"
    output = tmp_path / "extensions"
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    make_donor(donor, vllm_commit="1" * 40, sail_commit="2" * 40, fingerprint=previous)

    def reuse() -> None:
        manifest.reuse_native(
            donor,
            output,
            wheels,
            vllm_commit="1" * 40,
            sail_commit="2" * 40,
            sail_source=source,
            **settings,
        )

    if change is None:
        reuse()
        assert len(list(output.glob("*.so"))) == len(manifest.NATIVE_MODULES)
        assert len(list(wheels.glob("vllm-*.whl"))) == 1
    else:
        with pytest.raises(ValueError, match="native build fingerprint does not match"):
            reuse()
        assert not output.exists()
        assert not list(wheels.iterdir())


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
    assert data["native_reuse"]["fingerprint"] == manifest.native_fingerprint(
        ROOT, build_image="image@sha256:one", pytorch_sail_arch="ppu_15;ppu_10"
    )


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


def test_create_packages_deterministic_recipes_archive_and_binds_tree_sha256(
    tmp_path: Path,
) -> None:
    import hashlib
    import shutil
    import tarfile

    make_wheels(tmp_path)
    commit = "b" * 40
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
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    archive_path = tmp_path / manifest.RECIPES_ARCHIVE
    assert archive_path.is_file()
    first_bytes = archive_path.read_bytes()
    second_archive = tmp_path / "second-recipes.tar.gz"
    second_sha = manifest.build_recipes_archive(ROOT, second_archive)
    assert second_archive.read_bytes() == first_bytes

    with tarfile.open(archive_path, "r:gz") as tar:
        members = {
            member.name: hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            for member in tar.getmembers()
            if member.isfile()
        }
    assert "tests/ppu/recipes/deepseek-ai/DeepSeek-V4-Flash.yaml" in members
    assert "tests/ppu/recipes/zai-org/GLM-5.3.yaml" in members
    assert "scripts/ci/ppu_recipe.py" in members
    expected_canonical = json.dumps(
        sorted([name, sha] for name, sha in members.items()),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_tree_sha = hashlib.sha256(expected_canonical).hexdigest()
    assert second_sha == expected_tree_sha

    verified = manifest.verify(tmp_path, vllm_commit=commit, require_recipes=True)
    assert verified["recipe_tree_sha256"] == expected_tree_sha

    # Tampered archive bytes or missing archive must fail closed.
    tampered_dir = tmp_path / "tampered"
    shutil.copytree(tmp_path, tampered_dir, ignore=shutil.ignore_patterns("tampered"))
    with tarfile.open(tampered_dir / manifest.RECIPES_ARCHIVE, "w:gz") as tar:
        payload = b"schema_version: ppu-serving/v1\n"
        info = tarfile.TarInfo("tests/ppu/recipes/tampered.yaml")
        info.size = len(payload)
        info.mode = 0o644
        import io

        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="differ from build manifest"):
        manifest.verify(tampered_dir, vllm_commit=commit)

    (tampered_dir / manifest.RECIPES_ARCHIVE).unlink()
    with pytest.raises(ValueError, match="recipes.tar.gz is missing"):
        manifest.verify(tampered_dir, vllm_commit=commit)

    # Legacy manifest without recipe_tree_sha256 passes default verify, fails --require-recipes,
    # and rejects an unrecorded recipes.tar.gz dropped beside it.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    make_wheels(legacy_dir)
    record(legacy_dir)
    assert manifest.verify(legacy_dir)["schema"] == 1
    with pytest.raises(ValueError, match="lacks recipe_tree_sha256"):
        manifest.verify(legacy_dir, require_recipes=True)
    shutil.copyfile(archive_path, legacy_dir / manifest.RECIPES_ARCHIVE)
    with pytest.raises(ValueError, match="invalid recipe_tree_sha256"):
        manifest.verify(legacy_dir)


@pytest.mark.parametrize(
    "corruption", ["truncated_gzip", "non_tar_gzip", "bad_deflate", "bad_crc"]
)
def test_reuse_native_and_verify_fail_closed_on_corrupted_recipes_archive(
    tmp_path: Path, corruption: str
) -> None:
    import gzip

    source = make_native_source(tmp_path / "source")
    fingerprint = manifest.native_fingerprint(source)
    vllm_commit = "1" * 40
    sail_commit = "2" * 40

    # Normal control: donor with valid recipes.tar.gz + recipe_tree_sha256 succeeds.
    valid_donor = tmp_path / "valid-donor"
    data = make_donor(
        valid_donor,
        vllm_commit=vllm_commit,
        sail_commit=sail_commit,
        fingerprint=fingerprint,
    )
    tree_sha = manifest.build_recipes_archive(
        ROOT, valid_donor / manifest.RECIPES_ARCHIVE
    )
    data["recipe_tree_sha256"] = tree_sha
    (valid_donor / manifest.MANIFEST).write_text(json.dumps(data))

    ext_ok = tmp_path / "ext-ok"
    wheels_ok = tmp_path / "wheels-ok"
    wheels_ok.mkdir()
    manifest.reuse_native(
        valid_donor,
        ext_ok,
        wheels_ok,
        vllm_commit=vllm_commit,
        sail_commit=sail_commit,
        sail_source=source,
    )
    assert len(list(ext_ok.glob("*.so"))) == len(manifest.NATIVE_MODULES)
    assert len(list(wheels_ok.glob("vllm-*.whl"))) == 1

    # Corrupt recipes.tar.gz in ways that previously raised EOFError / TarError / zlib.error.
    archive_path = valid_donor / manifest.RECIPES_ARCHIVE
    valid_bytes = archive_path.read_bytes()
    if corruption == "truncated_gzip":
        archive_path.write_bytes(valid_bytes[: len(valid_bytes) // 2])
    elif corruption == "non_tar_gzip":
        archive_path.write_bytes(gzip.compress(b"not a tar stream" * 64, mtime=0))
    elif corruption == "bad_crc":
        corrupted = bytearray(valid_bytes)
        corrupted[-6] ^= 0xFF
        archive_path.write_bytes(bytes(corrupted))
    else:
        corrupted = bytearray(valid_bytes)
        corrupted[len(corrupted) // 2] ^= 0xFF
        archive_path.write_bytes(bytes(corrupted))

    with pytest.raises(ValueError, match="corrupted recipes.tar.gz"):
        manifest.verify(valid_donor, vllm_commit=vllm_commit, sail_commit=sail_commit)

    ext_bad = tmp_path / "ext-bad"
    wheels_bad = tmp_path / "wheels-bad"
    wheels_bad.mkdir()
    cli = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "reuse-native",
            str(valid_donor),
            "--output",
            str(ext_bad),
            "--wheel-output",
            str(wheels_bad),
            "--sail-source",
            str(source),
            "--vllm-commit",
            vllm_commit,
            "--sail-commit",
            sail_commit,
        ],
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 1
    assert "reuse-native: corrupted recipes.tar.gz" in cli.stderr
    assert "Traceback" not in cli.stderr
    assert not ext_bad.exists()
    assert not list(wheels_bad.iterdir())


def test_recipe_archive_rejects_dir_under_file_double_slash_and_ancestor_symlink(
    tmp_path: Path,
) -> None:
    import io
    import tarfile

    # 1. Directory member under an existing regular file path must be rejected.
    dir_under_file = tmp_path / "dir-under-file.tar.gz"
    with tarfile.open(dir_under_file, "w:gz") as tar:
        payload = b"schema_version: ppu-serving/v1\n"
        file_info = tarfile.TarInfo("tests/ppu/foo")
        file_info.size = len(payload)
        file_info.mode = 0o644
        tar.addfile(file_info, io.BytesIO(payload))
        dir_info = tarfile.TarInfo("tests/ppu/foo/bar/")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)
    with pytest.raises(ValueError, match="collides with a file"):
        manifest.recipe_tree_sha256(dir_under_file)

    # 2. Directory member with double slash must be rejected.
    double_slash = tmp_path / "double-slash.tar.gz"
    with tarfile.open(double_slash, "w:gz") as tar:
        dir_info = tarfile.TarInfo("tests/ppu//sub/")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)
    with pytest.raises(ValueError, match="unsafe recipe archive path"):
        manifest.recipe_tree_sha256(double_slash)

    # 3. Symlinked intermediate ancestor directory inside source must be rejected.
    fake_src = tmp_path / "src"
    (fake_src / "real_tests/ppu/recipes/org").mkdir(parents=True)
    (fake_src / "real_tests/ppu/recipes/org/model.yaml").write_text("a: 1\n")
    (fake_src / "tests").symlink_to(fake_src / "real_tests")
    (fake_src / "scripts/ci").mkdir(parents=True)
    (fake_src / "scripts/ci/ppu_recipe.py").write_text("# recipe\n")
    with pytest.raises(ValueError, match="symlinked ancestor"):
        manifest.build_recipes_archive(fake_src, tmp_path / "out.tar.gz")
