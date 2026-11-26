# SPDX-License-Identifier: Apache-2.0
"""Record and verify the exact wheels passed from the build to PPU E2E."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

MANIFEST = "wheel-manifest.json"
NATIVE_MODULES = ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C", "_cumem_allocator")
FINGERPRINT_COMPAT = "scripts/ci/ppu_native_fingerprint_compat.json"
FINGERPRINT_COMPAT_PATHS = (
    ".github/workflows/_ppu-build.yaml",
    "scripts/ci/build_ppu_wheels.sh",
    "scripts/ci/ppu_wheel_manifest.py",
)
NATIVE_INPUTS = (
    ".github/vllm-main-verified.commit",
    ".github/workflows/_ppu-build.yaml",
    "CMakeLists.txt",
    "cmake",
    "csrc",
    "pyproject.toml",
    "requirements/build.txt",
    "scripts/ci/build_ppu_wheels.sh",
    "scripts/ci/ppu_image_stack.py",
    FINGERPRINT_COMPAT,
    "scripts/ci/ppu_wheel_manifest.py",
    "setup.py",
    "vllm_sail/native/manifest.py",
    "vllm_sail/native/plugin_manifest.py",
    "vllm_sail/native/resources.py",
    "vllm_sail/native/toolchain.py",
)


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _native_members(archive: zipfile.ZipFile) -> dict[str, str]:
    members = archive.namelist()
    found = {}
    for module in NATIVE_MODULES:
        matches = [
            name
            for name in members
            if name.startswith(f"vllm_sail/{module}.") and name.endswith(".so")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"SAIL wheel must contain exactly one {module}, found {len(matches)}"
            )
        member = matches[0]
        if archive.getinfo(member).file_size == 0:
            raise ValueError(f"SAIL wheel contains empty native extension: {module}")
        found[module] = member
    return found


def native_extension_records(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        records = {}
        for module, member in _native_members(archive).items():
            with archive.open(member) as stream:
                digest = _sha256_stream(stream)
            records[module] = {"file": Path(member).name, "sha256": digest}
        return records


def wheel_records(directory: Path) -> dict:
    records = {}
    for name, pattern in (("vllm", "vllm-*.whl"), ("vllm-sail", "vllm_sail-*.whl")):
        wheels = list(directory.rglob(pattern))
        if len(wheels) != 1:
            raise ValueError(f"expected exactly one {name} wheel, found {len(wheels)}")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            metadata_paths = [p for p in members if p.endswith(".dist-info/METADATA")]
            if len(metadata_paths) != 1:
                raise ValueError(f"invalid wheel metadata: {wheel.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            if (
                metadata.get("Name", "").replace("_", "-") != name
                or not metadata["Version"]
            ):
                raise ValueError(f"unexpected package metadata: {wheel.name}")
            if name == "vllm":
                # Current upstream builds the frontend executable without the
                # removed PyO3 tool-parser bridge. Older wheels may retain it.
                if (
                    "vllm/vllm-rs" not in members
                    or archive.getinfo("vllm/vllm-rs").file_size == 0
                ):
                    raise ValueError("vLLM wheel is missing required Rust artifacts")
            else:
                _native_members(archive)
        records[name] = {
            "file": wheel.relative_to(directory).as_posix(),
            "version": metadata["Version"],
            "sha256": _sha256_file(wheel),
        }
    return records


def _native_files(source: Path):
    for relative in NATIVE_INPUTS:
        path = source / relative
        if not path.exists():
            raise ValueError(f"native build input is missing: {relative}")
        paths = path.rglob("*") if path.is_dir() else (path,)
        for candidate in paths:
            if (
                candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix != ".pyc"
            ):
                yield candidate.relative_to(source).as_posix(), candidate


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate fingerprint compatibility key")
    return result


def _legacy_native_digests(
    compat: bytes, current: dict[str, str]
) -> dict[str, str] | None:
    try:
        versions = json.loads(compat, object_pairs_hook=_unique_json_object)
    except ValueError:
        return None
    if not isinstance(versions, dict) or versions.keys() != set(
        FINGERPRINT_COMPAT_PATHS
    ):
        return None
    for path, pair in versions.items():
        if (
            not isinstance(pair, dict)
            or pair.keys() != {"current", "previous"}
            or pair["current"] != current[path]
            or not _is_sha256(pair["previous"])
        ):
            return None
    # Reconstruct the v2 inputs from before the compatibility table was added.
    return {
        path: versions[path]["previous"] if path in versions else digest
        for path, digest in current.items()
        if path != FINGERPRINT_COMPAT
    }


def _native_fingerprints(
    source: Path,
    *,
    build_image: str = "",
    pytorch_sail_arch: str = "",
    cmake_build_type: str = "Release",
    cmake_hgcc_dir: str = "",
    vllm_sail_hg_flags: str = "",
    vllm_sail_hg_std: str = "20",
    include_legacy: bool = False,
) -> tuple[str, str | None]:
    digest = hashlib.sha256()
    settings = {
        "build_image": build_image,
        "cmake_build_type": cmake_build_type,
        "cmake_hgcc_dir": cmake_hgcc_dir,
        "pytorch_sail_arch": pytorch_sail_arch,
        "vllm_sail_hg_flags": vllm_sail_hg_flags,
        "vllm_sail_hg_std": vllm_sail_hg_std,
    }
    digest.update(b"vllm-sail-native-v2\0")
    for name, value in sorted(settings.items()):
        digest.update(name.encode())
        digest.update(b"=")
        digest.update(value.encode())
        digest.update(b"\0")

    def fingerprint(file_digests: dict[str, str]) -> str:
        result = digest.copy()
        for relative, file_digest in sorted(file_digests.items()):
            result.update(relative.encode())
            result.update(b"\0")
            result.update(bytes.fromhex(file_digest))
        return result.hexdigest()

    file_digests = {
        relative: _sha256_file(path)
        for relative, path in _native_files(source)
        if relative != FINGERPRINT_COMPAT
    }
    compat = (source / FINGERPRINT_COMPAT).read_bytes()
    file_digests[FINGERPRINT_COMPAT] = hashlib.sha256(compat).hexdigest()
    legacy_digests = (
        _legacy_native_digests(compat, file_digests) if include_legacy else None
    )
    return fingerprint(file_digests), (
        fingerprint(legacy_digests) if legacy_digests is not None else None
    )


def native_fingerprint(source: Path, **settings: str) -> str:
    """Hash every input and build setting that can change native extensions."""
    return _native_fingerprints(source, **settings)[0]


def verify(
    directory: Path,
    *,
    vllm_commit: str | None = None,
    sail_commit: str | None = None,
) -> dict:
    manifest = json.loads((directory / MANIFEST).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("unsupported wheel manifest schema")
    if manifest.get("wheels") != wheel_records(directory):
        raise ValueError(
            "wheel filenames, versions or hashes differ from build manifest"
        )
    commits = manifest.get("commits")
    for name, expected in (("vllm", vllm_commit), ("vllm-sail", sail_commit)):
        if expected is not None and (
            not isinstance(commits, dict) or commits.get(name) != expected
        ):
            raise ValueError(
                f"{name} source commit differs from required revision {expected}"
            )
    if "native_reuse" in manifest:
        native = manifest["native_reuse"]
        if not isinstance(native, dict):
            raise ValueError("native reuse metadata must be an object")
        wheel = directory / manifest["wheels"]["vllm-sail"]["file"]
        if native.get("extensions") != native_extension_records(wheel):
            raise ValueError("native extension hashes differ from build manifest")
        if not _is_sha256(native.get("fingerprint")):
            raise ValueError("invalid native build fingerprint")
    return manifest


def _require_native(manifest: dict) -> dict:
    native = manifest.get("native_reuse")
    if not native:
        raise ValueError("wheel artifact predates native reuse metadata")
    return native


def _extract_native(wheel: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        members = _native_members(archive)
        for module, member in members.items():
            target = output / Path(member).name
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if target.stat().st_size == 0:
                raise ValueError(f"empty native extension: {module}")


def extract_native(directory: Path, output: Path) -> dict:
    manifest = verify(directory)
    native = _require_native(manifest)
    _extract_native(directory / manifest["wheels"]["vllm-sail"]["file"], output)
    return native


def reuse_native(
    directory: Path,
    output: Path,
    wheel_output: Path,
    *,
    vllm_commit: str,
    sail_commit: str | None,
    sail_source: Path,
    **settings: str,
) -> None:
    manifest = verify(directory, vllm_commit=vllm_commit, sail_commit=sail_commit)
    native = _require_native(manifest)
    if native["fingerprint"] not in _native_fingerprints(
        sail_source, include_legacy=True, **settings
    ):
        raise ValueError("native build fingerprint does not match")

    _extract_native(directory / manifest["wheels"]["vllm-sail"]["file"], output)
    wheel = directory / manifest["wheels"]["vllm"]["file"]
    # Stage on the destination filesystem so a failed copy cannot leave a wheel
    # for the next donor or the cold build to pick up.
    with tempfile.TemporaryDirectory(dir=wheel_output) as staging:
        staged_wheel = Path(staging) / wheel.name
        shutil.copyfile(wheel, staged_wheel)
        staged_wheel.replace(wheel_output / wheel.name)


def _git_commit(source: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={source.resolve()}",
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("create", "verify", "fingerprint", "extract-native", "reuse-native"),
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--sail-source", type=Path)
    parser.add_argument("--vllm-commit", help="require this upstream source commit")
    parser.add_argument("--sail-commit", help="require this plugin source commit")
    parser.add_argument("--build-image", default="")
    parser.add_argument("--pytorch-sail-arch", default="")
    parser.add_argument("--cmake-build-type", default="Release")
    parser.add_argument("--cmake-hgcc-dir", default="")
    parser.add_argument("--vllm-sail-hg-flags", default="")
    parser.add_argument("--vllm-sail-hg-std", default="20")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wheel-output", type=Path)
    args = parser.parse_args()
    settings = {
        "build_image": args.build_image,
        "pytorch_sail_arch": args.pytorch_sail_arch,
        "cmake_build_type": args.cmake_build_type,
        "cmake_hgcc_dir": args.cmake_hgcc_dir,
        "vllm_sail_hg_flags": args.vllm_sail_hg_flags,
        "vllm_sail_hg_std": args.vllm_sail_hg_std,
    }
    if args.command == "verify":
        result = verify(
            args.directory, vllm_commit=args.vllm_commit, sail_commit=args.sail_commit
        )
    elif args.command == "fingerprint":
        result = native_fingerprint(args.directory, **settings)
    elif args.command == "extract-native":
        if args.output is None:
            parser.error("extract-native requires --output")
        result = extract_native(args.directory, args.output)
    elif args.command == "reuse-native":
        if (
            args.output is None
            or args.wheel_output is None
            or args.sail_source is None
            or args.vllm_commit is None
        ):
            parser.error(
                "reuse-native requires --output, --wheel-output, "
                "--sail-source and --vllm-commit"
            )
        try:
            reuse_native(
                args.directory,
                args.output,
                args.wheel_output,
                vllm_commit=args.vllm_commit,
                sail_commit=args.sail_commit,
                sail_source=args.sail_source,
                **settings,
            )
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            parser.exit(1, f"reuse-native: {error}\n")
        return
    else:
        if (
            args.vllm_source is None and args.vllm_commit is None
        ) or args.sail_source is None:
            parser.error(
                "create requires --sail-source and either --vllm-source or --vllm-commit"
            )
        wheels = wheel_records(args.directory)
        sail_wheel = args.directory / wheels["vllm-sail"]["file"]
        result = {
            "schema": 1,
            "commits": {
                "vllm": (
                    args.vllm_commit
                    if args.vllm_commit is not None
                    else _git_commit(args.vllm_source)
                ),
                "vllm-sail": _git_commit(args.sail_source),
            },
            "wheels": wheels,
            "native_reuse": {
                "fingerprint": native_fingerprint(args.sail_source, **settings),
                "extensions": native_extension_records(sail_wheel),
            },
        }
        (args.directory / MANIFEST).write_text(json.dumps(result, indent=2) + "\n")
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
