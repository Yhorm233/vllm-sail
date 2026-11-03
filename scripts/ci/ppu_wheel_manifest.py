# SPDX-License-Identifier: Apache-2.0
"""Record and verify the exact wheels passed from the build to PPU E2E."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from email.parser import BytesParser
from pathlib import Path

MANIFEST = "wheel-manifest.json"
NATIVE_MODULES = ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C")
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
            if metadata["Name"].replace("_", "-") != name or not metadata["Version"]:
                raise ValueError(f"unexpected package metadata: {wheel.name}")
            if name == "vllm":
                if "vllm/vllm-rs" not in members or not any(
                    p.startswith("vllm/_rust_tool_parser.") and p.endswith(".so")
                    for p in members
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


def native_fingerprint(
    source: Path,
    *,
    build_image: str = "",
    pytorch_sail_arch: str = "",
    cmake_build_type: str = "Release",
    cmake_hgcc_dir: str = "",
    vllm_sail_hg_flags: str = "",
    vllm_sail_hg_std: str = "20",
) -> str:
    """Hash every input and build setting that can change native extensions."""
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
    for relative, path in sorted(_native_files(source)):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def verify(
    directory: Path,
    *,
    vllm_commit: str | None = None,
    sail_commit: str | None = None,
) -> dict:
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported wheel manifest schema")
    if manifest["wheels"] != wheel_records(directory):
        raise ValueError(
            "wheel filenames, versions or hashes differ from build manifest"
        )
    for name, expected in (("vllm", vllm_commit), ("vllm-sail", sail_commit)):
        if expected is not None and manifest.get("commits", {}).get(name) != expected:
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
        fingerprint = native.get("fingerprint", "")
        if len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            raise ValueError("invalid native build fingerprint")
    return manifest


def extract_native(directory: Path, output: Path) -> dict:
    manifest = verify(directory)
    native = manifest.get("native_reuse")
    if not native:
        raise ValueError("wheel artifact predates native reuse metadata")
    wheel = directory / manifest["wheels"]["vllm-sail"]["file"]
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        members = _native_members(archive)
        for module, member in members.items():
            target = output / Path(member).name
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if target.stat().st_size == 0:
                raise ValueError(f"empty native extension: {module}")
    return native


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
        "command", choices=("create", "verify", "fingerprint", "extract-native")
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
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(
            args.directory, vllm_commit=args.vllm_commit, sail_commit=args.sail_commit
        )
    elif args.command == "fingerprint":
        result = native_fingerprint(
            args.directory,
            build_image=args.build_image,
            pytorch_sail_arch=args.pytorch_sail_arch,
            cmake_build_type=args.cmake_build_type,
            cmake_hgcc_dir=args.cmake_hgcc_dir,
            vllm_sail_hg_flags=args.vllm_sail_hg_flags,
            vllm_sail_hg_std=args.vllm_sail_hg_std,
        )
    elif args.command == "extract-native":
        if args.output is None:
            parser.error("extract-native requires --output")
        result = extract_native(args.directory, args.output)
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
                "fingerprint": native_fingerprint(
                    args.sail_source,
                    build_image=args.build_image,
                    pytorch_sail_arch=args.pytorch_sail_arch,
                    cmake_build_type=args.cmake_build_type,
                    cmake_hgcc_dir=args.cmake_hgcc_dir,
                    vllm_sail_hg_flags=args.vllm_sail_hg_flags,
                    vllm_sail_hg_std=args.vllm_sail_hg_std,
                ),
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
