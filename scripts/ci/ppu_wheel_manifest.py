# SPDX-License-Identifier: Apache-2.0
"""Record and verify the exact wheels passed from the build to PPU E2E."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
import zlib
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

MANIFEST = "wheel-manifest.json"
RECIPES_ARCHIVE = "recipes.tar.gz"
RECIPE_BUNDLE_INPUTS = (
    "tests/ppu",
    "scripts/ci/ppu_recipe.py",
)
MAX_RECIPE_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_RECIPE_ARCHIVE_MEMBERS = 4096
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _safe_archive_relpath(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 or ord(c) == 127 for c in name)
    ):
        raise ValueError(f"invalid recipe archive path: {name!r}")
    parts = name.split("/")
    if name.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe recipe archive path: {name!r}")
    return PurePosixPath(name).as_posix()


def _recipe_bundle_files(source: Path) -> list[tuple[str, Path]]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("sail source must be a non-symlink directory")
    resolved_root = source.resolve()
    collected: dict[str, Path] = {}
    for relative in RECIPE_BUNDLE_INPUTS:
        root = source / relative
        if root.is_symlink() or not root.exists():
            raise ValueError(f"recipe bundle input is missing or symlinked: {relative}")
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for current, dirs, files in os.walk(root, followlinks=False):
                current_path = Path(current)
                for dirname in list(dirs):
                    if (current_path / dirname).is_symlink():
                        raise ValueError(
                            f"symlink directory in recipe bundle input: {current_path / dirname}"
                        )
                if "__pycache__" in dirs:
                    dirs.remove("__pycache__")
                dirs.sort()
                for filename in sorted(files):
                    candidate = current_path / filename
                    if candidate.is_symlink():
                        raise ValueError(
                            f"symlink file in recipe bundle input: {candidate}"
                        )
                    if candidate.suffix != ".pyc" and candidate.is_file():
                        candidates.append(candidate)
        for candidate in candidates:
            rel_unres = candidate.relative_to(source).as_posix()
            rel_res = candidate.resolve().relative_to(resolved_root).as_posix()
            if rel_unres != rel_res:
                raise ValueError(
                    f"symlinked ancestor in recipe bundle input: {rel_unres}"
                )
            rel_posix = _safe_archive_relpath(rel_res)
            collected[rel_posix] = candidate
    if not any(path.startswith("tests/ppu/recipes/") for path in collected):
        raise ValueError("recipe bundle must contain tests/ppu/recipes entries")
    if "scripts/ci/ppu_recipe.py" not in collected:
        raise ValueError("recipe bundle must contain scripts/ci/ppu_recipe.py")
    return sorted(collected.items())


class _BoundedArchiveStream:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.total = 0

    def read(self, size: int = -1) -> bytes:
        limit = 2 * MAX_RECIPE_ARCHIVE_BYTES - self.total + 1
        to_read = limit if size is None or size < 0 else min(size, limit)
        chunk = self.stream.read(to_read)
        self.total += len(chunk)
        if self.total > 2 * MAX_RECIPE_ARCHIVE_BYTES:
            raise ValueError("recipe archive stream size limit exceeded")
        return chunk


def recipe_tree_sha256(archive_path: Path) -> str:
    """Compute canonical SHA256 over sorted [relative_path, sha256] file pairs."""
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("recipes.tar.gz must be a regular file")
    if archive_path.stat().st_size > MAX_RECIPE_ARCHIVE_BYTES:
        raise ValueError("recipes.tar.gz exceeds size limit")
    pairs: list[list[str]] = []
    seen: set[str] = set()
    file_paths: set[str] = set()
    total = 0
    try:
        with gzip.open(archive_path, "rb") as compressed:
            bounded = _BoundedArchiveStream(compressed)
            with tarfile.open(fileobj=bounded, mode="r|") as archive:
                for member in archive:
                    raw_name = (
                        member.name[:-1]
                        if member.isdir() and member.name.endswith("/")
                        else member.name
                    )
                    name = _safe_archive_relpath(raw_name)
                    if name in seen or len(seen) >= MAX_RECIPE_ARCHIVE_MEMBERS:
                        raise ValueError(
                            "duplicate or excessive recipe archive members"
                        )
                    seen.add(name)
                    if member.mode & 0o7000 or not (member.isfile() or member.isdir()):
                        raise ValueError(
                            "unsafe recipe archive member type or permissions"
                        )
                    if (
                        member.size < 0
                        or member.size > MAX_RECIPE_ARCHIVE_BYTES
                        or total + member.size > MAX_RECIPE_ARCHIVE_BYTES
                    ):
                        raise ValueError("recipe archive expanded size limit exceeded")
                    for parent in PurePosixPath(name).parents:
                        if parent.as_posix() in file_paths:
                            raise ValueError("recipe archive path collides with a file")
                    if member.isdir():
                        if member.size:
                            raise ValueError(
                                "recipe archive directory member contains data"
                            )
                        continue
                    if any(
                        previous.startswith(name + "/")
                        for previous in seen
                        if previous != name
                    ):
                        raise ValueError(
                            "recipe archive file collides with a directory"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"cannot read recipe archive member: {name}")
                    digest = hashlib.sha256()
                    actual = 0
                    with extracted:
                        while chunk := extracted.read(65536):
                            actual += len(chunk)
                            total += len(chunk)
                            if (
                                actual > MAX_RECIPE_ARCHIVE_BYTES
                                or total > MAX_RECIPE_ARCHIVE_BYTES
                            ):
                                raise ValueError(
                                    "recipe archive expanded size limit exceeded"
                                )
                            digest.update(chunk)
                    if actual != member.size:
                        raise ValueError("truncated recipe archive member")
                    file_paths.add(name)
                    pairs.append([name, digest.hexdigest()])
            while bounded.read(65536):
                pass
    except (gzip.BadGzipFile, EOFError, tarfile.TarError, zlib.error) as error:
        raise ValueError("corrupted recipes.tar.gz") from error
    if not pairs:
        raise ValueError("empty recipe archive")
    return hashlib.sha256(_canonical_json(sorted(pairs))).hexdigest()


def build_recipes_archive(source: Path, destination: Path) -> str:
    """Write a deterministic recipes.tar.gz and return its canonical recipe_tree_sha256."""
    if destination.is_symlink():
        raise ValueError("recipes.tar.gz destination cannot be a symlink")
    entries = _recipe_bundle_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pairs: list[list[str]] = []
    total = 0
    fd, temp_name = tempfile.mkstemp(
        prefix=".recipes-", suffix=".tar.gz", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as gz_handle:
                with tarfile.open(
                    fileobj=gz_handle, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    for rel_posix, file_path in entries:
                        data = file_path.read_bytes()
                        total += len(data)
                        if total > MAX_RECIPE_ARCHIVE_BYTES:
                            raise ValueError(
                                "recipe bundle inputs exceed archive size limit"
                            )
                        info = tarfile.TarInfo(name=rel_posix)
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.type = tarfile.REGTYPE
                        archive.addfile(info, io.BytesIO(data))
                        pairs.append([rel_posix, hashlib.sha256(data).hexdigest()])
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.chmod(temp_path, 0o644)
        expected = hashlib.sha256(_canonical_json(sorted(pairs))).hexdigest()
        if recipe_tree_sha256(temp_path) != expected:
            raise ValueError("written recipes.tar.gz failed self-verification")
        os.replace(temp_path, destination)
        return expected
    finally:
        temp_path.unlink(missing_ok=True)


def verify(
    directory: Path,
    *,
    vllm_commit: str | None = None,
    sail_commit: str | None = None,
    require_recipes: bool = False,
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
    recipes_archive = directory / RECIPES_ARCHIVE
    has_recipes_archive = recipes_archive.exists() or recipes_archive.is_symlink()
    if require_recipes and "recipe_tree_sha256" not in manifest:
        raise ValueError("wheel manifest lacks recipe_tree_sha256")
    if "recipe_tree_sha256" in manifest or has_recipes_archive:
        expected_tree = manifest.get("recipe_tree_sha256")
        if not _is_sha256(expected_tree):
            raise ValueError("invalid recipe_tree_sha256 in wheel manifest")
        if not has_recipes_archive:
            raise ValueError("recipes.tar.gz is missing from wheel bundle")
        if recipe_tree_sha256(recipes_archive) != expected_tree:
            raise ValueError("recipes.tar.gz contents differ from build manifest")
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
    parser.add_argument(
        "--require-recipes",
        action="store_true",
        help="require recipes.tar.gz and recipe_tree_sha256 in wheel-manifest.json",
    )
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
            args.directory,
            vllm_commit=args.vllm_commit,
            sail_commit=args.sail_commit,
            require_recipes=args.require_recipes,
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
        tree_sha256 = build_recipes_archive(
            args.sail_source, args.directory / RECIPES_ARCHIVE
        )
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
            "recipe_tree_sha256": tree_sha256,
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
