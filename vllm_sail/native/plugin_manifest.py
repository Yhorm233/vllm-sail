# SPDX-License-Identifier: Apache-2.0
"""Read the native HGGC sources and host bindings owned by the plugin."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

__all__ = [
    "Extension",
    "PluginManifestError",
    "cmake_variable",
    "default_path",
    "load",
]

SCHEMA = 2
_PLUGIN_ROOT = "csrc/plugin/"
_CMAKE_VARIABLES = {
    "_C": "PPU_PLUGIN_EXT_SRC",
    "_moe_C": "PPU_PLUGIN_MOE_EXT_SRC",
    "_cumem_allocator": "PPU_PLUGIN_CUMEM_EXT_SRC",
}


class PluginManifestError(Exception):
    """The plugin-kernel manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class Extension:
    name: str
    device_sources: tuple[str, ...]
    bindings: tuple[str, ...]

    def cmake_sources(self) -> tuple[str, ...]:
        return self.device_sources + self.bindings


def default_path() -> Path:
    return Path(__file__).resolve().parents[2] / "csrc" / "plugin" / "manifest.toml"


def cmake_variable(extension: str) -> str:
    try:
        return _CMAKE_VARIABLES[extension]
    except KeyError as exc:
        raise PluginManifestError(
            f"unsupported plugin extension {extension!r}"
        ) from exc


def load(path: str | Path | None = None) -> tuple[Extension, ...]:
    src = Path(path) if path is not None else default_path()
    try:
        raw = tomllib.loads(src.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PluginManifestError(f"plugin manifest not found: {src}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PluginManifestError(f"{src}: {exc}") from exc

    if raw.get("schema") != SCHEMA:
        raise PluginManifestError(
            f"{src}: schema {raw.get('schema')!r}, this reader speaks {SCHEMA}"
        )
    extensions = tuple(_extension(src, item) for item in raw.get("extension") or ())
    names = [extension.name for extension in extensions]
    if len(names) != len(_CMAKE_VARIABLES) or set(names) != set(_CMAKE_VARIABLES):
        raise PluginManifestError(
            f"{src}: extensions must be exactly {', '.join(_CMAKE_VARIABLES)}"
        )
    paths = [
        path
        for extension in extensions
        for path in (*extension.device_sources, *extension.bindings)
    ]
    if len(paths) != len(set(paths)):
        raise PluginManifestError(f"{src}: a source path is listed more than once")
    return extensions


def _extension(src: Path, item: dict) -> Extension:
    name = str(item.get("name") or "").strip()
    cmake_variable(name)
    device_sources = tuple(item.get("device_sources") or ())
    bindings = tuple(item.get("bindings") or ())
    if not bindings or (name != "_cumem_allocator" and not device_sources):
        raise PluginManifestError(f"{src}: {name} needs device_sources and bindings")
    if name == "_cumem_allocator" and device_sources:
        raise PluginManifestError(f"{src}: {name} only accepts host bindings")
    for path in device_sources:
        if not str(path).startswith(_PLUGIN_ROOT) or not str(path).endswith(".hg"):
            raise PluginManifestError(f"{src}: invalid device source {path!r}")
    for path in bindings:
        if not str(path).startswith(_PLUGIN_ROOT) or not str(path).endswith(".cpp"):
            raise PluginManifestError(f"{src}: invalid binding source {path!r}")
    return Extension(
        name=name,
        device_sources=tuple(map(str, device_sources)),
        bindings=tuple(map(str, bindings)),
    )
