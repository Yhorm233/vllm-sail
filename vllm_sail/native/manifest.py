# SPDX-License-Identifier: Apache-2.0
"""Reader for the PPU kernel corpus manifest (``csrc/upstream/manifest.toml``).

The manifest is the single declarative surface behind which the whole
CUDA-free kernel story sits: the port tool, the CMake source lists, the runtime
capability stubs, the merge-gate integrity tests and the docs table all read
this one file. See ``docs/developer_guide/kernels.md``.

Import-safe with no torch, no vLLM and no PPU SDK: it is stdlib only, so the
build backend and the merge gate can both use it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

__all__ = [
    "Entry",
    "Manifest",
    "ManifestError",
    "default_manifest_path",
    "load",
    "vendored_name",
]

SCHEMA = 1

# Translate csrc/a/b.cu to csrc/upstream/a/b.hg.
VENDOR_ROOT = "csrc/upstream"


def vendored_name(upstream_rel: str) -> str:
    """Map upstream paths to the vendor tree, replacing .cu with .hg."""
    name = upstream_rel.removeprefix("csrc/")
    return f"{name[:-3]}.hg" if name.endswith(".cu") else name


# Bindings TUs never carry device code but do carry op registrations, so they
# are compiled alongside the kernels of their own tier.
_BINDINGS_SUFFIX = "torch_bindings.cpp"

# Upstream splits MoE into its own extension and op namespace; so do we.
_MOE_PREFIX = "csrc/libtorch_stable/moe/"


class ManifestError(Exception):
    """The manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Entry:
    """One upstream translation unit."""

    path: str
    tier: str
    hazards: tuple[str, ...] = ()
    ops: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_bindings(self) -> bool:
        return self.path.endswith(_BINDINGS_SUFFIX)

    @property
    def is_moe(self) -> bool:
        """Whether this TU belongs to the MoE extension rather than the main one."""
        return self.path.startswith(_MOE_PREFIX)

    @property
    def vendored(self) -> str:
        """Where the translated copy of this TU lives in this repo."""
        return f"{VENDOR_ROOT}/{vendored_name(self.path)}"


@dataclass(frozen=True)
class Manifest:
    """The kernel corpus, as declared.

    Every accessor works off validated data: :func:`load` refuses to return a
    ``Manifest`` that violates an integrity rule, so consumers do not re-check.
    """

    upstream_ref: str
    tiers: dict[str, str]
    hazards: dict[str, str]
    compiled_tiers: tuple[str, ...]
    entries: tuple[Entry, ...]
    drop_includes: tuple[str, ...] = ()
    prune_bindings: tuple[str, ...] = ()
    rename_extension: dict[str, str] = field(default_factory=dict)
    replace_text: dict[str, dict[str, str]] = field(default_factory=dict)

    def of_tier(self, *tiers: str) -> tuple[Entry, ...]:
        """Entries in the given tiers, in manifest order."""
        for tier in tiers:
            if tier not in self.tiers:
                raise ManifestError(f"unknown tier {tier!r}")
        return tuple(e for e in self.entries if e.tier in tiers)

    def compiled(self) -> tuple[Entry, ...]:
        """Entries that the current configuration compiles."""
        return self.of_tier(*self.compiled_tiers)

    def excluded(self) -> tuple[Entry, ...]:
        """Entries that the current configuration does not compile."""
        compiled = set(self.compiled_tiers)
        return tuple(e for e in self.entries if e.tier not in compiled)

    def compiled_sources(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Vendored source paths for ``_upstream_C`` and ``_upstream_moe_C``."""
        compiled = self.compiled()
        return (
            tuple(e.vendored for e in compiled if not e.is_moe),
            tuple(e.vendored for e in compiled if e.is_moe),
        )

    def ops(self, entries: Iterable[Entry] | None = None) -> tuple[str, ...]:
        """Sorted op names registered by the given entries."""
        chosen = self.entries if entries is None else entries
        return tuple(sorted({op for e in chosen for op in e.ops}))

    def excluded_ops(self) -> tuple[str, ...]:
        """Ops that need a runtime stub because their TU is not compiled."""
        return self.ops(self.excluded())

    def hazards_of(self, entries: Iterable[Entry] | None = None) -> tuple[str, ...]:
        chosen = self.entries if entries is None else entries
        return tuple(sorted({h for e in chosen for h in e.hazards}))

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)


def default_manifest_path() -> Path:
    """The installed manifest copy, or its canonical source-tree location."""
    from vllm_sail.native import resources

    return resources.resolve(Path(__file__).resolve().parent, "manifest.toml")


def load(path: str | Path | None = None) -> Manifest:
    """Parse and validate the manifest.

    Raises :class:`ManifestError` for anything a consumer would otherwise have
    to defend against: unknown tiers or hazards, duplicate paths, an op
    registered by two entries, a CUTLASS-hazard file in a compiled tier, or a
    missing ``upstream_ref``.
    """
    src = Path(path) if path is not None else default_manifest_path()
    try:
        raw = tomllib.loads(src.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {src}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{src}: {exc}") from exc

    if raw.get("schema") != SCHEMA:
        raise ManifestError(
            f"{src}: schema {raw.get('schema')!r}, this reader speaks {SCHEMA}"
        )

    upstream_ref = str(raw.get("upstream_ref") or "").strip()
    if not upstream_ref:
        raise ManifestError(f"{src}: upstream_ref must be a non-empty vLLM ref")

    tiers = dict(raw.get("tiers") or {})
    hazards = dict(raw.get("hazards") or {})
    if not tiers:
        raise ManifestError(f"{src}: no tiers declared")

    compiled = tuple(raw.get("compiled_tiers") or ())
    if not compiled:
        raise ManifestError(
            f"{src}: compiled_tiers is empty or missing, so nothing would be "
            "compiled; note a bare key written after a [table] header belongs "
            "to that table"
        )
    for tier in compiled:
        if tier not in tiers:
            raise ManifestError(f"{src}: compiled_tiers names unknown tier {tier!r}")

    overlay = raw.get("overlay") or {}
    entries = tuple(
        _entry(src, item, tiers, hazards) for item in raw.get("entry") or ()
    )
    if not entries:
        raise ManifestError(f"{src}: no entries")

    _check_unique(src, entries)
    _check_compiled(src, entries, compiled)

    manifest = Manifest(
        upstream_ref=upstream_ref,
        tiers=tiers,
        hazards=hazards,
        compiled_tiers=compiled,
        entries=entries,
        drop_includes=tuple(overlay.get("drop_includes") or ()),
        prune_bindings=tuple(overlay.get("prune_bindings") or ()),
        rename_extension=dict(overlay.get("rename_extension") or {}),
        replace_text=dict(overlay.get("replace_text") or {}),
    )
    for name, replacements in manifest.replace_text.items():
        if not isinstance(replacements, dict) or not replacements:
            raise ManifestError(f"{src}: overlay.replace_text.{name} must be a table")
        for before, after in replacements.items():
            if not before or not isinstance(after, str):
                raise ManifestError(
                    f"{src}: overlay.replace_text.{name} needs non-empty keys "
                    "and string values"
                )
    for name in manifest.prune_bindings:
        if name not in {e.path for e in entries}:
            raise ManifestError(f"{src}: overlay.prune_bindings names unknown {name!r}")
    return manifest


def _entry(
    src: Path, item: dict, tiers: dict[str, str], hazards: dict[str, str]
) -> Entry:
    path = str(item.get("path") or "").strip()
    if not path:
        raise ManifestError(f"{src}: an entry has no path")
    tier = str(item.get("tier") or "")
    if tier not in tiers:
        raise ManifestError(f"{src}: {path}: unknown tier {tier!r}")
    haz = tuple(item.get("hazards") or ())
    for name in haz:
        if name not in hazards:
            raise ManifestError(f"{src}: {path}: unknown hazard {name!r}")
    return Entry(
        path=path,
        tier=tier,
        hazards=haz,
        ops=tuple(item.get("ops") or ()),
        notes=str(item.get("notes") or ""),
    )


def _check_unique(src: Path, entries: tuple[Entry, ...]) -> None:
    seen_paths: set[str] = set()
    owner: dict[str, str] = {}
    for entry in entries:
        if entry.path in seen_paths:
            raise ManifestError(f"{src}: duplicate entry {entry.path!r}")
        seen_paths.add(entry.path)
        for op in entry.ops:
            if op in owner:
                raise ManifestError(
                    f"{src}: op {op!r} is claimed by both {owner[op]} and {entry.path}"
                )
            owner[op] = entry.path


def _check_compiled(
    src: Path, entries: tuple[Entry, ...], compiled: tuple[str, ...]
) -> None:
    for entry in entries:
        if entry.tier in compiled and "cutlass" in entry.hazards:
            raise ManifestError(
                f"{src}: {entry.path} has a cutlass hazard but sits in compiled "
                f"tier {entry.tier!r}; sailify carries no CUTLASS mappings and "
                f"this build wires no CUTLASS"
            )
