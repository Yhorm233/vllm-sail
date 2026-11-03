# SPDX-License-Identifier: Apache-2.0
"""Port upstream vLLM CUDA translation units into the vendored HGGC corpus.

The whole pipeline sits behind :func:`port`: select the manifest's compiled
entries, follow their local ``#include`` closure, copy the result into
``csrc/upstream/``, hand the copies to a translator, apply the manifest overlay,
and emit ``csrc/upstream/excluded_ops.toml`` for :mod:`vllm_sail.native.stubs`.

Two properties make this worth having as a module rather than a script:

* **Re-runnable.** The dest tree is rebuilt from scratch every time, so a tier
  demotion removes files rather than leaving them behind, and ``--check`` can
  detect drift by porting into a temp dir and diffing.
* **Translator-agnostic.** ``sailify`` is injected, so the merge gate exercises
  the copy/closure/overlay/pruning logic with a recording translator on a machine
  that has neither sailify nor a PPU SDK.

Stdlib only, and deliberately torch-free: ``tests/ut`` imports it directly.

Design: ``docs/developer_guide/kernels.md``.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vllm_sail.native import manifest as _manifest
from vllm_sail.native.manifest import Entry, Manifest
from vllm_sail.native.stubs import ExcludedOp

__all__ = [
    "EXCLUDED_OPS_NAME",
    "PRESERVED",
    "PortError",
    "PortResult",
    "TranslatedCorpus",
    "Translator",
    "port",
    "recording_translator",
    "render_excluded_ops",
    "report",
    "sailify_translator",
    "translate_corpus",
]

# The generated companion to manifest.toml, read at runtime by stubs.install().
EXCLUDED_OPS_NAME = "excluded_ops.toml"

# Upstream paths are relative to csrc/; vendored TUs use .hg.
_CSRC = "csrc/"

# Files in the dest tree that port() must not delete when it rebuilds:
# manifest.toml is the hand-written input that happens to live in its own output
# directory.
PRESERVED = frozenset({"manifest.toml"})

# Not a manifest tier: what an excluded op records when upstream registers it and
# the manifest never mentions it, so the stub message says so out loud.
_UNCLAIMED_TIER = "unclaimed"

#: Rewrites a copied corpus in place. Receives the dest root (so the translator
#: can resolve includes within the translated tree) and the files it may touch.
Translator = Callable[[Path, Sequence[Path]], None]

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
_BLOCK_RE = re.compile(
    r"^\s*STABLE_TORCH_LIBRARY(?:_FRAGMENT|_IMPL)?\s*\(\s*([A-Za-z_]\w*)\s*,"
)
_CALL_RE = re.compile(r"\b\w+\s*\.\s*(def|impl)\s*\(")
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_OPNAME_RE = re.compile(r"([A-Za-z_][\w.]*)")
_REGISTER_RE = re.compile(r"(REGISTER_EXTENSION\s*\(\s*)([A-Za-z_]\w*)(\s*\))")

# sailify 1.0.0 translates FP8 types but misses these SDK conversion functions.
# Keep this bounded supplement at the upstream generation boundary. Remove it
# when the supported sailify version provides these mappings itself.
_FP8_INTRINSICS = {
    f"__nv_cvt_{suffix}": f"__hg_cvt_{suffix}"
    for suffix in (
        "float_to_fp8",
        "float2_to_fp8x2",
        "fp8_to_halfraw",
        "fp8x2_to_halfraw2",
        "halfraw_to_fp8",
        "halfraw2_to_fp8x2",
    )
}
_FP8_TOKEN_RE = re.compile(
    r'R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"'
    r'|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r"|\b(?P<intrinsic>" + "|".join(_FP8_INTRINSICS) + r")\b",
    re.DOTALL,
)


class PortError(RuntimeError):
    """The corpus cannot be ported as the manifest describes it."""


@dataclass(frozen=True)
class PortResult:
    """What one :func:`port` run produced, in vendored-relative posix paths."""

    sources: tuple[str, ...]
    headers: tuple[str, ...]
    excluded_ops: tuple[ExcludedOp, ...]
    unclaimed_ops: tuple[str, ...]
    unresolved_includes: tuple[str, ...]
    pruned: Mapping[str, int]

    @property
    def files(self) -> tuple[str, ...]:
        return self.sources + self.headers


@dataclass(frozen=True)
class TranslatedCorpus:
    """One copied and translated source closure, relative to its destination."""

    sources: tuple[str, ...]
    headers: tuple[str, ...]
    unresolved_includes: tuple[str, ...]

    @property
    def files(self) -> tuple[str, ...]:
        return self.sources + self.headers


def sailify_translator(dest_root: Path, files: Sequence[Path]) -> None:
    """Translate CUDA to HGGC with ``t-head/sailify``, in place."""
    try:
        from sailify.sailify_python import sailify_extra_files_recursive
    except ImportError as exc:  # pragma: no cover - needs the real tool
        raise PortError(
            "porting a kernel corpus requires the sailify translator; clone "
            "https://github.com/t-head/sailify and put it on PYTHONPATH (or "
            "install it) before running this tool"
        ) from exc
    sailify_extra_files_recursive(
        str(dest_root),
        [str(path) for path in files],
        header_include_dirs=[str(dest_root)],
    )
    for path in files:
        source = path.read_text(encoding="utf-8")
        translated = _FP8_TOKEN_RE.sub(
            lambda match: (
                _FP8_INTRINSICS[match["intrinsic"]] if match["intrinsic"] else match[0]
            ),
            source,
        )
        if translated != source:
            path.write_text(translated, encoding="utf-8")


def recording_translator(seen: list[Path]) -> Translator:
    """A translator that changes nothing and records what it was offered."""

    def translate(dest_root: Path, files: Sequence[Path]) -> None:
        seen.extend(files)

    return translate


def port(
    src_root: str | Path,
    dest_root: str | Path,
    manifest: Manifest | None = None,
    translate: Translator | None = None,
) -> PortResult:
    """Rebuild the vendored corpus at ``dest_root`` from upstream ``src_root``.

    ``src_root`` is a vLLM source checkout (the directory holding ``csrc/``).
    Everything under ``dest_root`` except ``manifest.toml`` is replaced.
    """
    manifest = _manifest.load() if manifest is None else manifest
    translate = sailify_translator if translate is None else translate
    entries = manifest.compiled()
    corpus = translate_corpus(
        src_root,
        dest_root,
        tuple(entry.path for entry in entries),
        translate=translate,
        preserve=PRESERVED,
    )
    dest = Path(dest_root).resolve()
    # Rename TUs after sailify, which requires upstream names.
    sources = tuple(_manifest.vendored_name(_CSRC + rel) for rel in corpus.sources)
    for upstream, vendored in zip(corpus.sources, sources, strict=False):
        if upstream != vendored:
            (dest / upstream).rename(dest / vendored)
    copied = tuple(dest / path for path in (*sources, *corpus.headers))

    _drop_includes(copied, manifest.drop_includes)
    _replace_text(copied, dest, manifest.replace_text)
    pruned, excluded, unclaimed = _apply_bindings(dest, manifest)

    (dest / EXCLUDED_OPS_NAME).write_text(
        render_excluded_ops(manifest, excluded), encoding="utf-8"
    )
    return PortResult(
        sources=tuple(f"{_manifest.VENDOR_ROOT}/{rel}" for rel in sources),
        headers=tuple(f"{_manifest.VENDOR_ROOT}/{rel}" for rel in corpus.headers),
        excluded_ops=excluded,
        unclaimed_ops=unclaimed,
        unresolved_includes=corpus.unresolved_includes,
        pruned=pruned,
    )


# ---------------------------------------------------------------------------
# selection and include closure
# ---------------------------------------------------------------------------


def translate_corpus(
    src_root: str | Path,
    dest_root: str | Path,
    source_paths: Sequence[str],
    translate: Translator | None = None,
    preserve: frozenset[str] = frozenset(),
) -> TranslatedCorpus:
    """Copy, sailify, and return the local include closure of ``source_paths``."""
    translate = sailify_translator if translate is None else translate
    # Both resolved: _resolve() hands back resolved paths, so an unresolved root
    # breaks relative_to() wherever the path crosses a symlink (macOS /tmp).
    src = Path(src_root).resolve()
    dest = Path(dest_root).resolve()
    sources, headers, unresolved = _closure_paths(src, source_paths)

    _clean(dest, preserve)
    copied = tuple(_copy(src, dest, rel) for rel in (*sources, *headers))
    translate(dest, copied)
    return TranslatedCorpus(
        sources=tuple(rel.removeprefix(_CSRC) for rel in sources),
        headers=tuple(rel.removeprefix(_CSRC) for rel in headers),
        unresolved_includes=unresolved,
    )


def _closure(
    src: Path, entries: Sequence[Entry]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Selected TUs, the headers they reach, and includes that resolved nowhere.

    Resolution order matches sailify's and the compiler's: the including file's
    own directory first, then ``csrc/``. Upstream ships both ``csrc/ops.h`` and
    ``csrc/libtorch_stable/ops.h``, so own-directory-first is load-bearing.
    """
    return _closure_paths(src, tuple(entry.path for entry in entries))


def _closure_paths(
    src: Path, source_paths: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    sources: list[str] = []
    for path in source_paths:
        if not (src / path).is_file():
            raise PortError(
                f"the manifest names {path}, which does not exist under {src}"
            )
        if not path.startswith(_CSRC):
            raise PortError(f"the manifest path {path!r} is outside csrc/")
        sources.append(path)

    csrc = src / "csrc"
    headers: list[str] = []
    unresolved: set[str] = set()
    seen = set(sources)
    queue = list(sources)
    while queue:
        rel = queue.pop()
        includer = src / rel
        for line in includer.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _INCLUDE_RE.match(line)
            if match is None:
                continue
            target = _resolve(match.group(1), includer, csrc)
            if target is None:
                unresolved.add(match.group(1))
                continue
            found = target.relative_to(src).as_posix()
            if not found.startswith(_CSRC):
                raise PortError(f"{rel} includes {found}, which is outside csrc/")
            if found in seen:
                continue
            seen.add(found)
            headers.append(found)
            queue.append(found)

    return tuple(sources), tuple(sorted(headers)), tuple(sorted(unresolved))


def _resolve(include: str, includer: Path, csrc: Path) -> Path | None:
    for root in (includer.parent, csrc):
        candidate = root / include
        if candidate.is_file():
            return candidate.resolve()
    return None


def _vendored(rel: str) -> str:
    return f"{_manifest.VENDOR_ROOT}/{_manifest.vendored_name(rel)}"


def _clean(dest: Path, preserve: frozenset[str] = PRESERVED) -> None:
    """Empty the dest tree so a demoted or renamed TU cannot linger."""
    dest.mkdir(parents=True, exist_ok=True)
    for child in dest.iterdir():
        if child.name in preserve:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _copy(src: Path, dest: Path, rel: str) -> Path:
    target = dest / rel.removeprefix(_CSRC)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / rel, target)
    return target


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


def _drop_includes(files: Sequence[Path], names: Sequence[str]) -> None:
    """Remove ``#include "x"`` / ``<x>`` lines for headers we refuse to pull in."""
    if not names:
        return
    wanted = frozenset(names)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        kept = [
            line
            for line in text.splitlines(keepends=True)
            if not _includes_any(line, wanted)
        ]
        if len(kept) != len(text.splitlines()):
            path.write_text("".join(kept), encoding="utf-8")


def _includes_any(line: str, names: frozenset[str]) -> bool:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return False
    match = re.match(r'#\s*include\s*[<"]([^>"]+)[>"]', stripped)
    return match is not None and match.group(1).rsplit("/", 1)[-1] in names


def _replace_text(
    files: Sequence[Path], dest: Path, replacements: Mapping[str, Mapping[str, str]]
) -> None:
    """Apply exact post-sailify edits; stop on upstream drift, never guess."""
    # Map .hg files back to upstream .cu manifest keys.
    copied: dict[str, Path] = {}
    for path in files:
        rel = path.relative_to(dest).as_posix()
        upstream = f"{rel[:-3]}.cu" if rel.endswith(".hg") else rel
        copied[_CSRC + upstream] = path
    for name, edits in replacements.items():
        if name not in copied:
            raise PortError(f"overlay.replace_text names {name}, which was not copied")
        path = copied[name]
        source = path.read_text(encoding="utf-8")
        for before, after in edits.items():
            count = source.count(before)
            if count != 1:
                raise PortError(
                    f"overlay.replace_text in {name}: expected exactly one "
                    f"occurrence of {before!r}, found {count}; review upstream drift"
                )
            source = source.replace(before, after, 1)
        path.write_text(source, encoding="utf-8")


def _apply_bindings(
    dest: Path, manifest: Manifest
) -> tuple[Mapping[str, int], tuple[ExcludedOp, ...], tuple[str, ...]]:
    """Prune the binding TUs to the compiled tiers and fix their module names."""
    owner = {op: entry for entry in manifest for op in entry.ops}
    pruner = _BindingPruner(
        owner=owner,
        compiled=frozenset(manifest.compiled_tiers),
        hazards=manifest.hazards,
    )
    pruned: dict[str, int] = {}
    for rel in manifest.prune_bindings:
        path = dest / _manifest.vendored_name(rel)
        if not path.is_file():
            raise PortError(
                f"overlay.prune_bindings names {rel}, which this port did not "
                "copy; it must be an entry in a compiled tier"
            )
        text = pruner.prune(path.read_text(encoding="utf-8"))
        path.write_text(
            _rename_extension(text, manifest.rename_extension), encoding="utf-8"
        )
        pruned[_vendored(rel)] = pruner.dropped_in_last_file
    return pruned, pruner.excluded(), pruner.unclaimed()


def _rename_extension(text: str, renames: Mapping[str, str]) -> str:
    """Retarget ``REGISTER_EXTENSION`` so the TU exports our ``PyInit_`` symbol.

    ``REGISTER_EXTENSION(NAME)`` defines ``PyInit_<NAME>`` and names the module
    ``"<NAME>"``; a verbatim copy would export upstream's symbol and fail to
    import under our own extension name.
    """
    if not renames:
        return text

    def replace(match: re.Match[str]) -> str:
        new = renames.get(match.group(2))
        if new is None:
            return match.group(0)
        return f"{match.group(1)}{new}{match.group(3)}"

    return _REGISTER_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# binding pruning
# ---------------------------------------------------------------------------


class _BindingPruner:
    """Drops ``def``/``impl`` statements whose op lives in an uncompiled tier.

    Line-oriented on purpose. The binding TUs are generated-looking but not
    generated, and a real C++ parse would be far more machinery than the shape
    of these files warrants: every registration is a single ``x.def(...)`` or
    ``x.impl(...)`` statement terminated by ``;``, preprocessor lines only ever
    appear at statement boundaries, and the only lone ``}`` lines close the
    registration blocks themselves. All three are asserted by the merge gate,
    and anything unrecognised is preserved rather than guessed at.
    """

    def __init__(
        self,
        owner: Mapping[str, Entry],
        compiled: frozenset[str],
        hazards: Mapping[str, str],
    ) -> None:
        self._owner = owner
        self._compiled = compiled
        self._hazards = hazards
        self._excluded: dict[tuple[str, str], ExcludedOp] = {}
        self._unclaimed: set[str] = set()
        self.dropped_in_last_file = 0

    def excluded(self) -> tuple[ExcludedOp, ...]:
        return tuple(
            self._excluded[key] for key in sorted(self._excluded, key=lambda k: k[::-1])
        )

    def unclaimed(self) -> tuple[str, ...]:
        """Ops upstream registers that no manifest entry describes."""
        return tuple(sorted(self._unclaimed))

    def prune(self, text: str) -> str:
        self.dropped_in_last_file = 0
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        index = 0
        while index < len(lines):
            match = _BLOCK_RE.match(lines[index])
            if match is None:
                out.append(lines[index])
                index += 1
                continue
            while index < len(lines):
                out.append(lines[index])
                opens = "{" in lines[index]
                index += 1
                if opens:
                    break
            index = self._body(lines, index, out, match.group(1))
        return "".join(out)

    def _body(
        self, lines: Sequence[str], index: int, out: list[str], namespace: str
    ) -> int:
        pending: list[str] = []
        statement: list[str] = []
        while index < len(lines):
            line = lines[index]
            index += 1
            if statement:
                statement.append(line)
            else:
                stripped = line.strip()
                if stripped in ("}", "};"):
                    out.extend(pending)
                    out.append(line)
                    return index
                if stripped.startswith("#"):
                    out.extend(pending)
                    pending = []
                    out.append(line)
                    continue
                if not stripped or stripped.startswith("//"):
                    pending.append(line)
                    continue
                statement = [line]
            if _complete("".join(statement)):
                self._settle(out, pending, statement, namespace)
                pending, statement = [], []
        raise PortError(f"unterminated {namespace} registration block")

    def _settle(
        self, out: list[str], pending: list[str], statement: list[str], namespace: str
    ) -> None:
        """Emit or drop one complete statement, with the comments attached to it."""
        text = "".join(statement)
        call = _CALL_RE.search(text)
        name = None if call is None else _op_name(text, call.group(1))
        if name is None:
            out.extend(pending)
            out.append(text)
            return

        entry = self._owner.get(name)
        if entry is not None and entry.tier in self._compiled:
            out.extend(pending)
            out.append(text)
            return

        # No entry means no ported TU, hence no implementation: dropping and
        # stubbing is the only correct outcome. It is still worth reporting,
        # because it is how a vLLM bump's new kernels announce themselves.
        if entry is None:
            self._unclaimed.add(name)

        self.dropped_in_last_file += 1
        if call.group(1) == "def":
            schema = _schema(text)
            self._excluded[(namespace, name)] = ExcludedOp(
                namespace=namespace,
                name=name,
                schema=schema,
                kind="probe" if _is_probe(schema) else "compute",
                tier=entry.tier if entry else _UNCLAIMED_TIER,
                reason=self._reason(entry),
            )

    def _reason(self, entry: Entry | None) -> str:
        if entry is None:
            return "not declared in csrc/upstream/manifest.toml"
        described = [self._hazards.get(h, h) for h in entry.hazards]
        return "; ".join(described) if described else "no PPU implementation"


def _complete(text: str) -> bool:
    """Whether the accumulated text ends a statement, ignoring string bodies."""
    return ";" in _STRING_RE.sub('""', text)


def _op_name(text: str, kind: str) -> str | None:
    """The op a ``def``/``impl`` statement registers.

    ``def`` takes a schema built from adjacent string literals that the compiler
    concatenates, so the name is the leading identifier of the joined text.
    ``impl`` names the op in its first literal.
    """
    literals = _STRING_RE.findall(text)
    if not literals:
        return None
    joined = "".join(literals) if kind == "def" else literals[0]
    match = _OPNAME_RE.match(joined.strip())
    return None if match is None else match.group(1)


def _schema(text: str) -> str:
    return "".join(_STRING_RE.findall(text)).strip()


def _is_probe(schema: str) -> bool:
    """Whether the op is a capability query, which must answer ``False``.

    A probe takes no tensor and answers a bool; anything else that we cannot run
    has to raise instead, or a stub would silently produce wrong numerics.
    """
    args, _, result = schema.partition("->")
    return result.strip() == "bool" and "Tensor" not in args


# ---------------------------------------------------------------------------
# generated output
# ---------------------------------------------------------------------------


def render_excluded_ops(manifest: Manifest, ops: Sequence[ExcludedOp]) -> str:
    """Render ``excluded_ops.toml``: the stub registry's input."""
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# GENERATED by tools/port_upstream_kernels.py -- do not edit.",
        "#",
        "# Every upstream op whose translation unit sits outside the compiled",
        "# tiers. vllm_sail/native/stubs.py registers each one so a call fails",
        "# with a named NotImplementedError instead of an AttributeError, and so",
        "# capability probes answer False instead of raising.",
        f"# Upstream vLLM {manifest.upstream_ref}; compiled tiers "
        f"{', '.join(manifest.compiled_tiers)}.",
        "",
    ]
    for op in ops:
        lines += [
            "[[op]]",
            f'namespace = "{_escape(op.namespace)}"',
            f'name = "{_escape(op.name)}"',
            f'schema = "{_escape(op.schema)}"',
            f'kind = "{_escape(op.kind)}"',
            f'tier = "{_escape(op.tier)}"',
            f'reason = "{_escape(op.reason)}"',
            "",
        ]
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def report(manifest: Manifest, result: PortResult) -> str:
    """A human-readable summary of what was ported and what needs eyes."""
    compiled = manifest.compiled()
    lines = [
        f"ported {len(result.sources)} translation unit(s) and "
        f"{len(result.headers)} header(s) from vLLM {manifest.upstream_ref}",
        f"compiled tiers: {', '.join(manifest.compiled_tiers)}",
        f"stubbed ops: {len(result.excluded_ops)}",
    ]
    for path, dropped in sorted(result.pruned.items()):
        lines.append(f"pruned {dropped} registration(s) from {path}")
    if result.unclaimed_ops:
        lines += [
            "",
            f"{len(result.unclaimed_ops)} op(s) upstream registers that the "
            "manifest does not describe, so they were stubbed. Triage each into "
            "a tier (or leave it in none, deliberately):",
            *(f"  {name}" for name in result.unclaimed_ops),
        ]
    if result.unresolved_includes:
        lines += [
            "",
            "includes that resolved to nothing (external or provided by the SDK):",
            *(f"  {name}" for name in result.unresolved_includes),
        ]
    hazardous = [entry for entry in compiled if entry.hazards]
    if hazardous:
        lines += ["", "compiled files with hazards sailify does not fix:"]
        for entry in hazardous:
            lines.append(f"  {entry.path}: {', '.join(entry.hazards)}")
    return "\n".join(lines)
