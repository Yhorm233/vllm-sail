# SPDX-License-Identifier: Apache-2.0
"""The PPU/HGGC build toolchain gate.

``setup.py`` builds native extensions unless the caller explicitly opts out.
Before CMake runs, this module validates the required toolchain because enabling
cmake-hgcc's ``HG`` language without ``hgcc`` present is a ``FATAL_ERROR`` with
no recovery path. The decision is expressed as a pure function over
already-probed inputs, so it is table-testable at the merge gate on a machine
with no PPU SDK, no hgcc, no CMake and no torch.

See ``docs/developer_guide/kernels.md``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Decision",
    "NativeBuildError",
    "Probe",
    "cmake_defines",
    "decide",
    "find_cmake",
    "find_hgcc",
    "find_ninja",
    "probe",
    "skip_requested",
]

# Build both supported PPU generations unless the caller selects a subset.
DEFAULT_HG_ARCH = "ppu_15;ppu_10"
DEFAULT_HG_STANDARD = "20"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Sentinel so ``build_parallelism`` can tell "cpu_count omitted" (query the OS)
# apart from an explicit ``None`` (the count is unknown).
_UNSET = object()


def _truthy(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class Probe:
    """What the environment offers. Everything the decision depends on."""

    skip_requested: bool
    have_torch: bool
    hgcc: str | None
    cmake: str | None


@dataclass(frozen=True)
class Decision:
    """Whether to build, and the one-line reason to log either way."""

    build: bool
    reason: str


class NativeBuildError(RuntimeError):
    """A default native build is missing a required toolchain component."""


def _missing_prerequisite(reason: str) -> NativeBuildError:
    return NativeBuildError(
        f"{reason}. Native extensions are required by default; fix the build "
        "environment, or set VLLM_SAIL_SKIP_EXT=1 to intentionally build a "
        "Python-only wheel"
    )


def decide(p: Probe) -> Decision:
    """Resolve a :class:`Probe` into a native or explicit Python-only build.

    Order matters: an explicit opt-out wins over everything. Without it, each
    missing prerequisite fails the build with one actionable reason.
    """
    if p.skip_requested:
        return Decision(False, "VLLM_SAIL_SKIP_EXT requests a Python-only build")
    if not p.have_torch:
        raise _missing_prerequisite("torch is not importable in the build environment")
    if not p.hgcc:
        raise _missing_prerequisite(
            "hgcc was not found (set HGCC, or PPU_SDK with bin/hgcc, or put it on PATH)"
        )
    if not p.cmake:
        raise _missing_prerequisite("cmake was not found on PATH")
    return Decision(True, f"building with hgcc={p.hgcc}")


def skip_requested(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the caller explicitly requested a Python-only package."""
    env = os.environ if env is None else env
    return _truthy(env, "VLLM_SAIL_SKIP_EXT")


def build_parallelism(
    explicit: int | None = None,
    env: Mapping[str, str] | None = None,
    cpu_count: int | None | object = _UNSET,
) -> int:
    """Decide the ``cmake --build --parallel`` level for the native build.

    Native PPU kernels are memory-heavy (~GiB per compile job). A many-core
    CPU runner with a smaller memory limit OOM-kills the pod (exit 137) if the
    build fans out to ``os.cpu_count()``. Honour ``MAX_JOBS`` -- the same knob
    vLLM's own build reads, and the one this repo's CI exports -- so the cap is
    actually applied. Precedence: an explicit setuptools ``-j`` wins, then
    ``MAX_JOBS``, then the CPU count, then a single job.

    ``cpu_count`` defaults to ``os.cpu_count()`` when omitted; pass it (even
    ``None``) to override, where ``None`` means the count is unknown.
    """
    if explicit:
        return explicit
    env = os.environ if env is None else env
    raw = env.get("MAX_JOBS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    if cpu_count is _UNSET:
        cpu_count = os.cpu_count()
    return cpu_count or 1  # type: ignore[return-value]


def find_hgcc(env: Mapping[str, str] | None = None) -> str | None:
    """Locate ``hgcc`` the same way cmake-hgcc does: HGCC, PPU_SDK, then PATH."""
    env = os.environ if env is None else env
    explicit = env.get("HGCC", "").strip()
    if explicit:
        return explicit if Path(explicit).is_file() else None
    sdk = env.get("PPU_SDK", "").strip()
    if sdk:
        candidate = Path(sdk) / "bin" / "hgcc"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("hgcc", path=env.get("PATH"))


def find_cmake(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    return shutil.which("cmake", path=env.get("PATH"))


def find_ninja(env: Mapping[str, str] | None = None) -> str | None:
    """Locate ``ninja``. cmake-hgcc supports only Makefiles and Ninja."""
    env = os.environ if env is None else env
    return shutil.which("ninja", path=env.get("PATH"))


def probe(env: Mapping[str, str] | None = None) -> Probe:
    """Gather the environment facts. The only impure part of this module."""
    env = os.environ if env is None else env
    try:
        import torch  # noqa: F401 - importability is the only fact probed
    except ImportError:
        have_torch = False
    else:
        have_torch = True
    return Probe(
        skip_requested=skip_requested(env),
        have_torch=have_torch,
        hgcc=find_hgcc(env),
        cmake=find_cmake(env),
    )


def cmake_defines(
    sources: tuple[str, ...],
    moe_sources: tuple[str, ...],
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The full ``-D`` surface handed to CMake.

    Source lists come from the kernel manifest rather than from CMake globbing,
    so tier selection has exactly one home. ``CMAKE_HG_*`` values must reach
    CMake before ``project()`` enables the ``HG`` language, which is why they are
    cache entries rather than target properties.
    """
    env = os.environ if env is None else env
    architectures = _hg_architectures(env)
    defines = {
        "CMAKE_HG_ARCHITECTURES": architectures,
        "PYTORCH_SAIL_ARCH": architectures,
        "CMAKE_HG_STANDARD": env.get("VLLM_SAIL_HG_STD", "").strip()
        or DEFAULT_HG_STANDARD,
        "PPU_UPSTREAM_EXT_SRC": ";".join(sources),
        "PPU_UPSTREAM_MOE_EXT_SRC": ";".join(moe_sources),
    }
    hgcc = find_hgcc(env)
    if hgcc:
        defines["CMAKE_HG_COMPILER"] = hgcc
    extra = env.get("VLLM_SAIL_HG_FLAGS", "").strip()
    if extra:
        defines["CMAKE_HG_FLAGS"] = extra
    cmake_hgcc = env.get("CMAKE_HGCC_DIR", "").strip()
    if cmake_hgcc:
        defines["CMAKE_HGCC_DIR"] = cmake_hgcc
    sdk = env.get("PPU_SDK", "").strip()
    if sdk:
        defines["HGGCToolkit_ROOT"] = sdk
    return defines


def _hg_architectures(env: Mapping[str, str]) -> str:
    """Use one architecture list for cmake-hgcc and SAIL TorchConfig."""

    def parse(name: str) -> tuple[str, ...]:
        # SAIL torch accepts either semicolons or whitespace; CMake needs a list.
        return tuple(dict.fromkeys(env.get(name, "").replace(";", " ").split()))

    torch_archs = parse("PYTORCH_SAIL_ARCH")
    plugin_archs = parse("VLLM_SAIL_HG_ARCH")
    if torch_archs and plugin_archs and set(torch_archs) != set(plugin_archs):
        raise ValueError(
            "PYTORCH_SAIL_ARCH and VLLM_SAIL_HG_ARCH select different architectures; "
            "unset VLLM_SAIL_HG_ARCH or make both lists match"
        )
    return ";".join(torch_archs or plugin_archs) or DEFAULT_HG_ARCH
