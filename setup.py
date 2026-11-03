# SPDX-License-Identifier: Apache-2.0
"""Build wiring for the vllm-sail compiled extensions.

PPU is CUDA-free (docs/developer_guide/kernels.md): device code is
compiled by ``hgcc`` against HGGC headers, not by nvcc against CUDA headers.
So this file no longer speaks nvcc's dialect at all -- there is no
``CUDAExtension``, no ``TORCH_CUDA_ARCH_LIST``, no surgery on torch's
``COMMON_NVCC_FLAGS``. It only decides *whether* to build, and hands the *what*
and *how* to ``CMakeLists.txt`` via ``cmake-hgcc``.

Four extensions are produced (see ``CMakeLists.txt``):

* ``vllm_sail._upstream_C``     -> op namespaces ``_C``, ``_C_cache_ops``,
                                  ``_C_cuda_utils`` (kernels ported from vLLM)
* ``vllm_sail._upstream_moe_C`` -> op namespace ``_moe_C``
* ``vllm_sail._C``              -> ``_ppu_C``      (native HGGC BF16 sampler)
* ``vllm_sail._moe_C``          -> ``_ppu_moe_C``  (native HGGC ep_scatter)

Native extensions are the default. The only Python-only path is an explicit
opt-out; otherwise the build validates every prerequisite before running CMake:

* ``VLLM_SAIL_PREBUILT_EXT_DIR`` -> package four prebuilt native extensions
* ``VLLM_SAIL_SKIP_EXT`` truthy   -> pure-Python wheel
* otherwise                      -> require importable torch, hgcc and CMake,
                                    then configure and build

Missing native prerequisites fail with the first actionable reason. CPU-only
development and packaging remain available through the explicit opt-out.

Environment variables:

* ``PPU_SDK``             SDK root; also how cmake-hgcc finds ``bin/hgcc`` and
                          the HGGC toolkit.
* ``HGCC``                explicit ``hgcc`` path; wins over ``PPU_SDK/bin/hgcc``.
* ``VLLM_SAIL_SKIP_EXT``  force a pure-Python wheel.
* ``VLLM_SAIL_PREBUILT_EXT_DIR`` directory containing the four extension files
                                  to package without invoking CMake/HGCC.
* ``PYTORCH_SAIL_ARCH``  -> HG and Torch architecture lists (default ``ppu_15;ppu_10``).
* ``VLLM_SAIL_HG_ARCH``   plugin architecture list; must match ``PYTORCH_SAIL_ARCH``
                         when both are nonempty.
* ``VLLM_SAIL_HG_STD``    -> ``CMAKE_HG_STANDARD`` (default ``20``).
* ``VLLM_SAIL_HG_FLAGS``  extra HG compiler flags via ``CMAKE_HG_FLAGS``.
* ``CMAKE_HGCC_DIR``      optional path to a ``cmake-hgcc`` checkout. CMake
                          fetches a pinned revision when no local modules or
                          installed package are available.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

logger = logging.getLogger("vllm-sail-build")

_ROOT = Path(__file__).resolve().parent

# One Extension per CMake target. CMake still builds all four in a single
# invocation, but each output must be declared: PEP 660 editable installs build
# into build_lib and setuptools then copies back exactly the declared
# extensions, while the wheel packages build_lib as a whole.
_EXT_NAMES = (
    "vllm_sail._upstream_C",
    "vllm_sail._upstream_moe_C",
    "vllm_sail._C",
    "vllm_sail._moe_C",
)


def _load(name: str):
    """Load a ``vllm_sail.native`` module without importing the package.

    A build backend must not import the package it is building: ``vllm_sail``
    pulls in its version module and, transitively, torch-shaped code. These
    modules are deliberately stdlib-only so they can be loaded straight off disk.
    """
    path = _ROOT / "vllm_sail" / "native" / f"{name}.py"
    modname = f"_vllm_sail_build_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves field types through sys.modules[cls.__module__], so the
    # module must be registered before its body runs.
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


_toolchain = _load("toolchain")
_manifest = _load("manifest")
_plugin_manifest = _load("plugin_manifest")
_resources = _load("resources")


class _BuildPy(build_py):
    """Add generated native metadata to the importable package tree."""

    def run(self) -> None:
        super().run()
        _resources.copy_to_build(_ROOT, self.build_lib)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        outputs.extend(str(path) for path in _resources.build_outputs(self.build_lib))
        return outputs


class _CMakeBuildExt(build_ext):
    """Delegates the whole native build to CMake + cmake-hgcc.

    ``build_ext`` is kept only for its output-path bookkeeping: it knows where a
    wheel build and an ``--inplace`` editable build each want the shared objects,
    and CMake is told to put all four there via ``PPU_EXT_OUTPUT_DIR``.
    """

    def build_extensions(self) -> None:
        if self.extensions:
            self.build_extension(self.extensions[0])

    def build_extension(self, ext: Extension) -> None:
        decision = _toolchain.decide(_toolchain.probe())
        if not decision.build:  # The opt-out must be present when setup initializes.
            raise _toolchain.NativeBuildError(
                f"{decision.reason}; set VLLM_SAIL_SKIP_EXT before starting the build"
            )
        logger.warning("vllm-sail: building compiled extensions (%s)", decision.reason)
        out_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        build_dir = Path(self.build_temp).resolve() / "cmake"
        build_dir.mkdir(parents=True, exist_ok=True)

        manifest = _manifest.load(_resources.source_path(_ROOT, "manifest.toml"))
        plugin_extensions = _plugin_manifest.load(
            _ROOT / "csrc" / "plugin" / "manifest.toml"
        )
        sources, moe_sources = manifest.compiled_sources()
        logger.warning(
            "vllm-sail: building %d upstream kernel sources (%d MoE) from "
            "manifest tiers %s pinned at vLLM %s",
            len(sources),
            len(moe_sources),
            ",".join(manifest.compiled_tiers),
            manifest.upstream_ref,
        )

        defines = _toolchain.cmake_defines(sources, moe_sources)
        for extension in plugin_extensions:
            variable = _plugin_manifest.cmake_variable(extension.name)
            defines[variable] = ";".join(extension.cmake_sources())
        defines["PPU_EXT_OUTPUT_DIR"] = str(out_dir)
        defines["Python_EXECUTABLE"] = sys.executable
        defines["CMAKE_BUILD_TYPE"] = os.getenv("CMAKE_BUILD_TYPE", "Release")

        configure = [
            _toolchain.find_cmake() or "cmake",
            "-S",
            str(_ROOT),
            "-B",
            str(build_dir),
            *(f"-D{key}={value}" for key, value in sorted(defines.items())),
        ]
        if _toolchain.find_ninja():
            configure += ["-G", "Ninja"]
        self._run(configure)

        build = [configure[0], "--build", str(build_dir)]
        jobs = _toolchain.build_parallelism(self.parallel)
        build += ["--parallel", str(jobs)]
        self._run(build)

    def _run(self, argv: list[str]) -> None:
        logger.warning("vllm-sail: %s", " ".join(argv))
        subprocess.run(argv, check=True)


class _PrebuiltBuildExt(build_ext):
    """Copy verified native extensions into a wheel without recompiling them."""

    def build_extensions(self) -> None:
        source_dir = Path(os.environ["VLLM_SAIL_PREBUILT_EXT_DIR"]).resolve()
        if not source_dir.is_dir():
            raise RuntimeError(
                f"prebuilt extension directory does not exist: {source_dir}"
            )
        for ext in self.extensions:
            destination = Path(self.get_ext_fullpath(ext.name)).resolve()
            matches = [
                path
                for path in source_dir.rglob(destination.name)
                if path.is_file() and path.stat().st_size > 0
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one non-empty prebuilt {destination.name}, "
                    f"found {len(matches)} in {source_dir}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[0], destination)
            logger.warning("vllm-sail: reused native extension %s", destination.name)


class _SkippedBuildExt(build_ext):
    """No-op ``build_ext`` for the pure-Python wheel.

    The extension list is already empty in that case; this exists so a stray
    ``python setup.py build_ext`` does not try to find a compiler.
    """

    def run(self) -> None:
        return


_PREBUILT_EXT_DIR = os.getenv("VLLM_SAIL_PREBUILT_EXT_DIR", "").strip()
if _PREBUILT_EXT_DIR and _toolchain.skip_requested():
    raise RuntimeError(
        "VLLM_SAIL_PREBUILT_EXT_DIR and VLLM_SAIL_SKIP_EXT are mutually exclusive"
    )

if _PREBUILT_EXT_DIR:
    logger.warning(
        "vllm-sail: packaging prebuilt compiled extensions from %s",
        _PREBUILT_EXT_DIR,
    )
    _EXT_MODULES = [Extension(name, sources=[]) for name in _EXT_NAMES]
    _BUILD_EXT = _PrebuiltBuildExt
elif _toolchain.skip_requested():
    logger.warning(
        "vllm-sail: building a PURE-PYTHON wheel; compiled extensions skipped "
        "(VLLM_SAIL_SKIP_EXT requests it). PPU kernels will be unavailable until "
        "the package is rebuilt on a host with the PPU toolchain."
    )
    _EXT_MODULES = []
    _BUILD_EXT = _SkippedBuildExt
else:
    logger.warning(
        "vllm-sail: compiled extensions enabled by default; validating the PPU "
        "toolchain during build_ext"
    )
    _EXT_MODULES = [Extension(name, sources=[]) for name in _EXT_NAMES]
    _BUILD_EXT = _CMakeBuildExt

setup(
    ext_modules=_EXT_MODULES,
    cmdclass={"build_ext": _BUILD_EXT, "build_py": _BuildPy},
)
