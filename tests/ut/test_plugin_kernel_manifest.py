# SPDX-License-Identifier: Apache-2.0
"""Native plugin source ownership and the build inputs consumed by CMake."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vllm_sail.native import plugin_manifest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_MANIFEST = ROOT / "csrc" / "plugin" / "manifest.toml"


def test_shipped_manifest_owns_every_plugin_translation_unit() -> None:
    extensions = plugin_manifest.load(PLUGIN_MANIFEST)
    declared = {path for extension in extensions for path in extension.cmake_sources()}
    found = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "csrc").rglob("*")
        if path.suffix in {".hg", ".cu", ".cpp"} and "upstream" not in path.parts
    }

    assert declared
    assert declared == found
    for extension in extensions:
        assert all(path.endswith(".hg") for path in extension.device_sources)
        assert all(path.endswith(".cpp") for path in extension.bindings)
        assert all((ROOT / path).is_file() for path in extension.cmake_sources())


def test_cmake_uses_the_native_plugin_sources_and_include_root() -> None:
    extensions = plugin_manifest.load(PLUGIN_MANIFEST)
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    for extension in extensions:
        variable = plugin_manifest.cmake_variable(extension.name)
        if extension.name == "_cumem_allocator":
            assert not extension.device_sources
            assert (
                f"Python_add_library(_cumem_allocator MODULE WITH_SOABI ${{{variable}}})"
                in cmake
            )
            assert "PRIVATE HGGC::hggc HGGC::hggcrt" in cmake
            continue
        block = re.search(
            rf"ppu_add_extension\({re.escape(extension.name)}\b(?P<body>.*?)\)",
            cmake,
            re.DOTALL,
        )
        assert block is not None
        assert f"${{{variable}}}" in block.group("body")
        assert "INCLUDE_ROOT csrc/plugin\n" in block.group("body")


def test_native_plugin_sources_do_not_include_cuda_sdk_headers() -> None:
    files = sorted(path for path in (ROOT / "csrc/plugin").rglob("*") if path.is_file())

    assert files
    forbidden = re.compile(r"#\s*include\s*[<\"](?:cuda[^/\">]*|cublas_v2)\.h[>\"]")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if forbidden.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "path",
    sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "csrc/plugin").rglob("*")
        if path.suffix in {".hg", ".cuh", ".h", ".cpp"}
    ),
)
def test_native_plugin_sources_use_hggc_runtime_identifiers(path: str) -> None:
    source = (ROOT / path).read_text(encoding="utf-8")
    # SAIL PyTorch retains CUDAGuard and the CUDA dispatch key. Raw CUDA
    # runtime APIs, types and enums need sailify instead.
    # Ignore diagnostic strings and comments, including historical CUDA names.
    code = re.sub(
        r'R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"'
        r'|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        " ",
        source,
        flags=re.DOTALL,
    )
    assert re.findall(r"\bcuda[A-Z]\w*\b", code) == []
    if "step5" in Path(path).parts:
        # CUDAStream and CUDA error enums are distinct from HGGC's types in
        # SDK 2.2. Use the existing stable stream getter and HGGC checks.
        assert not re.search(
            r"\bC10_CUDA_(?:CHECK|KERNEL_LAUNCH_CHECK)\s*\("
            r"|\bat::cuda::getCurrentCUDAStream\s*\(",
            code,
        )


@pytest.mark.parametrize(
    ("original", "replacement", "error"),
    [
        ("schema = 2", "schema = 1", "schema 1"),
        ("sampler_bf16.hg", "sampler_bf16.cu", "invalid device source"),
        ("ppu_bindings.cpp", "ppu_bindings.hg", "invalid binding source"),
        ('name = "_moe_C"', 'name = "_C"', "extensions must be exactly"),
        (
            "csrc/plugin/libtorch_stable/moe/ep_scatter_kernels.hg",
            "csrc/plugin/sampler_bf16.hg",
            "source path is listed more than once",
        ),
    ],
)
def test_manifest_rejects_invalid_build_inputs(tmp_path, original, replacement, error):
    path = tmp_path / "manifest.toml"
    path.write_text(PLUGIN_MANIFEST.read_text().replace(original, replacement))
    with pytest.raises(plugin_manifest.PluginManifestError, match=error):
        plugin_manifest.load(path)
