# SPDX-License-Identifier: Apache-2.0
"""Tests for the PPU/HGGC build gate, ``vllm_sail.native.toolchain``.

Enabling cmake-hgcc's ``HG`` language without ``hgcc`` present is a CMake
``FATAL_ERROR`` with no recovery, so the whole decision has to be made in Python
first. That makes it testable here: ``decide`` is pure, and every discovery
function takes an explicit environment mapping. No test reads the real
``os.environ`` — a stray ``HGCC`` or ``PPU_SDK`` on a developer machine must not
change a result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_sail.native import toolchain

HGCC = "/opt/ppu/bin/hgcc"
CMAKE = "/usr/bin/cmake"


def _probe(**overrides: object) -> toolchain.Probe:
    """A fully-satisfied probe, so each case states only what it varies."""
    fields: dict[str, object] = {
        "skip_requested": False,
        "have_torch": True,
        "hgcc": HGCC,
        "cmake": CMAKE,
    }
    fields.update(overrides)
    return toolchain.Probe(**fields)  # type: ignore[arg-type]


def _exe(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def nowhere(tmp_path: Path) -> str:
    """A PATH that resolves nothing, so lookups cannot leak to the real one."""
    empty = tmp_path / "empty-path"
    empty.mkdir()
    return str(empty)


@pytest.mark.parametrize(
    ("probe", "build", "reason"),
    [
        pytest.param(_probe(), True, f"hgcc={HGCC}", id="all-present"),
        pytest.param(
            _probe(skip_requested=True), False, "VLLM_SAIL_SKIP_EXT", id="opt-out"
        ),
    ],
)
def test_decide(probe: toolchain.Probe, build: bool, reason: str) -> None:
    decision = toolchain.decide(probe)
    assert decision.build is build
    assert reason in decision.reason


@pytest.mark.parametrize(
    ("probe", "reason"),
    [
        pytest.param(_probe(have_torch=False), "torch", id="no-torch"),
        pytest.param(_probe(hgcc=None), "hgcc", id="no-hgcc"),
        pytest.param(_probe(hgcc=""), "hgcc", id="blank-hgcc"),
        pytest.param(_probe(cmake=None), "cmake", id="no-cmake"),
    ],
)
def test_missing_prerequisite_fails_the_default_native_build(
    probe: toolchain.Probe, reason: str
) -> None:
    with pytest.raises(toolchain.NativeBuildError) as caught:
        toolchain.decide(probe)

    assert reason in str(caught.value)
    assert "VLLM_SAIL_SKIP_EXT=1" in str(caught.value)


def test_skip_request_wins_over_a_complete_toolchain() -> None:
    decision = toolchain.decide(_probe(skip_requested=True))
    assert not decision.build
    assert "VLLM_SAIL_SKIP_EXT" in decision.reason
    assert "hgcc" not in decision.reason


def test_missing_torch_is_reported_before_missing_hgcc() -> None:
    with pytest.raises(toolchain.NativeBuildError) as caught:
        toolchain.decide(_probe(have_torch=False, hgcc=None, cmake=None))
    assert "torch is not importable" in str(caught.value)
    assert "hgcc" not in str(caught.value)


def test_missing_hgcc_is_reported_before_missing_cmake() -> None:
    with pytest.raises(toolchain.NativeBuildError) as caught:
        toolchain.decide(_probe(hgcc=None, cmake=None))
    assert "hgcc was not found" in str(caught.value)
    assert "cmake was not found" not in str(caught.value)


def test_find_hgcc_prefers_explicit_hgcc(tmp_path: Path, nowhere: str) -> None:
    explicit = _exe(tmp_path / "custom" / "hgcc")
    _exe(tmp_path / "sdk" / "bin" / "hgcc")
    env = {
        "HGCC": f"  {explicit}  ",
        "PPU_SDK": str(tmp_path / "sdk"),
        "PATH": nowhere,
    }
    assert toolchain.find_hgcc(env) == explicit


def test_find_hgcc_refuses_to_second_guess_a_wrong_explicit_path(
    tmp_path: Path,
) -> None:
    """An explicit HGCC that does not exist is a mistake worth surfacing.

    Falling through to PPU_SDK or PATH here would silently build with a compiler
    the maintainer did not ask for, so the miss is deliberate.
    """
    sdk = tmp_path / "sdk"
    _exe(sdk / "bin" / "hgcc")
    on_path = tmp_path / "bin"
    _exe(on_path / "hgcc")
    env = {
        "HGCC": str(tmp_path / "typo" / "hgcc"),
        "PPU_SDK": str(sdk),
        "PATH": str(on_path),
    }
    assert toolchain.find_hgcc(env) is None


def test_find_hgcc_uses_the_sdk_layout(tmp_path: Path, nowhere: str) -> None:
    sdk = tmp_path / "sdk"
    expected = _exe(sdk / "bin" / "hgcc")
    assert toolchain.find_hgcc({"PPU_SDK": str(sdk), "PATH": nowhere}) == expected


def test_find_hgcc_falls_through_an_sdk_without_a_compiler(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    on_path = tmp_path / "bin"
    expected = _exe(on_path / "hgcc")
    env = {"PPU_SDK": str(sdk), "PATH": str(on_path)}
    assert toolchain.find_hgcc(env) == expected


def test_find_hgcc_searches_path_last(tmp_path: Path) -> None:
    on_path = tmp_path / "bin"
    expected = _exe(on_path / "hgcc")
    assert toolchain.find_hgcc({"PATH": str(on_path)}) == expected


def test_find_hgcc_returns_none_when_nothing_is_installed(nowhere: str) -> None:
    assert toolchain.find_hgcc({"PATH": nowhere}) is None


@pytest.mark.parametrize(
    "value", ["1", "true", "TRUE", "Yes", "on", " ON ", "\ttrue\n"]
)
def test_probe_honours_a_skip_request(nowhere: str, value: str) -> None:
    env = {"VLLM_SAIL_SKIP_EXT": value, "PATH": nowhere}
    assert toolchain.probe(env).skip_requested


@pytest.mark.parametrize("value", ["0", "", "   ", "maybe", "false", "no", "off", "2"])
def test_probe_ignores_a_non_truthy_skip_request(nowhere: str, value: str) -> None:
    env = {"VLLM_SAIL_SKIP_EXT": value, "PATH": nowhere}
    assert not toolchain.probe(env).skip_requested


def test_probe_without_the_variable_does_not_skip(nowhere: str) -> None:
    assert not toolchain.probe({"PATH": nowhere}).skip_requested


def test_cmake_defines_defaults(nowhere: str) -> None:
    defines = toolchain.cmake_defines((), (), {"PATH": nowhere})
    assert (
        defines["CMAKE_HG_ARCHITECTURES"]
        == defines["PYTORCH_SAIL_ARCH"]
        == toolchain.DEFAULT_HG_ARCH
        == "ppu_15;ppu_10"
    )
    assert defines["CMAKE_HG_STANDARD"] == toolchain.DEFAULT_HG_STANDARD == "20"


@pytest.mark.parametrize(
    "key", ["CMAKE_HG_COMPILER", "CMAKE_HG_FLAGS", "CMAKE_HGCC_DIR", "HGGCToolkit_ROOT"]
)
def test_cmake_defines_omits_optional_keys_without_their_inputs(
    nowhere: str, key: str
) -> None:
    assert key not in toolchain.cmake_defines((), (), {"PATH": nowhere})


def test_cmake_defines_honours_arch_and_standard_overrides(nowhere: str) -> None:
    env = {"PATH": nowhere, "VLLM_SAIL_HG_ARCH": " ppu_10 ", "VLLM_SAIL_HG_STD": " 17 "}
    defines = toolchain.cmake_defines((), (), env)
    assert defines["CMAKE_HG_ARCHITECTURES"] == "ppu_10"
    assert defines["CMAKE_HG_STANDARD"] == "17"


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"PYTORCH_SAIL_ARCH": "ppu_15;ppu_10"}, "ppu_15;ppu_10"),
        ({"PYTORCH_SAIL_ARCH": "ppu_10"}, "ppu_10"),
        ({"VLLM_SAIL_HG_ARCH": "ppu_15;ppu_10"}, "ppu_15;ppu_10"),
        ({"PYTORCH_SAIL_ARCH": " ppu_15 ; ppu_10;ppu_15 "}, "ppu_15;ppu_10"),
        (
            {
                "PYTORCH_SAIL_ARCH": "ppu_15;ppu_10",
                "VLLM_SAIL_HG_ARCH": "ppu_10;ppu_15",
            },
            "ppu_15;ppu_10",
        ),
    ],
)
def test_torch_and_hgcc_receive_the_same_architectures(nowhere, settings, expected):
    defines = toolchain.cmake_defines((), (), {"PATH": nowhere, **settings})
    assert defines["CMAKE_HG_ARCHITECTURES"] == expected
    assert defines["PYTORCH_SAIL_ARCH"] == expected


def test_conflicting_architecture_settings_fail_instead_of_losing_a_target(nowhere):
    with pytest.raises(ValueError, match="PYTORCH_SAIL_ARCH.*VLLM_SAIL_HG_ARCH"):
        toolchain.cmake_defines(
            (),
            (),
            {
                "PATH": nowhere,
                "PYTORCH_SAIL_ARCH": "ppu_15;ppu_10",
                "VLLM_SAIL_HG_ARCH": "ppu_15",
            },
        )


def test_cmake_defines_joins_source_lists_for_cmake(nowhere: str) -> None:
    sources = (
        "csrc/upstream/libtorch_stable/layernorm_kernels.hg",
        "csrc/upstream/libtorch_stable/cache_kernels.hg",
    )
    moe_sources = ("csrc/upstream/libtorch_stable/moe/topk_softmax_kernels.hg",)
    defines = toolchain.cmake_defines(sources, moe_sources, {"PATH": nowhere})
    assert defines["PPU_UPSTREAM_EXT_SRC"] == ";".join(sources)
    assert defines["PPU_UPSTREAM_MOE_EXT_SRC"] == ";".join(moe_sources)


def test_cmake_defines_empty_source_lists_are_empty_strings(nowhere: str) -> None:
    defines = toolchain.cmake_defines((), (), {"PATH": nowhere})
    assert defines["PPU_UPSTREAM_EXT_SRC"] == ""
    assert defines["PPU_UPSTREAM_MOE_EXT_SRC"] == ""


def test_cmake_defines_passes_through_toolchain_locations(tmp_path: Path) -> None:
    hgcc = _exe(tmp_path / "bin" / "hgcc")
    env = {
        "PATH": str(tmp_path / "bin"),
        "HGCC": hgcc,
        "PPU_SDK": str(tmp_path / "sdk"),
        "CMAKE_HGCC_DIR": str(tmp_path / "cmake-hgcc"),
        "VLLM_SAIL_HG_FLAGS": " -dlto ",
    }
    defines = toolchain.cmake_defines(("a.cu",), (), env)
    assert defines["CMAKE_HG_COMPILER"] == hgcc
    assert defines["CMAKE_HG_FLAGS"] == "-dlto"
    assert defines["CMAKE_HGCC_DIR"] == str(tmp_path / "cmake-hgcc")
    assert defines["HGGCToolkit_ROOT"] == str(tmp_path / "sdk")


def test_cmake_defines_carry_nothing_from_the_nvcc_era(tmp_path: Path) -> None:
    """The regression guard for the whole phase: no CUDA reaches CMake.

    A leftover ``TORCH_CUDA_ARCH_LIST`` or ``CMAKE_CUDA_*`` would mean the build
    is still asking for nvcc semantics that hgcc does not provide.
    """
    env = {
        "PATH": str(tmp_path / "bin"),
        "HGCC": _exe(tmp_path / "bin" / "hgcc"),
        "PPU_SDK": str(tmp_path / "sdk"),
        "CMAKE_HGCC_DIR": str(tmp_path / "cmake-hgcc"),
        "VLLM_SAIL_HG_ARCH": "ppu_10",
        "VLLM_SAIL_HG_STD": "17",
        "VLLM_SAIL_HG_FLAGS": "-dlto",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "CUDA_HOME": "/usr/local/cuda",
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        "VLLM_PPU_NVCC_STD": "17",
        "VLLM_PPU_NVCC_FLAGS": "-O3",
    }
    defines = toolchain.cmake_defines(("a.cu",), ("b.cu",), env)
    assert "TORCH_CUDA_ARCH_LIST" not in defines
    assert "CUDA_HOME" not in defines
    assert [key for key in defines if key.startswith("CMAKE_CUDA")] == []
    assert [key for key in defines if "CUDA" in key or "NVCC" in key] == []


def test_old_ppu_build_variables_are_ignored(nowhere: str) -> None:
    env = {
        "PATH": nowhere,
        "VLLM_PPU_SKIP_EXT": "1",
        "VLLM_PPU_HG_ARCH": "ppu_10",
        "VLLM_PPU_HG_STD": "17",
        "VLLM_PPU_HG_FLAGS": "-dlto",
    }
    assert not toolchain.probe(env).skip_requested
    defines = toolchain.cmake_defines((), (), env)
    assert defines["CMAKE_HG_ARCHITECTURES"] == toolchain.DEFAULT_HG_ARCH
    assert defines["PYTORCH_SAIL_ARCH"] == toolchain.DEFAULT_HG_ARCH
    assert defines["CMAKE_HG_STANDARD"] == toolchain.DEFAULT_HG_STANDARD
    assert "CMAKE_HG_FLAGS" not in defines


@pytest.mark.parametrize("value", ["", " \t "])
def test_empty_build_settings_use_defaults(nowhere: str, value: str) -> None:
    defines = toolchain.cmake_defines(
        (),
        (),
        {
            "PATH": nowhere,
            "VLLM_SAIL_HG_ARCH": value,
            "VLLM_SAIL_HG_STD": value,
            "VLLM_SAIL_HG_FLAGS": value,
        },
    )
    assert defines["CMAKE_HG_ARCHITECTURES"] == toolchain.DEFAULT_HG_ARCH
    assert defines["PYTORCH_SAIL_ARCH"] == toolchain.DEFAULT_HG_ARCH
    assert defines["CMAKE_HG_STANDARD"] == toolchain.DEFAULT_HG_STANDARD
    assert "CMAKE_HG_FLAGS" not in defines


def test_build_parallelism_honours_explicit_setuptools_j() -> None:
    # An explicit ``setup.py build_ext -j N`` must win over everything else.
    jobs = toolchain.build_parallelism(4, env={"MAX_JOBS": "8"}, cpu_count=64)
    assert jobs == 4


def test_build_parallelism_uses_max_jobs_env() -> None:
    # Without an explicit -j, MAX_JOBS caps the cmake --parallel level so a
    # many-core, smaller-memory runner does not OOM (exit 137).
    jobs = toolchain.build_parallelism(None, env={"MAX_JOBS": "8"}, cpu_count=64)
    assert jobs == 8


def test_build_parallelism_falls_back_to_cpu_count() -> None:
    jobs = toolchain.build_parallelism(None, env={}, cpu_count=64)
    assert jobs == 64


@pytest.mark.parametrize("value", ["0", "-1", "abc", "", " "])
def test_build_parallelism_ignores_invalid_max_jobs(value: str) -> None:
    jobs = toolchain.build_parallelism(None, env={"MAX_JOBS": value}, cpu_count=64)
    assert jobs == 64


def test_build_parallelism_defaults_to_one_when_unknown() -> None:
    jobs = toolchain.build_parallelism(None, env={}, cpu_count=None)
    assert jobs == 1
