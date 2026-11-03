# SPDX-License-Identifier: Apache-2.0
"""Host compile regressions without torch, vLLM, or a PPU SDK."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        pytest.skip(f"optional host compile regression requires {name}")
    return found


def _run(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _write_native_sdk_stubs(sdk: Path) -> None:
    """Headers ppu_compat.h takes directly on the HG toolchain."""
    sdk.mkdir(exist_ok=True)
    for header, body in {
        "hggc.h": "#pragma once\n",
        "acblas_v2.h": "#pragma once\n",
        "hggc_fp16.h": "#pragma once\nstruct __ppu_half { unsigned short x; };\n",
        "hggc_bf16.h": (
            "#pragma once\n"
            "struct __ppu_bfloat16 { unsigned short x; };\n"
            "struct __ppu_bfloat162 { unsigned short x, y; };\n"
            "struct __ppu_bfloat16_raw { unsigned short x; };\n"
            "struct __ppu_bfloat162_raw { unsigned short x, y; };\n"
        ),
        "hggc_runtime_api.h": "#pragma once\n",
        "hggc_runtime_api.hpp": "#pragma once\n",
    }.items():
        (sdk / header).write_text(body, encoding="utf-8")


@pytest.mark.parametrize("cuda_checks_available", [True, False])
def test_stable_abi_macros_preserve_torch_error_checking(
    tmp_path: Path, cuda_checks_available: bool
) -> None:
    """Exercise wrapper include order, error checks and older torch headers."""
    compiler = _tool("c++")
    torch_include = tmp_path / "torch_include"
    header = torch_include / "torch/csrc/stable/macros.h"
    header.parent.mkdir(parents=True)
    header.write_text(
        r"""
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#define TORCH_STABLE_HEADER_SEEN 1
#define STABLE_TORCH_ERROR_CODE_CHECK(EXPR) (EXPR)
#if TEST_CUDA_CHECKS_AVAILABLE
// Fail if either original macro survives instead of the PPU adaptation.
#define STD_CUDA_CHECK(EXPR) static_assert(false, "unadapted CUDA check")
#define STD_CUDA_KERNEL_LAUNCH_CHECK() static_assert(false, "unadapted launch check")
#endif
""",
        encoding="utf-8",
    )
    unit = tmp_path / "stable_macros.cpp"
    unit.write_text(
        r"""
#include <cassert>
#include <cstring>
#include <torch/csrc/stable/macros.h>
static_assert(TORCH_STABLE_HEADER_SEEN == 1);
static_assert(STABLE_TORCH_ERROR_CODE_CHECK(42) == 42);

#if TEST_CUDA_CHECKS_AVAILABLE
enum cudaError_t { cudaSuccess = 0, cudaFailure = 7 };
enum hggcError_t { hggcSuccess = 0, hggcFailure = 8 };
int evaluations = 0, checks = 0, frees = 0, launches = 0;
int last_error = 0;
uint32_t expected_line = 0;
const char* expected_function = nullptr;
char* allocated_message = nullptr;
constexpr char error_message[] = "mock runtime error";
hggcError_t launch_error = hggcSuccess;

void torch_c10_cuda_check_msg(int32_t error, const char* file,
                            const char* function, uint32_t line,
                            bool device_assertions, char** message) {
  assert(std::strcmp(file, __FILE__) == 0);
  assert(std::strcmp(function, expected_function) == 0);
  assert(line == expected_line && device_assertions);
  assert(*message == nullptr && allocated_message == nullptr);
  ++checks;
  last_error = error;
  if (error != 0) {
    allocated_message = new char[sizeof(error_message)];
    std::memcpy(allocated_message, error_message, sizeof(error_message));
    *message = allocated_message;
  }
}
void torch_c10_cuda_free_error_msg(char* message) {
  assert(message != nullptr && message == allocated_message);
  ++frees;
  delete[] message;
  allocated_message = nullptr;
}
hggcError_t hggcGetLastError() {
  ++launches;
  return launch_error;
}
template <typename Error> Error evaluate(Error error) {
  ++evaluations;
  return error;
}
template <typename Error> void check(Error success, Error failure) {
  expected_function = __func__;
  const int previous_checks = checks, previous_frees = frees;
  const int previous_evaluations = evaluations;
  expected_line = __LINE__ + 1;
  if (true) STD_CUDA_CHECK(evaluate(success)); else assert(false);
  assert(checks == previous_checks + 1 && frees == previous_frees);
  assert(last_error == 0 && allocated_message == nullptr);
  assert(evaluations == previous_evaluations + 1);
  try {
    expected_line = __LINE__ + 1;
    STD_CUDA_CHECK(evaluate(failure));
    assert(false && "error must propagate");
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == error_message);
    assert(frees == previous_frees + 1 && allocated_message == nullptr);
  }
  assert(last_error == static_cast<int32_t>(failure));
  assert(checks == previous_checks + 2);
  assert(evaluations == previous_evaluations + 2);
}
int main() {
  check(cudaSuccess, cudaFailure);
  check(hggcSuccess, hggcFailure);
  expected_function = __func__;
  expected_line = __LINE__ + 1;
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  assert(launches == 1 && checks == 5 && frees == 2 && last_error == 0);
  launch_error = hggcFailure;
  try {
    expected_line = __LINE__ + 1;
    STD_CUDA_KERNEL_LAUNCH_CHECK();
    assert(false && "launch error must propagate");
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == error_message);
    assert(frees == 3 && allocated_message == nullptr);
  }
  assert(launches == 2 && checks == 6 && last_error == 8);
}
#else
#ifdef STD_CUDA_CHECK
#error CUDA checks must remain unavailable when torch does not define them
#endif
#ifdef STD_CUDA_KERNEL_LAUNCH_CHECK
#error CUDA launch checks must remain unavailable when torch does not define them
#endif
int main() {}
#endif
""",
        encoding="utf-8",
    )
    executable = tmp_path / "stable_macros"
    _run(
        [
            compiler,
            "-std=c++17",
            f"-DTEST_CUDA_CHECKS_AVAILABLE={int(cuda_checks_available)}",
            "-I",
            str(ROOT / "csrc"),
            "-I",
            str(torch_include),
            str(unit),
            "-o",
            str(executable),
        ]
    )
    _run([str(executable)])


def test_gdn_native_binding_guards_are_enabled_by_the_build(tmp_path):
    """Preprocess shipped bindings with the target's actual build definitions."""
    compiler = _tool("c++")
    source = (ROOT / "CMakeLists.txt").read_text()
    definitions = re.findall(
        r"target_compile_definitions\(_upstream_C\s+PRIVATE\s+([^)]*)\)", source
    )
    flags = [f"-D{item}" for definition in definitions for item in definition.split()]
    for filename in ("torch_bindings.cpp", "ops.h"):
        unit = tmp_path / filename
        body = (ROOT / "csrc/upstream/libtorch_stable" / filename).read_text()
        unit.write_text(re.sub(r"^\s*#\s*include[^\n]*", "", body, flags=re.MULTILINE))
        result = subprocess.run(
            [compiler, "-E", "-P", "-x", "c++", *flags, str(unit)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "fused_gdn_decode_post_conv_mtp" in result.stdout
        if filename == "torch_bindings.cpp":
            assert 'ops.impl("fused_gdn_decode_post_conv_mtp"' in result.stdout


def test_gdn_hg_helpers_commit_and_wait_for_pipeline_copies(tmp_path):
    """Exercise generated adapters against a deferred-copy SDK test double."""
    compiler = _tool("c++")
    source = (
        ROOT / "csrc/upstream/libtorch_stable/gdn/fused_gdn_decode_kernel.hg"
    ).read_text()
    # Retain the actual SDK pipeline include and its guard, without unrelated
    # device/torch headers. The host test does not establish SDK compatibility.
    includes = source[: source.index("namespace {")]
    includes = re.sub(
        r"^#include (?!<hggc_pipeline_primitives\.h>)[^\n]*",
        "",
        includes,
        flags=re.MULTILINE,
    )
    start = source.index("template <typename StateT>")
    end = source.index("template <typename StateT>\n__device__ __forceinline__ float4")
    (tmp_path / "hggc_pipeline_primitives.h").write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
struct Copy { void* dst; const void* src; };
std::vector<Copy> pending, committed;
std::string events;
void __pipeline_memcpy_async(void* dst, const void* src, std::size_t bytes) {
  assert(bytes == 16);
  pending.push_back({dst, src});
  events += 'i';
}
void __pipeline_commit() {
  committed.insert(committed.end(), pending.begin(), pending.end());
  pending.clear();
  events += 'c';
}
void __pipeline_wait_prior(int remaining) {
  assert(remaining == 0);
  assert(pending.empty());
  for (const auto& copy : committed) std::memcpy(copy.dst, copy.src, 16);
  committed.clear();
  events += 'w';
}
"""
    )
    checks = r"""
template <typename StateT> void check() {
  alignas(16) StateT src[16 / sizeof(StateT)], dst[16 / sizeof(StateT)] = {};
  for (unsigned i = 0; i < 16 / sizeof(StateT); ++i) src[i] = StateT(i + 1);
  events.clear();
  cp_async_16b(dst, src);
  assert(dst[0] == 0 && events == "i");
  cp_async_commit();
  assert(dst[0] == 0 && events == "ic" && pending.empty());
  cp_async_wait_all();
  assert(std::memcmp(dst, src, 16) == 0 && events == "iccw");

  // PTX wait_all implicitly commits outstanding, uncommitted copies too.
  std::memset(dst, 0, 16);
  events.clear();
  cp_async_16b(dst, src);
  cp_async_wait_all();
  assert(std::memcmp(dst, src, 16) == 0 && events == "icw");
}
int main() { check<float>(); check<uint16_t>(); }
"""
    unit = tmp_path / "gdn_pipeline.cpp"
    unit.write_text(
        includes
        + "\n#define __device__\n#define __forceinline__ inline\n"
        + source[start:end]
        + checks
    )
    executable = tmp_path / "gdn_pipeline"
    _run(
        [
            compiler,
            "-std=c++17",
            "-D__HGGCCC__=1",
            "-I",
            str(tmp_path),
            str(unit),
            "-o",
            str(executable),
        ]
    )
    _run([str(executable)])


@pytest.mark.parametrize(
    ("environment", "settings", "expected"),
    [
        ({}, "", "ppu_15;ppu_10"),
        ({"PYTORCH_SAIL_ARCH": "ppu_15;ppu_10"}, "", "ppu_15;ppu_10"),
        pytest.param(
            {"VLLM_PPU_HG_ARCH": "ppu_10"},
            "",
            "ppu_15;ppu_10",
            id="old-ppu-build-variable-is-ignored",
        ),
        ({"VLLM_SAIL_HG_ARCH": "ppu_10"}, "", "ppu_10"),
        (
            {"VLLM_SAIL_HG_ARCH": " ppu_15;ppu_15 "},
            "",
            "ppu_15",
        ),
        (
            {"VLLM_SAIL_HG_ARCH": ""},
            "",
            "ppu_15;ppu_10",
        ),
        (
            {"VLLM_SAIL_HG_ARCH": " ; \t "},
            "",
            "ppu_15;ppu_10",
        ),
        (
            {
                "PYTORCH_SAIL_ARCH": "ppu_10",
                "VLLM_SAIL_HG_ARCH": "",
            },
            "",
            "ppu_10",
        ),
        ({}, 'set(CMAKE_HG_ARCHITECTURES "ppu_10;ppu_15")', "ppu_10;ppu_15"),
        (
            {"VLLM_SAIL_HG_ARCH": "ppu_10"},
            'set(CMAKE_HG_ARCHITECTURES "ppu_10;ppu_15")',
            "ppu_10;ppu_15",
        ),
    ],
)
def test_cmake_selects_architectures_before_enabling_hg(
    tmp_path, environment, settings, expected
):
    cmake = _tool("cmake")
    # Pass real environment values: CMake's set(ENV{...} "") unsets a variable
    # instead of representing an explicitly empty value inherited from a shell.
    env = os.environ.copy()
    for name in ("PYTORCH_SAIL_ARCH", "VLLM_SAIL_HG_ARCH", "VLLM_PPU_HG_ARCH"):
        env.pop(name, None)
    env.update(environment)
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    start = source.index("if(NOT CMAKE_HG_ARCHITECTURES)")
    end = source.index("project(vllm_sail_native", start)
    script = tmp_path / "architectures.cmake"
    script.write_text(
        settings
        + "\n"
        + source[start:end]
        + f'\nif(NOT "${{CMAKE_HG_ARCHITECTURES}}" STREQUAL "{expected}" '
        + f'OR NOT "${{PYTORCH_SAIL_ARCH}}" STREQUAL "{expected}")\n'
        + 'message(FATAL_ERROR "HG and Torch architectures must match the request")\n'
        + "endif()\n",
        encoding="utf-8",
    )
    _run([cmake, "-P", str(script)], env=env)


def test_grouped_topk_finite_check_disambiguates_sdk_scalars(tmp_path: Path) -> None:
    compiler = _tool("c++")
    source = (
        ROOT / "csrc/upstream/libtorch_stable/moe/grouped_topk_kernels.hg"
    ).read_text(encoding="utf-8")
    start = source.index("template <typename T_OUT, typename T_IN>")
    end = source.index("// Scoring function enums", start)
    # SDK half types convert to both float and double; the math library has
    # overloads for both. Compile the actual generated helper and test NaN/Inf.
    prelude = r"""
#include <cassert>
#include <cmath>
#include <limits>
#define __device__
#define __COMPATIBLECC_VER_MAJOR__ 13
#define __COMPATIBLECC_VER_MINOR__ 0
namespace hggc { namespace std = ::std; }
using std::isfinite;
struct Half {
  float value;
  operator float() const { return value; }
  operator double() const { return value; }
};
struct __ppu_bfloat16 : Half {};
float __bfloat162float(__ppu_bfloat16 value) { return value.value; }
"""
    checks = r"""
int main() {
  for (float x : {0.0f, -1.0f, 65504.0f, INFINITY, -INFINITY, NAN}) {
    assert(is_finite(x) == std::isfinite(x));
    assert(is_finite(Half{x}) == std::isfinite(x));
    assert(is_finite(__ppu_bfloat16{{x}}) == std::isfinite(x));
  }
}
"""
    unit = tmp_path / "finite.cpp"
    unit.write_text(
        "#include <initializer_list>\n" + prelude + source[start:end] + checks,
        encoding="utf-8",
    )
    executable = tmp_path / "finite"
    _run([compiler, "-std=c++20", str(unit), "-o", str(executable)])
    _run([str(executable)])


@pytest.mark.parametrize("arch", [800, 890])
@pytest.mark.parametrize(
    "model", ["minimax_m3_qknorm_rope_kv_insert", "kimi_k3_mla_key_concat_kv_cache"]
)
def test_model_fused_kernels_use_ordinary_hg_launches(tmp_path, arch, model):
    """Preprocess actual generated sources for both PPU compatibility targets."""
    compiler = _tool("c++")
    source = (
        ROOT / f"csrc/upstream/libtorch_stable/fused_{model}_kernel.hg"
    ).read_text()
    # The preprocessor needs no SDK definitions to evaluate the launch guards.
    source = re.sub(r"^\s*#\s*include[^\n]*", "", source, flags=re.MULTILINE)
    unit = tmp_path / "launch.cpp"
    unit.write_text(source)
    result = subprocess.run(
        [
            compiler,
            "-E",
            "-P",
            "-x",
            "c++",
            "-D__HGGCCC__=1",
            f"-DCOMPATIBLE_ARCH={arch}",
            str(unit),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "<<<grid, kBlockSize, 0, stream>>>" in result.stdout
    assert "hggcLaunchKernelEx(" not in result.stdout
    assert "hggcGridDependencySynchronize(" not in result.stdout
    assert "hggcTriggerProgrammaticLaunchCompletion(" not in result.stdout


def test_mamba_scan_composition_and_prefix_carry(tmp_path):
    """Compile the real scan functors against a worked recurrence example."""
    compiler = _tool("c++")
    source = (ROOT / "csrc/upstream/libtorch_stable/mamba/selective_scan.h").read_text()
    start = source.index("template<typename scalar_t> struct SSMScanOp;")
    end = source.index("template<typename Ktraits>", start)
    unit = tmp_path / "scan.cpp"
    unit.write_text(
        "#include <cassert>\n#include <type_traits>\n"
        "#define __device__ constexpr\n#define __forceinline__ inline\n"
        "struct float2 { float x, y; }; struct float4 { float x, y, z, w; };\n"
        "constexpr float2 make_float2(float x, float y) { return {x, y}; }\n"
        + source[start:end]
        + """
constexpr bool check_scan() {
  SSMScanOp<float> op;
  auto ab = op(op({2, 3}, {5, 7}), {11, 13});
  if (ab.x != 110 || ab.y != 255) return false;
  SSMScanPrefixCallbackOp<float> callback({1, 17});
  auto old = callback(ab);
  if (old.x != 1 || old.y != 17) return false;
  if (callback.running_prefix.y != 2125) return false;
  old = callback({0.5f, -1});
  return old.y == 2125 && callback.running_prefix.y == 1061.5f;
}
static_assert(check_scan());
"""
    )
    _run([compiler, "-std=c++17", "-fsyntax-only", str(unit)])


@pytest.mark.parametrize("arch", [800, 890])
def test_fused_quant_and_qk_helpers_compile_without_ptx(tmp_path, arch):
    """Type-check generated HG helper branches; numerical tests need a PPU."""
    compiler = _tool("c++")
    root = ROOT / "csrc/upstream/libtorch_stable"
    quant = (root / "quantization/fused_kernels/quant_conversions.cuh").read_text()
    start = quant.index("static __device__ __forceinline__ int8_t float_to_int8_rn")
    end = quant.index("template <typename fp8_type>", start)
    copies = (root / "async_util.cuh").read_text()
    unit = tmp_path / "fused_helpers.cpp"
    unit.write_text(
        "#include <cstdint>\n#include <cmath>\nusing std::isnan;\n"
        "#define __device__\n#define __forceinline__ inline\n"
        "struct alignas(16) int4 { int x, y, z, w; };\n"
        "void __syncwarp() {}\n"
        + quant[start:end]
        + copies
        + """
int8_t compile_helpers(float input) {
  using namespace vllm::cuda_async;
  alignas(16) uint32_t src[] = {10, 20, 30, 40}, dst[] = {0, 0, 0, 0};
  cp_async_shared_global_ca(dst, src, 4);
  cp_async_shared_global_ca(dst, src, 8);
  cp_async_shared_global_ca(dst, src, 16);
  cp_async_shared_global_16_cg(dst, src);
  cp_async_commit_group();
  cp_async_wait_group<1>(); cp_async_wait_group<0>();
  return float_to_int8_rn(input);
}
"""
    )
    defines = ["-D__HGGCCC__=1", f"-DCOMPATIBLE_ARCH={arch}"]
    preprocessed = subprocess.run(
        [compiler, "-E", "-P", *defines, str(unit)], capture_output=True, text=True
    )
    assert preprocessed.returncode == 0, preprocessed.stderr
    assert "cvt.rni.sat.s8.f32" not in preprocessed.stdout
    assert '"cp.async.' not in preprocessed.stdout
    _run([compiler, "-std=c++17", "-fsyntax-only", *defines, str(unit)])


@pytest.mark.parametrize(
    ("hg_arch", "expected_arch"),
    [(100, 800), (150, 890), (200, 200)],
)
def test_ppu_compat_maps_hggc_arches(
    tmp_path: Path, hg_arch: int, expected_arch: int
) -> None:
    """Pin the arch mapping that used to come from the torch wheel."""
    compiler = _tool("c++")
    sdk = tmp_path / "sdk"
    _write_native_sdk_stubs(sdk)
    unit = tmp_path / "arch.cpp"
    unit.write_text(
        '#include "ppu_compat.h"\n'
        f"static_assert(COMPATIBLE_ARCH == {expected_arch});\n"
        f"static_assert(COMPATIBLE_ARCH_LIST == {expected_arch});\n",
        encoding="utf-8",
    )
    _run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "csrc"),
            "-I",
            str(sdk),
            "-D__HGGCCC__=1",
            f"-D__HGGC_ARCH__={hg_arch}",
            "-fsyntax-only",
            str(unit),
        ]
    )


@pytest.mark.parametrize(("hg_arch", "expected_arch"), [(100, 800), (150, 890)])
def test_ppu_compat_supplies_cuda_arch_on_the_device_pass(
    tmp_path: Path, hg_arch: int, expected_arch: int
) -> None:
    """Stock torch device branches compile per-arch under ``__CUDA_ARCH__``.

    The hgcc device pass defines ``__HGGC_ARCH__`` but not ``__CUDA_ARCH__``;
    the compat header must derive the NVIDIA spelling from it.
    """
    compiler = _tool("c++")
    sdk = tmp_path / "sdk"
    _write_native_sdk_stubs(sdk)
    unit = tmp_path / "device_emulation.cpp"
    unit.write_text(
        '#include "ppu_compat.h"\n'
        "#ifndef __CUDACC__\n"
        "#error both driver passes define __CUDACC__\n"
        "#endif\n"
        f"static_assert(__CUDA_ARCH__ == {expected_arch});\n",
        encoding="utf-8",
    )
    _run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "csrc"),
            "-I",
            str(sdk),
            "-D__HGGCCC__=1",
            f"-D__HGGC_ARCH__={hg_arch}",
            "-fsyntax-only",
            str(unit),
        ]
    )


def test_ppu_compat_supplies_cudacc_but_not_cuda_arch_on_the_host_pass(
    tmp_path: Path,
) -> None:
    """nvcc defines ``__CUDACC__`` in both passes and ``__CUDA_ARCH__`` in none.

    Stock torch headers key on the first; per-arch device code keys on the
    second, which must stay undefined wherever ``__HGGC_ARCH__`` is absent.
    """
    compiler = _tool("c++")
    sdk = tmp_path / "sdk"
    _write_native_sdk_stubs(sdk)
    unit = tmp_path / "host_emulation.cpp"
    unit.write_text(
        '#include "ppu_compat.h"\n'
        "#ifndef __CUDACC__\n"
        "#error stock torch device branches require __CUDACC__\n"
        "#endif\n"
        "#ifdef __CUDA_ARCH__\n"
        "#error __CUDA_ARCH__ must stay undefined without __HGGC_ARCH__\n"
        "#endif\n"
        "static_assert(__CUDACC_VER_MAJOR__ == 13);\n"
        "static_assert(__CUDACC_VER_MINOR__ == 0);\n"
        "static_assert(__CUDACC_VER_BUILD__ == 0);\n",
        encoding="utf-8",
    )
    _run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "csrc"),
            "-I",
            str(sdk),
            "-D__HGGCCC__=1",
            "-fsyntax-only",
            str(unit),
        ]
    )


def test_ppu_compat_resolves_stock_torch_cuda_headers(tmp_path: Path) -> None:
    """Stock torch requests CUDA scalar headers and the driver version macro."""
    compiler = _tool("c++")
    sdk = tmp_path / "sdk"
    _write_native_sdk_stubs(sdk)
    unit = tmp_path / "scalars.cpp"
    unit.write_text(
        '#include "ppu_compat.h"\n'
        "#include <cuda_fp16.h>\n"
        "#include <cuda_bf16.h>\n"
        "#include <cuda.h>\n"
        "static_assert(CUDA_VERSION == COMPATIBLE_VERSION);\n"
        "#ifndef __CUDA_FP16_H__\n"
        "#error NVIDIA cuda_fp16.h would redefine the native half types\n"
        "#endif\n"
        "#ifndef __CUDA_BF16_H__\n"
        "#error NVIDIA cuda_bf16.h would redefine the native bf16 types\n"
        "#endif\n"
        "static_assert(sizeof(__nv_bfloat16) == sizeof(__ppu_bfloat16));\n"
        "static_assert(sizeof(__nv_bfloat162) == sizeof(__ppu_bfloat162));\n"
        "static_assert(sizeof(__nv_bfloat16_raw) == sizeof(__ppu_bfloat16_raw));\n"
        "static_assert(sizeof(__nv_bfloat162_raw) == sizeof(__ppu_bfloat162_raw));\n",
        encoding="utf-8",
    )
    _run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "csrc"),
            "-I",
            str(sdk),
            "-D__HGGCCC__=1",
            "-fsyntax-only",
            str(unit),
        ]
    )


def test_ppu_compat_leaves_the_arch_undefined_on_the_host_pass(
    tmp_path: Path,
) -> None:
    compiler = _tool("c++")
    unit = tmp_path / "host.cpp"
    unit.write_text(
        '#include "ppu_compat.h"\n'
        "#ifdef COMPATIBLE_ARCH\n"
        "#error COMPATIBLE_ARCH must mirror the device-only __CUDA_ARCH__\n"
        "#endif\n"
        "#ifdef COMPATIBLE_ARCH_LIST\n"
        "#error COMPATIBLE_ARCH_LIST must mirror __CUDA_ARCH_LIST__\n"
        "#endif\n"
        "#ifdef __CUDACC__\n"
        "#error __CUDACC__ requires the hgcc driver\n"
        "#endif\n"
        "#ifdef __CUDA_ARCH__\n"
        "#error __CUDA_ARCH__ requires the hgcc device pass\n"
        "#endif\n"
        "static_assert(COMPATIBLE_VERSION == 13000);\n"
        "static_assert(COMPATIBLE_RT_VERSION == 13000);\n"
        "#if __COMPATIBLECC_VER_MAJOR__ < 11\n"
        "#error The corpus takes its CUDA >= 11 paths unconditionally\n"
        "#endif\n"
        "#if __COMPATIBLECC_VER_MAJOR__ * 10000 + "
        "__COMPATIBLECC_VER_MINOR__ * 100 < 120800\n"
        "#error grouped_topk_kernels.hg expects the CUDA >= 12.8 path\n"
        "#endif\n",
        encoding="utf-8",
    )
    _run([compiler, "-std=c++17", "-I", str(ROOT / "csrc"), "-fsyntax-only", str(unit)])


def test_ppu_compat_covers_upstream_headers_that_omit_their_includes() -> None:
    """cuda_utils.h uses int64_t and std::enable_if_t with only <stdio.h>."""
    compiler = _tool("c++")
    _run(
        [
            compiler,
            "-std=c++17",
            "-x",
            "c++",
            "-I",
            str(ROOT / "csrc"),
            "-include",
            "ppu_compat.h",
            "-fsyntax-only",
            str(ROOT / "csrc/upstream/cuda_utils.h"),
        ]
    )


def test_ppu_compat_supplies_the_sdk_types_cuda_utils_kernels_expands(
    tmp_path: Path,
) -> None:
    """The TU whose CUDA_CHECK expansion failed on hgcc without the torch header."""
    compiler = _tool("c++")
    sdk = tmp_path / "sdk"
    _write_native_sdk_stubs(sdk)
    (sdk / "hggc.h").write_text(
        "#pragma once\n"
        "typedef enum hggcError { hggcSuccess = 0 } hggcError_t;\n"
        "typedef enum { hggcDevAttrMaxSharedMemoryPerBlockOptin } hggcDeviceAttr;\n"
        "const char* hggcGetErrorString(hggcError_t e);\n"
        "hggcError_t hggcGetDevice(int* device);\n"
        "hggcError_t hggcDeviceGetAttribute(int* value, hggcDeviceAttr attr, "
        "int device);\n",
        encoding="utf-8",
    )
    argv = [
        compiler,
        "-std=c++17",
        "-x",
        "c++",
        "-D__HGGCCC__=1",
        "-D__host__=",
        "-D__device__=",
        "-D__forceinline__=inline",
        "-I",
        str(ROOT / "csrc"),
        "-I",
        str(ROOT / "csrc/upstream"),
        "-I",
        str(sdk),
        "-include",
        "ppu_compat.h",
        "-fsyntax-only",
        str(ROOT / "csrc/upstream/libtorch_stable/cuda_utils_kernels.hg"),
    ]
    _run(argv)
    # On the HG toolchain the umbrella is unconditional, like the torch
    # header it replaces: it must not fall back to an __has_include probe.
    bare = [
        arg
        for index, arg in enumerate(argv)
        if not (arg == "-I" and argv[index + 1] == str(sdk)) and arg != str(sdk)
    ]
    result = subprocess.run(bare, capture_output=True, text=True)
    assert result.returncode != 0
    assert "hggc.h" in result.stderr


@pytest.mark.parametrize("hg_toolchain", [True, False], ids=["hgcc", "host-probe"])
def test_ppu_compat_keeps_the_sdk_umbrella_when_the_sdk_is_visible(
    tmp_path: Path, hg_toolchain: bool
) -> None:
    """upstream torch_utils.h names acblasHandle_t without including acblas.

    The umbrella must not depend on __has_include: hgcc does not reliably
    evaluate it, so the HG compiler marker includes it unconditionally.
    """
    compiler = _tool("c++")
    sdk = tmp_path / "sdk"
    _write_native_sdk_stubs(sdk)
    (sdk / "hggc.h").write_text(
        "#pragma once\n#define HGGC_FROM_SDK 1\n", encoding="utf-8"
    )
    (sdk / "acblas_v2.h").write_text(
        "#pragma once\n#define ACBLAS_FROM_SDK 1\n", encoding="utf-8"
    )
    unit = tmp_path / "umbrella.cpp"
    unit.write_text(
        '#include "ppu_compat.h"\n'
        "static_assert(HGGC_FROM_SDK == 1);\n"
        "static_assert(ACBLAS_FROM_SDK == 1);\n",
        encoding="utf-8",
    )
    argv = [compiler, "-std=c++17", "-I", str(ROOT / "csrc"), "-I", str(sdk)]
    if hg_toolchain:
        argv.append("-D__HGGCCC__=1")
    argv += ["-fsyntax-only", str(unit)]
    _run(argv)


@pytest.mark.parametrize("torch_force_include", [False, True])
@pytest.mark.parametrize("hg_standard", [17, 20])
def test_extension_options_survive_torch_config(
    tmp_path: Path, torch_force_include: bool, hg_standard: int
) -> None:
    cmake = _tool("cmake")
    compiler = _tool("c++")
    torch_include = tmp_path / "torch include"
    torch_include.mkdir()
    (torch_include / "torch_force_include.h").write_text(
        "#pragma once\n#define TORCH_FORCE_INCLUDED 1\n", encoding="utf-8"
    )
    # Model the SDK namespace contract. Compile
    # our real compatibility headers, with no SDK or torch dependency in the gate.
    csrc = tmp_path / "csrc"
    (csrc / "cub").mkdir(parents=True)
    (csrc / "cub/util_namespace.cuh").write_text(
        "#pragma once\n"
        "#ifdef CUB_WRAPPED_NAMESPACE\n"
        "namespace CUB_WRAPPED_NAMESPACE { namespace cub { struct Scan {}; } }\n"
        "#define CUB_NS_QUALIFIER ::CUB_WRAPPED_NAMESPACE::cub\n"
        "#else\nnamespace cub { struct Scan {}; }\n#endif\n",
        encoding="utf-8",
    )
    for header in ("ppu_kernel_compat.h", "ppu_compat.h"):
        shutil.copyfile(ROOT / "csrc" / header, csrc / header)
    # Plugin headers must win over same-named upstream headers.
    for folder in ("plugin", "upstream"):
        (csrc / folder).mkdir()
        (csrc / folder / "owned_header.h").write_text(
            "#define PLUGIN_HEADER 1\n"
            if folder == "plugin"
            else '#error "upstream header shadows the plugin header"\n'
        )
    torch_gate = (
        "#ifndef TORCH_FORCE_INCLUDED\n"
        "#error Torch's force-include was dropped\n"
        "#endif\n"
        if torch_force_include
        else ""
    )
    (tmp_path / "binding.cpp").write_text(
        torch_gate + "static_assert(COMPATIBLE_VERSION == 13000);\n"
        "static_assert(__cplusplus == 202002L);\n"
        "#ifdef CUB_NS_QUALIFIER\n#error HG headers leaked into CXX\n#endif\n",
        encoding="utf-8",
    )
    (tmp_path / "kernel.hg").write_text(
        '#include "owned_header.h"\n'
        "static_assert(PLUGIN_HEADER == 1);\n"
        "#include <cub/util_namespace.cuh>\n"
        "using Scan = cub::Scan;\n"
        "#ifndef ENABLE_FP8\n#error FP8 conversions are disabled\n#endif\n"
        "#if defined(__HGGC_NO_HALF_OPERATORS__) || defined(__HGGC_NO_HALF_CONVERSIONS__)"
        " || defined(__HGGC_NO_HALF2_OPERATORS__)"
        " || defined(__HGGC_NO_BFLOAT16_CONVERSIONS__)\n"
        "#error Upstream kernels require SDK scalar operators and conversions\n#endif\n"
        + torch_gate
        + "static_assert(COMPATIBLE_VERSION == 13000);\n"
        f"static_assert(__cplusplus == {201703 if hg_standard == 17 else 202002}L);\n",
        encoding="utf-8",
    )
    # A minimal HG language shim routes *host-only* test sources through c++.
    # CMake still evaluates COMPILE_LANGUAGE:HG and orders flags itself; this
    # deliberately tests no cmake-hgcc or SDK implementation details.
    modules = tmp_path / "modules"
    modules.mkdir()
    compiler_config = f"""set(CMAKE_HG_COMPILER "{compiler}")
set(CMAKE_HG_COMPILER_ENV_VAR HGCC)
set(CMAKE_HG_COMPILER_LOADED TRUE)
set(CMAKE_HG_SOURCE_FILE_EXTENSIONS hg)
set(CMAKE_HG_OUTPUT_EXTENSION .o)
"""
    (modules / "CMakeDetermineHGCompiler.cmake").write_text(
        compiler_config
        + 'file(WRITE "${CMAKE_PLATFORM_INFO_DIR}/CMakeHGCompiler.cmake" [=['
        + compiler_config
        + "]=])\n",
        encoding="utf-8",
    )
    (modules / "CMakeTestHGCompiler.cmake").write_text(
        "set(CMAKE_HG_COMPILER_WORKS TRUE)\n", encoding="utf-8"
    )
    (modules / "CMakeHGInformation.cmake").write_text(
        """set(CMAKE_HG_COMPILE_OBJECT
  "<CMAKE_HG_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -x c++ -c <SOURCE> -o <OBJECT>")
set(CMAKE_INCLUDE_FLAG_HG "-I")
set(CMAKE_HG_INFORMATION_LOADED TRUE)
""",
        encoding="utf-8",
    )
    # Execute the production target function, with only the external Python,
    # torch and HGGC package discovery replaced. The real CMake generator and
    # compiler must preserve both -include/operand pairs from the failing log.
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    function = re.search(
        r"^function\(ppu_add_extension target\).*?^endfunction\(\)",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert function is not None
    torch_options = (
        "target_compile_options(torch INTERFACE -include "
        f'"{torch_include.as_posix()}/torch_force_include.h")\n'
        "target_compile_definitions(torch INTERFACE CUB_WRAPPED_NAMESPACE=at_cuda_detail)"
        if torch_force_include
        else ""
    )
    (tmp_path / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.26)
list(APPEND CMAKE_MODULE_PATH "{modules.as_posix()}")
project(ppu_option_regression LANGUAGES CXX HG)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HG_ARCHITECTURES "ppu_15;ppu_10")
set(CMAKE_HG_STANDARD {hg_standard})
# cmake-hgcc selects the requested standard before TorchConfig appends flags.
# Reproduce the log's later -std=c++17 overriding the requested C++20.
set(CMAKE_HG_FLAGS "-std=c++${{CMAKE_HG_STANDARD}} -std=c++17 -D__HGGC_NO_HALF_OPERATORS__ -D__HGGC_NO_HALF_CONVERSIONS__ -D__HGGC_NO_HALF2_OPERATORS__ -D__HGGC_NO_BFLOAT16_CONVERSIONS__")
set(PPU_EXT_OUTPUT_DIR "${{CMAKE_BINARY_DIR}}/extensions")
add_library(torch INTERFACE)
add_library(HGGC::toolkit INTERFACE IMPORTED)
{torch_options}
function(Python_add_library target kind soabi)
  add_library(${{target}} ${{kind}} ${{ARGN}})
endfunction()
{function.group()}
ppu_add_extension(_C INCLUDE_ROOT csrc/plugin SOURCES binding.cpp kernel.hg)
ppu_add_extension(_moe_C INCLUDE_ROOT csrc/plugin SOURCES binding.cpp kernel.hg)
foreach(target _C _moe_C)
  get_target_property(archs ${{target}} HG_ARCHITECTURES)
  if(NOT "${{archs}}" STREQUAL "ppu_15;ppu_10")
    message(FATAL_ERROR "The extension lost part of the architecture list")
  endif()
endforeach()
""",
        encoding="utf-8",
    )
    build = tmp_path / "build"
    _run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DCMAKE_CXX_COMPILER={compiler}",
        ]
    )
    _run([cmake, "--build", str(build)])


def test_bf16_register_loads_do_not_require_conversion_operators(
    tmp_path: Path,
) -> None:
    compiler = _tool("c++")
    source = (ROOT / "csrc/plugin/sampler_bf16.h").read_text(encoding="utf-8")
    start = source.index("  // Load scores into registers (bf16 -> float).")
    end = source.index("  // Accumulate histogram.", start)
    # Minimise to the exact failing block. Model the SDK's conversion-disabled
    # BF16 interface, and check both template branches including partial rows.
    # This is a host-side contract test, not an emulation of the device kernel.
    prelude = r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
struct BFloat16 { uint16_t bits; };  // deliberately no operator float()
float __bfloat162float(BFloat16 value) {
  uint32_t bits = uint32_t(value.bits) << 16;
  float result;
  static_assert(sizeof(result) == sizeof(bits));
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
constexpr int kBF16VecsPerThread = 4;
constexpr int kBF16BlockSize = 1024;
constexpr int kBF16Max1PassLen = 16384;
struct BF16Vec4F {
  float data[4];
  float& operator[](int i) { return data[i]; }
};
struct Smem { float scoreBuffer[kBF16Max1PassLen] = {}; };
void __syncthreads() {}
template <bool kIs2Pass>
void load(const BFloat16* scores, uint32_t tx, uint32_t length, Smem* smem) {
"""
    checks = r"""
  for (int v = 0; v < kBF16VecsPerThread; ++v) {
    uint32_t base = (tx + v * kBF16BlockSize) * 4;
    if (base >= length) break;
    for (int e = 0; e < 4; ++e) {
      uint32_t idx = base + e;
      float expected = (idx < length) ? __bfloat162float(scores[idx]) : 0.0f;
      assert(local[v][e] == expected);
    }
  }
}
int main() {
  for (uint32_t length : {1u, 4097u, 16384u, 16385u, 32768u}) {
    std::vector<BFloat16> scores(length);
    constexpr uint16_t values[] = {0, 0x3f80, 0xbf80, 0x3fc0, 0x7f80, 0xff80};
    for (uint32_t i = 0; i < length; ++i) scores[i].bits = values[i % 6];
    Smem smem;
    for (uint32_t tx = 0; tx < kBF16BlockSize; ++tx) {
      if (length > kBF16Max1PassLen) load<true>(scores.data(), tx, length, &smem);
      else load<false>(scores.data(), tx, length, &smem);
    }
    for (uint32_t i = kBF16Max1PassLen; i < length; ++i) {
      assert(smem.scoreBuffer[i - kBF16Max1PassLen] == __bfloat162float(scores[i]));
    }
  }
}
"""
    unit = tmp_path / "bf16_load.cpp"
    unit.write_text(prelude + source[start:end] + checks, encoding="utf-8")
    executable = tmp_path / "bf16_load"
    _run([compiler, "-std=c++20", str(unit), "-o", str(executable)])
    _run([str(executable)])
