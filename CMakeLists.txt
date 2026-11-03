# SPDX-License-Identifier: Apache-2.0
#
# Native build for the vllm-sail plugin.
#
# PPU is CUDA-free: device code is compiled by `hgcc` against HGGC headers, not
# by nvcc against CUDA headers. The `HG` language comes from t-head/cmake-hgcc.
#
# Everything that must be known before `project()` enables `HG` is set first --
# cmake-hgcc consumes those variables exactly once, at language-enable time.
#
# Invoked only by setup.py, and only after setup.py has confirmed hgcc exists:
# enabling `HG` without hgcc is a FATAL_ERROR with no recovery path.
#
# Design: docs/developer_guide/kernels.md

cmake_minimum_required(VERSION 3.26)

# ---------------------------------------------------------------------------
# cmake-hgcc discovery. Prefer an explicit checkout or an existing installation;
# fetch the pinned upstream revision when neither is configured. This must happen
# before project() enables the custom HG language.
# ---------------------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolveCmakeHgcc.cmake")
vllm_sail_resolve_cmake_hgcc()

# ---------------------------------------------------------------------------
# HG language configuration. MUST precede project().
# ---------------------------------------------------------------------------
# Compile the ported corpus with HGGC's .hg source extension.

if(NOT CMAKE_HG_ARCHITECTURES)
  if(PYTORCH_SAIL_ARCH)
    set(CMAKE_HG_ARCHITECTURES "${PYTORCH_SAIL_ARCH}")
  elseif(DEFINED ENV{PYTORCH_SAIL_ARCH} AND NOT "$ENV{PYTORCH_SAIL_ARCH}" STREQUAL "")
    set(CMAKE_HG_ARCHITECTURES "$ENV{PYTORCH_SAIL_ARCH}")
  elseif(DEFINED ENV{VLLM_SAIL_HG_ARCH})
    set(CMAKE_HG_ARCHITECTURES "$ENV{VLLM_SAIL_HG_ARCH}")
  else()
    set(CMAKE_HG_ARCHITECTURES "ppu_15;ppu_10")
  endif()
endif()
# setup.py resolves the environment once and passes both variables. Keep direct
# CMake invocations consistent too: Torch must not append a different -arch list.
string(REGEX REPLACE "[ \t\r\n]+" ";" CMAKE_HG_ARCHITECTURES "${CMAKE_HG_ARCHITECTURES}")
list(FILTER CMAKE_HG_ARCHITECTURES EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES CMAKE_HG_ARCHITECTURES)
if(NOT CMAKE_HG_ARCHITECTURES)
  set(CMAKE_HG_ARCHITECTURES "ppu_15;ppu_10")
endif()
set(PYTORCH_SAIL_ARCH "${CMAKE_HG_ARCHITECTURES}")
message(STATUS "vllm-sail: HG and SAIL Torch architectures: ${CMAKE_HG_ARCHITECTURES}")
if(NOT DEFINED CMAKE_HG_STANDARD)
  set(CMAKE_HG_STANDARD 20)
endif()
set(CMAKE_HG_STANDARD_REQUIRED ON)

project(vllm_sail_native LANGUAGES CXX HG)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(HGGCToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# ---------------------------------------------------------------------------
# Torch. Found via the caller's Python, not via a bundled FindTorch.
# ---------------------------------------------------------------------------
if(NOT DEFINED TORCH_CMAKE_PREFIX)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# The PPU torch fork gates its CUDA-named stable-ABI entry points behind
# USE_CUDA, which is why a CUDA-free build still defines it. Mirrors the
# fork's own in-tree build.
set(PPU_DEFINES
  USE_PPU=1
  USE_SAIL=1
  PPU_SDK=1
  USE_CUDA=1
  USE_DISTRIBUTED=1)

# ---------------------------------------------------------------------------
# Source lists come from the upstream and plugin manifests through setup.py.
# CMake never globs.
# ---------------------------------------------------------------------------
function(ppu_add_extension target)
  cmake_parse_arguments(ARG "" "INCLUDE_ROOT" "SOURCES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(STATUS "vllm-sail: ${target} has no sources; skipping")
    return()
  endif()
  Python_add_library(${target} MODULE WITH_SOABI ${ARG_SOURCES})
  # Resolve plugin headers within their owned tree before searching upstream.
  # Upstream targets use their generated include tree first.
  if(ARG_INCLUDE_ROOT)
    target_include_directories(${target} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INCLUDE_ROOT}")
  endif()
  target_include_directories(${target} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/upstream"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
  target_compile_definitions(${target} PRIVATE ${PPU_DEFINES})
  # Supply sailify compatibility gates; SHELL: keeps the -include operand paired.
  target_compile_options(${target} PRIVATE
    "SHELL:-include ppu_compat.h")
  # cmake-hgcc emits the standard in CMAKE_HG_FLAGS during project(), but the
  # SAIL TorchConfig can append its own -std=c++17 afterwards. Target options
  # follow global flags, so keep the requested HG standard authoritative here.
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:HG>:-std=c++${CMAKE_HG_STANDARD}>")
  # Upstream's FP8 conversion specializations are gated on ENABLE_FP8. The
  # native PPU SDK provides the HG equivalents translated by the port tool.
  target_compile_definitions(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:HG>:ENABLE_FP8>")
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:HG>:SHELL:-include ppu_kernel_compat.h>")
  # Match vLLM's get_torch_gpu_compiler_flags: its kernels require the SDK
  # half/BF16 operators and conversions that Torch's extension defaults disable.
  # Override only these targets, after Torch's global CMAKE_HG_FLAGS.
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:HG>:-U__HGGC_NO_HALF_OPERATORS__>"
    "$<$<COMPILE_LANGUAGE:HG>:-U__HGGC_NO_HALF_CONVERSIONS__>"
    "$<$<COMPILE_LANGUAGE:HG>:-U__HGGC_NO_HALF2_OPERATORS__>"
    "$<$<COMPILE_LANGUAGE:HG>:-U__HGGC_NO_BFLOAT16_CONVERSIONS__>")
  # Avoid the extended-lambda device builtin during host compilation.
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:HG>:-U__HGGCCC_EXTENDED_LAMBDA__>")
  target_link_libraries(${target} PRIVATE
    torch
    HGGC::toolkit)
  set_target_properties(${target} PROPERTIES
    HG_ARCHITECTURES "${CMAKE_HG_ARCHITECTURES}"
    HG_STANDARD "${CMAKE_HG_STANDARD}"
    LIBRARY_OUTPUT_DIRECTORY "${PPU_EXT_OUTPUT_DIR}")
  message(STATUS "vllm-sail: ${target} <- ${ARG_SOURCES}")
endfunction()

# Ported upstream kernels. These claim upstream's own op namespaces
# (_C, _C_cache_ops, _C_cuda_utils, _moe_C) -- see spec section 3.2.
ppu_add_extension(_upstream_C SOURCES ${PPU_UPSTREAM_EXT_SRC})
# Upstream guards both the declaration and schema/CUDA implementation binding.
# The independently built HGGC corpus must enable them as well as compile the TU.
target_compile_definitions(_upstream_C PRIVATE VLLM_ENABLE_FUSED_GDN_DECODE=1)
ppu_add_extension(_upstream_moe_C SOURCES ${PPU_UPSTREAM_MOE_EXT_SRC})

# Plugin-owned HGGC kernels and stable-ABI bindings, under private _ppu_* namespaces.
ppu_add_extension(_C
  INCLUDE_ROOT csrc/plugin
  SOURCES ${PPU_PLUGIN_EXT_SRC})
ppu_add_extension(_moe_C
  INCLUDE_ROOT csrc/plugin
  SOURCES ${PPU_PLUGIN_MOE_EXT_SRC})
