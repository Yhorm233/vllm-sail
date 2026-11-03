// SPDX-License-Identifier: Apache-2.0
#pragma once

// Compatibility gates for sailify-generated sources.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__cplusplus)
#include <type_traits>
#endif

// HG preprocessing does not reliably support __has_include.
#if defined(__HGGCCC__)
#define PPU_COMPAT_HAS_SDK
#elif defined(__has_include)
#if __has_include(<hggc.h>)
#define PPU_COMPAT_HAS_SDK
#endif
#endif

#ifdef PPU_COMPAT_HAS_SDK
#include <hggc.h>
#if defined(__cplusplus)
#include <acblas_v2.h>
#endif
#endif

// Use SDK scalar types for stock torch CUDA spellings.
#if defined(__HGGCCC__) && defined(__cplusplus)
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <hggc_runtime_api.h>
#include <hggc_runtime_api.hpp>
#define __CUDA_FP16_H__
#define __CUDA_BF16_H__
typedef __ppu_bfloat16 __nv_bfloat16;
typedef __ppu_bfloat162 __nv_bfloat162;
typedef __ppu_bfloat16_raw __nv_bfloat16_raw;
typedef __ppu_bfloat162_raw __nv_bfloat162_raw;
#endif

#ifndef COMPATIBLE_VERSION
#define COMPATIBLE_VERSION 13000
#endif

#ifndef COMPATIBLE_RT_VERSION
#define COMPATIBLE_RT_VERSION COMPATIBLE_VERSION
#endif

#ifndef __COMPATIBLECC_VER_MAJOR__
#define __COMPATIBLECC_VER_MAJOR__ 13
#endif
#ifndef __COMPATIBLECC_VER_MINOR__
#define __COMPATIBLECC_VER_MINOR__ 0
#endif
#ifndef __COMPATIBLECC_VER_BUILD__
#define __COMPATIBLECC_VER_BUILD__ 0
#endif

// Device-only, like __CUDA_ARCH__: undefined during the host pass.
#if defined(__HGGC_ARCH__) && !defined(COMPATIBLE_ARCH)
#if __HGGC_ARCH__ == 100
#define COMPATIBLE_ARCH 800
#elif __HGGC_ARCH__ == 150
#define COMPATIBLE_ARCH 890
#else
#define COMPATIBLE_ARCH __HGGC_ARCH__
#endif
#endif

// hgcc defines this only for device code, like __CUDA_ARCH_LIST__.
#ifndef COMPATIBLE_ARCH_LIST
#if defined(__HGGCCC__)
#if defined(COMPATIBLE_ARCH)
#define COMPATIBLE_ARCH_LIST COMPATIBLE_ARCH
#else
#define COMPATIBLE_ARCH_LIST 800
#endif
#endif
#endif

// CUDA-spelling compiler gates for stock torch.
#if defined(__HGGCCC__) && !defined(__CUDACC__)
#define __CUDACC__ 1
#define __CUDACC_VER_MAJOR__ __COMPATIBLECC_VER_MAJOR__
#define __CUDACC_VER_MINOR__ __COMPATIBLECC_VER_MINOR__
#define __CUDACC_VER_BUILD__ __COMPATIBLECC_VER_BUILD__
#endif
#if defined(__HGGC_ARCH__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ COMPATIBLE_ARCH
#endif
