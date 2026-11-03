// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// Adapted from PyTorch's torch/csrc/stable/macros.h (v2.10.0).
// Upstream copyright notices and BSD terms: LICENSES/PyTorch.txt.
// Project modifications are licensed under Apache-2.0.
#pragma once

// Accept HGGC error enums while retaining torch's checks.
#include_next <torch/csrc/stable/macros.h>

#ifdef STD_CUDA_CHECK
#undef STD_CUDA_CHECK
#undef STD_CUDA_KERNEL_LAUNCH_CHECK

#define STD_CUDA_CHECK(EXPR)                      \
  do {                                            \
    const auto __err = (EXPR);                    \
    char* __error_msg = nullptr;                  \
    torch_c10_cuda_check_msg(                     \
        static_cast<int32_t>(__err),              \
        __FILE__,                                 \
        __func__,                                 \
        static_cast<uint32_t>(__LINE__),          \
        true,                                     \
        &__error_msg);                            \
    if (__error_msg != nullptr) {                 \
      std::string __msg(__error_msg);             \
      torch_c10_cuda_free_error_msg(__error_msg); \
      throw std::runtime_error(__msg);            \
    }                                             \
  } while (0)

#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(hggcGetLastError())
#endif
