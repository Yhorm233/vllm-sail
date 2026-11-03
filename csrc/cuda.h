// SPDX-License-Identifier: Apache-2.0
#pragma once

// CUDA version gate for stock torch headers.
#include "ppu_compat.h"

#ifndef CUDA_VERSION
#define CUDA_VERSION COMPATIBLE_VERSION
#endif
