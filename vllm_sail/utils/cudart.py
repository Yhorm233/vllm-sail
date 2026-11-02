# SPDX-License-Identifier: Apache-2.0
"""Expose the selected SAIL runtime to TileLang through the platform kernel hook."""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

from vllm_sail import envs


@cache
def _load_sdk_cudart(sdk: str) -> ctypes.CDLL:
    # Keep the handle alive without mixing in system CUDA or requiring LD_PRELOAD.
    library = Path(sdk) / "CUDA_SDK/lib64/libcudart.so"
    return ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def load_sdk_cudart() -> ctypes.CDLL:
    """Read SDK configuration lazily and load once per SDK path."""
    return _load_sdk_cudart(envs.PPU_SDK)
