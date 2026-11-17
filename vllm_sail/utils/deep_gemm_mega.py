# SPDX-License-Identifier: Apache-2.0
"""Supported PPU MegaMoE formats and lazy access to DeepGEMM."""

from __future__ import annotations

import functools
import importlib
from typing import NamedTuple


class MegaMoEFormat(NamedTuple):
    kernel: str
    transform: str | None
    activations: tuple[str, ...]
    min_hidden: int
    hidden_multiple: int
    intermediate_multiple: int
    #: Routed-expert weight and scale dtype the kernel expects. MXFP4 packs
    #: two values per byte with UE8M0 block scales; INT8 stores one value per
    #: byte with a per-output-channel FP32 scale.
    weight_dtype: str = "uint8"
    scale_dtype: str = "uint8"
    #: Device capability the kernels behind ``kernel`` are compiled for. MXFP4
    #: modes also run on newer devices; the INT8 kernel is 8.0-only.
    device_capability: tuple[int, int] = (8, 9)
    #: ``mega_moe_pre_dispatch`` quantization group; INT8 scales per token.
    dispatch_group_size: int = 32

    @property
    def is_int8(self) -> bool:
        return self.weight_dtype == "int8"


MEGA_MOE_FORMATS = {
    "fp4xfp4": MegaMoEFormat(
        "fp4_fp4_mega_moe", None, ("swiglu", "situ"), 2048, 512, 64
    ),
    "fp8xfp4": MegaMoEFormat(
        "fp8_fp4_mega_moe",
        "transform_weights_for_w4a8_mega_moe",
        ("swiglu", "situ"),
        256,
        256,
        128,
    ),
    "bf16xfp4": MegaMoEFormat(
        "bf16_fp4_mega_moe",
        "transform_weights_for_w4a16_mega_moe",
        ("situ",),
        256,
        256,
        128,
    ),
    "int8xint8": MegaMoEFormat(
        "int8_int8_mega_moe",
        "transform_weights_for_mega_moe",
        ("swiglu", "swiglu_step", "situ", "swiglu_oai"),
        2048,
        512,
        256,
        weight_dtype="int8",
        scale_dtype="float32",
        device_capability=(8, 0),
        dispatch_group_size=0,
    ),
}

_DEVICE_NAMES = {
    (8, 0): "Zhenwu ZW-810E (8.0)",
    (8, 9): "Zhenwu M890 (8.9)",
}


def resolve_expert_dtype(vllm_config) -> str:
    """Routed-expert precision of the checkpoint behind ``vllm_config``.

    W8A8 exports publish ``expert_dtype`` through ``quantization_config``,
    which vLLM promotes onto the HF config; the compressed-tensors contract
    check also accepts exports that spell the alias differently. Everything
    else falls back to the attribute MegaMoE already keys on.
    """
    from vllm_sail.models.deepseek_v41.config import uses_int8_w8a8

    if uses_int8_w8a8(vllm_config):
        return "int8"
    model_config = getattr(vllm_config, "model_config", None)
    for config in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        expert_dtype = getattr(config, "expert_dtype", None)
        if expert_dtype is not None:
            return expert_dtype
    return "fp4"


def mma_type_for_expert_dtype(expert_dtype: str | None) -> str:
    """MMA type that matches a checkpoint's routed-expert precision.

    INT8 experts pin the 8.0 INT8 kernel because no MXFP4 kernel exists on
    ZW-810E and the two weight layouts are not interchangeable. Every other
    checkpoint keeps the configured MXFP4 mode; an explicit ``int8xint8`` there
    is an error rather than a silent fallback.
    """
    from vllm_sail import envs as sail_envs

    if expert_dtype == "int8":
        return "int8xint8"
    selected = sail_envs.VLLM_SAIL_MEGA_MOE_MMA_TYPE
    if MEGA_MOE_FORMATS[selected].is_int8:
        raise ValueError(
            "VLLM_SAIL_MEGA_MOE_MMA_TYPE=int8xint8 requires INT8 experts; "
            f"got expert_dtype={expert_dtype!r}."
        )
    return selected


class _Int8MegaMoEAPI:
    """DeepGEMM module view that pins the INT8 pre-dispatch group size.

    ``mega_moe_pre_dispatch`` asserts ``group_size == 0`` for INT8 and ``32``
    for MXFP4, and the shared forward passes it through a keyword only for
    modes that need it. Wrapping the module keeps the generic forward intact.
    """

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def mega_moe_pre_dispatch(self, *args, **kwargs):
        kwargs.setdefault("group_size", 0)
        return self._module.mega_moe_pre_dispatch(*args, **kwargs)


@functools.cache
def mega_moe_api(mma_type: str):
    """Resolve only the DeepGEMM API required by the selected mode."""
    fmt = MEGA_MOE_FORMATS[mma_type]
    try:
        module = importlib.import_module("deep_gemm.mega")
    except ImportError as exc:
        raise RuntimeError(
            "PPU MegaMoE requires DeepGEMM-for-sail with deep_gemm.mega support."
        ) from exc
    required = ["get_symm_buffer_for_mega_moe", "mega_moe_pre_dispatch", fmt.kernel]
    if fmt.transform is not None:
        required.append(fmt.transform)
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            f"PPU MegaMoE API is missing for {mma_type}: {', '.join(missing)}"
        )
    return _Int8MegaMoEAPI(module) if fmt.is_int8 else module


def check_mega_moe_supported(
    device, hidden_size: int, intermediate_size: int | None, mma_type: str
):
    """Gate one mode on its own kernels' device plus the shared geometry.

    Each entry advertises the capability it was built for: exact match for
    the single-arch INT8 kernel (ZW-810E), at-least for the MXFP4 modes.
    """
    from vllm.platforms import current_platform

    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    fmt = MEGA_MOE_FORMATS[mma_type]
    device_id = device.index or 0
    if fmt.is_int8:
        capability = current_platform.get_device_capability(device_id=device_id)
        current = None if capability is None else (capability.major, capability.minor)
        supported = current == fmt.device_capability
    else:
        supported = current_platform.has_device_capability(89, device_id=device_id)
    if not (current_platform.is_ppu() and supported and is_deep_gemm_supported()):
        raise NotImplementedError(
            f"SAIL MegaMoE {mma_type} requires "
            f"{_DEVICE_NAMES.get(fmt.device_capability, fmt.device_capability)} "
            "and DeepGEMM."
        )
    if intermediate_size is not None and (
        hidden_size < fmt.min_hidden
        or hidden_size % fmt.hidden_multiple
        or intermediate_size <= 0
        or intermediate_size % fmt.intermediate_multiple
    ):
        raise ValueError(
            f"SAIL MegaMoE {mma_type} requires H >= {fmt.min_hidden} divisible by "
            f"{fmt.hidden_multiple} and positive I divisible by {fmt.intermediate_multiple}."
        )
    mega_moe_api(mma_type)
