# SPDX-License-Identifier: Apache-2.0
"""Register PPU backends with the INT8 MoE oracle."""

from __future__ import annotations

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

logger = init_logger(__name__)

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel(), which would delete this patch."
)

Int8MoeBackend = oracle.Int8MoeBackend
extend_enum(Int8MoeBackend, "PPU_DEEPGEMM", "PPU_DEEPGEMM")
extend_enum(Int8MoeBackend, "BATCHED_PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM")
extend_enum(Int8MoeBackend, "ACEXT", "ACEXT")

_upstream_get_priority_backends = oracle._get_priority_backends
_upstream_backend_to_kernel_cls = oracle.backend_to_kernel_cls
_upstream_convert_to_int8_moe_kernel_format = oracle.convert_to_int8_moe_kernel_format
_upstream_map_int8_backend = oracle.map_int8_backend
_upstream_select_int8_moe_backend = oracle.select_int8_moe_backend


@patch(
    _MODULE,
    "_get_priority_backends",
    reason=(
        "The INT8 oracle has no registration API for PPU DeepGEMM and Acext "
        "experts, so PPU needs its own ordered candidate list."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends(moe_config: FusedMoEConfig) -> list[Int8MoeBackend]:
    backends = _upstream_get_priority_backends(moe_config)
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return backends

    backends = [
        backend
        for backend in backends
        if backend
        not in (
            Int8MoeBackend.PPU_DEEPGEMM,
            Int8MoeBackend.BATCHED_PPU_DEEPGEMM,
            Int8MoeBackend.ACEXT,
        )
    ]

    from vllm_sail.model_executor.layers.fused_moe.experts.acext import (
        is_acext_supported,
    )
    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    requested = ppu_envs.VLLM_SAIL_MOE_BACKEND
    deep_gemm_enabled = (
        is_deep_gemm_supported()
        and (not requested or requested == "deepgemm")
        and not moe_config.has_bias
    )
    acext_enabled = is_acext_supported() and (not requested or requested == "acext")
    ppu_backends = []
    if deep_gemm_enabled:
        ppu_backends.append(Int8MoeBackend.PPU_DEEPGEMM)
    if acext_enabled:
        ppu_backends.append(Int8MoeBackend.ACEXT)
    if deep_gemm_enabled:
        ppu_backends.append(Int8MoeBackend.BATCHED_PPU_DEEPGEMM)
    logger.info(
        "vllm-sail: int8 MoE priority backends: %s (deep_gemm=%s acext=%s "
        "has_bias=%s requested=%s)",
        [b.name for b in [*ppu_backends, *backends]],
        deep_gemm_enabled,
        acext_enabled,
        moe_config.has_bias,
        requested,
    )
    return [*ppu_backends, *backends]


@patch(
    _MODULE,
    "backend_to_kernel_cls",
    reason="The INT8 oracle cannot resolve plugin-owned PPU expert classes.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Int8MoeBackend.PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExperts,
        )

        return [PPUDeepGemmExperts]
    if backend == Int8MoeBackend.BATCHED_PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExperts,
        )

        return [PPUBatchedDeepGemmExperts]
    if backend == Int8MoeBackend.ACEXT:
        from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

        return [AcextExperts]
    return _upstream_backend_to_kernel_cls(backend)


@patch(
    _MODULE,
    "convert_to_int8_moe_kernel_format",
    reason=(
        "PPU DeepGEMM and Acext consume canonical INT8 MoE weights directly, "
        "but upstream rejects their plugin-added backend enum members during "
        "process_weights_after_loading."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream accepts out-of-tree INT8 backends in its weight-conversion "
        "hook, or exposes a backend-specific conversion registry."
    ),
)
def convert_to_int8_moe_kernel_format(
    int8_backend: Int8MoeBackend,
    w13: torch.Tensor,
    w2: torch.Tensor,
    layer: torch.nn.Module | None = None,
    w13_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int8_backend in (
        Int8MoeBackend.PPU_DEEPGEMM,
        Int8MoeBackend.BATCHED_PPU_DEEPGEMM,
        Int8MoeBackend.ACEXT,
    ):
        return w13, w2
    return _upstream_convert_to_int8_moe_kernel_format(
        int8_backend,
        w13,
        w2,
        layer=layer,
        w13_scale=w13_scale,
    )


@patch(
    _MODULE,
    "map_int8_backend",
    reason="The public PPU backend strings have no upstream INT8 member mapping.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def map_int8_backend(runner_backend: MoEBackend) -> Int8MoeBackend:
    mapping = {
        "ppu_deep_gemm": Int8MoeBackend.PPU_DEEPGEMM,
        "ppu_acext": Int8MoeBackend.ACEXT,
    }
    if runner_backend in mapping:
        return mapping[runner_backend]
    return _upstream_map_int8_backend(runner_backend)


@patch(
    _MODULE,
    "select_int8_moe_backend",
    reason=(
        "Batched INT8 activations require the batched PPU DeepGEMM expert and "
        "must reject standard-only Triton and Acext selections."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def select_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = oracle.kInt8StaticChannelSym,
    activation_key: QuantKey | None = oracle.kInt8DynamicTokenSym,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
    is_batched = config.activation_format.is_batched
    if not is_batched or config.moe_backend == "auto":
        return _upstream_select_int8_moe_backend(config, weight_key, activation_key)

    requested = map_int8_backend(config.moe_backend)
    if requested == Int8MoeBackend.PPU_DEEPGEMM:
        requested = Int8MoeBackend.BATCHED_PPU_DEEPGEMM
    elif requested == Int8MoeBackend.TRITON:
        raise ValueError("vLLM Triton MoE backend is disabled for this configuration.")
    elif requested == Int8MoeBackend.ACEXT:
        raise ValueError("vLLM ACEXT MoE backend is disabled for this configuration.")
    else:
        return _upstream_select_int8_moe_backend(config, weight_key, activation_key)

    for kernel_cls in backend_to_kernel_cls(requested):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls,
            config,
            weight_key,
            activation_key,
            mk.FusedMoEActivationFormat.BatchedExperts,
        )
        if supported:
            return requested, kernel_cls
    raise ValueError(f"INT8 MoE backend {requested.value} is unsupported: {reason}")
