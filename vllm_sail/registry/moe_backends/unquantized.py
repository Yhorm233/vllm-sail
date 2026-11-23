# SPDX-License-Identifier: Apache-2.0
"""Register PPU backends with the unquantized MoE oracle."""

from __future__ import annotations

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle import unquantized as oracle

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.unquantized"
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel(), which would delete this patch."
)

UnquantizedMoeBackend = oracle.UnquantizedMoeBackend
extend_enum(UnquantizedMoeBackend, "PPU_DEEPGEMM", "PPU_DEEPGEMM")
extend_enum(
    UnquantizedMoeBackend,
    "BATCHED_PPU_DEEPGEMM",
    "BATCHED_PPU_DEEPGEMM",
)
extend_enum(UnquantizedMoeBackend, "ACEXT", "ACEXT")

_upstream_get_priority_backends = oracle._get_priority_backends
_upstream_backend_to_kernel_cls = oracle.backend_to_kernel_cls
_upstream_map_unquantized_backend = oracle.map_unquantized_backend
_upstream_select_unquantized_moe_backend = oracle.select_unquantized_moe_backend


@patch(
    _MODULE,
    "_get_priority_backends",
    reason=(
        "The unquantized oracle has no registration API for PPU DeepGEMM and "
        "Acext experts, so PPU needs its own ordered candidate list."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[UnquantizedMoeBackend]:
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_get_priority_backends(moe_config)

    from vllm_sail.model_executor.layers.fused_moe.experts.acext import (
        is_acext_supported,
    )
    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    activation_format = moe_config.activation_format
    requested = ppu_envs.VLLM_SAIL_MOE_BACKEND
    deep_gemm_enabled = (
        is_deep_gemm_supported()
        and (not requested or requested == "deepgemm")
        and not moe_config.has_bias
    )
    acext_enabled = (
        is_acext_supported()
        and (not requested or requested == "acext")
        and activation_format == mk.FusedMoEActivationFormat.Standard
    )

    backends = []
    if deep_gemm_enabled:
        backends.append(UnquantizedMoeBackend.PPU_DEEPGEMM)
    if acext_enabled:
        backends.append(UnquantizedMoeBackend.ACEXT)
    if deep_gemm_enabled:
        backends.append(UnquantizedMoeBackend.BATCHED_PPU_DEEPGEMM)
    backends.extend(
        [UnquantizedMoeBackend.TRITON, UnquantizedMoeBackend.BATCHED_TRITON]
    )
    return backends


@patch(
    _MODULE,
    "backend_to_kernel_cls",
    reason="The unquantized oracle cannot resolve plugin-owned PPU expert classes.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def backend_to_kernel_cls(
    backend: UnquantizedMoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == UnquantizedMoeBackend.PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExperts,
        )

        return [PPUDeepGemmExperts]
    if backend == UnquantizedMoeBackend.BATCHED_PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExperts,
        )

        return [PPUBatchedDeepGemmExperts]
    if backend == UnquantizedMoeBackend.ACEXT:
        from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

        return [AcextExperts]
    return _upstream_backend_to_kernel_cls(backend)


@patch(
    _MODULE,
    "map_unquantized_backend",
    reason="The public MoE backend strings have no upstream mapping to PPU members.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def map_unquantized_backend(
    runner_backend: MoEBackend,
) -> UnquantizedMoeBackend:
    mapping = {
        "ppu_deep_gemm": UnquantizedMoeBackend.PPU_DEEPGEMM,
        "ppu_acext": UnquantizedMoeBackend.ACEXT,
    }
    if runner_backend in mapping:
        return mapping[runner_backend]
    return _upstream_map_unquantized_backend(runner_backend)


@patch(
    _MODULE,
    "select_unquantized_moe_backend",
    reason=(
        "Explicit ppu_deep_gemm requests need the batched PPU member when the "
        "prepare/finalize path uses batched expert activations."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
    # LoRA 的布局门控和专家选择归 upstream；不得由 batched PPU 快分支绕过。
    if moe_config.is_lora_enabled:
        return _upstream_select_unquantized_moe_backend(moe_config)
    if (
        moe_config.moe_backend == "ppu_deep_gemm"
        and moe_config.activation_format.is_batched
    ):
        backend = UnquantizedMoeBackend.BATCHED_PPU_DEEPGEMM
        for kernel_cls in backend_to_kernel_cls(backend):
            supported, reason = kernel_cls.is_supported_config(
                kernel_cls,
                moe_config,
                None,
                None,
                mk.FusedMoEActivationFormat.BatchedExperts,
            )
            if supported:
                return backend, kernel_cls
        raise ValueError(
            f"Unquantized MoE backend {backend.value} is unsupported: {reason}"
        )
    return _upstream_select_unquantized_moe_backend(moe_config)
