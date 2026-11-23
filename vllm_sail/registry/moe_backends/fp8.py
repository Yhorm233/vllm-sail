# SPDX-License-Identifier: Apache-2.0
"""Register PPU backends with the FP8 MoE oracle."""

from __future__ import annotations

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle import fp8 as oracle
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.fp8"
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "upstream gains a register_moe_backend() extension point mirroring "
    "register_linear_kernel(), which would delete this patch."
)

Fp8MoeBackend = oracle.Fp8MoeBackend
extend_enum(Fp8MoeBackend, "PPU_DEEPGEMM", "PPU_DEEPGEMM")
extend_enum(Fp8MoeBackend, "BATCHED_PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM")

_upstream_get_priority_backends = oracle._get_priority_backends
_upstream_backend_to_kernel_cls = oracle.backend_to_kernel_cls
_upstream_map_fp8_backend = oracle.map_fp8_backend
_upstream_select_fp8_moe_backend = oracle.select_fp8_moe_backend


@patch(
    _MODULE,
    "_get_priority_backends",
    reason="The FP8 oracle candidate list has no PPU DeepGEMM backends.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_priority_backends(
    moe_config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> list[Fp8MoeBackend]:
    backends = _upstream_get_priority_backends(moe_config, weight_key, activation_key)
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return backends
    if ppu_envs.is_set("VLLM_SAIL_MOE_BACKEND") and (
        ppu_envs.VLLM_SAIL_MOE_BACKEND != "deepgemm"
    ):
        return backends
    if Fp8MoeBackend.PPU_DEEPGEMM not in backends:
        backends.insert(
            backends.index(Fp8MoeBackend.DEEPGEMM) + 1,
            Fp8MoeBackend.PPU_DEEPGEMM,
        )
    if Fp8MoeBackend.BATCHED_PPU_DEEPGEMM not in backends:
        backends.insert(
            backends.index(Fp8MoeBackend.BATCHED_DEEPGEMM) + 1,
            Fp8MoeBackend.BATCHED_PPU_DEEPGEMM,
        )
    return backends


@patch(
    _MODULE,
    "backend_to_kernel_cls",
    reason="The FP8 oracle cannot resolve plugin-owned PPU expert classes.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Fp8MoeBackend.PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExperts,
        )

        return [PPUDeepGemmExperts]
    if backend == Fp8MoeBackend.BATCHED_PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExperts,
        )

        return [PPUBatchedDeepGemmExperts]
    return _upstream_backend_to_kernel_cls(backend)


@patch(
    _MODULE,
    "map_fp8_backend",
    reason="The public ppu_deep_gemm string has no upstream FP8 member mapping.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def map_fp8_backend(runner_backend: MoEBackend) -> Fp8MoeBackend:
    if runner_backend == "ppu_deep_gemm":
        return Fp8MoeBackend.PPU_DEEPGEMM
    return _upstream_map_fp8_backend(runner_backend)


def _select_ppu_backend(config, weight_key, activation_key, backend):
    activation_format = config.activation_format
    if (
        activation_format == mk.FusedMoEActivationFormat.BatchedExperts
        and backend == Fp8MoeBackend.PPU_DEEPGEMM
    ):
        backend = Fp8MoeBackend.BATCHED_PPU_DEEPGEMM
    for kernel_cls in backend_to_kernel_cls(backend):
        supported, reason = kernel_cls.is_supported_config(
            kernel_cls, config, weight_key, activation_key, activation_format
        )
        if supported:
            return backend, kernel_cls
    raise ValueError(f"FP8 MoE backend {backend.value} is unsupported: {reason}")


@patch(
    _MODULE,
    "select_fp8_moe_backend",
    reason=(
        "Explicit PPU FP8 selection must choose the batched PPU expert when "
        "needed and honor the plugin-specific DeepGEMM environment override."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def select_fp8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    allow_vllm_cutlass: bool = False,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts] | None]:
    if config.moe_backend == "ppu_deep_gemm":
        return _select_ppu_backend(
            config, weight_key, activation_key, Fp8MoeBackend.PPU_DEEPGEMM
        )
    from vllm.platforms import current_platform

    if (
        config.moe_backend == "auto"
        and current_platform.is_ppu()
        and ppu_envs.is_set("VLLM_SAIL_MOE_BACKEND")
        and ppu_envs.VLLM_SAIL_MOE_BACKEND == "deepgemm"
    ):
        return _select_ppu_backend(
            config, weight_key, activation_key, Fp8MoeBackend.PPU_DEEPGEMM
        )
    return _upstream_select_fp8_moe_backend(
        config, weight_key, activation_key, allow_vllm_cutlass
    )


_upstream_convert = oracle.convert_to_fp8_moe_kernel_format
_upstream_quant_config = oracle.make_fp8_moe_quant_config


@patch(
    _MODULE,
    "convert_to_fp8_moe_kernel_format",
    reason="PPU FP8 experts require block weight preparation or unchanged channelwise tensors.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def convert_to_fp8_moe_kernel_format(
    fp8_backend,
    layer,
    w13,
    w2,
    w13_scale,
    w2_scale,
    w13_input_scale,
    w2_input_scale,
):
    if fp8_backend in (Fp8MoeBackend.PPU_DEEPGEMM, Fp8MoeBackend.BATCHED_PPU_DEEPGEMM):
        block_shape = getattr(layer, "weight_block_size", None)
        if block_shape is not None:
            return oracle.prepare_fp8_moe_layer_for_deepgemm(
                w13, w2, w13_scale, w2_scale, tuple(block_shape)
            )
        return w13, w2, w13_scale, w2_scale
    return _upstream_convert(
        fp8_backend,
        layer,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_input_scale,
        w2_input_scale,
    )


@patch(
    _MODULE,
    "make_fp8_moe_quant_config",
    reason="FP8 channelwise quantization must preserve SwiGLU alpha and beta.",
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def make_fp8_moe_quant_config(*args, **kwargs):
    import inspect

    from vllm.platforms import current_platform

    result = _upstream_quant_config(*args, **kwargs)
    if current_platform.is_ppu():
        arguments = inspect.signature(_upstream_quant_config).bind(*args, **kwargs)
        arguments.apply_defaults()
        result.gemm1_alpha = arguments.arguments["gemm1_alpha"]
        result.gemm1_beta = arguments.arguments["gemm1_beta"]
    return result
