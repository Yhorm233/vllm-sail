# ruff: noqa: F821, I001
# SPDX-License-Identifier: Apache-2.0
"""Register PPU standard and batched INT4 W4A16 experts."""

from __future__ import annotations

import inspect

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle import int_wna16 as oracle

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int_wna16"
_META = dict(
    reason="The WNA16 oracle needs the plugin's PPU DeepGEMM experts and packed weight layout.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The MoE oracle supports registering plugin WNA16 backends.",
)
WNA16MoEBackend = oracle.WNA16MoEBackend
for _name in ("PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM"):
    extend_enum(WNA16MoEBackend, _name, _name)
_PPU = (WNA16MoEBackend.PPU_DEEPGEMM, WNA16MoEBackend.BATCHED_PPU_DEEPGEMM)
_PPU_ALIASES = ("ppu_deep_gemm", "ppu_deep_gemm_w4a16")

# Captured before patching so every replacement can fall back to upstream.
_upstream_backend_to_kernel_cls = oracle.backend_to_kernel_cls
_upstream_get_priority_backends = oracle._get_priority_backends
_upstream_map_wna16_backend = oracle.map_wna16_backend
_upstream_select_wna16_moe_backend = oracle.select_wna16_moe_backend
_upstream_reason = oracle._backend_incompatibility_reason
_upstream_convert = oracle.convert_to_wna16_moe_kernel_format


@patch(_MODULE, "backend_to_kernel_cls", **_META)
def backend_to_kernel_cls(
    backend: WNA16MoEBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Return the experts class for the given backend, or None for NONE."""
    # PPU MODIFICATION: begin — upstream cannot resolve plugin experts.
    if backend == WNA16MoEBackend.PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExperts,
        )

        return [PPUDeepGemmExperts]
    if backend == WNA16MoEBackend.BATCHED_PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExperts,
        )

        return [PPUBatchedDeepGemmExperts]
    # PPU MODIFICATION: end
    return _upstream_backend_to_kernel_cls(backend)


@patch(_MODULE, "_get_priority_backends", **_META)
def _get_priority_backends() -> list[WNA16MoEBackend]:
    """Get available backends in priority order based on platform and config."""
    from vllm.platforms import current_platform

    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        return [
            WNA16MoEBackend.PPU_DEEPGEMM,
            WNA16MoEBackend.BATCHED_PPU_DEEPGEMM,
            WNA16MoEBackend.MARLIN,
            WNA16MoEBackend.BATCHED_MARLIN,
            WNA16MoEBackend.TRITON,
            WNA16MoEBackend.HUMMING,
            WNA16MoEBackend.EMULATION,
        ]
    # PPU MODIFICATION: end
    return _upstream_get_priority_backends()


@patch(_MODULE, "map_wna16_backend", **_META)
def map_wna16_backend(runner_backend: MoEBackend) -> WNA16MoEBackend:
    """Map user's MoEBackend to WNA16MoEBackend."""
    # PPU MODIFICATION: begin
    if runner_backend in _PPU_ALIASES:
        return WNA16MoEBackend.PPU_DEEPGEMM
    # PPU MODIFICATION: end
    return _upstream_map_wna16_backend(runner_backend)


def _select_batched_ppu_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey,
    quant_config: QuantizationConfig | QuantizationArgs,
    may_have_zp: bool,
    may_have_bias: bool,
    allow_tile_padding: bool,
) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
    """Resolve the batched-layout expert for an explicit PPU request."""
    source = oracle.WNA16MoEKernelOracle.maybe_skip_for_requantization_source()
    if source is not None:
        return source
    backend = WNA16MoEBackend.BATCHED_PPU_DEEPGEMM
    reason = _backend_incompatibility_reason(
        backend,
        config,
        quant_config,
        may_have_zp,
        may_have_bias,
        allow_tile_padding,
    )
    if reason is not None:
        raise ValueError(
            f"WNA16 MoE backend '{backend.value}' does not support the "
            f"deployment configuration since {reason}."
        )
    fallback = oracle._activation_fallback_reason(
        backend, oracle._requested_activation_key()
    )
    if fallback is not None:
        raise ValueError(fallback)
    activation_format = mk.FusedMoEActivationFormat.BatchedExperts
    for k_cls in backend_to_kernel_cls(backend):
        supported, reason = k_cls.is_supported_config(
            k_cls, config, weight_key, None, activation_format
        )
        if supported:
            oracle.logger.info_once(
                f"Using '{backend.value}' WNA16 MoE backend.", scope="local"
            )
            return backend, k_cls
    raise ValueError(
        f"WNA16 MoE backend '{backend.value}' does not support the "
        f"deployment configuration since {reason}."
    )


@patch(_MODULE, "select_wna16_moe_backend", **_META)
def select_wna16_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey,
    quant_config: QuantizationConfig | QuantizationArgs,
    may_have_zp: bool,
    may_have_bias: bool,
    allow_tile_padding: bool = False,
) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
    """Select the WNA16 MoE backend.

    Args:
        config: the shared ``FusedMoEConfig`` for this layer.
        weight_key: The QuantKey describing the weight quantization.
                    Must have int4 or int8 type.
        quant_config: Quantization structure and checkpoint format description.
        may_have_zp: Whether the integration can provide weight zero points.
        may_have_bias: Whether the integration can provide expert bias.
        allow_tile_padding: Whether backends that require padding the weights
            up to a tile boundary may be selected.

    Returns:
        A tuple of (``WNA16MoEBackend``, experts class or ``None``).
    """
    # PPU MODIFICATION: begin — an explicit PPU DeepGEMM request cannot use the
    # standard-layout expert when the layer runs the batched activation format.
    if config.moe_backend in _PPU_ALIASES and config.activation_format.is_batched:
        return _select_batched_ppu_backend(
            config,
            weight_key,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
    # PPU MODIFICATION: end
    return _upstream_select_wna16_moe_backend(
        config,
        weight_key,
        quant_config,
        may_have_zp,
        may_have_bias,
        allow_tile_padding,
    )


def make_wna16_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    backend: WNA16MoEBackend = WNA16MoEBackend.MARLIN,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUExpertsInt4,
    )
    from vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe import (
        Int4EmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.rdna3_int4_int8_moe import (
        Rdna3Int4Int8Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
        Rdna3WNA16Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsWNA16,
    )
    from vllm.model_executor.layers.fused_moe.experts.zentorch_moe import (
        ZentorchExpertsWNA16,
    )

    # PPU MODIFICATION: begin — plugin experts are not in upstream's allowlist.
    from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        PPUBatchedDeepGemmExperts,
    )
    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        PPUDeepGemmExperts,
    )
    # PPU MODIFICATION: end

    # Currently, we only support TrtLlmMxint4ExpertsMonolithic, MarlinExperts,
    # BatchedMarlinExperts, XPUExpertsWNA16, CPUExpertsInt4,
    # ZentorchExpertsWNA16, the Humming grouped/indexed experts, and
    # Int4EmulationTritonExperts
    allowed_experts: tuple[type[mk.FusedMoEExperts], ...] = (
        MarlinExperts,
        BatchedMarlinExperts,
        TritonWNA16Experts,
        # PPU MODIFICATION: begin
        PPUDeepGemmExperts,
        PPUBatchedDeepGemmExperts,
        # PPU MODIFICATION: end
        TrtLlmMxint4ExpertsMonolithic,
        XPUExpertsWNA16,
        CPUExpertsInt4,
        ZentorchExpertsWNA16,
        Int4EmulationTritonExperts,
        Rdna3WNA16Experts,
        Rdna3Int4Int8Experts,
    )
    if backend == WNA16MoEBackend.HUMMING:
        allowed_experts += tuple(backend_to_kernel_cls(WNA16MoEBackend.HUMMING))
    assert experts_cls in allowed_experts

    return make_moe_kernel(
        moe_quant_config,
        moe_config,
        experts_cls,
        routing_tables,
        experts_log="Using %s",
    )


make_wna16_moe_kernel = patch(_MODULE, "make_wna16_moe_kernel", **_META)(
    bind_body(make_wna16_moe_kernel, oracle)
)


@patch(_MODULE, "_backend_incompatibility_reason", **_META)
def _backend_incompatibility_reason(
    backend, moe_config, quant_config, may_have_zp, may_have_bias, allow_tile_padding
):
    if backend not in _PPU:
        return _upstream_reason(
            backend,
            moe_config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    from vllm_sail.patch.enhancement.models.moe_marlin_gate import (
        supports_marlin_layout,
    )

    if may_have_zp or may_have_bias or moe_config.has_bias:
        return "PPU W4A16 does not consume zero points or expert bias"
    if not isinstance(quant_config, AutoGPTQConfig | QuantizationArgs):
        return "PPU W4A16 requires GPTQ or compressed-tensors symmetric INT4 weights"
    if getattr(quant_config, "desc_act", False) or getattr(
        quant_config, "actorder", None
    ) in ("group", "dynamic"):
        return "PPU W4A16 does not consume GPTQ activation ordering"
    if quant_config.group_size != 32 or not supports_marlin_layout(
        moe_config, 32, allow_tile_padding
    ):
        return "PPU W4A16 requires group32 and Marlin-compatible tile dimensions"
    return None


@patch(_MODULE, "convert_to_wna16_moe_kernel_format", **_META)
def convert_to_wna16_moe_kernel_format(*args, **kwargs):
    bound = inspect.signature(_upstream_convert).bind(*args, **kwargs)
    bound.apply_defaults()
    if bound.arguments["backend"] in _PPU:
        # Shared load-time Marlin packing, followed by PPU DeepGEMM inference.
        bound.arguments["backend"] = WNA16MoEBackend.MARLIN
    return _upstream_convert(*bound.args, **bound.kwargs)
