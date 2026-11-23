# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Marlin MoE gate — phase-1 decision D2, option (c).

The fork's Marlin support needs in-place edits to upstream CUDA sources
(``csrc/libtorch_stable/quantization/marlin/*``,
``csrc/libtorch_stable/moe/marlin_moe_wna16/*``) that a plugin cannot make.
Until those changes are upstreamed, PPU stays on non-Marlin MoE paths:
``check_moe_marlin_supports_config`` returns False on PPU unless
``VLLM_PPU_ENABLE_MOE_MARLIN`` is explicitly set (default off; declared in
``vllm_sail/envs.py``).

旧开关只控制 WNA16 的布局准入，不能提供被排除的 Marlin 计算内核。
专家类和 dense 的计算准入必须独立拒绝 PPU；非 PPU 完整委托 upstream。
保留原布局检查和 portable repack，供 PPU DeepGEMM 加载权重使用。
公共 check/verify 也用于 MP 方法的量化配置验证，不能在选择非 Marlin 内核前拒绝。
NVFP4 自动选择保留完整 upstream body，只取消 PPU 的强制 Marlin 快捷路径。
"""

from __future__ import annotations

import sys

from vllm.model_executor.kernels import linear as linear_kernels
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExpertsBase
from vllm.model_executor.layers.quantization.utils import (
    marlin_utils_fp4,
    marlin_utils_fp8,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_config,
)
from vllm.platforms import current_platform

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"

# Captured before patching so the replacement can delegate to upstream.
_upstream_check = check_moe_marlin_supports_config
# The layout check is also used by PPU DeepGEMM weight preparation.
supports_marlin_layout = _upstream_check


@patch(
    "vllm.model_executor.layers.quantization.utils.marlin_utils",
    "check_moe_marlin_supports_config",
    reason=(
        "The fork's Marlin kernels are in-place edits of upstream .cu/.h "
        "sources that a plugin cannot ship; v1 restricts PPU to non-Marlin "
        "MoE paths (phase-1 spec 1.8, decision (c)). Delegates to upstream "
        "unless PPU has the opt-out env var left at its default (off)."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the Marlin PPU kernel edits are upstreamed (or rebuilt inside "
        "vllm_sail._moe_C) and VLLM_PPU_ENABLE_MOE_MARLIN defaults on."
    ),
)
def gated_check_moe_marlin_supports_config(*args, **kwargs):
    if current_platform.is_ppu() and not ppu_envs.VLLM_SAIL_ENABLE_MOE_MARLIN:
        return False
    return _upstream_check(*args, **kwargs)


for _consumer in (
    "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16",
):
    _module = sys.modules.get(_consumer)
    if (
        _module is not None
        and getattr(_module, "check_moe_marlin_supports_config", None)
        is _upstream_check
    ):
        patch(
            _consumer,
            "check_moe_marlin_supports_config",
            reason="Preloaded WNA16 consumers must observe the PPU Marlin capability gate.",
            affected_versions=_AFFECTED,
            remove_when="WNA16 consumers resolve Marlin capabilities through the provider module.",
        )(gated_check_moe_marlin_supports_config)


_COMPUTE_REASON = "PPU 的 CUDA 兼容标识会通过 upstream Marlin 检查，但生成 manifest 排除了 Marlin 计算内核。"
_COMPUTE_REMOVE_WHEN = "Marlin 计算内核在 PPU manifest 中编译，并完成相应 device 验证。"
_NO_PPU_MARLIN = "PPU 没有 Marlin 计算内核；portable 权重重排不代表 Marlin GEMM 可用。"
_upstream_device_check = MarlinExpertsBase._supports_current_device
_upstream_can_implement = MarlinLinearKernel.can_implement.__func__
_upstream_fp4_check = marlin_utils_fp4.is_fp4_marlin_supported
_upstream_fp8_check = marlin_utils_fp8.is_fp8_marlin_supported


@patch(
    "vllm.model_executor.layers.fused_moe.experts.marlin_moe",
    "MarlinExpertsBase._supports_current_device",
    reason=_COMPUTE_REASON,
    affected_versions=_AFFECTED,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_marlin_experts_device():
    if current_platform.is_ppu():
        return False
    return _upstream_device_check()


@patch(
    "vllm.model_executor.kernels.linear.mixed_precision.marlin",
    "MarlinLinearKernel.can_implement",
    reason=_COMPUTE_REASON,
    affected_versions=_AFFECTED,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_marlin_linear_can_implement(cls, *args, **kwargs):
    if current_platform.is_ppu():
        return False, _NO_PPU_MARLIN
    return _upstream_can_implement(cls, *args, **kwargs)


@patch(
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4",
    "is_fp4_marlin_supported",
    reason=_COMPUTE_REASON,
    affected_versions=_AFFECTED,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_fp4_marlin_supported(*args, **kwargs):
    if current_platform.is_ppu():
        return False
    return _upstream_fp4_check(*args, **kwargs)


@patch(
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8",
    "is_fp8_marlin_supported",
    reason=_COMPUTE_REASON,
    affected_versions=_AFFECTED,
    remove_when=_COMPUTE_REMOVE_WHEN,
)
def gated_fp8_marlin_supported(*args, **kwargs):
    if current_platform.is_ppu():
        return False
    return _upstream_fp8_check(*args, **kwargs)


# 仅模块级按值缓存需要重绑；函数内导入会直接取得 patched provider。
for _consumer, _name, _original, _replacement in (
    (
        "vllm.model_executor.kernels.linear.nvfp4.marlin",
        "is_fp4_marlin_supported",
        _upstream_fp4_check,
        gated_fp4_marlin_supported,
    ),
    (
        "vllm.model_executor.kernels.linear.scaled_mm.marlin",
        "is_fp8_marlin_supported",
        _upstream_fp8_check,
        gated_fp8_marlin_supported,
    ),
):
    _module = sys.modules.get(_consumer)
    if _module is not None and getattr(_module, _name, None) is _original:
        patch(
            _consumer,
            _name,
            reason="预加载的 dense Marlin 消费者必须观察同一 PPU 计算准入。",
            affected_versions=_AFFECTED,
            remove_when="消费者不再按值缓存 Marlin capability probe。",
        )(_replacement)


# 完整借用第七日 upstream 的 NVFP4 selector；绑定真实 provider 的活跃 globals。
def init_nvfp4_linear_kernel(use_a16: bool = False) -> NvFp4LinearKernel:
    """Select and instantiate the best NVFP4 linear kernel for the
    current platform."""
    config = NvFp4LinearLayerConfig()
    a16_kernels = (
        B12xNvFp4W4A16LinearKernel,
        FlashInferCuteDslNvFp4W4A16LinearKernel,
        MarlinNvFp4LinearKernel,
        HummingNvFp4LinearKernel,
    )

    # VLLM_BATCH_INVARIANT forces deterministic execution: Humming for W4A16,
    # otherwise CUTLASS, or emulation where CUTLASS is unsupported. It
    # overrides --linear-backend.
    force_kernel: type[NvFp4LinearKernel] | None = None
    quantization = "nvfp4_w4a16" if use_a16 else "nvfp4_w4a4"
    linear_backend = _get_linear_backend(quantization=quantization)
    if envs.VLLM_BATCH_INVARIANT:
        if use_a16:
            force_kernel, bi_backend = HummingNvFp4LinearKernel, "humming"
        else:
            cutlass_supported, reason = CutlassNvFp4LinearKernel.is_supported()
            if cutlass_supported:
                force_kernel, bi_backend = CutlassNvFp4LinearKernel, "cutlass"
            else:
                logger.info_once(
                    "VLLM_BATCH_INVARIANT is set but the batch-invariant NVFP4 "
                    "kernel is not supported on this platform; falling back to "
                    "emulation for deterministic execution. Reason: %s",
                    reason,
                )
                force_kernel, bi_backend = EmulationNvFp4LinearKernel, "emulation"
        if linear_backend not in ("auto", bi_backend):
            logger.warning_once(
                "VLLM_BATCH_INVARIANT overrides --linear-backend=%s; using the "
                "%s backend for deterministic execution.",
                linear_backend,
                bi_backend,
            )
    elif linear_backend == "auto" and use_a16:
        _cc = current_platform.get_device_capability()
        compute_capability = _cc.to_int() if _cc is not None else None
        # Weight-only: prefer FlashInfer CuTe-DSL W4A16 on SM100/103,
        # Humming then Marlin where Humming is preferred, and Marlin elsewhere.
        cutedsl_ok, _ = FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported(
            compute_capability
        )
        if compute_capability in (100, 103) and cutedsl_ok:
            force_kernel = FlashInferCuteDslNvFp4W4A16LinearKernel
        elif not prefers_humming(compute_capability):
            # PPU MODIFICATION: begin
            # PPU 没有 Marlin 计算内核；继续 registry 选择，不能强制后再报错。
            if not current_platform.is_ppu():
                force_kernel = MarlinNvFp4LinearKernel
            # PPU MODIFICATION: end

    if force_kernel is not None:
        if use_a16 and force_kernel not in a16_kernels:
            raise ValueError(f"{force_kernel.__name__} does not support W4A16")
        is_supported, reason = force_kernel.is_supported()
        if not is_supported:
            raise ValueError(
                f"Forced NVFP4 kernel {force_kernel.__name__} is not "
                f"supported: {reason}"
            )
        logger.info_once("Using %s for NVFP4 GEMM", force_kernel.__name__)
        return force_kernel(config)

    # Auto-select from registry (or --linear-backend filtered).
    platform = current_platform._enum
    possible = list(_POSSIBLE_NVFP4_KERNELS.get(platform, []))
    if use_a16:
        possible = [kernel for kernel in possible if kernel in a16_kernels]

    # Apply --linear-backend filtering when set.
    possible = _resolve_backend_kernels(
        possible,
        "NVFP4",
        quantization=quantization,
    )

    failure_reasons = []
    for kernel_cls in possible:
        if kernel_cls.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f" {kernel_cls.__name__} disabled by environment variable"
            )
            continue

        is_supported, reason = kernel_cls.is_supported()
        if not is_supported:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        can_implement, reason = kernel_cls.can_implement(config)
        if not can_implement:
            failure_reasons.append(f"{kernel_cls.__name__}: {reason}")
            continue

        if kernel_cls is EmulationNvFp4LinearKernel and failure_reasons:
            logger.warning_once(
                "NVFP4 linear falling back to the slow and unoptimized "
                "emulation backend as no optimized backend is available "
                "(unavailable reasons:\n - %s\n). "
                "In case you expect one of these backends to be used, "
                "please verify your environment.",
                "\n - ".join(failure_reasons),
            )

        logger.info_once("Using %s for NVFP4 GEMM", kernel_cls.__name__)
        return kernel_cls(config)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "NVFP4 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


_upstream_init_nvfp4 = linear_kernels.init_nvfp4_linear_kernel
_ppu_init_nvfp4 = bind_body(init_nvfp4_linear_kernel, linear_kernels)


@patch(
    "vllm.model_executor.kernels.linear",
    "init_nvfp4_linear_kernel",
    reason="PPU 的 NVFP4 自动选择不能强制不可用的 Marlin 而绕过 registry 回退。",
    affected_versions=_AFFECTED,
    remove_when="upstream 的自动强制路径先检查计算内核支持并继续选择可用后端。",
)
def gated_init_nvfp4_linear_kernel(*args, **kwargs):
    if current_platform.is_ppu():
        return _ppu_init_nvfp4(*args, **kwargs)
    return _upstream_init_nvfp4(*args, **kwargs)


for _consumer in (
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4",
    "vllm.model_executor.layers.quantization.modelopt",
    "vllm.model_executor.layers.quantization.online.lm_head",
    "vllm.model_executor.layers.quantization.quark.schemes.quark_nvfp4",
):
    _module = sys.modules.get(_consumer)
    if (
        _module is not None
        and getattr(_module, "init_nvfp4_linear_kernel", None) is _upstream_init_nvfp4
    ):
        patch(
            _consumer,
            "init_nvfp4_linear_kernel",
            reason="预加载的 NVFP4 消费者必须观察同一 PPU 自动选择策略。",
            affected_versions=_AFFECTED,
            remove_when="消费者不再按值缓存 NVFP4 selector。",
        )(gated_init_nvfp4_linear_kernel)
