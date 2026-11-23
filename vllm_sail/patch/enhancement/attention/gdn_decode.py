# SPDX-License-Identifier: Apache-2.0
"""Select PPU GDN decode before upstream chooses a fused execution path."""

from __future__ import annotations

import torch
from vllm import envs as upstream_envs
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.attention.gdn_decode import (
    has_fused_gdn_kernel,
    resolve_gdn_decode_backend,
)
from vllm_sail.patch.utils import patch

logger = init_logger(__name__)
_META = dict(
    reason="PPU GDN decode selects PLA, native CUDA or Triton using real kernel capabilities, not stub schemas, and hashes the PLA preference for JIT/AOT caches.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="GDN exposes a platform decode backend resolver and native implementation capability checks.",
)

_upstream_getter = upstream_envs.environment_variables["VLLM_GDN_DECODE_KERNEL"]


def _decode_kernel():
    if current_platform.is_ppu():
        return envs.VLLM_GDN_DECODE_KERNEL
    return _upstream_getter()


class _EnvironmentVariables(dict):
    """Retain dict behavior while allowing the patch framework's metadata."""


# Replace the table rather than only __getattr__: compile_factors and the env
# cache also read these getters directly. All unrelated getters retain identity.
_getters = _EnvironmentVariables(upstream_envs.environment_variables)
_getters["VLLM_GDN_DECODE_KERNEL"] = _decode_kernel
patch("vllm.envs", "environment_variables", **_META)(_getters)

_upstream_compile_factors = upstream_envs.compile_factors


@patch("vllm.envs", "compile_factors", **_META)
def compile_factors():
    factors = _upstream_compile_factors()
    if not current_platform.is_ppu():
        return factors
    # auto 的 PLA 偏好会改变层的执行路径；JIT/AOT 共享此环境因子入口。
    # 仅追加已解析的规范值，保留上游忽略项、其他 getter 和非 PPU 行为。
    return {**factors, "VLLM_SAIL_USE_PLA": envs.VLLM_SAIL_USE_PLA}


_upstream_init = QwenGatedDeltaNetAttention.__init__
_upstream_unsupported_reason = (
    QwenGatedDeltaNetAttention._fused_gdn_decode_unsupported_reason
)
_upstream_can_use_mtp = QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode


def _pla_unsupported_reason(layer):
    if layer.head_k_dim != 128 or layer.head_v_dim != 128:
        return "PLA decode requires K=V=128"
    if layer.get_state_dtype()[1] != torch.float32:
        return (
            "PLA decode requires float32 SSM state; set --mamba-ssm-cache-dtype float32"
        )
    from vllm_sail.attention.pla_decode import (
        get_sail_cuda_pla_k_last,
        get_sail_cuda_pla_k_last_packed,
    )

    try:
        if (
            get_sail_cuda_pla_k_last() is not None
            and get_sail_cuda_pla_k_last_packed() is not None
        ):
            return None
    except ImportError as exc:
        return str(exc)
    return "PLA decode kernels are unavailable; install a compatible PLA library"


@patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "QwenGatedDeltaNetAttention.__init__",
    **_META,
)
def __init__(self, *args, **kwargs):
    _upstream_init(self, *args, **kwargs)
    if not current_platform.is_ppu():
        return
    requested = self.gdn_decode_kernel
    # Upstream __init__ has already validated an explicit CUDA request. Avoid
    # optional PLA imports for cuda/triton or a legacy opt-out.
    use_pla = envs.VLLM_SAIL_USE_PLA
    pla_reason = "PLA disabled by VLLM_SAIL_USE_PLA"
    if requested == "pla" or (requested == "auto" and use_pla):
        pla_reason = _pla_unsupported_reason(self)
    cuda_reason = None
    if requested == "auto":
        vllm_config = kwargs.get("vllm_config", args[1] if len(args) > 1 else None)
        cuda_reason = self._fused_gdn_decode_unsupported_reason(vllm_config)
    choice = resolve_gdn_decode_backend(
        requested, use_pla=use_pla, pla_reason=pla_reason, cuda_reason=cuda_reason
    )
    self.gdn_decode_kernel = choice.active
    self.enable_fused_gdn_spec_decode = (
        choice.active == "cuda" and self.speculative_config is not None
    )
    logger.info_once(
        "PPU GDN decode: requested=%s, active=%s, fallback_reason=%s",
        choice.requested,
        choice.active,
        choice.reason,
    )


@patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "QwenGatedDeltaNetAttention._fused_gdn_decode_unsupported_reason",
    **_META,
)
def _fused_gdn_decode_unsupported_reason(self, vllm_config):
    if current_platform.is_ppu() and not has_fused_gdn_kernel():
        return "_C::fused_gdn_decode_post_conv_mtp has no CUDA implementation (a schema or stub does not provide one)"
    return _upstream_unsupported_reason(self, vllm_config)


@patch(
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode",
    **_META,
)
def _can_use_fused_gdn_mtp_decode(self, attn_metadata):
    if current_platform.is_ppu() and not has_fused_gdn_kernel():
        return False
    return _upstream_can_use_mtp(self, attn_metadata)
