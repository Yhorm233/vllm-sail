# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""PPU FA3 default and FP8 support, before upstream compatibility fallbacks."""

from __future__ import annotations

import sys

from vllm.v1.attention.backends import fa_utils as _fa_utils

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.v1.attention.backends.fa_utils"
_AFFECTED = ">=0.32.0,<0.33.0"


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    requires_softcap: bool = False,
    kv_cache_block_size: int | None = None,
    supports_fa4_hd256: bool = False,
) -> int | None:
    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        # PPU MODIFICATION: begin
        elif current_platform.is_ppu() and is_fa_version_supported(3):
            # PPU NOTE (kai): prefer FA3
            fa_version = 3
        # PPU MODIFICATION: end
        else:
            # Fallback to FA2
            fa_version = 2

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        # Some FA3 unsupported SM90 cases can use FA4 when available.
        if (
            fa_version == 3
            and device_capability.major == 9
            and is_fa_version_supported(4)
        ):
            upgrade_reason = None
            if head_size is not None and head_size > 256:
                upgrade_reason = f"FA3 does not support head_size={head_size} on SM90"
            elif (
                has_sinks
                and head_size is not None
                and head_size_v is not None
                and head_size != head_size_v
            ):
                upgrade_reason = "Diff-KV with sinks"
            elif (
                vllm_config is not None
                and vllm_config.model_config is not None
                and vllm_config.model_config.is_diffusion
            ):
                upgrade_reason = "Per-sequence causal (dynamic_causal) requires FA4"
            if upgrade_reason:
                logger.info_once(
                    "%s: upgrading FlashAttention 3 -> 4",
                    upgrade_reason,
                    scope="local",
                )
                fa_version = 4

        # FA4 currently uses batch-shape-dependent scheduling
        # heuristics on SM100+, which breaks batch invariance.
        if envs.VLLM_BATCH_INVARIANT and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, "
                "defaulting to FA version 2.",
            )
            fa_version = 2

        if fa_version == 4 and uses_fa4_hd256_kernel(head_size, head_size_v):
            if not supports_fa4_hd256:
                fa_version = 2
            elif (
                reason := _fa4_hd256_fallback_reason(
                    has_sinks, requires_softcap, kv_cache_block_size, vllm_config
                )
            ) is not None:
                logger.warning_once(
                    "FA4's Blackwell head_size=256 kernel does not support %s, "
                    "defaulting to FA version 2.",
                    reason,
                )
                fa_version = 2

        # FA4 head dimensions on Blackwell are limited by TMEM capacity.
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and not (
                (head_size == 256 and head_size_v in (None, 256))
                or (head_size == 192 and head_size_v == 128)
            )
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


get_flash_attn_version = patch(
    _MODULE,
    "get_flash_attn_version",
    reason="PPU prefers FA3 before applying configuration and compatibility fallbacks, including ALiBi.",
    affected_versions=_AFFECTED,
    remove_when="The FlashAttention capability resolver supports PPU through a platform hook.",
)(bind_body(get_flash_attn_version, _fa_utils))


def flash_attn_supports_kv_cache_dtype(
    kv_cache_dtype: str = "fp8_e4m3",
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    requires_softcap: bool = False,
    kv_cache_block_size: int | None = None,
    supports_fa4_hd256: bool = False,
) -> bool:
    if kv_cache_dtype == "fp8_e5m2":
        return False
    if current_platform.is_xpu():
        return True
    fa_version = get_flash_attn_version(
        requires_alibi=requires_alibi,
        head_size=head_size,
        head_size_v=head_size_v,
        has_sinks=has_sinks,
        requires_softcap=requires_softcap,
        kv_cache_block_size=kv_cache_block_size,
        supports_fa4_hd256=supports_fa4_hd256,
    )
    is_sm90 = current_platform.is_device_capability_family(90)
    sm90_fp8_kv_supported = fa_version == 3 or (
        fa_version == 4 and head_size in (None, 512) and head_size_v in (None, 512)
    )
    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        # PPU FA3 takes q/k/v descales; sm_80 has no FP8 tensor support.
        return fa_version == 3 and current_platform.supports_fp8()
    # PPU MODIFICATION: end
    return (sm90_fp8_kv_supported and is_sm90) or (
        fa_version == 4 and current_platform.is_device_capability_family(100)
    )


flash_attn_supports_kv_cache_dtype = patch(
    _MODULE,
    "flash_attn_supports_kv_cache_dtype",
    reason="PPU 1.5 FA3 supports FP8 descales; the NVIDIA SM90 gate excludes PPU.",
    affected_versions=_AFFECTED,
    remove_when="The FlashAttention capability resolver supports PPU through a platform hook.",
)(bind_body(flash_attn_supports_kv_cache_dtype, _fa_utils))

_CONSUMERS = {
    "get_flash_attn_version": (
        "vllm.model_executor.layers.attention.sparse_mla_attention",
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        "vllm.v1.attention.backends.flash_attn_diffkv",
        "vllm.v1.attention.backends.mla.flashattn_mla",
        "vllm.v1.attention.backends.flash_attn",
        "vllm.v1.attention.backends.turboquant_attn",
        "vllm.v1.attention.backends.mla.prefill.flash_attn",
    ),
    "flash_attn_supports_kv_cache_dtype": ("vllm.v1.attention.backends.flash_attn",),
}
for _name, _consumers in _CONSUMERS.items():
    _replacement = getattr(_fa_utils, _name)
    _original = getattr(_replacement, PATCH_MARKER)[f"{_MODULE}.{_name}"]
    for _consumer in _consumers:
        _module = sys.modules.get(_consumer)
        if _module is not None and getattr(_module, _name, None) is _original:
            patch(
                _consumer,
                _name,
                reason="A preloaded attention consumer must use PPU FA capabilities.",
                affected_versions=_AFFECTED,
                remove_when="Attention consumers resolve FA capabilities through their provider module.",
            )(_replacement)
