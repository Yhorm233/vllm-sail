# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Allow sinks and FP8 for the PPU FA3 implementation."""

from __future__ import annotations

from vllm.v1.attention.backends import flash_attn as _flash_attn

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch


def supports_combination(
    cls,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: CacheDType | None,
    block_size: int | None,
    use_mla: bool,
    has_sink: bool,
    use_sparse: bool,
    use_mm_prefix: bool,
    device_capability: DeviceCapability,
) -> str | None:
    if (
        device_capability == DeviceCapability(9, 0)
        and kv_cache_dtype is not None
        and is_quantized_kv_cache(kv_cache_dtype)
        and head_size == 512
    ):
        if block_size is not None and block_size % 64 != 0:
            return "SM90 d512 FP8 KV requires a block size divisible by 64"
        if use_sparse:
            return "SM90 d512 FP8 KV does not support block sparsity"
    if has_sink and device_capability < DeviceCapability(9, 0):
        # PPU MODIFICATION: begin
        # PPU FA3 takes s_aux (sinks) through flash_attn_3.fwd; FA2 does not.
        fa_version = get_flash_attn_version(
            head_size=head_size, head_size_v=head_size, has_sinks=has_sink
        )
        if not (current_platform.is_ppu() and fa_version == 3):
            return "sink not supported on compute capability < 9.0"
        # PPU MODIFICATION: end
    if (
        use_mm_prefix
        and kv_cache_dtype is not None
        and is_quantized_kv_cache(kv_cache_dtype)
        and device_capability == DeviceCapability(9, 0)
    ):
        return "SM90 FP8 KV with mm_prefix requires Triton"
    if (
        kv_cache_dtype is not None
        and is_quantized_kv_cache(kv_cache_dtype)
        and not flash_attn_supports_kv_cache_dtype(
            kv_cache_dtype,
            head_size=head_size,
            head_size_v=getattr(cls, "head_size_v", None),
            has_sinks=has_sink,
        )
    ):
        # PPU MODIFICATION: begin
        return "FP8 KV requires FA3 (SM90/PPU sm_89), equal-d512 FA4 (SM90), or FA4 (SM100)."
        # PPU MODIFICATION: end
    if (
        use_mm_prefix
        and get_flash_attn_version(
            head_size=head_size,
            has_sinks=has_sink,
        )
        != 4
    ):
        return (
            "mm_prefix (PrefixLM bidirectional attention) requires "
            "FlashAttention v4, which does not resolve for this "
            "head_size"
        )
    return None


patch(
    "vllm.v1.attention.backends.flash_attn",
    "FlashAttentionBackend.supports_combination",
    reason="The upstream sink combination gate assumes only NVIDIA SM90 supports FA3 sinks.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="FlashAttentionBackend gates sinks through the selected implementation's capabilities.",
)(bind_body(supports_combination, _flash_attn))
