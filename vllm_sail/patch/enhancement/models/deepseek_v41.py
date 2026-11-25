# SPDX-License-Identifier: Apache-2.0
"""Select the V4.1 PPU FlashMLA subclass at the CUDA selector boundary.

PPU MODIFICATION (M2, fused norm-rope): the selector now has a second PPU
branch. When the PR#54 fused norm-rope sparse attention path passes every
gate (env switch, flash_mla wheel capability, PPU SM80/SM89, 528B KV record
switch, V4.1 head/group topology), the PPU mega-attention subclass is
selected; otherwise the selector falls back to the existing PPU baseline
subclass, and one ``info_once`` line records *which* gate closed so every run
ledger can prove which attention path was live.
"""

import sys

from vllm.logger import init_logger
from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention as UpstreamFlashMLAAttention,
)
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v41.flashmla import DeepseekV41FlashMLAAttention
from vllm_sail.patch.utils import PATCH_MARKER, patch

logger = init_logger(__name__)

_MODULE = "vllm.models.deepseek_v41.nvidia.model"
_TARGET = f"{_MODULE}._select_dsv4_attn_cls"


@patch(
    _MODULE,
    "_select_dsv4_attn_cls",
    reason=(
        "PPU requires unpadded FlashMLA heads and an MXFP8 BF16 output projection "
        "while retaining V4.1 shared KV/index metadata and CUDA model logic. "
        "With the PR#54 FlashMLA wheel and both runtime switches on, PPU instead "
        "selects the fused norm-rope mega-attention subclass (528B KV record, "
        "padded 64/128 heads, QuantizedActivation output)."
    ),
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="vLLM exposes a V4.1 attention-class registry supporting PPU overrides.",
)
def _select_dsv4_attn_cls(vllm_config):
    original = getattr(_select_dsv4_attn_cls, PATCH_MARKER)[_TARGET]
    selected = original(vllm_config)
    if not current_platform.is_ppu():
        return selected

    # Lazy import: keeps patch installation cheap and guarantees a baseline
    # wheel (b1b0e1f, no fused API) can never crash at import time.
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
        DeepseekV4MegaAttnAttention as UpstreamMegaAttnAttention,
    )

    from vllm_sail.models.deepseek_v41.flash_mla_mega_attn import (
        PPUDeepseekV4MegaAttnAttention,
        fused_gate,
    )

    if selected in (
        UpstreamFlashMLAAttention,
        UpstreamMegaAttnAttention,
        DeepseekV41FlashMLAAttention,
    ):
        ok, gate_report = fused_gate(vllm_config)
        if ok and getattr(vllm_config, "lora_config", None) is not None:
            # Upstream guards mega attention against LoRA: the grouped wo_a
            # direct read cannot apply the adapter delta, so LoRA must stay
            # on the split-KV FlashMLA path.
            ok, gate_report = False, "lora: mega attention needs the split-KV path"
        if ok:
            logger.info_once(
                "[vllm_sail] DSV4.1 attention class: "
                "PPUDeepseekV4MegaAttnAttention (fused norm-rope; %s)",
                gate_report,
            )
            return PPUDeepseekV4MegaAttnAttention
        logger.info_once(
            "[vllm_sail] DSV4.1 fused norm-rope gate closed (%s); "
            "falling back to the baseline attention class",
            gate_report,
        )

    if selected in (UpstreamFlashMLAAttention, UpstreamMegaAttnAttention):
        return DeepseekV41FlashMLAAttention
    return selected


@patch(
    _MODULE,
    "_linear_scale_param_name",
    reason="PPU channelwise FP8 dense layers register weight_scale for V4.1 target and draft loaders.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The V4.1 scale-name helper queries the linear quantization method's parameter contract.",
)
def _linear_scale_param_name(vllm_config, expert_dtype):
    if current_platform.is_ppu() and getattr(
        vllm_config.quant_config, "fp8_channelwise_layers", None
    ):
        return "weight_scale"
    original = getattr(_linear_scale_param_name, PATCH_MARKER)[
        f"{_MODULE}._linear_scale_param_name"
    ]
    return original(vllm_config, expert_dtype)


# The VL wrapper and DSpark loader import the helper by value.
def _rebind_scale_aliases():
    original = getattr(_linear_scale_param_name, PATCH_MARKER)[
        f"{_MODULE}._linear_scale_param_name"
    ]
    for name in (
        "vllm.models.deepseek_v41.nvidia.vl_model",
        "vllm.models.deepseek_v41.nvidia.dspark",
    ):
        module = sys.modules.get(name)
        if (
            module is not None
            and getattr(module, "_linear_scale_param_name", None) is original
        ):
            patch(
                name,
                "_linear_scale_param_name",
                reason="Preloaded V4.1 target and draft consumers must retain PPU channelwise scale names.",
                affected_versions=">=0.32.0,<0.33.0",
                remove_when="V4.1 loaders resolve the scale-name helper through its provider module.",
            )(_linear_scale_param_name)


_rebind_scale_aliases()
