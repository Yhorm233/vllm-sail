# SPDX-License-Identifier: Apache-2.0
"""Dependency-free normalization for PPU V4.1 mixed-precision exports."""

import re

# Resolved once while constructing the V4.1 target model. Module-free operators
# read it at forward time instead of querying the global vLLM config.
_int8_w8a8 = False


def set_int8_w8a8(enabled):
    """Record the construction-time W8A8 decision; the last target model wins."""
    global _int8_w8a8
    _int8_w8a8 = bool(enabled)


def int8_w8a8_enabled():
    return _int8_w8a8


def uses_int8_w8a8(vllm_config):
    """Accept only V4.1 symmetric W8A8 channel/token compressed-tensors contracts."""
    quant = getattr(vllm_config, "quant_config", None)
    if quant is None or quant.get_name() != "compressed-tensors":
        return False
    model = getattr(vllm_config, "model_config", None)
    hf = getattr(model, "hf_config", None)
    if getattr(hf, "model_type", None) not in ("deepseek_v41", "deepseek_v41_text"):
        return False
    raw = getattr(hf, "quantization_config", None) or {}
    if raw.get("expert_dtype") != "int8" or raw.get("format") != "int-quantized":
        return False
    for group in raw.get("config_groups", {}).values():
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if (
            "Linear" in group.get("targets", [])
            and weight.get("type") == activation.get("type") == "int"
            and weight.get("num_bits") == activation.get("num_bits") == 8
            and weight.get("strategy") == "channel"
            and activation.get("strategy") == "token"
            and weight.get("symmetric") is True
            and activation.get("symmetric") is True
            and weight.get("dynamic") is False
            and activation.get("dynamic") is True
        ):
            return True
    return False


def normalize_expert_dtype(vllm_config):
    """Canonicalize the exporter alias in resolved configs, never checkpoint files."""
    configs = [getattr(vllm_config, "model_config", None)]
    speculative = getattr(vllm_config, "speculative_config", None)
    configs.append(getattr(speculative, "draft_model_config", None))
    for model in configs:
        for attr in ("hf_config", "hf_text_config"):
            hf = getattr(model, attr, None)
            if getattr(hf, "model_type", None) not in (
                "deepseek_v41",
                "deepseek_v41_text",
            ):
                continue
            quant = getattr(hf, "quantization_config", None) or {}
            if quant.get("quant_method") == "mxfp4" and quant.get(
                "fp8_channelwise_layers"
            ):
                if getattr(hf, "expert_dtype", None) == "mxfp4":
                    hf.expert_dtype = "fp4"


def checkpoint_layer_name(prefix: str, num_hidden_layers: int) -> str:
    """Reverse V4.1 runtime prefixes, including DSpark's absolute layer indices."""
    prefix = prefix.removeprefix("language_model.").removeprefix("model.")
    match = re.match(r"layers\.(\d+)\.(.*)", prefix)
    if match and int(match[1]) >= num_hidden_layers:
        prefix = f"mtp.{int(match[1]) - num_hidden_layers}.{match[2]}"
    if prefix == "main_proj":
        prefix = "mtp.0.main_proj"
    return prefix.replace(".shared_experts.down_proj", ".shared_experts.w2")
