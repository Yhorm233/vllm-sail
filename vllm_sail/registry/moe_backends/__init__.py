# SPDX-License-Identifier: Apache-2.0
"""PPU MoE backend registration helpers."""

from __future__ import annotations

from vllm_sail.registry.moe_backends._extend import extend_enum

__all__ = ["extend_enum", "register"]

_registered = False
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "vLLM imports MoE selectors through a registration API instead of "
    "capturing module-level aliases before plugin registration."
)

# Module-scope consumers captured before plugin registration at the pinned vLLM.
_ORACLE_ALIAS_CONSUMERS = {
    "unquantized": {
        "select_unquantized_moe_backend": (
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
        )
    },
    "fp8": {
        "backend_to_kernel_cls": ("vllm.model_executor.layers.fused_moe.oracle.mxfp8",),
        "select_fp8_moe_backend": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8",
            "vllm.model_executor.layers.quantization.fp8",
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.online.fp8",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "convert_to_fp8_moe_kernel_format": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8",
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8",
            "vllm.model_executor.layers.quantization.fp8",
            "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp8_moe",
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "make_fp8_moe_quant_config": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8",
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8",
            "vllm.model_executor.layers.quantization.fp8",
            "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp8_moe",
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
    },
    "int8": {
        "convert_to_int8_moe_kernel_format": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
            "vllm.model_executor.layers.quantization.online.int8",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "make_int8_moe_quant_config": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
            "vllm.model_executor.layers.quantization.online.int8",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "select_int8_moe_backend": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
            "vllm.model_executor.layers.quantization.online.int8",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
    },
    "mxfp4": {
        "backend_to_kernel_cls": (
            "vllm.model_executor.layers.quantization.mxfp4",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "_backend_activation_key": (
            "vllm.model_executor.layers.quantization.online.mxfp4",
        ),
        "convert_gpt_oss_weight_to_mxfp4_moe_kernel_format": (
            "vllm.model_executor.layers.quantization.mxfp4",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "convert_weight_to_mxfp4_moe_kernel_format": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4",
            "vllm.model_executor.layers.quantization.mxfp4",
            "vllm.model_executor.layers.quantization.online.mxfp4",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "make_mxfp4_moe_quant_config": (
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4",
            "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe",
            "vllm.model_executor.layers.quantization.mxfp4",
            "vllm.model_executor.layers.quantization.online.fp8_requant",
            "vllm.model_executor.layers.quantization.online.mxfp4",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "mxfp4_round_up_hidden_size_and_intermediate_size": (
            "vllm.model_executor.layers.quantization.mxfp4",
            "vllm.model_executor.layers.quantization.online.mxfp4",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        # select_mxfp4_moe_backend is absent on purpose: the PPU differences
        # live in the helpers it calls (map_mxfp4_backend,
        # backend_to_kernel_cls, _backend_activation_key), so upstream's selector
        # object is never replaced and a consumer's captured alias stays valid.
        "select_deepseek_v4_mxfp4_moe_backend": (
            "vllm.model_executor.layers.quantization.mxfp4",
        ),
    },
    "int_wna16": {
        "backend_to_kernel_cls": (),
        "select_wna16_moe_backend": (
            "vllm.model_executor.layers.quantization.auto_awq",
            "vllm.model_executor.layers.quantization.auto_gptq",
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16",
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.moe_wna16",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "make_wna16_moe_kernel": (
            "vllm.model_executor.layers.quantization.auto_awq",
            "vllm.model_executor.layers.quantization.auto_gptq",
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16",
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.moe_wna16",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
        "convert_to_wna16_moe_kernel_format": (
            "vllm.model_executor.layers.quantization.auto_awq",
            "vllm.model_executor.layers.quantization.auto_gptq",
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16",
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.moe_wna16",
            "vllm.model_executor.layers.quantization.quark.quark_moe",
        ),
    },
}


def _rebind_loaded_oracle_aliases(backend_modules: dict[str, object]) -> None:
    """Update vLLM 0.31 consumers that captured patched oracle functions."""
    import importlib
    import sys

    from vllm_sail.patch.utils import PATCH_MARKER, patch

    for backend_name, aliases in _ORACLE_ALIAS_CONSUMERS.items():
        # Access through the upstream oracle module after importing the plugin
        # backend module: the latter installs its replacements on the former.
        if backend_name not in backend_modules:
            raise RuntimeError(f"PPU MoE backend module {backend_name} was not loaded")
        oracle_name = f"vllm.model_executor.layers.fused_moe.oracle.{backend_name}"
        oracle_module = importlib.import_module(oracle_name)
        for alias_name, consumer_names in aliases.items():
            replacement = getattr(oracle_module, alias_name)
            target = f"{oracle_name}.{alias_name}"

            for consumer_name in consumer_names:
                consumer = sys.modules.get(consumer_name)
                if consumer is None:
                    continue

                captured = getattr(consumer, alias_name)
                if captured is replacement:
                    continue

                originals = getattr(replacement, PATCH_MARKER, {})
                upstream = originals.get(target)
                if upstream is None or captured is not upstream:
                    raise RuntimeError(
                        f"{consumer_name}.{alias_name} is not the expected "
                        "vLLM 0.31 oracle alias"
                    )

                patch(
                    consumer_name,
                    alias_name,
                    reason=(
                        "vLLM 0.31 captured this MoE oracle function before "
                        "PPU registration, so the consumer would bypass PPU "
                        "DeepGEMM selection or preparation."
                    ),
                    affected_versions=_AFFECTED,
                    remove_when=_REMOVE_WHEN,
                )(replacement)


def register() -> None:
    """Extend the MoE backend enums and patch their oracles. Idempotent."""
    global _registered
    if _registered:
        return

    import importlib

    backend_modules = {
        name: importlib.import_module(f"{__name__}.{name}")
        for name in _ORACLE_ALIAS_CONSUMERS
    }
    _rebind_loaded_oracle_aliases(backend_modules)
    _registered = True
