# SPDX-License-Identifier: Apache-2.0
"""PPU fp8-channelwise plumbing for the CompressedTensors quantization family.

The fork modifies three compressed_tensors files:

* ``compressed_tensors.py`` (+53): ``CompressedTensorsConfig`` gains
  ``fp8_channelwise_layers`` (constructor/from_config plumbing plus the
  ``_is_fp8_channelwise_layer`` helper), ``get_scheme`` overrides matched dense
  layers with an FP8 channelwise W8A8 scheme on PPU, and the KV-cache
  ``create_weights`` marks its scale/zero-point parameters non-trainable.
* ``compressed_tensors_moe.py`` (+8): PPU MXFP4 MoE layers redirect to
  ``Mxfp4MoEMethod`` instead of the upstream W4A4-MXFP4 method.
* ``compressed_tensors_moe_w8a8_int8.py`` (+2): ``swiglu_limit`` is threaded
  into the int8 MoE quant config (on PPU it becomes ``gemm1_clamp_limit``,
  which the PPU experts read).

All patches delegate to upstream; only the PPU-specific case is intercepted.
Leaf-style module (like ``patch/enhancement/residual/``): importable on a bare
CPU runner, patches install when ``install()`` runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_sail.patch.utils import patch

if TYPE_CHECKING:
    import torch

_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
)
_MOE_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
_INT8_MOE_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
AFFECTED_VERSIONS = ">=0.31.0,<0.32.0"

REASON_CHANNELWISE = (
    "The fork adds fp8_channelwise_layers to CompressedTensorsConfig: dense "
    "layers listed there are quantized FP8 channelwise on PPU instead of with "
    "the checkpoint's scheme; upstream has no such plumbing."
)
REASON_MOE_REDIRECT = (
    "PPU runs compressed-tensors MXFP4 MoE through Mxfp4MoEMethod (whose PPU "
    "backend selection is patched separately); upstream picks the W4A4-MXFP4 "
    "method, which has no PPU experts."
)
REASON_SWIGLU = (
    "The fork threads swiglu_limit into the int8 MoE quant config; on PPU it "
    "becomes gemm1_clamp_limit, which the PPU int8 experts read. Upstream "
    "drops it."
)
REASON_KV_REQUIRES_GRAD = (
    "The fork marks the KV-cache scale/zero-point Parameters requires_grad="
    "False; upstream leaves the Parameter wrappers trainable."
)
REMOVE_WHEN_CHANNELWISE = (
    "upstream grows fp8_channelwise_layers support in CompressedTensorsConfig, "
    "or the PPU mixed-precision checkpoints stop using it."
)
REMOVE_WHEN_MOE_REDIRECT = (
    "upstream's compressed-tensors MXFP4 MoE path supports the PPU DeepGEMM "
    "experts directly."
)
REMOVE_WHEN_SWIGLU = (
    "upstream threads swiglu_limit through the int8 MoE quant config itself."
)
REMOVE_WHEN_KV = (
    "upstream constructs the KV-cache scale/zero-point Parameters with "
    "requires_grad=False."
)

METADATA = (
    (
        f"{_MODULE}.CompressedTensorsConfig.__init__",
        REASON_CHANNELWISE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_CHANNELWISE,
    ),
    (
        f"{_MODULE}.CompressedTensorsConfig.from_config",
        REASON_CHANNELWISE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_CHANNELWISE,
    ),
    (
        f"{_MODULE}.CompressedTensorsConfig._is_fp8_channelwise_layer",
        REASON_CHANNELWISE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_CHANNELWISE,
    ),
    (
        f"{_MODULE}.CompressedTensorsConfig.get_scheme",
        REASON_CHANNELWISE,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_CHANNELWISE,
    ),
    (
        f"{_MODULE}.CompressedTensorsKVCacheMethod.create_weights",
        REASON_KV_REQUIRES_GRAD,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_KV,
    ),
    (
        f"{_MOE_MODULE}.CompressedTensorsMoEMethod.get_moe_method",
        REASON_MOE_REDIRECT,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_MOE_REDIRECT,
    ),
    (
        f"{_INT8_MOE_MODULE}.CompressedTensorsW8A8Int8MoEMethod."
        "get_fused_moe_quant_config",
        REASON_SWIGLU,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_SWIGLU,
    ),
)

METADATA += (
    (
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe."
        "compressed_tensors_moe_w8a8_fp8.CompressedTensorsW8A8Fp8MoEMethod.get_fused_moe_quant_config",
        REASON_SWIGLU,
        AFFECTED_VERSIONS,
        REMOVE_WHEN_SWIGLU,
    ),
)

_KV_PARAM_NAMES = ("k_scale", "k_zero_point", "v_zero_point", "q_zero_point")

_installed = False


def install() -> None:
    """Apply the patches. Requires vLLM to be importable; idempotent."""
    global _installed
    if _installed:
        return

    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as _ct,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe as _ct_moe,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_w8a8_int8 as _ct_int8,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    _upstream_init = _ct.CompressedTensorsConfig.__init__
    _upstream_from_config = _ct.CompressedTensorsConfig.from_config
    _upstream_get_scheme = _ct.CompressedTensorsConfig.get_scheme
    _upstream_kv_create_weights = _ct.CompressedTensorsKVCacheMethod.create_weights
    _upstream_get_moe_method = _ct_moe.CompressedTensorsMoEMethod.get_moe_method
    _upstream_int8_gfqc = (
        _ct_int8.CompressedTensorsW8A8Int8MoEMethod.get_fused_moe_quant_config
    )

    @patch(
        _MODULE,
        "CompressedTensorsConfig.__init__",
        reason=REASON_CHANNELWISE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_CHANNELWISE,
    )
    def _config_init(self, *args: Any, fp8_channelwise_layers=None, **kwargs: Any):
        _upstream_init(self, *args, **kwargs)
        self.fp8_channelwise_layers = fp8_channelwise_layers

    @patch(
        _MODULE,
        "CompressedTensorsConfig.from_config",
        reason=REASON_CHANNELWISE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_CHANNELWISE,
    )
    def _from_config(cls, config):
        instance = _upstream_from_config.__func__(cls, config)
        if isinstance(config, dict):
            instance.fp8_channelwise_layers = config.get("fp8_channelwise_layers")
        return instance

    @patch(
        _MODULE,
        "CompressedTensorsConfig._is_fp8_channelwise_layer",
        allow_missing=True,
        reason=REASON_CHANNELWISE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_CHANNELWISE,
    )
    def _is_fp8_channelwise_layer(self, layer_name: str) -> bool:
        """Check if layer uses FP8 channelwise quantization (substring)."""
        if not self.fp8_channelwise_layers:
            return False
        return is_layer_skipped(
            prefix=layer_name,
            ignored_layers=self.fp8_channelwise_layers,
            fused_mapping=self.packed_modules_mapping,
            match_mode="substring",
        )

    @patch(
        _MODULE,
        "CompressedTensorsConfig.get_scheme",
        reason=REASON_CHANNELWISE,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_CHANNELWISE,
    )
    def _get_scheme(self, layer, layer_name: str | None = None):
        from vllm.platforms import current_platform

        if (
            current_platform.is_ppu()
            and self.fp8_channelwise_layers
            and layer_name is not None
            and self._is_fp8_channelwise_layer(layer_name)
        ):
            from compressed_tensors.quantization import (
                QuantizationArgs,
                QuantizationStrategy,
                QuantizationType,
            )
            from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa: E501
                CompressedTensorsW8A8Fp8,
            )

            channelwise_args = QuantizationArgs(
                strategy=QuantizationStrategy.CHANNEL,
                type=QuantizationType.FLOAT,
                num_bits=8,
                symmetric=True,
            )
            return CompressedTensorsW8A8Fp8(
                weight_quant=channelwise_args,
                is_static_input_scheme=False,
            )
        return _upstream_get_scheme(self, layer, layer_name)

    @patch(
        _MODULE,
        "CompressedTensorsKVCacheMethod.create_weights",
        reason=REASON_KV_REQUIRES_GRAD,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_KV,
    )
    def _kv_create_weights(self, layer, *args: Any, **kwargs: Any):
        _upstream_kv_create_weights(self, layer, *args, **kwargs)
        for name in _KV_PARAM_NAMES:
            param = getattr(layer, name, None)
            if param is not None:
                param.requires_grad_(False)

    @patch(
        _MOE_MODULE,
        "CompressedTensorsMoEMethod.get_moe_method",
        reason=REASON_MOE_REDIRECT,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_MOE_REDIRECT,
    )
    def _get_moe_method(quant_config, layer, layer_name: str):
        method = _upstream_get_moe_method(quant_config, layer, layer_name)
        from vllm.platforms import current_platform

        if not current_platform.is_ppu():
            return method
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
            CompressedTensorsW4A4Mxfp4MoEMethod,
        )

        if isinstance(method, CompressedTensorsW4A4Mxfp4MoEMethod):
            from vllm.model_executor.layers.quantization.mxfp4 import (
                Mxfp4MoEMethod,
            )

            return Mxfp4MoEMethod(layer.moe_config)
        return method

    @patch(
        _INT8_MOE_MODULE,
        "CompressedTensorsW8A8Int8MoEMethod.get_fused_moe_quant_config",
        reason=REASON_SWIGLU,
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_SWIGLU,
    )
    def _int8_get_fused_moe_quant_config(self, layer: torch.nn.Module):
        config = _upstream_int8_gfqc(self, layer)
        from vllm.platforms import current_platform

        if current_platform.is_ppu() and config is not None:
            config.gemm1_clamp_limit = getattr(layer, "swiglu_limit", None)
            config.gemm1_alpha = getattr(layer, "swiglu_alpha", None)
            config.gemm1_beta = getattr(layer, "swiglu_beta", None)
        return config

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        compressed_tensors_moe_w8a8_fp8 as _ct_fp8,
    )

    _upstream_fp8_gfqc = (
        _ct_fp8.CompressedTensorsW8A8Fp8MoEMethod.get_fused_moe_quant_config
    )

    @patch(
        _ct_fp8.__name__,
        "CompressedTensorsW8A8Fp8MoEMethod.get_fused_moe_quant_config",
        reason="PPU FP8 experts need the model's SwiGLU alpha and beta parameters.",
        affected_versions=AFFECTED_VERSIONS,
        remove_when=REMOVE_WHEN_SWIGLU,
    )
    def _fp8_get_fused_moe_quant_config(self, layer):
        from vllm.platforms import current_platform

        config = _upstream_fp8_gfqc(self, layer)
        if current_platform.is_ppu() and config is not None:
            config.gemm1_alpha = getattr(layer, "swiglu_alpha", None)
            config.gemm1_beta = getattr(layer, "swiglu_beta", None)
        return config

    _installed = True
