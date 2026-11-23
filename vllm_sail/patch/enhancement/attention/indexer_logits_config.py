# SPDX-License-Identifier: Apache-2.0
"""Resolve DeepSeek logits before upstream consumes the user's auto setting."""

from vllm.model_executor.models import config as _config
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_original = _config._set_deepseek_indexer_logits_dtype


@patch(
    "vllm.model_executor.models.config",
    "_set_deepseek_indexer_logits_dtype",
    reason=(
        "The model config hook resolves auto before platform defaults run. "
        "Its SM100-only probe selects FP32 on M890, but SAIL MXFP4 logits "
        "reduce in BF16. Resolve from the selected PPU cache format while "
        "preserving explicit dtype requests."
    ),
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The model config default queries the selected indexer provider's reduction dtype.",
)
def _set_deepseek_indexer_logits_dtype(vllm_config):
    if not current_platform.is_ppu():
        return _original(vllm_config)
    from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

    attention = vllm_config.attention_config
    default = "bfloat16" if dsa_indexer_uses_fp4(vllm_config) else "float32"
    attention.indexer_logits_dtype = attention.resolve_indexer_logits_dtype(default)
