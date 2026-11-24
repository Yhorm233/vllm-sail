# SPDX-License-Identifier: Apache-2.0
"""Keep MiniMax DSpark on its base context writer when PPU has only a stub."""

from vllm.models.minimax_m3.common import dspark as _dspark
from vllm.models.minimax_m3.common.ops.dspark_context_kv import DSparkContextKVWriter
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_original_supported = DSparkContextKVWriter.supported
_original_normalize_context_k = _dspark.MiniMaxM3DSparkModel._normalize_context_k


@patch(
    "vllm.models.minimax_m3.common.ops.dspark_context_kv",
    "DSparkContextKVWriter.supported",
    reason="PPU advertises CUDA identity and compute stubs, which do not implement the fused DSpark context writer.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The context writer checks a real platform implementation or PPU implements and verifies this operator.",
)
def supported(attn_layers: list) -> bool:
    if current_platform.is_ppu():
        return False
    return _original_supported(attn_layers)


@patch(
    "vllm.models.minimax_m3.common.dspark",
    "MiniMaxM3DSparkModel._normalize_context_k",
    reason="The PPU context-KV fallback passes three-dimensional K to MiniMAXGemmaRMSNorm, whose non-matrix path calls a two-dimensional-only FlashInfer kernel.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="MiniMax's context-K normalization or Gemma RMSNorm supports the fallback's per-head tensor shape.",
)
def normalize_context_k(self, all_k):
    if not current_platform.is_ppu():
        return _original_normalize_context_k(self, all_k)
    normalized = [
        layer.self_attn.k_norm(k.reshape(-1, k.shape[-1])).reshape(k.shape)
        for layer, k in zip(self.layers, all_k)
    ]
    return _dspark.torch.stack(normalized)
