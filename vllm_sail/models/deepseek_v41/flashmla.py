# SPDX-License-Identifier: Apache-2.0
"""810E head geometry and INT8 output projection for V4.1 FlashMLA."""

import torch
from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekV4FlashMLAAttention as UpstreamAttention,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLABackend

from vllm_sail.models.deepseek_v4.ops.o_proj import deep_gemm_int8_o_proj


class PPUDeepseekV41FlashMLABackend(DeepseekV4FlashMLABackend):
    @classmethod
    def supports_compute_capability(cls, capability):
        return capability.major == 8 and capability.minor == 0


class DeepseekV41FlashMLAAttention(UpstreamAttention):
    backend_cls = PPUDeepseekV41FlashMLABackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        if num_heads not in (8, 64):
            raise ValueError(f"810E V4.1 supports TP8/TP1; actual head count is {num_heads}")
        return num_heads

    def _o_proj(self, attn_out, positions):
        if self.wo_a.weight.dtype != torch.int8:
            raise ValueError("810E V4.1 requires W8A8 INT8 wo_a")
        return deep_gemm_int8_o_proj(
            attn_out[:, : self.n_local_heads],
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self._wo_b_proj,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=(1, 1, 128),
        )
