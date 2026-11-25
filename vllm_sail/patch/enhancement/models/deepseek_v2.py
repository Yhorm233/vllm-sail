# SPDX-License-Identifier: Apache-2.0
"""Keep separate quantized wk and BF16 score projections on PPU when needed.

The upstream model and MTP loaders own the checkpoint parameters. Only the
indexer decides whether the forward needs a fused BF16 copy after loading.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_group_quant_int8
from vllm.model_executor.layers.sparse_attn_indexer import fused_indexer_q_rope_quant
from vllm.model_executor.models import deepseek_v2 as upstream
from vllm.platforms import current_platform

from vllm_sail.patch.utils import patch

_AFFECTED = "upstream main e83d82a"
_upstream_init = upstream.init_indexer_wk_weights_proj
_upstream_finish = upstream.Indexer.process_weights_after_loading
_upstream_indexer_init = upstream.Indexer.__init__


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "init_indexer_wk_weights_proj",
    reason="PPU channelwise wk must load into its own quantized parameter without a derived BF16 weight.",
    affected_versions=_AFFECTED,
    remove_when="The upstream projection helper supports separate PPU quantized wk and BF16 scoring.",
)
def init_projections(indexer, hidden_size, quant_config, prefix, **kwargs):
    indexer.use_ppu_split_indexer = current_platform.is_ppu() and not (
        quant_config is not None and quant_config.get_name() == "modelopt_fp4"
    )
    if not indexer.use_ppu_split_indexer:
        return _upstream_init(indexer, hidden_size, quant_config, prefix, **kwargs)
    indexer.wk = ReplicatedLinear(
        hidden_size, indexer.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.wk"
    )
    indexer.weights_proj = ReplicatedLinear(
        hidden_size, indexer.n_head, bias=False, quant_config=None, prefix=f"{prefix}.weights_proj"
    )


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "Indexer.process_weights_after_loading",
    reason="The PPU split projection executes checkpoint weights directly and has no derived fused storage to refresh.",
    affected_versions=_AFFECTED,
    remove_when="Upstream refreshes derived weights only for fused indexer projections.",
)
def process_weights_after_loading(self):
    if not self.use_ppu_split_indexer:
        return _upstream_finish(self)


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "Indexer.__init__",
    reason="Select unfused q and INT8 quantization on PPU indexers without the matching FP8 projection path.",
    affected_versions=_AFFECTED,
    remove_when="Upstream dispatches indexer query quantization using platform and projection capabilities.",
)
def indexer_init(self, *args, **kwargs):
    _upstream_indexer_init(self, *args, **kwargs)
    self.use_ppu_int8_indexer = current_platform.is_ppu() and not current_platform.supports_fp8()
    if self.use_ppu_split_indexer or self.use_ppu_int8_indexer:
        self.use_fused_indexer_q = False


@patch(
    "vllm.model_executor.models.deepseek_v2",
    "Indexer.forward",
    reason=(
        "Selects between the fused wk_weights_proj path (modelopt_fp4 only) "
        "and the separate wk/weights_proj projections installed by the "
        "Indexer.__init__ patch, and uses int8 q quantization on PPU without "
        "FP8 support. Verbatim upstream body with two marked branches; the "
        "changes sit mid-body with no delegation seam."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream parameterises the Indexer wk projection on quant_config "
        "and exposes a platform hook for the q-quant format."
    ),
)
def forward(
    self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
) -> torch.Tensor:
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)

    if current_platform.is_rocm() and self.is_inplace_rope:
        # This path should works on all platform, will remove extra
        # branches in the future
        # This fast path relies on rotary_emb mutating q and k inplace.
        # On ROCm, this is only valid for kernels used as custom ops.
        # In pytorch-native rope for inductor fusion, rotated q/k tensors
        # are not mutated inplace but returned as new tensors.
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)

        rotary_emb(
            positions, q[..., : self.rope_dim], k[..., : self.rope_dim].unsqueeze(1)
        )
    elif self.use_fused_indexer_q and q.dtype == torch.bfloat16:
        # fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_fp8, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head_scale,
            rotary_emb.is_neox_style,
        )

        # rotate only the MQA K
        k_pe = k_pe.unsqueeze(1)
        q_dummy = torch.empty_like(k_pe)
        _, k_pe = rotary_emb(positions, q_dummy, k_pe)
        k_pe = k_pe.reshape(-1, self.rope_dim)
        k = torch.cat([k_pe, k_nope], dim=-1)

        return self.indexer_op(hidden_states, q_fp8, k, weights)
    else:
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        # PPU MODIFICATION: begin
        if not self.use_ppu_split_indexer:
            # Fused wk + weights_proj: one GEMM, then split
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]
        else:
            k, _ = self.wk(hidden_states)
            weights, _ = self.weights_proj(hidden_states)
        # PPU MODIFICATION: end

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during
        # compilation so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, rope_dim] (MQA).
        k = torch.cat([k_pe, k_nope], dim=-1)

    # we only quant q here since k quant is fused with cache insertion
    q = q.view(-1, self.head_dim)
    # PPU MODIFICATION: begin

    if self.use_ppu_int8_indexer:
        q_fp8, q_scale = per_token_group_quant_int8(q, self.quant_block_size)
    else:
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
    # PPU MODIFICATION: end
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head)

    weights = weights * q_scale * self.softmax_scale * self.n_head_scale

    return self.indexer_op(hidden_states, q_fp8, k, weights)
