# SPDX-License-Identifier: Apache-2.0
"""SAIL MSA sparse attend (PPU) for MiniMax M3.

``fmha_sail`` consumes the per-token block ids the indexer wrote into the
model-shared ``topk_indices_buffer`` (``[tokens, kv_heads, topk]``, ascending,
``-1`` padded) and attends those pages of the paged K/V cache. One plan covers
both decode and prefill rows: SAIL's sparse plan expands the batch per query
token, so a mixed step needs no decode/prefill split.

BF16 K/V keeps the model-dtype query. FP8 K/V is served by the fp8 sparse
kernel, which takes the E4M3 query the fused norm/RoPE/insert kernel writes --
the same ``use_cutlass_decode`` switch the NVIDIA MSA impl uses to allocate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_sail.models.minimax_m3_msa.adapter import require_sail_msa_ops
from vllm_sail.models.minimax_m3_msa.common import (
    PAGE_SIZE,
    PlanWorkspacePool,
    flat_page_table,
    plan_device,
)
from vllm_sail.models.minimax_m3_msa.gating import (
    SailMSASelection,
    decide_from_config,
    decide_from_current_config,
)

logger = init_logger(__name__)


class MiniMaxM3SparseSailMSABackend(MiniMaxM3SparseBackend):
    """Backend marker selecting the SAIL sparse attend builder + impl."""

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_SAIL_MSA"

    @staticmethod
    def get_impl_cls() -> type[MiniMaxM3SparseSailMSAImpl]:
        return MiniMaxM3SparseSailMSAImpl

    @staticmethod
    def get_builder_cls() -> type[MiniMaxM3SparseSailMSAMetadataBuilder]:
        return MiniMaxM3SparseSailMSAMetadataBuilder


@dataclass
class MiniMaxM3SparseSailMSAMetadata(MiniMaxM3SparseMetadata):
    """SAIL sparse plan for the whole (decode + prefill) batch."""

    sail_plan: tuple | None = None
    # Compacted paged page table ([sum_pages] int32) the plan was built for.
    sail_kv_indices: torch.Tensor | None = None


class MiniMaxM3SparseSailMSAMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Builds the SAIL sparse attend plan per step (attention runs eagerly)."""

    # Same reason as the indexer builder: the plan and compacted page table are
    # rebuilt per step, so FULL-graph replay would use stale pointers.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_q_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.topk_blocks = config.index_topk_blocks
        self.selection: SailMSASelection = decide_from_config(vllm_config)
        if not self.selection.enable:
            raise ValueError(
                "MiniMax M3 SAIL MSA attend selected but its policy is off: "
                f"{self.selection.reason}"
            )
        self.use_fp8_kv = self.selection.attend_fp8
        self._ops = require_sail_msa_ops()
        self._plans = PlanWorkspacePool()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseSailMSAMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)

        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        qo_lens = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int32)
        kv_lens = seq_lens[:num_reqs].cpu().to(torch.int32)

        device = plan_device(seq_lens)
        workspace = self._plans.get(
            num_tokens=int(qo_lens.sum()),
            max_qo_len=max(int(qo_lens.max()), 1),
            num_qo_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            kv_block_num=self.topk_blocks,
            output_maxscore=False,
            use_fp8_kvcache=self.use_fp8_kv,
            device=device,
        )
        plan = self._ops.fmha_sail_plan(
            qo_lens,
            kv_lens,
            self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            page_size=PAGE_SIZE,
            kv_block_num=self.topk_blocks,
            output_maxscore=False,
            causal=True,
            use_fp8_kvcache=self.use_fp8_kv,
            # Sparse rows attend at most topk_blocks pages, so one KV pass per
            # row needs no split; this also keeps the plan workspace minimal.
            num_kv_splits=1,
            device=device,
            given_workspace=workspace,
        )
        has_mixed_prefill, _, batch_size, first_plan, second_plan = plan
        if batch_size != num_reqs:
            raise RuntimeError(
                "SAIL sparse attend plan returned an unexpected batch size "
                f"({batch_size}/{num_reqs})"
            )
        # A mixed step is split into decode/prefill sub-plans by the library;
        # both must be sparse (no max-score tiles).
        sub_plans = (first_plan, second_plan) if has_mixed_prefill else (first_plan,)
        if any(int(p["max_k_tiles"]) != -1 for p in sub_plans):
            raise RuntimeError(
                "SAIL sparse attend plan unexpectedly allocated max_score tiles"
            )

        kv_indices = flat_page_table(block_table, seq_lens, num_reqs)
        return MiniMaxM3SparseSailMSAMetadata(
            seq_lens=base.seq_lens,
            max_seq_len=base.max_seq_len,
            slot_mapping=base.slot_mapping,
            page16_slot_mapping=base.page16_slot_mapping,
            num_actual_tokens=base.num_actual_tokens,
            num_decodes=base.num_decodes,
            num_decode_tokens=base.num_decode_tokens,
            num_prefills=base.num_prefills,
            num_prefill_tokens=base.num_prefill_tokens,
            prefill=base.prefill,
            decode=base.decode,
            sail_plan=plan,
            sail_kv_indices=kv_indices,
        )


class MiniMaxM3SparseSailMSAImpl(MiniMaxM3SparseImpl):
    """One ``fmha_sail`` call over the indexer-selected pages."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
        msa_decode_backend: str = "triton",
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
            msa_decode_backend=msa_decode_backend,
        )
        selection = decide_from_current_config()
        if not selection.enable:
            raise ValueError(
                "MiniMax M3 SAIL MSA attend impl built while the policy is "
                f"off: {selection.reason}"
            )
        # fp8 sparse attend needs the E4M3 query; reuse the switch that makes
        # the layer allocate/emit it (same as the NVIDIA CUTLASS decode path).
        self.use_cutlass_decode = self.use_fp8_kv
        self._ops = require_sail_msa_ops()
        logger.warning_once(
            "MiniMax M3 sparse attention: SAIL sparse fmha "
            "[kv_cache_dtype=%s, topk=%d]",
            kv_cache_dtype,
            self.topk_blocks,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseSailMSAMetadata)
        assert main_md.sail_plan is not None
        assert main_md.sail_kv_indices is not None

        num_tokens = main_md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None

        hd = self.head_size
        # The insert kernel emits an E4M3 query when fp8 attend is selected:
        # as ``query`` itself on vLLM main, as ``query_fp8`` on the 0.33
        # release (``query`` is ``None`` there). Either way it is quantised
        # with ``_q_scale_float``.
        src = query if query_fp8 is None else query_fp8
        q = src[:num_tokens].view(-1, self.num_heads, hd)
        q_is_fp8 = q.element_size() == 1
        q_scale = float(getattr(layer, "_q_scale_float", 1.0)) if q_is_fp8 else 1.0
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        kv = kv_cache
        if self.use_fp8_kv:
            # vLLM stores fp8 pages as uint8; K and V share the page's last dim.
            kv = kv.view(self.kv_cache_fp8_dtype)
            if not q_is_fp8:
                # Model-dtype query: K/V are inserted at unit scale, so
                # quantise the query at unit scale too.
                q = q.to(self.kv_cache_fp8_dtype)
        k_cache, v_cache = kv.split(hd, dim=-1)
        # The cache interleaves K|V in the page's last dim, so the split views
        # jump by 2*hd between tokens. SAIL's HND layout assumes a packed
        # in-page stride of head_dim and would silently mis-address them; its
        # NHD layout reads the token/head strides, and a permuted view is
        # exactly that layout ([pages, page, heads, hd]).
        k_cache = k_cache.permute(0, 2, 1, 3)
        v_cache = v_cache.permute(0, 2, 1, 3)

        self._ops.fmha_sail(
            q,
            k_cache,
            v_cache,
            main_md.sail_plan,
            kv_indices=main_md.sail_kv_indices,
            kv_block_indexes=topk[:num_tokens],
            out=out,
            sm_scale=self.scale,
            # K/V pages are inserted at unit scale, so only the query's
            # dequantisation scale enters; fmha_sail folds q_scale * k_scale
            # into the softmax scale.
            q_scale=q_scale,
            output_o=True,
            output_maxscore=False,
        )
        return output
