# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
"""PPU MLA profile allocation and INT8 input-dtype handling.

The copied forward body preserves SAIL's profile-run allocation workaround.
vLLM 0.30 centralizes projection input casting in a helper; patching that helper
keeps floating activations for ACEXT across all prefill paths.
"""

from __future__ import annotations

from vllm.model_executor.layers.attention import mla_attention as _mla_attention

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.model_executor.layers.attention.mla_attention"


def _with_target_globals(fn):
    return bind_body(fn, _mla_attention)


def _forward_impl_body(
    self,
    q: torch.Tensor,
    k_c_normed: torch.Tensor,  # key in unified attn
    k_pe: torch.Tensor,  # value in unified attn
    kv_cache: torch.Tensor,
    attn_metadata: "MLACommonMetadata",
    output: torch.Tensor,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    quant_group_size: int | None = None,
    quant_scale_ue8m0: bool | None = None,
    quant_col_major: bool | None = None,
    quant_tma_aligned: bool | None = None,
    q_dcp_replicated: torch.Tensor | None = None,
    mqa_q_prepared: torch.Tensor | None = None,
) -> torch.Tensor:
    assert output is not None, "Output tensor must be provided."

    quant_key = _detect_output_quant_key(
        output, output_scale, output_block_scale, self.num_heads * self.v_head_dim
    )
    if quant_key is not None:
        # The fusion pass has allocated output with quantized dtype
        # (FP8 or uint8 for FP4). We can't write into it directly,
        # so we swap in a temp buffer for computation, then quantize
        # into the real output at the end.
        # NOTE(carlyou): this is temporary until kernels support fp8 output
        quant_output = output
        output = torch.empty(
            output.shape[0],
            self.num_heads * self.v_head_dim,
            dtype=q.dtype,
            device=output.device,
        )

    if attn_metadata is None:
        # During the profile run try to simulate to worse case output size
        # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
        # since this can be large
        # PPU MODIFICATION: begin
        # PPU FIXME: WA for deepseek dp + ep OOM on PPU, need investigate later.
        if not current_platform.is_ppu():
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )
        # PPU MODIFICATION: end
        self.impl.reserve_workspace()
        if self.use_shardq and self.impl.dcp_world_size > 1:
            from vllm.v1.attention.ops.dcp_shardq.context import (
                reserve_dcp_shardq,
            )

            reserve_dcp_shardq(self, k_c_normed.device)

        # The zero fill is required when used with DP + EP
        # to ensure all ranks within a DP group compute the
        # same expert outputs.
        if quant_key is not None:
            return quant_output.fill_(0)
        return output.fill_(0)

    fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)

    num_actual_toks = attn_metadata.num_actual_tokens
    if self.use_pcp and self.impl.dcp_world_size > 1 and quant_key is not None:
        raise NotImplementedError(
            "MRV2 MLA PCP+DCP does not support fused output quantization yet."
        )

    # Inputs and outputs may be padded for CUDA graphs
    output_padded = output
    output = output[:num_actual_toks, ...]
    q = q[:num_actual_toks, ...]
    if q_dcp_replicated is not None:
        q_dcp_replicated = q_dcp_replicated[:num_actual_toks, ...]
    k_c_normed = k_c_normed[:num_actual_toks, ...]
    k_pe = k_pe[:num_actual_toks, ...]

    if fp8_attention and self.kv_cache_dtype not in (
        # Opaque per-token byte formats stay as raw uint8
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ):
        kv_cache = kv_cache.view(current_platform.fp8_dtype())

    assert (
        attn_metadata.num_decodes is not None
        and attn_metadata.num_prefills is not None
        and attn_metadata.num_decode_tokens is not None
    )
    num_mqa_tokens = attn_metadata.num_decode_tokens
    num_mha_tokens = q.size(0) - num_mqa_tokens
    use_mha = True

    if self.impl.is_sparse and num_mha_tokens > 0:
        use_mha = self._use_sparse_mha(attn_metadata)
        if not use_mha:
            num_mqa_tokens = q.size(0)
            num_mha_tokens = 0

    shardq = getattr(attn_metadata, "shardq", None)
    gathered_context = None if shardq is None else shardq.gathered_context
    if gathered_context is not None:
        # A DCP ShardQ step's prefill tokens attend a gathered copy of
        # their context, which still lacks the step's new tokens; every
        # rank writes all of them.
        num_dcp_decode_tokens = gathered_context.num_decode_tokens
        self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            k_c_normed[num_dcp_decode_tokens:],
            k_pe[num_dcp_decode_tokens:],
            gathered_context.mla_view(kv_cache),
            gathered_context.slot_mapping,
            self.kv_cache_dtype,
            self._k_scale,
        )

    mha_use_quant_output = (
        quant_key is not None
        and self.prefill_backend is not None
        and self.prefill_backend.supports_quant_output(quant_key)
        and attn_metadata is not None
        and attn_metadata.prefill is not None
        and (
            not self.impl.is_sparse
            or attn_metadata.prefill.route  # type: ignore[attr-defined]
            is SparseMLAPrefillRoute.DENSE_MHA
        )
        and attn_metadata.prefill.chunked_context is None
        and self.impl.dcp_world_size <= 1
    )

    if num_mha_tokens > 0:
        if mha_use_quant_output:
            mha_output = quant_output
            mha_output_scale = output_scale
        else:
            mha_output = output
            mha_output_scale = None

        self.impl.forward_mha(  # type: ignore[attr-defined]
            q[num_mqa_tokens:],
            k_c_normed[num_mqa_tokens:],
            k_pe[num_mqa_tokens:],
            kv_cache,
            attn_metadata,
            self._k_scale,
            output=mha_output[num_mqa_tokens:num_actual_toks],
            output_scale=mha_output_scale,
        )

    if num_mqa_tokens > 0:
        has_mla_lora = hasattr(self.kv_b_proj, "apply_mla_absorbed_kv_b_lora")
        use_fused_v_up_proj = (
            self.W_UV_dcp is not None
            and not has_mla_lora
            and gathered_context is None
        )
        mqa_output_slice = output[:num_mqa_tokens]
        if mqa_q_prepared is not None:
            mqa_q = mqa_q_prepared[:num_mqa_tokens]
        elif q_dcp_replicated is not None and gathered_context is None:
            mqa_q = self._prepare_mqa_query(
                q_dcp_replicated[:num_mqa_tokens], qrep_decode=True
            )
        else:
            # A DCP ShardQ step exchanges the local heads' query (shard_q)
            # instead of gathering it.
            mqa_q = self._prepare_mqa_query(
                q[:num_mqa_tokens],
                qrep_decode=False,
                fused_q_heads=self._fused_decode_query_heads(attn_metadata),
                gather_query=gathered_context is None,
            )
        mqa_metadata = attn_metadata
        mqa_kv_cache = kv_cache
        if shardq is not None:
            decode_q, mqa_q = shard_q(mqa_q, shardq)
            decode_out = None
            if decode_q.shape[0] and gathered_context is not None:
                # Decode tokens keep the DCP path.
                decode_out = self._forward_dcp_decode(
                    decode_q, kv_cache, attn_metadata, gathered_context
                )
            elif decode_q.shape[0]:
                # Decode tokens attend head-sharded.
                decode_out, _ = self.impl.forward_mqa(  # type: ignore[attr-defined]
                    decode_q, kv_cache, attn_metadata, self
                )
            mqa_metadata = shardq.attn_metadata
            if gathered_context is not None:
                mqa_kv_cache = gathered_context.mla_view(kv_cache)

        # call decode attn
        if not self.impl.is_sparse:
            assert attn_metadata.decode is not None
        attn_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
            mqa_q, mqa_kv_cache, mqa_metadata, self
        )
        if shardq is not None:
            attn_out = unshard_out(attn_out, shardq, decode_out)

        # correct dcp attn_out with lse.
        if self.impl.dcp_world_size > 1 and gathered_context is None:
            assert lse is not None
            assert self.dcp_manager is not None
            decode_metadata = attn_metadata.decode
            if not use_mha and not self.use_pcp:
                seq_lens = cast(torch.Tensor, attn_metadata.seq_lens)  # type: ignore[attr-defined]
                query_start_loc = attn_metadata.query_start_loc
            else:
                seq_lens = (
                    decode_metadata.seq_lens
                    if decode_metadata is not None
                    else cast(torch.Tensor, attn_metadata.seq_lens)[  # type: ignore[attr-defined]
                        : attn_metadata.num_decodes
                    ]
                )
                query_start_loc = attn_metadata.query_start_loc[
                    : attn_metadata.num_decodes + 1
                ]
            # Under PCP+DCP the prefill rows attended over the gathered KV,
            # so only the decode rows carry an LSE and take part in the merge.
            num_merge_rows = lse.shape[0]
            if use_fused_v_up_proj:
                assert self.dcp_manager.v_up_proj_combine is not None
                self.dcp_manager.v_up_proj_combine(
                    attn_out,
                    lse,
                    self.W_UV_dcp,
                    mqa_output_slice,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
            elif num_merge_rows == attn_out.shape[0]:
                attn_out = self.dcp_manager.combine(
                    attn_out,
                    lse,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
            elif num_merge_rows > 0:
                attn_out[:num_merge_rows] = self.dcp_manager.combine(
                    attn_out[:num_merge_rows],
                    lse,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
            if self.use_pcp:
                attn_out = finalize_mla_pcp_decode(attn_out, self.num_heads)

        # v_up projection, unless the DCP merge already applied it
        if not use_fused_v_up_proj:
            self._v_up_proj(attn_out, out=mqa_output_slice)

    if quant_key is not None:
        quant_idx = num_mqa_tokens if mha_use_quant_output else num_actual_toks
        if quant_idx == 0:
            return quant_output
        actual = output[:quant_idx]
        if quant_key == kNvfp4Dynamic:
            # NVFP4: two FP4 values packed into one uint8
            assert output_block_scale is not None
            fp4_data, fp4_scales = ops.scaled_fp4_quant(actual, output_scale)
            quant_output[:quant_idx].copy_(fp4_data)
            output_block_scale[: fp4_scales.shape[0]].copy_(fp4_scales)
        elif quant_key in (kFp8Dynamic128Sym, kFp8Dynamic64Sym):
            # Per-group FP8
            assert output_block_scale is not None
            assert quant_group_size is not None, (
                "Group FP8 output quant requested but "
                "quant_group_size not passed through custom op"
            )
            finfo = torch.finfo(_FP8_DTYPE)
            torch.ops._C.per_token_group_fp8_quant(
                actual,
                quant_output[:quant_idx],
                output_block_scale[:quant_idx],
                quant_group_size,
                1e-10,  # eps
                finfo.min,
                finfo.max,
                quant_scale_ue8m0,
                quant_col_major,
                quant_tma_aligned,
            )
        elif quant_key == kFp8StaticTensorSym:
            # Static FP8 quantization
            if self._aiter_output_quant:
                torch.ops.vllm.rocm_aiter_per_tensor_quant(
                    quant_output[:quant_idx], actual, output_scale, False
                )
            else:
                fp8_data, _ = self._quant_fp8_op(actual, output_scale)
                quant_output[:quant_idx].copy_(fp8_data)
        else:
            raise ValueError(f"Unsupported quant_key: {quant_key}")
        return quant_output

    if self.use_pcp and output_padded.shape[0] > num_actual_toks:
        output_padded[num_actual_toks:].zero_()
    return output_padded


patch(
    _MODULE,
    "MLAAttention.forward_impl",
    reason=(
        "PPU FIXME: WA for deepseek dp + ep OOM on PPU — the fork skips the "
        "profile-run torch.empty scratch allocation in forward_impl. The "
        "allocation is mid-method, so this is a verbatim upstream body with "
        "only the fork's change marked."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the PPU OOM workaround is root-caused upstream-side, or upstream stops "
        "allocating the profile-run scratch buffer."
    ),
)(_with_target_globals(_forward_impl_body))


_upstream_kv_b_proj_input_dtype = _mla_attention._get_kv_b_proj_input_dtype


@patch(
    _MODULE,
    "_get_kv_b_proj_input_dtype",
    reason="SAIL ACEXT quantizes floating activations internally for INT8 kv_b_proj weights.",
    affected_versions=_AFFECTED,
    remove_when="The upstream input-dtype helper recognizes SAIL INT8 linear kernels.",
)
def _get_kv_b_proj_input_dtype(kv_b_proj, use_fp8_prefill):
    import torch
    from vllm.platforms import current_platform

    weight = getattr(kv_b_proj, "weight", None)
    if current_platform.is_ppu() and weight is not None and weight.dtype == torch.int8:
        return None
    return _upstream_kv_b_proj_input_dtype(kv_b_proj, use_fp8_prefill)
