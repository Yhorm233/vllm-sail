# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_sail.utils.deep_select import (
    can_use_deep_select,
    deep_select_topk,
    warn_deep_select_fallback,
)

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops

if current_platform.is_ppu():
    # PPU-private ops live in torch.ops._ppu_C, which only exists once
    # vllm_sail._C has been imported. The wrapper below performs that import;
    # reaching into torch.ops._ppu_C directly would race the lazy load.
    from vllm_sail import _custom_ops as sail_ops
    from vllm_sail.utils.deep_gemm import (
        fp8_fp4_mqa_logits,
        fp8_fp4_paged_mqa_logits,
        fp8_mqa_logits,
        fp8_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
        int8_mqa_logits,
        int8_paged_mqa_logits,
        is_deep_gemm_supported,
    )

logger = init_logger(__name__)


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


@triton.jit
def _scatter_indexer_int8_cache(
    k_bytes,
    scale_bytes,
    cache,
    slots,
    BLOCK_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    if slot >= 0:
        tl.device_assert(slot < NUM_BLOCKS * BLOCK_SIZE, "indexer slot out of bounds")
        if slot < NUM_BLOCKS * BLOCK_SIZE:
            page = slot // BLOCK_SIZE
            token = slot % BLOCK_SIZE
            head = tl.arange(0, 128)
            values = tl.load(k_bytes + row * 128 + head)
            tl.store(cache + page * PAGE_STRIDE + token * 128 + head, values)
            byte = tl.arange(0, 4)
            scale = tl.load(scale_bytes + row * 4 + byte)
            tl.store(
                cache + page * PAGE_STRIDE + BLOCK_SIZE * 128 + token * 4 + byte,
                scale,
            )


def indexer_k_quant_and_cache_int8(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
) -> None:
    assert current_platform.is_ppu()
    assert k.ndim == 2 and k.shape[1] == 128 and k.dtype == torch.bfloat16
    assert quant_block_size == 128
    assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
    assert kv_cache.shape[1] > 0 and kv_cache.shape[2] == 132
    assert kv_cache.stride(2) == 1 and kv_cache.stride(1) == 132
    assert kv_cache.stride(0) >= kv_cache.shape[1] * 132
    assert kv_cache.stride(0) % 4 == 0 and kv_cache.data_ptr() % 4 == 0
    assert k.device == kv_cache.device == slot_mapping.device
    assert k.device.type == "cuda"
    assert slot_mapping.ndim == 1 and slot_mapping.dtype == torch.int64
    assert slot_mapping.is_contiguous() and slot_mapping.shape[0] == k.shape[0]
    if k.shape[0] == 0:
        return

    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_group_quant_int8,
    )

    values, scales = per_token_group_quant_int8(k.contiguous(), 128)
    # INT8 consumers need FP32 scales and separate value/scale planes per page.
    _scatter_indexer_int8_cache[(k.shape[0],)](
        values.view(torch.uint8),
        scales.view(torch.uint8),
        kv_cache,
        slot_mapping,
        kv_cache.shape[1],
        kv_cache.stride(0),
        kv_cache.shape[0],
        num_warps=4,
    )


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    q_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), q_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def ppu_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    q_dtype = q_quant.dtype
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, q_dtype, use_fp4_cache
        )

        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return ppu_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
            candidate_blocks,
            candidate_block_size,
            candidate_write,
            topk_backend,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if q_dtype == torch.int8:
            indexer_k_quant_and_cache_int8(k, kv_cache, slot_mapping, quant_block_size)
        else:
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # ShardQ 的 chunk 行号对应紧凑查询；cache 写入和 decode 保留原始行序。
        prefill_q, prefill_q_scale, prefill_weights = q_quant, q_scale, weights
        if (indices := prefill_metadata.shardq_gather_indices) is not None:
            assert candidate_blocks is None
            prefill_q, prefill_weights = q_quant[indices], weights[indices]
            if q_scale is not None:
                prefill_q_scale = q_scale[indices]

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, q_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            # Both host-known bounds cover every row's valid K window. Cap
            # short batches too, so a large configured context cannot inflate
            # a small chunk's allocation beyond its gathered KV extent.
            logits_capacity = min(max_model_len, chunk.total_seq_lens)

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = prefill_q[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                prefill_q_scale[chunk.token_start : chunk.token_end]
                if prefill_q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)

            if is_deep_gemm_supported():
                if use_fp4_cache:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        prefill_weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                        logits_dtype=torch.bfloat16,
                        max_seqlen_k=logits_capacity,
                    )
                elif q_dtype == torch.int8:
                    logits = int8_mqa_logits(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        prefill_weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                        max_seqlen_k=logits_capacity,
                    )
                elif q_dtype == torch.float8_e4m3fn:
                    logits = fp8_mqa_logits(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        prefill_weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                        max_seqlen_k=logits_capacity,
                    )
                else:
                    raise RuntimeError(
                        f"PPU mqa_logits get unsupported q_dtype: {q_dtype}"
                    )
            else:
                raise RuntimeError("indexer need PPU deep gemm installed")

            num_rows = logits.shape[0]
            row_starts = chunk.cu_seqlen_ks
            row_ends = chunk.cu_seqlen_ke
            # All non-paged policies use compressed, query-relative columns.
            row_ends = (row_ends - row_starts).clamp_min(0)
            row_starts = torch.zeros_like(row_ends)
            if candidate_blocks is not None:
                _filter_candidates(
                    logits,
                    row_starts,
                    row_ends,
                    candidate_blocks[chunk.token_start : chunk.token_end],
                    candidate_block_size,
                    candidate_write,
                )

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if _use_deep_select(logits, topk_tokens, topk_backend):
                # Compressed rows start at each query's first K position;
                # BF16 scores use the wheel with per-row valid lengths.
                deep_select_topk(logits, topk_tokens, row_ends, topk_indices)
            elif logits.dtype == torch.bfloat16:
                warn_deep_select_fallback()
                sail_ops.top_k_per_row_prefill_bf16(
                    logits,
                    row_starts,
                    row_ends,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    row_starts,
                    row_ends,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )

        if is_deep_gemm_supported():
            # All paged DSA policies in 4869f1e use next_n=1. Q, weights,
            # contexts and page tables must share the same query row order.
            decode_weights = weights[:num_decode_tokens]
            if decode_metadata.requires_padding:
                decode_weights = pack_seq_triton(
                    decode_weights, decode_lens, pad_value=0
                ).reshape(num_padded_tokens, -1)
            contexts, tables, request_ids, schedule = _paged_decode_inputs(
                attn_metadata_narrowed,
                seq_lens,
                decode_lens,
                next_n,
                kv_cache.shape[1],
                padded_q_quant_cast.shape[-2],
                head_dim,
                q_dtype,
                use_fp4_cache,
            )
            flat_q = padded_q_quant_cast.reshape(
                num_padded_tokens, 1, *padded_q_quant_cast.shape[2:]
            )
            if use_fp4_cache:
                logits = fp8_fp4_paged_mqa_logits(
                    (
                        flat_q,
                        padded_q_scale.reshape(
                            num_padded_tokens, 1, *padded_q_scale.shape[2:]
                        ),
                    ),
                    kv_cache,
                    decode_weights,
                    contexts,
                    tables,
                    schedule,
                    max_model_len=max_model_len,
                    clean_logits=False,
                    indices=request_ids,
                )
            elif q_dtype == torch.int8:
                logits = int8_paged_mqa_logits(
                    flat_q,
                    kv_cache,
                    decode_weights,
                    contexts,
                    tables,
                    schedule,
                    max_model_len=max_model_len,
                    clean_logits=False,
                )
            elif q_dtype == torch.float8_e4m3fn:
                logits = fp8_paged_mqa_logits(
                    flat_q,
                    kv_cache,
                    decode_weights,
                    contexts,
                    tables,
                    schedule,
                    max_model_len=max_model_len,
                    clean_logits=False,
                )
            else:
                raise RuntimeError(
                    f"PPU paged mqa_logits get unsupported q_dtype: {q_dtype}"
                )
        else:
            raise RuntimeError("indexer need ppu deep gemm installed")
        seq_lens = contexts.reshape(batch_size, next_n)
        num_rows = logits.shape[0]
        if candidate_blocks is not None:
            decode_candidates = candidate_blocks[:num_decode_tokens]
            if decode_metadata.requires_padding:
                decode_candidates = pack_seq_triton(
                    decode_candidates, decode_lens, pad_value=-1
                ).reshape(num_padded_tokens, -1)
            vis = seq_lens.reshape(-1)
            row_repeat = next_n if vis.numel() != num_rows else 1
            _filter_candidates(
                logits,
                None,
                vis[:num_rows],
                decode_candidates,
                candidate_block_size,
                candidate_write,
                row_repeat,
            )
            if candidate_write and decode_metadata.requires_padding:
                candidate_blocks[:num_decode_tokens].copy_(
                    unpack_seq_triton(
                        decode_candidates.reshape(batch_size, next_n, -1), decode_lens
                    )
                )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if _use_deep_select(logits, topk_tokens, topk_backend):
            row_ends = seq_lens.reshape(-1)
            deep_select_topk(logits, topk_tokens, row_ends, topk_indices)
        elif topk_tokens in (512, 1024, 2048):
            # The existing decode kernels consume FP32 only. This expansion
            # preserves BF16 scores exactly and never rounds FP32 into BF16.
            if logits.dtype == torch.bfloat16:
                warn_deep_select_fallback()
                logits = logits.float()
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            if logits.dtype == torch.bfloat16:
                warn_deep_select_fallback()
                logits = logits.float()
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def ppu_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    return topk_indices_buffer


def _use_deep_select(logits, topk, backend):
    supported = can_use_deep_select(logits, topk)
    if backend == "deep_select" and not supported:
        raise RuntimeError(
            "deep_select was requested but the SAIL BF16 wheel, device or "
            "shape is unsupported; FP32 logits are never rounded to BF16"
        )
    return backend in ("auto", "deep_select") and supported


def _paged_decode_inputs(
    step, seq_lens, decode_lens, next_n, page_size, heads, head_dim, q_dtype, fp4
):
    """Reuse a schedule only for the matching builder-owned decode layout."""
    num_sms = get_num_sms()
    geometry = (1, heads, head_dim, torch.int8 if fp4 else q_dtype)
    metadata = step.decode
    tables = metadata.block_table
    rows = decode_lens.shape[0]
    indices = metadata.indices if fp4 else None
    if (
        getattr(step, "_sail_paged_schedule_key", None)
        == (page_size, num_sms, geometry, fp4)
        and next_n == 1
        and not metadata.requires_padding
        and seq_lens.shape == (rows, 1)
        and seq_lens.is_contiguous()
        and tables.shape[0] == rows
        and tables.is_contiguous()
        and current_platform.is_ppu()
    ):
        if not fp4 and q_dtype == torch.int8:
            if current_platform.is_device_capability((8, 0)):
                return seq_lens, tables, None, metadata.schedule_metadata
        elif (
            (fp4 or q_dtype == torch.float8_e4m3fn)
            and current_platform.is_device_capability(89)
            and (
                not fp4
                or (
                    indices is not None
                    and indices.shape == (rows,)
                    and indices.is_contiguous()
                )
            )
        ):
            return (
                seq_lens,
                tables,
                indices,
                metadata.schedule_metadata,
            )

    contexts, tables, request_ids = _paged_decode_rows(
        seq_lens,
        tables,
        decode_lens,
        next_n,
        indices,
        metadata.per_req_decode_lens,
        need_request_ids=fp4,
    )
    schedule = get_paged_mqa_logits_metadata(
        contexts, page_size, num_sms, geometry, indices=request_ids if fp4 else None
    )
    return contexts, tables, request_ids if fp4 else None, schedule


def _paged_decode_rows(
    seq_lens,
    block_table,
    decode_lens,
    next_n,
    indices,
    per_req_decode_lens=None,
    *,
    need_request_ids=True,
):
    """Request-major [Q,1] rows, including safe, separately labelled padding."""
    batch = decode_lens.shape[0]
    device = seq_lens.device
    positions = torch.arange(next_n, device=device, dtype=torch.int32)
    if seq_lens.numel() == batch * next_n:
        contexts = seq_lens.reshape(batch, next_n)
        # Native metadata is right-aligned to the maximum verify count,
        # whereas pack_seq_triton left-aligns each request's actual queries.
        # Gather the already compressed contexts rather than adding token
        # offsets in compressed units (which is wrong for C2 indexers).
        source_cols = (next_n - decode_lens.reshape(batch, 1) + positions).clamp(
            0, next_n - 1
        )
        contexts = contexts.gather(1, source_cols.to(torch.int64))
    else:
        contexts = (
            seq_lens.reshape(batch, 1) - decode_lens.reshape(batch, 1) + 1 + positions
        ).clamp_min(0)
    valid = positions.unsqueeze(0) < decode_lens.reshape(batch, 1)
    contexts = torch.where(valid, contexts, 0).reshape(-1, 1).contiguous()
    rows = batch * next_n
    tables = block_table.repeat_interleave(next_n, dim=0)
    if not need_request_ids:
        if next_n == 1 and per_req_decode_lens is not None:
            valid_rows = torch.arange(rows, device=device) < per_req_decode_lens.sum()
            contexts = torch.where(valid_rows[:, None], contexts, 0)
        return contexts, tables, None

    if indices is None and next_n == 1 and per_req_decode_lens is not None:
        # The non-varlen upstream flattening path replaces decode_lens by
        # ones, but retains original verify counts separately. Recover true
        # owners without a GPU->CPU read or a dynamic repeat_interleave size.
        request_ends = per_req_decode_lens.to(torch.int32).cumsum(0)
        request_ids = torch.searchsorted(
            request_ends, torch.arange(rows, device=device), right=True
        ).to(torch.int32)
        valid = valid.reshape(-1) & (request_ids < per_req_decode_lens.numel())
        contexts = torch.where(valid.reshape(-1, 1), contexts, 0)
    elif indices is None:
        request_ids = torch.arange(batch, device=device, dtype=torch.int32)
        request_ids = request_ids.repeat_interleave(next_n)
    else:
        request_ids = indices.reshape(-1).to(torch.int32)
        if request_ids.numel() != rows:
            raise ValueError("Request IDs must cover every flattened query")
    # Real request IDs are nonnegative local indices; negative labels isolate
    # padding without colliding with any real request, even after flattening.
    dummy_ids = -1 - torch.arange(rows, device=device, dtype=torch.int32)
    request_ids = torch.where(valid.reshape(-1), request_ids, dummy_ids).contiguous()
    return contexts, tables, request_ids


def _filter_candidates(
    logits, starts, ends, candidates, block_size, write, row_repeat=1
):
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
        select_candidate_blocks,
    )

    assert block_size > 0
    if write:
        select_candidate_blocks(
            logits,
            starts,
            ends,
            candidates.shape[1],
            block_size,
            candidates,
            row_repeat,
        )
    else:
        apply_candidate_mask(logits, starts, ends, candidates, block_size, row_repeat)
