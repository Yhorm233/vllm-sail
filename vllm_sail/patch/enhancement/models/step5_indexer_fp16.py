# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""FP16 storage for the Step5 indexer (``VLLM_STEP5_INDEXER_FP8_AS_FP16=1``).

Values keep their E4M3 quantization while the summary cache, packed pages and
Q are stored as FP16. Scoring uses an FP16 dot with FP32 accumulation and the
summary cache takes twice the memory; the projection weights and the main
attention KV cache quantization are unchanged.

The kernels themselves branch on tensor dtype in
``vllm_sail/models/step5/ops/soft_fp8.py`` (installed by ``step5_soft_fp8``).
This module switches the storage:

* the summary-cache spec and the two indexer launchers allocate FP16;
* summary packing copies FP16 pages (258 elements per key: 256 values and
  one FP32 unit scale), through the dtype-aware page-copy helpers;
* ``Step5FlashInferSparse`` accepts FP16 query/summary pairs in its decode and
  prefill scorers; its ``__init__`` (``step5_dsa_backend``) selects the
  FP16-aware Triton scorer.

With the switch unset every wrapper calls upstream unchanged; the page-copy
helpers reduce to upstream's byte arithmetic.
"""

from __future__ import annotations

import dataclasses
import functools

import torch
from vllm.models.step5.nvidia import flashinfer_sparse as _sparse
from vllm.models.step5.nvidia.flashinfer_sparse import Step5FlashInferSparse
from vllm.models.step5.common.ops import (
    indexer_norm_rope as _indexer,
    sparse_indexer as _pack,
)
from vllm.models.step5.nvidia.ops import qkv_indexer_norm_rope as _joint
from vllm.models.step5.common.sparse_summary_cache import (
    Step5SparseSummaryCacheLayer,
)

from vllm_sail.models.step5.ops import soft_fp8
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.32.0,<0.33.0"
_REASON = (
    "VLLM_STEP5_INDEXER_FP8_AS_FP16=1 stores E4M3-quantized values as FP16, "
    "using FP16 dot products with FP32 accumulation; the main attention KV cache "
    "quantization is unchanged. "
    "Upstream only supports E4M3 storage."
)
_REMOVE_WHEN = (
    "Step5 DSA supports an FP16 indexer storage mode, or PPU drops it after "
    "native E4M3 scoring is preferred everywhere."
)

_upstream_indexer_norm_rope = _indexer.indexer_norm_rope
_upstream_qkv_indexer_norm_rope = _joint.qkv_indexer_norm_rope
_upstream_pack_summaries = _pack._pack_summaries
_upstream_get_kv_cache_config = Step5SparseSummaryCacheLayer.get_kv_cache_config
_upstream_decode_logits = Step5FlashInferSparse.decode_logits
_upstream_logits = Step5FlashInferSparse.logits


def _install(module: str, name: str, replacement) -> None:
    patch(
        module,
        name,
        reason=_REASON,
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )(replacement)


# Copied from vllm.models.step5.common.ops.indexer_norm_rope.indexer_norm_rope.
def indexer_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    zero_centered: bool,
    native_rope: bool = False,
    *,
    num_groups: int = 1,
    decode_cache=None,
    decode_ring=None,
    decode_z: torch.Tensor | None = None,
    decode_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare indexer Q/K/weights and update the decode CSA state."""
    tokens, width = query.shape
    if (decode_cache is None) != (decode_ring is None):
        raise ValueError("CSA fusion needs both the summary cache and the ring")
    if decode_cache is not None:
        if decode_z is None or decode_slots is None:
            raise ValueError("CSA fusion needs decode z and slots")
    elif decode_z is not None or decode_slots is not None:
        raise ValueError("CSA decode inputs need a summary cache")
    if decode_slots is not None and decode_slots.numel() > tokens:
        raise ValueError("CSA decode rows must fit the indexer token count")
    if num_groups < 1 or width % (num_groups * 256):
        raise ValueError("Indexer query heads must divide evenly into provider groups")
    heads = width // (num_groups * 256)
    padded_heads = max(heads, 8)
    q = torch.empty(
        (num_groups, tokens, padded_heads, 256),
        device=query.device,
        # PPU MODIFICATION: begin — FP16 holds the kernel's E4M3-quantized values.
        dtype=torch.float16,
        # PPU MODIFICATION: end
    )
    k = torch.empty((tokens, 1, 256), device=key.device, dtype=key.dtype)
    w = torch.empty(
        (num_groups, tokens, padded_heads), device=weights.device, dtype=torch.float32
    )
    if tokens:
        _indexer_norm_rope[(tokens, num_groups * padded_heads + 1)](
            query,
            key,
            weights,
            q_weight,
            k_weight,
            k_bias,
            cos_sin_cache,
            positions,
            q,
            k,
            w,
            decode_z,
            decode_slots,
            decode_cache.mean if decode_cache is not None else None,
            decode_ring.rows if decode_ring is not None else None,
            decode_ring.metadata().base if decode_ring is not None else None,
            decode_cache is not None,
            decode_slots.numel() if decode_slots is not None else 0,
            decode_z.stride(0) if decode_z is not None else 0,
            decode_slots.stride(0) if decode_slots is not None else 0,
            decode_ring.capacity if decode_ring is not None else 8,
            *(
                summary_layout(decode_cache.mean)
                if decode_cache is not None
                else (0, 8)
            ),
            query.stride(0),
            key.stride(0),
            weights.stride(0),
            heads,
            padded_heads,
            num_groups,
            tokens,
            cos_sin_cache.shape[1],
            cos_sin_cache.stride(0),
            eps,
            float(zero_centered),
            native_rope,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.transpose(0, 1), k, w.transpose(0, 1)


# Copied from
# vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope.qkv_indexer_norm_rope.
def qkv_indexer_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    zero_centered: bool,
    native_rope: bool = False,
    *,
    main_inputs,
    decode_cache=None,
    decode_ring=None,
    decode_z: torch.Tensor | None = None,
    decode_slots: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Prepare indexer/attention tensors and write their separate cache states."""
    (
        main_qkv,
        main_qw,
        main_kw,
        main_rope,
        main_slots,
        main_kc,
        main_vc,
        main_qh,
        main_kh,
        main_rotary,
        main_eps,
        main_bias,
        main_k_scale,
        main_v_scale,
    ) = main_inputs
    fp8_cache = main_kc.dtype in (torch.uint8, torch.float8_e4m3fn)
    if fp8_cache:
        main_kc = main_kc.view(torch.float8_e4m3fn)
        main_vc = main_vc.view(torch.float8_e4m3fn)
    tokens, width = query.shape
    main_q = main_qkv.new_empty((tokens, main_qh * 192))
    main_k = main_qkv.new_empty((tokens, main_kh * 192))
    main_v = torch.empty_like(main_k)
    main_group = 4 if main_rotary > 128 else 8
    if (decode_cache is None) != (decode_ring is None):
        raise ValueError("CSA fusion needs both the summary cache and the ring")
    if decode_cache is not None:
        if decode_z is None or decode_slots is None:
            raise ValueError("CSA fusion needs decode z and slots")
    elif decode_z is not None or decode_slots is not None:
        raise ValueError("CSA decode inputs need a summary cache")
    if decode_slots is not None and decode_slots.numel() > tokens:
        raise ValueError("CSA decode rows must fit the indexer token count")
    # One indexer group per local KV head; outputs are group-major.
    groups = main_kh
    if width % (groups * 256):
        raise ValueError("Indexer query heads must divide evenly into KV groups")
    heads = width // (groups * 256)
    padded_heads = max(heads, 8)
    q = torch.empty(
        (groups, tokens, padded_heads, 256),
        device=query.device,
        # PPU MODIFICATION: begin — FP16 holds the kernel's E4M3-quantized values.
        dtype=torch.float16,
        # PPU MODIFICATION: end
    )
    k = torch.empty((tokens, 1, 256), device=key.device, dtype=key.dtype)
    w = torch.empty(
        (groups, tokens, padded_heads), device=weights.device, dtype=torch.float32
    )
    if tokens:
        indexer_programs = groups * padded_heads + 1
        main_programs = triton.cdiv(main_qh + 2 * main_kh, main_group)
        _joint_prepare[(tokens, indexer_programs + main_programs)](
            query,
            key,
            weights,
            q_weight,
            k_weight,
            k_bias,
            cos_sin_cache,
            positions,
            q,
            k,
            w,
            decode_z,
            decode_slots,
            decode_cache.mean if decode_cache is not None else None,
            decode_ring.rows if decode_ring is not None else None,
            decode_ring.metadata().base if decode_ring is not None else None,
            decode_cache is not None,
            decode_slots.numel() if decode_slots is not None else 0,
            decode_z.stride(0) if decode_z is not None else 0,
            decode_slots.stride(0) if decode_slots is not None else 0,
            decode_ring.capacity if decode_ring is not None else 8,
            *(
                summary_layout(decode_cache.mean)
                if decode_cache is not None
                else (0, 8)
            ),
            query.stride(0),
            key.stride(0),
            weights.stride(0),
            heads,
            padded_heads,
            groups,
            tokens,
            cos_sin_cache.shape[1],
            cos_sin_cache.stride(0),
            eps,
            float(zero_centered),
            native_rope,
            main_qkv,
            main_qw,
            main_kw,
            main_rope,
            main_slots,
            main_q,
            main_k,
            main_v,
            main_kc,
            main_vc,
            main_qh,
            main_kh,
            main_qkv.stride(0),
            main_rotary,
            main_rope.stride(0),
            main_slots.numel(),
            main_kc.shape[1],
            main_kc.stride(0),
            main_kc.stride(1),
            main_kc.stride(2),
            main_eps,
            main_bias,
            main_k_scale,
            main_v_scale,
            fp8_cache,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.transpose(0, 1), k, w.transpose(0, 1), main_q, main_k, main_v


# Copied from vllm.models.step5.common.ops.sparse_indexer._pack_summaries.
def _pack_summaries(
    mean, block_table, regions, slots=None, seq_lens=None, num_sms=0, *, live_only=False
):
    # Blocks may be strided (block-outer KV layout); regions within one are not.
    assert mean.stride(1) == 256 and mean.stride(2) == 1
    pages = triton.cdiv(regions, 128) * 4
    batch = block_table.shape[0]
    # PPU MODIFICATION: begin — FP16 pages: 256 values and one FP32 unit scale
    # (two FP16 elements) per key.
    assert mean.dtype == torch.float16
    packed_width = 256 + 4 // mean.element_size()
    packed = torch.empty(
        (batch * pages, 32 * packed_width), device=mean.device, dtype=mean.dtype
    )
    # PPU MODIFICATION: end
    metadata = None
    if slots is not None:
        assert seq_lens is not None
        metadata = DecodeIndexerMetadata(
            torch.empty(batch, device=mean.device, dtype=torch.bool),
            torch.empty(batch, device=mean.device, dtype=torch.int64),
            torch.empty(batch, device=mean.device, dtype=torch.int64),
            torch.empty(batch, device=mean.device, dtype=torch.int32),
            torch.empty((batch, pages), device=mean.device, dtype=torch.int32),
            torch.empty(batch, device=mean.device, dtype=torch.int32),
            torch.empty((num_sms + 1, 2), device=mean.device, dtype=torch.int32)
            if num_sms
            else None,
        )
    if live_only and slots is None:
        raise ValueError("Live packing needs decode slots")
    tile_pages = 4 if batch >= 16 and pages >= 256 else 2
    workers = max(triton.cdiv(pages, tile_pages), 1) if live_only else max(pages, 1)
    # Constexprs only the decode schedule and live packing read. Prefill
    # passes constants: specializing on the context length or batch would
    # JIT a new variant mid-serving (~0.4 s) each time one crosses a power of 2.
    decode = metadata is not None
    block_batch = triton.next_power_of_2(max(batch, 1)) if decode else 1
    search_steps = max(batch, 1).bit_length() if decode else 1
    block_pages = triton.next_power_of_2(max(pages, 1)) if live_only else 1
    if batch and (pages or metadata is not None):
        _pack_summary_pages[(workers, batch)](
            mean,
            block_table,
            packed,
            packed[:, 32 * 256 :].view(torch.float32),
            slots,
            seq_lens,
            *(metadata if metadata is not None else (None,) * 7),
            regions,
            pages,
            mean.shape[0],
            mean.shape[1],
            mean.stride(0),
            block_table.shape[1],
            block_table.stride(0),
            metadata is not None,
            slots.stride(0) if slots is not None else 0,
            seq_lens.stride(0) if seq_lens is not None else 0,
            batch,
            block_batch,
            num_sms,
            triton.next_power_of_2(num_sms + 1),
            search_steps,
            LIVE_ONLY=live_only,
            BLOCK_PAGES=block_pages,
            TILE_PAGES=tile_pages if live_only else 1,
        )
    # PPU MODIFICATION: begin — FP16 page width.
    return packed.view(-1, 32, 1, packed_width), pages, metadata
    # PPU MODIFICATION: end


# Copied from vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.
def decode_logits(
    self,
    query,
    weights,
    mean,
    table,
    regions,
    slots,
    seq_lens,
    row_groups=1,
    query_len=1,
):
    """Score regions for (token, KV group) rows.

    ``query`` is ``[tokens, row_groups, heads, 256]`` and ``weights`` is
    ``[tokens, row_groups, heads]``; row ``r`` is group ``r % row_groups``
    of token ``r // row_groups``. ``query_len`` tokens share a request of
    ``table``/``seq_lens``.
    """
    assert query.shape[1] == row_groups
    if (
        query.shape[0] > 0
        and query.shape[2:] == (8, 256)
        # PPU MODIFICATION: begin — FP16 query over an FP16 summary cache.
        and query.dtype == torch.float16
        # PPU MODIFICATION: end
        and query.stride(3) == 1
        and query.stride(2) == 256
        and weights.shape == query.shape[:3]
        and weights.dtype == torch.float32
        and weights.stride(-1) == 1
        and (mean.shape[1:] == (8, 256) or mean.shape[1:] in ((2, 256), (4, 256)))
        # PPU MODIFICATION: begin — FP16 query over an FP16 summary cache.
        and mean.dtype == torch.float16
        # PPU MODIFICATION: end
        and mean.stride(1) == 256
        and mean.stride(2) == 1
        and table.stride(1) == 1
        and regions > 0
    ):
        return decode_summary_scores(
            query,
            weights,
            mean,
            table,
            slots,
            seq_lens,
            regions,
            num_sms=self._num_sms,
            row_groups=row_groups,
            query_len=query_len,
        )
    if query_len > 1:
        offset = torch.arange(slots.shape[0], device=slots.device) % query_len
        seq_lens = seq_lens.repeat_interleave(query_len) - (query_len - 1 - offset)
        table = table.repeat_interleave(query_len, dim=0)
    if row_groups > 1:
        slots, table, seq_lens = (
            t.repeat_interleave(row_groups, dim=0) for t in (slots, table, seq_lens)
        )
    rows = query.shape[0] * row_groups
    query = query.reshape(rows, 1, *query.shape[2:])
    weights = weights.reshape(rows, 1, weights.shape[2])
    packed, decode = self.pack_decode_summaries(
        mean, table, regions, slots, seq_lens
    )
    scores = self.logits(
        query,
        weights,
        packed,
        decode.block_table.shape[1],
        None,
        decode.lengths,
        regions,
        block_table=decode.block_table,
        schedule_meta=decode.schedule,
    )
    return scores, decode


# Copied from vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.
def logits(
    self,
    query,
    weights,
    packed,
    pages,
    requests,
    lengths,
    regions,
    *,
    block_table=None,
    schedule_meta=None,
    pairs=None,
    paired_heads=0,
):
    """With ``pairs`` (from ``prefill_pair_inputs``), consecutive row pairs
    of one request are scored as ``next_n=2`` on their ``paired_heads``
    real heads rather than zero-padding each row to the MMA minimum."""
    q = query[:, 0]
    w = weights[:, 0]
    # PPU MODIFICATION: begin — FP16 pages score an FP16 query as is.
    assert q.dtype == torch.float16
    if pairs is not None:
        assert q.shape[0] % 2 == 0
        # PPU MODIFICATION: end
        pair_lengths, pair_table, pair_schedule = pairs
        return self._logits(
            q=q[:, :paired_heads].unflatten(0, (-1, 2)),
            kv_fused=packed,
            weights=w[:, :paired_heads],
            seq_lens=pair_lengths,
            block_tables=pair_table,
            max_seq_len=regions,
            schedule_meta=pair_schedule,
        )
    q = q.unsqueeze(1)
    if block_table is None:
        block_table = (
            requests[:, None] * pages + torch.arange(pages, device=q.device)
        ).int()
    return self._logits(
        q=q,
        kv_fused=packed,
        weights=w.contiguous(),
        seq_lens=lengths.int().contiguous(),
        block_tables=block_table,
        max_seq_len=regions,
        schedule_meta=schedule_meta,
    )


_fp16_indexer_norm_rope = bind_body(indexer_norm_rope, _indexer)
_fp16_qkv_indexer_norm_rope = bind_body(qkv_indexer_norm_rope, _joint)
_fp16_pack_summaries = bind_body(_pack_summaries, _pack)
_fp16_decode_logits = bind_body(decode_logits, _sparse)
_fp16_logits = bind_body(logits, _sparse)


# Device helpers resolved by _pack_summary_pages through module globals; with
# byte summaries they compute upstream's offsets.
_install(
    "vllm.models.step5.common.ops.sparse_indexer",
    "_copy_summary_page",
    soft_fp8._copy_summary_page,
)
_install(
    "vllm.models.step5.common.ops.sparse_indexer",
    "_copy_summary_tile",
    soft_fp8._copy_summary_tile,
)


@functools.wraps(_upstream_indexer_norm_rope)
def _indexer_norm_rope_launcher(*args, **kwargs):
    if not soft_fp8.indexer_fp16():
        return _upstream_indexer_norm_rope(*args, **kwargs)
    return _fp16_indexer_norm_rope(*args, **kwargs)


@functools.wraps(_upstream_qkv_indexer_norm_rope)
def _qkv_indexer_norm_rope_launcher(*args, **kwargs):
    if not soft_fp8.indexer_fp16():
        return _upstream_qkv_indexer_norm_rope(*args, **kwargs)
    return _fp16_qkv_indexer_norm_rope(*args, **kwargs)


@functools.wraps(_upstream_pack_summaries)
def _pack_summaries_launcher(mean, *args, **kwargs):
    if mean.dtype != torch.float16:
        return _upstream_pack_summaries(mean, *args, **kwargs)
    return _fp16_pack_summaries(mean, *args, **kwargs)


# model.py imports both launchers inside its forward, so the module attributes
# are what it calls; pack_*summary_pages resolve _pack_summaries the same way.
_install(
    "vllm.models.step5.common.ops.indexer_norm_rope",
    "indexer_norm_rope",
    _indexer_norm_rope_launcher,
)
_install(
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope",
    "qkv_indexer_norm_rope",
    _qkv_indexer_norm_rope_launcher,
)
_install(
    "vllm.models.step5.common.ops.sparse_indexer",
    "_pack_summaries",
    _pack_summaries_launcher,
)


@patch(
    "vllm.models.step5.common.sparse_summary_cache",
    "Step5SparseSummaryCacheLayer.get_kv_cache_config",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _get_kv_cache_config(self, vllm_config):
    layer = _upstream_get_kv_cache_config(self, vllm_config)
    spec = layer.spec
    if soft_fp8.indexer_fp16():
        # Each region holds 256 E4M3-quantized FP16 values.
        spec = dataclasses.replace(spec, dtype=torch.float16)
        return dataclasses.replace(layer, spec=spec)
    return layer


@patch(
    "vllm.models.step5.nvidia.flashinfer_sparse",
    "Step5FlashInferSparse.decode_logits",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _decode_logits(
    self,
    query,
    weights,
    mean,
    table,
    regions,
    slots,
    seq_lens,
    row_groups=1,
    query_len=1,
):
    if query.dtype != torch.float16:
        return _upstream_decode_logits(
            self, query, weights, mean, table, regions, slots, seq_lens,
            row_groups, query_len,
        )
    return _fp16_decode_logits(
        self, query, weights, mean, table, regions, slots, seq_lens,
        row_groups, query_len,
    )


@patch(
    "vllm.models.step5.nvidia.flashinfer_sparse",
    "Step5FlashInferSparse.logits",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _logits(
    self,
    query,
    weights,
    packed,
    pages,
    requests,
    lengths,
    regions,
    *,
    block_table=None,
    schedule_meta=None,
    pairs=None,
    paired_heads=0,
):
    run = _fp16_logits if packed.dtype == torch.float16 else _upstream_logits
    return run(
        self, query, weights, packed, pages, requests, lengths, regions,
        block_table=block_table, schedule_meta=schedule_meta, pairs=pairs,
        paired_heads=paired_heads,
    )
