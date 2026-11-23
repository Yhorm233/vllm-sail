# SPDX-License-Identifier: Apache-2.0
"""Step5 selector_v2: exact FP32 cutoff, reusing upstream's stabilization and
page-write contract.
"""

from __future__ import annotations


def workspace_capacity(regions, topk):
    if type(regions) is not int or regions <= 0:
        raise ValueError("regions must be a positive integer")
    if type(topk) is not int or not 0 < topk <= 1024:
        raise ValueError("topk must be in 1..1024")
    return 0 if regions <= 16384 else ((regions + 16383) // 16384) * topk


def raw_select(
    scores,
    positions,
    table,
    workspace,
    regions,
    starts,
    *,
    region_block=8,
    page_size=64,
    topk=512,
    packed=False,
    threads=0,
):
    """Raw unordered candidates, current in the last column.

    select_regions must order tied IDs stably.
    """
    from vllm_sail import _custom_ops

    _custom_ops.step5_dsa_selector_v2(
        scores,
        positions,
        table,
        workspace,
        regions,
        starts,
        region_block,
        page_size,
        topk,
        packed,
        threads,
    )


def allocate(scores, topk):
    import torch

    rows, regions = scores.shape
    capacity = workspace_capacity(regions, topk)
    return (
        torch.empty((rows, 1, capacity), dtype=torch.int32, device=scores.device),
        torch.empty((rows, 1, topk + 1), dtype=torch.int32, device=scores.device),
        torch.empty((rows, 1, topk + 1), dtype=torch.int32, device=scores.device),
        torch.empty(0, dtype=torch.int32, device=scores.device),
    )


def select_regions(
    scores,
    current,
    topk,
    backend="auto",
    *,
    topk_lengths=None,
    paged_prefill=None,
    sort=True,
    threads=0,
):
    """Ordered int64 selections identical to Step5 per_row.

    Includes all allocation and stabilization cost.
    """
    import torch
    from vllm.models.step5.common.ops.sparse_indexer import (
        _stable_topk_regions,
        select_all_regions,
    )
    from vllm.triton_utils import triton

    if scores.ndim != 2 or scores.dtype != torch.float32 or not scores.is_contiguous():
        raise ValueError("scores must be contiguous FP32 [rows,regions]")
    rows, regions = scores.shape
    workspace_capacity(regions, topk)
    if backend not in ("auto", "per_row"):
        raise ValueError("selector_v2 only supports auto/per_row")
    k = min(topk, regions)
    if (
        current.shape != (rows,)
        or current.dtype != torch.int64
        or not current.is_contiguous()
    ):
        raise ValueError("current must be contiguous int64 [rows]")
    if current.device != scores.device:
        raise ValueError("scores/current must be on the same device")
    if regions <= topk:
        if paged_prefill is not None:
            raise ValueError("Fused prefill pages require scored prefill only")
        return select_all_regions(current, k)
    output = torch.empty((rows, k + 1), device=scores.device, dtype=torch.int64)
    if not rows:
        return output
    if paged_prefill is not None and paged_prefill.block_size not in (16, 32, 64, 128):
        raise ValueError("Unsupported prefill cache block size")
    selection_lengths = current
    if topk_lengths is not None:
        if (
            topk_lengths.shape != (rows,)
            or topk_lengths.dtype != torch.int32
            or topk_lengths.device != scores.device
            or not topk_lengths.is_contiguous()
        ):
            raise ValueError(
                "topk_lengths must be contiguous int32 [rows] on the same device"
            )
        selection_lengths = topk_lengths.to(torch.int64)
    workspace, candidates, starts, table = allocate(scores, k)
    raw_select(
        scores.view(rows, 1, regions),
        selection_lengths * 8,
        table,
        workspace,
        candidates,
        starts,
        topk=k,
        threads=threads,
    )
    scratch = torch.empty((rows, k + 1), device=scores.device, dtype=torch.int32)
    page_kwargs = {}
    if paged_prefill is not None:
        page_kwargs = dict(
            PREPARE_PREFILL=True,
            Owners=paged_prefill.owners,
            Table=paged_prefill.block_table,
            Pages=paged_prefill.indices,
            Indptr=paged_prefill.indptr,
            OWNER_STRIDE=paged_prefill.owners.stride(0),
            TABLE_ROW=paged_prefill.block_table.stride(0),
            TABLE_COL=paged_prefill.block_table.stride(1),
            INDPTR_STRIDE=paged_prefill.indptr.stride(0),
            BLOCK_SIZE=paged_prefill.block_size,
            PAGE_STRIDE=paged_prefill.page_stride,
        )
    # The original per_row path sorts stably even with sort=False; both paths match.
    _stable_topk_regions[(rows,)](
        scores,
        current,
        candidates,
        scratch,
        output,
        scores.stride(0),
        k + 1,
        k,
        triton.next_power_of_2(k),
        1024,
        triton.next_power_of_2(k + 1),
        **page_kwargs,
        SORT=True,
    )
    return output
