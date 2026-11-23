# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Run Step5 DSA on PPU (SM 8.x) without FlashInfer's attn_scores module.

Upstream admits only SM90/SM100/SM103 and builds its indexer scorer from
``flashinfer.attn_scores``, which the PPU FlashInfer build does not ship. PPU
runs upstream's own Triton scorer (``paged_mqa_logits_sm90``), exactly like
SM90: natively on SM 8.9, through the software E4M3 kernels of
``step5_soft_fp8`` on PPU 1.0 (SM 8.0). That scorer ignores
``schedule_meta``, so the FlashInfer schedule is replaced with ``None``.

The copied bodies run in ``flashinfer_sparse``'s globals, which have no
``current_platform``; the platform check therefore sits in the wrappers below,
and non-PPU platforms call upstream unchanged unless
``VLLM_STEP5_INDEXER_FP8_AS_FP16`` is set: FP16 indexer storage
(``step5_indexer_fp16``) is scored only by the Triton scorer of ``soft_fp8``,
which then replaces upstream's on every platform.

On PPU, ``select_regions`` also resolves ``auto`` / ``deep_select`` like
upstream does on SM100: the SAIL DeepSelect wheel runs when it applies
(``vllm_sail.utils.deep_select.can_use_deep_select``: SM 8.9, BF16,
``VLLM_SAIL_USE_DEEPSELECT``); otherwise an explicit ``deep_select`` falls back
to ``per_row`` (topKPerRowDecode) before upstream picks the sort and stride,
and ``auto`` keeps upstream's ``per_row``.
"""

from __future__ import annotations

from vllm.models.step5.nvidia import flashinfer_sparse as _sparse
from vllm.models.step5.nvidia.flashinfer_sparse import Step5FlashInferSparse
from vllm.models.step5.nvidia.sparse_attention import Step5DSAAttentionBackend
from vllm.platforms import current_platform

from vllm_sail.models.step5.ops import soft_fp8
from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch
from vllm_sail.utils import deep_select

_TARGET = "vllm.models.step5.nvidia.flashinfer_sparse"
_AFFECTED = ">=0.32.0,<0.33.0"
_REASON_SCORES = (
    "Step5's sparse adapter imports flashinfer.attn_scores for its indexer "
    "scorer and schedule; the PPU FlashInfer build has no such module. PPU "
    "runs upstream's Triton SM90 scorer, which ignores schedule_meta."
)
_REMOVE_WHEN = (
    "Step5 selects its indexer scorer and schedule through platform "
    "capabilities instead of importing flashinfer.attn_scores unconditionally."
)

_upstream_supports_compute_capability = (
    Step5DSAAttentionBackend.supports_compute_capability.__func__
)
_upstream_init = Step5FlashInferSparse.__init__
_upstream_warmup_prefill_scoring = Step5FlashInferSparse.warmup_prefill_scoring
_upstream_prefill_pair_inputs = Step5FlashInferSparse.prefill_pair_inputs
_upstream_select_regions = Step5FlashInferSparse.select_regions


@patch(
    "vllm.models.step5.nvidia.sparse_attention",
    "Step5DSAAttentionBackend.supports_compute_capability",
    reason=(
        "Step5 DSA admits only SM90, SM100 and SM103. PPU reports SM 8.x; its "
        "Triton DSA kernels use native E4M3 on SM 8.9 and the software E4M3 "
        "of step5_soft_fp8 on SM 8.0."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "Step5DSAAttentionBackend admits SM 8.x or defers to platform "
        "capability checks."
    ),
)
@classmethod
def supports_compute_capability(cls, capability) -> bool:
    # PPU MODIFICATION: begin — SM 8.9 PPU has native E4M3; PPU 1.0 (SM 8.0)
    # runs the software E4M3 kernels installed by step5_soft_fp8.
    if current_platform.is_ppu() and capability.major == 8:
        return True
    # PPU MODIFICATION: end
    return _upstream_supports_compute_capability(cls, capability)


def __init__(self, device: torch.device):
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    config = get_current_vllm_config_or_none()
    self._topk_backend = (
        config.kernel_config.sparse_indexer_topk_backend if config else "auto"
    )
    # PPU MODIFICATION: begin — no flashinfer.attn_scores on PPU; its native
    # E4M3 runs upstream's Triton scorer like SM90.
    from ..common.ops.sparse_indexer_scores import paged_mqa_logits_sm90

    self._logits = paged_mqa_logits_sm90
    # PPU MODIFICATION: end
    properties = torch.cuda.get_device_properties(device)
    self._num_sms = properties.multi_processor_count
    self._workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
    self._attention = BatchPrefillWithRaggedKVCacheWrapper(
        self._workspace, "NHD", backend="fa2"
    )
    self._prefill_plan_event = torch.cuda.Event()
    self._paged_prefill_plans: list[CompactPagedPrefill | None] = [None, None]
    self._paged_prefill_index = 0


def warmup_prefill_scoring(self, max_regions, topk, paired_heads, device):
    """JIT the scored-prefill variants a startup warmup step cannot reach.

    Per-chunk metadata specializes on the chunk size, which shrinks as the
    context grows, and paired logits compile for ``next_n = 2``; either
    would otherwise compile mid-serving on the first such prefill.
    """
    from ..common.ops.prefill_indexer_metadata import prefill_indexer_chunk_size

    rows = torch.arange(2, device=device)
    owners = torch.zeros(2, dtype=torch.long, device=device)
    chunk_regions: dict[int, int] = {}
    for regions in range(topk + 1, max(max_regions, topk + 1) + 1, 256):
        chunk_regions.setdefault(prefill_indexer_chunk_size(regions), regions)
    for regions in chunk_regions.values():
        positions = rows + (regions - 1) * 8
        self.prefill_metadata(owners, positions, regions, topk)
    if paired_heads:
        regions = topk + 1
        pages = (regions + 127) // 128 * 4
        # PPU MODIFICATION: begin — FP16 indexer storage packs 258 FP16
        # elements per key and scores an FP16 query.
        from vllm_sail.models.step5.ops.soft_fp8 import indexer_query_dtype

        query_dtype = indexer_query_dtype()
        half = query_dtype == torch.float16
        packed = torch.zeros(
            (pages, 32, 1, 258 if half else 260),
            dtype=torch.float16 if half else torch.uint8,
            device=device,
        )
        # PPU MODIFICATION: end
        lengths = torch.full((1,), regions, dtype=torch.int32, device=device)
        self._logits(
            q=torch.zeros(
                # PPU MODIFICATION: begin — FP16 or E4M3 indexer query.
                (1, 2, paired_heads, 256),
                dtype=query_dtype,
                device=device,
                # PPU MODIFICATION: end
            ),
            kv_fused=packed,
            weights=torch.zeros((2, paired_heads), device=device),
            seq_lens=lengths,
            block_tables=torch.arange(pages, dtype=torch.int32, device=device)[
                None
            ],
            max_seq_len=regions,
            # PPU MODIFICATION: begin — the Triton scorer ignores the schedule.
            schedule_meta=None,
            # PPU MODIFICATION: end
        )


def prefill_pair_inputs(self, indexer):
    """Per-chunk (context, table, schedule) of consecutive row pairs."""
    pairs = []
    for meta in indexer:
        lengths = meta.lengths[1::2].contiguous()
        pairs.append(
            (
                lengths,
                meta.block_table[::2].contiguous(),
                # PPU MODIFICATION: begin — the Triton scorer ignores the
                # schedule.
                None,
                # PPU MODIFICATION: end
            )
        )
    return pairs


_ppu_init = bind_body(__init__, _sparse)
_ppu_warmup_prefill_scoring = bind_body(warmup_prefill_scoring, _sparse)
_ppu_prefill_pair_inputs = bind_body(prefill_pair_inputs, _sparse)


def _use_triton_scorer() -> bool:
    return current_platform.is_ppu() or soft_fp8.indexer_fp16()


@patch(
    _TARGET,
    "Step5FlashInferSparse.__init__",
    reason=_REASON_SCORES,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _init(self, device):
    if not _use_triton_scorer():
        return _upstream_init(self, device)
    _ppu_init(self, device)
    if soft_fp8.indexer_fp16():
        # Upstream's scorer (and its performance wrapper) asserts E4M3 bytes;
        # this one also scores FP16 pages.
        self._logits = soft_fp8.paged_mqa_logits_sm90


@patch(
    _TARGET,
    "Step5FlashInferSparse.warmup_prefill_scoring",
    reason=_REASON_SCORES,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _warmup_prefill_scoring(self, max_regions, topk, paired_heads, device):
    if not _use_triton_scorer():
        return _upstream_warmup_prefill_scoring(
            self, max_regions, topk, paired_heads, device
        )
    return _ppu_warmup_prefill_scoring(self, max_regions, topk, paired_heads, device)


@patch(
    _TARGET,
    "Step5FlashInferSparse.prefill_pair_inputs",
    reason=_REASON_SCORES,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def _prefill_pair_inputs(self, indexer):
    if not _use_triton_scorer():
        return _upstream_prefill_pair_inputs(self, indexer)
    return _ppu_prefill_pair_inputs(self, indexer)


@patch(
    _TARGET,
    "Step5FlashInferSparse.select_regions",
    reason=(
        "Step5 resolves auto to deep_select only on the SM100 family and an "
        "explicit deep_select raises elsewhere; PPU selects the deep_select "
        "wheel (SM 8.9, BF16) when it applies and per_row topKPerRowDecode "
        "otherwise."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "Step5 resolves its region top-k backend through platform "
        "capabilities that PPU can provide."
    ),
)
def _select_regions(
    self, scores, current, topk, *, topk_lengths=None, paged_prefill=None, sort=True
):
    if not current_platform.is_ppu():
        return _upstream_select_regions(
            self,
            scores,
            current,
            topk,
            topk_lengths=topk_lengths,
            paged_prefill=paged_prefill,
            sort=sort,
        )
    backend = self._topk_backend
    if scores.shape[1] > topk and backend in ("auto", "deep_select"):
        if deep_select.can_use_deep_select(scores, topk):
            backend = "deep_select"
        elif backend == "deep_select":
            # Step5 scores are FP32, which the BF16 wheel never rounds.
            deep_select.warn_deep_select_fallback()
            backend = "per_row"
            # topKPerRowDecode reads FP32 only; the upcast keeps the order.
            scores = scores.float()
    # Looked up at call time: the selector_v2 performance wrapper replaces
    # the module-level select_regions and still sees auto / per_row.
    return _sparse.select_regions(
        scores,
        current,
        topk,
        backend,
        topk_lengths=topk_lengths,
        paged_prefill=paged_prefill,
        sort=sort,
    )
