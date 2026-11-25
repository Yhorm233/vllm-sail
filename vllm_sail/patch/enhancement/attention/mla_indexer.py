# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
"""PPU patches for ``vllm.v1.attention.backends.mla.indexer`` (DSA indexer).

The fork's changes, translated to plugin patches. Everything else is left to
upstream: ``__init__`` delegates to the original method and re-applies only the
fork's PPU state, ``build`` delegates untouched, so upstream drift (new
helpers, new parameters, refactors) cannot break this plugin.

* ``PPU_DEEP_GEMM_TB_PER_CU`` sizes the PPU schedule buffer and is installed
  with ``patch_value``. Metadata describes quantization using torch.dtype.
* The fork's import switch (``vllm.utils.ppu_deep_gemm`` instead of
  ``vllm.utils.deep_gemm`` under ``is_ppu()``): ``has_deep_gemm`` is an
  identity (same object); ``get_paged_mqa_logits_metadata`` is rebound on the
  target module to the PPU implementation. PPU declares
  ``PlatformEnum.CUDA``, so upstream's decode build DOES enter its
  ``is_cuda() and has_deep_gemm()`` block; the rebound helper derives the
  fork's ``metadata_extra`` from the builder currently inside ``build``, so
  upstream's own buffer fill already uses the fp4 layout.
* Prefill chunking keeps upstream's FP32 budget. This is conservative for
  dense BF16 logits; the candidate-sparse backend already accounts for BF16
  in its split sequence lengths and must not have that saving applied twice.
* PPU MXFP4 uses upstream's SM100 varlen builder and its request IDs.
  Legacy FP8/INT8 use upstream's non-varlen rows, including the SM90-style
  flattening path for adaptive verification.
* ``DeepseekV32IndexerMetadataBuilder.__init__`` delegates to upstream, then
  adds the fork's PPU-only state: ``indexer_n_head``/``indexer_q_head_dim``
  from the KV-cache spec, the SM count from PPU DeepGEMM, and the enlarged
  fp4 scheduler-metadata buffer.
* ``DeepseekV32IndexerMetadataBuilder.build`` delegates to upstream, keeping
  the builder on ``_current_builder`` for the duration of the call so the
  rebound paged-metadata helper can read the fork's fp4 capability. Upstream
  owns the whole step build.
* ``_prepare_decode_tensors`` preserves MXFP4 query ownership from the
  separate per-request counts. The prebuilt metadata call uses these IDs
  instead of numbering already-flattened MTP queries as separate requests;
  ``build`` clears the temporary IDs on both success and failure.
* ``DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata`` (the MTP
  draft refresh) is the other call site that reaches the rebound helper, so it
  keeps the builder on ``_current_builder`` the same way; without it the fp4
  scheduler metadata would be rebuilt with the fp8 layout and fail upstream's
  shape assertion.
"""

from __future__ import annotations

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as _indexer_module
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
)

import vllm_sail.utils.deep_gemm as _ppu_deep_gemm
from vllm_sail.patch.utils import patch, patch_value

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.backends.mla.indexer"

#: Builder currently inside ``build``. The rebound paged-MQA-logits metadata
#: helper reads the fork's fp4 capability from it; upstream does not pass it.
_current_builder: list = []


patch_value(
    _MODULE,
    "PPU_DEEP_GEMM_TB_PER_CU",
    8,
    allow_missing=True,
    reason=(
        "Fork constant: thread blocks per compute unit used by PPU DeepGEMM "
        "when sizing the fp4 indexer scheduler-metadata buffer."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream defines the constant itself (patch_value becomes a no-op).",
)

# Fork's import switch: on PPU the indexer takes its DeepGEMM helpers from the
# PPU DeepGEMM wrapper. Only get_paged_mqa_logits_metadata needs rebinding —
# the plugin's wrapper re-exports the very same has_deep_gemm object upstream
# already binds, so that half of the switch is an identity. The patched build()
# and update_draft_decode_metadata() below keep their builder on
# _current_builder, so upstream's calls resolve the rebound name at call time
# with the fork's fp4 geometry available.
_upstream_paged_metadata = _indexer_module.get_paged_mqa_logits_metadata


def _ppu_paged_metadata(
    context_lens, block_size, num_sms, metadata_extra=None, indices=None
):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_paged_metadata(
            context_lens, block_size, num_sms, indices=indices
        )
    builder = _current_builder[-1] if _current_builder else None
    weighted_fp4 = builder is not None and getattr(builder, "indexer_uses_fp4", False)
    if metadata_extra is None:
        # PPU declares PlatformEnum.CUDA, so upstream's decode build does reach
        # its is_cuda() block and calls this helper without metadata_extra.
        # Derive the SAIL tuple from the builder inside build.
        # Every PPU-reachable call
        # site (build and update_draft_decode_metadata) runs behind a wrapper
        # that puts its builder on the stack.
        n_head = getattr(builder, "indexer_n_head", 0) if builder else 0
        if n_head > 0:
            heads_per_token = context_lens.shape[1] if context_lens.dim() == 2 else 1
            metadata_extra = (
                heads_per_token,
                n_head,
                builder.indexer_q_head_dim,
                (
                    torch.int8
                    if weighted_fp4 or current_platform.is_device_capability((8, 0))
                    else current_platform.fp8_dtype()
                ),
            )
    has_varlen_ids = indices is not None
    if metadata_extra is not None:
        # 4869f1e weighted metadata uses next_n=1; MXFP4 also needs request
        # ownership.
        batch = context_lens.shape[0]
        next_n = context_lens.shape[1] if context_lens.ndim == 2 else 1
        if weighted_fp4 and indices is None:
            indices = getattr(builder, "_sail_decode_request_ids", None)
            if indices is None:
                # Draft refresh has one query per request. During build,
                # _prepare_decode_tensors supplies the original ownership,
                # including when upstream has flattened MTP into Q rows.
                indices = torch.arange(
                    batch, dtype=torch.int32, device=context_lens.device
                ).repeat_interleave(next_n)
        context_lens = context_lens.reshape(-1, 1).contiguous()
        metadata_extra = (1, *metadata_extra[1:])
    # Legacy FP8/INT8 kernels do not accept request grouping.
    if not weighted_fp4:
        indices = None
    if metadata_extra is None:
        raise ValueError("SAIL paged DSA metadata requires indexer query head geometry")
    schedule = _ppu_deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_size, num_sms, metadata_extra, indices=indices
    )
    # PPU MODIFICATION: begin -- size upstream's copy buffer for the active ABI.
    buffer = getattr(builder, "scheduler_metadata_buffer", None)
    if buffer is not None and (
        buffer.shape[0] < schedule.shape[0] or buffer.shape[1] != schedule.shape[1]
    ):
        builder.scheduler_metadata_buffer = schedule.new_empty(schedule.shape)
    # PPU MODIFICATION: end
    if builder is not None and (
        (weighted_fp4 and getattr(builder, "supports_varlen", False) and has_varlen_ids)
        or (
            getattr(builder, "use_flattening", False)
            and not weighted_fp4
            and metadata_extra[-1] in (torch.int8, current_platform.fp8_dtype())
        )
    ):
        builder._sail_paged_schedule_key = (
            block_size,
            num_sms,
            metadata_extra,
            weighted_fp4,
        )
    return schedule


patch(
    _MODULE,
    "get_paged_mqa_logits_metadata",
    reason=(
        "Fork's is_ppu() import switch selects the PPU DeepGEMM wrapper's "
        "get_paged_mqa_logits_metadata, which accepts the PPU metadata_extra "
        "tuple. On PPU the rebound helper also derives metadata_extra from the "
        "builder inside build (PPU declares PlatformEnum.CUDA, so upstream's "
        "decode build reaches the same call), keeping the fp4 scheduler "
        "metadata correct in upstream's single fill. The CUDA helper has no "
        "such parameter."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vllm.utils.deep_gemm dispatches on current_platform internally, or "
        "upstream accepts a DeepGEMM backend registry."
    ),
)(_ppu_paged_metadata)


_upstream_supports_varlen_paged_mqa_logits = (
    _indexer_module._supports_varlen_paged_mqa_logits
)


@patch(
    _MODULE,
    "_supports_varlen_paged_mqa_logits",
    reason=(
        "SAIL SM89 accepts device-built varlen query rows and request IDs "
        "through the PPU next_n=1 paged-logits adapter, without an SM100 gate."
    ),
    affected_versions=_AFFECTED,
    remove_when="The indexer delegates varlen query-row support to the platform.",
)
def _supports_varlen_paged_mqa_logits() -> bool:
    from vllm.platforms import current_platform

    if current_platform.is_ppu():
        return (
            current_platform.is_device_capability(89)
            and _ppu_deep_gemm.is_deep_gemm_supported()
        )
    return _upstream_supports_varlen_paged_mqa_logits()


_upstream_supports_flattened_device_query_lens = (
    _indexer_module._supports_flattened_device_query_lens
)


def _supports_flattened_device_query_lens() -> bool:
    from vllm.platforms import current_platform

    if current_platform.is_ppu():
        # SAIL consumes one expanded query per row, including on SM89.
        # Preserve legacy PPU flattened-query support as well.
        return _ppu_deep_gemm.is_deep_gemm_supported()
    return _upstream_supports_flattened_device_query_lens()


patch(
    _MODULE,
    "_supports_flattened_device_query_lens",
    reason="Use device query lengths for adaptive V4.1 verification with SAIL fixed-length paged logits.",
    affected_versions=_AFFECTED,
    remove_when="The indexer delegates flattened device-query-length support to the platform.",
)(_supports_flattened_device_query_lens)


_upstream_uses_fp4 = _indexer_module.dsa_indexer_uses_fp4


@patch(
    _MODULE,
    "dsa_indexer_uses_fp4",
    reason=(
        "PPU MXFP4 indexer support is supplied by SAIL DeepGEMM, not CUDA "
        "SM100. On sm_89, only V4/V4.1 have MXFP4 producers and default "
        "indexer_kv_dtype to mxfp4."
    ),
    affected_versions=_AFFECTED,
    remove_when="The indexer exposes a platform capability hook.",
)
def dsa_indexer_uses_fp4(vllm_config):
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return _upstream_uses_fp4(vllm_config)
    # PPU MODIFICATION: begin -- only V4/V4.1 have MXFP4 query/cache producers.
    model_config = vllm_config.model_config
    hf_config = model_config.hf_text_config
    supports_fp4 = hf_config.model_type in (
        "deepseek_v4",
        "deepseek_v41",
        "deepseek_v41_text",
    )
    # sm_89 (Zhenwu M890) defaults these models to the MXFP4 indexer
    # cache. It halves the per-token K cache (132B -> 68B) and keeps the whole
    # indexer on the upstream Triton MXFP4 route instead of the INT8 quantising
    # kernels that only exist for PPU 1.0. Upstream's fp8 default is kept on
    # every other capability, where mxfp4 is rejected just below.
    default_kv_dtype = (
        "mxfp4" if supports_fp4 and current_platform.is_device_capability(89) else "fp8"
    )
    # PPU MODIFICATION: end
    kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype(default_kv_dtype)
    if kv_dtype not in _indexer_module.DSA_INDEXER_KV_DTYPES:
        raise ValueError(f"Unsupported PPU indexer_kv_dtype={kv_dtype!r}")
    if kv_dtype == "mxfp4" and not current_platform.is_device_capability(89):
        raise ValueError("PPU indexer_kv_dtype='mxfp4' requires capability sm_89")
    if kv_dtype == "mxfp4" and not supports_fp4:
        raise ValueError("PPU indexer_kv_dtype='mxfp4' requires DeepSeek V4 or V4.1")
    return kv_dtype == "mxfp4"


# Models and the sparse layer can capture the helper before general plugins load.
import sys

for _consumer_name in (
    "vllm.model_executor.models.deepseek_v2",
    "vllm.models.deepseek_v4.attention",
    "vllm.models.deepseek_v41.attention",
    "vllm.model_executor.layers.sparse_mqa_indexer",
):
    _consumer = sys.modules.get(_consumer_name)
    if (
        _consumer is not None
        and getattr(_consumer, "dsa_indexer_uses_fp4", None) is _upstream_uses_fp4
    ):
        patch(
            _consumer_name,
            "dsa_indexer_uses_fp4",
            reason="Preloaded indexer consumers must use SAIL's MXFP4 capability.",
            affected_versions=_AFFECTED,
            remove_when="Consumers resolve the helper through its provider module.",
        )(dsa_indexer_uses_fp4)


_upstream_builder_init = DeepseekV32IndexerMetadataBuilder.__init__


def _select_paged_decode_layout(builder):
    """Use the upstream non-varlen flattening path for legacy SAIL logits."""
    if builder.supports_varlen and not builder.indexer_uses_fp4:
        builder.supports_varlen = False
        builder.use_flattening = True


def _builder_init_wrapper(self, *args, **kwargs) -> None:
    """Delegate to upstream, then add the fork's PPU-only builder state."""
    _upstream_builder_init(self, *args, **kwargs)
    # PPU MODIFICATION: begin -- SAIL DSA metadata needs query geometry,
    # the DeepGEMM SM count and capacity for the selected blocks per CU.
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        return
    _select_paged_decode_layout(self)
    hf_config = self.vllm_config.model_config.hf_text_config
    self.indexer_n_head = getattr(self.kv_cache_spec, "indexer_n_head", 0) or getattr(
        hf_config, "index_n_heads", 0
    )
    self.indexer_q_head_dim = getattr(
        self.kv_cache_spec, "indexer_q_head_dim", 0
    ) or getattr(hf_config, "index_head_dim", 0)
    if (getattr(hf_config, "index_kpool", 1) or 1) > 1:
        # GLM-5-Next zero-pads its queries and weights to at least 32 heads.
        self.indexer_n_head = max(32, self.indexer_n_head)
    from vllm_sail.utils.deep_gemm import get_num_sms

    self.num_sms = get_num_sms()
    # Grouped MXFP4 schedules contain four int32 fields per row plus one
    # row per expanded query. Legacy INT8/FP8 schedules have two fields and
    # no query rows. Reserve the full decode capacity before graph capture;
    # upstream narrows only the row dimension when copying each schedule.
    rows = self.num_sms * _indexer_module.PPU_DEEP_GEMM_TB_PER_CU + 1
    columns = 2
    if self.indexer_uses_fp4:
        rows += self.vllm_config.scheduler_config.max_num_batched_tokens
        columns = 4
    self.scheduler_metadata_buffer = torch.empty(
        (rows, columns),
        dtype=torch.int32,
        device=self.device,
    )
    # PPU MODIFICATION: end


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.__init__",
    reason=(
        "PPU-only builder state from the fork: indexer_n_head/indexer_q_head_dim "
        "read from the KV-cache spec, SM count from PPU DeepGEMM's get_num_sms, "
        "and the enlarged num_sms * PPU_DEEP_GEMM_TB_PER_CU scheduler-metadata "
        "buffer for fp4 paged logits. Delegates to the upstream __init__ so "
        "upstream buffer/refactor drift cannot break this plugin."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream grows PPU branches in the DSA indexer metadata builder.",
)(_builder_init_wrapper)


_upstream_build = DeepseekV32IndexerMetadataBuilder.build
_upstream_prepare_decode = DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors


def _decode_request_ids(context_lens, request_lens, flattened):
    """Recover real request owners before upstream's flattened counts are lost."""
    rows = context_lens.numel()
    row = torch.arange(rows, device=context_lens.device, dtype=torch.int32)
    if flattened:
        ids = torch.searchsorted(request_lens.cumsum(0), row, right=True).to(
            torch.int32
        )
        valid = ids < request_lens.numel()
    else:
        next_n = context_lens.shape[1] if context_lens.ndim == 2 else 1
        ids = torch.arange(
            request_lens.numel(), device=context_lens.device, dtype=torch.int32
        ).repeat_interleave(next_n)
        valid = row % next_n < request_lens.repeat_interleave(next_n)
    return torch.where(valid, ids, -1 - row).contiguous()


@patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors",
    reason=(
        "SAIL MXFP4 prebuilt metadata needs real request ownership. Upstream's "
        "non-varlen MTP preparation flattens queries and replaces decode_lens "
        "with ones without publishing indices, so preserve ownership from the "
        "separate per-request counts before the metadata call."
    ),
    affected_versions=_AFFECTED,
    remove_when="The non-varlen indexer decode preparation publishes request IDs to metadata generation.",
)
def _prepare_decode_wrapper(
    self,
    seq_lens,
    block_table,
    decode_lens,
    decode_lens_cpu,
    query_start_loc,
    num_decodes,
    num_decode_tokens,
    use_native,
    next_n,
    max_decode_len,
):
    result = _upstream_prepare_decode(
        self,
        seq_lens=seq_lens,
        block_table=block_table,
        decode_lens=decode_lens,
        decode_lens_cpu=decode_lens_cpu,
        query_start_loc=query_start_loc,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        use_native=use_native,
        next_n=next_n,
        max_decode_len=max_decode_len,
    )
    from vllm.platforms import current_platform

    if (
        current_platform.is_ppu()
        and self.indexer_uses_fp4
        and _current_builder
        and _current_builder[-1] is self
    ):
        self._sail_decode_request_ids = _decode_request_ids(
            result[0], self.per_req_decode_lens_buffer[:num_decodes], not use_native
        )
    return result


def _build_wrapper(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> DeepseekV32IndexerMetadata:
    """Delegate to upstream with the builder visible to the static helpers.

    Upstream owns the whole step build, including the fp4 paged-MQA-logits
    scheduler metadata: PPU declares ``PlatformEnum.CUDA``, so its decode
    block runs here and the rebound ``get_paged_mqa_logits_metadata`` adds the
    fork's ``metadata_extra`` itself. Always-flattened and varlen decode
    publish stable row buffers; draft refresh updates them and the schedule
    in place.
    """
    self._sail_decode_request_ids = None
    self._sail_paged_schedule_key = None
    _current_builder.append(self)
    try:
        metadata = _upstream_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        if self._sail_paged_schedule_key is not None and metadata.decode is not None:
            # The decode record is slotted upstream. Keep SAIL provenance on
            # the step that owns it, not on a shared tensor or the decode row.
            metadata._sail_paged_schedule_key = self._sail_paged_schedule_key
        return metadata
    finally:
        _current_builder.pop()
        self._sail_decode_request_ids = None


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.build",
    reason=(
        "Keeps the builder on _current_builder while upstream builds the step, "
        "so the rebound paged-MQA-logits metadata helper can read the fork's "
        "fp4 capability. Upstream keeps ownership of the step build "
        "(including TP prefill row sharding), so "
        "upstream drift cannot break this plugin."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream exposes the builder to the paged metadata helper without "
        "module-level context."
    ),
)(_build_wrapper)


_upstream_update_draft_decode = (
    DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata
)


def _update_draft_decode_wrapper(self, metadata) -> None:
    """Delegate to upstream with the builder visible to the static helpers."""
    _current_builder.append(self)
    try:
        return _upstream_update_draft_decode(self, metadata)
    finally:
        _current_builder.pop()


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata",
    reason=(
        "Second PPU-reachable caller of the rebound paged-MQA-logits metadata "
        "helper (the MTP draft refresh). Keeps the builder on "
        "_current_builder so the helper re-derives the fork's fp4 "
        "metadata_extra; otherwise the fp4 scheduler metadata would be "
        "refreshed in the fp8 layout and upstream's shape assertion would fail."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream exposes the builder to the paged metadata helper without "
        "module-level context."
    ),
)(_update_draft_decode_wrapper)
