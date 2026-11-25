# SPDX-License-Identifier: Apache-2.0
"""Warm the V4 query path that short-context eager forwards omit."""

from contextvars import ContextVar

import torch
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

from vllm_sail.patch.utils import PATCH_MARKER, patch

_GRAPH_WARMUP = ContextVar("ppu_dsv4_graph_warmup", default=False)
_MANAGER = "vllm.v1.worker.gpu.cudagraph_utils"
_MANAGER_TARGET = f"{_MANAGER}.CudaGraphManager.capture"
_INDEXER = "vllm.models.deepseek_v4.attention"
_INDEXER_TARGET = f"{_INDEXER}.DeepseekV4Indexer.forward"
_REMOVE_WHEN = (
    "Upstream graph warmup exercises the V4 indexer query projection and "
    "quantization even when all short-context candidates are selected."
)


@patch(
    _MANAGER,
    "CudaGraphManager.capture",
    reason="V2 runner graph warmup must initialize the omitted V4 query path too.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when=_REMOVE_WHEN,
)
def capture(self, *args, **kwargs):
    original = getattr(capture, PATCH_MARKER)[_MANAGER_TARGET]
    return _run_graph_warmup(original, self, args, kwargs)


def _run_graph_warmup(original, self, args, kwargs):
    if not current_platform.is_ppu():
        return original(self, *args, **kwargs)
    token = _GRAPH_WARMUP.set(True)
    try:
        return original(self, *args, **kwargs)
    finally:
        _GRAPH_WARMUP.reset(token)


@patch(
    _INDEXER,
    "DeepseekV4Indexer.forward",
    reason=(
        "Short-context eager warmup skips the indexer Q projection, but capture "
        "runs it. DeepGEMM then loads a new shape inside capture and raises "
        "HGGC_ERROR_STREAM_CAPTURE_UNSUPPORTED. Warm that omitted path first."
    ),
    affected_versions=">=0.32.0,<0.33.0",
    remove_when=_REMOVE_WHEN,
)
def indexer_forward(
    self,
    hidden_states,
    qr,
    compressed_kv_score,
    indexer_weights,
    positions,
    rotary_emb,
    qr_scale=None,
    skip_compressor=False,
):
    if (
        _GRAPH_WARMUP.get()
        and current_platform.is_ppu()
        and not torch.cuda.is_current_stream_capturing()
    ):
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            indexer_metadata = metadata[self.k_cache.prefix]
            if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
                q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_head, self.head_dim)
                self._q_rope_quant(positions, q, rotary_emb, indexer_weights)
    original = getattr(indexer_forward, PATCH_MARKER)[_INDEXER_TARGET]
    return original(
        self, hidden_states, qr, compressed_kv_score, indexer_weights,
        positions, rotary_emb, qr_scale, skip_compressor,
    )
