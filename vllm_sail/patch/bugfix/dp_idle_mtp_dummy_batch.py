# SPDX-License-Identifier: Apache-2.0
"""Keep DP idle-replica dummy batches in the agreed query-length geometry.

With ``data-parallel-size > 1`` every rank must agree on the step's batch
shape.  When a replica has no real requests it still joins the step through
``execute_model(..., dummy_run=True)`` and ``InputBatch.make_dummy`` builds a
padded batch.  0.33.0 splits the padded graph capacity evenly across dummy
requests, so the idle replica's per-request query length no longer matches the
active replicas.  MTP/EP requires that agreement; the divergence fails in
attention metadata construction or the expert all-to-all.

The fix follows the upstream change: active dummy rows must adopt the agreed
``uniform_token_count`` and only consume capacity that is an integer multiple
of it, while unused graph slots keep zero query length.  ``set_dummy_context``
must only offset real rows, and dummy attention metadata must cover the full
padded capacity.

``execute_model`` is not copied.  Its ``dispatch_cg_and_sync_dp`` call already
produces the cross-rank agreement; a wrapper records the agreed token count and
``make_dummy`` reads it.  That keeps this patch small and still uses the
synchronised value rather than this rank's own vote.
"""

from __future__ import annotations

import functools
import sys
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm_sail.patch.utils import patch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

#: Per-thread agreed uniform query length for the step being built.  The v1
#: engine runs one step per thread and the value is overwritten by every
#: dispatch, so it never survives into an unrelated batch.
_UNIFORM_TOK_COUNT = threading.local()

_CONTEXT_CONSUMERS = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.cudagraph_utils",
)


def _set_uniform_token_count(value: int | None) -> None:
    _UNIFORM_TOK_COUNT.value = value


def _get_uniform_token_count() -> int | None:
    return getattr(_UNIFORM_TOK_COUNT, "value", None)


def _install_make_dummy() -> None:
    import numpy as np
    import torch
    from vllm.utils import random_uuid

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        max_query_len: int | None = None,
        is_padding: bool = True,
        uniform_token_count: int | None = None,
    ) -> InputBatch:
        # PPU MODIFICATION: begin
        if uniform_token_count is None:
            uniform_token_count = _get_uniform_token_count()
        # PPU MODIFICATION: end

        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device
        num_reqs_after_padding = num_reqs
        num_tokens_after_padding = num_tokens
        if uniform_token_count is not None:
            # A varlen graph's request capacity is not a new batch partition.
            num_reqs = min(num_reqs, num_tokens // uniform_token_count)
            assert num_reqs > 0
            num_tokens = num_reqs * uniform_token_count

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.intp)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # Active rows follow the uniform query length when specified. Otherwise,
        # spread the tokens evenly across requests. Unused graph slots have
        # zero query length.
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        assert max_query_len is None or base_tokens + (num_extra > 0) <= max_query_len
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs_after_padding]

        query_start_loc_np = np.full(
            num_reqs_after_padding + 1, num_tokens, dtype=np.int32
        )
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens[:num_reqs],
            dim=0,
            out=input_buffers.query_start_loc[1 : num_reqs + 1],
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs_after_padding + 1]

        input_ids = input_buffers.input_ids[:num_tokens_after_padding].zero_()
        positions = input_buffers.positions[:num_tokens_after_padding].zero_()

        input_buffers.is_padding[:num_tokens_after_padding].fill_(is_padding)
        is_padding = input_buffers.is_padding[:num_tokens_after_padding]

        logits_indices = query_start_loc[1 : num_reqs + 1] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Copy so set_dummy_context can add context in place without touching
        # num_scheduled_tokens.
        seq_lens_cpu_upper_bound = torch.from_numpy(
            np.pad(num_scheduled_tokens, (0, num_reqs_after_padding - num_reqs))
        )
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_after_padding,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            has_prefill=False,
            max_seq_len_np=None,
            decode_graph_eligible=True,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=max_query_len,
        )

    patch(
        "vllm.v1.worker.gpu.input_batch",
        "InputBatch.make_dummy",
        reason=(
            "Idle DP replicas build dummy batches; the agreed uniform_token_count "
            "must not be repartitioned by varlen graph capacity."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="upstream vLLM 0.33.x backports the DP dummy-batch geometry fix",
    )(make_dummy)


def _install_set_dummy_context() -> None:
    import numpy as np
    import torch
    from vllm.utils.math_utils import cdiv
    from vllm.v1.worker.gpu import input_batch as _input_batch

    original = _input_batch.set_dummy_context

    def set_dummy_context(
        input_batch: InputBatch,
        block_tables: BlockTables,
        context_len: int,
        num_kv_blocks: int,
        max_model_len: int,
        input_block_tables: Sequence[torch.Tensor] | None = None,
    ) -> None:
        """Give each dummy request context_len of context, for profiling step cost
        and for DCP FULL cudagraph capture."""
        if input_block_tables is None:
            input_block_tables = block_tables.input_block_tables
        if not input_block_tables:
            # Attention-free models have no KV context to fabricate.
            return
        num_reqs = input_batch.num_reqs
        query_len = input_batch.max_query_len or int(
            input_batch.num_scheduled_tokens.max()
        )
        context_len = max(min(context_len, max_model_len - query_len), 0)
        if not context_len:
            return

        # Decode-like shape: each request continues after context_len
        # already-computed tokens.
        input_batch.seq_lens[:num_reqs] += context_len
        input_batch.seq_lens_cpu_upper_bound[:num_reqs] += context_len
        input_batch.num_computed_tokens_np.fill(context_len)
        input_batch.num_computed_prefill_tokens_np.fill(context_len)
        local_pos = np.arange(input_batch.num_tokens, dtype=np.int64) - np.repeat(
            input_batch.query_start_loc_np[:num_reqs],
            input_batch.num_scheduled_tokens,
        )
        input_batch.positions[: input_batch.num_tokens].copy_(
            torch.from_numpy(local_pos + context_len)
        )

        seq_len = context_len + query_len
        for block_table, block_size in zip(
            input_block_tables, block_tables.block_sizes, strict=False
        ):
            num_blocks = min(cdiv(seq_len, block_size), block_table.shape[1])
            # Spans are disjoint until the pool runs out, then they wrap and
            # share blocks: profiling only needs reads to be realistic.
            block_ids = (
                torch.arange(
                    num_reqs * num_blocks,
                    dtype=block_table.dtype,
                    device=block_table.device,
                )
                % num_kv_blocks
            )
            block_table[:num_reqs, :num_blocks] = block_ids.view(num_reqs, num_blocks)

    metadata = dict(
        reason=(
            "Profiling context belongs only to real dummy rows, never graph "
            "padding rows."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="upstream vLLM 0.33.x backports the DP dummy-batch geometry fix",
    )
    patch("vllm.v1.worker.gpu.input_batch", "set_dummy_context", **metadata)(
        set_dummy_context
    )
    # These modules import the function by value before plugin installation.
    # Rebind only aliases still pointing at the original, without importing
    # unloaded consumers or overwriting an independent replacement.
    for name in _CONTEXT_CONSUMERS:
        consumer = sys.modules.get(name)
        if (
            consumer is not None
            and getattr(consumer, "set_dummy_context", None) is original
        ):
            patch(name, "set_dummy_context", **metadata)(set_dummy_context)


def _install_prepare_dummy_attn() -> None:
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner.prepare_dummy_attn

    @functools.wraps(original)
    def prepare_dummy_attn(
        self, input_batch: InputBatch, valid_state_slots: bool = False
    ):
        block_table_provider = self.pcp_manager or self.block_tables
        block_tables = block_table_provider.get_dummy_block_tables(
            input_batch.num_reqs_after_padding
        )
        if valid_state_slots:
            state_slots = torch.arange(
                1, input_batch.num_reqs + 1, dtype=torch.int32, device=self.device
            )
            for block_table in block_tables:
                block_table[: input_batch.num_reqs, 0].copy_(state_slots)
        slot_mappings = block_table_provider.get_dummy_slot_mappings(
            input_batch.num_tokens_after_padding
        )
        return block_tables, slot_mappings

    patch(
        "vllm.v1.worker.gpu.model_runner",
        "GPUModelRunner.prepare_dummy_attn",
        reason=(
            "Dummy attention metadata must cover the full padded capacity, while "
            "state slots only belong to active rows."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="upstream vLLM 0.33.x backports the DP dummy-batch geometry fix",
    )(prepare_dummy_attn)


def _install_dispatch_wrapper() -> None:
    from vllm.v1.worker.gpu import model_runner as _model_runner

    original = _model_runner.dispatch_cg_and_sync_dp

    @functools.wraps(original)
    def dispatch_cg_and_sync_dp(*args, **kwargs):
        batch_desc, sync = original(*args, **kwargs)
        # Only record the agreed value when cross-rank coordination actually
        # happened (dp_size > 1).  For dp_size == 1 the returned sync is None
        # and uniform_token_count is merely this rank's own vote; using it
        # would incorrectly repartition non-DP dummy batches and mismatch the
        # model's expected batch geometry.
        if sync is not None:
            _set_uniform_token_count(sync.uniform_token_count)
        else:
            _set_uniform_token_count(None)
        return batch_desc, sync

    patch(
        "vllm.v1.worker.gpu.model_runner",
        "dispatch_cg_and_sync_dp",
        reason=(
            "0.33.0 execute_model does not pass the DP-agreed "
            "uniform_token_count to InputBatch.make_dummy; record it from the "
            "DP sync for dummy runs, but only when dp_size > 1."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="upstream vLLM 0.33.x backports the DP dummy-batch geometry fix",
    )(dispatch_cg_and_sync_dp)


def _install_execute_model_scope() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner.execute_model

    @functools.wraps(original)
    def execute_model(self, *args, **kwargs):
        # The agreed token count is only valid for the dummy batch this call
        # builds. Later, unrelated ``make_dummy(num_reqs, num_reqs, ...)`` calls
        # (e.g. the profile sampler's batch) must not inherit it, or the sampler
        # sees fewer requests than logits rows.
        _set_uniform_token_count(None)
        try:
            return original(self, *args, **kwargs)
        finally:
            _set_uniform_token_count(None)

    patch(
        "vllm.v1.worker.gpu.model_runner",
        "GPUModelRunner.execute_model",
        reason=(
            "Scope the DP-agreed uniform_token_count to the execute_model dummy "
            "batch; it must not leak into the profile sampler's own dummy batch."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="upstream vLLM 0.33.x backports the DP dummy-batch geometry fix",
    )(execute_model)


def install() -> None:
    _install_make_dummy()
    _install_set_dummy_context()
    _install_prepare_dummy_attn()
    _install_dispatch_wrapper()
    _install_execute_model_scope()
