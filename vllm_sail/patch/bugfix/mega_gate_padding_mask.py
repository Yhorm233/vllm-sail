# SPDX-License-Identifier: Apache-2.0
"""Pass the padding mask to the MegaGate so NaN padding rows cannot trap it.

DSpark's padded drafter batches leave padding rows of the hidden-state buffer
uninitialized; they routinely hold NaN bit patterns. The upstream
``DeepseekV4MoE.forward`` calls ``bf16_mega_gate`` without a token mask, so the
DeepGEMM gate kernel validates every row's GEMM scores
(``device_check(gate_is_valid)``) and executes ``s.trap`` on the NaN padding
scores. The trap surfaces as a sticky ``unspecified launch failure`` (719) at
the next MoE launch/sync, which misdirected the original diagnosis toward the
INT8 MegaMoE kernel.

The gate API already supports this exact case: ``mask[t]=False`` skips the
score validation and emits ``topk_idx=-1, topk_weights=0`` for the row --
the same contract ``VLLM_MOE_SKIP_PADDING`` applies on the expert side
(``topk_ids`` of padding rows become -1 before dispatch). The upstream
``vllm.utils.deep_gemm.bf16_mega_gate`` wrapper simply never exposes ``mask``.

The replacement below rebinds the model module's by-value ``bf16_mega_gate``
import. It derives ``mask = ~is_padding`` from the forward context and calls
the DeepGEMM implementation directly (the upstream wrapper cannot forward a
mask). Without padding information it delegates to the original wrapper, so
non-padded batches keep the upstream path byte-for-byte.

Verified on ZW-810E: SPEC=1 + GSM8K load crashed within ~8 minutes before the
fix (gate trap, NaN rows exactly matching ``is_padding``); after the fix the
full 1319-question eval completes at 97.2% accuracy with zero traps.
"""

from __future__ import annotations

import torch

from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.models.deepseek_v4.nvidia.model"
_TARGET = "bf16_mega_gate"
_AFFECTED = ">=0.32.0,<0.34.0"
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    @patch(
        _MODULE,
        _TARGET,
        reason=(
            "Padded drafter batches carry NaN hidden-state rows; without a "
            "token mask the MegaGate kernel traps on their scores and the "
            "sticky launch failure kills the engine under SPEC=1 load."
        ),
        affected_versions=_AFFECTED,
        remove_when=(
            "vLLM's bf16_mega_gate wrapper accepts a mask and the DeepSeek V4 "
            "MoE forward passes the forward-context padding mask to it."
        ),
    )
    def _bf16_mega_gate_padding_masked(
        x: torch.Tensor,
        weight: torch.Tensor,
        num_topk: int,
        *,
        scoring_func: str,
        routed_scaling_factor: float,
        ep_rank: int,
        bias: torch.Tensor | None = None,
        image_bias: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        fix_routing_mask: torch.Tensor | None = None,
        unmapped_topk_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.utils import deep_gemm as dg_utils

        mask = None
        if is_forward_context_available():
            padding = get_forward_context().is_padding
            if padding is not None:
                mask = ~padding[: x.shape[0]]
        if mask is None:
            original = getattr(_bf16_mega_gate_padding_masked, PATCH_MARKER)[
                f"{_MODULE}.{_TARGET}"
            ]
            return original(
                x,
                weight,
                num_topk,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                ep_rank=ep_rank,
                bias=bias,
                image_bias=image_bias,
                image_token_mask=image_token_mask,
                fix_routing_mask=fix_routing_mask,
                unmapped_topk_idx=unmapped_topk_idx,
            )
        # Mirror the upstream wrapper's output contract (weights, indices)
        # while forwarding the mask the wrapper cannot express.
        dg_utils._lazy_init()
        if dg_utils._bf16_mega_gate_impl is None:
            raise RuntimeError(
                "DeepGEMM MegaMoE requires a DeepGEMM build with bf16_mega_gate."
            )
        topk_idx = torch.empty(
            (x.shape[0], num_topk), dtype=torch.int64, device=x.device
        )
        topk_weights = torch.empty(
            (x.shape[0], num_topk), dtype=torch.float32, device=x.device
        )
        dg_utils._bf16_mega_gate_impl(
            x,
            weight,
            num_topk,
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=routed_scaling_factor,
            ep_rank=ep_rank,
            scoring_func=scoring_func,
            bias=bias,
            image_bias=image_bias,
            image_token_mask=image_token_mask,
            fix_routing_mask=fix_routing_mask,
            unmapped_topk_idx=unmapped_topk_idx,
            mask=mask,
            out=(topk_idx, topk_weights),
        )
        return topk_weights, topk_idx

    _installed = True