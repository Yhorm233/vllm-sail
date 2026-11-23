# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Install PPU DeepEP preparation through the all-to-all factory seam."""

from __future__ import annotations

import sys

from vllm.model_executor.layers.fused_moe import all2all_utils as _all2all
from vllm.platforms import current_platform

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.model_executor.layers.fused_moe.all2all_utils"
_META = dict(
    reason="The all-to-all factory needs PPU DeepEP quantized dispatch and completion support.",
    affected_versions=">=0.32.0,<0.33.0",
    remove_when="The prepare/finalize factory supports plugin backend registration.",
)


_upstream_factory = _all2all.maybe_make_prepare_finalize
_upstream_roundup = _all2all.maybe_roundup_layer_hidden_size


def _maybe_make_prepare_finalize_body(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    allow_new_interface: bool = False,
    use_monolithic: bool = False,
    all2all_manager: Any | None = None,
    input_dtype: torch.dtype | None = None,
) -> FusedMoEPrepareAndFinalize | None:
    if moe.moe_parallel_config.use_passthrough_all2all:
        # The experts dispatch and combine themselves, in every EP topology.
        return PassThroughPrepareAndFinalize()

    if not moe.moe_parallel_config.use_all2all_kernels:
        if not allow_new_interface:
            return None

        # Opt-in XPU batched path: reorganize tokens into E x T x K locally
        # (no all-to-all) so BatchedTritonExperts (moe_mmk TD) can run.
        if current_platform.is_xpu() and moe.moe_backend == "batched_triton":
            return BatchedPrepareAndFinalize(
                max_num_tokens=moe.max_num_tokens,
                num_local_experts=moe.num_local_experts,
                num_dispatchers=1,
                rank=moe.moe_parallel_config.ep_rank,
            )

        # For DP/TP case, fall back to naive P/F.
        if moe.moe_parallel_config.dp_size > 1:
            logger.info_once(
                "Detected DP deployment with no --enable-expert-parallel. "
                "Falling back to AllGather+ReduceScatter dispatch/combine."
            )
            if all2all_manager is None:
                all2all_manager = get_ep_all2all_manager()
            return make_moe_prepare_and_finalize_naive_dp_ep(
                is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
                num_dispatchers=all2all_manager.world_size,
                use_monolithic=use_monolithic,
            )
        else:
            return make_moe_prepare_and_finalize_no_dp_ep(use_monolithic)

    if all2all_manager is None:
        all2all_manager = get_ep_all2all_manager()

    prepare_finalize: FusedMoEPrepareAndFinalize | None = None

    if moe.use_deepep_ht_kernels:
        assert moe.dp_size == all2all_manager.dp_world_size

        all_to_all_args: dict[str, Any] = dict()
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = DeepEPHTPrepareAndFinalize(
            handle,
            num_dispatchers=all2all_manager.world_size,
            dp_size=all2all_manager.dp_world_size,
            rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
        )

    elif moe.use_deepep_ll_kernels:
        assert quant_config is not None
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_ep_ranks=all2all_manager.world_size,
            num_global_experts=moe.num_experts,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        # PPU MODIFICATION: begin
        if current_platform.is_ppu():
            use_fp8_dispatch = (
                quant_config.quant_dtype == current_platform.fp8_dtype()
                and (
                    quant_config.block_shape is None
                    or quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
                )
            )
            use_int8_dispatch = (
                quant_config.quant_dtype == torch.int8
                and quant_config.block_shape is None
            )
            use_mxfp4_dispatch = quant_config.quant_dtype == "mxfp4"
        else:
            use_fp8_dispatch = (
                quant_config.quant_dtype == current_platform.fp8_dtype()
                and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
            )
            use_int8_dispatch = False
            use_mxfp4_dispatch = False

        from vllm_sail.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
            DeepEPLLPrepareAndFinalize,
        )
        # PPU MODIFICATION: end

        prepare_finalize = DeepEPLLPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            # PPU MODIFICATION: begin
            use_int8_dispatch=use_int8_dispatch,
            use_mxfp4_dispatch=use_mxfp4_dispatch,
            # PPU MODIFICATION: end
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )
    elif moe.use_deepep_v2_kernels:
        assert moe.dp_size == all2all_manager.dp_world_size

        use_fp8_dispatch = (
            quant_config is not None
            and quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.is_block_quantized
        )
        all_to_all_args = dict(
            num_max_tokens_per_rank=moe.max_num_tokens,
            hidden=moe.hidden_dim,
            num_topk=moe.experts_per_token,
            num_experts=moe.num_experts,
            use_fp8_dispatch=use_fp8_dispatch,
        )
        handle = all2all_manager.get_handle(all_to_all_args)
        vllm_config = get_current_vllm_config()
        use_cudagraph = not vllm_config.model_config.enforce_eager

        prepare_finalize = DeepEPV2PrepareAndFinalize(
            buffer=handle,
            num_dispatchers=all2all_manager.world_size,
            dp_size=all2all_manager.dp_world_size,
            rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
            num_experts=moe.num_experts,
            num_topk=moe.experts_per_token,
            use_fp8_dispatch=use_fp8_dispatch,
            use_cudagraph=use_cudagraph,
            sp_size=moe.moe_parallel_config.sp_size,
        )

    elif moe.use_moonep_kernels:
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_topk=moe.experts_per_token,
            num_global_experts=moe.num_experts,
            num_prefetch_slots=MOONEP_DEFAULT_NUM_PREFETCH_SLOTS,
            token_padding=MOONEP_DEFAULT_TOKEN_PADDING,
            num_sms=MOONEP_DEFAULT_NUM_SMS,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # The [E+B] weight layout is picked up from the experts (their
        # process_weights_after_loading hook) once the layer has loaded and
        # converted its weights.
        prepare_finalize = MoonEPPrepareAndFinalize(
            buffer_pool=handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            num_global_experts=moe.num_experts,
        )

    elif moe.use_mori_epv2_kernels:
        num_local_experts = moe.num_experts // all2all_manager.world_size
        # MORI EPv2 routes expert e to rank e // num_local_experts, so every rank
        # must hold the same number of experts.
        if num_local_experts * all2all_manager.world_size != moe.num_experts:
            raise ValueError(
                f"MORI EPv2 needs the {moe.num_experts} experts to split evenly "
                f"over {all2all_manager.world_size} expert-parallel ranks"
            )
        all_to_all_args = dict(
            hidden_dim=moe.hidden_dim,
            max_num_inp_token_per_rank=moe.max_num_tokens,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=moe.experts_per_token,
        )
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = MoriEPv2PrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            num_local_experts=num_local_experts,
            rank_expert_offset=all2all_manager.rank * num_local_experts,
        )

    elif moe.use_mori_epv1_kernels:
        assert quant_config is not None

        # MXFP4 activations are dispatched as packed FP4 with their e8m0
        # block scales, which the experts consume without re-quantizing.
        use_fp4_dispatch = quant_config.use_mxfp4_w4a4
        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = not use_fp4_dispatch and (
            quant_config.is_per_act_token
            or quant_config.is_block_quantized
            or quant_config.is_per_tensor
        )
        if use_fp4_dispatch:
            # One e8m0 scale per 32-element block. token_hidden_size stays
            # the unpacked hidden size: mori sizes its token buffers by the
            # BF16 input type, which also covers the BF16 combine.
            quant_dtype = torch.float4_e2m1fn_x2
            scale_dim = moe.hidden_dim // 32
            scale_type_size = torch.float8_e8m0fnu.itemsize
        elif use_fp8_dispatch:
            # For PTPC (per token per channel) or per-tensor quant,
            # scale dim is 1. For 1x128 quant, scale dim is
            # hidden_dim // 128
            quant_dtype = quant_config.quant_dtype
            scale_dim = (
                1
                if (quant_config.is_per_act_token or quant_config.is_per_tensor)
                else moe.hidden_dim // 128
            )
            scale_type_size = torch.float32.itemsize
        else:
            # Unquantized dispatch (e.g. AITER with defer_input_quant):
            # dispatch raw BF16/FP16 data, no scales needed.
            quant_dtype = moe.in_dtype
            scale_dim = 0
            scale_type_size = 0
        all_to_all_args = dict(
            rank=all2all_manager.rank,
            num_ep_ranks=all2all_manager.world_size,
            quant_dtype=quant_dtype,
            token_hidden_size=moe.hidden_dim,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            input_dtype=moe.in_dtype,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
            num_experts_per_token=moe.experts_per_token,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        prepare_finalize = MoriPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            use_fp4_dispatch=use_fp4_dispatch,
        )

    elif moe.use_fi_nvl_two_sided_kernels:
        assert quant_config is not None
        prepare_finalize = FlashInferNVLinkTwoSidedPrepareAndFinalize(
            max_num_tokens=(
                get_current_vllm_config().scheduler_config.max_num_batched_tokens
            ),
            top_k=moe.experts_per_token,
            num_experts=moe.num_experts,
            hidden_size=moe.hidden_dim,
            num_dispatchers=all2all_manager.world_size,
        )

    elif moe.use_fi_nvl_one_sided_kernels:
        assert quant_config is not None
        max_num_tokens = (
            get_current_vllm_config().scheduler_config.max_num_batched_tokens
        )
        dispatch_layout = flashinfer_one_sided_dispatch_layout(
            moe.hidden_dim, quant_config, input_dtype=input_dtype
        )
        prepare_finalize = FlashInferNVLinkOneSidedPrepareAndFinalize(
            max_num_tokens=max_num_tokens,
            top_k=moe.experts_per_token,
            num_experts=moe.num_experts,
            hidden_size=moe.hidden_dim,
            num_dispatchers=all2all_manager.world_size,
            dispatch_format=dispatch_layout.dispatch_format,
            extra_payload_bytes_per_token=dispatch_layout.extra_payload_bytes_per_token,
        )

    elif moe.use_ag_rs_all2all_kernels and allow_new_interface:
        prepare_finalize = make_moe_prepare_and_finalize_naive_dp_ep(
            use_monolithic=use_monolithic,
            is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
            num_dispatchers=all2all_manager.world_size,
        )

    elif moe.use_nixl_ep_kernels:
        assert quant_config is not None
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_local_experts=moe.num_local_experts,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.block_shape == NIXL_EP_QUANT_BLOCK_SHAPE
        )

        prepare_finalize = NixlEPPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.max_num_ep_ranks,
            expert_capacity=(moe.num_local_experts * all2all_manager.max_num_ep_ranks),
            use_fp8_dispatch=use_fp8_dispatch,
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )

    return prepare_finalize


_ppu_factory = bind_body(_maybe_make_prepare_finalize_body, _all2all)


@patch(_MODULE, "maybe_make_prepare_finalize", **_META)
def maybe_make_prepare_finalize(*args, **kwargs):
    if current_platform.is_ppu():
        return _ppu_factory(*args, **kwargs)
    return _upstream_factory(*args, **kwargs)


@patch(_MODULE, "maybe_roundup_layer_hidden_size", **_META)
def maybe_roundup_layer_hidden_size(hidden_size, act_dtype, moe_parallel_config):
    if current_platform.is_ppu() and moe_parallel_config.use_deepep_ll_kernels:
        from vllm_sail.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
            DeepEPLLPrepareAndFinalize,
        )

        return DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(hidden_size)
    return _upstream_roundup(hidden_size, act_dtype, moe_parallel_config)


_CONSUMERS = {
    "maybe_make_prepare_finalize": [
        "vllm.model_executor.layers.fused_moe.eep_reconfigure",
        "vllm.model_executor.layers.fused_moe.oracle.base",
        "vllm.model_executor.layers.quantization.utils.humming.moe",
        "vllm.model_executor.layers.quantization.modelopt_block_quant",
    ],
    "maybe_roundup_layer_hidden_size": [],
}

for _name, _consumers in _CONSUMERS.items():
    _replacement = getattr(_all2all, _name)
    _original = getattr(_replacement, PATCH_MARKER)[f"{_MODULE}.{_name}"]
    for _consumer in _consumers:
        _module = sys.modules.get(_consumer)
        if _module is not None and getattr(_module, _name, None) is _original:
            patch(_consumer, _name, **_META)(_replacement)
