# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAIL DeepGEMM MegaMoE expert layouts and dispatch."""

from __future__ import annotations

import torch
import vllm.envs as envs
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    DeepGemmMegaMoEExperts,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

import vllm_sail.envs as sail_envs
from vllm_sail.utils.deep_gemm_mega import (
    MEGA_MOE_FORMATS,
    check_mega_moe_supported,
    mega_moe_api,
)


class PPUDeepGemmMegaMoEExperts(DeepGemmMegaMoEExperts):
    """Use the selected PPU MegaMoE kernel with native vLLM routing."""

    _symm_buffers: dict[tuple, object] = {}

    def __init__(
        self,
        vllm_config,
        *,
        num_experts,
        num_local_experts,
        experts_start_idx,
        top_k,
        hidden_size,
        intermediate_size,
        activation,
        weight_format,
        shared_experts=None,
        num_logical_experts=None,
        fast_math=True,
        max_num_tokens=None,
        prefix="",
        l2_input_scale=None,
    ):
        if weight_format != MEGA_MOE_MXFP4:
            raise ValueError("SAIL MegaMoE requires MXFP4 weights.")
        self.mma_type = sail_envs.VLLM_SAIL_MEGA_MOE_MMA_TYPE
        self.mma_format = MEGA_MOE_FORMATS[self.mma_type]
        if activation.name not in self.mma_format.activations:
            raise ValueError(
                f"SAIL MegaMoE {self.mma_type} requires "
                f"{' or '.join(self.mma_format.activations)} activation, "
                f"got {activation.name}."
            )
        max_local_experts = 127 if self.mma_type == "bf16xfp4" else 128
        if num_local_experts > max_local_experts:
            raise ValueError(
                f"SAIL MegaMoE {self.mma_type} supports at most "
                f"{max_local_experts} local experts per EP rank."
            )
        self.l2_input_scale = l2_input_scale
        self.weight_format = weight_format
        self.activation = activation
        self.fast_math = fast_math
        torch.nn.Module.__init__(self)
        self.prefix = prefix
        self.capture_fn = None
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_shared_experts = 0
        self.max_num_tokens = (
            max_num_tokens
            if max_num_tokens is not None
            else vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )
        self.eplb_state = EplbLayerState()
        self._transformed = False
        self._l1_inv_scale = None
        self._l2_inv_scale = None
        self._register_weight(
            "w13", (num_local_experts, 2 * intermediate_size, hidden_size), None
        )
        self._register_weight(
            "w2", (num_local_experts, hidden_size, intermediate_size), None
        )
        for name, param in self.named_parameters():
            set_weight_attrs(param, {"weight_loader": self.weight_loader})
            if name.endswith("_scale"):
                param.quant_method = "block"
        self._check_runtime_supported()
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def l1_weights(self):
        return self._kernel_weights("w13")

    @property
    def l2_weights(self):
        return self._kernel_weights("w2")

    def _kernel_weights(self, name):
        weight = getattr(self, f"{name}_weight")
        scale = getattr(self, f"{name}_weight_scale", None)
        weights = (weight, scale)
        if self.mma_type == "fp8xfp4":
            # W4A8 kernel expects each bank's inverse scale, returned by the
            # weight transform.
            weights += (self._l1_inv_scale if name == "w13" else self._l2_inv_scale,)
        return weights

    def _check_runtime_supported(self):
        check_mega_moe_supported(
            self.w13_weight.device,
            self.hidden_size,
            self.intermediate_size,
            self.mma_type,
        )

    def _transform_weights(self, api):
        """Transform the raw banks in place and drop the raw storages.

        The transformed banks are the only consumer once loading is done and
        EPLB is disabled; keeping both copies doubles the expert memory
        footprint. EPLB rebalancing writes into the registered raw banks, so
        the raw layout must stay resident and each step re-transforms.
        """
        transform_name = self.mma_format.transform
        if (
            transform_name is None
            or self._transformed
            or self.eplb_state.logical_to_physical_map is not None
        ):
            return
        transform = getattr(api, transform_name)
        # Raw (weight, scale) pairs: l1_weights would append inv_scale, which
        # is only produced by this transform.
        args = (
            (self.w13_weight, self.w13_weight_scale),
            (self.w2_weight, self.w2_weight_scale),
        )
        if self.mma_type == "bf16xfp4":
            args += (self.activation.name,)
        transformed_l1, transformed_l2 = transform(*args)
        if self.mma_type == "fp8xfp4":
            (
                self.w13_weight.data,
                self.w13_weight_scale.data,
                self._l1_inv_scale,
            ) = transformed_l1
            (
                self.w2_weight.data,
                self.w2_weight_scale.data,
                self._l2_inv_scale,
            ) = transformed_l2
        else:
            self.w13_weight.data, self.w13_weight_scale.data = transformed_l1
            self.w2_weight.data, self.w2_weight_scale.data = transformed_l2
        self._transformed = True

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        expected = "w2_" if shard_id == "w2" else "w13_"
        local_ids = (
            self._map_global_expert_id(expert_id) if expected in weight_name else []
        )
        for local_id in local_ids:
            self._load_weight(
                param.data[local_id], loaded_weight, shard_id, "scale" in weight_name
            )
        return bool(local_ids) if return_success else None

    weight_loader.supports_moe_loading = True

    def _register_weight(self, name, shape, spec):
        experts, n, k = shape
        self.register_parameter(
            f"{name}_weight",
            torch.nn.Parameter(
                torch.zeros(experts, n, k // 2, dtype=torch.uint8), requires_grad=False
            ),
        )
        if self.mma_format.transform is None:
            scales = torch.zeros(experts, k // 64, n, dtype=torch.uint16).transpose(
                1, 2
            )
        else:
            scales = torch.zeros(experts, n, k // 32, dtype=torch.uint8)
        self.register_parameter(
            f"{name}_weight_scale", torch.nn.Parameter(scales, requires_grad=False)
        )

    def _load_weight(self, dst, src, shard_id, is_scale):
        if self._transformed:
            # The first forward overwrote the raw banks with the kernel layout,
            # so a checkpoint shard can no longer be decoded into them.
            raise RuntimeError(
                "MegaMoE expert weights cannot be reloaded after the first "
                "forward: the raw banks were transformed in place."
            )
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        if not is_scale:
            src = src.view(torch.uint8)
            rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
            if src.shape != (rows, dst.shape[1]):
                raise ValueError(f"Invalid {shard_id} packed weight shape: {src.shape}")
        else:
            if src.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise ValueError(
                    "MegaMoE requires raw UE8M0 scales, not NVFP4 E4M3 scales."
                )
            rows, k = (
                (self.hidden_size, self.intermediate_size)
                if shard_id == "w2"
                else (self.intermediate_size, self.hidden_size)
            )
            if src.shape != (rows, k // 32):
                raise ValueError(f"Invalid {shard_id} UE8M0 scale shape: {src.shape}")
            src = src.view(torch.uint8)
            if self.mma_format.transform is None:
                src = src.contiguous().view(torch.uint16)
        if shard_id != "w2":
            if self.mma_format.transform is None:
                dst = dst[0::2] if shard_id == "w1" else dst[1::2]
            else:
                rows = dst.shape[0] // 2
                dst = dst[:rows] if shard_id == "w1" else dst[rows:]
        dst.copy_(src)

    def _kernel_kwargs(self):
        activation = self.activation
        kwargs = {"activation": activation.name, "fast_math": self.fast_math}
        if activation.name == "situ":
            kwargs["situ_beta"] = activation.alpha
            kwargs["situ_linear_beta"] = activation.beta or None
        else:
            kwargs["activation_clamp"] = activation.clamp
        if self.mma_format.transform is None:
            kwargs["recipe"] = (1, 1, 32)
        return kwargs

    def get_symm_buffer(self):
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.cuda.current_device(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.mma_type,
            self.activation.name,
        )
        if key not in self._symm_buffers:
            self._symm_buffers[key] = mega_moe_api(
                self.mma_type
            ).get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type=self.mma_type,
                activation=self.activation.name,
            )
        return self._symm_buffers[key]

    def forward(self, hidden_states, topk_weights, topk_ids, shared_experts=None):
        tokens = hidden_states.shape[0]
        if tokens > self.max_num_tokens:
            raise ValueError(
                f"MegaMoE token count {tokens} exceeds capacity {self.max_num_tokens}."
            )
        self._synchronize_first_launch()
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            padding = get_forward_context().is_padding
            if padding is not None:
                topk_ids = torch.where(padding[:tokens, None], -1, topk_ids)
        state = self.eplb_state
        if state.logical_to_physical_map is not None:
            if self.mma_type == "fp8xfp4":
                raise NotImplementedError(
                    "SAIL W4A8 MegaMoE does not support EPLB because rebalancing "
                    "can change the inverse weight scale captured by the kernel."
                )
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=state.expert_load_view,
                logical_to_physical_map=state.logical_to_physical_map,
                logical_replica_count=state.logical_replica_count,
                record_enabled=state.should_record_tensor,
                num_unpadded_tokens=(
                    state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
                    if state.num_unpadded_tokens_tensors is not None
                    else None
                ),
            )
        api = mega_moe_api(self.mma_type)
        self._transform_weights(api)
        l1, l2 = self.l1_weights, self.l2_weights
        if (
            self.eplb_state.logical_to_physical_map is not None
            and self.mma_format.transform is not None
        ):
            # EPLB writes into the registered raw banks, so re-transform each
            # step; without EPLB the banks were transformed in place at load.
            l1, l2 = api.transform_weights_for_w4a16_mega_moe(
                l1, l2, self.activation.name
            )
        buffer = self.get_symm_buffer()
        x = hidden_states.contiguous()
        api.mega_moe_pre_dispatch(
            x,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            buf_x=buffer.x,
            buf_x_sf=buffer.x_sf,
            buf_topk_idx=buffer.topk_idx,
            buf_topk_weights=buffer.topk_weights,
            num_tokens=tokens,
            mma_type=self.mma_type,
        )
        output = torch.empty_like(x, dtype=torch.bfloat16)
        # Every EP rank participates, including ranks with zero local tokens.
        kernel = getattr(api, self.mma_format.kernel)
        kernel(output, l1, l2, buffer, **self._kernel_kwargs())
        if shared_experts is not None:
            output += shared_experts(hidden_states)
        return output
