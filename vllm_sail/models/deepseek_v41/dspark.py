# SPDX-License-Identifier: Apache-2.0
"""Provide quantization mapping and destination-aware scale loading for V4.1 DSpark."""

from __future__ import annotations

from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.models.deepseek_v41.nvidia import dspark as _dspark
from vllm.models.deepseek_v41.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as UpstreamDSparkV41,
)
from vllm.models.deepseek_v41.nvidia.model import _make_deepseek_v4_weights_mapper

from vllm_sail.patch.bodies import bind_body


class DSparkDeepseekV41ForCausalLM(UpstreamDSparkV41, SupportsQuant):
    """Preserve V4.1 draft layers, weight loader and target-state sharing."""

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # SupportsQuant uses this mapper only to match quantization config patterns.
    # The loader preserves V4.1 expert counts and TP slicing, adapting only scale names.
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4", "weight_scale")

    @staticmethod
    def _resolve_scale_name(name, params_dict):
        """Resolve scale suffixes against target parameters for INT8 and mixed precision."""
        if name not in params_dict:
            for suffix, alternate in (
                (".weight_scale_inv", ".weight_scale"),
                (".weight_scale", ".weight_scale_inv"),
                ("_weight_scale_inv", "_weight_scale"),
                ("_weight_scale", "_weight_scale_inv"),
            ):
                if name.endswith(suffix):
                    candidate = name.removesuffix(suffix) + alternate
                    if candidate in params_dict:
                        return candidate
        return name


# Preserve V4.1 draft expert counts, name mapping and TP slicing without the V4 loader.
# This function retains the upstream vLLM 0.31 body and globals.
# ruff: noqa: F821
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

    Non-mtp weights (embed/head/main layers) belong to the target model and
    are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
    """
    first_layer = self.model.layers[0]
    use_mega_moe = first_layer.ffn.use_mega_moe
    # Draft MoE layers use the dspark_* expert counts, not the
    # backbone's (see DeepseekV4MoE and the reference
    # ModelArgs.get_moe_config).
    n_draft_experts = (
        getattr(self.config, "dspark_n_routed_experts", 0)
        or self.config.n_routed_experts
    )
    if use_mega_moe:
        expert_mapping = make_deepseek_v4_expert_params_mapping(n_draft_experts)
    else:
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=n_draft_experts,
        )
    expert_scale_suffix = (
        ".weight_scale"
        if getattr(self.config, "expert_dtype", "fp4") == "fp4"
        else ".weight_scale_inv"
    )

    # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("attn.fused_wqa_wkv", "attn.wq_a", 0),
        ("attn.fused_wqa_wkv", "attn.wkv", 1),
    ]

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    loaded_confidence_head = False

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    n_local_head = self.config.num_attention_heads // tp_size
    head_start = n_local_head * tp_rank
    head_end = n_local_head * (tp_rank + 1)

    for name, loaded_weight in weights:
        mapped = self._remap_dspark_name(name)
        if mapped is None:
            continue
        name = mapped
        if "confidence_head." in name:
            loaded_confidence_head = True

        # ``.scale`` -> per-method scale suffix.
        if name.endswith(".scale"):
            suffix = (
                expert_scale_suffix
                if _EXPERT_SCALE_RE.search(name)
                else f".{self.linear_scale_name}"
            )
            name = name.removesuffix(".scale") + suffix
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if self.pad_shared_expert and ".shared_experts." in name:
            loaded_weight = DeepseekV4Model._pad_shared_expert_weight(
                self.quant_config, name, loaded_weight
            )

        # E8M0 expert scales: keep raw exponent bytes.
        if ".experts." in name:
            if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                loaded_weight = loaded_weight.view(torch.uint8)
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # PPU MODIFICATION: begin
                # INT8 fused MoE registers weight_scale; resolve the mapped destination.
                name_mapped = self._resolve_scale_name(name_mapped, params_dict)
                # PPU MODIFICATION: end
                param = params_dict[name_mapped]
                success = param.weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                    break
            continue

        # Stacked rules only apply to decoder-layer weights. Head-stack params
        # (main_proj/norm/markov_head/confidence_head) load directly —
        # otherwise e.g. "markov_w1" would collide with the "w1" shard rule.
        is_layer_param = name.startswith("model.layers.")
        for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
            if not is_layer_param or weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # PPU MODIFICATION: begin
            # The quantization method registers fused attention/shared-expert scales.
            name = self._resolve_scale_name(name, params_dict)
            # PPU MODIFICATION: end
            param = params_dict[name]
            param.weight_loader(param, loaded_weight, stacked_shard_id)
            loaded_params.add(name)
            break
        else:
            if "attn_sink" in name:
                narrow = loaded_weight[head_start:head_end]
                params_dict[name][: narrow.shape[0]].copy_(narrow)
                loaded_params.add(name)
                continue
            if name.endswith(".ffn.gate.bias"):
                name = name.replace(
                    ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                )
            # PPU MODIFICATION: begin
            # Standalone layers include main_proj, attention and shared-expert down_proj.
            name = self._resolve_scale_name(name, params_dict)
            # PPU MODIFICATION: end
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    if self.model.confidence_head is not None and not loaded_confidence_head:
        self.model.confidence_head = None
    self.process_weights_after_loading()
    logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
    return loaded_params


DSparkDeepseekV41ForCausalLM.load_weights = bind_body(load_weights, _dspark)
