# ruff: noqa: E402
# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific DSpark draft model for DeepSeek V4."""

from __future__ import annotations

from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as NvidiaDSparkDeepseekV4ForCausalLM,
)

from vllm_sail.models.deepseek_v4.model import _make_deepseek_v4_weights_mapper


class DSparkDeepseekV4ForCausalLM(NvidiaDSparkDeepseekV4ForCausalLM, SupportsQuant):
    """Inject PPU mixed-precision mappings into DSpark's fresh quant config."""

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # DeepseekV4FP8Config only consumes the name-level parts of this mapper
    # while preparing the draft quant config. Those are identical for every
    # expert dtype; the fp4 default matches the target model's class mapping.
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")


# Reuse upstream loader globals and retain its tensor-parallel slicing rules.
from vllm.models.deepseek_v4.nvidia import dspark as _dspark

from vllm_sail.patch.bodies import bind_body


def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

    Non-mtp weights (embed/head/main layers) belong to the target model and
    are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
    """
    first_layer = self.model.layers[0]
    use_native_mega_moe = first_layer.ffn.use_native_mega_moe
    if use_native_mega_moe:
        expert_mapping = make_deepseek_v4_expert_params_mapping(
            self.config.n_routed_experts
        )
    else:
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=first_layer.ffn.n_redundant_experts,
        )
    # PPU MODIFICATION: begin
    # Expert scale suffix: FP4/INT8/INT4 experts use .weight_scale;
    # FP8 experts use .weight_scale_inv.
    expert_dtype = getattr(self.config, "expert_dtype", "fp4")
    if expert_dtype == "fp8":
        expert_scale_suffix = ".weight_scale_inv"
    else:
        expert_scale_suffix = ".weight_scale"
    # PPU MODIFICATION: end

    # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("attn.fused_wqa_wkv", "attn.wq_a", 0),
        ("attn.fused_wqa_wkv", "attn.wkv", 1),
    ]

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    # PPU MODIFICATION: begin
    # 0.31 keeps a cross-layer ``context_wkv_proj`` that the checkpoint does
    # not contain: every draft layer's ``mtp.{i}.attn.wkv.*`` must be
    # duplicated into it (carrying ``shard_id=i``) and loaded separately.
    # Without this the draft model runs on uninitialised context-KV weights
    # and the acceptance rate collapses to ~0.
    loaded_confidence_head = False
    weights = _duplicate_context_wkv_weights(weights, len(self.model.layers))

    # Detect non-expert scale suffix from actual model parameters.
    # INT8/FP8 channelwise quant registers .weight_scale; FP8 block
    # quant registers .weight_scale_inv.
    non_expert_scale_suffix = ".weight_scale_inv"
    for param_key in params_dict:
        if ".experts." not in param_key and (
            param_key.endswith(".weight_scale")
            or param_key.endswith(".weight_scale_inv")
        ):
            non_expert_scale_suffix = (
                ".weight_scale"
                if param_key.endswith(".weight_scale")
                else ".weight_scale_inv"
            )
            break
    # PPU MODIFICATION: end

    for name, loaded_weight in weights:
        mapped = self._remap_dspark_name(name)
        if mapped is None:
            continue
        name = mapped

        # PPU MODIFICATION: begin
        if "confidence_head." in name:
            loaded_confidence_head = True
        # PPU MODIFICATION: end

        # ``.scale`` -> per-method scale suffix.
        if name.endswith(".scale"):
            suffix = (
                expert_scale_suffix
                if _EXPERT_SCALE_RE.search(name)
                # PPU MODIFICATION: begin
                else non_expert_scale_suffix
                # PPU MODIFICATION: end
            )
            name = name.removesuffix(".scale") + suffix
        if ".shared_experts." in name:
            experts = self.model.layers[extract_layer_index(name)].ffn.experts
            if getattr(experts, "has_fused_shared_experts", False):
                loaded_params.add(
                    experts.load_shared_expert_weight(name, loaded_weight)
                )
                continue
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if self.pad_shared_expert and ".shared_experts." in name:
            loaded_weight = DeepseekV4Model._pad_shared_expert_weight(
                self.quant_config, name, loaded_weight
            )

        # ``context_wkv_proj`` is a MergedColumnParallelLinear whose shards are
        # produced by ``_duplicate_context_wkv_weights``; the destination shard
        # rides on the tensor as ``loaded_weight.shard_id``. Falling through to
        # the default loader would overwrite every shard with the last one.
        if name.startswith("model.context_wkv_proj."):
            param = params_dict[name]
            param.weight_loader(param, loaded_weight, loaded_weight.shard_id)
            loaded_params.add(name)
            continue

        # E8M0 expert scales: keep raw exponent bytes.
        if ".experts." in name:
            if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                loaded_weight = loaded_weight.view(torch.uint8)
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
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
        # (main_proj/norm/hc_head/markov_head) load directly — otherwise e.g.
        # "markov_w1" would collide with the "w1" shard rule.
        is_layer_param = name.startswith("model.layers.")
        for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
            if not is_layer_param or weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            param.weight_loader(param, loaded_weight, stacked_shard_id)
            loaded_params.add(name)
            break
        else:
            if name.endswith(".ffn.gate.bias"):
                name = name.replace(
                    ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    # PPU MODIFICATION: begin
    # The checkpoint may not ship confidence-head weights; drop the randomly
    # initialised module so acceptance estimation never consumes garbage.
    # Upstream removed this fallback in #49459 (config-driven
    # enable_confidence_head instead); PPU keeps it because the speculator
    # still gates on ``confidence_head is not None`` and PPU checkpoints may
    # lack the weights.
    if self.model.confidence_head is not None and not loaded_confidence_head:
        self.model.confidence_head = None
    # PPU MODIFICATION: end
    logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
    return loaded_params


DSparkDeepseekV4ForCausalLM.load_weights = bind_body(load_weights, _dspark)
