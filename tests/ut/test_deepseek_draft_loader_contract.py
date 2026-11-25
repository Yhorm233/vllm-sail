# SPDX-License-Identifier: Apache-2.0
"""Draft loaders delegate sink sharding and fused shared-expert ownership."""

from types import SimpleNamespace as NS

import pytest

from tests.support.source import function


def check_loader(load, kind, shared):
    calls = []
    loaded_weight = tuple(range(16))
    sink_name = "model.layers.0.attn.attn_sink"
    shared_name = "model.layers.0.ffn.experts.shared_l2_weight"

    def load_shared(name, weight):
        assert name.endswith(
            ("shared_experts.w2.weight", "shared_experts.down_proj.weight")
        )
        calls.append(("shared", weight))
        return shared_name

    def load_sink(param, weight):
        assert param is sink
        calls.append(("sink", weight))

    sink = NS(weight_loader=load_sink)
    experts = NS(has_fused_shared_experts=shared, load_shared_expert_weight=load_shared)
    ffn = NS(use_native_mega_moe=True, experts=experts)
    layers = [NS(ffn=ffn)] if kind == "dspark" else {"0": NS(mtp_block=NS(ffn=ffn))}
    model = NS(
        layers=layers, confidence_head=None, mtp_start_layer_idx=0, num_mtp_layers=1
    )
    layer = NS(
        model=model,
        config=NS(n_routed_experts=2, num_attention_heads=16, num_hidden_layers=0),
        pad_shared_expert=False,
        named_parameters=lambda: [(sink_name, sink)],
        _remap_dspark_name=lambda name: name,
        _to_spec_layer_name=lambda name: name,
        _rewrite_spec_layer_name=lambda index, name: name,
    )

    def unexpected_default(*args):
        pytest.fail("the sink's registered weight loader must own its shard")

    namespace = {
        "make_deepseek_v4_expert_params_mapping": lambda count: [],
        "_duplicate_context_wkv_weights": lambda weights, count: weights,
        "get_spec_layer_idx_from_weight_name": lambda config, name: 0,
        "extract_layer_index": lambda name: 0,
        "default_weight_loader": unexpected_default,
        "get_tensor_model_parallel_world_size": lambda: 8,
        "get_tensor_model_parallel_rank": lambda: 3,
        "current_platform": NS(is_ppu=lambda: True),
        "logger": NS(info_once=lambda *args: None),
    }
    checkpoint_name = (
        "model.layers.0.ffn.shared_experts.w2.weight" if shared else sink_name
    )
    result = load(namespace)(layer, [(checkpoint_name, loaded_weight)])
    assert result == {shared_name if shared else sink_name}
    assert calls == [("shared" if shared else "sink", loaded_weight)]
    assert calls[0][1] is loaded_weight
    assert not hasattr(layer, "process_weights_after_loading")


@pytest.mark.parametrize("kind", ["dspark", "mtp"])
@pytest.mark.parametrize("shared", [False, True])
def test_plugin_loader_hands_complete_tensor_to_its_registered_owner(kind, shared):
    path = f"vllm_sail/models/deepseek_v4/{kind}.py"
    name = "load_weights" if kind == "dspark" else "DeepSeekV4MTP.load_weights"
    check_loader(lambda ns: function(path, name, ns), kind, shared)


@pytest.mark.upstream_source
@pytest.mark.parametrize("kind", ["dspark", "mtp"])
@pytest.mark.parametrize("shared", [False, True])
def test_actual_upstream_uses_the_same_tensor_ownership(
    upstream_source_root, kind, shared
):
    path = upstream_source_root / f"vllm/models/deepseek_v4/nvidia/{kind}.py"
    cls = "DSparkDeepseekV4ForCausalLM" if kind == "dspark" else "DeepSeekV4MTP"
    check_loader(lambda ns: function(path, cls + ".load_weights", ns), kind, shared)
