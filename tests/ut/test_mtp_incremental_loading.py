# SPDX-License-Identifier: Apache-2.0
"""Checkpoint ownership and reload for the DeepSeek indexer used by MTP."""

from types import SimpleNamespace

import pytest

from tests.support.source import function


@pytest.fixture
def install_loader(upstream_source_root):
    def install(
        *,
        spec_layer=lambda config, name: 0,
        resolved_shared=False,
        global_shared=False,
        expert_mapping=lambda *a, **kw: [],
        remap_scale=lambda name, parameters: name,
    ):
        class Target:
            def __init__(self, *, vllm_config):
                self.config = SimpleNamespace(n_routed_experts=0, n_shared_experts=0)
                self.model = SimpleNamespace(mtp_start_layer_idx=0, num_mtp_layers=1)
                self.parameters = vllm_config.parameters
                self.fused_shared_expert_layers = {0} if resolved_shared else set()

            def named_parameters(self):
                return self.parameters.items()

            def _rewrite_spec_layer_name(self, index, name):
                assert index == 0
                return name

        Target.load_weights = function(
            upstream_source_root / "vllm/model_executor/models/deepseek_mtp.py",
            "DeepSeekMTP.load_weights",
            dict(
                get_spec_layer_idx_from_weight_name=spec_layer,
                extract_layer_index=lambda name: int(
                    name.split(".layers.")[1].split(".")[0]
                ),
                rocm_aiter_ops=SimpleNamespace(
                    is_fusion_moe_shared_experts_enabled=lambda: global_shared
                ),
                fused_moe_make_expert_params_mapping=expert_mapping,
                maybe_remap_kv_scale_name=remap_scale,
                default_weight_loader=lambda *a: pytest.fail(
                    "use the checkpoint parameter's own loader"
                ),
            ),
        )
        return Target

    return install


@pytest.mark.upstream_source
@pytest.mark.parametrize("reverse", [False, True])
def test_scale_and_weight_load_into_separate_parameters_across_calls(
    install_loader, reverse
):
    prefix = "model.layers.0.indexer.wk."
    records = []
    weights = {
        prefix + name: SimpleNamespace(
            weight_loader=lambda param, value: records.append((param, value))
        )
        for name in ("weight", "weight_scale_inv")
    }
    model = install_loader()(vllm_config=SimpleNamespace(parameters=weights))
    for name in sorted(weights, reverse=reverse):
        value = object()
        assert model.load_weights([(name, value)]) == {name}
        assert records[-1] == (weights[name], value)


@pytest.mark.parametrize(
    "ppu, scheme, split",
    [
        (True, "channelwise", True),
        (True, None, True),
        (True, "modelopt_fp4", False),
        (False, "channelwise", False),
    ],
)
def test_ppu_projection_ownership_survives_reload(ppu, scheme, split):
    path = "vllm_sail/patch/enhancement/models/deepseek_v2.py"
    owner = SimpleNamespace(head_dim=128, n_head=32)
    cfg = SimpleNamespace(get_name=lambda: scheme) if scheme else None
    native = []

    def linear(hidden, width, **kwargs):
        return SimpleNamespace(width=width, **kwargs)

    scope = dict(
        current_platform=SimpleNamespace(is_ppu=lambda: ppu),
        ReplicatedLinear=linear,
        _upstream_init=lambda *args, **kw: native.append("build"),
        _upstream_finish=lambda layer: native.append("refresh"),
    )
    function(path, "init_projections", scope)(owner, 256, cfg, "indexer")
    finish = function(path, "process_weights_after_loading", scope)
    finish(owner)
    finish(owner)
    assert owner.use_ppu_split_indexer is split
    if split:
        assert native == []
        assert (owner.wk.width, owner.wk.quant_config, owner.wk.prefix) == (
            128,
            cfg,
            "indexer.wk",
        )
        assert (
            owner.weights_proj.width,
            owner.weights_proj.quant_config,
            owner.weights_proj.prefix,
        ) == (32, None, "indexer.weights_proj")
        assert not hasattr(owner, "wk_weights_proj")
    else:
        assert native == ["build", "refresh", "refresh"]


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.upstream_source
def test_tied_top_level_embedding_loads_without_rewriting(install_loader, fp4):
    writes = []
    embedding_name = "model.embed_tokens.weight"
    ordinary_name = "model.layers.0.norm.weight"
    embedding = SimpleNamespace(
        weight_loader=lambda param, value: writes.append((param, value))
    )
    ordinary = SimpleNamespace(weight_loader=lambda param, value: None)
    value = object()
    target = install_loader(
        spec_layer=lambda config, name: 0 if ".layers.0." in name else None,
    )
    model = target(
        vllm_config=SimpleNamespace(
            fp4=fp4, parameters={embedding_name: embedding, ordinary_name: ordinary}
        )
    )
    loaded = model.load_weights(
        [
            (embedding_name, value),
            ("unowned.weight", object()),
            (ordinary_name, object()),
        ]
    )
    assert loaded == {embedding_name, ordinary_name}
    assert writes == [(embedding, value)]


@pytest.mark.parametrize("resolved_shared", [False, True])
@pytest.mark.parametrize("global_shared", [False, True])
@pytest.mark.upstream_source
def test_expert_mapping_uses_the_models_resolved_shared_flag(
    install_loader, resolved_shared, global_shared
):
    observed = []

    def mapping(model, **kwargs):
        observed.append(kwargs["num_experts"])
        return []

    name = "model.layers.0.norm.weight"
    target = install_loader(
        resolved_shared=resolved_shared,
        global_shared=global_shared,
        expert_mapping=mapping,
    )
    model = target(
        vllm_config=SimpleNamespace(
            fp4=False,
            parameters={name: SimpleNamespace(weight_loader=lambda *a: None)},
        )
    )
    model.config.n_routed_experts = 3
    model.config.n_shared_experts = 2
    assert model.load_weights([(name, object())]) == {name}
    assert observed == [5 if resolved_shared else 3]


@pytest.mark.upstream_source
def test_disabled_model_fusion_keeps_shared_projection_whole(install_loader):
    writes = []
    name = "model.layers.0.mlp.shared_experts.gate_up_proj.weight"
    parameter = SimpleNamespace(
        weight_loader=lambda param, value, shard: writes.append((param, value, shard))
    )
    target = install_loader(resolved_shared=False, global_shared=True)
    model = target(vllm_config=SimpleNamespace(fp4=False, parameters={name: parameter}))
    model.config.n_shared_experts = 2
    value = object()
    assert model.load_weights(
        [("model.layers.0.mlp.shared_experts.gate_proj.weight", value)]
    ) == {name}
    assert writes == [(parameter, value, 0)]


@pytest.mark.upstream_source
def test_another_layers_fusion_keeps_this_shared_projection_whole(install_loader):
    writes = []
    name = "model.layers.0.mlp.shared_experts.gate_up_proj.weight"
    parameter = SimpleNamespace(
        weight_loader=lambda param, value, shard: writes.append((param, value, shard))
    )
    target = install_loader(resolved_shared=True, global_shared=True)
    model = target(vllm_config=SimpleNamespace(fp4=False, parameters={name: parameter}))
    model.fused_shared_expert_layers = {1}
    model.config.n_shared_experts = 2
    value = object()
    assert model.load_weights(
        [("model.layers.0.mlp.shared_experts.gate_proj.weight", value)]
    ) == {name}
    assert writes == [(parameter, value, 0)]


@pytest.mark.upstream_source
def test_ignored_kv_scale_preserves_string_name(install_loader):
    classified = []
    ordinary_name = "model.layers.0.norm.weight"
    ignored_name = "model.layers.0.self_attn.k_scale"

    def layer(config, name):
        assert isinstance(name, str)
        classified.append(name)
        return 0

    target = install_loader(
        spec_layer=layer,
        remap_scale=lambda name, parameters: None if name == ignored_name else name,
    )
    model = target(
        vllm_config=SimpleNamespace(
            fp4=False,
            parameters={ordinary_name: SimpleNamespace(weight_loader=lambda *a: None)},
        )
    )
    assert model.load_weights(
        [(ignored_name, object()), (ordinary_name, object())]
    ) == {
        ignored_name,
        ordinary_name,
    }
    assert classified.count(ignored_name) == 1
