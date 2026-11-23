# SPDX-License-Identifier: Apache-2.0
"""DeepEP prepare/finalize integration accepts upstream keyword arguments."""

from __future__ import annotations

import ast
import re
from enum import Enum
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import (
    ROOT,
    assert_accepts_upstream_keywords,
    definition,
    function,
)


@pytest.mark.upstream_source
def test_prepare_finalize_consumer_inventory_matches_upstream(upstream_source_root):
    tree = ast.parse((ROOT / "vllm_sail/patch/enhancement/deepep.py").read_text())
    inventory = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_CONSUMERS"
            for target in node.targets
        )
    )
    discovered = {name: set() for name in inventory}
    source = upstream_source_root / "vllm"
    for path in (source / "model_executor").rglob("*.py"):
        module = ".".join(("vllm", *path.relative_to(source).with_suffix("").parts))
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "vllm.model_executor.layers.fused_moe.all2all_utils"
            ):
                for alias in node.names:
                    if alias.name in discovered:
                        assert alias.asname is None, "renamed alias needs rebinding"
                        discovered[alias.name].add(module)
            pending.extend(ast.iter_child_nodes(node))
    assert discovered == {name: set(consumers) for name, consumers in inventory.items()}


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "deepep",
            "_maybe_make_prepare_finalize_body",
            "model_executor/layers/fused_moe/all2all_utils",
            "maybe_make_prepare_finalize",
        )
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )


@pytest.mark.upstream_source
def test_complete_factory_body_diff_is_limited_to_ppu_dispatch(upstream_source_root):
    """比较完整复制体，只还原已记录的 PPU LL dispatch 差异。"""
    path = ROOT / "vllm_sail/patch/enhancement/deepep.py"
    node = definition(path, "_maybe_make_prepare_finalize_body")
    source = "\n".join(path.read_text().splitlines()[node.lineno - 1 : node.end_lineno])
    regions = list(
        re.finditer(
            r"(?m)^ *# PPU MODIFICATION: begin\n.*?^ *# PPU MODIFICATION: end\n?",
            source,
            flags=re.DOTALL,
        )
    )
    assert len(regions) == 2
    original_dispatch = (
        "        use_fp8_dispatch = (\n"
        "            quant_config.quant_dtype == current_platform.fp8_dtype()\n"
        "            and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE\n"
        "        )\n"
    )
    for region, replacement in reversed(
        list(zip(regions, [original_dispatch, ""], strict=False))
    ):
        source = source[: region.start()] + replacement + source[region.end() :]
    local = ast.parse(source).body[0]
    upstream = definition(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/all2all_utils.py",
        "maybe_make_prepare_finalize",
    )
    local.name = upstream.name
    assert ast.dump(local) == ast.dump(upstream)


@pytest.mark.upstream_source
@pytest.mark.parametrize("upstream", [False, True])
@pytest.mark.parametrize("experts", [16, 17])
def test_mori_epv2_preserves_expert_partition_and_rank_offset(
    upstream_source_root, upstream, experts
):
    """复制体须采用新 EPv2 路由，并在分区不整除时拒绝创建 handle。"""
    calls = []
    handle = object()
    manager = NS(world_size=4, rank=2)

    def get_handle(args):
        calls.append(args)
        return handle

    manager.get_handle = get_handle
    factory = function(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/all2all_utils.py"
        if upstream
        else "vllm_sail/patch/enhancement/deepep.py",
        "maybe_make_prepare_finalize"
        if upstream
        else "_maybe_make_prepare_finalize_body",
        {"MoriEPv2PrepareAndFinalize": lambda *args, **kwargs: NS(args=args, **kwargs)},
    )
    moe = NS(
        moe_parallel_config=NS(use_passthrough_all2all=False, use_all2all_kernels=True),
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_deepep_v2_kernels=False,
        use_moonep_kernels=False,
        use_mori_epv2_kernels=True,
        num_experts=experts,
        hidden_dim=3584,
        max_num_tokens=32,
        experts_per_token=8,
    )
    if experts == 17:
        with pytest.raises(ValueError, match="split evenly"):
            factory(moe, None, all2all_manager=manager)
        assert calls == []
        return
    result = factory(moe, None, all2all_manager=manager)
    assert calls == [
        dict(
            hidden_dim=3584,
            max_num_inp_token_per_rank=32,
            num_experts_per_rank=4,
            num_experts_per_token=8,
        )
    ]
    assert result.args == (handle,)
    assert result.max_tokens_per_rank == 32
    assert result.num_dispatchers == 4
    assert result.num_local_experts == 4
    assert result.rank_expert_offset == 8


@pytest.mark.upstream_source
@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("max_tokens,dispatchers", [(64, 1), (512, 8)])
def test_two_sided_factory_uses_real_constructor_dimensions(
    upstream_source_root, copied, max_tokens, dispatchers
):
    """真实 factory 和完整 constructor；通信 manager 是边界，不提供设备证据。"""
    calls = []
    communication = object()

    class Manager:
        def get_communication(self, **kwargs):
            calls.append(kwargs)
            return communication

    manager = Manager()
    constructor = function(
        upstream_source_root
        / "vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py",
        "FlashInferNVLinkTwoSidedPrepareAndFinalize",
        {
            "mk": NS(FusedMoEPrepareAndFinalizeModular=type("Base", (), {})),
            "FlashInferNVLinkTwoSidedManager": Manager,
            "get_ep_group": lambda: NS(device_communicator=NS(all2all_manager=manager)),
        },
    )
    factory = function(
        ROOT / "vllm_sail/patch/enhancement/deepep.py"
        if copied
        else upstream_source_root
        / "vllm/model_executor/layers/fused_moe/all2all_utils.py",
        "_maybe_make_prepare_finalize_body"
        if copied
        else "maybe_make_prepare_finalize",
        {
            "FlashInferNVLinkTwoSidedPrepareAndFinalize": constructor,
            "get_current_vllm_config": lambda: NS(
                scheduler_config=NS(max_num_batched_tokens=max_tokens)
            ),
        },
    )
    moe = NS(
        moe_parallel_config=NS(use_passthrough_all2all=False, use_all2all_kernels=True),
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_deepep_v2_kernels=False,
        use_moonep_kernels=False,
        use_mori_epv2_kernels=False,
        use_mori_epv1_kernels=False,
        use_fi_nvl_two_sided_kernels=True,
        experts_per_token=8,
        num_experts=256,
        hidden_dim=4096,
    )
    result = factory(moe, object(), all2all_manager=NS(world_size=dispatchers))
    assert isinstance(result, constructor)
    assert result.communication is communication
    assert result.all2all_manager is manager
    assert result.num_dispatchers() == dispatchers
    assert calls == [
        dict(max_num_tokens=max_tokens, top_k=8, num_experts=256, hidden_size=4096)
    ]


_ACTIVATION_FORMAT_CASES = [
    ("Standard", "Standard", True),
    ("Standard", "PaddedStandard", False),
    ("Standard", "BatchedExperts", False),
    ("PaddedStandard", "Standard", True),
    ("PaddedStandard", "PaddedStandard", True),
    ("PaddedStandard", "BatchedExperts", False),
    ("BatchedExperts", "Standard", False),
    ("BatchedExperts", "PaddedStandard", False),
    ("BatchedExperts", "BatchedExperts", True),
]


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize(
    "experts_format,pf_format,compatible", _ACTIVATION_FORMAT_CASES
)
def test_real_shared_oracle_factory_observes_installed_ppu_rebinding(
    modules, upstream_source_root, preloaded, experts_format, pf_format, compatible
):
    """真实共享 helper 和格式关系；专家构造是边界，不宣称 PPU 支持 padding。"""
    formats = function(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/modular_kernel.py",
        "FusedMoEActivationFormat",
        {"Enum": Enum},
    )
    factory_calls, expert_calls = [], []
    quant, config, routing = object(), object(), (object(), object(), object())
    pf = NS(
        activation_format=getattr(formats, pf_format),
        max_num_tokens_per_rank=lambda: 96,
        num_dispatchers=lambda: 4,
    )

    def original(*args, **kwargs):
        pytest.fail("预加载共享 oracle 必须观察已安装的 PPU factory")

    def local_factory(monolithic):
        factory_calls.append(monolithic)
        return pf

    platform = NS(is_ppu=lambda: True, is_xpu=lambda: False)
    modules("vllm.platforms", current_platform=platform)
    provider = modules(
        "vllm.model_executor.layers.fused_moe.all2all_utils",
        current_platform=platform,
        maybe_make_prepare_finalize=original,
        maybe_roundup_layer_hidden_size=lambda *args: 0,
        make_moe_prepare_and_finalize_no_dp_ep=local_factory,
    )
    consumer_name = "vllm.model_executor.layers.fused_moe.oracle.base"
    if preloaded:
        consumer = modules(consumer_name, maybe_make_prepare_finalize=original)
    patch_module = load_patch("vllm_sail/patch/enhancement/deepep.py")
    if not preloaded:
        consumer = modules(
            consumer_name,
            maybe_make_prepare_finalize=provider.maybe_make_prepare_finalize,
        )
    assert consumer.maybe_make_prepare_finalize is provider.maybe_make_prepare_finalize
    assert (
        consumer.maybe_make_prepare_finalize is patch_module.maybe_make_prepare_finalize
    )
    config = NS(
        moe_parallel_config=NS(
            use_passthrough_all2all=False, use_all2all_kernels=False, dp_size=1
        )
    )

    class Experts:
        @staticmethod
        def is_monolithic():
            return False

        @staticmethod
        def activation_format():
            return getattr(formats, experts_format)

        def __init__(self, **kwargs):
            expert_calls.append(kwargs)

    consumer.logger = NS(info_once=lambda *args: None)
    consumer.mk = NS(FusedMoEKernel=lambda prepare, experts: (prepare, experts))
    make_kernel = function(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/base.py",
        "make_moe_kernel",
        consumer.__dict__,
    )
    if not compatible:
        with pytest.raises(ValueError, match="expects .* activations"):
            make_kernel(quant, config, Experts, routing)
        assert expert_calls == []
    else:
        prepare, experts = make_kernel(quant, config, Experts, routing)
        assert prepare is pf
        assert isinstance(experts, Experts)
        expected = dict(moe_config=config, quant_config=quant)
        if pf.activation_format.is_batched:
            expected.update(max_num_tokens=96, num_dispatchers=4)
        assert expert_calls == [expected]
    assert factory_calls == [False]


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "experts_format,pf_format,compatible", _ACTIVATION_FORMAT_CASES
)
def test_real_kernel_checks_final_format_after_prepare_setup(
    upstream_source_root, experts_format, pf_format, compatible
):
    """真实末端检查必须使用 post-init 后格式；不执行通信或专家计算。"""
    path = (
        upstream_source_root / "vllm/model_executor/layers/fused_moe/modular_kernel.py"
    )
    formats = function(path, "FusedMoEActivationFormat", {"Enum": Enum})
    setup = function(path, "FusedMoEKernel._post_init_setup", {})
    calls = []
    experts = NS(activation_format=lambda: getattr(formats, experts_format))
    pf = NS(activation_format=formats.Standard)

    def post_init_setup(actual_experts):
        calls.append(actual_experts)
        pf.activation_format = getattr(formats, pf_format)

    pf.post_init_setup = post_init_setup
    kernel = NS(
        prepare_finalize=pf, fused_experts=experts, impl=NS(fused_experts=experts)
    )
    if compatible:
        setup(kernel)
    else:
        with pytest.raises(AssertionError):
            setup(kernel)
    assert calls == [experts]
    assert pf.activation_format is getattr(formats, pf_format)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "primary_format,fallback_format,expected",
    [
        ("Standard", "Standard", "Standard"),
        ("Standard", "PaddedStandard", "Standard"),
        ("Standard", "BatchedExperts", None),
        ("PaddedStandard", "Standard", "Standard"),
        ("PaddedStandard", "PaddedStandard", "PaddedStandard"),
        ("PaddedStandard", "BatchedExperts", None),
        ("BatchedExperts", "Standard", None),
        ("BatchedExperts", "PaddedStandard", None),
        ("BatchedExperts", "BatchedExperts", "BatchedExperts"),
    ],
)
def test_real_fallback_format_requires_both_runtime_branches(
    upstream_source_root, primary_format, fallback_format, expected
):
    """真实 fallback 格式关系取共同能力，不宣称 Standard 专家支持 padding。"""
    root = upstream_source_root / "vllm/model_executor/layers/fused_moe"
    formats = function(
        root / "modular_kernel.py", "FusedMoEActivationFormat", {"Enum": Enum}
    )
    activation_format = function(
        root / "experts/fallback.py", "FallbackExperts.activation_format", {}
    )
    primary = type(
        "Primary",
        (),
        {"activation_format": staticmethod(lambda: getattr(formats, primary_format))},
    )
    fallback = type(
        "Fallback",
        (),
        {"activation_format": staticmethod(lambda: getattr(formats, fallback_format))},
    )
    combined = NS(get_clses=lambda: (primary, fallback))
    if expected is None:
        with pytest.raises(AssertionError, match="incompatible activation formats"):
            activation_format(combined)
    else:
        assert activation_format(combined) is getattr(formats, expected)
