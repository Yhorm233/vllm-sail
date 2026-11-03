# SPDX-License-Identifier: Apache-2.0
"""810E delayed mHC dispatch, alias lifecycle, and pinned upstream body contracts."""

import ast
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, definition
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch
PATH = "vllm_sail/patch/enhancement/attention/mhc_delayed.py"
NAMES = ("mhc_pre_delayed_tilelang", "mhc_fused_post_pre_delayed_tilelang")


class Tensor:
    def __init__(self, *shape, dtype="bf16", device="ppu"):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)

    def is_contiguous(self):
        return True

    def view(self, *shape):
        return Tensor(*shape, dtype=self.dtype)

    def unsqueeze(self, dim):
        assert dim == -1
        return self.view(*self.shape, 1)


@pytest.fixture
def delayed(modules):
    calls = []
    platform = SimpleNamespace(ppu=True, capability=(8, 0))
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: platform.ppu,
            is_device_capability=lambda value: platform.capability == value,
        ),
    )
    modules(
        "torch",
        Tensor=Tensor,
        bfloat16="bf16",
        float32="fp32",
        empty=Tensor,
        empty_like=lambda x: Tensor(*x.shape, dtype=x.dtype),
    )

    def record(kind):
        def call(*args, **kwargs):
            calls.append((kind, args, kwargs))

        return call

    def prenorm(x, fn, **kwargs):
        calls.append(("ppu", (x, fn), kwargs))
        return Tensor(1, x.shape[0], fn.shape[0], dtype="fp32"), Tensor(
            1, x.shape[0], dtype="fp32"
        )

    modules(
        "vllm_sail.patch.enhancement.attention.mhc_tilelang",
        _hc_prenorm_gemm_outputs=prenorm,
    )
    modules(
        "vllm.model_executor.kernels.mhc.tilelang_kernels",
        _HC_PRENORM_GEMM_TILELANG_KERNEL=record("tilelang"),
        _MHC_FUSED_TILELANG_KERNEL=record("fused"),
        _MHC_POST_TILELANG_KERNEL=record("post"),
        mhc_fused_post_pre_split_config=lambda *args: None,
        mhc_pre_big_fuse_tilelang=record("epilogue"),
    )
    modules(
        "vllm.model_executor.kernels.mhc.warmup",
        MHC_PRE_NORM_KERNEL=record("norm"),
        compute_mhc_pre_num_splits=lambda *args: 4,
    )
    modules(
        "vllm.utils.deep_gemm",
        is_deep_gemm_supported=lambda: True,
        tf32_hc_prenorm_gemm=record("cuda"),
    )
    originals = {name: record(name) for name in NAMES}
    provider = modules("vllm.model_executor.kernels.mhc.tilelang", **originals)
    consumer = modules("vllm.models.deepseek_v41.nvidia.model", **originals)
    dispatch = modules("vllm.models.deepseek_v41.nvidia.ops.mega_mhc", **originals)
    fixed_dispatch = modules("vllm.models.deepseek_v41.nvidia.ops.mhc", **originals)
    patch = load_patch(PATH)
    return SimpleNamespace(
        patch=patch,
        provider=provider,
        consumer=consumer,
        dispatch=dispatch,
        fixed_dispatch=fixed_dispatch,
        originals=originals,
        calls=calls,
        platform=platform,
    )


def test_delayed_aliases_and_double_install_guard(delayed):
    for name, consumer in zip(NAMES, (delayed.consumer, delayed.dispatch), strict=True):
        replacement = getattr(delayed.patch, name)
        assert getattr(delayed.provider, name) is replacement
        assert getattr(consumer, name) is replacement
        marker = getattr(replacement, delayed.patch.PATCH_MARKER)
        assert marker[f"{delayed.patch._MODULE}.{name}"] is delayed.originals[name]
    assert (
        delayed.fixed_dispatch.mhc_fused_post_pre_delayed_tilelang
        is delayed.dispatch.mhc_fused_post_pre_delayed_tilelang
        is delayed.patch.mhc_fused_post_pre_delayed_tilelang
    )
    delayed.patch._rebind_delayed_aliases()
    delayed.consumer.mhc_pre_delayed_tilelang = object()
    delayed.patch._rebind_delayed_aliases()
    assert (
        delayed.consumer.mhc_pre_delayed_tilelang
        is not delayed.patch.mhc_pre_delayed_tilelang
    )
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch(PATH)


@pytest.mark.parametrize(
    "ppu,capability", [(True, (8, 0)), (True, (9, 0)), (False, (8, 0))]
)
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("tokens", [0, 7])
@pytest.mark.parametrize("norm", [True, False])
def test_delayed_prenorm_and_epilogue_contract(
    delayed, ppu, capability, fused, tokens, norm
):
    delayed.platform.ppu = ppu
    delayed.platform.capability = capability
    use_ppu = ppu
    residual = Tensor(tokens, 4, 1024)
    fn = Tensor(24, 4096, dtype="fp32")
    common = (
        residual,
        fn,
        Tensor(3, dtype="fp32"),
        Tensor(24, dtype="fp32"),
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
    )
    kwargs = {
        "pre_mix": Tensor(tokens, 4, dtype="fp32"),
        "norm_weight": Tensor(1024) if norm else None,
    }
    if fused:
        result = delayed.patch.mhc_fused_post_pre_delayed_tilelang(
            Tensor(tokens, 1024),
            residual,
            Tensor(tokens, 4, 1, dtype="fp32"),
            Tensor(tokens, 4, 4, dtype="fp32"),
            *common[1:],
            capture_aux=True,
            **kwargs,
        )
        assert result[-1].shape == (tokens, 1024)
    else:
        result = delayed.patch.mhc_pre_delayed_tilelang(*common, **kwargs)
        assert result[-1].shape == (tokens, 4)
    kinds = [call[0] for call in delayed.calls]
    if tokens == 0:
        assert kinds == []
        return
    assert kinds == (["post"] if fused else []) + [
        "ppu" if use_ppu else "cuda",
        "norm" if norm else "epilogue",
    ]
    _, args, epilogue = delayed.calls[-1]
    assert args[0].shape == (1 if use_ppu else 4, tokens, 24)
    assert epilogue["save_pre_mix"] and epilogue["use_pre_mix_in"]
    assert epilogue["rms_numel"] == 4096
    if fused:
        assert epilogue["write_aux"]


@pytest.mark.upstream_source
@pytest.mark.parametrize("name", NAMES)
def test_non_ppu_delayed_body_matches_upstream(upstream_source_root, name):
    class RemovePPUBranch(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "current_platform.is_ppu()":
                return node.orelse
            return self.generic_visit(node)

    local = definition(ROOT / PATH, name)
    upstream = definition(
        upstream_source_root / "vllm/model_executor/kernels/mhc/tilelang.py", name
    )
    local.decorator_list = []
    assert ast.dump(RemovePPUBranch().visit(local)) == ast.dump(upstream)


@pytest.mark.upstream_source
def test_delayed_consumer_inventory_matches_fixed_source(upstream_source_root):
    tree = ast.parse((ROOT / PATH).read_text())
    inventory = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_ALIAS_CONSUMERS"
            for target in node.targets
        )
    )
    found = {name: set() for name in NAMES}
    source = upstream_source_root / "vllm/models/deepseek_v41"
    for path in source.rglob("*.py"):
        module = ".".join(path.relative_to(upstream_source_root).with_suffix("").parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == (
                "vllm.model_executor.kernels.mhc.tilelang"
            ):
                for alias in node.names:
                    if alias.name in found:
                        assert alias.asname is None
                        found[alias.name].add(module)
    fixed = {
        "mhc_pre_delayed_tilelang": {"vllm.models.deepseek_v41.nvidia.model"},
        "mhc_fused_post_pre_delayed_tilelang": {
            "vllm.models.deepseek_v41.nvidia.ops.mhc"
        },
    }
    assert found == fixed
    assert {name: set(consumers) for name, consumers in inventory.items()} == {
        NAMES[0]: fixed[NAMES[0]],
        NAMES[1]: fixed[NAMES[1]] | {"vllm.models.deepseek_v41.nvidia.ops.mega_mhc"},
    }


def test_delayed_future_consumer_and_unrelated_alias(delayed, modules):
    future = modules("vllm.models.deepseek_v41.future_consumer")
    exec(
        "from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_delayed_tilelang",
        future.__dict__,
    )
    assert future.mhc_pre_delayed_tilelang is delayed.patch.mhc_pre_delayed_tilelang
    unrelated = modules("vllm.models.deepseek_v4.model", **delayed.originals)
    delayed.patch._rebind_delayed_aliases()
    assert unrelated.mhc_pre_delayed_tilelang is delayed.originals[NAMES[0]]
