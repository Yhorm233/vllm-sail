# SPDX-License-Identifier: Apache-2.0
"""PPU channelwise Engram storage and factory contracts."""

import ast
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, definition
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("use_thp", [False, True])
def test_engram_channelwise_storage_keeps_real_fp32_scales(modules, offload, use_thp):
    allocations = []

    def allocate(*shape, **kwargs):
        allocations.append((shape, kwargs))
        return SimpleNamespace()

    def set_weight_attrs(weight, attrs):
        for key, value in attrs.items():
            assert not hasattr(weight, key), key
            setattr(weight, key, value)

    modules(
        "torch",
        empty=allocate,
        float8_e4m3fn="fp8",
        int8="int8",
        float32="fp32",
    )
    modules("vllm.model_executor.utils", set_weight_attrs=set_weight_attrs)

    class Upstream:
        lookup = object()
        _storage = object()
        _weight_loader = object()

        def __init__(self, n, dim, heads, **kwargs):
            self.dim, self.part_num_embeddings = dim, n
            self.__dict__.update(kwargs)
            assert kwargs["block_size"] == dim
            assert kwargs["use_thp"] is use_thp
            self.weight, self.weight_scale_inv = self._allocate_weights()
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})

    modules(
        "vllm.distributed",
        get_engram_dp_group=lambda: None,
        get_engram_dp_size=lambda: 1,
    )
    modules("vllm.distributed.device_communicators.shm_broadcast", SHM_PATH="/dev/shm")
    modules(
        "vllm.models.deepseek_v41.common.engram",
        _engram_head_shard_weight_loader=lambda *args: None,
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.engram",
        ParallelEngramEmbedding=Upstream,
        DPSharedEngramStorage=type("Storage", (), {}),
        _allocate_huge_page_storage=lambda *args: None,
    )
    cls = load_patch(
        "vllm_sail/models/deepseek_v41/engram.py"
    ).ChannelwiseEngramEmbedding
    embedding = cls(10, 256, (10,), cpu_offload=offload, use_thp=use_thp)
    assert [item[0] for item in allocations] == [(10, 256), (10, 1)]
    assert [item[1]["dtype"] for item in allocations] == ["fp8", "fp32"]
    assert all((item[1].get("device") == "cpu") is offload for item in allocations)
    assert embedding._weight_loader is Upstream._weight_loader
    assert embedding.weight_scale_inv.dummy_weight_value == 1.0
    assert (
        embedding.lookup is Upstream.lookup and embedding._storage is Upstream._storage
    )
    with pytest.raises(ValueError, match="dp_shared_memory"):
        cls(10, 256, (10,), dp_shared_memory=True)


def test_engram_factory_keeps_offload_and_original_layouts(modules):
    platform = SimpleNamespace(ppu=True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                quantization_config={
                    "fp8_channelwise_layers": ["layers.14.engram.embed"],
                }
            )
        ),
        engram_config=SimpleNamespace(cpu_offload=True, dp_shared_memory=False),
    )
    modules("vllm.config", get_current_vllm_config=lambda: config)
    modules(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_ppu=lambda: platform.ppu, is_device_capability=lambda cap: False
        ),
    )
    modules("torch", int8="int8")
    modules("vllm_sail.models.deepseek_v41.config", int8_w8a8_enabled=lambda: False)
    modules(
        "vllm.triton_utils",
        tl=SimpleNamespace(constexpr=object()),
        triton=SimpleNamespace(jit=lambda **kw: lambda fn: fn),
    )
    calls = []
    modules(
        "vllm_sail.models.deepseek_v41.engram",
        ChannelwiseEngramEmbedding=lambda *args, **kw: calls.append((args, kw)),
    )
    original = object()

    class Engram:
        def _create_embedding(self, *args):
            return original

    original_factory = Engram._create_embedding
    factory = modules(
        "vllm.models.deepseek_v41.nvidia.engram",
        Engram=Engram,
        can_share_engram_tables=lambda *args: False,
    )
    kernel = modules(
        "vllm.models.deepseek_v41.common.engram",
        _engram_lookup_kernel=lambda: None,
        ParallelEngramEmbedding=type("Embedding", (), {"lookup": lambda *args: None}),
    )
    path = "vllm_sail/patch/enhancement/models/deepseek_v41_engram.py"
    patch = load_patch(path)
    assert factory.Engram._create_embedding is patch._create_embedding
    assert kernel._engram_lookup_kernel is patch._engram_lookup_kernel
    marker = getattr(patch._create_embedding, patch.PATCH_MARKER)
    assert marker[f"{patch._FACTORY}.Engram._create_embedding"] is original_factory
    layout = SimpleNamespace(
        layer_ids=[1, 14],
        num_embeddings=[10, 20],
        head_dim=256,
        primes=[[(4, 6)], [(8, 12)]],
    )
    embedding = Engram()
    assert embedding._create_embedding(layout, 0) is original
    embedding._create_embedding(layout, 1)
    assert calls == [
        ((20, 256, (8, 12)), {"cpu_offload": True, "dp_shared_memory": False})
    ]
    platform.ppu = False
    assert embedding._create_embedding(layout, 1) is original
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch(path)


@pytest.fixture
def int8_storage(modules):
    allocations, loads, constructors = [], [], []

    class Tensor:
        def __init__(self, shape, dtype):
            self.shape, self.dtype, self.ndim = shape, dtype, len(shape)
            self.engram_vocab_start = 2

        def __getitem__(self, index):
            return Tensor((0,), self.dtype)

        def view(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], str):
                return Tensor(self.shape, shape[0])
            return Tensor(shape, self.dtype)

    def allocate(*shape, **kw):
        allocations.append((shape, kw))
        return Tensor(shape, kw["dtype"])

    class Shared:
        def _allocate(self, size):
            allocations.append((size, "shared"))
            return Tensor((size,), "uint8")

        def load_weight(self, param, loaded):
            loads.append(("shared", param, loaded))

    class Upstream:
        lookup = object()
        _storage = object()
        _weight_loader = staticmethod(lambda *args: loads.append(("upstream", *args)))

        def __init__(self, n, dim, heads, **kw):
            constructors.append((n, dim, heads, kw))
            if kw["dp_shared_memory"] and not kw["cpu_offload"]:
                raise ValueError("upstream requires CPU offload for shared UVA storage")
            self.part_num_embeddings, self.dim = n, dim
            self.__dict__.update(kw)
            self.weight, self.weight_scale_inv = self._allocate_weights()
            self.weight_scale_inv.dummy_weight_value = 127

    modules("torch", int8="int8", float8_e4m3fn="fp8", float32="fp32", empty=allocate)
    group = SimpleNamespace(value=object())
    modules(
        "vllm.distributed",
        get_engram_dp_group=lambda: group.value,
        get_engram_dp_size=lambda: 2,
    )
    modules("vllm.distributed.device_communicators.shm_broadcast", SHM_PATH="/dev/shm")
    modules(
        "vllm.models.deepseek_v41.common.engram",
        _engram_head_shard_weight_loader=lambda *args: loads.append(("shard", *args)),
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.engram",
        ParallelEngramEmbedding=Upstream,
        DPSharedEngramStorage=Shared,
        _allocate_huge_page_storage=lambda size: None,
    )
    module = load_patch("vllm_sail/models/deepseek_v41/engram.py")
    return SimpleNamespace(
        module=module,
        allocations=allocations,
        loads=loads,
        constructors=constructors,
        group=group,
        Tensor=Tensor,
        upstream=Upstream,
    )


@pytest.mark.parametrize("offload", [False, True])
def test_int8_embedding_reuses_constructor_storage_and_loader(int8_storage, offload):
    state = int8_storage
    embedding = state.module.ChannelwiseEngramEmbedding(
        10, 256, (4, 6), cpu_offload=offload, weight_dtype="int8"
    )
    assert state.constructors == [
        (
            10,
            256,
            (4, 6),
            dict(
                block_size=256,
                cpu_offload=offload,
                dp_shared_memory=False,
                use_thp=False,
            ),
        )
    ]
    assert [a[0] for a in state.allocations] == [(10, 256), (10, 1)]
    assert [a[1]["dtype"] for a in state.allocations] == ["int8", "fp32"]
    assert all((a[1].get("device") == "cpu") is offload for a in state.allocations)
    assert embedding.weight_scale_inv.dummy_weight_value == 1.0
    assert embedding.lookup is state.upstream.lookup
    assert embedding._storage is state.upstream._storage
    loaded = state.Tensor((12, 256), "int8")
    embedding._weight_loader(embedding.weight, loaded)
    assert state.loads == [("shard", embedding.weight, loaded)]
    with pytest.raises(ValueError, match="implicit casting"):
        embedding._weight_loader(embedding.weight, state.Tensor((12, 256), "fp32"))
    with pytest.raises(ValueError, match="all rows"):
        embedding._weight_loader(embedding.weight, state.Tensor((11, 256), "int8"))
    with pytest.raises(ValueError, match="shape"):
        embedding._weight_loader(
            embedding.weight_scale_inv, state.Tensor((12, 8), "fp32")
        )


@pytest.mark.parametrize("use_thp", [False, True])
def test_int8_shared_storage_preserves_upstream_allocation_and_barrier(
    int8_storage, use_thp
):
    state = int8_storage
    cls = state.module.ChannelwiseEngramEmbedding
    with pytest.raises(ValueError, match="CPU offload"):
        cls(10, 256, (4, 6), dp_shared_memory=True, weight_dtype="int8")
    embedding = cls(
        10,
        256,
        (4, 6),
        dp_shared_memory=True,
        cpu_offload=True,
        use_thp=use_thp,
        weight_dtype="int8",
    )
    assert state.allocations == [(10 * 260, "shared")]
    assert (embedding.weight.dtype, embedding.weight_scale_inv.dtype) == (
        "int8",
        "fp32",
    )
    assert embedding.weight_scale_inv.shape == (10, 1)
    loaded = state.Tensor((12, 256), "int8")
    embedding._weight_loader(embedding.weight, loaded)
    assert state.loads == [("shared", embedding.weight, loaded)]
    state.group.value = None
    with pytest.raises(RuntimeError, match="node-local"):
        cls(
            10,
            256,
            (4, 6),
            dp_shared_memory=True,
            cpu_offload=True,
            weight_dtype="int8",
        )
    with pytest.raises(ValueError, match="four-byte aligned"):
        state.module.Int8DPSharedEngramStorage(3, 3, 3, object())


@pytest.mark.parametrize("dtypes", [("fp8", "int8"), ("int8", "fp8")])
def test_channelwise_loader_is_selected_per_instance(int8_storage, dtypes):
    state = int8_storage
    for dtype in dtypes:
        kwargs = {"weight_dtype": dtype} if dtype == "int8" else {}
        embedding = state.module.ChannelwiseEngramEmbedding(10, 256, (4, 6), **kwargs)
        expected = (
            state.module.load_weight
            if dtype == "int8"
            else state.upstream._weight_loader
        )
        assert embedding.weight.dtype == dtype
        assert embedding._weight_loader is expected
        loaded = state.Tensor((12, 256), dtype)
        embedding._weight_loader(embedding.weight, loaded)
        assert state.loads[-1] == (
            "shard" if dtype == "int8" else "upstream",
            embedding.weight,
            loaded,
        )


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_int8_thp_storage_and_pinned_fallback(
    int8_storage, monkeypatch, offload, available
):
    state = int8_storage
    packed = state.Tensor((10 * 260,), "uint8") if available else None
    requests = []

    def allocate(size):
        requests.append(size)
        return packed

    monkeypatch.setattr(state.module, "_allocate_huge_page_storage", allocate)
    embedding = state.module.ChannelwiseEngramEmbedding(
        10, 256, (4, 6), cpu_offload=offload, use_thp=True, weight_dtype="int8"
    )
    assert requests == ([10 * 260] if offload else [])
    assert (embedding.weight.dtype, embedding.weight_scale_inv.dtype) == (
        "int8",
        "fp32",
    )
    assert (embedding.weight.shape, embedding.weight_scale_inv.shape) == (
        (10, 256),
        (10, 1),
    )
    assert embedding._weight_loader is state.module.load_weight
    assert embedding.weight_scale_inv.dummy_weight_value == 1.0
    if offload and available:
        assert embedding._packed is packed
        assert state.allocations == []
    else:
        assert [a[0] for a in state.allocations] == [(10, 256), (10, 1)]
        assert all((a[1].get("device") == "cpu") is offload for a in state.allocations)


@pytest.mark.upstream_source
def test_engram_e8m0_lookup_body_matches_upstream(upstream_source_root):
    class RemoveChannelwiseBranch(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "scales.dtype.element_ty == tl.float32":
                return node.orelse
            return self.generic_visit(node)

    local = definition(
        ROOT / "vllm_sail/patch/enhancement/models/deepseek_v41_engram.py",
        "_engram_lookup_kernel",
    )
    upstream = definition(
        upstream_source_root / "vllm/models/deepseek_v41/common/engram.py",
        "_engram_lookup_kernel",
    )
    local.decorator_list = local.decorator_list[1:]
    assert ast.dump(RemoveChannelwiseBranch().visit(local)) == ast.dump(upstream)
