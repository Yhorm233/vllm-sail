# SPDX-License-Identifier: Apache-2.0
"""V4.1 runtime contract tests without device dependencies."""

import ast
import importlib.util
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BASE = "vllm.models.deepseek_v41"
PLUGIN = "vllm_sail.models.deepseek_v41"


@pytest.fixture
def installed(monkeypatch, patch_utils_module):
    modules = {}

    def module(name, **attrs):
        if name not in modules:
            obj = types.ModuleType(name)
            obj.__path__ = []
            modules[name] = obj
            monkeypatch.setitem(sys.modules, name, obj)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(module(parent), leaf, obj)
        modules[name].__dict__.update(attrs)
        return modules[name]

    calls = []

    def operation(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return call

    platform = types.SimpleNamespace(ppu=True, capability=(8, 0))
    platform.is_ppu = lambda: platform.ppu
    platform.is_device_capability = lambda cap: cap == platform.capability
    module("vllm.platforms", current_platform=platform)
    module(
        "torch",
        float32="fp32",
        uint8="uint8",
        ops=types.SimpleNamespace(
            vllm=types.SimpleNamespace(
                ppu_deepseek_v4_dequant_gather=operation("gather810")
            )
        ),
    )
    module("vllm_sail.patch.utils", patch=patch_utils_module.patch)

    @dataclass(frozen=True)
    class Spec:
        indexer_n_head: int = 64
        indexer_q_head_dim: int = 128
        alignment: int = 576
        tokens_per_state: int = 2
        block_size: int = 256
        page_size_padded: int | None = 17280

    class Cache:
        compress_ratio = 2

        def get_kv_cache_spec(self, config):
            return Spec()

    class Embedding:
        lookup = operation("lookup_upstream")

    nvidia = type("NvidiaAttention", (), {})
    ppu = type("PPUAttention", (), {})
    module(BASE + ".nvidia.flashmla", DeepseekV4FlashMLAAttention=nvidia)
    sharing = types.SimpleNamespace(available=True)

    def can_share(layout, block_size=32):
        return sharing.available

    class Engram:
        _create_embedding = operation("embedding_upstream")

    config = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(get_name=lambda: "compressed-tensors"),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                model_type="deepseek_v41",
                quantization_config={
                    "expert_dtype": "int8",
                    "format": "int-quantized",
                    "config_groups": {
                        "W8A8": {
                            "targets": ["Linear"],
                            "weights": dict(
                                type="int",
                                num_bits=8,
                                strategy="channel",
                                symmetric=True,
                                dynamic=False,
                            ),
                            "input_activations": dict(
                                type="int",
                                num_bits=8,
                                strategy="token",
                                symmetric=True,
                                dynamic=True,
                            ),
                        },
                    },
                },
            )
        ),
        engram_config=types.SimpleNamespace(
            cpu_offload=True, dp_shared_memory=True, use_thp=False
        ),
    )
    module("vllm.config", get_current_vllm_config=lambda: config)
    module(BASE + ".nvidia.engram", can_share_engram_tables=can_share, Engram=Engram)

    def strict_share(layout, block_size=32):
        if not sharing.available:
            raise RuntimeError("fallback to per-rank full tables is forbidden")
        return True

    module(
        PLUGIN + ".engram",
        can_share_engram_tables=strict_share,
        Int8ParallelEngramEmbedding=operation("embedding810"),
    )
    module(
        BASE + ".nvidia.model",
        _select_dsv4_attn_cls=lambda config: nvidia,
        can_share_engram_tables=can_share,
    )
    module(PLUGIN + ".flashmla", DeepseekV41FlashMLAAttention=ppu)
    module(BASE + ".common.engram", ParallelEngramEmbedding=Embedding)
    q, k, store, gather = [
        operation(name)
        for name in ("q_upstream", "k_upstream", "store_upstream", "gather_upstream")
    ]
    module(
        "vllm.models.deepseek_v4.common.ops.fused_indexer_q",
        fused_indexer_q_rope_quant=q,
    )
    module(
        BASE + ".common.ops",
        fused_indexer_q_rope_quant=q,
        indexer_k_norm_rope_store=k,
        dequantize_and_gather_k_cache=gather,
    )
    module(BASE + ".common.ops.indexer_k_store", indexer_k_norm_rope_store=k)
    module(BASE + ".common.ops.fused_compress_quant_cache", rope_quant_insert=store)
    module(BASE + ".common.ops.cache_utils", dequantize_and_gather_k_cache=gather)
    module(
        BASE + ".attention",
        DeepseekV4IndexerCache=Cache,
        fused_indexer_q_rope_quant=q,
        indexer_k_norm_rope_store=k,
    )
    module(BASE + ".compressor", rope_quant_insert=store)
    module(BASE + ".nvidia.prefill", dequantize_and_gather_k_cache=gather)
    backend = type(
        "Backend",
        (),
        {
            "get_supported_kernel_block_sizes": staticmethod(
                lambda kv_cache_spec=None: [128]
            )
        },
    )
    module("vllm.v1.attention.backends.mla.indexer", DeepseekV41IndexerBackend=backend)
    module(
        PLUGIN + ".ops.indexer_tilelang",
        fused_indexer_q_rope_quant_int8=operation("q810"),
        indexer_k_norm_rope_store_int8=operation("k810"),
    )
    module(PLUGIN + ".ops.latent_store", rope_quant_insert=operation("store810"))
    module(PLUGIN + ".ops.engram_lookup", lookup=operation("lookup810"))
    path = ROOT / "vllm_sail/patch/enhancement/models/deepseek_v41.py"
    spec = importlib.util.spec_from_file_location("_test_v41_patch", path)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    return types.SimpleNamespace(
        platform=platform,
        modules=modules,
        calls=calls,
        patch=patched,
        module=module,
        original_q=q,
        utils=patch_utils_module,
        ppu=ppu,
        nvidia=nvidia,
        sharing=sharing,
        config=config,
    )


@pytest.mark.parametrize("consumer", ["common.ops", "attention"])
def test_query_aliases_and_buffers(installed, consumer):
    state = installed
    buffers = (object(), object())
    fn = state.modules[BASE + "." + consumer].fused_indexer_q_rope_quant
    assert fn(*range(6), output_buffers=buffers) == "q810"
    assert state.calls[-1][2]["output_buffers"] is buffers
    old = state.modules["vllm.models.deepseek_v4.common.ops.fused_indexer_q"]
    assert old.fused_indexer_q_rope_quant is state.original_q


@pytest.mark.parametrize(
    "consumer", ["common.ops.indexer_k_store", "common.ops", "attention"]
)
def test_k_aliases_forward_exact_contract(installed, consumer):
    fn = installed.modules[BASE + "." + consumer].indexer_k_norm_rope_store
    assert fn(*range(9)) == "k810"
    assert installed.calls[-1][1] == tuple(range(9))


@pytest.mark.parametrize("ppu,cap", [(False, (8, 0)), (True, (8, 9))])
def test_non_810e_delegation(installed, ppu, cap):
    state = installed
    state.platform.ppu, state.platform.capability = ppu, cap
    assert state.patch.fused_indexer_q_rope_quant(*range(6)) == "q_upstream"
    assert "output_buffers" not in state.calls[-1][2]
    assert state.patch.indexer_k_norm_rope_store(*range(9)) == "k_upstream"
    assert state.patch.rope_quant_insert(*range(6)) == "store_upstream"
    assert state.patch.lookup(None, 1, 2) == "lookup_upstream"
    assert state.patch.select_attention(None) is state.nvidia
    with pytest.raises(ValueError, match="output_buffers"):
        state.patch.fused_indexer_q_rope_quant(*range(6), output_buffers=(1, 2))


def test_selector_cache_and_engram(installed):
    state = installed
    assert state.patch.select_attention(None) is state.ppu
    assert (
        state.modules[BASE + ".compressor"].rope_quant_insert(*range(6)) == "store810"
    )
    assert state.patch.lookup(None, 1, 2, True) == "lookup810"
    assert state.calls[-1][1] == (None, 1, 2, True)
    cache = types.SimpleNamespace(dtype="uint8", shape=(2, 16, 584))
    assert (
        state.modules[BASE + ".nvidia.prefill"].dequantize_and_gather_k_cache(
            0, cache, 2, 3, 4, 16, 0
        )
        == "gather810"
    )
    cache.shape = (2, 16, 132)
    with pytest.raises(ValueError, match="584"):
        state.patch.dequantize_and_gather_k_cache(0, cache, 2, 3, 4, 16, 0)


def test_engram_shared_failure_does_not_replicate(installed):
    state = installed
    fn = state.modules[BASE + ".nvidia.model"].can_share_engram_tables
    assert fn(None) is True
    state.sharing.available = False
    with pytest.raises(
        RuntimeError, match="fallback to per-rank full tables is forbidden"
    ):
        fn(None)
    state.platform.ppu = False
    assert fn(None) is False


def test_int8_engram_factory_and_non_quant_isolation(installed):
    state = installed
    layout = types.SimpleNamespace(
        num_embeddings=[123, 456], head_dim=256, primes=[[[60, 63]], [[200, 256]]]
    )
    assert state.patch.create_embedding(None, layout, 1) == "embedding810"
    assert state.calls[-1][1] == (456, 256, (200, 256))
    assert state.calls[-1][2] == dict(
        cpu_offload=True, dp_shared_memory=True, use_thp=False
    )
    state.config.quant_config = None
    assert state.patch.create_embedding(None, layout, 1) == "embedding_upstream"
    state.sharing.available = False
    assert state.patch.can_share_engram_tables(layout) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "float"),
        ("num_bits", 4),
        ("strategy", "tensor"),
        ("symmetric", False),
        ("dynamic", True),
    ],
)
def test_other_compressed_formats_do_not_use_int8_engram(installed, field, value):
    state = installed
    raw = state.config.model_config.hf_config.quantization_config
    raw["config_groups"]["W8A8"]["weights"][field] = value
    assert state.patch._uses_int8_engram() is False
    assert state.patch.create_embedding(None, None, 0) == "embedding_upstream"


def test_metadata_preserves_ratio_alignment(installed):
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(index_n_heads=32, index_head_dim=128)
        )
    )
    cache = installed.modules[BASE + ".attention"].DeepseekV4IndexerCache()
    spec = cache.get_kv_cache_spec(config)
    assert (spec.indexer_n_head, spec.indexer_q_head_dim) == (32, 128)
    assert (spec.alignment, spec.tokens_per_state) == (576, 2)
    assert spec.block_size == 128 and spec.page_size_padded is None
    installed.platform.ppu = False
    assert cache.get_kv_cache_spec(config).indexer_n_head == 64


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "ppu,capability",
    [(True, (8, 0)), (True, (8, 9)), (False, (9, 0)), (False, (10, 0))],
)
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("manager_block", [128, 256])
def test_real_indexer_spec_and_kernel_selection(
    installed, upstream_source_root, ppu, capability, ratio, manager_block
):
    from tests.support.source import function

    @dataclass(frozen=True)
    class Spec:
        block_size: int
        num_kv_heads: int
        head_size: int
        dtype: object
        tokens_per_state: int
        alignment: int
        block_stride_alignment: int | None = None
        page_size_padded: int | None = None
        indexer_n_head: int = 64
        indexer_q_head_dim: int = 128

        def __post_init__(self):
            if self.page_size_padded is None:
                size = self.block_size // self.tokens_per_state * self.head_size
                object.__setattr__(
                    self,
                    "page_size_padded",
                    -(-size // self.alignment) * self.alignment,
                )

    state = installed
    state.platform.ppu, state.platform.capability = ppu, capability
    state.platform.is_device_capability_family = lambda value: (
        capability[0] * 10 == value
    )
    state.patch._spec = function(
        upstream_source_root / "vllm/models/deepseek_v41/attention.py",
        "DeepseekV4IndexerCache.get_kv_cache_spec",
        {
            "MLAAttentionSpec": Spec,
            "_use_v41_mxfp8_kv_record": lambda: capability == (10, 0),
        },
    )
    state.patch._kernel_sizes = function(
        upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py",
        "DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
        {"current_platform": state.platform},
    )
    cache = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=manager_block),
        compress_ratio=ratio,
        head_dim=132,
        dtype="uint8",
        sparse_logits=False,
    )
    config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(cache_dtype="fp8_ds_mla"),
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(index_n_heads=32, index_head_dim=128)
        ),
    )
    before = state.patch._spec(cache, config)
    spec = state.patch.get_kv_cache_spec(cache, config)
    backend = state.modules[
        "vllm.v1.attention.backends.mla.indexer"
    ].DeepseekV41IndexerBackend
    select = function(
        upstream_source_root / "vllm/v1/worker/utils.py",
        "select_common_block_size",
        {"MultipleOf": type("MultipleOf", (), {})},
    )
    selected = select(spec.block_size, [backend])
    if ppu and capability == (8, 0):
        assert selected == spec.block_size == 64 * ratio
        assert spec.block_size // spec.tokens_per_state == 64
        assert spec.page_size_padded == 8640
        assert (spec.indexer_n_head, spec.indexer_q_head_dim) == (32, 128)
    else:
        assert spec == before
        assert selected == (64 if capability == (9, 0) else 128)
    assert cache.cache_config.block_size == manager_block


def test_early_late_import_and_idempotent_rebinding(installed):
    state = installed
    late = state.module(BASE + ".late_consumer")
    exec(
        "from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant",
        late.__dict__,
    )
    assert late.fused_indexer_q_rope_quant is state.patch.fused_indexer_q_rope_quant
    state.patch._rebind_aliases()
    state.patch._rebind_aliases()
    other = object()
    state.modules[BASE + ".attention"].fused_indexer_q_rope_quant = other
    state.patch._rebind_aliases()
    assert state.modules[BASE + ".attention"].fused_indexer_q_rope_quant is other


def test_patch_marker_metadata_and_duplicate_rejection(installed):
    state = installed
    fn = state.patch.fused_indexer_q_rope_quant
    target = BASE + ".common.ops.fused_indexer_q_rope_quant"
    assert getattr(fn, state.utils.PATCH_MARKER)[target] is state.original_q
    records = [r for r in state.utils.PATCH_REGISTRY if r.target.startswith(BASE)]
    assert len(records) >= 10
    assert all(r.reason and r.affected_versions and r.remove_when for r in records)
    with pytest.raises(RuntimeError, match="already patched"):
        state.utils.patch(
            BASE + ".common.ops", "fused_indexer_q_rope_quant", **state.patch._METADATA
        )(fn)


def test_install_order_and_lazy_registration():
    source = (ROOT / "vllm_sail/patch/enhancement/models/__init__.py").read_text()
    tree = ast.parse(source)
    names = [
        a.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for a in node.names
    ]
    assert names.index("deepseek_v41") < names.index("deepseek_v4")
    scope = {"__name__": "_test_v41_registry"}
    exec(
        compile((ROOT / "vllm_sail/models/__init__.py").read_text(), "models", "exec"),
        scope,
    )
    assert (
        scope["_REQUIRED_PPU_MODELS"]["DeepseekV41ForCausalLM"]
        == PLUGIN + ".ppu_model:DeepseekV41ForCausalLM"
    )
    assert scope["_registered"] is False


def test_int8_sharing_uses_actual_free_bytes_and_margin():
    tree = ast.parse((ROOT / "vllm_sail/models/deepseek_v41/engram.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "can_share_engram_tables"
    )
    required = (384006168 + 384016682) * 260
    margin = max(8 * 1024**3, (required + 9) // 10)
    available = types.SimpleNamespace(free=required + margin)
    group = types.SimpleNamespace(size=16)
    scope = dict(
        os=types.SimpleNamespace(path=types.SimpleNamespace(isdir=lambda path: True)),
        shutil=types.SimpleNamespace(disk_usage=lambda path: available),
        get_engram_dp_size=lambda: group.size,
        SHM_PATH="/shared",
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "engram", "exec"), scope)
    layout = types.SimpleNamespace(num_embeddings=[384006168, 384016682], head_dim=256)
    assert scope["can_share_engram_tables"](layout)
    available.free -= 1
    with pytest.raises(RuntimeError, match="insufficient capacity"):
        scope["can_share_engram_tables"](layout)
    available.free += 1
    group.size = 1
    with pytest.raises(
        RuntimeError, match="fallback to per-rank full tables is forbidden"
    ):
        scope["can_share_engram_tables"](layout)


def test_int8_mapper_is_local_and_preserves_engram():
    tree = ast.parse((ROOT / "vllm_sail/models/deepseek_v41/ppu_model.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "int8_weights_mapper"
    )
    import copy

    scope = {"deepcopy": copy.deepcopy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "mapper", "exec"), scope)
    engram = re.compile(r"engram\.embed_tokens\.scale$")
    dense = re.compile(r"\.scale$")
    original = types.SimpleNamespace(
        orig_to_new_regex={
            engram: "engram.embed_tokens.weight_scale_inv",
            dense: ".weight_scale_inv",
        }
    )
    mapped = scope["int8_weights_mapper"](original)
    assert mapped is not original
    assert mapped.orig_to_new_regex[engram] == original.orig_to_new_regex[engram]
    assert mapped.orig_to_new_regex[dense] == ".weight_scale"
    assert original.orig_to_new_regex[dense] == ".weight_scale_inv"


def test_cudart_loading_is_lazy_and_cached_by_sdk(monkeypatch):
    import ctypes

    calls = []
    handles = []

    def load(path, mode):
        calls.append((path, mode))
        handle = object()
        handles.append(handle)
        return handle

    monkeypatch.setattr(ctypes, "CDLL", load)
    spec = importlib.util.spec_from_file_location(
        "_test_v41_cudart", ROOT / "vllm_sail/utils/cudart.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not calls
    monkeypatch.setenv("PPU_SDK", "/sdk/one")
    first = module.load_sdk_cudart()
    assert module.load_sdk_cudart() is first
    monkeypatch.setenv("PPU_SDK", "/sdk/two")
    second = module.load_sdk_cudart()
    assert second is not first
    assert module.load_sdk_cudart() is second
    assert calls == [
        ("/sdk/one/CUDA_SDK/lib64/libcudart.so", ctypes.RTLD_GLOBAL),
        ("/sdk/two/CUDA_SDK/lib64/libcudart.so", ctypes.RTLD_GLOBAL),
    ]
