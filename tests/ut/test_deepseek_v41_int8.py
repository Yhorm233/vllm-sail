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
        int8="int8",
        ops=types.SimpleNamespace(
            vllm=types.SimpleNamespace(
                ppu_deepseek_v4_dequant_gather=operation("gather810")
            )
        ),
    )
    module(
        "vllm_sail.patch.utils",
        patch=patch_utils_module.patch,
        PATCH_MARKER=patch_utils_module.PATCH_MARKER,
    )
    module(
        "vllm.triton_utils",
        tl=types.SimpleNamespace(constexpr=object()),
        triton=types.SimpleNamespace(jit=lambda **kw: lambda fn: fn),
    )

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

    def int8_share(layout, block_size=32):
        return sharing.available

    module(
        PLUGIN + ".engram",
        can_share_engram_tables=int8_share,
        Int8ParallelEngramEmbedding=operation("embedding810"),
    )
    module(
        BASE + ".nvidia.model",
        _select_dsv4_attn_cls=lambda config: nvidia,
        _linear_scale_param_name=lambda config, dtype: "weight_scale_inv",
        can_share_engram_tables=can_share,
    )
    module(PLUGIN + ".flashmla", DeepseekV41FlashMLAAttention=ppu)
    module(
        BASE + ".common.engram",
        ParallelEngramEmbedding=Embedding,
        _engram_lookup_kernel=operation("engram_kernel_upstream"),
    )
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

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        spec.loader.exec_module(loaded)
        return loaded

    load(PLUGIN + ".config", "vllm_sail/models/deepseek_v41/config.py")
    prefix = "vllm_sail/patch/enhancement/models/"
    patched = load("_test_v41_int8", prefix + "deepseek_v41_int8.py")
    selector = load("_test_v41_selector", prefix + "deepseek_v41.py")
    engram = load("_test_v41_engram", prefix + "deepseek_v41_engram.py")
    indexer = load("_test_v41_indexer", prefix + "deepseek_v41_indexer.py")
    return types.SimpleNamespace(
        platform=platform,
        modules=modules,
        calls=calls,
        patch=patched,
        selector=selector,
        engram=engram,
        indexer=indexer,
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
    args = (*range(8), False)
    assert fn(*args) == "k810"
    assert installed.calls[-1][1] == args
    assert fn(*range(8), True) == "k_upstream"


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_non_ppu_delegation(installed, cap):
    state = installed
    state.platform.ppu, state.platform.capability = False, cap
    assert state.patch.fused_indexer_q_rope_quant(*range(6)) == "q_upstream"
    assert "output_buffers" not in state.calls[-1][2]
    assert state.patch.indexer_k_norm_rope_store(*range(9)) == "k_upstream"
    assert state.patch.rope_quant_insert(*range(6)) == "store_upstream"
    assert state.engram.lookup(None, 1, 2) == "lookup_upstream"
    assert state.selector._select_dsv4_attn_cls(state.config) is state.nvidia
    with pytest.raises(ValueError, match="output_buffers"):
        state.patch.fused_indexer_q_rope_quant(*range(6), output_buffers=(1, 2))


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_selector_cache_and_engram(installed, cap):
    state = installed
    state.platform.capability = cap

    def reject_capability_query(*args):
        raise AssertionError("V4.1 INT8 dispatch must not query chip capability")

    state.platform.is_device_capability = reject_capability_query
    assert state.patch._uses_int8() is True
    assert state.patch.fused_indexer_q_rope_quant(*range(6)) == "q810"
    assert state.patch.indexer_k_norm_rope_store(*range(8), False) == "k810"
    assert state.selector._select_dsv4_attn_cls(state.config) is state.ppu
    assert (
        state.modules[BASE + ".compressor"].rope_quant_insert(*range(6)) == "store810"
    )
    embedding = types.SimpleNamespace(weight=types.SimpleNamespace(dtype="int8"))
    assert state.engram.lookup(embedding, 1, 2, True) == "lookup810"
    assert state.calls[-1][1] == (embedding, 1, 2, True)
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


@pytest.mark.parametrize("record_bytes", [528, 584, 288])
@pytest.mark.parametrize("ppu,cap", [(True, (8, 0)), (True, (8, 9)), (False, (8, 0))])
@pytest.mark.parametrize("use_fnuz", [False, True])
def test_fp8_main_cache_decoding_does_not_enable_int8(
    installed, record_bytes, ppu, cap, use_fnuz
):
    state = installed
    state.config.quant_config = None
    state.platform.ppu, state.platform.capability = ppu, cap
    assert state.patch._uses_int8() is False
    cache = types.SimpleNamespace(dtype="uint8", shape=(2, 64, record_bytes))
    args = (0, cache, 2, None, 4, 64, 3)
    selected = ppu and not use_fnuz and record_bytes in (528, 584)
    consumer = state.modules[BASE + ".nvidia.prefill"].dequantize_and_gather_k_cache
    assert consumer(*args, use_fnuz=use_fnuz) == (
        "gather810" if selected else "gather_upstream"
    )
    assert state.calls[-1][1] == args
    assert state.calls[-1][2] == ({} if selected else dict(use_fnuz=use_fnuz))
    assert state.patch.fused_indexer_q_rope_quant(*range(6)) == "q_upstream"
    assert state.patch.indexer_k_norm_rope_store(*range(8), False) == "k_upstream"


@pytest.mark.parametrize("record_bytes", [528, 584])
def test_int8_main_cache_keeps_584_byte_contract(installed, record_bytes):
    cache = types.SimpleNamespace(dtype="uint8", shape=(2, 64, record_bytes))
    if record_bytes == 528:
        with pytest.raises(ValueError, match="584"):
            installed.patch.dequantize_and_gather_k_cache(0, cache, 2, None, 4, 64, 0)
    else:
        assert (
            installed.patch.dequantize_and_gather_k_cache(0, cache, 2, None, 4, 64, 0)
            == "gather810"
        )
    with pytest.raises(ValueError, match="584"):
        installed.patch.dequantize_and_gather_k_cache(
            0, cache, 2, None, 4, 64, 0, use_fnuz=True
        )


@pytest.mark.parametrize(
    "record_bytes,fp8_dim,bf16_dim,scale_dim,quant_block",
    [(528, 512, 0, 16, 32), (584, 448, 64, 8, 64)],
)
def test_software_main_cache_launch_preserves_plane_geometry(
    record_bytes, fp8_dim, bf16_dim, scale_dim, quant_block
):
    from tests.support.source import function

    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    dispatch = function(
        "vllm_sail/models/deepseek_v4/ops/cache.py",
        "_dequantize_and_gather_k_cache_triton",
        {"_dequantize_and_gather_k_kernel_sm80": Kernel()},
    )
    out = types.SimpleNamespace(stride=lambda dim: (102400, 1024)[dim])
    cache = types.SimpleNamespace(
        shape=(4, 64, record_bytes), stride=lambda dim: 64 * record_bytes + 512
    )
    seq_lens = types.SimpleNamespace(shape=(2,))
    table = types.SimpleNamespace(shape=(2, 4))
    dispatch(out, cache, seq_lens, None, table, 64, 3)
    grid, args, kwargs = calls.pop()
    assert grid == (2, 128)
    assert args == (out, 102400, 1024, cache, seq_lens, table, 3, None)
    assert kwargs == dict(
        max_blocks_per_seq=4,
        fp8_dim=fp8_dim,
        bf16_dim=bf16_dim,
        scale_dim=scale_dim,
        quant_block=quant_block,
        cache_block_size=64,
        token_data_size=fp8_dim + bf16_dim * 2,
        block_stride=64 * record_bytes + 512,
        output_dim=512,
        fp8_max=448.0,
        n_quant_blocks=fp8_dim // quant_block,
    )
    cache.shape = (4, 64, 132)
    with pytest.raises(ValueError, match="528- or 584-byte"):
        dispatch(out, cache, seq_lens, None, table, 64, 3)
    assert not calls


def test_engram_sharing_retains_upstream_capacity_fallback(installed):
    state = installed
    fn = state.modules[BASE + ".nvidia.model"].can_share_engram_tables
    assert fn(None) is True
    state.sharing.available = False
    assert fn(None) is False
    state.platform.ppu = False
    assert fn(None) is False


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_int8_engram_factory_and_non_quant_isolation(installed, cap):
    state = installed
    state.platform.capability = cap
    layout = types.SimpleNamespace(
        layer_ids=[1, 14],
        num_embeddings=[123, 456],
        head_dim=256,
        primes=[[[60, 63]], [[200, 256]]],
    )
    assert state.engram._create_embedding(None, layout, 1) == "embedding810"
    assert state.calls[-1][1] == (456, 256, (200, 256))
    assert state.calls[-1][2] == dict(
        cpu_offload=True, dp_shared_memory=True, use_thp=False
    )
    state.config.quant_config = None
    assert state.engram._create_embedding(None, layout, 1) == "embedding_upstream"
    state.sharing.available = False
    assert state.engram.can_share_engram_tables(layout) is False


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
@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
def test_other_compressed_formats_do_not_use_int8_engram(installed, field, value, cap):
    state = installed
    raw = state.config.model_config.hf_config.quantization_config
    raw["config_groups"]["W8A8"]["weights"][field] = value
    state.platform.capability = cap
    assert state.patch._uses_int8() is False
    layout = types.SimpleNamespace(layer_ids=[1])
    assert state.engram._create_embedding(None, layout, 0) == "embedding_upstream"


@pytest.mark.parametrize("kind", ["weights", "input_activations"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "float"),
        ("num_bits", 4),
        ("strategy", "tensor"),
        ("symmetric", False),
    ],
)
@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
def test_non_w8a8_quantization_delegates_all_int8_operators(
    installed, kind, field, value, cap
):
    state = installed
    raw = state.config.model_config.hf_config.quantization_config
    raw["config_groups"]["W8A8"][kind][field] = value
    state.platform.capability = cap
    assert state.patch._uses_int8() is False
    assert state.patch.fused_indexer_q_rope_quant(*range(6)) == "q_upstream"
    assert state.patch.indexer_k_norm_rope_store(*range(8), False) == "k_upstream"
    assert state.patch.rope_quant_insert(*range(6)) == "store_upstream"
    cache = types.SimpleNamespace(dtype="fp8", shape=(2, 64, 132))
    assert (
        state.patch.dequantize_and_gather_k_cache(0, cache, 2, 3, 4, 64, 0)
        == "gather_upstream"
    )
    embedding = types.SimpleNamespace(weight=types.SimpleNamespace(dtype="int8"))
    assert state.engram.lookup(embedding, 1, 2) == "lookup_upstream"


@pytest.mark.parametrize("model_type", ["deepseek_v4", "dspark", "other"])
def test_other_models_do_not_enter_v41_int8(installed, model_type):
    installed.config.model_config.hf_config.model_type = model_type
    assert installed.patch._uses_int8() is False


def test_fp4_query_keeps_original_provider(installed):
    assert (
        installed.patch.fused_indexer_q_rope_quant(*range(6), use_fp4=True)
        == "q_upstream"
    )


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_metadata_preserves_ratio_alignment(installed, cap):
    installed.platform.capability = cap
    config = installed.config
    config.model_config.hf_text_config = types.SimpleNamespace(
        index_n_heads=32, index_head_dim=128
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
    state.indexer._cache_spec = function(
        upstream_source_root / "vllm/models/deepseek_v41/attention.py",
        "DeepseekV4IndexerCache.get_kv_cache_spec",
        {
            "MLAAttentionSpec": Spec,
            "_use_v41_mxfp8_kv_record": lambda: capability == (10, 0),
        },
    )
    state.indexer._kernel_sizes = function(
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
    config = state.config
    config.cache_config = types.SimpleNamespace(cache_dtype="fp8_ds_mla")
    config.model_config.hf_text_config = types.SimpleNamespace(
        index_n_heads=32, index_head_dim=128
    )
    before = state.indexer._cache_spec(cache, config)
    spec = state.indexer.get_kv_cache_spec(cache, config)
    backend = state.modules[
        "vllm.v1.attention.backends.mla.indexer"
    ].DeepseekV41IndexerBackend
    select = function(
        upstream_source_root / "vllm/v1/worker/utils.py",
        "select_common_block_size",
        {"MultipleOf": type("MultipleOf", (), {})},
    )
    selected = select(spec.block_size, [backend])
    if ppu:
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
            BASE + ".common.ops",
            "fused_indexer_q_rope_quant",
            reason=state.patch.REASON,
            affected_versions=state.patch.AFFECTED_VERSIONS,
            remove_when=state.patch.REMOVE_WHEN,
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
    assert names.index("deepseek_v41_int8") < names.index("deepseek_v4")
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
    assert scope["can_share_engram_tables"](layout) is False
    available.free += 1
    group.size = 1
    assert scope["can_share_engram_tables"](layout) is False


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


@pytest.fixture
def sparse_indexer(installed):
    from tests.support.source import function

    state = installed
    state.config.parallel_config = types.SimpleNamespace(
        decode_context_parallel_size=1, prefill_context_parallel_size=1
    )
    state.config.kernel_config = types.SimpleNamespace(
        sparse_indexer_topk_backend="auto", enable_jit_warmup=False
    )
    state.platform.is_cuda = lambda: True
    state.platform.fp8_dtype = lambda: "float8_e4m3fn"
    state.module("vllm_sail.utils.deep_gemm", is_deep_gemm_supported=lambda: True)
    warmups = []
    state.module(
        "vllm.v1.attention.ops.common",
        _PACK_SEQ_TRITON_KERNEL=types.SimpleNamespace(
            register_warmup=lambda **kwargs: warmups.append(kwargs)
        ),
        _UNPACK_SEQ_TRITON_KERNEL=types.SimpleNamespace(
            register_warmup=lambda: warmups.append("unpack")
        ),
    )

    class Parent:
        def __init__(self):
            self.parent_initialized = True

    class Indexer(Parent):
        pass

    path = "vllm_sail/patch/enhancement/attention/sparse_attn_indexer.py"
    namespace = dict(
        SparseAttnIndexer=Indexer,
        torch=state.modules["torch"],
        current_platform=state.platform,
        get_current_vllm_config=lambda: state.config,
        has_deep_gemm=lambda: True,
    )
    init = function(path, "sparse_attn_indexer_init", namespace)
    calls = []

    def op(*args):
        calls.append(args)
        return "sparse_ppu"

    forward = function(
        path,
        "sparse_attn_indexer_forward_ppu",
        dict(
            torch=types.SimpleNamespace(
                int8="int8",
                ops=types.SimpleNamespace(
                    vllm=types.SimpleNamespace(ppu_sparse_attn_indexer=op)
                ),
            ),
            _encode_layer_name=lambda name: name,
        ),
    )
    Indexer.forward_ppu = forward
    native = function(
        path,
        "sparse_attn_indexer_forward_native",
        dict(
            current_platform=state.platform,
            _upstream_forward_native=lambda *args: "sparse_upstream",
        ),
    )
    Indexer.forward_native = native

    def construct(use_fp4_cache=False):
        indexer = Indexer.__new__(Indexer)
        cache = types.SimpleNamespace(prefix="v41.indexer", kv_cache=object())
        init(
            indexer,
            cache,
            128,
            "scale",
            2,
            128,
            32,
            32,
            object(),
            use_fp4_cache=use_fp4_cache,
        )
        return indexer

    return types.SimpleNamespace(
        construct=construct, calls=calls, namespace=namespace, warmups=warmups
    )


@pytest.mark.parametrize("ppu,contract", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("fp4_cache", [False, True])
def test_indexer_warmup_uses_int8_only_for_w8a8(
    installed, sparse_indexer, ppu, contract, fp4_cache
):
    installed.platform.ppu = ppu
    if not contract:
        installed.config.quant_config = None
    installed.config.kernel_config.enable_jit_warmup = True
    sparse_indexer.construct(use_fp4_cache=fp4_cache)
    dtype = "int8" if ppu and contract else "uint8" if fp4_cache else "float8_e4m3fn"
    assert sparse_indexer.warmups == [
        dict(dtype=dtype, pad_value=0 if fp4_cache else -float("inf")),
        "unpack",
    ]


@pytest.mark.parametrize("ppu,contract", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_indexer_int8_contract_is_frozen_at_construction(
    installed, sparse_indexer, ppu, contract, cap
):
    state = installed
    state.platform.ppu, state.platform.capability = ppu, cap
    if not contract:
        state.config.quant_config = None

    def unavailable(*args):
        raise AssertionError("indexer must not query capability or runtime config")

    state.platform.is_device_capability = unavailable
    indexer = sparse_indexer.construct()
    assert indexer.parent_initialized
    assert indexer._v41_int8_indexer is (ppu and contract)
    state.config.quant_config = None
    sparse_indexer.namespace["get_current_vllm_config"] = unavailable
    q, weights = types.SimpleNamespace(dtype="int8"), object()
    assert indexer.forward_ppu(None, q, None, weights) == "sparse_ppu"
    assert sparse_indexer.calls[0][3:7] == (q, None, None, weights)
    assert indexer._v41_int8_indexer is (ppu and contract)


@pytest.mark.parametrize("dispatch", ["forward_ppu", "forward_native"])
@pytest.mark.parametrize(
    "dtype,has_scale,fp4_cache",
    [
        ("int8", False, False),
        ("float8_e4m3fn", False, False),
        ("bfloat16", False, False),
        ("uint8", True, True),
        ("int8", True, False),
        ("int8", False, True),
    ],
)
@pytest.mark.parametrize("contract", [True, False])
def test_w8a8_indexer_rejects_non_int8_before_custom_op(
    installed, sparse_indexer, dispatch, dtype, has_scale, fp4_cache, contract
):
    if not contract:
        installed.config.quant_config = None
    indexer = sparse_indexer.construct(use_fp4_cache=fp4_cache)
    q, scale = types.SimpleNamespace(dtype=dtype), object() if has_scale else None
    quant = (q, scale) if has_scale else q
    invalid = dtype != "int8" or has_scale or fp4_cache
    if contract and invalid:
        with pytest.raises(ValueError, match="W8A8 indexer requires torch.int8"):
            getattr(indexer, dispatch)(None, quant, None, None)
        assert not sparse_indexer.calls
    else:
        assert getattr(indexer, dispatch)(None, quant, None, None) == "sparse_ppu"
        assert sparse_indexer.calls[0][3:5] == (q, scale)


def test_non_ppu_native_indexer_keeps_upstream(installed, sparse_indexer):
    installed.platform.ppu = False
    indexer = sparse_indexer.construct()
    assert indexer.forward_native(None, object(), None, None) == "sparse_upstream"
    assert not sparse_indexer.calls


@pytest.mark.parametrize("kind", ["int8", "fp8"])
@pytest.mark.parametrize("missing", [False, True])
def test_paged_logits_delegates_sdk_without_software_fallback(kind, missing):
    from tests.support.source import function

    calls = []

    def initialize():
        calls.append("initialize")

    def unavailable():
        raise RuntimeError("missing SDK logits")

    output = object()

    def implementation(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    name = kind + "_paged_mqa_logits"
    namespace = dict(_lazy_init=initialize, _missing=unavailable)
    namespace["_" + name + "_impl"] = None if missing else implementation
    logits = function("vllm_sail/utils/deep_gemm.py", name, namespace)
    args = [object() for _ in range(8)]
    if missing:
        with pytest.raises(RuntimeError, match="missing SDK logits"):
            logits(*args)
        assert calls == ["initialize"]
    else:
        assert logits(*args) is output
        assert calls == ["initialize", (tuple(args[:7]), dict(clean_logits=args[7]))]
        kwargs = dict(
            zip(
                (
                    "q_fp8",
                    "kv_cache_fp8",
                    "weights",
                    "context_lens",
                    "block_tables",
                    "schedule_metadata",
                    "max_model_len",
                    "clean_logits",
                ),
                args,
                strict=True,
            )
        )
        assert logits(**kwargs) is output


def test_fp8_indexer_software_fallback_is_removed():
    tree = ast.parse((ROOT / "vllm_sail/utils/deep_gemm.py").read_text())
    assert not any(
        isinstance(node, ast.Name | ast.FunctionDef)
        and getattr(node, "id", getattr(node, "name", None))
        == "_fp8_paged_mqa_logits_ppu10"
        for node in ast.walk(tree)
    )


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
