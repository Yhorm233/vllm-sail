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


@dataclass(frozen=True)
class LayerConfig:
    spec: object
    manager_config: object = object()
    is_draft: bool = True


ROOT = Path(__file__).parents[2]
BASE = "vllm.models.deepseek_v41"
PLUGIN = "vllm_sail.models.deepseek_v41"


@pytest.mark.upstream_source
def test_operator_alias_inventory_matches_main(installed, upstream_source_root):
    state = installed
    aliases = {
        name: (original, replacement)
        for name, original, replacement in state.patch._ALIASES
    }
    found = {name: set() for name in aliases}
    for path in (upstream_source_root / "vllm/models/deepseek_v41").rglob("*.py"):
        module = ".".join(path.relative_to(upstream_source_root).with_suffix("").parts)
        module = module.removesuffix(".__init__")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in found:
                        assert alias.asname is None
                        found[alias.name].add(module)
    assert found == {
        "fused_indexer_q_rope_quant": {
            BASE + ".attention",
            BASE + ".common.ops",
            BASE + ".nvidia.specialized_decode.attention.indexer",
        },
        "indexer_k_norm_rope_store": {BASE + ".attention", BASE + ".common.ops"},
        "rope_quant_insert": {BASE + ".compressor"},
        "dequantize_and_gather_k_cache": {
            BASE + ".amd.rocm",
            BASE + ".common.ops",
            BASE + ".nvidia.flash_mla_mega_attn",
            BASE + ".nvidia.flashmla",
        },
    }
    for name, consumers in found.items():
        for consumer in consumers:
            state.module(consumer, **{name: aliases[name][0]})
    state.patch._rebind_aliases()
    for name, consumers in found.items():
        for consumer in consumers:
            assert getattr(state.modules[consumer], name) is aliases[name][1]


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
        "vllm.logger",
        init_logger=lambda name: types.SimpleNamespace(info_once=lambda *a, **k: None),
    )
    module(
        "torch",
        float32="fp32",
        uint8="uint8",
        int8="int8",
        bfloat16="bf16",
        ops=types.SimpleNamespace(
            vllm=types.SimpleNamespace(
                ppu_deepseek_v4_dequant_gather=operation("gather810"),
                ppu_deepseek_v41_mxfp8_dequant_gather=operation("gather810_mxfp8"),
                ppu_deepseek_v41_nvfp4_dequant_gather=operation("gather810_nvfp4"),
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
        tl=types.SimpleNamespace(constexpr=object(), uint8="uint8"),
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
        block_stride_alignment: int | None = None

    class Cache:
        compress_ratio = 2
        sparse_logits = False
        n_head = 64
        q_head_dim = 128

        def get_kv_cache_config(self, config):
            return LayerConfig(Spec())

    class Indexer:
        def __init__(self, cache):
            self.k_cache, self.n_head, self.head_dim = cache, 64, 128
            self.use_fp4_kv = False

    class Embedding:
        lookup = operation("lookup_upstream")

    nvidia = type("NvidiaAttention", (), {})
    ppu = type("PPUAttention", (), {})
    module(BASE + ".nvidia.flashmla", DeepseekV4FlashMLAAttention=nvidia)
    module(
        BASE + ".nvidia.flash_mla_mega_attn",
        DeepseekV4MegaAttnAttention=type("NvidiaMegaAttention", (), {}),
    )
    module(
        PLUGIN + ".flash_mla_mega_attn",
        PPUDeepseekV4MegaAttnAttention=type("PPUMegaAttention", (), {}),
        fused_gate=lambda config: (False, "INT8 baseline fixture"),
    )
    sharing = types.SimpleNamespace(available=True, calls=[])

    def can_share(layout, block_size=32):
        sharing.calls.append((layout, block_size))
        return sharing.available

    class Engram:
        _create_embedding = operation("embedding_upstream")

    checkpoint = {}

    def safetensors_metadata(model, revision=None):
        calls.append(("safetensors_metadata", (model,), {"revision": revision}))
        return checkpoint

    module(
        "vllm.transformers_utils.config",
        get_safetensors_params_metadata=safetensors_metadata,
    )

    config = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(get_name=lambda: "compressed-tensors"),
        model_config=types.SimpleNamespace(
            model="/ckpt/v41-w8a8",
            revision=None,
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
            ),
        ),
        engram_config=types.SimpleNamespace(
            cpu_offload=True, dp_shared_memory=True, use_thp=False
        ),
    )
    module("vllm.config", get_current_vllm_config=lambda: config)
    module(BASE + ".common.engram", can_share_engram_tables=can_share)
    module(BASE + ".common.engram", Engram=Engram)

    module(
        PLUGIN + ".engram",
        ChannelwiseEngramEmbedding=operation("embedding810"),
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
        DeepseekV4Indexer=Indexer,
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
    module(
        PLUGIN + ".ops.cache_gather",
        register_ops=operation("register_v41_gather"),
    )

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        spec.loader.exec_module(loaded)
        return loaded

    contract = load(PLUGIN + ".config", "vllm_sail/models/deepseek_v41/config.py")

    def resolve():
        # Mirror DeepseekV41ForCausalLM.__init__: decide once at construction.
        contract.set_int8_w8a8(platform.is_ppu() and contract.uses_int8_w8a8(config))

    resolve()
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
        original_share=can_share,
        utils=patch_utils_module,
        ppu=ppu,
        nvidia=nvidia,
        sharing=sharing,
        config=config,
        checkpoint=checkpoint,
        resolve=resolve,
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


@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("peer_rows", [0, 8])
def test_fp8_lookup_preserves_main_native_routes(installed, cap, packed, peer_rows):
    state = installed
    state.platform.capability = cap
    embedding = types.SimpleNamespace(
        weight=types.SimpleNamespace(dtype="fp8"), peer_rows=peer_rows
    )
    output = types.SimpleNamespace(dtype="uint8" if packed else "bf16")
    expected = (
        "lookup810"
        if cap == (8, 0) and not packed and not peer_rows
        else "lookup_upstream"
    )
    assert state.engram.lookup(embedding, object(), output) == expected


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_selector_cache_and_engram(installed, cap):
    state = installed
    state.platform.capability = cap

    def reject_capability_query(*args):
        raise AssertionError("V4.1 INT8 dispatch must not query chip capability")

    state.platform.is_device_capability = reject_capability_query

    def unavailable():
        raise AssertionError("forward dispatch must not query the runtime config")

    state.modules["vllm.config"].get_current_vllm_config = unavailable
    state.engram.get_current_vllm_config = unavailable
    assert state.patch._uses_int8() is True
    assert state.patch.fused_indexer_q_rope_quant(*range(6)) == "q810"
    assert state.patch.indexer_k_norm_rope_store(*range(8), False) == "k810"
    assert state.selector._select_dsv4_attn_cls(state.config) is state.ppu
    assert (
        state.modules[BASE + ".compressor"].rope_quant_insert(*range(6)) == "store810"
    )
    embedding = types.SimpleNamespace(
        weight=types.SimpleNamespace(dtype="int8"), weight_dtype="int8"
    )
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
    """Only 810E software-decodes, and each record width keeps its own kernel."""
    state = installed
    state.config.quant_config = None
    state.platform.ppu, state.platform.capability = ppu, cap
    state.resolve()
    assert state.patch._uses_int8() is False
    cache = types.SimpleNamespace(dtype="uint8", shape=(2, 64, record_bytes))
    args = (0, cache, 2, None, 4, 64, 3)
    selected = ppu and cap == (8, 0) and not use_fnuz
    software = {584: "gather810", 528: "gather810_mxfp8", 288: "gather810_nvfp4"}
    consumer = state.modules[BASE + ".nvidia.prefill"].dequantize_and_gather_k_cache
    assert consumer(*args, use_fnuz=use_fnuz) == (
        software[record_bytes] if selected else "gather_upstream"
    )
    assert state.calls[-1][1] == args
    assert state.calls[-1][2] == (
        {} if selected else dict(use_fnuz=use_fnuz, max_gather_len=None)
    )
    if selected and record_bytes in (528, 288):
        assert state.calls[-2][0] == "register_v41_gather"
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


def _planar_dispatch(calls):
    from tests.support.source import function

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    return function(
        "vllm_sail/models/deepseek_v4/ops/cache.py",
        "_dequantize_and_gather_k_cache_triton",
        {"_dequantize_and_gather_k_kernel_sm80": Kernel()},
    )


def test_software_main_cache_launch_preserves_plane_geometry():
    calls = []
    dispatch = _planar_dispatch(calls)
    out = types.SimpleNamespace(stride=lambda dim: (102400, 1024)[dim])
    cache = types.SimpleNamespace(shape=(4, 64, 584), stride=lambda dim: 64 * 584 + 512)
    seq_lens = types.SimpleNamespace(shape=(2,))
    table = types.SimpleNamespace(shape=(2, 4))
    dispatch(out, cache, seq_lens, None, table, 64, 3)
    grid, args, kwargs = calls.pop()
    assert grid == (2, 128)
    assert args == (out, 102400, 1024, cache, seq_lens, table, 3, None)
    assert kwargs == dict(
        max_blocks_per_seq=4,
        fp8_dim=448,
        bf16_dim=64,
        scale_dim=8,
        quant_block=64,
        cache_block_size=64,
        token_data_size=576,
        block_stride=64 * 584 + 512,
        output_dim=512,
        fp8_max=448.0,
        n_quant_blocks=7,
    )
    assert not calls


@pytest.mark.parametrize("record_bytes", [528, 288, 132])
def test_software_planar_gather_rejects_interleaved_records(record_bytes):
    """The planar V4 kernel must never silently read a V4.1 interleaved record.

    Regression guard for the 528B branch that reused the planar address formula
    and misread every token's values and scales.
    """
    calls = []
    dispatch = _planar_dispatch(calls)
    out = types.SimpleNamespace(stride=lambda dim: (102400, 1024)[dim])
    cache = types.SimpleNamespace(
        shape=(4, 64, record_bytes), stride=lambda dim: 64 * record_bytes + 512
    )
    seq_lens = types.SimpleNamespace(shape=(2,))
    table = types.SimpleNamespace(shape=(2, 4))
    with pytest.raises(ValueError, match="584-byte V4 record"):
        dispatch(out, cache, seq_lens, None, table, 64, 3)
    assert not calls


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_engram_sharing_keeps_upstream_provider_and_consumer(
    installed, ppu, quantized, available
):
    state = installed
    state.platform.ppu = ppu
    state.sharing.available = available
    if not quantized:
        state.config.quant_config = None

    def unavailable():
        raise AssertionError("upstream sharing must not query the plugin config")

    state.modules["vllm.config"].get_current_vllm_config = unavailable
    state.engram.get_current_vllm_config = unavailable
    assert not hasattr(state.engram, "can_share_engram_tables")
    assert not hasattr(state.engram, "_rebind_share_aliases")
    targets = {
        BASE + ".common.engram.can_share_engram_tables",
        BASE + ".nvidia.model.can_share_engram_tables",
    }
    assert targets.isdisjoint(record.target for record in state.utils.PATCH_REGISTRY)
    layout = object()
    for consumer in ("common.engram", "nvidia.model"):
        fn = state.modules[BASE + "." + consumer].can_share_engram_tables
        assert fn is state.original_share
        assert fn(layout, block_size=64) is available
    assert state.sharing.calls == [(layout, 64), (layout, 64)]


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("use_thp", [False, True])
def test_int8_engram_factory_and_non_quant_isolation(installed, cap, shared, use_thp):
    state = installed
    state.platform.capability = cap
    state.config.engram_config.dp_shared_memory = shared
    state.config.engram_config.use_thp = use_thp
    layout = types.SimpleNamespace(
        layer_ids=[1, 14],
        num_embeddings=[123, 456],
        head_dim=256,
        primes=[[[60, 63]], [[200, 256]]],
    )
    assert state.engram._create_embedding(None, layout, 1) == "embedding810"
    assert state.calls[-1][1] == (456, 256, (200, 256))
    assert state.calls[-1][2] == dict(
        cpu_offload=True, dp_shared_memory=shared, use_thp=use_thp, weight_dtype="int8"
    )
    state.config.quant_config = None
    state.resolve()
    assert state.engram._create_embedding(None, layout, 1) == "embedding_upstream"
    state.sharing.available = False
    assert (
        state.modules[BASE + ".common.engram"].can_share_engram_tables(layout) is False
    )


def test_w8a8_engram_follows_checkpoint_table_dtype(installed):
    # W8A8 Linear exports may keep Engram as MXFP8 (E4M3 + E8M0 block scales).
    state = installed
    state.checkpoint.update(
        {
            "layers.1.engram.embed.weight": {"dtype": "I8"},
            "layers.1.engram.embed.scale": {"dtype": "F32"},
            "model.layers.14.engram.embed.weight": {"dtype": "F8_E4M3"},
            "model.layers.14.engram.embed.scale": {"dtype": "F8_E8M0"},
            "__metadata__": {"format": "pt"},
        }
    )
    layout = types.SimpleNamespace(
        layer_ids=[1, 14, 30],
        num_embeddings=[123, 456, 789],
        head_dim=256,
        primes=[[[60, 63]], [[200, 256]], [[300, 489]]],
    )
    assert state.engram._create_embedding(None, layout, 0) == "embedding810"
    assert state.calls[-1][2]["weight_dtype"] == "int8"
    assert state.engram._create_embedding(None, layout, 1) == "embedding_upstream"
    # Unknown tables (no safetensors headers) keep the model-level W8A8 decision.
    assert state.engram._create_embedding(None, layout, 2) == "embedding810"
    probes = [call for call in state.calls if call[0] == "safetensors_metadata"]
    assert probes == [("safetensors_metadata", ("/ckpt/v41-w8a8",), {"revision": None})]
    state.config.quant_config = None
    state.resolve()
    assert state.engram._create_embedding(None, layout, 0) == "embedding_upstream"


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
    state.resolve()
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
    state.resolve()
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
    installed.resolve()
    assert installed.patch._uses_int8() is False


def test_fp4_query_keeps_original_provider(installed):
    assert (
        installed.patch.fused_indexer_q_rope_quant(*range(6), use_fp4=True)
        == "q_upstream"
    )


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_indexer_spec_preserves_geometry_without_quantization_queries(installed, cap):
    class UnavailableModelConfig:
        @property
        def model_config(self):
            raise AssertionError("cache sizing must not query model geometry")

        @property
        def quant_config(self):
            raise AssertionError("cache sizing must not query quantization")

    installed.platform.capability = cap
    config = UnavailableModelConfig()
    cache = installed.modules[BASE + ".attention"].DeepseekV4IndexerCache()
    before = installed.indexer._cache_spec(cache, config).spec
    spec = cache.get_kv_cache_config(config).spec
    assert (
        (spec.indexer_n_head, spec.indexer_q_head_dim)
        == (before.indexer_n_head, before.indexer_q_head_dim)
        == (64, 128)
    )
    assert (spec.alignment, spec.tokens_per_state) == (576, 2)
    assert spec.block_size == 128 and spec.page_size_padded is None
    assert not hasattr(installed.indexer, "uses_int8_w8a8")
    installed.platform.ppu = False
    assert cache.get_kv_cache_config(config).spec == before


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
        max_tp_shards: int
        head_size: int
        dtype: object
        tokens_per_state: int
        alignment: int
        block_stride_alignment: int | None = None
        page_size_padded: int | None = None
        indexer_n_head: int | None = None
        indexer_q_head_dim: int | None = None

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
        "DeepseekV4IndexerCache.get_kv_cache_config",
        {
            "MLAAttentionSpec": Spec,
            "FullAttentionManager": types.SimpleNamespace(layer_config=LayerConfig),
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
    before = state.indexer._cache_spec(cache, config).spec
    spec = state.indexer.get_kv_cache_config(cache, config).spec
    backend = state.modules[
        "vllm.v1.attention.backends.mla.indexer"
    ].DeepseekV41IndexerBackend
    scope = {"MultipleOf": type("MultipleOf", (), {})}
    scope["_block_size_is_supported"] = function(
        upstream_source_root / "vllm/v1/worker/utils.py",
        "_block_size_is_supported",
        scope,
    )
    select = function(
        upstream_source_root / "vllm/v1/worker/utils.py",
        "select_common_block_size",
        scope,
    )
    selected = select(spec.block_size, [backend])
    if ppu:
        assert selected == spec.block_size == 64 * ratio
        assert spec.block_size // spec.tokens_per_state == 64
        assert spec.page_size_padded == 8640
    else:
        assert spec == before
        assert selected == (64 if capability == (9, 0) else 128)
    assert (
        (spec.indexer_n_head, spec.indexer_q_head_dim)
        == (before.indexer_n_head, before.indexer_q_head_dim)
        == (None, None)
    )
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
        == PLUGIN + ".model:DeepseekV41ForCausalLM"
    )
    assert scope["_registered"] is False


def test_int8_mapper_is_local_and_preserves_engram():
    tree = ast.parse((ROOT / "vllm_sail/models/deepseek_v41/model.py").read_text())
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
    state.config.model_config.max_model_len = 32
    state.config.parallel_config = types.SimpleNamespace(
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pcp_shard_decode_requests=False,
    )
    state.config.kernel_config = types.SimpleNamespace(
        sparse_indexer_topk_backend="auto", enable_jit_warmup=False
    )
    attention = types.SimpleNamespace(indexer_logits_dtype="auto")
    attention.resolve_indexer_logits_dtype = lambda default: (
        default
        if attention.indexer_logits_dtype == "auto"
        else attention.indexer_logits_dtype
    )
    state.config.attention_config = attention
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
        get_indexer_topk=lambda backend: object(),
        STR_DTYPE_TO_TORCH_DTYPE={"float32": "fp32", "bfloat16": "bf16"},
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

    def construct(
        use_fp4_cache=False,
        candidate_blocks=None,
        candidate_block_size=0,
        candidate_write=False,
        compress_ratio=1,
        max_model_len=32,
    ):
        indexer = Indexer.__new__(Indexer)
        cache = types.SimpleNamespace(prefix="v41.indexer", kv_cache=object())
        init(
            indexer,
            cache,
            128,
            "scale",
            2,
            128,
            max_model_len,
            32,
            object(),
            use_fp4_cache=use_fp4_cache,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
            compress_ratio=compress_ratio,
        )
        return indexer

    return types.SimpleNamespace(
        construct=construct, calls=calls, namespace=namespace, warmups=warmups
    )


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("fp4_cache", [False, True])
def test_dense_indexer_owns_the_ppu_weights_dtype(
    installed, sparse_indexer, ppu, fp4_cache
):
    installed.platform.ppu = ppu
    indexer = sparse_indexer.construct(use_fp4_cache=fp4_cache)
    if ppu:
        assert indexer.weights_dtype == ("bf16" if fp4_cache else "fp32")
    else:
        assert not hasattr(indexer, "weights_dtype")


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("fp4_cache", [False, True])
@pytest.mark.parametrize("requested", ["auto", "float32", "bfloat16"])
def test_dense_indexer_resolves_real_logits_policy(
    installed, sparse_indexer, upstream_source_root, ppu, fp4_cache, requested
):
    from tests.support.source import function

    config = installed.config.attention_config
    config.indexer_logits_dtype = requested
    resolver = function(
        upstream_source_root / "vllm/config/attention.py",
        "AttentionConfig.resolve_indexer_logits_dtype",
        {},
    )
    config.resolve_indexer_logits_dtype = types.MethodType(resolver, config)
    installed.platform.ppu = ppu
    default = "bfloat16" if ppu and fp4_cache else "float32"
    resolved = default if requested == "auto" else requested
    if ppu and resolved != default:
        with pytest.raises(ValueError, match="native score reduction"):
            sparse_indexer.construct(use_fp4_cache=fp4_cache)
        assert sparse_indexer.calls == []
    else:
        layer = sparse_indexer.construct(use_fp4_cache=fp4_cache)
        assert layer.logits_dtype == {"float32": "fp32", "bfloat16": "bf16"}[resolved]
        assert layer.parent_initialized
        if ppu:
            assert layer.weights_dtype == layer.logits_dtype


@pytest.mark.upstream_source
def test_complete_dense_constructor_preserves_upstream_outside_ppu_regions(
    upstream_source_root,
):
    from tests.support.source import definition

    path = ROOT / "vllm_sail/patch/enhancement/attention/sparse_attn_indexer.py"
    copied = definition(path, "sparse_attn_indexer_init")
    original = definition(
        upstream_source_root / "vllm/model_executor/layers/sparse_attn_indexer.py",
        "SparseAttnIndexer.__init__",
    )
    copied.name = original.name
    copied.decorator_list = original.decorator_list
    assert ast.unparse(copied.body[0]) == "super(SparseAttnIndexer, self).__init__()"
    copied.body[0] = original.body[0]
    kept = []
    ppu_conditions = []
    for node in copied.body:
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "current_platform.is_ppu()"
        ):
            ppu_conditions.append(node)
            continue
        if (
            isinstance(node, ast.AnnAssign)
            and ast.unparse(node.target) == "self._cp_kv_cache_interleave_size"
        ):
            assert ast.unparse(node.value) == "None"
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "vllm_sail.utils.deep_gemm"
        ):
            continue
        if isinstance(node, ast.If) and "is_deep_gemm_supported" in ast.unparse(
            node.test
        ):
            assert len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
            node = node.orelse[0]
            assert (
                ast.unparse(node.test)
                == "not current_platform.is_ppu() and current_platform.is_cuda() and (not has_deep_gemm())"
            )
            node.test = next(
                n.test
                for n in original.body
                if isinstance(n, ast.If) and "has_deep_gemm" in ast.unparse(n.test)
            )
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "vllm_config.kernel_config.enable_jit_warmup"
        ):
            warmup = node.body[-1]
            assert isinstance(warmup, ast.If)
            assert (
                ast.unparse(warmup.test)
                == "self.dcp_world_size > 1 and (not current_platform.is_ppu()) and current_platform.is_cuda() and has_cutedsl()"
            )
            warmup.test = original.body[-1].body[-1].test
        if isinstance(node, ast.If) and "_get_direct_dcp_topk_workspace" in ast.unparse(
            node
        ):
            assert isinstance(node.test, ast.BoolOp)
            guard = next(
                v
                for v in node.test.values
                if ast.unparse(v) == "not current_platform.is_ppu()"
            )
            node.test.values.remove(guard)
        kept.append(node)
    assert len(ppu_conditions) == 2
    assert "self.weights_dtype" in ast.unparse(ppu_conditions[0])
    assert "self.max_model_len" in ast.unparse(ppu_conditions[0])
    assert "self.logits_dtype != self.weights_dtype" in ast.unparse(ppu_conditions[1])
    copied.body = kept
    assert ast.dump(copied) == ast.dump(original)


@pytest.mark.parametrize(
    "runtime_len,ratio,caller_bound,expected",
    [
        (257, 2, 128, 129),
        (256, 2, 128, 128),
        (257, 1, 257, 257),
        (257, 2, 144, 144),
    ],
)
def test_sparse_indexer_window_uses_ceil_without_shrinking_explicit_bounds(
    sparse_indexer, installed, runtime_len, ratio, caller_bound, expected
):
    installed.config.model_config.max_model_len = runtime_len
    indexer = sparse_indexer.construct(compress_ratio=ratio, max_model_len=caller_bound)
    assert indexer.max_model_len == expected


@pytest.mark.parametrize("ppu,contract", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("fp4_cache", [False, True])
def test_indexer_warmup_preserves_cache_format(
    installed, sparse_indexer, ppu, contract, fp4_cache
):
    installed.platform.ppu = ppu
    if not contract:
        installed.config.quant_config = None
    installed.config.kernel_config.enable_jit_warmup = True
    sparse_indexer.construct(use_fp4_cache=fp4_cache)
    dtype = "uint8" if fp4_cache else "float8_e4m3fn"
    assert sparse_indexer.warmups == [
        dict(dtype=dtype, pad_value=0 if fp4_cache else -float("inf")),
        "unpack",
    ]


@pytest.mark.parametrize("ppu,contract", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0)])
def test_indexer_forward_does_not_depend_on_model_quantization(
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
    assert not hasattr(indexer, "_v41_int8_indexer")
    state.config.quant_config = None
    indexer.forward_ppu.__func__.__globals__["get_current_vllm_config"] = unavailable
    indexer.forward_ppu.__func__.__globals__["current_platform"] = state.platform
    q, weights = types.SimpleNamespace(dtype="int8"), object()
    assert indexer.forward_ppu(None, q, None, weights) == "sparse_ppu"
    assert sparse_indexer.calls[0][3:7] == (q, None, None, weights)
    assert not hasattr(indexer, "_v41_int8_indexer")


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
@pytest.mark.parametrize("candidate_write", [False, True])
def test_indexer_wrapper_forwards_inputs_to_custom_op(
    installed,
    sparse_indexer,
    dispatch,
    dtype,
    has_scale,
    fp4_cache,
    contract,
    candidate_write,
):
    if not contract:
        installed.config.quant_config = None
    candidates = object()
    indexer = sparse_indexer.construct(
        use_fp4_cache=fp4_cache,
        candidate_blocks=candidates,
        candidate_block_size=8,
        candidate_write=candidate_write,
    )
    q, scale = types.SimpleNamespace(dtype=dtype), object() if has_scale else None
    quant = (q, scale) if has_scale else q
    hidden_states, k, weights = object(), object(), object()
    assert getattr(indexer, dispatch)(hidden_states, quant, k, weights) == "sparse_ppu"
    assert sparse_indexer.calls == [
        (
            hidden_states,
            indexer.k_cache.prefix,
            indexer.k_cache.kv_cache,
            q,
            scale,
            k,
            weights,
            indexer.quant_block_size,
            indexer.scale_fmt,
            indexer.topk_tokens,
            indexer.head_dim,
            indexer.max_model_len,
            indexer.max_total_seq_len,
            indexer.topk_indices_buffer,
            indexer.skip_k_cache_insert,
            fp4_cache,
            candidates,
            8,
            candidate_write,
            indexer.topk_backend,
        )
    ]


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
