# SPDX-License-Identifier: Apache-2.0
"""Exercise V4.1 cache sizing and upstream kernel selection without torch."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.support.source import function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu,capability", [(True, 89), (False, 90), (False, 100)])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("manager_block", [128, 256])
def test_indexer_cache_pages(
    modules, upstream_source_root, ppu, capability, ratio, manager_block
):
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

        def __post_init__(self):
            size = self.block_size // self.tokens_per_state * self.head_size
            if self.page_size_padded is None:
                object.__setattr__(
                    self,
                    "page_size_padded",
                    -(-size // self.alignment) * self.alignment,
                )

    platform = SimpleNamespace(
        is_ppu=lambda: ppu,
        is_device_capability_family=lambda value: capability == value,
        is_device_capability=lambda value: capability == value[0] * 10 + value[1],
    )
    modules("vllm.platforms", current_platform=platform)
    modules("vllm_sail.models.deepseek_v41.config", uses_int8_w8a8=lambda config: False)
    indexer_source = upstream_source_root / "vllm/v1/attention/backends/mla/indexer.py"
    kernel_sizes = function(
        indexer_source,
        "DeepseekV41IndexerBackend.get_supported_kernel_block_sizes",
        {"current_platform": platform},
    )
    backend = type(
        "Backend", (), {"get_supported_kernel_block_sizes": staticmethod(kernel_sizes)}
    )
    modules("vllm.v1.attention.backends.mla.indexer", DeepseekV41IndexerBackend=backend)
    cache_spec = function(
        upstream_source_root / "vllm/models/deepseek_v41/attention.py",
        "DeepseekV4IndexerCache.get_kv_cache_spec",
        {
            "MLAAttentionSpec": Spec,
            "_use_v41_mxfp8_kv_record": lambda: capability == 100,
        },
    )
    cache_cls = type("Cache", (), {"get_kv_cache_spec": cache_spec})
    modules("vllm.models.deepseek_v41.attention", DeepseekV4IndexerCache=cache_cls)
    cache = cache_cls()
    cache.cache_config = SimpleNamespace(block_size=manager_block)
    cache.compress_ratio, cache.head_dim, cache.dtype = ratio, 132, "uint8"
    cache.sparse_logits = False
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"))
    before = cache.get_kv_cache_spec(config)
    harness.load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py")
    spec = cache.get_kv_cache_spec(config)
    select = function(
        upstream_source_root / "vllm/v1/worker/utils.py",
        "select_common_block_size",
        {"MultipleOf": type("MultipleOf", (), {})},
    )
    selected = select(spec.block_size, [backend])
    assert backend.get_supported_kernel_block_sizes(kv_cache_spec=spec) == (
        [64, 128] if ppu else [64 if capability == 90 else 128]
    )
    if ppu:
        assert spec.block_size == selected == 64 * ratio
        assert spec.block_size // spec.tokens_per_state == 64
        # Recompute alignment after changing the block size; never retain the
        # old 128/256-row page padding. No virtual splitting is needed.
        assert spec.page_size_padded == 8640
    else:
        assert spec == before
        assert selected == (64 if capability == 90 else 128)
    assert cache.cache_config.block_size == manager_block


@pytest.fixture
def q_launcher():
    class Tensor:
        def __init__(self, shape, dtype, *, strided=False):
            self.shape, self.dtype = shape, dtype
            self.ndim = len(shape)
            self.device, self.is_cuda = "ppu", True
            strides, span = [], 1
            for size in reversed(shape):
                strides.insert(0, span)
                span *= size
            if strided:
                strides[0] *= 2
            self.strides = tuple(strides)
            self.contiguous = not strided

        def stride(self, dim):
            return self.strides[dim]

        def is_contiguous(self):
            return self.contiguous

        def view(self, *shape):
            return self

    torch = SimpleNamespace(
        bfloat16="bf16",
        int8="int8",
        int64="int64",
        float32="fp32",
        contiguous_format="contiguous",
        empty_like=lambda tensor, **kw: Tensor(tensor.shape, kw["dtype"]),
        empty=lambda shape, **kw: Tensor(shape, kw["dtype"]),
    )
    calls = []
    launcher = function(
        "vllm_sail/models/deepseek_v41/ops/indexer_tilelang.py",
        "fused_indexer_q_rope_quant_int8",
        {
            "torch": torch,
            "_DTYPES": {"bf16": "bfloat16", "fp32": "float32"},
            "_flat": lambda tensor: tensor,
            "_q_kernel": lambda *args: calls.append(args),
        },
    )
    return Tensor, launcher, calls


@pytest.mark.parametrize("heads", [4, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("n", [0, 3])
@pytest.mark.parametrize("reuse_buffers", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_q_launcher_uses_actual_head_count(
    q_launcher, heads, n, reuse_buffers, strided
):
    Tensor, launch, calls = q_launcher
    q = Tensor((n, heads, 128), "bf16", strided=strided)
    positions = Tensor((n,), "int64")
    cs = Tensor((1024, 64), "fp32", strided=strided)
    w = Tensor((n, heads), "bf16", strided=strided)
    buffers = (
        (Tensor(q.shape, "int8"), Tensor(w.shape, "fp32")) if reuse_buffers else None
    )
    out, weights = launch(
        positions, q, cs, w, 0.125, heads**-0.5, output_buffers=buffers
    )
    assert out.shape == q.shape and out.dtype == "int8"
    assert weights.shape == (n, heads) and weights.dtype == "fp32"
    if reuse_buffers:
        assert out is buffers[0] and weights is buffers[1]
    assert len(calls) == bool(n)
    if n:
        args = calls[0]
        assert args[:6] == (positions, q, cs, w, out, weights)
        assert args[6:10] == (q.stride(0), q.stride(1), cs.stride(0), w.stride(0))
        assert args[10:] == (0.125, heads**-0.5, heads, "float32", "bfloat16")


@pytest.mark.parametrize("heads", [0, 1, 2, 3, 6, 31, 33])
def test_q_launcher_rejects_incomplete_head_groups(q_launcher, heads):
    Tensor, launch, calls = q_launcher
    with pytest.raises(AssertionError, match="complete groups of 4 heads"):
        launch(None, Tensor((1, heads, 128), "bf16"), None, None, 1.0, 1.0)
    assert not calls


@pytest.mark.parametrize(
    "shape,dtype,use_fp4,weights_dtype",
    [
        ((1, 32, 64), "bf16", False, "fp32"),
        ((1, 32, 128), "int8", False, "fp32"),
        ((1, 32, 128), "bf16", True, "fp32"),
        ((1, 32, 128), "bf16", False, "bf16"),
    ],
)
def test_q_launcher_keeps_quantization_geometry_contract(
    q_launcher, shape, dtype, use_fp4, weights_dtype
):
    Tensor, launch, calls = q_launcher
    with pytest.raises(AssertionError):
        launch(
            None,
            Tensor(shape, dtype),
            None,
            None,
            1.0,
            1.0,
            use_fp4=use_fp4,
            weights_out_dtype=weights_dtype,
        )
    assert not calls
