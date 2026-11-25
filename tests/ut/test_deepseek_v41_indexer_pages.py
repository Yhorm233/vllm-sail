# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed V4.1 indexer patch with strict upstream doubles."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch
from tests.support.source import function


@dataclass(frozen=True)
class LayerConfig:
    spec: object
    manager_config: object = object()
    is_draft: bool = True


@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_indexer_publishes_geometry_and_physical_pages(
    modules, ppu, ratio, fp4, sparse
):
    @dataclass(frozen=True)
    class Spec:
        block_size: int = 256
        tokens_per_state: int = ratio
        page_size_padded: int | None = 33792
        block_stride_alignment: int | None = 256
        indexer_n_head: int | None = None
        indexer_q_head_dim: int | None = None

    class Cache:
        def get_kv_cache_config(self, config):
            return LayerConfig(Spec())

    class Indexer:
        def __init__(self, cache):
            self.k_cache = cache
            self.n_head = 32
            self.head_dim = 128
            self.use_fp4_kv = fp4
            # A distinct sparse contract catches accidental cache-dtype
            # overrides of the upstream operator's producer requirements.
            dtype = "sparse-dtype" if sparse else "bf16" if ppu and fp4 else "fp32"
            self.indexer_op = SimpleNamespace(weights_dtype=dtype)
            self.indexer_weights_dtype = "sparse-dtype" if sparse else "fp32"

    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return [128]

    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules("torch", bfloat16="bf16")
    modules(
        "vllm.models.deepseek_v41.attention",
        DeepseekV4IndexerCache=Cache,
        DeepseekV4Indexer=Indexer,
    )
    modules("vllm.v1.attention.backends.mla.indexer", DeepseekV41IndexerBackend=Backend)
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_indexer.py")
    cache = Cache()
    cache.compress_ratio = ratio
    cache.sparse_logits = sparse
    indexer = Indexer(cache)
    assert indexer.indexer_weights_dtype == (
        "sparse-dtype" if sparse else "bf16" if ppu and fp4 else "fp32"
    )
    layer = cache.get_kv_cache_config(object())
    assert layer.manager_config is LayerConfig.manager_config and layer.is_draft
    spec = layer.spec
    if ppu:
        assert spec.block_size == 64 * ratio
        assert spec.page_size_padded is None
        assert spec.block_stride_alignment == (512 if sparse else 256)
        assert (spec.indexer_n_head, spec.indexer_q_head_dim) == (32, 128)
        assert Backend.get_supported_kernel_block_sizes(spec) == [64, 128]
    else:
        assert spec == Spec()
        assert Backend.get_supported_kernel_block_sizes(spec) == [128]


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
