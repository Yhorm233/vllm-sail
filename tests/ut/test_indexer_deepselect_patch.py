# SPDX-License-Identifier: Apache-2.0
"""Installed DeepSelect providers and preloaded consumers use the SAIL wheel."""

from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import function

PATCH = "vllm_sail/patch/enhancement/attention/indexer_topk.py"
HELPER = "vllm_sail/utils/deep_select.py"
PROVIDER = "vllm.model_executor.layers.indexer_topk"
SPARSE = "vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits"
CANDIDATES = "vllm.model_executor.kernels.attention.dsa.candidate_blocks"


@pytest.mark.parametrize(
    "capability,enabled,wheel,expected",
    [
        (89, True, True, True),
        (80, True, True, False),
        (89, False, True, False),
        (89, True, False, False),
    ],
)
def test_sail_availability_uses_capability_flag_and_wheel(
    capability, enabled, wheel, expected
):
    namespace = {
        "current_platform": NS(
            is_ppu=lambda: True,
            has_device_capability=lambda cap: capability >= cap,
        ),
        "envs": NS(VLLM_SAIL_USE_DEEPSELECT=enabled),
        "_wheel": lambda: object() if wheel else None,
    }
    assert function(HELPER, "is_deep_select_available", namespace)() is expected


@pytest.fixture
def installed(modules):
    calls = []
    platform = NS(is_ppu=lambda: True)
    sail = NS(
        is_deep_select_available=lambda: True,
        can_use_deep_select=lambda logits, k: logits.dtype == "bf16",
        _wheel=lambda: NS(get_stride_requirement=lambda: (512, 32)),
        deep_select_topk=lambda *args: calls.append(args) or "sail-indices",
    )

    def original(*args, **kwargs):
        return "cuda-result"

    class Topk:
        resolve_backend = original

    modules("torch", int32="int32")
    modules("vllm.platforms", current_platform=platform)
    provider = modules(
        PROVIDER,
        get_deep_select_stride_requirement=original,
        is_deep_select_supported=original,
        deep_select_topk=original,
        SparseIndexerTopk=Topk,
    )
    sparse = modules(
        SPARSE,
        has_deep_select=original,
        get_deep_select_stride_requirement=original,
        deep_select_topk=original,
    )
    candidates = modules(
        CANDIDATES,
        has_deep_select=original,
        get_deep_select_stride_requirement=original,
        is_deep_select_supported=original,
        deep_select_topk=original,
    )
    layer = modules(
        "vllm.model_executor.layers.sparse_mqa_indexer", has_deep_select=original
    )
    backend = modules(
        "vllm.v1.attention.backends.mla.sparse_indexer", has_deep_select=original
    )
    step5 = modules(
        "vllm.models.step5.common.ops.sparse_indexer", is_deep_select_supported=original
    )
    modules("vllm_sail.utils.deep_select", **vars(sail))
    patch = load_patch(PATCH)
    return NS(
        patch=patch,
        provider=provider,
        sparse=sparse,
        candidates=candidates,
        layer=layer,
        backend=backend,
        step5=step5,
        platform=platform,
        sail=patch._sail,
        original=original,
        calls=calls,
    )


def test_preloaded_consumers_get_all_matching_replacements(
    installed, patch_utils_module
):
    p = installed.patch
    assert installed.sparse.has_deep_select is p.has_deep_select
    assert installed.candidates.has_deep_select is p.has_deep_select
    assert installed.layer.has_deep_select is p.has_deep_select
    assert installed.backend.has_deep_select is p.has_deep_select
    assert installed.step5.is_deep_select_supported is p.is_deep_select_supported
    for owner in (installed.provider, installed.candidates):
        for name in (
            "get_deep_select_stride_requirement",
            "is_deep_select_supported",
            "deep_select_topk",
        ):
            assert getattr(owner, name) is getattr(p, name)
    assert installed.sparse.deep_select_topk is p.deep_select_topk
    assert (
        installed.sparse.get_deep_select_stride_requirement
        is p.get_deep_select_stride_requirement
    )
    assert p.has_deep_select()
    assert p.get_deep_select_stride_requirement() == (512, 32)
    assert p.is_deep_select_supported(NS(dtype="bf16"), 64)
    assert not p.is_deep_select_supported(NS(dtype="fp32"), 64)
    assert installed.provider.SparseIndexerTopk.resolve_backend is p.resolve_backend
    for module, name, replacement in (
        (SPARSE, "has_deep_select", p.has_deep_select),
        (
            PROVIDER,
            "get_deep_select_stride_requirement",
            p.get_deep_select_stride_requirement,
        ),
        (PROVIDER, "is_deep_select_supported", p.is_deep_select_supported),
        (PROVIDER, "deep_select_topk", p.deep_select_topk),
        (PROVIDER, "SparseIndexerTopk.resolve_backend", p.resolve_backend),
    ):
        assert (
            patch_utils_module.original_of(replacement, module + "." + name)
            is installed.original
        )
        with pytest.raises(RuntimeError, match="already"):
            patch_utils_module.patch(
                module,
                name,
                reason=p._REASON,
                affected_versions=p._AFFECTED,
                remove_when=p._REMOVE_WHEN,
            )(lambda *args: None)
    with pytest.raises(RuntimeError, match="already"):
        load_patch(PATCH)


def test_upstream_entry_returns_sail_indices_and_preserves_optional_arguments(
    installed,
):
    logits = NS(dtype="bf16")
    assert (
        installed.patch.deep_select_topk(logits, 64, indices_dtype="int64")
        == "sail-indices"
    )
    assert installed.calls == [(logits, 64, None, None, "int64")]


@pytest.mark.parametrize("backend", ["auto", "deep_select"])
def test_cached_topk_objects_do_not_retain_cuda_only_support_flags(installed, backend):
    cached = NS(_backend=backend, _has_deep_select=False)
    resolve = installed.provider.SparseIndexerTopk.resolve_backend
    assert resolve(cached, NS(dtype="bf16"), 64, 2) == "deep_select"
    if backend == "auto":
        assert resolve(cached, NS(dtype="fp32"), 64, 2) == "cuda-result"
    else:
        with pytest.raises(RuntimeError, match="never rounded"):
            resolve(cached, NS(dtype="fp32"), 64, 2)


def test_non_ppu_probes_and_entry_preserve_upstream(installed):
    installed.platform.is_ppu = lambda: False
    p = installed.patch
    assert p.has_deep_select() == "cuda-result"
    assert p.get_deep_select_stride_requirement() == "cuda-result"
    assert p.is_deep_select_supported(None, 64) == "cuda-result"
    assert p.deep_select_topk(None, 64) == "cuda-result"
    assert p.resolve_backend(NS(_backend="deep_select"), None, 64, 2) == "cuda-result"


def test_upstream_entry_preserves_larger_strided_output():
    torch = pytest.importorskip("torch")
    out = torch.full((3, 8), -9, dtype=torch.int64)[:, ::2]

    def select(logits, k, end, output, dtype):
        assert output.shape == (2, 3) and dtype == torch.int64
        output.copy_(torch.tensor([[2, 1, 0], [0, -1, -1]]))
        return output

    topk = function(
        PATCH,
        "deep_select_topk",
        {
            "torch": torch,
            "current_platform": NS(is_ppu=lambda: True),
            "_sail": NS(deep_select_topk=select),
        },
    )
    assert (
        topk(torch.zeros((2, 8)), 3, output_idx=out, indices_dtype=torch.int64) is out
    )
    assert out[:2, :3].tolist() == [[2, 1, 0], [0, -1, -1]]
    assert out[:, 3].tolist() == [-9, -9, -9]
    assert out[2].tolist() == [-9, -9, -9, -9]
