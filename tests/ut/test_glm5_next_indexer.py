# SPDX-License-Identifier: Apache-2.0
"""810E paged-KV block-size contract for the GLM-5.3-Flash kpool indexer.

Runs on a bare CPU runner: the patch module imports its vLLM targets at import
time, so this test loads ``glm5_next_indexer.py`` through importlib with those
targets stubbed in ``sys.modules`` (via the shared ``patch_utils_module``
harness) instead of importing the real framework.

The platform stub models vLLM's real capability semantics: an integer ``arch``
equal to ``DeviceCapability.to_int()`` (810E ``(8, 0)`` -> 80, 890P ``(8, 9)``
-> 89). ``is_device_capability_family`` matches by major (``arch // 10``) — the
behaviour that made a naive ``family(80)`` predicate capture 890P — while
``is_device_capability`` matches exactly, which is what the shipped predicate
uses.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).parents[2]
PATCH_FILE = (
    ROOT / "vllm_sail" / "patch" / "enhancement" / "models" / "glm5_next_indexer.py"
)
BACKEND = "vllm.v1.attention.backends.mla.indexer"
UPSTREAM_BLOCK_SIZES = [128, 256]  # the upstream default this patch must preserve

CAP_810E = 80  # capability (8, 0)
CAP_890P = 89  # capability (8, 9)


class _MultipleOf:
    """Stand-in for ``vllm.v1.attention.backend.MultipleOf``."""

    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        return isinstance(other, _MultipleOf) and other.n == self.n

    def __repr__(self):
        return f"MultipleOf({self.n})"


def _cdiv(a, b):
    return -(-a // b)


def _next_power_of_2(n):
    return 1 << (max(1, n) - 1).bit_length()


@pytest.fixture
def installed(monkeypatch, patch_utils_module):
    modules: dict[str, types.ModuleType] = {}

    def module(name, **attrs):
        if name not in modules:
            result = types.ModuleType(name)
            result.__path__ = []
            modules[name] = result
            monkeypatch.setitem(sys.modules, name, result)
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(module(parent), leaf, result)
        modules[name].__dict__.update(attrs)
        return modules[name]

    platform = types.SimpleNamespace(
        ppu=True,
        arch=CAP_810E,
        is_ppu=lambda: platform.ppu,
        # Exact capability match (major, minor) — what the shipped predicate uses.
        is_device_capability=(
            lambda capability, device_id=0: (
                platform.ppu
                and (
                    platform.arch == capability
                    if isinstance(capability, int)
                    else (platform.arch // 10, platform.arch % 10) == tuple(capability)
                )
            )
        ),
        # Real major-match semantics: any <major>.x returns True (8.9 == 80//10).
        is_device_capability_family=(
            lambda family, device_id=0: platform.arch // 10 == family // 10
        ),
    )
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(index_kpool=4)
        )
    )

    class Glm5NextIndexerBackend:
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return list(UPSTREAM_BLOCK_SIZES)

    module("vllm.config", get_current_vllm_config=lambda: config)
    module("vllm.platforms", current_platform=platform)
    module("vllm.v1.attention.backend", MultipleOf=_MultipleOf)
    module(BACKEND, Glm5NextIndexerBackend=Glm5NextIndexerBackend)
    module(
        "vllm_sail.patch.utils",
        patch=patch_utils_module.patch,
        PATCH_MARKER=patch_utils_module.PATCH_MARKER,
    )

    spec = importlib.util.spec_from_file_location("_test_glm5_indexer", PATCH_FILE)
    assert spec is not None and spec.loader is not None
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    assert patched is not None

    return platform, Glm5NextIndexerBackend, config


def _sizes(backend_cls, **kwargs):
    return backend_cls.get_supported_kernel_block_sizes(**kwargs)


def test_810e_advertises_multiple_of_index_kpool_times_64(installed):
    _, backend_cls, _ = installed
    result = _sizes(
        backend_cls, kv_cache_spec=types.SimpleNamespace(tokens_per_state=4)
    )
    assert result == [_MultipleOf(4 * 64)]
    # page_kv = block // index_kpool stays a positive multiple of 64 for every
    # block the selector may pick from the advertised granularity.
    for block in range(result[0].n, 8 * result[0].n + 1, result[0].n):
        assert (block // 4) % 64 == 0


def test_810e_follows_runtime_index_kpool_not_hardcoded(installed):
    _, backend_cls, _ = installed
    assert _sizes(
        backend_cls, kv_cache_spec=types.SimpleNamespace(tokens_per_state=3)
    ) == [_MultipleOf(3 * 64)]


def test_810e_uses_config_when_no_spec(installed):
    _, backend_cls, config = installed
    config.model_config.hf_text_config.index_kpool = 8
    assert _sizes(backend_cls, kv_cache_spec=None) == [_MultipleOf(8 * 64)]


def test_810e_block_divisible_by_mtp_tail_ring(installed):
    _, backend_cls, _ = installed
    result = _sizes(
        backend_cls, kv_cache_spec=types.SimpleNamespace(tokens_per_state=4)
    )
    index_kpool, num_spec = 4, 2
    ring = index_kpool * _next_power_of_2(_cdiv(index_kpool + num_spec, index_kpool))
    assert ring == 8
    assert result[0].n % ring == 0


def test_890p_passthrough_is_unchanged(installed):
    platform, backend_cls, _ = installed
    platform.ppu, platform.arch = True, CAP_890P  # Zhenwu M890 / SM89 -> (8, 9)
    result = _sizes(
        backend_cls, kv_cache_spec=types.SimpleNamespace(tokens_per_state=4)
    )
    assert result == UPSTREAM_BLOCK_SIZES


def test_890p_excluded_despite_family_major_match(installed):
    # Regression for review feedback on the original predicate:
    # ``is_device_capability_family(80)``
    # matches by major, so 890P (8, 9) also returns True and the old predicate
    # wrongly captured it. The shipped predicate uses the exact (8, 0) 810E
    # capability, so 890P stays on the upstream path.
    platform, backend_cls, _ = installed
    platform.ppu, platform.arch = True, CAP_890P
    assert platform.is_device_capability_family(80) is True  # the trap family(80) sets
    assert (
        platform.is_device_capability((8, 0)) is False
    )  # exact 810E test is False here
    assert (
        _sizes(backend_cls, kv_cache_spec=types.SimpleNamespace(tokens_per_state=4))
        == UPSTREAM_BLOCK_SIZES
    )


def test_810e_enters_branch_under_exact_capability(installed):
    # Symmetric guard: under the exact (8, 0) predicate, real 810E hardware still
    # routes into the page-size branch.
    platform, backend_cls, _ = installed
    platform.ppu, platform.arch = True, CAP_810E
    assert platform.is_device_capability((8, 0)) is True
    assert _sizes(
        backend_cls, kv_cache_spec=types.SimpleNamespace(tokens_per_state=4)
    ) == [_MultipleOf(4 * 64)]


def test_non_ppu_passthrough_is_unchanged(installed):
    platform, backend_cls, _ = installed
    platform.ppu = False
    platform.arch = CAP_810E
    assert _sizes(backend_cls, kv_cache_spec=None) == UPSTREAM_BLOCK_SIZES
