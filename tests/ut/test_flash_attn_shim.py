# SPDX-License-Identifier: Apache-2.0
"""Exercise FA imports in fresh interpreters without torch, vLLM or a device."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.support.source import assert_accepts_upstream_keywords


@pytest.fixture
def run_shim(tmp_path: Path):
    upstream = tmp_path / "vllm"
    upstream.mkdir()
    (upstream / "__init__.py").write_text("")
    (upstream / "logger.py").write_text(
        "from logging import getLogger as init_logger\n"
    )
    fa = upstream / "vllm_flash_attn"
    fa.mkdir()
    # Either upstream file executing is the regression, even if CUDA binaries
    # happen to be installed on the machine running this test.
    for name in ("__init__.py", "flash_attn_interface.py"):
        (fa / name).write_text(
            "raise AssertionError('upstream CUDA FA binary import reached')\n"
        )

    def run(code: str) -> None:
        prelude = f"""
import sys
sys.path[:0] = [{str(tmp_path)!r}, {str(Path(__file__).parents[2])!r}]
class BlockSDK:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'flash_attn', 'flash_attn_3'}}:
            raise AssertionError('eager SDK import: ' + fullname)
sys.meta_path.insert(0, BlockSDK())
import vllm_sail
"""
        result = subprocess.run(
            [sys.executable, "-c", prelude + textwrap.dedent(code)],
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


@pytest.mark.parametrize("interface_first", [False, True])
def test_platform_hook_redirects_package_and_interface_without_loading_sdk(
    run_shim, interface_first: bool
) -> None:
    run_shim(f"""
        import importlib
        vllm_sail.register()
        vllm_sail.register()
        package_name = 'vllm.vllm_flash_attn'
        interface_name = package_name + '.flash_attn_interface'
        names = [interface_name, package_name] if {interface_first!r} else [package_name, interface_name]
        for name in names:
            importlib.import_module(name)
        from vllm.vllm_flash_attn import (
            flash_attn_varlen_func, get_scheduler_metadata,
            is_fa_version_supported, compile_flash_attn_varlen_func_from_specs,
        )
        import vllm
        package = sys.modules[package_name]
        interface = sys.modules[interface_name]
        assert vllm.vllm_flash_attn is package
        assert package.flash_attn_interface is interface
        assert package.flash_attn_varlen_func is interface.flash_attn_varlen_func
        assert callable(flash_attn_varlen_func)
        assert callable(get_scheduler_metadata)
        assert callable(is_fa_version_supported)
        assert compile_flash_attn_varlen_func_from_specs is None
    """)


def test_already_loaded_package_and_interface_are_both_rebound(run_shim) -> None:
    run_shim("""
        import types
        import vllm
        package = types.ModuleType('vllm.vllm_flash_attn')
        interface = types.ModuleType(package.__name__ + '.flash_attn_interface')
        old = lambda: 'upstream'
        for module in (package, interface):
            module.flash_attn_varlen_func = old
            module.compile_flash_attn_varlen_func_from_specs = old
            module.FA2_AVAILABLE = True
            sys.modules[module.__name__] = module
        package.flash_attn_interface = interface
        vllm.vllm_flash_attn = package
        vllm_sail.register()
        assert sys.modules[package.__name__] is package
        assert sys.modules[interface.__name__] is interface
        assert package.flash_attn_varlen_func is interface.flash_attn_varlen_func
        assert package.flash_attn_varlen_func is not old
        assert package.compile_flash_attn_varlen_func_from_specs is None
        assert 'FA2_AVAILABLE' not in vars(package)
        assert 'FA2_AVAILABLE' not in vars(interface)
    """)


def test_broken_sdk_wheels_do_not_block_import_and_keep_the_real_error(run_shim):
    run_shim("""
        import types
        vllm_sail.register()
        from vllm.vllm_flash_attn import (
            flash_attn_varlen_func, get_scheduler_metadata,
            is_fa_version_supported, fa_version_unsupported_reason,
        )
        # Only signature annotations need torch for this CPU-only failure path.
        torch = types.ModuleType('torch')
        torch.Tensor = type('Tensor', (), {})
        torch.bfloat16 = object()
        sys.modules['torch'] = torch
        sys.meta_path.pop(0)
        class BrokenWheels:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'flash_attn', 'flash_attn_3'}:
                    raise ImportError('undefined symbol: ppu_fa_torch_abi')
        sys.meta_path.insert(0, BrokenWheels())
        for version in (2, 3):
            assert not is_fa_version_supported(version)
            assert 'undefined symbol: ppu_fa_torch_abi' in fa_version_unsupported_reason(version)
            try:
                flash_attn_varlen_func(None, None, None, 1, None, 1, fa_version=version)
            except ImportError as exc:
                assert f'PPU FlashAttention FA{version}' in str(exc)
                assert 'undefined symbol: ppu_fa_torch_abi' in str(exc)
                assert 'PPU SDK' in str(exc)
            else:
                raise AssertionError('unavailable FA must not return a fake result')
        try:
            get_scheduler_metadata(1, 1, 1, 1, 1, 1, None)
        except ImportError as exc:
            assert 'PPU FlashAttention FA3' in str(exc)
        else:
            raise AssertionError('unavailable FA3 metadata must fail')
        import vllm.vllm_flash_attn as package
        assert package.FA2_AVAILABLE is False
        assert package.FA3_AVAILABLE is False
    """)


@pytest.mark.parametrize(
    ("sail", "ppu", "expected"),
    [(None, "1", True), ("0", "1", False), ("", "1", False), ("yes", "0", True)],
)
def test_kernel_nvtx_flags_follow_central_env_precedence(
    run_shim, sail: str | None, ppu: str, expected: bool
) -> None:
    run_shim(f"""
        import importlib
        import os
        import types
        # Import the real kernel module with only annotation/SDK stubs.
        torch = types.ModuleType('torch')
        torch.Tensor = type('Tensor', (), {{}})
        torch.bfloat16 = object()
        sys.modules['torch'] = torch
        for name in ('torch.cuda', 'torch.cuda.nvtx', 'flash_attn',
                     'flash_attn_3', 'flash_attn_3._C'):
            sys.modules[name] = types.ModuleType(name)
        nvtx = sys.modules['torch.cuda.nvtx']
        nvtx.range_push = lambda label: None
        nvtx.range_pop = lambda: None
        os.environ.pop('SAIL_NVTX_PROFILE', None)
        for suffix in ('NVTX_PROFILE', 'NVTX_VFA_DUMP_SEQLEN'):
            canonical = 'VLLM_SAIL_' + suffix
            os.environ.pop(canonical, None)
            if {sail!r} is not None:
                os.environ[canonical] = {sail!r}
            os.environ['VLLM_PPU_' + suffix] = {ppu!r}
        kernels = importlib.import_module('vllm_sail.attention.flash_attn._kernels')
        assert kernels.NVTX_PROFILE is {expected!r}
        assert kernels.NVTX_PROFILE_DUMP_SEQLEN is {expected!r}
    """)


def test_lazy_api_forwards_arguments_and_results_to_the_ppu_implementation(run_shim):
    run_shim("""
        import importlib
        import types
        vllm_sail.register()
        api = importlib.import_module('vllm.vllm_flash_attn.flash_attn_interface')
        impl = types.ModuleType('vllm_sail.attention.flash_attn._kernels')
        sys.modules[impl.__name__] = impl
        calls = []
        result = object()
        def kernel(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        for name in ('flash_attn_varlen_func', 'get_scheduler_metadata',
                     'sparse_attn_func', 'sparse_attn_varlen_func'):
            setattr(impl, name, kernel)
            assert getattr(api, name)(1, 2, out=3) is result
            assert calls[-1] == ((1, 2), {'out': 3})
        impl.is_fa_version_supported = lambda version, device: (version, device) == (3, 0)
        impl.fa_version_unsupported_reason = lambda version, device: 'SDK reason'
        assert api.is_fa_version_supported(3, device=0)
        assert api.fa_version_unsupported_reason(2) == 'SDK reason'
    """)


def test_fa4_probe_does_not_load_sdk(monkeypatch):
    from vllm_sail.attention.flash_attn import flash_attn_interface as interface

    def forbidden():
        pytest.fail("a capability probe for FA4 must not import PPU binaries")

    monkeypatch.setattr(interface, "_kernels", forbidden)
    assert interface.is_fa_version_supported(4) is False
    assert "FA4" in interface.fa_version_unsupported_reason(4)


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/fa_utils",
            "get_flash_attn_version",
            "v1/attention/backends/fa_utils",
            "get_flash_attn_version",
        ),
        (
            "attention/fa_utils",
            "flash_attn_supports_kv_cache_dtype",
            "v1/attention/backends/fa_utils",
            "flash_attn_supports_kv_cache_dtype",
        ),
        (
            "attention/flash_attn",
            "supports_combination",
            "v1/attention/backends/flash_attn",
            "FlashAttentionBackend.supports_combination",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
