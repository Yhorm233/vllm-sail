# SPDX-License-Identifier: Apache-2.0
"""PPU identity must be available before the general-plugin patch lifecycle."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest

from tests.support.source import ROOT, definition, function


@pytest.mark.upstream_source
@pytest.mark.parametrize("enabled", [False, True])
def test_real_hisparse_admission_rejects_ppu_before_other_config_probes(
    upstream_source_root, enabled
):
    """只执行真实配置中的完整 HiSparse 准入区段，不模拟整个配置初始化。"""
    import ast

    node = definition(
        upstream_source_root / "vllm/config/vllm.py", "VllmConfig.__post_init__"
    )
    admission = [
        child
        for child in node.body
        if isinstance(child, ast.If)
        and ast.unparse(child.test)
        == "self.attention_config.hisparse_config is not None"
    ]
    assert len(admission) == 1
    node.name = "admit_hisparse"
    node.body = admission
    node.decorator_list = []
    probes = []
    hook = function(
        ROOT / "vllm_sail/platform.py", "PPUPlatform.get_hisparse_backend_cls", {}
    )

    def backend():
        probes.append("platform")
        return hook(None)

    namespace = {
        "current_platform": types.SimpleNamespace(get_hisparse_backend_cls=backend)
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "hisparse-admission", "exec"),
        namespace,
    )
    config = types.SimpleNamespace(
        attention_config=types.SimpleNamespace(
            hisparse_config=object() if enabled else None
        )
    )
    # 其他配置字段故意缺失：PPU 拒绝必须先于并行/缓存与原生算子检查。
    if enabled:
        with pytest.raises(
            ValueError, match="The current platform does not support HiSparse"
        ):
            namespace["admit_hisparse"](config)
        assert probes == ["platform"]
    else:
        assert namespace["admit_hisparse"](config) is None
        assert probes == []


@pytest.fixture
def platform_classes(monkeypatch):
    cuda_enum = object()

    class NvmlCudaPlatform:
        @classmethod
        def get_hisparse_backend_cls(cls):
            return "vllm.v1.hisparse.backend.CUDAHiSparseBackend"

        @classmethod
        def check_and_update_config(cls, config):
            config.platform_configured = True

        def __getattr__(self, name):
            # vLLM's fallback returns None for unknown platform capabilities.
            return None

        def is_cuda(self):
            return self._enum is cuda_enum

    modules = {
        "torch": types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(enable_cudnn_sdp=lambda value: None)
            )
        ),
        "vllm.logger": types.SimpleNamespace(init_logger=logging.getLogger),
        "vllm.platforms.cuda": types.SimpleNamespace(NvmlCudaPlatform=NvmlCudaPlatform),
        "vllm.platforms.interface": types.SimpleNamespace(
            DeviceCapability=tuple, PlatformEnum=types.SimpleNamespace(CUDA=cuda_enum)
        ),
        "vllm.v1.attention.backends.registry": types.SimpleNamespace(
            AttentionBackendEnum=object
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "vllm_sail/platform.py"
    spec = importlib.util.spec_from_file_location("_ppu_identity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.PPUPlatform, NvmlCudaPlatform


def test_ppu_identity_exists_before_general_patches(platform_classes, monkeypatch):
    PPUPlatform, NvmlCudaPlatform = platform_classes
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    PPUPlatform()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER")
    platform = PPUPlatform()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert platform.is_ppu() is True
    assert PPUPlatform.is_ppu() is True
    assert platform.is_cuda() is True
    assert NvmlCudaPlatform.get_hisparse_backend_cls() is not None
    assert platform.get_hisparse_backend_cls() is None
    assert PPUPlatform.get_hisparse_backend_cls() is None
    # Installing a predicate on the upstream base later must not change PPU
    # identity or become a prerequisite for it.
    NvmlCudaPlatform.is_ppu = lambda self: False
    assert platform.is_ppu() is True


@pytest.mark.parametrize(
    ("use_cumem", "requested", "expected"),
    [
        (False, True, True),
        (False, None, None),
        (True, None, False),
        (True, False, False),
    ],
)
def test_cumem_owns_the_kv_cache(platform_classes, use_cumem, requested, expected):
    platform, _ = platform_classes
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enable_cumem_allocator=use_cumem),
        cache_config=types.SimpleNamespace(enable_extensible_kv_cache=requested),
    )
    platform.check_and_update_config(config)
    assert config.cache_config.enable_extensible_kv_cache is expected
    assert config.platform_configured


def test_cumem_rejects_explicit_extensible_kv_cache(platform_classes):
    platform, _ = platform_classes
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enable_cumem_allocator=True),
        cache_config=types.SimpleNamespace(enable_extensible_kv_cache=True),
    )
    with pytest.raises(ValueError, match="Extensible KV cache bypasses"):
        platform.check_and_update_config(config)
