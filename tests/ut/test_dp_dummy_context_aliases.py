# SPDX-License-Identifier: Apache-2.0
"""Dummy context patches reach consumers imported before plugin installation."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from vllm_sail.patch.utils import PATCH_REGISTRY, original_of

PROVIDER = "vllm.v1.worker.gpu.input_batch"
CONSUMERS = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.cudagraph_utils",
)


@pytest.fixture
def context_modules(monkeypatch):
    path = (
        Path(__file__).parents[2] / "vllm_sail/patch/bugfix/dp_idle_mtp_dummy_batch.py"
    )
    spec = importlib.util.spec_from_file_location("_dp_context_alias_test", path)
    patch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch_module)
    count = len(PATCH_REGISTRY)

    def original(*args, **kwargs):
        raise AssertionError("the original context function must be replaced")

    for name in (
        "vllm",
        "vllm.utils",
        "vllm.utils.math_utils",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        PROVIDER,
        "numpy",
        "torch",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.utils.math_utils"].cdiv = lambda a, b: -(-a // b)
    provider = sys.modules[PROVIDER]
    provider.set_dummy_context = original
    try:
        yield patch_module, provider, original
    finally:
        del PATCH_REGISTRY[count:]


@pytest.mark.parametrize("loaded", [(), CONSUMERS[:1], CONSUMERS[1:], CONSUMERS])
def test_context_patch_rebinds_loaded_consumers(context_modules, monkeypatch, loaded):
    patch_module, provider, original = context_modules
    for name in CONSUMERS:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in loaded:
        consumer = types.ModuleType(name)
        consumer.set_dummy_context = original
        monkeypatch.setitem(sys.modules, name, consumer)

    patch_module._install_set_dummy_context()

    assert provider.set_dummy_context is not original
    assert (
        original_of(provider.set_dummy_context, f"{PROVIDER}.set_dummy_context")
        is original
    )
    expected_targets = {f"{name}.set_dummy_context" for name in (PROVIDER, *loaded)}
    assert {
        record.target for record in PATCH_REGISTRY[-len(expected_targets) :]
    } == expected_targets
    for name in loaded:
        replacement = sys.modules[name].set_dummy_context
        assert replacement is provider.set_dummy_context
        assert original_of(replacement, f"{name}.set_dummy_context") is original
    for name in set(CONSUMERS) - set(loaded):
        assert name not in sys.modules
    from vllm.v1.worker.gpu.input_batch import set_dummy_context

    assert set_dummy_context is provider.set_dummy_context


def test_context_patch_preserves_independently_replaced_consumer(
    context_modules, monkeypatch
):
    patch_module, provider, _ = context_modules
    consumer = types.ModuleType(CONSUMERS[0])

    def independent(*args, **kwargs):
        return None

    consumer.set_dummy_context = independent
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    patch_module._install_set_dummy_context()
    assert consumer.set_dummy_context is independent
    assert provider.set_dummy_context is not independent


@pytest.mark.upstream_source
def test_context_consumer_inventory(upstream_source_root):
    import vllm_sail.patch.bugfix.dp_idle_mtp_dummy_batch as patch_module

    found = set()
    for path in (upstream_source_root / "vllm").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == PROVIDER
                and any(alias.name == "set_dummy_context" for alias in node.names)
            ):
                found.add(
                    ".".join(
                        path.relative_to(upstream_source_root).with_suffix("").parts
                    )
                )
    assert found == set(patch_module._CONTEXT_CONSUMERS)
