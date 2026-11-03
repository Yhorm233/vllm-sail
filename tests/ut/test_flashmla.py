# SPDX-License-Identifier: Apache-2.0
"""FlashMLA patches cover upstream consumers that import operations by value."""

from __future__ import annotations

import ast

import pytest

from tests.support.source import ROOT


@pytest.mark.parametrize(
    "patch_module,upstream_module",
    (
        ("flashmla_backend", "vllm.v1.attention.backends.mla.flashmla"),
        ("flashmla_ops", "vllm.v1.attention.ops.flashmla"),
    ),
)
def test_flashmla_expected_patch_targets_match_declarations(
    patch_module, upstream_module
):
    """Catch stale runtime-test expectations even without torch or vLLM."""
    path = ROOT / "tests/ut/test_attention_patches.py"
    assignment = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_ATTENTION_TARGETS"
    )
    expected = {
        attr
        for module, attr in ast.literal_eval(assignment.value)
        if module == upstream_module
    }
    path = ROOT / f"vllm_sail/patch/enhancement/attention/{patch_module}.py"
    declared = {
        ast.literal_eval(decorator.args[1])
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "patch"
    }
    assert expected == declared


@pytest.mark.upstream_source
def test_flashmla_consumer_inventory_against_source_without_vllm_imports(
    upstream_source_root,
):
    """Keep preloaded consumer coverage checkable on the dependency-free runner."""
    path = ROOT / "vllm_sail/patch/enhancement/attention/flashmla_ops.py"
    assignment = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FLASHMLA_ALIAS_CONSUMERS"
    )
    expected = {
        name: set(consumers)
        for name, consumers in ast.literal_eval(assignment.value).items()
    }
    discovered = {name: set() for name in expected}
    source = upstream_source_root / "vllm"
    for path in source.rglob("*.py"):
        consumer = ".".join(("vllm", *path.relative_to(source).with_suffix("").parts))
        for node in ast.parse(path.read_text()).body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "vllm.v1.attention.ops.flashmla"
            ):
                for alias in node.names:
                    if alias.name in discovered:
                        discovered[alias.name].add(consumer)
    assert discovered == expected
