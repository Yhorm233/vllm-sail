# SPDX-License-Identifier: Apache-2.0
"""Keep speculative sampling consumers on the platform-specific dispatcher."""

from __future__ import annotations

import ast
import importlib.util
from types import SimpleNamespace

import pytest

from tests.support.source import function

PROVIDER = "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils"
CONSUMERS = (
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    "vllm.v1.watermarking.spec_decode",
)


@pytest.mark.upstream_source
def test_consumer_inventory_matches_all_module_scope_imports(upstream_source_root):
    discovered = set()
    for path in (upstream_source_root / "vllm").rglob("*.py"):
        relative = path.relative_to(upstream_source_root).with_suffix("")
        parts = list(relative.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module_name = ".".join(parts)
        package = module_name if is_package else module_name.rpartition(".")[0]
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(node, ast.ImportFrom):
                imported = node.module or ""
                if node.level:
                    imported = importlib.util.resolve_name(
                        "." * node.level + imported, package
                    )
                if imported == PROVIDER:
                    for alias in node.names:
                        if alias.name in {"rejection_sample", "*"}:
                            discovered.add(
                                (module_name, alias.asname or "rejection_sample")
                            )
            pending.extend(ast.iter_child_nodes(node))

    assert discovered == {(name, "rejection_sample") for name in CONSUMERS}


@pytest.mark.upstream_source
def test_real_watermark_wrapper_preserves_sampling_inputs(upstream_source_root):
    watermarker = SimpleNamespace(context_width=3)

    def resolve_key(actual):
        assert actual is watermarker
        return 123

    wrapper = function(
        upstream_source_root / "vllm/v1/watermarking/spec_decode.py",
        "watermarked_rejection_sample",
        dict(
            rejection_sample=lambda *args, **kwargs: (args, kwargs),
            _resolve_watermark_key=resolve_key,
        ),
    )
    args = [SimpleNamespace(shape=(2, 32)), *(object() for _ in range(9)), 7]
    contexts = SimpleNamespace(shape=(2, 3))
    watermarking, skip = object(), object()
    result = wrapper(
        *args,
        contexts,
        watermarking,
        watermarker,
        watermarking_skip_mask=skip,
        use_fp64=True,
    )

    assert result == (
        tuple(args),
        dict(
            use_fp64=True,
            contexts=contexts,
            watermarking=watermarking,
            watermarking_skip_mask=skip,
            watermark_key=123,
        ),
    )
