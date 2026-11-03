# SPDX-License-Identifier: Apache-2.0
"""Regression guards for the DSpark draft loader against the pinned v0.31.0.

``vllm_sail/models/deepseek_v4/dspark.py`` does not subclass-and-override the
upstream loader; it copies the body and rebinds it through ``bind_body``, so
every global the body reads is resolved in
``vllm.models.deepseek_v4.nvidia.dspark``. That makes the port silently
version-coupled in both directions:

* a name that only exists after v0.31.0 (``use_native_mega_moe``) raises
  ``AttributeError`` on the pinned tag, before a single tensor is loaded;
* a step that exists only in the port (the ``context_wkv_proj`` duplication)
  disappears the moment the body is refreshed from upstream, and the failure
  is silent -- the draft model predicts noise and the acceptance rate drops
  to ~0 rather than erroring.

Each test pins one of those, plus the quant routing the duplicated shards
depend on. Everything is static or AST-extracted, so the CPU-only tier keeps
running without vLLM, torch or an accelerator.
"""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.source import ROOT, definition, function

DSPARK_REL = "vllm_sail/models/deepseek_v4/dspark.py"
QUANT_CONFIG_REL = "vllm_sail/models/deepseek_v4/quant_config.py"
UPSTREAM_DSPARK = "vllm/models/deepseek_v4/nvidia/dspark.py"
UPSTREAM_MODEL = "vllm/models/deepseek_v4/nvidia/model.py"
UPSTREAM_QUANT_UTILS = "vllm/model_executor/layers/quantization/utils/quant_utils.py"

DSPARK = ROOT / DSPARK_REL
QUANT_CONFIG = ROOT / QUANT_CONFIG_REL


def _bound_names(node: ast.AST) -> set[str]:
    """Names the body introduces itself: arguments, assignments, imports."""
    bound: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            bound.add(child.id)
        elif isinstance(child, ast.Import | ast.ImportFrom):
            for alias in child.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(child.name)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        elif isinstance(child, ast.Global | ast.Nonlocal):
            bound.update(child.names)
    return bound


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _module_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope when the module is imported."""
    names: set[str] = set()
    for child in tree.body:
        if isinstance(child, ast.Import | ast.ImportFrom):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(child.name)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            names.add(child.target.id)
        elif isinstance(child, ast.If | ast.Try):
            # Version gates and ``if TYPE_CHECKING`` both add module names.
            names |= _module_names(ast.Module(body=child.body, type_ignores=[]))
    return names


def _self_attributes(class_def: ast.ClassDef) -> set[str]:
    """Attributes the class assigns on ``self`` anywhere in its body."""
    attributes: set[str] = set()
    for node in ast.walk(class_def):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attributes.add(target.attr)
    return attributes


@pytest.mark.upstream_source
def test_loader_globals_resolve_in_the_pinned_module(
    upstream_source_root: Path,
) -> None:
    """``bind_body`` resolves the copied body against upstream's namespace.

    A global that the port added but upstream lacks fails at load time with
    ``NameError`` -- after the engine already allocated the draft model.
    """
    body = definition(DSPARK, "load_weights")
    upstream = ast.parse((upstream_source_root / UPSTREAM_DSPARK).read_text())
    available = _module_names(upstream) | _bound_names(body) | set(dir(builtins))
    unresolved = _loaded_names(body) - available
    assert not unresolved, (
        f"{DSPARK_REL}:load_weights reads names absent from {UPSTREAM_DSPARK}: "
        f"{sorted(unresolved)}"
    )


@pytest.mark.upstream_source
def test_loader_reads_only_moe_attributes_the_pin_defines(
    upstream_source_root: Path,
) -> None:
    """``first_layer.ffn.<attr>`` must exist on the pinned ``DeepseekV4MoE``.

    ``use_native_mega_moe`` is a post-v0.31.0 rename of ``use_mega_moe``.
    Reading the new name on the pinned tag aborts the whole load with
    ``AttributeError`` and no draft weights are installed at all.
    """
    body = definition(DSPARK, "load_weights")
    read = {
        node.attr
        for node in ast.walk(body)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "ffn"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "first_layer"
    }
    assert read, "the loader is expected to branch on the draft MoE backend"
    moe = definition(upstream_source_root / UPSTREAM_MODEL, "DeepseekV4MoE")
    missing = read - _self_attributes(moe)
    assert not missing, (
        f"DeepseekV4MoE from {UPSTREAM_MODEL} assigns no {sorted(missing)}; "
        "the port targets the v0.31.0 tag, not a later release"
    )


def test_loader_materialises_the_cross_layer_context_projection() -> None:
    """v0.31.0's ``context_wkv_proj`` has no entry in the checkpoint.

    It is a ``MergedColumnParallelLinear`` filled from every draft layer's
    ``attn.wkv`` shard, so the loader must (a) wrap the weight stream in
    ``_duplicate_context_wkv_weights`` with the draft depth and (b) hand each
    duplicate to the merger's loader together with its ``shard_id``. Without
    (a) the projection keeps its ``torch.empty`` contents. Without (b) the
    default loader overwrites every shard with the last one seen.
    """
    body = definition(DSPARK, "load_weights")
    duplicate = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_duplicate_context_wkv_weights"
    ]
    assert len(duplicate) == 1, "load_weights must wrap the weight stream once"
    args = duplicate[0].args
    assert len(args) == 2, "expected (weights, num_draft_layers)"
    depth = args[1]
    assert (
        isinstance(depth, ast.Call)
        and isinstance(depth.func, ast.Name)
        and depth.func.id == "len"
        and isinstance(depth.args[0], ast.Attribute)
        and depth.args[0].attr == "layers"
    ), "the shard count must be len(self.model.layers)"

    branch = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "startswith"
        and any(
            isinstance(arg, ast.Constant) and arg.value == "model.context_wkv_proj."
            for arg in node.test.args
        )
    ]
    assert len(branch) == 1, "missing the model.context_wkv_proj load branch"
    assert [
        node
        for node in ast.walk(branch[0])
        if isinstance(node, ast.Attribute) and node.attr == "shard_id"
    ], "each duplicate must be routed by loaded_weight.shard_id"


def test_loader_discards_an_unshipped_confidence_head() -> None:
    """A checkpoint without confidence-head weights leaves a random module.

    Acceptance estimation would then consume garbage, so the loader records
    whether the checkpoint supplied the head and drops the module otherwise.
    """
    body = definition(DSPARK, "load_weights")
    observers = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Constant)
        and node.test.left.value == "confidence_head."
        and any(isinstance(op, ast.In) for op in node.test.ops)
    ]
    assert len(observers) == 1, "the loader must observe confidence-head weights"
    assert [
        node
        for node in ast.walk(observers[0])
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    ], "observing a confidence-head weight must set the loaded flag"

    droppers = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and any(
            isinstance(value, ast.Compare)
            and isinstance(value.left, ast.Attribute)
            and value.left.attr == "confidence_head"
            for value in node.test.values
        )
    ]
    assert len(droppers) == 1, "the unused head must be dropped after the loop"
    assert [
        node
        for node in ast.walk(droppers[0])
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        and any(
            isinstance(target, ast.Attribute) and target.attr == "confidence_head"
            for target in node.targets
        )
    ], "the unused head must be set to None"


def _channelwise_match_probe():
    """Compile the channel-wise decision prefix of ``get_quant_method``.

    The full method imports compressed-tensors schemes the CPU-only tier
    cannot load, so only the statements that compute ``matched`` are kept.
    """
    method = definition(QUANT_CONFIG, "DeepseekV4FP8Config.get_quant_method")
    guard = next(
        node
        for node in method.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and any(
            isinstance(value, ast.Attribute) and value.attr == "fp8_channelwise_layers"
            for value in node.test.values
        )
    )
    statements = guard.body
    stop = next(
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "matched"
    )
    probe = ast.FunctionDef(
        name="probe",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="self"),
                ast.arg(arg="layer"),
                ast.arg(arg="prefix"),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            *statements[:stop],
            ast.Return(value=ast.Name(id="matched", ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    namespace: dict[str, object] = {"LinearBase": object}
    module = ast.Module(body=[probe], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(QUANT_CONFIG), "exec"),
        namespace,
    )
    return namespace["probe"]


@pytest.mark.upstream_source
def test_context_wkv_proj_routes_to_the_channelwise_scheme(
    upstream_source_root: Path,
) -> None:
    """The projection's shards are channel-wise, so its scheme must be too.

    ``model.context_wkv_proj`` never matches the checkpoint's ``...attn.wkv``
    entries, and ``configure_quant_config`` skips ``apply_vllm_mapper`` for
    ``SupportsQuant`` models, so without an explicit name check the layer
    falls back to per-tensor FP8. Its ``PerTensorScaleParameter`` then
    rejects the ``[head_dim, 1]`` shard with
    ``assert loaded_weight.shape[0] == 1``.
    """
    matcher = function(
        upstream_source_root / UPSTREAM_QUANT_UTILS,
        "is_layer_skipped",
        {"MappingProxyType": dict},
    )
    probe = _channelwise_match_probe()
    probe.__globals__["is_layer_skipped"] = matcher
    config = SimpleNamespace(
        ignored_layers=[],
        fp8_channelwise_layers=["attn.wq_a", "attn.wkv"],
        packed_modules_mapping={"fused_wqa_wkv": ["wq_a", "wkv"]},
    )
    assert probe(config, object(), "model.context_wkv_proj") is True
    # An explicit ignore must take precedence over the projection override.
    unquantized_method = object()
    probe.__globals__["UnquantizedLinearMethod"] = lambda: unquantized_method
    config.ignored_layers = ["context_wkv_proj"]
    assert probe(config, object(), "model.context_wkv_proj") is unquantized_method
    config.ignored_layers = []
    # A genuine match, not an unconditional True: unrelated layers stay on
    # the default path.
    config.fp8_channelwise_layers = ["attn.unrelated"]
    assert not probe(config, object(), "model.layers.0.attn.fused_wqa_wkv")
    assert not probe(config, object(), "model.layers.0.ffn.shared_experts.w2")


@pytest.mark.upstream_source
def test_duplicate_context_wkv_weights_emits_one_shard_per_layer(
    upstream_source_root: Path,
) -> None:
    """Contract of the upstream helper the loader relies on.

    The original stream passes through untouched; each in-range
    ``mtp.<n>.attn.wkv.<suffix>`` entry additionally yields a
    ``context_wkv_proj.<suffix>`` duplicate tagged with ``shard_id=n``.
    Out-of-range layers and every other ``mtp.<n>.*`` weight yield nothing.
    """
    upstream = ast.parse((upstream_source_root / UPSTREAM_DSPARK).read_text())
    pattern = next(
        node.value.args[0].value
        for node in upstream.body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "_CONTEXT_WKV_RE" for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.args[0], ast.Constant)
    )
    duplicate = function(
        upstream_source_root / UPSTREAM_DSPARK,
        "_duplicate_context_wkv_weights",
        {"_CONTEXT_WKV_RE": re.compile(pattern)},
    )

    class _Tensor:
        """Stand-in for ``torch.Tensor``; only ``detach`` plus attribute
        assignment are needed, and the tier must not import torch."""

        def __init__(self, name: str) -> None:
            self.name = name

        def detach(self) -> _Tensor:
            return _Tensor(self.name)

    stream = [
        ("mtp.0.attn.wkv.weight", _Tensor("w0")),
        ("mtp.1.attn.wkv.weight", _Tensor("w1")),
        ("mtp.2.attn.wkv.weight", _Tensor("w2")),
        ("mtp.0.attn.wq_a.weight", _Tensor("q0")),
        ("mtp.0.confidence_head.weight", _Tensor("c0")),
    ]
    emitted = list(duplicate(iter(stream), 2))
    assert [name for name, _ in emitted] == [
        "mtp.0.attn.wkv.weight",
        "context_wkv_proj.weight",
        "mtp.1.attn.wkv.weight",
        "context_wkv_proj.weight",
        "mtp.2.attn.wkv.weight",
        "mtp.0.attn.wq_a.weight",
        "mtp.0.confidence_head.weight",
    ]
    shards = [
        weight.shard_id
        for name, weight in emitted
        if name.startswith("context_wkv_proj.")
    ]
    assert shards == [0, 1]
