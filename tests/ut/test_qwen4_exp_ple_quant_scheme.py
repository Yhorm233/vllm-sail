# SPDX-License-Identifier: Apache-2.0
"""Tests for the PLE ple_quant_scheme bridge (bugfix/qwen4_exp_ple_quant_scheme).

Two-tier design like ``test_mxfp4_patches.py``:

* **Bare-CPU import + metadata shape.** The leaf imports no vLLM at module
  scope; ``METADATA`` and every ``patch`` call carry the mandatory fields.
* **Behaviour against stubs.** ``install()`` runs against throwaway stand-ins
  of the upstream ngram module (no torch needed), asserting the dispatch
  priority matches the checkpoint contract.  The config half of the bridge
  (``ple_quant_scheme`` attached in ``from_config``) lives in
  ``enhancement/mxfp4.py`` and is asserted in ``test_mxfp4_patches.py`` plus
  the combined install test below.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.patch_inventory import scan_patch_file
from tests.support.source import function
from tests.support.upstream import SUPPORTED_RANGE
from vllm_sail.patch.utils import PATCH_REGISTRY, original_of, patch

MODULE_PATH = (
    Path(__file__).parents[2]
    / "vllm_sail"
    / "patch"
    / "bugfix"
    / "qwen4_exp_ple_quant_scheme.py"
)
ENHANCEMENT_PATH = (
    Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "mxfp4.py"
)
NGRAM_TARGET = (
    "vllm.models.qwen4_exp.common.ngram_embedding."
    "Qwen4ExpPLEEmbeddingMethod.from_quant_config"
)
REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")


def _load_leaf(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ple_module() -> Iterator[types.ModuleType]:
    """A fresh leaf-module instance per test (install state must not leak)."""
    name = f"_ple_scheme_leaf_test_{id(object())}"
    module = _load_leaf(MODULE_PATH, name)
    registry_before = len(PATCH_REGISTRY)
    try:
        yield module
    finally:
        del PATCH_REGISTRY[registry_before:]
        sys.modules.pop(name, None)


def _register_chain(
    monkeypatch: pytest.MonkeyPatch, leaf_name: str, leaf_module: types.ModuleType
) -> None:
    parts = leaf_name.split(".")
    chain = [".".join(parts[: i + 1]) for i in range(len(parts))]
    modules: dict[str, types.ModuleType] = {}
    for name in chain[:-1]:
        existing = sys.modules.get(name)
        module = (
            existing
            if isinstance(existing, types.ModuleType)
            else types.ModuleType(name)
        )
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules[leaf_name] = leaf_module
    monkeypatch.setitem(sys.modules, leaf_name, leaf_module)
    for parent_name, child_name in zip(chain, chain[1:], strict=False):
        parent = modules[parent_name]
        attribute = child_name.rpartition(".")[2]
        # Force the binding: with real vLLM already imported, a from-import
        # resolves through the parent's attribute and would bypass the stub
        # leaf in sys.modules entirely.
        monkeypatch.setattr(parent, attribute, modules[child_name], raising=False)


class _StubState:
    def __init__(self) -> None:
        self.upstream_dispatch: list = []
        self.warnings: list = []


def _stub_logger(monkeypatch: pytest.MonkeyPatch, state: _StubState) -> None:
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: NS(  # type: ignore[attr-defined]
        warning=lambda message, *args: state.warnings.append(message % args),
        info=lambda *args, **kwargs: None,
    )
    _register_chain(monkeypatch, "vllm.logger", logger_mod)


def _stub_ngram(monkeypatch: pytest.MonkeyPatch, state: _StubState) -> types.ModuleType:
    class Qwen4ExpPLEUnquantizedEmbeddingMethod:
        pass

    class Qwen4ExpPLEFp8EmbeddingMethod:
        pass

    class Qwen4ExpPLENvFp4EmbeddingMethod:
        pass

    class Qwen4ExpPLEEmbeddingMethod:
        @staticmethod
        def from_quant_config(quant_config, prefix, embedding_dtype=None):
            state.upstream_dispatch.append((quant_config, prefix, embedding_dtype))
            if embedding_dtype == "nvfp4":
                return Qwen4ExpPLENvFp4EmbeddingMethod()
            if embedding_dtype == "float8_e4m3fn":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            if embedding_dtype is not None:
                raise ValueError(f"bad embedding_dtype {embedding_dtype!r}")
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()

    ngram = types.ModuleType("vllm.models.qwen4_exp.common.ngram_embedding")
    ngram.Qwen4ExpPLEEmbeddingMethod = Qwen4ExpPLEEmbeddingMethod  # type: ignore[attr-defined]
    ngram.Qwen4ExpPLEUnquantizedEmbeddingMethod = (  # type: ignore[attr-defined]
        Qwen4ExpPLEUnquantizedEmbeddingMethod
    )
    ngram.Qwen4ExpPLEFp8EmbeddingMethod = Qwen4ExpPLEFp8EmbeddingMethod  # type: ignore[attr-defined]
    ngram.Qwen4ExpPLENvFp4EmbeddingMethod = (  # type: ignore[attr-defined]
        Qwen4ExpPLENvFp4EmbeddingMethod
    )
    _register_chain(monkeypatch, "vllm.models.qwen4_exp.common.ngram_embedding", ngram)
    return ngram


def _build_stub_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_StubState, types.ModuleType]:
    state = _StubState()
    _stub_logger(monkeypatch, state)
    return state, _stub_ngram(monkeypatch, state)


# ---------------------------------------------------------------------------
# Bare-CPU import and metadata shape
# ---------------------------------------------------------------------------


def test_module_imports_on_bare_cpu(ple_module) -> None:
    if "vllm" in sys.modules:
        pytest.skip("bare-CPU contract test; real vLLM is loaded (mode B)")
    assert callable(ple_module.install)
    assert ple_module.__doc__


def test_metadata_shape(ple_module) -> None:
    assert ple_module.METADATA == (
        (
            NGRAM_TARGET,
            ple_module.REASON_DISPATCH,
            SUPPORTED_RANGE,
            ple_module.REMOVE_WHEN,
        ),
    )
    for target, reason, affected, remove_when in ple_module.METADATA:
        assert target.startswith("vllm."), target
        for value in (reason, affected, remove_when):
            assert isinstance(value, str) and value.strip(), target
        assert ">=" in affected and "<" in affected, target


def test_no_top_level_vllm_import() -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for imported in names:
            assert imported != "vllm" and not imported.startswith("vllm."), (
                f"the leaf imports {imported!r} at module level; it must stay "
                "importable without vLLM."
            )


def test_every_patch_call_carries_metadata() -> None:
    tree = ast.parse(MODULE_PATH.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "patch"
    ]
    assert len(calls) == 1
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        missing = set(REQUIRED_METADATA) - keywords
        assert not missing, f"patch call at line {call.lineno} lacks {missing}"


def test_static_discovery_resolves_the_target() -> None:
    hooks, unresolved = scan_patch_file(MODULE_PATH)
    assert not unresolved
    assert [hook.target for hook in hooks] == [NGRAM_TARGET]
    assert all(hook.node is not None and not hook.additive for hook in hooks)


# ---------------------------------------------------------------------------
# Behaviour against stubs
# ---------------------------------------------------------------------------


def test_install_lands_target_and_reapplication_guard(
    monkeypatch: pytest.MonkeyPatch, ple_module
) -> None:
    _, ngram = _build_stub_vllm(monkeypatch)
    method_cls = ngram.Qwen4ExpPLEEmbeddingMethod
    original_dispatch = inspect.getattr_static(method_cls, "from_quant_config")

    registry_before = len(PATCH_REGISTRY)
    ple_module.install()
    records = PATCH_REGISTRY[registry_before:]
    assert [record.target for record in records] == [NGRAM_TARGET]
    assert all(record.was_missing is False for record in records)
    assert all(record.affected_versions == SUPPORTED_RANGE for record in records)
    assert all(record.reason and record.remove_when for record in records)

    dispatch_replacement = method_cls.from_quant_config
    assert dispatch_replacement is not original_dispatch
    assert original_of(dispatch_replacement, NGRAM_TARGET) is original_dispatch

    ple_module.install()  # the guard, not the double-patch error
    assert len(PATCH_REGISTRY) == registry_before + 1
    with pytest.raises(RuntimeError, match="already patched"):
        patch(
            "vllm.models.qwen4_exp.common.ngram_embedding",
            "Qwen4ExpPLEEmbeddingMethod.from_quant_config",
            reason="test",
            affected_versions="test",
            remove_when="test",
        )(method_cls.from_quant_config)


def test_dispatch_honors_scheme_with_dtype_priority(
    monkeypatch: pytest.MonkeyPatch, ple_module
) -> None:
    state, ngram = _build_stub_vllm(monkeypatch)
    ple_module.install()
    method_cls = ngram.Qwen4ExpPLEEmbeddingMethod
    fp8_cls = ngram.Qwen4ExpPLEFp8EmbeddingMethod
    nvfp4_cls = ngram.Qwen4ExpPLENvFp4EmbeddingMethod
    unquantized_cls = ngram.Qwen4ExpPLEUnquantizedEmbeddingMethod
    quant_config = NS(ple_quant_scheme="fp8_pertensor")

    method = method_cls.from_quant_config(quant_config, "model.ple")
    assert type(method) is fp8_cls
    assert state.upstream_dispatch == []  # the scheme short-circuits inference

    # An explicit embedding_dtype still wins over the declared scheme.
    state.upstream_dispatch.clear()
    method = method_cls.from_quant_config(
        quant_config, "model.ple", embedding_dtype="nvfp4"
    )
    assert type(method) is nvfp4_cls
    assert state.upstream_dispatch == [(quant_config, "model.ple", "nvfp4")]
    method = method_cls.from_quant_config(
        quant_config, "model.ple", embedding_dtype="float8_e4m3fn"
    )
    assert type(method) is fp8_cls
    with pytest.raises(ValueError, match="embedding_dtype"):
        method_cls.from_quant_config(quant_config, "model.ple", embedding_dtype="int8")

    # Without a declared scheme upstream inference runs unchanged.
    state.upstream_dispatch.clear()
    for config in (None, object(), NS()):
        method = method_cls.from_quant_config(config, "model.ple")
        assert type(method) is unquantized_cls
    assert len(state.upstream_dispatch) == 3

    # An unsupported scheme fails loudly instead of silently misloading.
    state.upstream_dispatch.clear()
    with pytest.raises(ValueError, match="ple_quant_scheme"):
        method_cls.from_quant_config(NS(ple_quant_scheme="fp8_perchannel"), "model.ple")
    assert state.upstream_dispatch == []
    # ... unless an explicit embedding_dtype takes priority.
    method = method_cls.from_quant_config(
        NS(ple_quant_scheme="fp8_perchannel"),
        "model.ple",
        embedding_dtype="float8_e4m3fn",
    )
    assert type(method) is fp8_cls


def test_install_skips_when_upstream_modules_are_missing(
    monkeypatch: pytest.MonkeyPatch, ple_module
) -> None:
    state = _StubState()
    _stub_logger(monkeypatch, state)
    # No ngram stub chain: install() probes via importlib.import_module, which
    # consults sys.modules directly, where a None entry raises.
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.common.ngram_embedding", None
    )
    registry_before = len(PATCH_REGISTRY)
    ple_module.install()
    assert ple_module._unavailable is True
    assert state.warnings and "ple_quant_scheme" in state.warnings[0]
    assert len(PATCH_REGISTRY) == registry_before
    ple_module.install()
    assert len(PATCH_REGISTRY) == registry_before


def test_install_skips_even_when_the_parent_attribute_is_bound(
    monkeypatch: pytest.MonkeyPatch, ple_module
) -> None:
    # Mode B regression: with real vLLM imported earlier in the session, the
    # parent package carries a bound ngram_embedding attribute. A from-import
    # probe would resolve through it and bypass the None entry in sys.modules,
    # then patch()'s importlib.import_module would raise uncaught.
    state = _StubState()
    _stub_logger(monkeypatch, state)
    parent = types.ModuleType("vllm.models.qwen4_exp.common")
    parent.ngram_embedding = types.ModuleType(  # type: ignore[attr-defined]
        "vllm.models.qwen4_exp.common.ngram_embedding"
    )
    _register_chain(monkeypatch, "vllm.models.qwen4_exp.common", parent)
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.common.ngram_embedding", None
    )
    registry_before = len(PATCH_REGISTRY)
    ple_module.install()
    assert ple_module._unavailable is True
    assert state.warnings and "ple_quant_scheme" in state.warnings[0]
    assert len(PATCH_REGISTRY) == registry_before


@pytest.mark.requires_runtime
def test_scheme_survives_the_enhancement_from_config(
    monkeypatch: pytest.MonkeyPatch, ple_module
) -> None:
    """The two installers together must load an FP8-PLE MXFP4 checkpoint.

    Reuses the enhancement test module's full stub builder (torch-gated) so
    the two installers run against the same stand-ins they use in their own
    suites: enhancement attaches the scheme in from_config, the bugfix
    dispatch consumes it.
    """
    helper = _load_leaf(
        Path(__file__).parent / "test_mxfp4_patches.py",
        f"_mxfp4_test_helper_{id(object())}",
    )
    try:
        _, modules = helper._build_stub_vllm(monkeypatch)
        state = _StubState()
        _stub_logger(monkeypatch, state)
        ngram = _stub_ngram(monkeypatch, state)
        enhancement = _load_leaf(
            ENHANCEMENT_PATH, f"_mxfp4_enh_leaf_test_{id(object())}"
        )
        try:
            ple_module.install()
            enhancement.install()
            config_cls = modules["quant_mxfp4"].Mxfp4Config
            helper._Platform.ppu = True
            config = config_cls.from_config(
                {
                    "ignore": ["mtp"],
                    "fp8_channelwise_layers": ["gate"],
                    "ple_quant_scheme": "fp8_pertensor",
                }
            )
            assert config.ple_quant_scheme == "fp8_pertensor"
            assert config.ignored_layers == ["mtp"]
            assert config.fp8_channelwise_layers == ["gate"]

            method_cls = ngram.Qwen4ExpPLEEmbeddingMethod
            method = method_cls.from_quant_config(config, "model.ple")
            assert type(method).__name__ == "Qwen4ExpPLEFp8EmbeddingMethod"
        finally:
            sys.modules.pop(enhancement.__name__, None)
    finally:
        sys.modules.pop(helper.__name__, None)


# ---------------------------------------------------------------------------
# Pinned-source contracts
# ---------------------------------------------------------------------------


@pytest.mark.upstream_source
def test_pinned_targets_and_bindings_fit(upstream_source_root) -> None:
    contracts = _load_leaf(
        Path(__file__).parent / "test_patch_contracts.py",
        f"_patch_contracts_helper_{id(object())}",
    )
    try:
        hooks, unresolved = scan_patch_file(MODULE_PATH)
        assert not unresolved
        assert not contracts.target_problems(upstream_source_root, hooks)
    finally:
        sys.modules.pop(contracts.__name__, None)


@pytest.mark.upstream_source
def test_pinned_upstream_drops_the_scheme(upstream_source_root) -> None:
    """Document the defect against the pinned source: both drops are real."""
    namespace: dict = {}
    from_config = function(
        upstream_source_root / "vllm/model_executor/layers/quantization/mxfp4.py",
        "Mxfp4Config.from_config",
        namespace,
    )

    class Config:
        def __init__(self):
            pass

    instance = from_config(Config, {"ple_quant_scheme": "fp8_pertensor"})
    assert not hasattr(instance, "ple_quant_scheme")

    for name in (
        "Qwen4ExpPLEUnquantizedEmbeddingMethod",
        "Qwen4ExpPLEFp8EmbeddingMethod",
        "Qwen4ExpPLENvFp4EmbeddingMethod",
        "ModelOptQuantConfigBase",
        "ModelOptMixedPrecisionConfig",
        "CompressedTensorsConfig",
        "INCConfig",
        "QuarkConfig",
        "Fp8Config",
    ):
        namespace[name] = type(name, (), {})
    namespace["is_layer_skipped"] = lambda *args, **kwargs: False
    namespace["logger"] = NS(warning=lambda *args, **kwargs: None)
    dispatch = function(
        upstream_source_root / "vllm/models/qwen4_exp/common/ngram_embedding.py",
        "Qwen4ExpPLEEmbeddingMethod.from_quant_config",
        namespace,
    )
    method = dispatch(NS(ple_quant_scheme="fp8_pertensor"), "model.ple")
    assert type(method) is namespace["Qwen4ExpPLEUnquantizedEmbeddingMethod"]
