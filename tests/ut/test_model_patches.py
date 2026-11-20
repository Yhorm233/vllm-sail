# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-5 model patch modules.

Two tiers, following the suite's established split:

* Bare CPU (no vLLM, no torch required): every module in
  ``vllm_sail/patch/enhancement/models/`` is parsed with ``ast`` and must
  carry well-formed mandatory metadata (``reason`` / ``affected_versions``
  / ``remove_when``) on every ``@patch`` call. The patch modules themselves
  import their vLLM targets at import time — that is the framework's
  convention (see ``custom_op_dispatch.py``) — so actually importing them
  is gated on vLLM being installed, exactly like ``test_patch_install.py``.
* With vLLM: importing the subpackage must apply every patch, land each
  replacement on its target with the marker recording the original, and
  record complete metadata in ``PATCH_REGISTRY``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pathlib
import py_compile
import sys

import pytest

from tests.support.source import assert_accepts_upstream_keywords

MODELS_PATCH_DIR = (
    pathlib.Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "models"
)
REPO_ROOT = pathlib.Path(__file__).parents[2]

EXPECTED_MODULES = {
    "deepseek_mtp",
    "deepseek_v2",
    "deepseek_v4",
    "deepseek_v4_cache",
    "deepseek_v4_metadata",
    "deepseek_v4_compressor",
    "deepseek_v4_graph_warmup",
    "deepseek_v41",
    "deepseek_v41_engram",
    "deepseek_v41_indexer",
    "deepseek_v41_int8",
    "deepseek_v41_mega_mhc",
    "glm5_next",
    "glm5_next_indexer",
    "glm5_next_mtp",
    "kimi_k3",
    "llama4",
    "minimax_m3",
    "minimax_m3_msa",
    "moe_marlin_gate",
    "qwen3_fused_quant",
    "qwen3_moe",
    "qwen3_next",
    "qwen4_exp_qsa",
    "qwen4_exp_qsa_fp4",
    "step3p5_mtp",
    "step5_dsa_backend",
    "step5_indexer_fp16",
    "step5_mtp",
    "step5_norm",
    "step5_soft_fp8",
}

REQUIRED_METADATA = ("reason", "affected_versions", "remove_when")


def _patch_module_files() -> list[pathlib.Path]:
    return sorted(
        path for path in MODELS_PATCH_DIR.glob("*.py") if path.name != "__init__.py"
    )


def _patch_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "patch":
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == "patch":
            calls.append(node)
    return calls


def _string_value(node: ast.expr, constants: dict[str, str]) -> str:
    """Constant string, a module-level constant name, or a concatenation."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    if isinstance(node, ast.Name):
        return constants.get(node.id, "")
    if isinstance(node, ast.JoinedStr):
        return "<f-string>"
    parts = [_string_value(part, constants) for part in getattr(node, "values", [])]
    return "".join(parts)


def _module_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    constants[target.id] = node.value.value
    return constants


def test_all_expected_patch_modules_exist() -> None:
    on_disk = {path.stem for path in _patch_module_files()}
    assert on_disk == EXPECTED_MODULES, (
        "Model patch modules are the audited port of the fork's modified model "
        "files, the Marlin gate, the Kimi K3 latent-MoE tail and the Qwen4Exp QSA "
        "FA3 dispatch; a missing or extra module means that inventory drifted."
    )


def test_patch_modules_compile_on_bare_cpu() -> None:
    for path in _patch_module_files():
        py_compile.compile(str(path), doraise=True)
    py_compile.compile(str(MODELS_PATCH_DIR / "__init__.py"), doraise=True)


def test_every_patch_has_well_formed_metadata() -> None:
    """AST-level metadata check: runs with no vLLM and no torch."""
    for path in _patch_module_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        constants = _module_constants(tree)
        calls = _patch_calls(tree)
        assert calls, f"{path.name} contains no @patch call"
        for call in calls:
            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            for field in REQUIRED_METADATA:
                assert field in keywords, f"{path.name}: patch missing {field}"
                value = _string_value(keywords[field], constants)
                assert value.strip(), f"{path.name}: patch {field} is empty"
            remove_when = _string_value(keywords["remove_when"], constants)
            assert remove_when.strip().lower() != "todo", (
                f"{path.name}: remove_when must be a verifiable condition"
            )


def _plugin_source(dotted: str) -> pathlib.Path | None:
    """The file that defines plugin module ``dotted``, if it exists in-tree."""
    path = REPO_ROOT.joinpath(*dotted.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _top_level_names(path: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update(
                (alias.asname or alias.name).split(".")[0] for alias in node.names
            )
    return names


def test_plugin_imports_and_targets_exist_in_tree() -> None:
    """Every ``vllm_sail`` module a patch module imports or targets is in-tree.

    The modules import their targets eagerly and ``install()`` has no fallback,
    so a dangling plugin path breaks plugin load for every model on PPU. Only
    a vLLM run would notice; this AST check catches it on a bare CPU.
    """
    for path in _patch_module_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        constants = _module_constants(tree)
        modules = {
            _string_value(call.args[0], constants)
            for call in _patch_calls(tree)
            if call.args
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules.add(node.module)
                source = _plugin_source(node.module)
                if source is None or source.name == "__init__.py":
                    continue
                # A plain module has no submodules: every name must be defined.
                missing = {alias.name for alias in node.names} - _top_level_names(
                    source
                )
                assert not missing, (
                    f"{path.name}: {node.module} defines no {sorted(missing)}"
                )
        for module in sorted(m for m in modules if m.startswith("vllm_sail.")):
            assert _plugin_source(module) is not None, (
                f"{path.name}: {module} does not exist in the source tree"
            )


def test_subpackage_wiring_is_present() -> None:
    """The parent enhancement package imports models/.

    Guard against a silent behaviour change: the one-line wiring in
    ``vllm_sail/patch/enhancement/__init__.py`` is what makes
    ``vllm_sail.patch.install()`` apply the model patches, so deleting it must
    fail loudly here, not pass unnoticed.
    """
    parent = MODELS_PATCH_DIR.parent / "__init__.py"
    text = parent.read_text()
    assert "enhancement.models" in text or "import models" in text, (
        "models/ patches are no longer wired into the parent package; "
        "restore `from vllm_sail.patch.enhancement import models`."
    )


@pytest.mark.requires_runtime
def test_full_package_chain_on_clean_upstream() -> None:
    """The full enhancement chain installs on a clean upstream vLLM.

    Regression guard for a former known issue: on clean upstream,
    ``SparseAttnIndexer`` subclasses ``CustomOp`` and inherits the
    ``forward_ppu`` attribute installed by the Phase-2 ``custom_op_dispatch``
    patch, which used to trip the double-patch guard of the Phase-4 additive
    ``SparseAttnIndexer.forward_ppu`` patch. The patch framework now shadows
    additive patches into the subclass namespace instead (see
    ``test_patch_framework.py``), so the chain must import cleanly.
    """
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    module = importlib.import_module("vllm_sail.patch.enhancement.models")
    assert module is not None


# ---------------------------------------------------------------------------
# Everything below needs a real vLLM to patch against.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def installed_model_patches():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    # Load each models/ patch module in isolation instead of importing the
    # package: the parent enhancement/__init__ chain installs many more
    # patches, and loading in isolation keeps this fixture independent of
    # unrelated patch failures. (The former clean-upstream forward_ppu
    # collision that also motivated this is fixed in the patch framework.)
    import uuid

    from vllm_sail.patch.utils import PATCH_REGISTRY

    # If the full enhancement chain already ran in this process (fork mode:
    # test_full_package_chain_on_clean_upstream imported it successfully),
    # the targets are already patched; reuse those records instead of
    # re-applying, which the double-patch guard would reject.
    existing = [r for r in PATCH_REGISTRY if r.target in _EXPECTED_TARGETS]
    if len(existing) >= len(_EXPECTED_TARGETS):
        return existing

    registry_before = len(PATCH_REGISTRY)
    loaded = []
    for path in _patch_module_files():
        name = f"vllm_sail_model_patch_test_{path.stem}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            # A missing third-party dep of a vLLM model module (e.g.
            # compressed_tensors on a bare source checkout) is an
            # environment property; a broken plugin import is not.
            if (exc.name or "").startswith(("vllm_sail", "vllm")):
                raise
            pytest.skip(f"vLLM model modules need an unavailable dependency: {exc}")
        loaded.append(module)

    records = [
        r for r in PATCH_REGISTRY[registry_before:] if r.target in _EXPECTED_TARGETS
    ]
    assert records, "no model patch records were installed"
    return records


_EXPECTED_TARGETS = {
    "vllm.model_executor.models.deepseek_v2.Indexer.__init__",
    "vllm.model_executor.models.deepseek_v2.Indexer.forward",
    "vllm.model_executor.models.deepseek_v2.DeepseekV2Model.__init__",
    "vllm.model_executor.models.deepseek_v2.DeepseekV2Model.load_weights",
    "vllm.model_executor.models.deepseek_mtp.DeepSeekMTP.__init__",
    "vllm.model_executor.models.deepseek_mtp.DeepSeekMTP.load_weights",
    "vllm.models.deepseek_v4.nvidia.model._select_dsv4_attn_cls",
    (
        "vllm.models.kimi_k3.nvidia.latent_moe_runner.LatentMoERunner."
        "_overlap_allreduce_tail"
    ),
    ("vllm.models.kimi_k3.nvidia.latent_moe_runner.LatentMoERunner._select_tail_tier"),
    "vllm.models.kimi_k3.nvidia.latent_moe_runner.LatentMoERunner.__init__",
    "vllm.models.kimi_k3.nvidia.model.KimiRoutedOutputTransform.forward",
    "vllm.model_executor.models.llama4.Llama4MoE.custom_routing_function",
    "vllm.models.minimax_m3.nvidia.model.MiniMaxM3Model._init_fused_moe_finalize",
    "vllm.models.minimax_m3.nvidia.model.MiniMaxM3MLP.__init__",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock.__init__",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer.__init__",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeModel.__init__",
    "vllm.model_executor.models.qwen3_next.th_nvtx_range_push",
    "vllm.model_executor.models.qwen3_next.th_nvtx_range_pop",
    "vllm.models.qwen4_exp.nvidia.qsa.Qwen4ExpQSAImpl.__init__",
    "vllm.models.qwen4_exp.nvidia.qsa.Qwen4ExpQSAImpl.forward",
    "vllm.model_executor.models.step3p5_mtp.Step3p5MTP.load_weights",
    "vllm.models.qwen4_exp.common.qsa_indexer.QSACompressedKeyCache.__init__",
    (
        "vllm.models.qwen4_exp.common.qsa_indexer."
        "QSAIndexerBackend.get_supported_kernel_block_sizes"
    ),
    "vllm.models.qwen4_exp.nvidia.ops.qsa_prepare.qsa_prepare",
    "vllm.models.qwen4_exp.nvidia.qsa_indexer.qsa_prepare",
    "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer.qsa_select_paged_decode",
    "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer.qsa_select_paged_prefill",
    "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer.warmup_qsa_mqa_paged_decode",
    (
        "vllm.models.step5.nvidia.sparse_attention.Step5DSAAttentionBackend."
        "supports_compute_capability"
    ),
    "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.__init__",
    (
        "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse."
        "warmup_prefill_scoring"
    ),
    (
        "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse."
        "prefill_pair_inputs"
    ),
    ("vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.select_regions"),
    "vllm.models.step5.nvidia.layernorm.apply_step5_rms_norm",
    "vllm.models.step5.nvidia.layernorm.apply_step5_add_rms_norm",
    "vllm.models.step5.nvidia.ops.summary_update._store_summary",
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope._indexer_norm_rope",
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope._joint_prepare",
    "vllm.models.step5.nvidia.ops.indexer_norm_rope._indexer_norm_rope",
    "vllm.models.step5.nvidia.ops.sparse_indexer_scores._direct_scores",
    "vllm.models.step5.nvidia.ops.sparse_indexer_scores._paged_scores_sm90",
    "vllm.models.step5.nvidia.ops.sparse_indexer._copy_summary_page",
    "vllm.models.step5.nvidia.ops.sparse_indexer._copy_summary_tile",
    "vllm.models.step5.nvidia.ops.sparse_indexer._pack_summaries",
    "vllm.models.step5.nvidia.ops.indexer_norm_rope.indexer_norm_rope",
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope.qkv_indexer_norm_rope",
    (
        "vllm.models.step5.nvidia.sparse_summary_cache."
        "Step5SparseSummaryCacheLayer.get_kv_cache_spec"
    ),
    "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.decode_logits",
    "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.logits",
    "vllm.models.step5.nvidia.mtp.Step5MTP.__init__",
    (
        "vllm.model_executor.layers.quantization.utils.marlin_utils."
        "check_moe_marlin_supports_config"
    ),
}


@pytest.mark.requires_runtime
def test_all_model_patches_land(installed_model_patches) -> None:
    from vllm_sail.patch.utils import PATCH_MARKER

    targets = {record.target for record in installed_model_patches}
    assert _EXPECTED_TARGETS <= targets, _EXPECTED_TARGETS - targets

    for record in installed_model_patches:
        module_path, _, attribute_path = record.target.rpartition(".")
        if record.kind == "class":
            module_name, _, class_path = module_path.rpartition(".")
            module = importlib.import_module(module_name)
            owner = module
            for class_name in class_path.split("."):
                owner = getattr(owner, class_name)
            installed = getattr(owner, attribute_path)
        else:
            module = importlib.import_module(module_path)
            installed = getattr(module, attribute_path)
        marker_source = getattr(installed, PATCH_MARKER, None) or getattr(
            getattr(installed, "__wrapped__", None), PATCH_MARKER, None
        )
        assert marker_source is not None or record.was_missing, record.target


@pytest.mark.requires_runtime
def test_kimi_k3_quantized_up_proj_forces_tier_one(
    installed_model_patches,
) -> None:
    """A quantized routed up-projection must never reach a raw-weight tier.

    Tier selection short-circuits before reading the batch, so ``None`` tensors
    are safe here.
    """
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
        LatentMoERunner,
        LatentTailTier,
    )

    runner = SimpleNamespace(_up_proj_is_quantized=True)
    assert (
        LatentMoERunner._select_tail_tier(runner, None, None)
        is LatentTailTier.ALLREDUCE_OVERLAP
    )


@pytest.mark.requires_runtime
def test_deepseek_v4_selector_uses_ppu_flashmla(installed_model_patches) -> None:
    from types import SimpleNamespace
    from unittest import mock

    from vllm.models.deepseek_v4.nvidia import model as nvidia_model
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLAAttention,
    )
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        DeepseekV4FlashMLAAttention as NvidiaDeepseekV4FlashMLAAttention,
    )
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sail.models.deepseek_v4.flashmla import (
        DeepseekV4FlashMLAAttention as PPUDeepseekV4FlashMLAAttention,
    )

    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHMLA_SPARSE,
        )
    )
    platform_type = type(current_platform)
    with (
        mock.patch.object(platform_type, "get_device_capability", return_value=None),
        mock.patch.object(platform_type, "is_ppu", return_value=True),
    ):
        assert (
            nvidia_model._select_dsv4_attn_cls(config) is PPUDeepseekV4FlashMLAAttention
        )

    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4
    with (
        mock.patch.object(platform_type, "get_device_capability", return_value=None),
        mock.patch.object(platform_type, "is_ppu", return_value=True),
    ):
        assert (
            nvidia_model._select_dsv4_attn_cls(config)
            is DeepseekV4FlashInferMLAAttention
        )

    config.attention_config.backend = AttentionBackendEnum.FLASHMLA_SPARSE
    with (
        mock.patch.object(platform_type, "get_device_capability", return_value=None),
        mock.patch.object(platform_type, "is_ppu", return_value=False),
    ):
        assert (
            nvidia_model._select_dsv4_attn_cls(config)
            is NvidiaDeepseekV4FlashMLAAttention
        )


@pytest.mark.requires_runtime
def test_ppu_flashmla_inherits_upstream_attention_flow(
    installed_model_patches,
) -> None:
    """Only the PPU output projection and head-padding policy may diverge."""
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        DeepseekV4FlashMLAAttention as NvidiaDeepseekV4FlashMLAAttention,
    )

    from vllm_sail.models.deepseek_v4.flashmla import (
        DeepseekV4FlashMLAAttention as PPUDeepseekV4FlashMLAAttention,
    )

    assert issubclass(
        PPUDeepseekV4FlashMLAAttention,
        NvidiaDeepseekV4FlashMLAAttention,
    )
    assert (
        PPUDeepseekV4FlashMLAAttention.forward_mqa
        is NvidiaDeepseekV4FlashMLAAttention.forward_mqa
    )
    assert (
        PPUDeepseekV4FlashMLAAttention._forward_prefill
        is NvidiaDeepseekV4FlashMLAAttention._forward_prefill
    )


@pytest.mark.requires_runtime
def test_deepseek_v4_mapper_remains_unpatched(installed_model_patches) -> None:
    """Installing the PPU model selector must not alter CUDA weight mapping."""
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model

    from vllm_sail.patch.utils import PATCH_MARKER

    assert not hasattr(nvidia_model._make_deepseek_v4_weights_mapper, PATCH_MARKER)


@pytest.mark.requires_runtime
def test_marlin_gate_defaults_off_on_ppu(installed_model_patches) -> None:
    """Decision D2(c): MoE Marlin stays disabled on PPU unless opted in.

    The gate short-circuits before touching its config argument, so a
    sentinel object is safe here.
    """
    import os
    from unittest import mock

    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.platforms import current_platform

    os.environ.pop("VLLM_PPU_ENABLE_MOE_MARLIN", None)
    with mock.patch.object(type(current_platform), "is_ppu", return_value=True):
        assert marlin_utils.check_moe_marlin_supports_config(object()) is False


@pytest.mark.requires_runtime
def test_step5_dsa_admits_only_ppu_sm8x(installed_model_patches) -> None:
    from unittest import mock

    from vllm.models.step5.nvidia.sparse_attention import Step5DSAAttentionBackend
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    supports = Step5DSAAttentionBackend.supports_compute_capability
    platform_type = type(current_platform)
    with mock.patch.object(platform_type, "is_ppu", return_value=True):
        assert supports(DeviceCapability(8, 9)) is True
        assert supports(DeviceCapability(8, 0)) is True
        assert supports(DeviceCapability(7, 5)) is False
        assert supports(DeviceCapability(9, 0)) is True
    with mock.patch.object(platform_type, "is_ppu", return_value=False):
        assert supports(DeviceCapability(8, 9)) is False
        assert supports(DeviceCapability(8, 0)) is False
        assert supports(DeviceCapability(9, 0)) is True


@pytest.mark.requires_runtime
def test_step5_pair_inputs_skip_flashinfer_schedule_on_ppu(
    installed_model_patches,
) -> None:
    """PPU's Triton scorer needs no FlashInfer schedule; nothing imports it."""
    from types import SimpleNamespace
    from unittest import mock

    import torch
    from vllm.models.step5.nvidia.flashinfer_sparse import Step5FlashInferSparse
    from vllm.platforms import current_platform

    meta = SimpleNamespace(
        lengths=torch.tensor([3, 4, 5, 6], dtype=torch.int32),
        block_table=torch.arange(8, dtype=torch.int32).view(4, 2),
    )
    with (
        mock.patch.object(type(current_platform), "is_ppu", return_value=True),
        mock.patch.dict(sys.modules, {"flashinfer.attn_scores": None}),
    ):
        ((lengths, table, schedule),) = Step5FlashInferSparse.prefill_pair_inputs(
            object(), [meta]
        )
    assert lengths.tolist() == [4, 6]
    assert table.tolist() == [[0, 1], [4, 5]]
    assert schedule is None


@pytest.fixture
def installed_step5_patches(monkeypatch):
    """Install only Step5 patches so unrelated model imports never probe a GPU."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (8, 0))
    from vllm_sail.patch.utils import PATCH_REGISTRY

    for name, target in (
        (
            "step5_soft_fp8",
            "vllm.models.step5.nvidia.ops.summary_update._store_summary",
        ),
        (
            "step5_dsa_backend",
            "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.__init__",
        ),
        (
            "step5_indexer_fp16",
            "vllm.models.step5.nvidia.ops.indexer_norm_rope.indexer_norm_rope",
        ),
    ):
        if any(record.target == target for record in PATCH_REGISTRY):
            continue
        spec = importlib.util.spec_from_file_location(
            f"_step5_cpu_patch_{name}", MODELS_PATCH_DIR / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)


@pytest.mark.requires_runtime
@pytest.mark.parametrize("mode", ["1"])
def test_step5_summary_cache_dtype_follows_fp16_switch(
    installed_step5_patches, monkeypatch, mode
) -> None:
    from types import SimpleNamespace

    import torch
    from vllm.models.step5.nvidia.sparse_summary_cache import (
        Step5SparseSummaryCacheLayer,
    )

    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    get_spec = Step5SparseSummaryCacheLayer.get_kv_cache_spec
    monkeypatch.delenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", raising=False)
    byte_spec = get_spec(object(), config)
    assert byte_spec.dtype == torch.uint8
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", mode)
    half_spec = get_spec(object(), config)
    assert half_spec.dtype == torch.float16
    assert (half_spec.head_size, half_spec.tokens_per_state) == (
        byte_spec.head_size,
        byte_spec.tokens_per_state,
    )
    assert half_spec.page_size_bytes == 2 * byte_spec.page_size_bytes


@pytest.mark.requires_runtime
@pytest.mark.parametrize("mode", ["1"])
def test_step5_fp16_switch_selects_triton_scorer_off_ppu(
    installed_step5_patches, monkeypatch, mode
) -> None:
    """FP16 pages are scored only by soft_fp8's scorer, on any platform."""
    from types import SimpleNamespace

    import torch
    from vllm.models.step5.nvidia.flashinfer_sparse import Step5FlashInferSparse
    from vllm.platforms import current_platform

    from vllm_sail.models.step5.ops import soft_fp8

    meta = SimpleNamespace(
        lengths=torch.tensor([3, 4], dtype=torch.int32),
        block_table=torch.arange(4, dtype=torch.int32).view(2, 2),
    )
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", mode)
    assert soft_fp8.indexer_fp16() is True
    assert soft_fp8.indexer_query_dtype() == torch.float16
    with (
        _mock_is_ppu(current_platform, False),
        _no_module("flashinfer.attn_scores"),
    ):
        ((_, _, schedule),) = Step5FlashInferSparse.prefill_pair_inputs(
            object(), [meta]
        )
    assert schedule is None

    captured = {}
    owner = SimpleNamespace(_logits=lambda **kwargs: captured.update(kwargs))
    query = torch.zeros((4, 1, 8, 256), dtype=torch.float16)
    weights = torch.zeros((4, 1, 8))
    packed = torch.zeros((4, 32, 1, 258), dtype=torch.float16)
    Step5FlashInferSparse.logits(
        owner,
        query,
        weights,
        packed,
        4,
        torch.zeros(4, dtype=torch.long),
        torch.ones(4, dtype=torch.int32),
        8,
    )
    assert captured["q"].dtype == torch.float16
    assert captured["q"].shape == (4, 1, 8, 256)
    assert captured["kv_fused"] is packed


def _load_step5_soft_bodies(names, namespace):
    """Run the real production bodies; CPU numeric tests import no device module."""
    path = MODELS_PATCH_DIR.parents[2] / "models/step5/ops/soft_fp8.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    bodies = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names
    ]
    assert {node.name for node in bodies} == set(names)
    for node in bodies:
        node.decorator_list = []
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, *bodies],
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.requires_runtime
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_step5_store_value_modes_on_cpu(dtype):
    """FP16 Q and summary outputs keep the E4M3 quantization and saturation."""
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    namespace = dict(
        tl=SimpleNamespace(float16=torch.float16),
        encode_e4m3=lambda x: (
            x.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        ),
        e4m3_to_fp16=lambda x: x.view(torch.float8_e4m3fn).to(torch.float16),
    )
    _load_step5_soft_bodies(["_indexer_store_value"], namespace)
    value = torch.tensor([0.0, -0.0, 0.333, 1.125, -2.75, 500, -1000]).to(
        getattr(torch, dtype)
    )
    store = namespace["_indexer_store_value"]
    raw = store(value, False)
    assert torch.equal(store(value, True), raw.view(torch.float8_e4m3fn).half())
    half = store(value, True)
    assert half.dtype == torch.float16
    assert not torch.equal(half, value.half())
    assert half[-1].item() == -448


@pytest.mark.requires_runtime
def test_step5_summary_quantizes_after_bf16_rounding_on_cpu():
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    output = torch.zeros(4096, dtype=torch.float16)

    class Pointer:
        dtype = SimpleNamespace(element_ty=torch.float16)

        def __init__(self, offset=0):
            self.offset = offset

        def __add__(self, offset):
            return Pointer(self.offset + offset)

    def encode(value):
        assert value.dtype == torch.bfloat16
        return value.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)

    namespace = dict(
        tl=SimpleNamespace(
            float16=torch.float16,
            bfloat16=torch.bfloat16,
            div_rn=lambda a, b: a / b,
            store=lambda pointer, values: output.__setitem__(pointer.offset, values),
        ),
        _maximum=lambda a, b: torch.clamp_min(a, b),
        encode_e4m3=encode,
        e4m3_to_fp16=lambda x: x.view(torch.float8_e4m3fn).half(),
    )
    _load_step5_soft_bodies(["_indexer_store_value", "_store_summary"], namespace)
    numerator = torch.linspace(-1000, 1000, 256)
    denominator = torch.full((256,), 1.7)
    namespace["_store_summary"](
        Pointer(), 23, torch.arange(256), numerator, denominator, 4096, 64
    )
    ref = (
        (numerator / denominator)
        .bfloat16()
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .half()
    )
    assert torch.equal(output[512:768], ref)
    assert ref.abs().max() == 448
    assert torch.count_nonzero(output[:512]) == 0
    assert torch.count_nonzero(output[768:]) == 0


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("half", [False, True])
def test_step5_kernel_launch_modes_are_lazy_on_cpu(monkeypatch, ppu, half):
    """Toggling modes 0/1 repeatedly converts only the indexer pointers.

    The main attention FP8 cache is never touched.
    """
    from types import SimpleNamespace

    from vllm_sail import envs

    class Tensor:
        def __init__(self, dtype):
            self.dtype = dtype

        def view(self, dtype):
            return Tensor(dtype)

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kw: (self.name, args, kw)

    namespace = dict(
        torch=SimpleNamespace(
            Tensor=Tensor, float16="half", float8_e4m3fn="fp8", uint8="byte"
        ),
        current_platform=SimpleNamespace(is_ppu=lambda: ppu),
        envs=envs,
    )
    _load_step5_soft_bodies(["indexer_fp16", "ByteFP8Launch"], namespace)
    launch = namespace["ByteFP8Launch"](Kernel("upstream"), Kernel("software"), (0,))
    query, main_cache = Tensor("half" if half else "fp8"), Tensor("fp8")
    for mode in (0, 1, 0, 1):
        monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", str(mode))
        assert namespace["indexer_fp16"]() is bool(mode)
        name, args, kw = launch[(1,)](query, main_cache, sentinel=3)
        expected = "software" if half or ppu else "upstream"
        assert name == expected
        assert args[0].dtype == ("byte" if not half and ppu else query.dtype)
        assert args[1] is main_cache
        assert kw == {"sentinel": 3}


def _mock_is_ppu(platform, value: bool):
    from unittest import mock

    return mock.patch.object(type(platform), "is_ppu", return_value=value)


def _no_module(name: str):
    from unittest import mock

    return mock.patch.dict(sys.modules, {name: None})


@pytest.mark.requires_runtime
def test_step5_mtp_quant_config_ignores_bf16_draft_layers(
    installed_model_patches, monkeypatch
) -> None:
    """Checkpoint ignore names of a BF16 MTP layer reach its module paths."""
    import dataclasses
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.models.step5.nvidia import mtp
    from vllm.models.step5.nvidia.model import STEP5_PACKED_MODULES_MAPPING

    int8 = {"num_bits": 8, "type": "int", "symmetric": True}
    projections = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
    checkpoint_names = [f"self_attn.{name}" for name in projections] + [
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "eh_proj",
        "transformer.shared_head.output",
    ]
    target = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {
                "W8A8": {
                    "targets": ["Linear"],
                    "weights": {**int8, "strategy": "channel", "dynamic": False},
                    "input_activations": {
                        **int8,
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            },
            # The VL target's view: hf_to_vllm_mapper added language_model.
            "ignore": [
                f"language_model.model.layers.{layer}.{name}"
                for layer in (1, 2)
                for name in checkpoint_names
            ],
        }
    )
    target.packed_modules_mapping = dict(STEP5_PACKED_MODULES_MAPPING)
    snapshot = list(target.ignore)

    @dataclasses.dataclass
    class _Config:
        quant_config: object

    captured = {}
    init = mtp.Step5MTP.__init__
    monkeypatch.setitem(
        init.__globals__,
        "_upstream_init",
        lambda self, *, vllm_config, prefix: captured.update(config=vllm_config),
    )
    monkeypatch.setattr(
        mtp,
        "_get_mtp_config",
        lambda _: SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1),
    )
    init(object(), vllm_config=_Config(target), prefix="")

    draft = captured["config"].quant_config
    # Upstream's own Step5MTP strip.
    draft.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix={"language_model.": ""}))
    for module in (
        "mtp_block.self_attn.qkvg_proj",
        "mtp_block.self_attn.o_proj",
        "mtp_block.mlp.gate_up_proj",
        "mtp_block.mlp.down_proj",
        "shared_head.head",
    ):
        name = f"model.layers.2.{module}"
        assert should_ignore_layer(name, draft.ignore, draft.packed_modules_mapping)
    # Only the draft layers move, and the target's config is untouched.
    assert "model.layers.1.mlp.down_proj" in draft.ignore
    assert "model.layers.2.mlp.down_proj" not in draft.ignore
    assert target.ignore == snapshot


@pytest.mark.requires_runtime
def test_reapplication_raises(installed_model_patches) -> None:
    """The double-patch guard must still fire for these targets."""
    from vllm_sail.patch.utils import patch

    with pytest.raises(RuntimeError, match="already patched"):

        @patch(
            "vllm.model_executor.models.llama4",
            "Llama4MoE.custom_routing_function",
            reason="test",
            affected_versions=">=0.27.0,<0.28.0",
            remove_when="never; test-only",
        )
        def custom_routing_function(*args, **kwargs):
            raise AssertionError("must not install")


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "models/deepseek_v4_compressor",
            "__init__",
            "models/deepseek_v4/compressor",
            "DeepseekCompressor.__init__",
        ),
        (
            "models/deepseek_v4_cache",
            "dequantize_and_gather_k_cache",
            "models/deepseek_v4/common/ops/cache_utils",
            "dequantize_and_gather_k_cache",
        ),
        (
            "models/qwen4_exp_qsa_fp4",
            "__init__",
            "models/qwen4_exp/common/qsa_indexer",
            "_QSAIndexerCache.__init__",
        ),
        (
            "models/qwen4_exp_qsa_fp4",
            "get_supported_kernel_block_sizes",
            "v1/attention/backend",
            "AttentionBackend.get_supported_kernel_block_sizes",
        ),
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
