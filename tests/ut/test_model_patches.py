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
from enum import Enum
from types import SimpleNamespace as NS

import pytest

from tests.support.modules import load_patch
from tests.support.source import (
    assert_accepts_upstream_keywords,
    definition,
    function,
    load_moe_oracle,
)

MODELS_PATCH_DIR = (
    pathlib.Path(__file__).parents[2] / "vllm_sail" / "patch" / "enhancement" / "models"
)
REPO_ROOT = pathlib.Path(__file__).parents[2]

EXPECTED_MODULES = {
    "deepseek_v2",
    "deepseek_v4",
    "deepseek_v4_cache",
    "deepseek_v4_metadata",
    "deepseek_v4_compressor",
    "deepseek_v4_graph_warmup",
    "deepseek_v41",
    "deepseek_v41_device",
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
    "vllm.model_executor.models.deepseek_v2.init_indexer_wk_weights_proj",
    "vllm.model_executor.models.deepseek_v2.Indexer.process_weights_after_loading",
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
    "vllm.models.minimax_m3.nvidia.model._supports_qkv_fused_epilogue",
    "vllm.models.minimax_m3.nvidia.model._maybe_init_gemm_rs",
    "vllm.models.minimax_m3.nvidia.model._bind_gemm_rs",
    "vllm.models.minimax_m3.nvidia.model.MiniMAXGemmaRMSNorm.forward",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock.__init__",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer.__init__",
    "vllm.model_executor.models.qwen3_moe.Qwen3MoeModel.__init__",
    "vllm.model_executor.models.qwen3_next.th_nvtx_range_push",
    "vllm.model_executor.models.qwen3_next.th_nvtx_range_pop",
    "vllm.models.qwen4_exp.nvidia.qsa.Qwen4ExpQSAImpl.__init__",
    "vllm.models.qwen4_exp.nvidia.qsa.Qwen4ExpQSAImpl.forward",
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
    "vllm.models.step5.common.ops.summary_update._store_summary",
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope._indexer_norm_rope",
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope._joint_prepare",
    "vllm.models.step5.common.ops.indexer_norm_rope._indexer_norm_rope",
    "vllm.models.step5.common.ops.sparse_indexer_scores._direct_scores",
    "vllm.models.step5.common.ops.sparse_indexer_scores._paged_scores_sm90",
    "vllm.models.step5.common.ops.sparse_indexer._copy_summary_page",
    "vllm.models.step5.common.ops.sparse_indexer._copy_summary_tile",
    "vllm.models.step5.common.ops.sparse_indexer._pack_summaries",
    "vllm.models.step5.common.ops.indexer_norm_rope.indexer_norm_rope",
    "vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope.qkv_indexer_norm_rope",
    (
        "vllm.models.step5.common.sparse_summary_cache."
        "Step5SparseSummaryCacheLayer.get_kv_cache_config"
    ),
    "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.decode_logits",
    "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.logits",
    "vllm.models.step5.nvidia.mtp.Step5MTP.__init__",
    (
        "vllm.model_executor.layers.quantization.utils.marlin_utils."
        "check_moe_marlin_supports_config"
    ),
    "vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExpertsBase._supports_current_device",
    "vllm.model_executor.kernels.linear.mixed_precision.marlin.MarlinLinearKernel.can_implement",
    "vllm.model_executor.kernels.linear.init_nvfp4_linear_kernel",
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4.is_fp4_marlin_supported",
    "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8.is_fp8_marlin_supported",
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
def test_marlin_gate_defaults_off_on_ppu(installed_model_patches, monkeypatch) -> None:
    """Decision D2(c): MoE Marlin stays disabled on PPU unless opted in.

    The gate short-circuits before touching its config argument, so a
    sentinel object is safe here.
    """
    from unittest import mock

    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.platforms import current_platform

    monkeypatch.delenv("VLLM_SAIL_ENABLE_MOE_MARLIN", raising=False)
    monkeypatch.delenv("VLLM_PPU_ENABLE_MOE_MARLIN", raising=False)
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
            "vllm.models.step5.common.ops.summary_update._store_summary",
        ),
        (
            "step5_dsa_backend",
            "vllm.models.step5.nvidia.flashinfer_sparse.Step5FlashInferSparse.__init__",
        ),
        (
            "step5_indexer_fp16",
            "vllm.models.step5.common.ops.indexer_norm_rope.indexer_norm_rope",
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
    from vllm.models.step5.common.sparse_summary_cache import (
        Step5SparseSummaryCacheLayer,
    )

    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    get_spec = Step5SparseSummaryCacheLayer.get_kv_cache_config
    monkeypatch.delenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", raising=False)
    byte_spec = get_spec(object(), config).spec
    assert byte_spec.dtype == torch.uint8
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", mode)
    half_spec = get_spec(object(), config).spec
    assert half_spec.dtype == torch.float16
    assert (half_spec.head_size, half_spec.tokens_per_state) == (
        byte_spec.head_size,
        byte_spec.tokens_per_state,
    )
    assert half_spec.page_size_bytes == 2 * byte_spec.page_size_bytes


@pytest.mark.requires_runtime
def test_step5_warmup_resolves_common_prefill_metadata(installed_step5_patches):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.models.step5.nvidia.flashinfer_sparse import Step5FlashInferSparse
    from vllm.platforms import current_platform

    owner = SimpleNamespace(prefill_metadata=Mock())
    with _mock_is_ppu(current_platform, True):
        Step5FlashInferSparse.warmup_prefill_scoring(owner, 5, 4, 0, "cpu")
    owner.prefill_metadata.assert_called_once()
    owners, positions, regions, topk = owner.prefill_metadata.call_args.args
    assert (regions, topk) == (5, 4)
    assert owners.tolist() == [0, 0]
    assert positions.tolist() == [32, 33]


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


def _marlin_boundary(modules, root, preloaded):
    """加载真实准入函数及 authored patch；不是完整 upstream 导入或设备证据。"""
    platform = NS(
        ppu=True,
        capability=80,
        device_name="ppu",
        is_cuda=lambda: True,
        is_cpu=lambda: False,
        is_xpu=lambda: False,
        is_rocm=lambda: False,
    )
    platform.is_ppu = lambda: platform.ppu

    class Capability(tuple):
        def to_int(self):
            return self[0] * 10 + self[1]

    platform._enum = "cuda"
    platform.get_device_capability = lambda: Capability(divmod(platform.capability, 10))
    platform.has_device_capability = lambda value: platform.capability >= (
        value[0] * 10 + value[1] if isinstance(value, tuple) else value
    )
    modules("vllm.platforms", current_platform=platform)
    envs = modules("vllm_sail.envs", VLLM_SAIL_ENABLE_MOE_MARLIN=False)
    modules(
        "vllm.model_executor.layers.quantization.online.requantization",
        is_requantization_source=lambda: False,
    )
    utils_path = root / "vllm/model_executor/layers/quantization/utils"
    scalar_types = NS(
        **{
            name: type(name, (), {"size_bits": 4 if "4" in name else 8})()
            for name in (
                "uint4",
                "uint8",
                "uint4b8",
                "uint8b128",
                "float8_e4m3fn",
                "float4_e2m1f",
            )
        }
    )
    utils = modules(
        "vllm.model_executor.layers.quantization.utils.marlin_utils",
        current_platform=platform,
        scalar_types=scalar_types,
        MARLIN_SUPPORTED_GROUP_SIZES=[-1, 32, 64, 128],
    )
    for name in (
        "query_marlin_supported_quant_types",
        "_check_marlin_supported",
        "check_marlin_supported",
        "verify_marlin_supported",
        "check_moe_marlin_supports_config",
    ):
        function(utils_path / "marlin_utils.py", name, utils.__dict__)
    layout = utils.check_moe_marlin_supports_config
    dense_config_check = utils._check_marlin_supported
    linear = modules("vllm.model_executor.kernels.linear", current_platform=platform)
    upstream_init_nvfp4 = function(
        root / "vllm/model_executor/kernels/linear/__init__.py",
        "init_nvfp4_linear_kernel",
        linear.__dict__,
    )
    nvfp4_consumers = (
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4",
        "vllm.model_executor.layers.quantization.modelopt",
        "vllm.model_executor.layers.quantization.online.nvfp4",
        "vllm.model_executor.layers.quantization.quark.schemes.quark_nvfp4",
    )
    if preloaded:
        for consumer in nvfp4_consumers:
            modules(consumer, init_nvfp4_linear_kernel=upstream_init_nvfp4)
    probes = []
    for bits, consumer in (
        (4, "vllm.model_executor.kernels.linear.nvfp4.marlin"),
        (8, "vllm.model_executor.kernels.linear.scaled_mm.marlin"),
    ):
        name = f"is_fp{bits}_marlin_supported"
        provider = modules(
            f"vllm.model_executor.layers.quantization.utils.marlin_utils_fp{bits}",
            current_platform=platform,
        )
        original = function(
            utils_path / f"marlin_utils_fp{bits}.py", name, provider.__dict__
        )
        if preloaded:
            modules(consumer, **{name: original})
        probes.append((consumer, name, provider))
    experts_path = root / "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
    device_check = function(
        experts_path,
        "MarlinExpertsBase._supports_current_device",
        {
            "current_platform": platform,
        },
    )
    experts = type(
        "MarlinExpertsBase",
        (),
        {
            "_supports_current_device": staticmethod(device_check),
            "executed_activation_key": staticmethod(
                function(experts_path, "MarlinExpertsBase.executed_activation_key", {})
            ),
            "_supports_quant_scheme": staticmethod(lambda *a: True),
        },
    )
    modules(
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe",
        MarlinExpertsBase=experts,
    )
    dense_check = function(
        root / "vllm/model_executor/kernels/linear/mixed_precision/marlin.py",
        "MarlinLinearKernel.can_implement",
        utils.__dict__,
    )
    dense = type("MarlinLinearKernel", (), {"can_implement": classmethod(dense_check)})
    modules(
        "vllm.model_executor.kernels.linear.mixed_precision.marlin",
        MarlinLinearKernel=dense,
    )
    if preloaded:
        for consumer in (
            "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16",
        ):
            modules(consumer, check_moe_marlin_supports_config=layout)
    expert_descriptor = experts.__dict__["_supports_current_device"]
    dense_descriptor = dense.__dict__["can_implement"]
    patch_module = load_patch("vllm_sail/patch/enhancement/models/moe_marlin_gate.py")
    if not preloaded:
        for consumer in nvfp4_consumers:
            modules(consumer, init_nvfp4_linear_kernel=linear.init_nvfp4_linear_kernel)
    for consumer, name, provider in probes:
        if not preloaded:
            modules(consumer, **{name: getattr(provider, name)})
    config = NS(
        act_type=NS(itemsize=2),
        weight_type=scalar_types.uint4b8,
        zero_points=False,
        group_size=32,
        full_weight_shape=(128, 128),
        partition_weight_shape=(128, 128),
    )
    return NS(
        platform=platform,
        envs=envs,
        utils=utils,
        scalar_types=scalar_types,
        experts=experts,
        dense=dense,
        config=config,
        probes=probes,
        patch=patch_module,
        layout=layout,
        device_check=device_check,
        dense_check=dense_check,
        dense_config_check=dense_config_check,
        expert_descriptor=expert_descriptor,
        dense_descriptor=dense_descriptor,
        linear=linear,
        nvfp4_consumers=nvfp4_consumers,
        upstream_init_nvfp4=upstream_init_nvfp4,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("capability", [80, 89])
@pytest.mark.parametrize("legacy_opt_in", [False, True])
def test_real_marlin_compute_admission_rejects_ppu_but_preserves_layout(
    modules,
    patch_utils_module,
    upstream_source_root,
    preloaded,
    capability,
    legacy_opt_in,
):
    state = _marlin_boundary(modules, upstream_source_root, preloaded)
    state.platform.capability = capability
    state.envs.VLLM_SAIL_ENABLE_MOE_MARLIN = legacy_opt_in
    layout_config = NS(hidden_dim=128, intermediate_size_per_partition_unpadded=128)
    assert state.layout(layout_config, 32) is True
    assert state.patch.supports_marlin_layout is state.layout
    assert (
        state.utils.check_moe_marlin_supports_config(layout_config, 32) is legacy_opt_in
    )
    assert state.device_check() is True
    assert state.experts._supports_current_device() is False
    assert (
        type("PreloadedExperts", (state.experts,), {})._supports_current_device()
        is False
    )
    assert isinstance(state.experts.__dict__["_supports_current_device"], staticmethod)
    assert isinstance(state.dense.__dict__["can_implement"], classmethod)
    assert state.dense.can_implement(state.config) == (
        False,
        state.patch._NO_PPU_MARLIN,
    )
    quant_type = state.scalar_types.uint4b8
    # 公共量化配置校验不能阻断之后的 Triton/Humming 等 MP 内核选择。
    assert state.utils._check_marlin_supported is state.dense_config_check
    assert state.utils.check_marlin_supported(quant_type, 32) is True
    assert state.utils.verify_marlin_supported(quant_type, 32) is None
    with pytest.raises(ValueError, match="group_size"):
        state.utils.verify_marlin_supported(quant_type, 16)
    for consumer, name, provider in state.probes:
        assert getattr(provider, name)() is False
        assert getattr(sys.modules[consumer], name) is getattr(provider, name)
    records = patch_utils_module.PATCH_REGISTRY
    assert len(records) == (14 if preloaded else 6)
    assert all(
        r.affected_versions == ">=0.32.0,<0.33.0" and r.reason and r.remove_when
        for r in records
    )
    target = "vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExpertsBase._supports_current_device"
    assert (
        patch_utils_module.original_of(state.experts._supports_current_device, target)
        is state.expert_descriptor
    )
    assert state.expert_descriptor.__func__ is state.device_check
    dense_target = "vllm.model_executor.kernels.linear.mixed_precision.marlin.MarlinLinearKernel.can_implement"
    assert (
        patch_utils_module.original_of(state.dense.can_implement, dense_target)
        is state.dense_descriptor
    )
    assert state.dense_descriptor.__func__ is state.dense_check
    nvfp4_target = "vllm.model_executor.kernels.linear.init_nvfp4_linear_kernel"
    assert (
        patch_utils_module.original_of(
            state.linear.init_nvfp4_linear_kernel, nvfp4_target
        )
        is state.upstream_init_nvfp4
    )
    for consumer in state.nvfp4_consumers:
        assert (
            sys.modules[consumer].init_nvfp4_linear_kernel
            is state.linear.init_nvfp4_linear_kernel
        )
    with pytest.raises(RuntimeError, match="already patched"):
        patch_utils_module.patch(
            "vllm.model_executor.layers.fused_moe.experts.marlin_moe",
            "MarlinExpertsBase._supports_current_device",
            reason="test",
            affected_versions=">=0.32.0,<0.33.0",
            remove_when="test-only",
        )(lambda: True)


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("capability", [74, 80, 89])
def test_real_marlin_non_ppu_admission_delegates_and_keeps_original_errors(
    modules,
    upstream_source_root,
    preloaded,
    capability,
):
    state = _marlin_boundary(modules, upstream_source_root, preloaded)
    state.platform.ppu, state.platform.capability = False, capability
    assert state.experts._supports_current_device() is (capability >= 75)
    assert state.dense.can_implement(state.config) == state.dense_check(
        state.dense, state.config
    )
    for _, name, provider in state.probes:
        assert getattr(provider, name)() is (capability >= 75)
    assert state.utils.check_marlin_supported(state.scalar_types.uint4b8, 32) is (
        capability >= 75
    )
    error = RuntimeError("upstream capability query failed")

    def fail(*args):
        raise error

    state.platform.has_device_capability = state.platform.get_device_capability = fail
    for call in (
        state.experts._supports_current_device,
        lambda: state.dense.can_implement(state.config),
        lambda: state.utils.check_marlin_supported(state.scalar_types.uint4b8, 32),
        *(getattr(provider, name) for _, name, provider in state.probes),
    ):
        with pytest.raises(RuntimeError) as raised:
            call()
        assert raised.value is error


@pytest.mark.upstream_source
@pytest.mark.parametrize("deepseek", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_real_mxfp4_oracle_rejects_marlin_compute_and_keeps_auto_fallback(
    modules,
    upstream_source_root,
    deepseek,
    batched,
    explicit,
):
    state = _marlin_boundary(modules, upstream_source_root, True)
    state.envs.VLLM_SAIL_ENABLE_MOE_MARLIN = True
    backend = Enum(
        "Backend",
        {
            name: name.lower()
            for name in (
                "MARLIN",
                "BATCHED_MARLIN",
                "EMULATION",
                "FALLBACK",
                "NONE",
                "XPU",
            )
        },
    )
    fmt = function(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/modular_kernel.py",
        "FusedMoEActivationFormat",
        {"Enum": Enum},
    )
    config = NS(
        moe_backend="marlin" if explicit else "auto",
        is_lora_enabled=False,
        activation_format=fmt.BatchedExperts if batched else fmt.Standard,
    )
    admission = function(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/modular_kernel.py",
        "FusedMoEExperts.is_supported_config",
        {"current_platform": state.platform},
    )
    state.experts.is_supported_config = staticmethod(admission)
    fallback = type(
        "Fallback", (), {"is_supported_config": staticmethod(lambda *a: (True, None))}
    )
    marlin = backend.BATCHED_MARLIN if batched else backend.MARLIN
    namespace = {
        "current_platform": state.platform,
        "Mxfp4MoeBackend": backend,
        "mk": NS(FusedMoEActivationFormat=fmt),
        "kMxfp4Static": object(),
        "TRTLLM_BACKENDS": (),
        "_resolve_activation_key": lambda key: key,
        "_backend_activation_key": lambda b: None,
        "_get_requested_backends": lambda *a: [backend.MARLIN],
        "map_mxfp4_backend": lambda *a: [backend.MARLIN],
        "_get_priority_backends_for_gpt_oss": lambda: [marlin, backend.FALLBACK],
        "_get_priority_backends": lambda: [marlin, backend.FALLBACK],
        "backend_to_kernel_cls": lambda b: [
            fallback if b == backend.FALLBACK else state.experts
        ],
        "_warn_on_bf16_fallback": lambda *a: None,
        "logger": NS(info_once=lambda *a, **kw: None, debug_once=lambda *a, **kw: None),
    }
    path = upstream_source_root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
    load_moe_oracle(path, "Mxfp4MoEKernelOracle", namespace)
    for name in (
        "_requested_kernel_cls",
        "_filter_by_activation",
        "_mxfp4_quant_keys_for_experts",
        "_validate_mxfp4_experts_config",
    ):
        function(path, name, namespace)
    select = function(
        path,
        "select_deepseek_v4_mxfp4_moe_backend"
        if deepseek
        else "select_mxfp4_moe_backend",
        namespace,
    )
    if explicit:
        with pytest.raises(ValueError, match="current device ppu"):
            select(config)
    else:
        assert select(config) == (backend.FALLBACK, fallback)
    namespace["_MXFP4_SCHEME_CANDIDATES"] = [(namespace["kMxfp4Static"], None)]
    make_kernel = function(path, "make_mxfp4_moe_kernel", namespace)
    with pytest.raises(ValueError, match="current device ppu"):
        make_kernel(object(), config, state.experts, marlin)


def _linear_selection_boundary(state, root, backend):
    """真实筛选 helpers；候选列表和可选 Humming 优先级是本测试的外部边界。"""
    linear = state.linear
    linear.envs = NS(VLLM_DISABLED_KERNELS=[], VLLM_BATCH_INVARIANT=False)
    linear.logger = NS(
        debug=lambda *a, **kw: None,
        debug_once=lambda *a, **kw: None,
        info=lambda *a, **kw: None,
        info_once=lambda *a, **kw: None,
        warning_once=lambda *a, **kw: None,
    )
    linear._get_linear_backend = lambda *, quantization: backend
    linear.prioritize_humming = lambda kernels, compute_capability=None: kernels
    path = root / "vllm/model_executor/kernels/linear/__init__.py"
    for name in (
        "_filter_kernels_by_backend",
        "_resolve_backend_kernels",
        "choose_mp_linear_kernel",
    ):
        function(path, name, linear.__dict__)
    return linear


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize(
    "method_name", ["AutoGPTQLinearMethod", "AutoAWQMarlinLinearMethod"]
)
@pytest.mark.parametrize("backend", ["auto", "triton", "marlin"])
def test_real_mp_constructor_and_weight_creation_preserve_non_marlin_selection(
    modules,
    upstream_source_root,
    preloaded,
    method_name,
    backend,
):
    """真实 constructor/create_weights/selector/Triton 准入；不执行分配或 GEMM。"""
    state = _marlin_boundary(modules, upstream_source_root, preloaded)
    linear = _linear_selection_boundary(state, upstream_source_root, backend)
    dtype = state.config.act_type
    torch = NS(
        float16=dtype, bfloat16=object(), int32=object(), empty=lambda *a, **kw: NS()
    )
    triton_path = (
        upstream_source_root
        / "vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py"
    )
    triton_ns = {
        "current_platform": state.platform,
        "torch": torch,
        "TRITON_W4A16_SUPPORTED_GROUP_SIZES": [-1, 32, 64, 128, 256],
    }
    triton = type(
        "TritonW4A16LinearKernel",
        (),
        {
            "SUPPORTED_QUANT_TYPES": [
                state.scalar_types.uint4b8,
                state.scalar_types.uint4,
            ],
            "get_min_capability": classmethod(
                function(
                    triton_path, "TritonW4A16LinearKernel.get_min_capability", triton_ns
                )
            ),
            "can_implement": classmethod(
                function(
                    triton_path, "TritonW4A16LinearKernel.can_implement", triton_ns
                )
            ),
            "__init__": lambda self, config, **kwargs: setattr(self, "config", config),
        },
    )
    marlin_path = (
        upstream_source_root
        / "vllm/model_executor/kernels/linear/mixed_precision/marlin.py"
    )
    state.dense.get_min_capability = classmethod(
        function(marlin_path, "MarlinLinearKernel.get_min_capability", {})
    )
    for cls in (state.dense, triton):
        cls.is_supported = classmethod(
            function(
                marlin_path.with_name("MPLinearKernel.py"),
                "MPLinearKernel.is_supported",
                {"current_platform": state.platform},
            )
        )
    linear._POSSIBLE_KERNELS = {state.platform._enum: [state.dense, triton]}
    linear._LINEAR_BACKEND_KERNEL_MAP = {"marlin": {state.dense}, "triton": {triton}}
    awq = method_name == "AutoAWQMarlinLinearMethod"
    path = (
        upstream_source_root
        / f"vllm/model_executor/layers/quantization/{'auto_awq' if awq else 'auto_gptq'}.py"
    )
    namespace = {
        "torch": torch,
        "current_platform": state.platform,
        "envs": linear.envs,
        "verify_marlin_supported": state.utils.verify_marlin_supported,
        "scalar_types": state.scalar_types,
        "MPLinearLayerConfig": NS,
        "choose_mp_linear_kernel": linear.choose_mp_linear_kernel,
        "logger": linear.logger,
        **{
            name: NS
            for name in (
                "PackedvLLMParameter",
                "GroupQuantScaleParameter",
                "ChannelQuantScaleParameter",
                "PackedColumnParameter",
            )
        },
    }
    function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/marlin_utils.py",
        "marlin_repeat_scales_on_all_ranks",
        namespace,
    )
    cls = type(
        method_name,
        (),
        {
            "_kernel_backends_being_used": set(),
            "__init__": function(path, f"{method_name}.__init__", namespace),
            "create_weights": function(
                path, f"{method_name}.create_weights", namespace
            ),
        },
    )
    config = NS(
        quant_type=state.scalar_types.uint4 if awq else state.scalar_types.uint4b8,
        group_size=32,
        zero_point=awq,
        pack_factor=8,
    )
    method = cls(config)
    assert method.quant_config is config
    parameters = {}
    layer = NS(
        register_parameter=lambda name, value: parameters.__setitem__(name, value)
    )
    if backend == "marlin":
        with pytest.raises(ValueError, match="PPU 没有 Marlin"):
            method.create_weights(layer, 128, [128], 128, 128, dtype)
        assert parameters == {}
    else:
        method.create_weights(layer, 128, [128], 128, 128, dtype)
        assert isinstance(method.kernel, triton)
        assert method.kernel.config.weight_type is config.quant_type
        assert method.kernel.config.zero_points is awq
        assert set(parameters) == {"qweight", "scales", "qzeros"}


@pytest.mark.upstream_source
def test_nvfp4_copied_selector_keeps_complete_upstream_body(upstream_source_root):
    copied = definition(
        MODELS_PATCH_DIR / "moe_marlin_gate.py", "init_nvfp4_linear_kernel"
    )
    original = definition(
        upstream_source_root / "vllm/model_executor/kernels/linear/__init__.py",
        "init_nvfp4_linear_kernel",
    )
    branches = [
        node
        for node in ast.walk(copied)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not prefers_humming(compute_capability)"
    ]
    assert len(branches) == 1
    branch = branches[0]
    assert len(branch.body) == 1 and isinstance(branch.body[0], ast.If)
    guard = branch.body[0]
    assert ast.unparse(guard.test) == "not current_platform.is_ppu()"
    assert not guard.orelse
    branch.body = guard.body
    assert ast.dump(copied, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def test_marlin_gate_static_inventory_resolves_all_compute_and_consumer_targets():
    from tests.support.patch_inventory import scan_patch_file

    hooks, unresolved = scan_patch_file(MODELS_PATCH_DIR / "moe_marlin_gate.py")
    assert not unresolved
    targets = {hook.target for hook in hooks}
    required = {target for target in _EXPECTED_TARGETS if "marlin" in target.lower()}
    selector = "vllm.model_executor.kernels.linear.init_nvfp4_linear_kernel"
    assert required | {selector} <= targets
    assert len(hooks) == 14 and len(targets) == 14
    assert len([hook for hook in hooks if hook.conditional_alias]) == 8
    assert all(isinstance(hook.node, ast.FunctionDef) for hook in hooks)
    assert not any(target.endswith("._check_marlin_supported") for target in targets)


def _nvfp4_selection_boundary(state, modules, root, backend, has_humming):
    """真实 Marlin/Humming 准入，第三方可用性和实例构造是外部边界。"""
    linear = _linear_selection_boundary(state, root, backend)

    def initialize(self, config):
        self.config = config

    for name, provider, prefix in (
        (
            "MarlinNvFp4LinearKernel",
            modules("vllm.model_executor.kernels.linear.nvfp4.marlin"),
            "marlin",
        ),
        (
            "HummingNvFp4LinearKernel",
            modules(
                "vllm.model_executor.kernels.linear.nvfp4.humming",
                current_platform=state.platform,
                has_humming=lambda: has_humming,
            ),
            "humming",
        ),
    ):
        path = root / f"vllm/model_executor/kernels/linear/nvfp4/{prefix}.py"
        cls = type(
            name,
            (),
            {
                "__init__": initialize,
                **{
                    method: classmethod(
                        function(path, f"{name}.{method}", provider.__dict__)
                    )
                    for method in ("is_supported", "can_implement")
                },
            },
        )
        setattr(linear, name, cls)
    for name in (
        "B12xNvFp4W4A16LinearKernel",
        "FlashInferCuteDslNvFp4W4A16LinearKernel",
        "CutlassNvFp4LinearKernel",
        "EmulationNvFp4LinearKernel",
    ):
        setattr(
            linear,
            name,
            type(
                name,
                (),
                {
                    "__init__": initialize,
                    "is_supported": classmethod(
                        lambda cls, *a: (False, "测试边界：该后端不可用")
                    ),
                },
            ),
        )
    linear.LinearKernelOracle = NS(maybe_skip_for_requantization_source=lambda: False)
    linear._first_usable_kernel = function(
        root / "vllm/model_executor/kernels/linear/__init__.py",
        "_first_usable_kernel",
        linear.__dict__,
    )
    linear._check_activation_fallback = (
        lambda cls, key: None
    )  # Activation policy is outside the hardware-admission test.
    linear.kNvfp4Dynamic = object()
    linear.NvFp4LinearLayerConfig = NS
    linear.prefers_humming = lambda *a: False
    linear._POSSIBLE_NVFP4_KERNELS = {
        state.platform._enum: [
            linear.MarlinNvFp4LinearKernel,
            linear.HummingNvFp4LinearKernel,
        ]
    }
    linear._LINEAR_BACKEND_KERNEL_MAP = {
        "marlin": {linear.MarlinNvFp4LinearKernel},
        "humming": {linear.HummingNvFp4LinearKernel},
    }
    return linear


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("capability", [80, 89])
@pytest.mark.parametrize(
    "backend,batch_invariant,has_humming",
    [
        ("auto", False, True),
        ("auto", True, True),
        ("marlin", False, True),
        ("auto", False, False),
        ("humming", False, False),
    ],
)
def test_real_nvfp4_auto_fallback_and_explicit_marlin_rejection(
    modules,
    upstream_source_root,
    preloaded,
    capability,
    backend,
    batch_invariant,
    has_humming,
):
    state = _marlin_boundary(modules, upstream_source_root, preloaded)
    state.platform.capability = capability
    linear = _nvfp4_selection_boundary(
        state, modules, upstream_source_root, backend, has_humming
    )
    linear.envs.VLLM_BATCH_INVARIANT = batch_invariant
    assert state.patch._ppu_init_nvfp4.__globals__ is linear.__dict__
    if backend == "marlin" or not has_humming:
        with pytest.raises(
            ValueError, match="Marlin FP4 not available|Humming is not installed"
        ):
            linear.init_nvfp4_linear_kernel(use_a16=True)
    else:
        result = linear.init_nvfp4_linear_kernel(use_a16=True)
        assert isinstance(result, linear.HummingNvFp4LinearKernel)
        # 模拟预加载和后加载的消费者别名使用同一 selector 和活跃 registry。
        for consumer in state.nvfp4_consumers:
            result = sys.modules[consumer].init_nvfp4_linear_kernel(use_a16=True)
            assert isinstance(result, linear.HummingNvFp4LinearKernel)


@pytest.mark.upstream_source
def test_nvfp4_non_ppu_selector_delegates_and_preserves_exception_identity(
    modules, upstream_source_root
):
    state = _marlin_boundary(modules, upstream_source_root, True)
    linear = _nvfp4_selection_boundary(
        state, modules, upstream_source_root, "auto", True
    )
    state.platform.ppu = False
    assert isinstance(
        linear.init_nvfp4_linear_kernel(use_a16=True), linear.MarlinNvFp4LinearKernel
    )
    error = RuntimeError("upstream NVFP4 selection failed")

    def fail(**kwargs):
        raise error

    linear._get_linear_backend = fail
    with pytest.raises(RuntimeError) as raised:
        linear.init_nvfp4_linear_kernel(use_a16=True)
    assert raised.value is error


def _nvfp4_cutlass_bias_boundary(root, capability, padded_rows, padded_k, fail_at=None):
    """真实 class、padding/slice helpers；tensor、native 量化/GEMM 与支持查询是替身。"""
    events = []
    error = RuntimeError("NVFP4 native boundary failed")
    dtype = object()

    def record(stage, *args):
        events.append((stage, *args))
        if stage == fail_at:
            raise error

    class Tensor:
        def __init__(self, shape):
            self.shape, self.dtype = shape, dtype

        def __getitem__(self, key):
            record("slice", self, key)
            assert key[0] is Ellipsis and key[1] == slice(None, 80)
            return Tensor((*self.shape[:-1], 80))

        def contiguous(self):
            record("contiguous", self)
            return self

        def __add__(self, other):
            record("add", self, other)
            return Tensor(self.shape)

        def view(self, *shape):
            record("view", self, shape)
            return Tensor(shape)

    x = Tensor((2, 3, 64))
    layer = NS(
        output_size_per_partition=80,
        input_size_per_partition=64,
        weight=Tensor((96 if padded_rows else 80, 48 if padded_k else 32)),
        weight_scale=object(),
        input_global_scale_inv=object(),
        alpha=object(),
    )
    packed, scales = object(), object()
    bias = object()

    def quantize(inputs, scale, **kwargs):
        record("quantize", inputs, scale, kwargs)
        return packed, scales

    def gemm(*args, **kwargs):
        record("gemm", args, kwargs)
        return Tensor((6, layer.weight.shape[0]))

    kernel_root = root / "vllm/model_executor/kernels/linear/nvfp4"
    utils = root / "vllm/model_executor/layers/quantization/utils/nvfp4_utils.py"
    base_init = function(kernel_root.parent / "base.py", "LinearKernel.__init__", {})
    base = type("NvFp4LinearKernel", (), {"__init__": base_init})
    namespace = {
        "NvFp4LinearKernel": base,
        "current_platform": NS(
            is_device_capability_family=lambda family: capability // 10 == family // 10
        ),
        "cutlass_fp4_supported": lambda: True,
        "scaled_fp4_quant": quantize,
        "cutlass_scaled_fp4_mm": gemm,
        "nvfp4_weight_padding_bytes": function(utils, "nvfp4_weight_padding_bytes", {}),
        "slice_nvfp4_output": function(utils, "slice_nvfp4_output", {}),
    }
    namespace.update(
        torch=NS(Tensor=Tensor),
        kNvfp4Dynamic="nvfp4",
        as_quantized_activation=lambda value, key: None,
        QuantizedActivation=lambda data, scale, dtype, shape, key: NS(
            data=data, scale=scale, orig_dtype=dtype, orig_shape=shape
        ),
    )
    base.prepare_input = function(
        kernel_root / "base.py", "NvFp4LinearKernel.prepare_input", namespace
    )
    cls = function(kernel_root / "cutlass.py", "CutlassNvFp4LinearKernel", namespace)
    config = object()
    kernel = cls(config)
    assert kernel.config is config
    assert kernel.bias_epilogue is (capability // 10 == 12)
    return NS(
        kernel=kernel,
        x=x,
        layer=layer,
        bias=bias,
        packed=packed,
        scales=scales,
        dtype=dtype,
        events=events,
        error=error,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("capability", [80, 89, 100, 120])
@pytest.mark.parametrize("padded_rows", [False, True])
@pytest.mark.parametrize("padded_k", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_nvfp4_cutlass_bias_applies_once_after_padding_or_in_epilogue(
    upstream_source_root, capability, padded_rows, padded_k, with_bias
):
    """仅验证真实 provider 的调用契约；模拟支持查询不代表该设备支持 CUTLASS FP4。"""
    f = _nvfp4_cutlass_bias_boundary(
        upstream_source_root, capability, padded_rows, padded_k
    )
    bias = f.bias if with_bias else None
    result = f.kernel.apply_weights(f.layer, f.x, bias)
    fuse = with_bias and capability == 120 and not padded_rows
    assert result.shape == (2, 3, 80) and result.dtype is f.dtype
    assert f.events[0] == (
        "quantize",
        f.x,
        f.layer.input_global_scale_inv,
        dict(
            is_sf_swizzled_layout=True,
            padded_n=96 if padded_k else 64,
        ),
    )
    assert f.events[1] == (
        "gemm",
        (
            f.packed,
            f.layer.weight,
            f.scales,
            f.layer.weight_scale,
            f.layer.alpha,
            f.dtype,
        ),
        {"bias": bias if fuse else None},
    )
    stages = [event[0] for event in f.events]
    assert stages == [
        "quantize",
        "gemm",
        *(["slice", "contiguous"] if padded_rows else []),
        *(["add"] if with_bias and not fuse else []),
        "view",
    ]
    additions = [event for event in f.events if event[0] == "add"]
    assert len(additions) == int(with_bias and not fuse)
    if additions:
        assert additions[0][1].shape == (6, 80) and additions[0][2] is bias


@pytest.mark.upstream_source
@pytest.mark.parametrize("fail_at", ["quantize", "gemm", "add", "view"])
def test_nvfp4_cutlass_bias_preserves_native_failure_and_stops_later_stages(
    upstream_source_root, fail_at
):
    f = _nvfp4_cutlass_bias_boundary(upstream_source_root, 120, True, True, fail_at)
    with pytest.raises(RuntimeError) as raised:
        f.kernel.apply_weights(f.layer, f.x, f.bias)
    assert raised.value is f.error
    stages = [event[0] for event in f.events]
    complete = ["quantize", "gemm", "slice", "contiguous", "add", "view"]
    assert stages == complete[: complete.index(fail_at) + 1]


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("capability", [80, 89])
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    "backend,batch_invariant",
    [("auto", False), ("auto", True), ("marlin", False), ("humming", False)],
)
def test_real_nvfp4_selector_observes_optional_provider_gate_before_compute(
    modules,
    upstream_source_root,
    preloaded,
    capability,
    installed,
    backend,
    batch_invariant,
):
    """真实 selector、准入与安装重绑；包探测是边界，不提供第三方或设备运行证据。"""
    state = _marlin_boundary(modules, upstream_source_root, preloaded)
    state.platform.capability = capability
    package_reads = []

    def package_available(name):
        assert name == "humming"
        package_reads.append(name)
        return installed

    import_utils = modules("vllm.utils.import_utils", has_cutedsl=lambda: False)
    original = function(
        upstream_source_root / "vllm/utils/import_utils.py",
        "has_humming",
        {"_has_module": package_available},
    )
    import_utils.has_humming = original
    if not preloaded:
        load_patch("vllm_sail/patch/enhancement/import_gates.py")
    linear = _nvfp4_selection_boundary(
        state, modules, upstream_source_root, backend, installed
    )
    humming = sys.modules["vllm.model_executor.kernels.linear.nvfp4.humming"]
    # 模拟真实模块级 from-import；旧测试中的假定可用 provider 边界保持不变。
    humming.has_humming = import_utils.has_humming
    if preloaded:
        load_patch("vllm_sail/patch/enhancement/import_gates.py")
    assert humming.has_humming is import_utils.has_humming
    assert humming.has_humming is not original
    assert not humming.has_humming()
    assert package_reads == ["humming"]

    def forbidden_provider():
        pytest.fail("80/89 的准入必须在查询 FlashInfer provider 前拒绝")

    flashinfer_path = (
        upstream_source_root / "vllm/model_executor/kernels/linear/nvfp4/flashinfer.py"
    )
    cute = linear.FlashInferCuteDslNvFp4W4A16LinearKernel
    cute.is_supported = classmethod(
        function(
            flashinfer_path,
            "FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported",
            {
                "current_platform": state.platform,
                "has_flashinfer_bf16_fp4": forbidden_provider,
            },
        )
    )
    for explicit_capability in (None, capability):
        assert cute.is_supported(explicit_capability) == (
            False,
            "FlashInfer CuTe-DSL W4A16 requires sm_100 or sm_12x",
        )
    linear._POSSIBLE_NVFP4_KERNELS[state.platform._enum].insert(0, cute)
    linear.envs.VLLM_BATCH_INVARIANT = batch_invariant
    # 无可用 W4A16 时应给出诊断，不能进入被排除的 Marlin/Humming 计算路径。
    for select in (
        linear.init_nvfp4_linear_kernel,
        *(sys.modules[name].init_nvfp4_linear_kernel for name in state.nvfp4_consumers),
    ):
        assert select is linear.init_nvfp4_linear_kernel
        with pytest.raises(
            ValueError, match="Marlin FP4 not available|Humming is not installed"
        ):
            select(use_a16=True)
    assert package_reads and set(package_reads) == {"humming"}
    state.platform.ppu = False
    assert humming.has_humming() is installed
    assert linear.HummingNvFp4LinearKernel.is_supported() == (
        (True, None) if installed else (False, "Humming is not installed")
    )


def _real_nvfp4_consumer_call(state, root, consumer, use_a16):
    """真实入口 body；模块、基类、分配、量化和 kernel post-load 是测试边界。"""
    linear = state.linear
    selected, processed, exposed = [], [], []
    for kernel_cls in (linear.HummingNvFp4LinearKernel, linear.MarlinNvFp4LinearKernel):
        kernel_cls.process_weights_after_loading = lambda self, layer: processed.append(
            layer
        )

    def select(*args, **kwargs):
        selected.append((args, kwargs))
        return linear.init_nvfp4_linear_kernel(*args, **kwargs)

    # 复用 fixture 中的消费者 module 与活跃 globals；不是 installed module 导入证据。
    provider = sys.modules[consumer]
    assert provider.init_nvfp4_linear_kernel is linear.init_nvfp4_linear_kernel
    provider.init_nvfp4_linear_kernel = select
    provider.logger = linear.logger
    base = root / "vllm/model_executor/layers/quantization"
    layer = NS(weight=NS(shape=(128, 128)))
    if consumer.endswith("modelopt"):
        path = base / "modelopt.py"
        key_type = type("QuantKey", (), {})
        key = key_type()
        key.dtype = object()
        provider.kInt8Static = object()
        provider.QuantKey, provider.FP4_DTYPE = key_type, key.dtype
        function(path, "select_linear_kernel", provider.__dict__)
        provider.Shapes = lambda *args: args
        provider.RuntimeDtypes = lambda input_dtype, out_dtype: NS(
            input_dtype=input_dtype, out_dtype=out_dtype
        )
        provider.WEIGHT, provider.ACT = object(), object()
        provider.expose_input_quant_key = lambda layer, kernel: exposed.append(kernel)
        scheme_calls = []
        scheme = NS(create_weights=lambda *args: scheme_calls.append(args))
        method = NS(
            spec=NS(weight=key, activation=None if use_a16 else object()),
            ctx=object(),
            wkey=scheme,
            akey=None if use_a16 else scheme,
            fmt=NS(
                extra_weights=lambda *args: None,
                kernel_weight_shape=lambda layer: layer.weight.shape,
            ),
            input_dtype=object(),
            out_dtype=object(),
        )
        create = function(
            path, "ModelOptLinearMethod.create_weights", provider.__dict__
        )

        def invoke():
            return create(method, layer, 128, [128], 128, 128, object())

        expected_call = ((), {"use_a16": use_a16})
    else:
        if consumer.endswith("compressed_tensors_w4a4_nvfp4"):
            path = base / "compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py"
            class_name = "CompressedTensorsW4A4Fp4"
            provider.CompressedTensorsScheme = object
            invoke_args = {"use_a16": use_a16}
            expected_call = ((), invoke_args)
        elif consumer.endswith("quark_nvfp4"):
            path = base / "quark/schemes/quark_nvfp4.py"
            class_name = "QuarkNVFP4"
            # 保留真实 class 编译产生的 __class__ closure，使零参数 super 正常工作。
            provider.QuarkScheme = type(
                "QuarkScheme", (), {"__init__": lambda self, *args: None}
            )
            provider.kNvfp4Static, provider.kNvfp4Dynamic = object(), object()
            provider.EmulationNvFp4LinearKernel = linear.EmulationNvFp4LinearKernel
            invoke_args = {"activation_quant_key": provider.kNvfp4Dynamic}
            expected_call = ((), {})
        else:
            path = base / "online/nvfp4.py"
            class_name = "Nvfp4WeightOnlyOnlineLinearMethod"
            provider.OnlineLinearBase = type(
                "Base", (), {"create_weights": lambda self, *a, **kw: None}
            )
            provider.NVFP4_BLOCK_SIZE = 16
            params_dtype = object()
            expected_call = ((), {"use_a16": True, "act_dtype": params_dtype})
        tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                definition(path, class_name),
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(tree), str(path), "exec"),
            provider.__dict__,
        )
        cls = getattr(provider, class_name)
        if consumer.endswith("online.nvfp4"):
            method = cls()

            def invoke():
                method.create_weights(layer, 128, [128], 128, 128, params_dtype)
                return method
        else:

            def invoke():
                return cls(**invoke_args)

    return NS(
        invoke=invoke,
        selected=selected,
        expected_call=expected_call,
        processed=processed,
        exposed=exposed,
        layer=layer,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("backend", ["auto", "marlin"])
@pytest.mark.parametrize(
    "consumer_index,use_a16",
    [(0, False), (0, True), (1, False), (1, True), (2, True), (3, False)],
)
def test_real_nvfp4_consumers_keep_actual_call_shapes_and_ppu_selection(
    modules,
    upstream_source_root,
    preloaded,
    backend,
    consumer_index,
    use_a16,
):
    state = _marlin_boundary(modules, upstream_source_root, preloaded)
    linear = _nvfp4_selection_boundary(
        state, modules, upstream_source_root, backend, True
    )
    consumer = state.nvfp4_consumers[consumer_index]
    call = _real_nvfp4_consumer_call(state, upstream_source_root, consumer, use_a16)
    if backend == "marlin":
        with pytest.raises(ValueError, match="Marlin FP4 not available"):
            call.invoke()
        assert not call.processed and not call.exposed
    else:
        result = call.invoke()
        if consumer_index in (0, 3):
            assert isinstance(result.kernel, linear.HummingNvFp4LinearKernel)
        elif consumer_index == 1:
            assert len(call.exposed) == 1
            assert isinstance(call.exposed[0], linear.HummingNvFp4LinearKernel)
        else:
            assert isinstance(result.kernel, linear.HummingNvFp4LinearKernel)
            assert result.kernel.config.act_dtype is call.expected_call[1]["act_dtype"]
    assert call.selected == [call.expected_call]


@pytest.mark.upstream_source
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("batch_invariant", [False, True])
def test_real_nvfp4_w4a4_preserves_emulation_fallback_but_never_uses_it_for_a16(
    modules,
    upstream_source_root,
    preloaded,
    batch_invariant,
):
    state = _marlin_boundary(modules, upstream_source_root, preloaded)
    linear = _nvfp4_selection_boundary(
        state, modules, upstream_source_root, "auto", False
    )
    path = (
        upstream_source_root / "vllm/model_executor/kernels/linear/nvfp4/emulation.py"
    )
    for name in ("is_supported", "can_implement"):
        setattr(
            linear.EmulationNvFp4LinearKernel,
            name,
            classmethod(function(path, f"EmulationNvFp4LinearKernel.{name}", {})),
        )
    linear._POSSIBLE_NVFP4_KERNELS[state.platform._enum].append(
        linear.EmulationNvFp4LinearKernel
    )
    linear.envs.VLLM_BATCH_INVARIANT = batch_invariant
    assert isinstance(
        linear.init_nvfp4_linear_kernel(), linear.EmulationNvFp4LinearKernel
    )
    with pytest.raises(ValueError, match="Humming is not installed"):
        linear.init_nvfp4_linear_kernel(use_a16=True)


def _glm_mtp_quant_boundary(modules, patch_utils_module, root, torch):
    """真实 MTP/MRO/mapper 源码；模块导入、配置构造和预测器分配是边界。"""
    from dataclasses import dataclass, field, replace
    from typing import Protocol

    modules(
        "vllm_sail.patch.utils",
        PATCH_REGISTRY=patch_utils_module.PATCH_REGISTRY,
        PatchRecord=patch_utils_module.PatchRecord,
    )
    modules("vllm.platforms", current_platform=NS(is_ppu=lambda: True))
    logger = NS(warning=lambda *a: None, warning_once=lambda *a: None)
    modules("vllm.logger", init_logger=lambda name: logger)
    config_type = type("VllmConfig", (NS,), {})
    modules("vllm.config", VllmConfig=config_type)
    model_dir = root / "vllm/model_executor/models"
    mapper_module = modules(
        "vllm.model_executor.models.utils", field=field, replace=replace, logger=logger
    )
    mapper = dataclass(slots=True)(
        function(model_dir / "utils.py", "WeightsMapper", mapper_module.__dict__)
    )
    mapper_module.WeightsMapper = mapper
    quant_base = type(
        "QuantizationConfig",
        (),
        {
            "apply_vllm_mapper": function(
                root / "vllm/model_executor/layers/quantization/base_config.py",
                "QuantizationConfig.apply_vllm_mapper",
                {},
            )
        },
    )
    interfaces = modules(
        "vllm.model_executor.models.interfaces",
        Protocol=Protocol,
        QuantizationConfig=quant_base,
        torch=torch,
    )
    mixture = function(
        model_dir / "interfaces.py", "MixtureOfExperts", interfaces.__dict__
    )
    supports = function(
        model_dir / "interfaces.py", "SupportsQuant", interfaces.__dict__
    )
    deepseek = function(
        model_dir / "deepseek_v2.py",
        "DeepseekV2MixtureOfExperts",
        {"MixtureOfExperts": mixture, "logger": logger},
    )
    observations = []

    class Predictor(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            quant = vllm_config.quant_config
            observations.append(
                None
                if quant is None
                else (
                    list(quant.fp8_channelwise_layers),
                    dict(quant.packed_modules_mapping),
                )
            )
            self.layers = {}

    mtp_module = modules(
        "vllm.models.glm5next.common.mtp",
        nn=torch.nn,
        DeepseekV2MixtureOfExperts=deepseek,
        NATIVE_MEGA_MOE_BACKENDS={"deep_gemm_mega_moe"},
        Glm5NextMultiTokenPredictor=Predictor,
        Glm5NextMoE=type("Glm5NextMoE", (torch.nn.Module,), {}),
        maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name,
        get_fused_shared_expert_layers=lambda *a, **kw: set(),
    )
    mtp = function(
        root / "vllm/models/glm5next/common/mtp.py", "Glm5NextMTP", mtp_module.__dict__
    )
    apply_mapper = function(
        "vllm_sail/patch/enhancement/mxfp4.py",
        "install._apply_vllm_mapper",
        {"_upstream_apply_mapper": quant_base.apply_vllm_mapper},
    )

    class Config(quant_base):
        apply_vllm_mapper = apply_mapper

        def __init__(self, names):
            self.fp8_channelwise_layers = list(names)
            self.ignored_layers = ["load_only.keep"]
            self.packed_modules_mapping = {"existing": ["left", "right"]}

    return NS(
        module=mtp_module,
        model=mtp,
        supports=supports,
        config=Config,
        vllm_config=config_type,
        observations=observations,
    )


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize(
    "prefix",
    ["model.language_model", "language_model.model", "language_model", "model"],
)
def test_glm_mtp_real_mro_applies_rename_only_mapping_before_predictor(
    modules,
    patch_utils_module,
    upstream_source_root,
    preloaded,
    prefix,
):
    """真实 Torch Module、Protocol 和 __new__；不是完整模型导入或权重数值证据。"""
    from types import MappingProxyType

    torch = pytest.importorskip("torch")
    state = _glm_mtp_quant_boundary(
        modules, patch_utils_module, upstream_source_root, torch
    )
    captured = state.model if preloaded else None
    bases = state.model.__bases__
    assert not issubclass(state.model, state.supports)
    patch_module = load_patch("vllm_sail/patch/enhancement/models/glm5_next_mtp.py")
    assert state.model.__bases__ == (*bases, state.supports)
    if not preloaded:
        namespace = {}
        exec("from vllm.models.glm5next.common.mtp import Glm5NextMTP", namespace)
        captured = namespace["Glm5NextMTP"]
    assert captured is state.model
    records = list(patch_utils_module.PATCH_REGISTRY)
    assert len(records) == 3
    patch_module._enable_supports_quant()
    patch_module._enable_packed_modules_mapping()
    assert patch_utils_module.PATCH_REGISTRY == records
    assert all("e83d82a" in r.affected_versions for r in records)
    mapper = captured.hf_to_vllm_mapper
    assert not hasattr(mapper, "__dict__")
    assert f"{state.module.__name__}.Glm5NextMTP.hf_to_vllm_mapper" in getattr(
        mapper, patch_utils_module.PATCH_MARKER
    )
    # 加载专用丢弃/堆叠/变换不能流入量化层列表；保留原 mapper 以独立比较。
    mapper.orig_to_new_substr = {"load_only": None}
    mapper.orig_to_new_stacked = {"gate_proj": ("wrong_fused", 0)}
    mapper.weights_transforms = [lambda weights: iter(())]
    names = [
        f"{prefix}.layers.45.mlp.shared_experts.{name}"
        for name in ("gate_proj", "up_proj", "down_proj")
    ]
    names.append("load_only.keep")
    expected = [
        f"model.layers.45.mlp.shared_experts.{name}"
        for name in ("gate_proj", "up_proj", "down_proj")
    ] + ["load_only.keep"]
    quant = state.config(names)
    target_snapshot = list(quant.fp8_channelwise_layers)
    config = state.vllm_config(
        quant_config=quant,
        kernel_config=NS(moe_backend="auto"),
        model_config=NS(hf_text_config=NS(num_nextn_predict_layers=1, n_group=1)),
        parallel_config=NS(enable_eplb=False),
    )
    first = captured(vllm_config=config)
    second = captured(vllm_config=config)
    assert first.quant_config is second.quant_config is quant
    assert quant.fp8_channelwise_layers == expected and target_snapshot == names
    packed = {"existing": ["left", "right"], "gate_up_proj": ["gate_proj", "up_proj"]}
    assert state.observations == [(expected, packed), (expected, packed)]
    assert quant.ignored_layers == ["load_only.keep"]
    assert mapper.orig_to_new_stacked and mapper.weights_transforms
    assert mapper.map_name("load_only.keep") is None
    assert mapper.get_rename_mapper().map_name("load_only.keep") == "load_only.keep"
    skipped = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/quant_utils.py",
        "is_layer_skipped",
        {"MappingProxyType": MappingProxyType},
    )
    assert skipped(
        "model.layers.45.mlp.shared_experts.gate_up_proj",
        expected,
        packed,
        match_mode="substring",
    )
    with pytest.raises(ValueError, match="some but not all shards"):
        skipped(
            "model.layers.45.mlp.shared_experts.gate_up_proj",
            expected[:1],
            packed,
            match_mode="substring",
        )
    config.quant_config = None
    unquantized = captured(vllm_config=config)
    assert unquantized.quant_config is None and state.observations[-1] is None


@pytest.mark.upstream_source
@pytest.mark.requires_runtime
@pytest.mark.parametrize("is_mtp", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_glm_real_decoder_keeps_mtp_topk_layout_and_unsummed_residual(
    upstream_source_root,
    is_mtp,
    sequence_parallel,
):
    """完整真实 decoder constructor/forward；注意力、MLP、norm 是显式数值边界。"""
    torch = pytest.importorskip("torch")
    attention_calls, mlp_calls, kda_calls = [], [], []

    class Attention(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            attention_calls.append(kwargs)
            self.o_proj = NS(reduce_results=True)

        def forward(self, *, hidden_states, positions):
            return hidden_states * 2

    class MLP(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            mlp_calls.append(kwargs)

        def forward(self, hidden_states):
            return hidden_states + 3

    class Norm(torch.nn.Module):
        def __init__(self, hidden_size, eps):
            super().__init__()

        def forward(self, hidden_states, residual=None):
            return hidden_states if residual is None else (hidden_states, residual)

    def is_kda(config, layer):
        kda_calls.append(layer)
        return False

    decoder = function(
        upstream_source_root / "vllm/models/glm5next/common/model.py",
        "Glm5NextDecoderLayer",
        {
            "nn": torch.nn,
            "torch": torch,
            "_is_moe": lambda config: False,
            "_is_kda_layer": is_kda,
            "_mla_dequantized_at_load": function(
                upstream_source_root / "vllm/models/glm5next/common/model.py",
                "_mla_dequantized_at_load",
                {},
            ),
            "Glm5NextMLAAttention": Attention,
            "Glm5NextMLP": MLP,
            "Glm5NextMoE": type("MoE", (), {}),
            "RMSNorm": Norm,
        },
    )
    topk, quant, cache = object(), object(), object()
    config = NS(
        hidden_size=4,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        n_routed_experts=None,
        mhc=is_mtp,
        v_head_dim=4,
        kv_lora_rank=4,
        num_attention_heads=1,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=4,
        max_position_embeddings=64,
        mla_use_nope=True,
        mlp_layer_types=["dense"] * 2,
        intermediate_size=8,
        hidden_act="silu",
        swiglu_limit=None,
    )
    owner = decoder(
        NS(
            cache_config=cache,
            quant_config=quant,
            parallel_config=NS(use_sequence_parallel_moe=sequence_parallel),
        ),
        config,
        2 if is_mtp else 0,
        prefix="model.layers.2",
        topk_indices_buffer=topk,
        is_mtp_layer=is_mtp,
    )
    assert kda_calls == ([] if is_mtp else [0])
    assert owner.layer_kind == "mla" and len(attention_calls) == len(mlp_calls) == 1
    assert attention_calls[0]["shardq_eligible"] is (not is_mtp)
    assert attention_calls[0]["topk_indices_buffer"] is topk
    assert attention_calls[0]["quant_config"] is quant
    assert attention_calls[0]["cache_config"] is cache
    assert mlp_calls[0]["quant_config"] is quant
    assert owner.self_attn.o_proj.reduce_results is (is_mtp or not sequence_parallel)
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    result, residual, post, comb = owner(torch.arange(2), hidden)
    expected = hidden * (2 if is_mtp else 3) + 3
    assert torch.equal(result, expected) and residual is hidden
    assert post is None and comb is None
