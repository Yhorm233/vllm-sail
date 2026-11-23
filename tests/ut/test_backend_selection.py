# SPDX-License-Identifier: Apache-2.0
"""Category-2: per-chip backend selection is the plan's core gate.

The user's requirement is unambiguous: *if the selector does not return the
expected backend for a chip, this UT must FAIL* -- never skip, never warn. Every
assertion below is therefore a hard ``assert`` on an exact ordered candidate
list, and the parameterization covers both ZW-810E ``(8, 0)`` and ZW-890P
``(8, 9)``.

This module is CPU-only and imports neither ``torch`` nor ``vllm``. Instead of
mocking the selectors, it compiles the *production* functions in isolation (AST
extraction with a stub namespace) and injects fakes for their function-local
imports through ``sys.modules``. That way the real branch logic runs, and the
assertions read the genuine candidate ordering.

Two selection layers are covered and kept distinct:

* **MoE oracle candidate lists** (``registry/moe_backends/*.py``) -- the ordered
  backends offered per quant path. ACEXT is present only on ZW-810E; the fp8 /
  mxfp4 DeepGEMM paths are gated to ZW-890P.
* **Linear-kernel selection plan** (``registry/linear_kernels``) and
  **attention backend priorities** (``platform._get_backend_priorities``).

The fp8 hardware gate lives at the *kernel* ``is_supported`` layer, not in the
fp8 oracle candidate list (which is capability-independent); both facts are
asserted so the "no fp8 DeepGEMM on 810E" claim is proven where it is enforced.
"""

from __future__ import annotations

import ast
import copy
import enum
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import CHIP_PROFILES

ROOT = Path(__file__).resolve().parents[2]
CHIPS = sorted(CHIP_PROFILES)

# Forbidden attention backends: PPU does not ship these upstream CUDA paths.
FORBIDDEN_ATTENTION = frozenset({"FLASHINFER", "CUTLASS_MLA", "CUDNN", "CUDNN_ATTN"})


# ---------------------------------------------------------------------------
# AST extraction helpers (Python 3.9-safe; ``tests/support/source.py`` uses the
# ``X | Y`` isinstance form that only runs on 3.10+, so it cannot be reused).
# ---------------------------------------------------------------------------
def _find_def(tree: ast.AST, qualname: str) -> ast.AST:
    node: ast.AST = tree
    for part in qualname.split("."):
        match = next(
            (
                child
                for child in node.body  # type: ignore[attr-defined]
                if (
                    isinstance(child, ast.FunctionDef)
                    or isinstance(child, ast.ClassDef)
                )
                and child.name == part
            ),
            None,
        )
        assert match is not None, f"missing definition {qualname!r} (at {part!r})"
        node = match
    return node


def extract(relative: str, qualname: str, namespace: dict):
    """Compile one production function/method with ``namespace`` as globals."""
    path = ROOT / relative
    node = copy.deepcopy(_find_def(ast.parse(path.read_text()), qualname))
    node.decorator_list = []  # type: ignore[attr-defined]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[qualname.rpartition(".")[2]]


def platform_double(chip: str) -> types.SimpleNamespace:
    """A ``current_platform`` double carrying only this chip's capability."""
    capability = tuple(CHIP_PROFILES[chip].capability)

    def is_device_capability(value) -> bool:
        if isinstance(value, tuple):
            return capability == tuple(value)
        return capability[0] * 10 + capability[1] >= value

    return types.SimpleNamespace(
        is_ppu=lambda: True,
        is_cuda=lambda: True,
        get_device_capability=lambda device_id=0: capability,
        get_device_name=lambda device_id=0: CHIP_PROFILES[chip].name,
        is_device_capability=is_device_capability,
    )


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs) -> None:
    """Bind ``name`` in ``sys.modules`` so function-local imports resolve."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# Stub enums mirroring the upstream/PPU-extended backend members. Only ``.name``
# matters: the selectors build lists of enum members and the assertions compare
# their names, exactly as ``int8.py``'s own logging does.
# ---------------------------------------------------------------------------
class _ActivationFormat(enum.Enum):
    Standard = "Standard"
    BatchedExperts = "BatchedExperts"


class _UnquantizedMoeBackend(enum.Enum):
    PPU_DEEPGEMM = "PPU_DEEPGEMM"
    BATCHED_PPU_DEEPGEMM = "BATCHED_PPU_DEEPGEMM"
    ACEXT = "ACEXT"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"


class _Int8MoeBackend(enum.Enum):
    PPU_DEEPGEMM = "PPU_DEEPGEMM"
    BATCHED_PPU_DEEPGEMM = "BATCHED_PPU_DEEPGEMM"
    ACEXT = "ACEXT"
    TRITON = "TRITON"
    MARLIN = "MARLIN"


class _Fp8MoeBackend(enum.Enum):
    DEEPGEMM = "DEEPGEMM"
    PPU_DEEPGEMM = "PPU_DEEPGEMM"
    BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
    BATCHED_PPU_DEEPGEMM = "BATCHED_PPU_DEEPGEMM"
    TRITON = "TRITON"


class _AttentionBackendEnum(enum.Enum):
    FLASHMLA = "FLASHMLA"
    TRITON_MLA = "TRITON_MLA"
    FLASHMLA_SPARSE = "FLASHMLA_SPARSE"
    FLASH_ATTN = "FLASH_ATTN"
    TRITON_ATTN = "TRITON_ATTN"
    FLEX_ATTENTION = "FLEX_ATTENTION"


def _moe_config(*, batched: bool = False, has_bias: bool = False, backend="auto"):
    return types.SimpleNamespace(
        activation_format=(
            _ActivationFormat.BatchedExperts
            if batched or backend == "batched_triton"
            else _ActivationFormat.Standard
        ),
        moe_backend=backend,
        has_bias=has_bias,
    )


# ---------------------------------------------------------------------------
# MoE oracle candidate lists.
# ---------------------------------------------------------------------------
def _unquantized_backends(
    monkeypatch: pytest.MonkeyPatch,
    chip: str,
    *,
    acext_available: bool,
    deep_gemm_available: bool,
    requested: str | None = None,
) -> list[str]:
    _stub_module(monkeypatch, "vllm.platforms", current_platform=platform_double(chip))
    _stub_module(
        monkeypatch,
        "vllm_sail.model_executor.layers.fused_moe.experts.acext",
        is_acext_supported=lambda: acext_available,
    )
    _stub_module(
        monkeypatch,
        "vllm_sail.utils.deep_gemm",
        is_deep_gemm_supported=lambda: deep_gemm_available,
    )
    namespace = {
        "mk": types.SimpleNamespace(FusedMoEActivationFormat=_ActivationFormat),
        "ppu_envs": types.SimpleNamespace(VLLM_SAIL_MOE_BACKEND=requested),
        "UnquantizedMoeBackend": _UnquantizedMoeBackend,
        "_upstream_get_priority_backends": lambda config: [],
    }
    fn = extract(
        "vllm_sail/registry/moe_backends/unquantized.py",
        "_get_priority_backends",
        namespace,
    )
    return [b.name for b in fn(_moe_config())]


def _int8_backends(
    monkeypatch: pytest.MonkeyPatch,
    chip: str,
    *,
    acext_available: bool,
    deep_gemm_available: bool,
) -> list[str]:
    _stub_module(monkeypatch, "vllm.platforms", current_platform=platform_double(chip))
    _stub_module(
        monkeypatch,
        "vllm_sail.model_executor.layers.fused_moe.experts.acext",
        is_acext_supported=lambda: acext_available,
    )
    _stub_module(
        monkeypatch,
        "vllm_sail.utils.deep_gemm",
        is_deep_gemm_supported=lambda: deep_gemm_available,
    )
    namespace = {
        "Int8MoeBackend": _Int8MoeBackend,
        "_upstream_get_priority_backends": lambda config: [
            _Int8MoeBackend.TRITON,
            _Int8MoeBackend.MARLIN,
        ],
        "ppu_envs": types.SimpleNamespace(VLLM_SAIL_MOE_BACKEND=None),
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
    }
    fn = extract(
        "vllm_sail/registry/moe_backends/int8.py", "_get_priority_backends", namespace
    )
    return [b.name for b in fn(_moe_config())]


def _fp8_backends(monkeypatch: pytest.MonkeyPatch, chip: str) -> list[str]:
    import vllm_sail.envs as ppu_envs

    _stub_module(monkeypatch, "vllm.platforms", current_platform=platform_double(chip))
    namespace = {
        "Fp8MoeBackend": _Fp8MoeBackend,
        "_upstream_get_priority_backends": lambda config, weight_key, activation_key: [
            _Fp8MoeBackend.DEEPGEMM,
            _Fp8MoeBackend.BATCHED_DEEPGEMM,
            _Fp8MoeBackend.TRITON,
        ],
        "ppu_envs": ppu_envs,
    }
    fn = extract(
        "vllm_sail/registry/moe_backends/fp8.py", "_get_priority_backends", namespace
    )
    return [b.name for b in fn(None, None, None)]


def _mxfp4_activation_key(monkeypatch: pytest.MonkeyPatch, chip: str):
    """Return the activation key ``select_deepseek_v4_mxfp4_moe_backend`` picks."""
    _stub_module(monkeypatch, "vllm.platforms", current_platform=platform_double(chip))
    seen: dict = {}

    def _select(config, activation_key=None):
        seen["activation_key"] = activation_key
        return (None, None)

    namespace = {
        "oracle": types.SimpleNamespace(
            select_mxfp4_moe_backend=_select, kMxfp4Dynamic="kMxfp4Dynamic"
        ),
        "_upstream_select_deepseek_v4": lambda config: (None, None),
    }
    fn = extract(
        "vllm_sail/registry/moe_backends/mxfp4.py",
        "select_deepseek_v4_mxfp4_moe_backend",
        namespace,
    )
    fn(_moe_config(backend="auto"))
    return seen["activation_key"]


# ---------------------------------------------------------------------------
# Linear-kernel selection plan and per-chip support gates.
# ---------------------------------------------------------------------------
def _linear_selection_plan_names() -> dict[str, list[str]]:
    """Statically read ``_selection_plan()``'s returned (type, [classes]) pairs."""
    path = ROOT / "vllm_sail/registry/linear_kernels/__init__.py"
    fn = _find_def(ast.parse(path.read_text()), "_selection_plan")
    returned = next(node for node in ast.walk(fn) if isinstance(node, ast.Return))
    plan: dict[str, list[str]] = {}
    for entry in returned.value.elts:  # type: ignore[attr-defined]
        kernel_type = entry.elts[0].value
        classes = [name.id for name in entry.elts[1].elts]
        plan[kernel_type] = classes
    return plan


def _kernel_is_supported(chip: str, class_name: str) -> bool:
    supported, _reason = extract(
        "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py",
        f"{class_name}.is_supported",
        {
            "current_platform": platform_double(chip),
            "is_deep_gemm_supported": lambda: True,
        },
    )(None)
    return bool(supported)


def _attention_priorities(chip: str, use_mla: bool) -> list[str]:
    namespace = {"AttentionBackendEnum": _AttentionBackendEnum}
    fn = extract("vllm_sail/platform.py", "_get_backend_priorities", namespace)
    capability = tuple(CHIP_PROFILES[chip].capability)
    return [b.name for b in fn(use_mla, capability)]


# ===========================================================================
# MoE backend selection.
# ===========================================================================
@pytest.mark.parametrize("chip", CHIPS)
def test_unquantized_moe_candidate_ordering(monkeypatch, chip) -> None:
    """BF16 MoE: ACEXT is offered only on ZW-810E; DeepGEMM leads on both."""
    acext_available = CHIP_PROFILES[chip].acext
    got = _unquantized_backends(
        monkeypatch, chip, acext_available=acext_available, deep_gemm_available=True
    )
    if chip == "810e":
        expected = [
            "PPU_DEEPGEMM",
            "ACEXT",
            "BATCHED_PPU_DEEPGEMM",
            "TRITON",
            "BATCHED_TRITON",
        ]
    else:
        expected = [
            "PPU_DEEPGEMM",
            "BATCHED_PPU_DEEPGEMM",
            "TRITON",
            "BATCHED_TRITON",
        ]
    assert got == expected, f"{chip}: unquantized MoE candidates {got} != {expected}"
    assert ("ACEXT" in got) is (chip == "810e")


@pytest.mark.parametrize("chip", CHIPS)
def test_int8_moe_candidate_ordering(monkeypatch, chip) -> None:
    """W8A8-INT8 MoE: PPU DeepGEMM/ACEXT are prepended; ACEXT only on 810E."""
    acext_available = CHIP_PROFILES[chip].acext
    got = _int8_backends(
        monkeypatch, chip, acext_available=acext_available, deep_gemm_available=True
    )
    if chip == "810e":
        expected = [
            "PPU_DEEPGEMM",
            "ACEXT",
            "BATCHED_PPU_DEEPGEMM",
            "TRITON",
            "MARLIN",
        ]
    else:
        expected = ["PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM", "TRITON", "MARLIN"]
    assert got == expected, f"{chip}: int8 MoE candidates {got} != {expected}"
    assert ("ACEXT" in got) is (chip == "810e")


@pytest.mark.parametrize("chip", CHIPS)
def test_fp8_moe_candidate_insertion(monkeypatch, chip) -> None:
    """FP8 MoE: PPU DeepGEMM slots in right after the upstream DeepGEMM entry.

    The fp8 oracle candidate list is deliberately capability-independent; the
    ``(8, 0)`` exclusion is enforced at the kernel ``is_supported`` layer (see
    ``test_fp8_deepgemm_hardware_gate_excludes_810e``).
    """
    got = _fp8_backends(monkeypatch, chip)
    expected = [
        "DEEPGEMM",
        "PPU_DEEPGEMM",
        "BATCHED_DEEPGEMM",
        "BATCHED_PPU_DEEPGEMM",
        "TRITON",
    ]
    assert got == expected, f"{chip}: fp8 MoE candidates {got} != {expected}"
    assert got.index("PPU_DEEPGEMM") == got.index("DEEPGEMM") + 1


@pytest.mark.parametrize("chip", CHIPS)
def test_fp8_deepgemm_hardware_gate_excludes_810e(chip) -> None:
    """The fp8 DeepGEMM *selection* gate: unavailable on ZW-810E ``(8, 0)``."""
    supported = _kernel_is_supported(chip, "PPUDeepGemmFP8ScaledMMLinearKernel")
    assert supported is (chip == "890p")
    # And the chip profile agrees: fp8 is a ZW-890P-only hardware capability.
    assert CHIP_PROFILES[chip].fp8 is (chip == "890p")


@pytest.mark.parametrize("chip", CHIPS)
def test_mxfp4_deepseek_v4_activation_gate(monkeypatch, chip) -> None:
    """MXFP4 native W4A4 DeepGEMM (dynamic act key) is a ZW-890P-only path."""
    activation_key = _mxfp4_activation_key(monkeypatch, chip)
    if chip == "890p":
        assert activation_key == "kMxfp4Dynamic", activation_key
    else:
        assert activation_key is None, activation_key


@pytest.mark.parametrize("chip", CHIPS)
def test_selectable_deepgemm_moe_paths_match_matrix(monkeypatch, chip) -> None:
    """Roll up every MoE quant path into the authoritative DeepGEMM matrix.

    This is the user's core requirement expressed as one hard assertion: the
    set of DeepGEMM MoE paths actually selectable on a chip must equal the
    confirmed capability matrix -- anything else fails the test.
    """
    acext_available = CHIP_PROFILES[chip].acext
    unquantized = _unquantized_backends(
        monkeypatch, chip, acext_available=acext_available, deep_gemm_available=True
    )
    int8 = _int8_backends(
        monkeypatch, chip, acext_available=acext_available, deep_gemm_available=True
    )
    selectable = set()
    if "PPU_DEEPGEMM" in unquantized:
        selectable.add("bf16")
    if "PPU_DEEPGEMM" in int8:
        selectable.add("w8a8-int8")
    if _kernel_is_supported(chip, "PPUDeepGemmFP8ScaledMMLinearKernel"):
        selectable.add("w8a8-fp8")
    if _mxfp4_activation_key(monkeypatch, chip) == "kMxfp4Dynamic":
        selectable.add("mxfp4")

    expected = {
        "810e": {"bf16", "w8a8-int8"},
        "890p": {"bf16", "w8a8-int8", "w8a8-fp8", "mxfp4"},
    }[chip]
    assert selectable == expected, (
        f"{chip}: DeepGEMM MoE paths {selectable} != {expected}"
    )


# ===========================================================================
# Linear-kernel selection plan.
# ===========================================================================
def test_linear_kernel_selection_plan_structure() -> None:
    """The prepend order PPU writes into upstream's CUDA bucket is fixed."""
    assert _linear_selection_plan_names() == {
        "mxfp8": ["PPUEmulationMxfp8LinearKernel"],
        "int8": ["PPUInt8ScaledMMLinearKernel"],
        "fp8": [
            "PPUDeepGemmFP8ScaledMMLinearKernel",
            "PPUCutlassFP8ScaledMMLinearKernel",
        ],
        "fp8_block": [
            "PPUDeepGemmFp8BlockScaledMMKernel",
            "PPUCutlassFp8BlockScaledMMKernel",
        ],
    }


@pytest.mark.parametrize("chip", CHIPS)
def test_linear_kernel_support_per_chip(chip) -> None:
    """int8/Cutlass kernels run on both chips; DeepGEMM fp8 needs ZW-890P."""
    expected = {
        "PPUInt8ScaledMMLinearKernel": True,
        "PPUDeepGemmFP8ScaledMMLinearKernel": chip == "890p",
        "PPUCutlassFP8ScaledMMLinearKernel": True,
        "PPUDeepGemmFp8BlockScaledMMKernel": chip == "890p",
        "PPUCutlassFp8BlockScaledMMKernel": True,
    }
    for class_name, want in expected.items():
        got = _kernel_is_supported(chip, class_name)
        assert got is want, f"{chip}: {class_name}.is_supported={got}, want {want}"


@pytest.mark.parametrize("chip", CHIPS)
def test_first_supported_fp8_kernel_differs_by_chip(chip) -> None:
    """Selection takes the first supported candidate, so the fp8 winner is
    DeepGEMM on ZW-890P but Cutlass on ZW-810E (DeepGEMM gated out)."""
    plan = _linear_selection_plan_names()
    for kernel_type in ("fp8", "fp8_block"):
        winner = next(
            name for name in plan[kernel_type] if _kernel_is_supported(chip, name)
        )
        want = plan[kernel_type][1] if chip == "810e" else plan[kernel_type][0]
        assert winner == want, f"{chip}/{kernel_type}: selected {winner}, want {want}"
        assert ("DeepGemm" in winner) is (chip == "890p")


# ===========================================================================
# Attention backend priorities.
# ===========================================================================
@pytest.mark.parametrize("chip", CHIPS)
def test_attention_backend_priorities(chip) -> None:
    """MLA and non-MLA priority lists are exact and identical on both chips."""
    assert _attention_priorities(chip, use_mla=True) == [
        "FLASHMLA",
        "TRITON_MLA",
        "FLASHMLA_SPARSE",
    ]
    assert _attention_priorities(chip, use_mla=False) == [
        "FLASH_ATTN",
        "TRITON_ATTN",
        "FLEX_ATTENTION",
    ]


@pytest.mark.parametrize("chip", CHIPS)
def test_attention_never_offers_forbidden_backends(chip) -> None:
    """FlashInfer / CUTLASS-MLA / cuDNN are not available on PPU."""
    for use_mla in (True, False):
        offered = set(_attention_priorities(chip, use_mla))
        assert offered.isdisjoint(FORBIDDEN_ATTENTION), (chip, use_mla, offered)


@pytest.mark.parametrize("chip", CHIPS)
def test_vit_attention_backends_are_cuda_subset_source(chip) -> None:
    """``get_supported_vit_attn_backends`` must never list a forbidden backend.

    Verified structurally (CPU-only): every ``AttentionBackendEnum`` member the
    method can return is in the PPU-allowed vision set.
    """
    path = ROOT / "vllm_sail/platform.py"
    fn = _find_def(
        ast.parse(path.read_text()), "PPUPlatform.get_supported_vit_attn_backends"
    )
    referenced = {
        node.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "AttentionBackendEnum"
    }
    assert referenced == {"FLASH_ATTN", "TRITON_ATTN", "TORCH_SDPA"}
    assert referenced.isdisjoint(FORBIDDEN_ATTENTION)
