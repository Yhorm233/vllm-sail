# SPDX-License-Identifier: Apache-2.0
"""Category-2: ``VLLM_SAIL_*`` env vars override backend auto-selection.

CPU-only; imports neither ``torch`` nor ``vllm``. The real ``vllm_sail.envs``
module *is* imported (it is pure-stdlib), so the override paths read genuine
environment values and ``is_set`` semantics rather than a mock. The selectors
themselves are the production functions, compiled in isolation via AST
extraction with stub namespaces.

Covered:

* ``envs.is_set()`` distinguishes "explicitly set" (including the ``VLLM_PPU_*``
  alias) from "left at the default".
* Illegal ``VLLM_SAIL_MOE_BACKEND`` / ``VLLM_SAIL_DENSE_BACKEND`` values raise
  ``ValueError``; every legal choice is accepted verbatim.
* ``VLLM_SAIL_MOE_BACKEND`` overrides the unquantized / fp8 MoE candidate lists.
* ``VLLM_SAIL_DENSE_BACKEND`` overrides the dense fp8 DeepGEMM decision.
* ``VLLM_SAIL_DENSE_BF16_DEEPGEMM`` gates BF16 dense DeepGEMM auto-enable, and
  only on ZW-890P (PPU 1.5). ZW-810E hardware supports BF16 DeepGEMM but its
  default dense backend stays ACEXT, so the env does not flip it there -- the
  "capability support != default auto-selection" distinction the user called out.
* With every env unset, selection falls back exactly to the defaults asserted in
  ``test_backend_selection.py`` (the default path is not polluted by env logic).
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
from vllm_sail import envs as ppu_envs

ROOT = Path(__file__).resolve().parents[2]
CHIPS = sorted(CHIP_PROFILES)

# Canonical env name -> its backward-compatible alias.
_ENV_ALIASES = {
    "VLLM_SAIL_MOE_BACKEND": "VLLM_PPU_MOE_BACKEND",
    "VLLM_SAIL_DENSE_BACKEND": "VLLM_PPU_DENSE_BACKEND",
    "VLLM_SAIL_DENSE_BF16_DEEPGEMM": "VLLM_PPU_DENSE_BF16_DEEPGEMM",
}


# ---------------------------------------------------------------------------
# AST extraction helpers (Python 3.9-safe; see test_backend_selection.py).
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
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def _clear_env(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Unset a variable and its alias so a default path is truly default."""
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(_ENV_ALIASES.get(name, name), raising=False)


class _ActivationFormat(enum.Enum):
    Standard = "Standard"
    BatchedExperts = "BatchedExperts"


class _UnquantizedMoeBackend(enum.Enum):
    PPU_DEEPGEMM = "PPU_DEEPGEMM"
    BATCHED_PPU_DEEPGEMM = "BATCHED_PPU_DEEPGEMM"
    ACEXT = "ACEXT"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"


class _Fp8MoeBackend(enum.Enum):
    DEEPGEMM = "DEEPGEMM"
    PPU_DEEPGEMM = "PPU_DEEPGEMM"
    BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
    BATCHED_PPU_DEEPGEMM = "BATCHED_PPU_DEEPGEMM"
    TRITON = "TRITON"


_TORCH = types.SimpleNamespace(bfloat16="bf16")


def _moe_config(*, batched: bool = False, has_bias: bool = False, backend="auto"):
    return types.SimpleNamespace(
        moe_parallel_config=types.SimpleNamespace(
            use_batched_activation_format=batched
        ),
        moe_backend=backend,
        has_bias=has_bias,
    )


# ---------------------------------------------------------------------------
# Selector drivers (env values flow in through the real ``ppu_envs`` module).
# ---------------------------------------------------------------------------
def _unquantized_backends(
    monkeypatch: pytest.MonkeyPatch, chip: str, *, acext_available: bool
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
        is_deep_gemm_supported=lambda: True,
    )
    namespace = {
        "mk": types.SimpleNamespace(FusedMoEActivationFormat=_ActivationFormat),
        "ppu_envs": ppu_envs,
        "UnquantizedMoeBackend": _UnquantizedMoeBackend,
        "_upstream_get_priority_backends": lambda config: [],
    }
    fn = extract(
        "vllm_sail/registry/moe_backends/unquantized.py",
        "_get_priority_backends",
        namespace,
    )
    return [b.name for b in fn(_moe_config())]


def _fp8_backends(monkeypatch: pytest.MonkeyPatch, chip: str) -> list[str]:
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


def _should_use_fp8_linear(weight_shape=(128, 256)) -> bool:
    fn = extract(
        "vllm_sail/utils/deep_gemm.py",
        "should_use_deepgemm_for_fp8_linear",
        {
            "ppu_envs": ppu_envs,
            "torch": _TORCH,
            "is_deep_gemm_supported": lambda: True,
        },
    )
    return bool(fn("bf16", weight_shape))


def _should_use_bf16_linear(chip: str, *, bias: bool, n: int = 64) -> bool:
    platform = platform_double(chip)
    x = types.SimpleNamespace(dtype="bf16", is_contiguous=lambda: True)
    weight = types.SimpleNamespace(
        dtype="bf16", dim=lambda: 2, shape=(n, 16), is_contiguous=lambda: True
    )
    fn = extract(
        "vllm_sail/utils/deep_gemm.py",
        "should_use_deepgemm_for_bf16_linear",
        {
            "ppu_envs": ppu_envs,
            "torch": _TORCH,
            "current_platform": platform,
            "has_deep_gemm": lambda: True,
            "is_ppu_15": lambda: platform.is_device_capability((8, 9)),
        },
    )
    return bool(fn(x, weight, object() if bias else None))


# ===========================================================================
# is_set / validation semantics.
# ===========================================================================
def test_is_set_distinguishes_explicit_from_default(monkeypatch) -> None:
    _clear_env(monkeypatch, "VLLM_SAIL_MOE_BACKEND")
    assert ppu_envs.is_set("VLLM_SAIL_MOE_BACKEND") is False
    assert ppu_envs.VLLM_SAIL_MOE_BACKEND is None

    monkeypatch.setenv("VLLM_SAIL_MOE_BACKEND", "deepgemm")
    assert ppu_envs.is_set("VLLM_SAIL_MOE_BACKEND") is True
    assert ppu_envs.VLLM_SAIL_MOE_BACKEND == "deepgemm"


def test_is_set_honors_legacy_alias(monkeypatch) -> None:
    """The ``VLLM_PPU_*`` alias still counts as an explicit user choice."""
    _clear_env(monkeypatch, "VLLM_SAIL_MOE_BACKEND")
    monkeypatch.setenv("VLLM_PPU_MOE_BACKEND", "acext")
    assert ppu_envs.is_set("VLLM_SAIL_MOE_BACKEND") is True
    assert ppu_envs.VLLM_SAIL_MOE_BACKEND == "acext"


@pytest.mark.parametrize("name", ["VLLM_SAIL_MOE_BACKEND", "VLLM_SAIL_DENSE_BACKEND"])
def test_illegal_backend_value_raises(monkeypatch, name) -> None:
    monkeypatch.setenv(name, "not-a-backend")
    with pytest.raises(ValueError, match="Invalid value"):
        getattr(ppu_envs, name)


@pytest.mark.parametrize("value", list(ppu_envs.GEMM_BACKENDS))
def test_every_legal_choice_is_accepted(monkeypatch, value) -> None:
    monkeypatch.setenv("VLLM_SAIL_DENSE_BACKEND", value)
    assert ppu_envs.VLLM_SAIL_DENSE_BACKEND == value


# ===========================================================================
# MoE backend override.
# ===========================================================================
@pytest.mark.parametrize("chip", CHIPS)
def test_moe_backend_env_overrides_unquantized(monkeypatch, chip) -> None:
    """``VLLM_SAIL_MOE_BACKEND`` forces the unquantized candidate list.

    ``deepgemm`` drops ACEXT even on ZW-810E (where it is otherwise present);
    ``acext`` drops the PPU DeepGEMM entries and keeps ACEXT (only meaningful on
    ZW-810E, since ACEXT is unavailable on ZW-890P).
    """
    acext_available = CHIP_PROFILES[chip].acext

    _clear_env(monkeypatch, "VLLM_SAIL_MOE_BACKEND")
    monkeypatch.setenv("VLLM_SAIL_MOE_BACKEND", "deepgemm")
    got = _unquantized_backends(monkeypatch, chip, acext_available=acext_available)
    assert "ACEXT" not in got
    assert got == [
        "PPU_DEEPGEMM",
        "BATCHED_PPU_DEEPGEMM",
        "TRITON",
        "BATCHED_TRITON",
    ], got

    _clear_env(monkeypatch, "VLLM_SAIL_MOE_BACKEND")
    monkeypatch.setenv("VLLM_SAIL_MOE_BACKEND", "acext")
    got = _unquantized_backends(monkeypatch, chip, acext_available=acext_available)
    assert "PPU_DEEPGEMM" not in got
    # ACEXT only survives where the hardware supports it.
    assert ("ACEXT" in got) is (chip == "810e")
    assert got == (
        ["ACEXT", "TRITON", "BATCHED_TRITON"]
        if chip == "810e"
        else ["TRITON", "BATCHED_TRITON"]
    ), got


@pytest.mark.parametrize("chip", CHIPS)
@pytest.mark.parametrize("value", ["deepgemm", "acext", "triton"])
def test_moe_backend_env_overrides_fp8(monkeypatch, chip, value) -> None:
    """On fp8, only ``deepgemm`` (or unset) keeps the PPU DeepGEMM candidate."""
    _clear_env(monkeypatch, "VLLM_SAIL_MOE_BACKEND")
    monkeypatch.setenv("VLLM_SAIL_MOE_BACKEND", value)
    got = _fp8_backends(monkeypatch, chip)
    if value == "deepgemm":
        assert "PPU_DEEPGEMM" in got
        assert got.index("PPU_DEEPGEMM") == got.index("DEEPGEMM") + 1
    else:
        # An explicit non-deepgemm request short-circuits to the upstream list.
        assert got == ["DEEPGEMM", "BATCHED_DEEPGEMM", "TRITON"], got


# ===========================================================================
# Dense backend override.
# ===========================================================================
def test_dense_backend_env_overrides_fp8_linear(monkeypatch) -> None:
    """``VLLM_SAIL_DENSE_BACKEND`` steers the dense fp8 DeepGEMM decision."""
    _clear_env(monkeypatch, "VLLM_SAIL_DENSE_BACKEND")
    assert _should_use_fp8_linear() is True  # default: DeepGEMM eligible

    monkeypatch.setenv("VLLM_SAIL_DENSE_BACKEND", "deepgemm")
    assert _should_use_fp8_linear() is True

    monkeypatch.setenv("VLLM_SAIL_DENSE_BACKEND", "acext")
    assert _should_use_fp8_linear() is False

    monkeypatch.setenv("VLLM_SAIL_DENSE_BACKEND", "triton")
    assert _should_use_fp8_linear() is False


def test_dense_fp8_linear_still_respects_shape_gate(monkeypatch) -> None:
    """An N not divisible by 64 defeats DeepGEMM regardless of the env."""
    _clear_env(monkeypatch, "VLLM_SAIL_DENSE_BACKEND")
    assert _should_use_fp8_linear((100, 256)) is False


@pytest.mark.parametrize("chip", CHIPS)
def test_dense_bf16_deepgemm_env_autonables_only_on_890p(monkeypatch, chip) -> None:
    """BF16 dense DeepGEMM auto-enable needs the env *and* PPU 1.5 (890P).

    ZW-810E hardware supports BF16 DeepGEMM, but its default dense backend is
    ACEXT, so ``VLLM_SAIL_DENSE_BF16_DEEPGEMM=1`` must not flip BF16 dense onto
    DeepGEMM there -- capability support is not default auto-selection.
    """
    name = "VLLM_SAIL_DENSE_BF16_DEEPGEMM"

    _clear_env(monkeypatch, name)
    assert _should_use_bf16_linear(chip, bias=False) is False

    monkeypatch.setenv(name, "1")
    assert _should_use_bf16_linear(chip, bias=False) is (chip == "890p")

    # A present bias always forces the acblas/F.linear fallback.
    assert _should_use_bf16_linear(chip, bias=True) is False


def test_env_unset_matches_default_selection(monkeypatch) -> None:
    """With every override cleared, selection equals the documented default."""
    for name in _ENV_ALIASES:
        _clear_env(monkeypatch, name)
    # ZW-810E default: ACEXT sits between the two PPU DeepGEMM entries.
    got = _unquantized_backends(monkeypatch, "810e", acext_available=True)
    assert got == [
        "PPU_DEEPGEMM",
        "ACEXT",
        "BATCHED_PPU_DEEPGEMM",
        "TRITON",
        "BATCHED_TRITON",
    ], got
    # ZW-890P default: no ACEXT.
    got = _unquantized_backends(monkeypatch, "890p", acext_available=False)
    assert got == [
        "PPU_DEEPGEMM",
        "BATCHED_PPU_DEEPGEMM",
        "TRITON",
        "BATCHED_TRITON",
    ], got
