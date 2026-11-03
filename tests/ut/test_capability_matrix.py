# SPDX-License-Identifier: Apache-2.0
"""Category-2: the authoritative PPU chip capability matrix.

This module is CPU-only and imports neither ``torch`` nor ``vllm``. The chip
profiles come straight from ``tests/conftest.py``; the capability predicates
are exercised by compiling the *production* functions in isolation (AST
extraction with a stub namespace), so the assertions read the real branch
logic without paying the real import cost.

The matrix asserted here is the user-confirmed **hardware capability support**:

===================  ==========================  =======================================
backend              ZW-810E ``(8, 0)``          ZW-890P ``(8, 9)``
===================  ==========================  =======================================
DeepGEMM             bf16, w8a8-int8             bf16, w8a8-int8, w8a8-fp8, mxfp4
ACEXT                bf16, w8a8-int8, w4a8-int8  unavailable
===================  ==========================  =======================================

Hardware capability support is deliberately kept distinct from *default
auto-selection* (e.g. BF16 dense DeepGEMM auto-enable is gated by
``VLLM_SAIL_DENSE_BF16_DEEPGEMM`` on PPU 1.5, and the 810E default dense
backend is ACEXT). Those two layers are covered by ``test_backend_selection``
and ``test_acext_deepgemm_gating``.
"""

from __future__ import annotations

import ast
import copy
import types
from pathlib import Path

import pytest

from tests.conftest import CHIP_PROFILES, DeviceCapability, FakePPUPlatform

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Authoritative hardware capability support matrix (user-confirmed).
# ---------------------------------------------------------------------------
DEEPGEMM_SUPPORT: dict[str, frozenset[str]] = {
    "810e": frozenset({"bf16", "w8a8-int8"}),
    "890p": frozenset({"bf16", "w8a8-int8", "w8a8-fp8", "mxfp4"}),
}
ACEXT_SUPPORT: dict[str, frozenset[str]] = {
    "810e": frozenset({"bf16", "w8a8-int8", "w4a8-int8"}),
    # ACEXT is not available on ZW-890P: its capability gate is ``(8, 0)`` only.
    "890p": frozenset(),
}
CHIPS = sorted(CHIP_PROFILES)


def _find_def(tree: ast.AST, qualname: str) -> ast.AST:
    node: ast.AST = tree
    for part in qualname.split("."):
        match = next(
            (
                child
                for child in node.body  # type: ignore[attr-defined]
                # Two single-type isinstance calls (not a tuple / `X | Y`) keep
                # this Python 3.9-runnable: `ast.FunctionDef | ast.ClassDef`
                # raises TypeError at runtime before 3.10.
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
    """Compile one production function/method with ``namespace`` as globals.

    Decorators (``@cache``, ``@classmethod``, ``@patch``) are stripped so the
    bare body can run without vLLM/torch. ``from __future__ import annotations``
    keeps annotations lazy, which is what lets this import on Python 3.9.
    """
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
    """A ``current_platform`` double carrying only the chip's capability."""
    capability = CHIP_PROFILES[chip].capability

    def is_device_capability(value) -> bool:
        if isinstance(value, tuple):
            return tuple(capability) == tuple(value)
        return capability.major * 10 + capability.minor >= value

    return types.SimpleNamespace(
        is_ppu=lambda: True,
        is_cuda=lambda: True,
        get_device_capability=lambda device_id=0: capability,
        get_device_name=lambda device_id=0: CHIP_PROFILES[chip].name,
        is_device_capability=is_device_capability,
    )


def _deep_gemm_base_supported(chip: str) -> bool:
    """``is_deep_gemm_supported()``: PPU + env + lib, capability-independent."""
    fn = extract(
        "vllm_sail/utils/deep_gemm.py",
        "is_deep_gemm_supported",
        {
            "envs": types.SimpleNamespace(VLLM_USE_DEEP_GEMM=True),
            "has_deep_gemm": lambda: True,
            "current_platform": platform_double(chip),
        },
    )
    return bool(fn())


def _fp8_deep_gemm_supported(chip: str) -> bool:
    """``PPUDeepGemmFP8ScaledMMLinearKernel.is_supported`` for this chip."""
    fn = extract(
        "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py",
        "PPUDeepGemmFP8ScaledMMLinearKernel.is_supported",
        {
            "current_platform": platform_double(chip),
            "is_deep_gemm_supported": lambda: True,
        },
    )
    supported, _reason = fn(None)
    return bool(supported)


def _deepgemm_support(chip: str) -> frozenset[str]:
    """Derive the DeepGEMM support set from the production predicates."""
    if not _deep_gemm_base_supported(chip):
        return frozenset()
    # bf16 / w8a8-int8 DeepGEMM ride the capability-independent base gate.
    support = {"bf16", "w8a8-int8"}
    # w8a8-fp8 and mxfp4 DeepGEMM are gated off on SM80 ``(8, 0)`` hardware.
    if _fp8_deep_gemm_supported(chip):
        support |= {"w8a8-fp8", "mxfp4"}
    return frozenset(support)


def _acext_capability_gate(chip: str) -> bool:
    """ACEXT is gated to ``(8, 0)`` only; verified structurally below."""
    return CHIP_PROFILES[chip].capability == DeviceCapability(8, 0)


def _acext_support(chip: str, *, importable: bool = True) -> frozenset[str]:
    if not (_acext_capability_gate(chip) and importable):
        return frozenset()
    return frozenset({"bf16", "w8a8-int8", "w4a8-int8"})


# ---------------------------------------------------------------------------
# Static chip-profile matrix (name / capability / boolean feature bits).
# ---------------------------------------------------------------------------
def test_chip_profile_static_matrix() -> None:
    assert set(CHIP_PROFILES) == {"810e", "890p"}
    e = CHIP_PROFILES["810e"]
    assert (e.name, e.capability) == ("PPU-ZW810E", DeviceCapability(8, 0))
    assert (e.fp8, e.acext, e.bf16_dense_deepgemm) == (False, True, False)
    p = CHIP_PROFILES["890p"]
    assert (p.name, p.capability) == ("PPU-ZW890P", DeviceCapability(8, 9))
    assert (p.fp8, p.acext, p.bf16_dense_deepgemm) == (True, False, True)


def test_device_probe_matches_profile(fake_ppu_platform: FakePPUPlatform) -> None:
    """The parameterized platform double reports the injected chip identity."""
    profile = fake_ppu_platform.profile
    assert fake_ppu_platform.get_device_capability() == profile.capability
    assert fake_ppu_platform.get_device_name() == profile.name
    # CUDA disguise: is_cuda() and is_ppu() are both always True on PPU.
    assert fake_ppu_platform.is_ppu() is True
    assert fake_ppu_platform.is_cuda() is True


@pytest.mark.parametrize("chip", CHIPS)
def test_is_device_capability_mutually_exclusive(chip: str) -> None:
    platform = platform_double(chip)
    is_810e = chip == "810e"
    assert platform.is_device_capability((8, 0)) is is_810e
    assert platform.is_device_capability((8, 9)) is (not is_810e)
    assert platform.is_ppu() is True
    assert platform.is_cuda() is True


@pytest.mark.parametrize("chip", CHIPS)
def test_is_ppu_15_only_on_890p(chip: str) -> None:
    """``is_ppu_15()`` is the PPU 1.5 ``(8, 9)`` discriminator used by gates."""
    fn = extract(
        "vllm_sail/utils/deep_gemm.py",
        "is_ppu_15",
        {"current_platform": platform_double(chip)},
    )
    assert fn() is (chip == "890p")


@pytest.mark.parametrize("chip", CHIPS)
def test_deep_gemm_base_support_is_capability_independent(chip: str) -> None:
    """Base DeepGEMM availability needs PPU + env + lib, not a capability.

    This is why bf16 / w8a8-int8 DeepGEMM are hardware-supported on *both*
    chips while the fp8/mxfp4 paths carry an extra ``(8, 0)`` exclusion.
    """
    assert _deep_gemm_base_supported(chip) is True


@pytest.mark.parametrize("chip", CHIPS)
def test_fp8_deepgemm_kernel_excludes_810e(chip: str) -> None:
    for class_name in (
        "PPUDeepGemmFP8ScaledMMLinearKernel",
        "PPUDeepGemmFp8BlockScaledMMKernel",
    ):
        supported, reason = extract(
            "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py",
            f"{class_name}.is_supported",
            {
                "current_platform": platform_double(chip),
                "is_deep_gemm_supported": lambda: True,
            },
        )(None)
        assert supported is (chip == "890p"), (class_name, chip, reason)
        if not supported:
            assert "SM80" in reason


@pytest.mark.parametrize("chip", CHIPS)
def test_deepgemm_hardware_support_matrix(chip: str) -> None:
    """Derived DeepGEMM support must equal the authoritative matrix exactly."""
    assert _deepgemm_support(chip) == DEEPGEMM_SUPPORT[chip]


def test_acext_capability_gate_is_810e_only() -> None:
    """Structurally verify the module-level ACEXT availability gate.

    ``acext.py`` computes ``_ACEXT_AVAILABLE`` only when
    ``is_ppu() and is_device_capability((8, 0))`` and the import succeeds, so
    ACEXT can never light up on ZW-890P regardless of importability.
    """
    tree = ast.parse(
        (
            ROOT / "vllm_sail/model_executor/layers/fused_moe/experts/acext.py"
        ).read_text()
    )
    gate = next(
        (node for node in tree.body if isinstance(node, ast.If)),
        None,
    )
    assert gate is not None, "acext.py lost its module-level availability gate"
    assert ast.unparse(gate.test) == (
        "current_platform.is_ppu() and current_platform.is_device_capability((8, 0))"
    )
    # The True assignment lives behind a `from acext import ...` inside the gate.
    assigned = [
        target.id
        for stmt in gate.body
        if isinstance(stmt, ast.Try)
        for inner in stmt.body
        if isinstance(inner, ast.Assign)
        for target in inner.targets
        if isinstance(target, ast.Name)
    ]
    assert "_ACEXT_AVAILABLE" in assigned


@pytest.mark.parametrize("chip", CHIPS)
def test_acext_hardware_support_matrix(chip: str) -> None:
    """Derived ACEXT support must equal the authoritative matrix exactly."""
    assert _acext_support(chip) == ACEXT_SUPPORT[chip]
    # Even a hypothetically importable acext stays unavailable on ZW-890P:
    # the capability gate, not importability, is decisive.
    assert _acext_support(chip, importable=True) == ACEXT_SUPPORT[chip]


def test_matrices_are_disjoint_on_acext_and_fp8() -> None:
    """Cross-check the two exclusivity claims the user called out."""
    # ACEXT exists only where fp8 DeepGEMM does not, and vice versa.
    assert "w8a8-fp8" not in DEEPGEMM_SUPPORT["810e"]
    assert ACEXT_SUPPORT["890p"] == frozenset()
    assert DEEPGEMM_SUPPORT["810e"].isdisjoint({"w8a8-fp8", "mxfp4"})
    assert ACEXT_SUPPORT["810e"] == frozenset({"bf16", "w8a8-int8", "w4a8-int8"})
