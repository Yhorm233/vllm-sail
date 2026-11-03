# SPDX-License-Identifier: Apache-2.0
"""Category-2: ACEXT / FP8-DeepGEMM / BF16-DeepGEMM availability gating.

CPU-only; imports neither ``torch`` nor ``vllm``. Rather than reimplementing the
gates, this module executes the *production* gate statements verbatim: the
module-level ACEXT availability ``if`` in ``.../fused_moe/experts/acext.py`` is
compiled and run against a per-chip ``current_platform`` double, with the
``acext`` SDK made importable / missing through ``sys.modules`` (coordinating
with the ``ppu_sdk_stubs`` fixture). The DeepGEMM gates are the real
``is_supported`` classmethods and ``should_use_deepgemm_for_bf16_linear``.

Three gating branches, each asserted hard (fail on mismatch):

* **ACEXT** -- available only on ZW-810E ``(8, 0)`` *and* only when ``acext``
  imports. On ZW-890P the capability gate wins: even an importable ``acext``
  leaves it unavailable.
* **w8a8-fp8 / mxfp4 DeepGEMM** -- gated to ZW-890P ``(8, 9)``; never
  constructed on ZW-810E ``(8, 0)``.
* **bf16 / w8a8-int8 DeepGEMM** -- hardware-supported (constructible) on *both*
  chips, but BF16 *dense* auto-enable additionally requires
  ``VLLM_SAIL_DENSE_BF16_DEEPGEMM`` and PPU 1.5, so it fires only on ZW-890P.
  This is the "capability support != default auto-enable" distinction.
"""

from __future__ import annotations

import ast
import copy
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import CHIP_PROFILES
from vllm_sail import envs as ppu_envs

ROOT = Path(__file__).resolve().parents[2]
CHIPS = sorted(CHIP_PROFILES)
ACEXT_PATH = "vllm_sail/model_executor/layers/fused_moe/experts/acext.py"
PPU_KERNEL_PATH = "vllm_sail/model_executor/kernels/linear/scaled_mm/ppu.py"
DEEP_GEMM_PATH = "vllm_sail/utils/deep_gemm.py"


# ---------------------------------------------------------------------------
# AST extraction helpers (Python 3.9-safe; see test_backend_selection.py).
# ---------------------------------------------------------------------------
def _future_import() -> ast.ImportFrom:
    return ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )


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
    module = ast.Module(body=[_future_import(), node], type_ignores=[])
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


_TORCH = types.SimpleNamespace(bfloat16="bf16")


# ---------------------------------------------------------------------------
# ACEXT availability: run the real module-level gate + is_acext_supported().
# ---------------------------------------------------------------------------
def _acext_availability(
    monkeypatch: pytest.MonkeyPatch, chip: str, *, importable: bool
) -> tuple[bool, bool]:
    """Execute acext.py's availability gate; return (flag, is_acext_supported())."""
    if importable:
        # Declare exactly the two names the gate imports so a strict stub module
        # (from ppu_sdk_stubs) or a fresh module both satisfy the `from acext`.
        module = sys.modules.get("acext")
        if module is None:
            module = types.ModuleType("acext")
            monkeypatch.setitem(sys.modules, "acext", module)
        module.fusedmoe_wrapper = lambda *a, **k: None
        module.get_enum_from_booleans = lambda *a, **k: None
    else:
        # A sys.modules entry of None makes `from acext import ...` raise
        # ImportError, exercising the gate's except branch.
        monkeypatch.setitem(sys.modules, "acext", None)

    tree = ast.parse((ROOT / ACEXT_PATH).read_text())
    gate = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.If)))
    predicate = copy.deepcopy(_find_def(tree, "is_acext_supported"))
    predicate.decorator_list = []
    namespace: dict = {"current_platform": platform_double(chip)}
    module_ast = ast.Module(body=[_future_import(), gate, predicate], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    exec(compile(module_ast, str(ROOT / ACEXT_PATH), "exec"), namespace)
    return bool(namespace["_ACEXT_AVAILABLE"]), bool(namespace["is_acext_supported"]())


# ---------------------------------------------------------------------------
# DeepGEMM gates.
# ---------------------------------------------------------------------------
def _kernel_is_supported(chip: str, class_name: str) -> bool:
    supported, _reason = extract(
        PPU_KERNEL_PATH,
        f"{class_name}.is_supported",
        {
            "current_platform": platform_double(chip),
            "is_deep_gemm_supported": lambda: True,
        },
    )(None)
    return bool(supported)


def _deep_gemm_base_supported(chip: str) -> bool:
    fn = extract(
        DEEP_GEMM_PATH,
        "is_deep_gemm_supported",
        {
            "envs": types.SimpleNamespace(VLLM_USE_DEEP_GEMM=True),
            "has_deep_gemm": lambda: True,
            "current_platform": platform_double(chip),
        },
    )
    return bool(fn())


def _mxfp4_activation_key(monkeypatch: pytest.MonkeyPatch, chip: str):
    module = types.ModuleType("vllm.platforms")
    module.current_platform = platform_double(chip)
    monkeypatch.setitem(sys.modules, "vllm.platforms", module)
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
    fn(types.SimpleNamespace(moe_backend="auto"))
    return seen["activation_key"]


def _should_use_bf16_linear(
    monkeypatch: pytest.MonkeyPatch,
    chip: str,
    *,
    bf16_env: bool,
    bias: bool,
    n: int = 64,
) -> bool:
    if bf16_env:
        monkeypatch.setenv("VLLM_SAIL_DENSE_BF16_DEEPGEMM", "1")
    else:
        monkeypatch.delenv("VLLM_SAIL_DENSE_BF16_DEEPGEMM", raising=False)
        monkeypatch.delenv("VLLM_PPU_DENSE_BF16_DEEPGEMM", raising=False)
    platform = platform_double(chip)
    x = types.SimpleNamespace(dtype="bf16", is_contiguous=lambda: True)
    weight = types.SimpleNamespace(
        dtype="bf16", dim=lambda: 2, shape=(n, 16), is_contiguous=lambda: True
    )
    fn = extract(
        DEEP_GEMM_PATH,
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
# ACEXT availability gating.
# ===========================================================================
@pytest.mark.parametrize("importable", [True, False])
@pytest.mark.parametrize("chip", CHIPS)
def test_acext_availability_requires_810e_and_importable(
    monkeypatch, ppu_sdk_stubs, chip, importable
) -> None:
    """ACEXT lights up only on ZW-810E with an importable ``acext`` SDK."""
    flag, predicate = _acext_availability(monkeypatch, chip, importable=importable)
    expected = chip == "810e" and importable
    assert flag is expected, (chip, importable, flag)
    # is_acext_supported() is the public read of the same gate.
    assert predicate is expected, (chip, importable, predicate)


def test_acext_capability_gate_beats_importability(monkeypatch, ppu_sdk_stubs) -> None:
    """On ZW-890P an importable ``acext`` still yields unavailable.

    The capability gate ``(8, 0)`` short-circuits before the import is even
    attempted, so ACEXT can never be selected on ZW-890P.
    """
    flag, predicate = _acext_availability(monkeypatch, "890p", importable=True)
    assert flag is False
    assert predicate is False


def test_acext_supports_bf16_and_int8_schemes() -> None:
    """The ACEXT quant-scheme whitelist is exactly bf16 and w8a8-int8."""
    int8_w = object()
    int8_a = object()
    fn = extract(
        ACEXT_PATH,
        "AcextExperts._supports_quant_scheme",
        {
            "QuantKey": object,
            "kInt8StaticChannelSym": int8_w,
            "kInt8DynamicTokenSym": int8_a,
        },
    )
    assert fn(None, None) is True  # bf16
    assert fn(int8_w, int8_a) is True  # w8a8-int8
    assert fn(object(), object()) is False  # anything else


# ===========================================================================
# FP8 / MXFP4 DeepGEMM gating (ZW-890P only).
# ===========================================================================
@pytest.mark.parametrize("chip", CHIPS)
def test_fp8_deepgemm_gated_to_890p(ppu_sdk_stubs, chip) -> None:
    for class_name in (
        "PPUDeepGemmFP8ScaledMMLinearKernel",
        "PPUDeepGemmFp8BlockScaledMMKernel",
    ):
        assert _kernel_is_supported(chip, class_name) is (chip == "890p"), (
            class_name,
            chip,
        )


@pytest.mark.parametrize("chip", CHIPS)
def test_mxfp4_deepgemm_gated_to_890p(monkeypatch, chip) -> None:
    """Native mxfp4 W4A4 (dynamic activation key) is a ZW-890P-only path."""
    activation_key = _mxfp4_activation_key(monkeypatch, chip)
    assert (activation_key == "kMxfp4Dynamic") is (chip == "890p")


# ===========================================================================
# BF16 / INT8 DeepGEMM: constructible on both, dense auto-enable only on 890P.
# ===========================================================================
@pytest.mark.parametrize("chip", CHIPS)
def test_bf16_and_int8_deepgemm_constructible_on_both_chips(
    ppu_sdk_stubs, chip
) -> None:
    """The capability-independent base gate + int8 kernel run on both chips."""
    assert _deep_gemm_base_supported(chip) is True
    assert _kernel_is_supported(chip, "PPUInt8ScaledMMLinearKernel") is True


@pytest.mark.parametrize("chip", CHIPS)
def test_bf16_dense_autoenable_only_on_890p_with_env(monkeypatch, chip) -> None:
    """BF16 dense DeepGEMM auto-enable needs the env *and* PPU 1.5 (890P)."""
    # Env unset: never auto-enabled, on either chip.
    assert (
        _should_use_bf16_linear(monkeypatch, chip, bf16_env=False, bias=False) is False
    )
    # Env set: auto-enabled only on ZW-890P; ZW-810E default dense stays ACEXT.
    assert _should_use_bf16_linear(monkeypatch, chip, bf16_env=True, bias=False) is (
        chip == "890p"
    )
    # A present bias always forces the acblas/F.linear fallback.
    assert _should_use_bf16_linear(monkeypatch, chip, bf16_env=True, bias=True) is False


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, False),  # odd N, scalar projection included: kernel asserts even N
        (2, True),
        (64, True),
        (4096, True),
        (3, False),
        (5, False),
        (129, False),
    ],
)
def test_bf16_dense_accepts_only_even_n(monkeypatch, n, expected) -> None:
    """DeepGEMM BF16 dense NT tiling accepts an even N only.

    Every odd N -- N == 1 (Kimi-K3 AttnRes scalar projections) included --
    is rejected at dispatch so it falls back to acblas instead of failing
    inside the kernel (the PPU "BF16 dense GEMM requires even N" JIT
    assertion).
    """
    assert (
        _should_use_bf16_linear(monkeypatch, "890p", bf16_env=True, bias=False, n=n)
        is expected
    )
