# SPDX-License-Identifier: Apache-2.0
"""PPU gate for DSV4.1's fused shifted post/pre Mega mHC dispatch.

The gate decides whether V4.1 runs one DeepGEMM kernel instead of the separate
post and pre pair, so both directions matter: a false negative silently keeps
the slow path, a false positive would hand CUDA's Ampere capability (8, 0) the
PPU kernel.
"""

from types import SimpleNamespace

import pytest

from tests.support.modules import load_patch

PATH = "vllm_sail/patch/enhancement/models/deepseek_v41_mega_mhc.py"
MODULE = "vllm.models.deepseek_v41.nvidia.ops.mega_mhc"
MHC_MODULE = "vllm.models.deepseek_v41.nvidia.ops.mhc"


class Tensor:
    """Annotation-only stand-in; the gate reads shapes, not tensor data."""

    def __init__(self, *shape):
        self.shape = shape


def _load(
    modules,
    capability=(8, 0),
    *,
    ppu=True,
    deep_gemm_supported=True,
    kernel=True,
):
    """Install the patch against fakes and return the live module objects."""
    platform = SimpleNamespace(
        ppu=ppu,
        capability=capability
        and SimpleNamespace(major=capability[0], minor=capability[1]),
    )
    platform.is_ppu = lambda: platform.ppu
    platform.get_device_capability = lambda: platform.capability
    modules("vllm.platforms", current_platform=platform)
    logger = SimpleNamespace(info_once=lambda *a, **k: None)
    modules("vllm.logger", init_logger=lambda name: logger)
    modules("torch", Tensor=Tensor)

    calls = []
    deep_gemm = modules(
        "vllm_sail.utils.deep_gemm",
        is_deep_gemm_supported=lambda: deep_gemm_supported,
        has_mega_mhc=lambda: kernel,
        mega_mhc=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    envs = modules("vllm_sail.envs", VLLM_SAIL_DISABLE_MEGA_MHC=False)

    originals = {
        "is_mega_mhc_supported": lambda *args: ("cuda_gate", args),
        "mega_mhc": lambda *args, **kwargs: calls.append(("cuda_kernel", args, kwargs)),
    }
    provider = modules(f"{MODULE}", **originals)
    mhc_originals = {
        "can_use_mega_mhc": lambda *args: ("cuda_dispatch", args),
    }
    dispatcher = modules(MHC_MODULE, **mhc_originals)
    patch = load_patch(PATH)
    return SimpleNamespace(
        patch=patch,
        platform=platform,
        deep_gemm=deep_gemm,
        envs=envs,
        provider=provider,
        dispatcher=dispatcher,
        originals=originals,
        mhc_originals=mhc_originals,
        calls=calls,
    )


@pytest.mark.parametrize("capability", [(8, 0), (8, 9)])
def test_gate_enables_mega_mhc_on_both_ppu_generations(modules, capability) -> None:
    gate = _load(modules, capability)
    assert gate.patch.is_mega_mhc_supported(5120, 4) is True
    assert gate.provider.is_mega_mhc_supported is gate.patch.is_mega_mhc_supported


@pytest.mark.parametrize("capability", [(8, 6), (9, 0), (10, 0), None])
def test_gate_rejects_capabilities_without_the_kernel(modules, capability) -> None:
    gate = _load(modules, capability)
    assert gate.patch.is_mega_mhc_supported(5120, 4) is False


@pytest.mark.parametrize(
    "hidden_size,hc_mult",
    [(5000, 4), (4096, 4), (5120, 2), (0, 4), (-1024, 4)],
)
def test_gate_rejects_shapes_outside_the_kernel_contract(
    modules, hidden_size, hc_mult
) -> None:
    gate = _load(modules)
    assert gate.patch.is_mega_mhc_supported(hidden_size, hc_mult) is False


@pytest.mark.parametrize("deep_gemm_supported,kernel", [(False, True), (True, False)])
def test_gate_requires_deepgemm_and_kernel(
    modules, deep_gemm_supported, kernel
) -> None:
    gate = _load(modules, deep_gemm_supported=deep_gemm_supported, kernel=kernel)
    assert gate.patch.is_mega_mhc_supported(5120, 4) is False


def test_kill_switch_beats_the_shape_cache(modules) -> None:
    """The switch is read outside the cache, so a live server can fall back."""
    gate = _load(modules)
    assert gate.patch.is_mega_mhc_supported(5120, 4) is True
    gate.envs.VLLM_SAIL_DISABLE_MEGA_MHC = True
    assert gate.patch.is_mega_mhc_supported(5120, 4) is False


def test_mega_mhc_call_goes_to_the_ppu_wrapper(modules) -> None:
    gate = _load(modules)
    gate.patch.mega_mhc(x="x", y_bf16="y")
    assert gate.calls == [((), {"x": "x", "y_bf16": "y"})]


def test_non_ppu_delegates_to_the_upstream_symbols(modules) -> None:
    gate = _load(modules, ppu=False)
    assert gate.patch.is_mega_mhc_supported(5120, 4) == ("cuda_gate", (5120, 4))
    gate.patch.mega_mhc(x="x")
    assert gate.calls == [("cuda_kernel", (), {"x": "x"})]
    args = (Tensor(7, 5120), Tensor(7, 4, 5120), None, None, False)
    assert gate.patch.can_use_mega_mhc(*args) == ("cuda_dispatch", args)


@pytest.mark.parametrize(
    "pre_mix,norm_weight,capture_aux,tokens,expected",
    [
        (True, True, False, 7, True),
        (False, True, False, 7, False),
        (True, False, False, 7, False),
        (True, True, True, 7, False),
        (True, True, False, (1 << 20) + 1, False),
    ],
)
def test_dispatcher_consults_the_ppu_gate(
    modules, pre_mix, norm_weight, capture_aux, tokens, expected
) -> None:
    gate = _load(modules)
    assert gate.dispatcher.can_use_mega_mhc is gate.patch.can_use_mega_mhc
    usable = gate.patch.can_use_mega_mhc(
        Tensor(tokens, 5120),
        Tensor(tokens, 4, 5120),
        Tensor(tokens, 4) if pre_mix else None,
        Tensor(5120) if norm_weight else None,
        capture_aux,
    )
    assert usable is expected


def test_dispatcher_rejects_shapes_the_gate_rejects(modules) -> None:
    gate = _load(modules)
    assert (
        gate.patch.can_use_mega_mhc(
            Tensor(7, 4096), Tensor(7, 4, 4096), Tensor(7, 4), Tensor(4096), False
        )
        is False
    )