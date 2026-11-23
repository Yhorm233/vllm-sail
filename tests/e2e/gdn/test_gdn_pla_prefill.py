# ruff: noqa: E402
# Plugin registration precedes imports of patched providers.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection, numerical, and performance tests for PPU PLA (FlashQLA) GDN prefill.

``VLLM_SAIL_USE_PLA`` defaults to enabled; ``VLLM_PPU_USE_PLA`` is its
lower-priority alias. ``ChunkGatedDeltaRule`` calls
``pla.prefill.flashqla.chunk_gated_delta_rule_fwd`` through the ``pla_prefill``
resolver.

Cover environment variables, missing-package fallback, backend selection, TP
head configurations, duplicate l2norm prevention, and output/final-state parity
between ``forward_pla`` and ``forward_native``. Performance tests record time
and memory without performance assertions.
This entire E2E file requires a real PPU; environment variables and resolver
stubs also have device-free CPU regressions.

Run on PPU::

    VLLM_SAIL_USE_PLA=1 pytest tests/e2e/gdn/test_gdn_pla_prefill.py
"""

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires real PPU for native kernel parity", allow_module_level=True)
pytest.importorskip("vllm")
import vllm_sail

vllm_sail.register_out_of_tree()
pytestmark = pytest.mark.ppu

import torch.nn.functional as F
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.attention import pla_prefill as sail_cuda_pla_prefill

if not current_platform.is_ppu():
    pytest.skip("requires the PPU platform", allow_module_level=True)

DEVICE = torch.device("cuda")

# Per-rank GDN dims. FlashQLA hard-requires head dims of 128, and (8, 4) is in
# its SUPPORTED_HEAD_CONFIGS whitelist as (num_v_heads, num_k_heads).
HV = 8
HK = 4
DK = 128
DV = 128

# Parity bounds against the Triton/FLA oracle. FlashQLA chunks at 32 while FLA
# chunks at 64, and the two use different WY factorizations, so bitwise or
# 1e-3-style agreement is not expected. cos > 0.99 is the bound pla's own
# FlashQLA acceptance suite used against its FLA-Triton oracle; the tighter
# rel < 5e-3 bound there applies only against pla's fp32 self-reference. rel is
# reported rather than asserted so a gradual regression shows up in the log.
COS_MIN = 0.99


# ---------------------------------------------------------------------------
# helpers: resolver / platform / config stubs (platform-independent tests)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_pla_prefill_resolver():
    """The resolver memoizes its answer; clear it around every test.

    Without this a stub installed by one test would leak into the next -- and
    into the real-PPU parity tests, which need a genuine resolution.
    """

    def _reset() -> None:
        sail_cuda_pla_prefill._resolved = False
        sail_cuda_pla_prefill._chunk_fwd_fn = None
        sail_cuda_pla_prefill._supported_head_configs = frozenset()

    _reset()
    yield
    _reset()


def _install_fake_pla(
    monkeypatch: pytest.MonkeyPatch,
    *,
    head_configs: frozenset[tuple[int, int]] = frozenset({(HV, HK)}),
):
    """Inject a stub ``pla`` package so the resolver can run off-PPU.

    Returns the stub kernel function -- the object the resolver caches -- so a
    test can prove it resolved the stub rather than a real ``pla``.
    """
    ops_mod = types.ModuleType("pla.prefill.flashqla.ops")
    ops_mod.SUPPORTED_HEAD_CONFIGS = head_configs

    def stub_fwd(*args, **kwargs):
        return None

    flashqla_mod = types.ModuleType("pla.prefill.flashqla")
    flashqla_mod.chunk_gated_delta_rule_fwd = stub_fwd
    flashqla_mod.ops = ops_mod

    prefill_mod = types.ModuleType("pla.prefill")
    prefill_mod.flashqla = flashqla_mod
    pla_mod = types.ModuleType("pla")
    pla_mod.prefill = prefill_mod

    for name, mod in (
        ("pla", pla_mod),
        ("pla.prefill", prefill_mod),
        ("pla.prefill.flashqla", flashqla_mod),
        ("pla.prefill.flashqla.ops", ops_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return stub_fwd


def _stub_resolver_env(
    monkeypatch: pytest.MonkeyPatch, *, use_pla: bool, is_ppu: bool
) -> None:
    """Point the resolver at stub ``envs``/``current_platform`` objects.

    Stubbing the names the resolver imported into its own namespace keeps the
    global ``vllm.envs`` and ``vllm.platforms`` untouched, so no state leaks.
    """
    monkeypatch.setattr(
        sail_cuda_pla_prefill, "envs", SimpleNamespace(VLLM_SAIL_USE_PLA=use_pla)
    )
    monkeypatch.setattr(
        sail_cuda_pla_prefill,
        "current_platform",
        SimpleNamespace(is_ppu=lambda: is_ppu),
    )


def _stub_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_cuda: bool = True,
    is_ppu: bool = False,
    capability: int = 90,
    family: int | None = None,
    runtime_major: int = 12,
) -> None:
    monkeypatch.setattr(
        gdn_mod,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: is_cuda,
            is_ppu=lambda: is_ppu,
            is_rocm=lambda: False,
            is_xpu=lambda: False,
            is_device_capability=lambda c: c == capability,
            is_device_capability_family=lambda f: f == family,
            get_cuda_runtime_major=lambda: runtime_major,
        ),
    )


def _stub_pla_kernel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    head_configs: frozenset[tuple[int, int]] = frozenset({(HV, HK)}),
) -> None:
    """Stub the resolver getters as seen by the backend-selection logic."""
    monkeypatch.setattr(
        sail_cuda_pla_prefill,
        "get_sail_cuda_pla_prefill_fwd",
        lambda: (lambda *a, **kw: None) if available else None,
    )
    monkeypatch.setattr(
        sail_cuda_pla_prefill,
        "get_sail_cuda_pla_prefill_head_configs",
        lambda: head_configs if available else frozenset(),
    )


def _make_vllm_config(
    *,
    gdn_prefill_backend: str = "auto",
    head_k_dim: int | None = DK,
    head_v_dim: int | None = DV,
    num_k_heads: int | None = HK,
    num_v_heads: int | None = HV,
    tp_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"gdn_prefill_backend": gdn_prefill_backend},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                linear_key_head_dim=head_k_dim,
                linear_value_head_dim=head_v_dim,
                linear_num_key_heads=num_k_heads,
                linear_num_value_heads=num_v_heads,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )


# ---------------------------------------------------------------------------
# helpers: kernel-level inputs and metrics (PPU parity tests)
# ---------------------------------------------------------------------------
def _requires_ppu_pla():
    """Skip unless PLA prefill is actually runnable; return the pla module."""
    if not envs.VLLM_PPU_USE_PLA:
        pytest.skip("VLLM_PPU_USE_PLA is disabled; the PLA prefill path is off")
    if not current_platform.is_ppu():
        pytest.skip("PPU PLA prefill tests require the PPU platform")
    return pytest.importorskip(
        "pla.prefill.flashqla", reason="PPU SAIL `pla` prefill package not found"
    )


def _bare_op() -> gdn_mod.ChunkGatedDeltaRule:
    """A ChunkGatedDeltaRule without __init__.

    ``forward_pla``/``forward_native`` are pure functions of their arguments,
    so the backend selection in ``__init__`` (which needs a live VllmConfig)
    can be skipped entirely.
    """
    return object.__new__(gdn_mod.ChunkGatedDeltaRule)


def _make_gate(mode: str, total: int, num_v_heads: int, rng) -> torch.Tensor:
    """Build a raw log-space gate, mirroring pla's FlashQLA accuracy harness.

    The modes exercise distinct FlashQLA paths: "logsigmoid" is the
    production-like moderate decay; "slow" (-1e-3) keeps every CP segment above
    the warmup threshold so the ``correct_h0`` / ``mt @ h`` path runs; "fast"
    forces the kernel to evaluate ``exp(g_i - g_j)`` directly.
    """
    shape = (1, total, num_v_heads)
    if mode == "slow":
        return torch.full(shape, -1e-3, device=DEVICE, dtype=torch.float32)
    if mode == "fast":
        noise = torch.rand(shape, device=DEVICE, dtype=torch.float32, generator=rng)
        return -(noise * 0.5 + 4.0)
    return F.logsigmoid(
        torch.randn(shape, device=DEVICE, dtype=torch.float32, generator=rng)
    )


def _make_prefill_inputs(
    seq_lens: list[int],
    *,
    num_v_heads: int = HV,
    num_k_heads: int = HK,
    head_k_dim: int = DK,
    head_v_dim: int = DV,
    gate_mode: str = "logsigmoid",
    state_dtype: torch.dtype = torch.float32,
    dtype: torch.dtype = torch.bfloat16,
    with_initial_state: bool = True,
    seed: int = 0,
) -> SimpleNamespace:
    """Build GDN prefill inputs mirroring the production call site.

    Reproduces what ``fused_post_conv_prep(apply_l2norm=True,
    output_g_exp=False)`` hands to ``ChunkGatedDeltaRule``: varlen-packed q/k/v
    under a leading batch dim of 1, l2-normalized q/k, a raw log-space gate and
    ``beta = sigmoid(b)``, plus per-sequence initial states in vLLM's
    ``[N, H, DV, DK]`` layout.
    """
    rng = torch.Generator("cuda").manual_seed(seed)
    ends = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32), ends]).to(DEVICE)
    total = int(ends[-1].item())

    def _randn(*shape):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=rng) * 0.1

    # FLA prefill semantics: q/k are l2-normalized by the caller, so both paths
    # are invoked with use_qk_l2norm_in_kernel=False.
    q = F.normalize(_randn(1, total, num_k_heads, head_k_dim).float(), p=2, dim=-1)
    k = F.normalize(_randn(1, total, num_k_heads, head_k_dim).float(), p=2, dim=-1)
    q, k = q.to(dtype), k.to(dtype)
    v = _randn(1, total, num_v_heads, head_v_dim)

    g = _make_gate(gate_mode, total, num_v_heads, rng)
    beta = torch.rand(
        1, total, num_v_heads, device=DEVICE, dtype=torch.float32, generator=rng
    ).sigmoid()

    initial_state = None
    if with_initial_state:
        initial_state = (
            _randn(len(seq_lens), num_v_heads, head_v_dim, head_k_dim)
            .to(state_dtype)
            .contiguous()
        )

    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        total=total,
    )


def _run(op, method: str, inp: SimpleNamespace, output_final_state: bool = True):
    return getattr(op, method)(
        q=inp.q,
        k=inp.k,
        v=inp.v,
        g=inp.g,
        beta=inp.beta,
        initial_state=inp.initial_state,
        output_final_state=output_final_state,
        cu_seqlens=inp.cu_seqlens,
        # FlashQLA builds its own chunk metadata at CS=32 and ignores the
        # FLA_CHUNK_SIZE=64 metadata vLLM precomputes, so pass neither.
        chunk_indices=None,
        chunk_offsets=None,
        use_qk_l2norm_in_kernel=False,
    )


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max-abs error relative to the reference's max-abs magnitude."""
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-8)).item()


def _assert_parity(pla_o, tri_o, pla_state, tri_state) -> None:
    assert torch.isfinite(pla_o).all(), "PLA output contains non-finite values"
    o_cos, o_rel = _cos(pla_o, tri_o), _rel(pla_o, tri_o)
    msg = f"o cos={o_cos:.6f} rel={o_rel:.2e}"
    if pla_state is not None:
        assert torch.isfinite(pla_state).all(), "PLA state has non-finite values"
        s_cos, s_rel = _cos(pla_state, tri_state), _rel(pla_state, tri_state)
        msg += f" | state cos={s_cos:.6f} rel={s_rel:.2e}"
    print(f"\n[pla prefill parity] {msg}")

    assert pla_o.shape == tri_o.shape, (pla_o.shape, tri_o.shape)
    assert o_cos > COS_MIN, f"attention output diverged: {msg}"
    if pla_state is not None:
        assert pla_state.shape == tri_state.shape, (pla_state.shape, tri_state.shape)
        assert s_cos > COS_MIN, f"final ssm state diverged: {msg}"


def _bench(fn, *, warmup: int = 3, iters: int = 20) -> tuple[float, float]:
    """Return (avg milliseconds per call, peak CUDA memory in MiB)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1 << 20)
    return start.elapsed_time(end) / iters, peak_mib


# ---------------------------------------------------------------------------
# env var resolution (platform-independent)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "new_val,expected",
    [
        (None, True),  # default: on
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
    ],
)
def test_ppu_pla_env_resolution(
    monkeypatch: pytest.MonkeyPatch,
    new_val: str | None,
    expected: bool,
) -> None:
    """Verify the canonical name and compatibility alias shared by prefill and decode."""
    names = ("VLLM_SAIL_USE_PLA", "VLLM_PPU_USE_PLA")
    for name in names:
        with monkeypatch.context() as isolated:
            for candidate in names:
                isolated.delenv(candidate, raising=False)
            if new_val is not None:
                isolated.setenv(name, new_val)
            assert envs.environment_variables["VLLM_SAIL_USE_PLA"]() is expected
            assert envs.VLLM_SAIL_USE_PLA is expected
            assert envs.VLLM_PPU_USE_PLA is expected


@pytest.mark.parametrize(
    "canonical,alias,expected",
    [("1", "0", True), ("0", "1", False), ("false", "true", False), ("", "1", False)],
)
def test_sail_pla_env_takes_precedence(monkeypatch, canonical, alias, expected):
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", canonical)
    monkeypatch.setenv("VLLM_PPU_USE_PLA", alias)
    assert envs.VLLM_SAIL_USE_PLA is expected
    assert envs.VLLM_PPU_USE_PLA is expected


# ---------------------------------------------------------------------------
# lazy resolver gating (platform-independent: stub `pla` package + platform)
# ---------------------------------------------------------------------------
def test_resolver_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_PPU_USE_PLA=0 leaves the kernel unset even on PPU with pla."""
    _install_fake_pla(monkeypatch)
    _stub_resolver_env(monkeypatch, use_pla=False, is_ppu=True)

    assert sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_fwd() is None
    configs = sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_head_configs()
    assert configs == frozenset()


def test_resolver_skips_non_ppu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-PPU CUDA (SM90/SM100) keeps using flashinfer/cutedsl/Triton."""
    _install_fake_pla(monkeypatch)
    _stub_resolver_env(monkeypatch, use_pla=True, is_ppu=False)

    assert sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_fwd() is None
    configs = sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_head_configs()
    assert configs == frozenset()


def test_resolver_enabled_on_ppu(monkeypatch: pytest.MonkeyPatch) -> None:
    """On PPU with pla importable the resolver hands out the FlashQLA kernel."""
    stub_fwd = _install_fake_pla(monkeypatch, head_configs=frozenset({(HV, HK)}))
    _stub_resolver_env(monkeypatch, use_pla=True, is_ppu=True)

    assert sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_fwd() is stub_fwd
    configs = sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_head_configs()
    assert configs == frozenset({(HV, HK)})


def test_resolver_tolerates_missing_pla(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing `pla` package must warn and fall back, never raise.

    This is the deliberate difference from the decode resolver, which raises
    ImportError: prefill has a fully functional Triton/FLA fallback, so
    deployments without `pla` must keep serving.
    """
    _stub_resolver_env(monkeypatch, use_pla=True, is_ppu=True)
    # Block cached submodules too, so the missing-package simulation is real;
    # monkeypatch restores them afterward.
    for name in (
        "pla",
        "pla.prefill",
        "pla.prefill.flashqla",
        "pla.prefill.flashqla.ops",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    assert sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_fwd() is None
    configs = sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_head_configs()
    assert configs == frozenset()


def test_resolver_honors_env_after_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caching the SDK handle must not bypass a subsequent disabled gate."""
    stub_fwd = _install_fake_pla(monkeypatch)
    _stub_resolver_env(monkeypatch, use_pla=True, is_ppu=True)

    first = sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_fwd()
    monkeypatch.setattr(
        sail_cuda_pla_prefill, "envs", SimpleNamespace(VLLM_SAIL_USE_PLA=False)
    )
    second = sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_fwd()

    assert first is stub_fwd
    assert second is None
    monkeypatch.setattr(
        sail_cuda_pla_prefill, "envs", SimpleNamespace(VLLM_SAIL_USE_PLA=True)
    )
    assert sail_cuda_pla_prefill.get_sail_cuda_pla_prefill_fwd() is stub_fwd


# ---------------------------------------------------------------------------
# backend selection (platform-independent: stub platform + resolver getters)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested", ["auto", "pla"])
def test_backend_selects_pla_on_ppu(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=True)

    cfg = _make_vllm_config(gdn_prefill_backend=requested)
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == (requested, "pla")


@pytest.mark.parametrize("requested", ["auto", "pla", "triton"])
def test_backend_falls_back_when_pla_unavailable(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    """env off / pla missing -> Triton, even for an explicit "pla" request."""
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=False)

    cfg = _make_vllm_config(gdn_prefill_backend=requested)
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == (requested, "triton")


def test_backend_honours_explicit_triton_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit triton request must win over an available PLA kernel.

    This is the A/B accuracy-comparison switch documented for the backend.
    """
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=True)

    cfg = _make_vllm_config(gdn_prefill_backend="triton")
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == ("triton", "triton")


@pytest.mark.parametrize(
    "head_k_dim,head_v_dim", [(64, 64), (128, 64), (64, 128), (None, None)]
)
def test_backend_rejects_non_128_head_dims(
    monkeypatch: pytest.MonkeyPatch, head_k_dim: int | None, head_v_dim: int | None
) -> None:
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=True)

    cfg = _make_vllm_config(head_k_dim=head_k_dim, head_v_dim=head_v_dim)
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == ("auto", "triton")


def test_backend_rejects_unlisted_head_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlashQLA only instantiates a fixed (num_v_heads, num_k_heads) set."""
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=True, head_configs=frozenset({(HV, HK)}))

    # (5, 3) is not whitelisted ...
    cfg = _make_vllm_config(num_v_heads=5, num_k_heads=3)
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == ("auto", "triton")
    # ... while the whitelisted pair is.
    cfg = _make_vllm_config(num_v_heads=HV, num_k_heads=HK)
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == ("auto", "pla")


@pytest.mark.parametrize(
    "num_v_heads,num_k_heads,tp_size,expected",
    [
        (16, 8, 2, "pla"),  # shards to the whitelisted (8, 4)
        (16, 8, 4, "triton"),  # shards to (4, 2), not whitelisted here
        (8, 4, 4, "triton"),  # shards to (2, 1)
    ],
)
def test_backend_validates_tp_sharded_head_counts(
    monkeypatch: pytest.MonkeyPatch,
    num_v_heads: int,
    num_k_heads: int,
    tp_size: int,
    expected: str,
) -> None:
    """The whitelist applies to per-rank head counts, not global ones."""
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=True, head_configs=frozenset({(8, 4)}))

    cfg = _make_vllm_config(
        num_v_heads=num_v_heads, num_k_heads=num_k_heads, tp_size=tp_size
    )
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == ("auto", expected)


@pytest.mark.parametrize("num_k_heads", [None, 4])
def test_backend_rejects_missing_head_counts(
    monkeypatch: pytest.MonkeyPatch, num_k_heads: int | None
) -> None:
    """A config that omits linear_num_key_heads must not reach the whitelist."""
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=True)

    cfg = _make_vllm_config(num_k_heads=num_k_heads)
    expected = "triton" if num_k_heads is None else "pla"
    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == ("auto", expected)


def test_backend_leaves_non_ppu_platforms_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Hopper, "auto" still resolves to flashinfer even with pla present."""
    _stub_platform(monkeypatch, is_ppu=False, capability=90)
    _stub_pla_kernel(monkeypatch, available=True)

    assert gdn_mod._resolve_gdn_prefill_backend(_make_vllm_config()) == (
        "auto",
        "flashinfer",
    )


def test_backend_non_cuda_falls_back_to_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_platform(monkeypatch, is_cuda=False)
    _stub_pla_kernel(monkeypatch, available=True)

    assert gdn_mod._resolve_gdn_prefill_backend(_make_vllm_config()) == (
        "auto",
        "triton",
    )


def test_backend_tolerates_missing_additional_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-dict additional_config must default to "auto", not crash."""
    _stub_platform(monkeypatch, is_ppu=True)
    _stub_pla_kernel(monkeypatch, available=True)
    cfg = _make_vllm_config()
    cfg.additional_config = None

    assert gdn_mod._resolve_gdn_prefill_backend(cfg) == ("auto", "pla")


# ---------------------------------------------------------------------------
# the l2norm guard (CPU tensors: the assert fires before any kernel work)
# ---------------------------------------------------------------------------
def test_forward_pla_rejects_in_kernel_l2norm() -> None:
    """forward_pla must refuse to double-normalize q/k.

    ``fused_post_conv_prep(apply_l2norm=True)`` already normalized them, so a
    second in-kernel normalization would silently corrupt the recurrence.
    """
    tiny = dict(
        q=torch.zeros(1, 4, HK, DK),
        k=torch.zeros(1, 4, HK, DK),
        v=torch.zeros(1, 4, HV, DV),
        g=torch.zeros(1, 4, HV),
        beta=torch.zeros(1, 4, HV),
        initial_state=torch.zeros(1, HV, DV, DK),
    )
    with pytest.raises(AssertionError, match="l2-normalized"):
        _bare_op().forward_pla(
            **tiny, output_final_state=True, use_qk_l2norm_in_kernel=True
        )


# ---------------------------------------------------------------------------
# FlashQLA (CUDA) vs Triton/FLA numerical parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gate_mode", ["logsigmoid", "slow", "fast"])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [128],  # single sequence, exact multiple of both chunk sizes
        [33],  # shortest multi-chunk case, with a remainder
        [1024],  # long enough for FlashQLA's intracard-CP splitting
        [129, 64, 255, 32],  # varlen pack crossing 32/64 boundaries
    ],
)
def test_prefill_pla_matches_triton(seq_lens: list[int], gate_mode: str) -> None:
    _requires_ppu_pla()
    op = _bare_op()
    inp = _make_prefill_inputs(seq_lens, gate_mode=gate_mode, seed=len(seq_lens))

    tri_o, tri_state = _run(op, "forward_native", inp)
    pla_o, pla_state = _run(op, "forward_pla", inp)

    _assert_parity(pla_o, tri_o, pla_state, tri_state)


@pytest.mark.parametrize("num_v_heads,num_k_heads", [(8, 4), (16, 4), (32, 8)])
def test_prefill_pla_supported_head_configs(num_v_heads: int, num_k_heads: int) -> None:
    """Every whitelisted per-rank head config must run and match Triton."""
    _requires_ppu_pla()
    op = _bare_op()
    inp = _make_prefill_inputs(
        [256], num_v_heads=num_v_heads, num_k_heads=num_k_heads, seed=num_v_heads
    )

    tri_o, tri_state = _run(op, "forward_native", inp)
    pla_o, pla_state = _run(op, "forward_pla", inp)

    _assert_parity(pla_o, tri_o, pla_state, tri_state)


def test_prefill_pla_bf16_state_matches_triton() -> None:
    """A bf16 ssm state cache must work: forward_pla upcasts it to fp32.

    FlashQLA's h0 is fp32-only, so the ``.to(torch.float32)`` in forward_pla is
    load-bearing for the bf16 state dtype vLLM may select.
    """
    _requires_ppu_pla()
    op = _bare_op()
    inp = _make_prefill_inputs([256], state_dtype=torch.bfloat16, seed=31)
    assert inp.initial_state.dtype is torch.bfloat16

    tri_o, tri_state = _run(op, "forward_native", inp)
    pla_o, pla_state = _run(op, "forward_pla", inp)

    assert pla_state.dtype is torch.float32
    _assert_parity(pla_o, tri_o, pla_state, tri_state)


def test_prefill_pla_without_initial_state() -> None:
    """A cold start (initial_state=None) must match Triton."""
    _requires_ppu_pla()
    op = _bare_op()
    inp = _make_prefill_inputs([256], with_initial_state=False, seed=41)
    assert inp.initial_state is None

    tri_o, tri_state = _run(op, "forward_native", inp)
    pla_o, pla_state = _run(op, "forward_pla", inp)

    _assert_parity(pla_o, tri_o, pla_state, tri_state)


def test_prefill_pla_output_final_state_false() -> None:
    """output_final_state=False still returns a correct output and no state."""
    _requires_ppu_pla()
    op = _bare_op()
    inp = _make_prefill_inputs([256], seed=51)

    tri_o, tri_state = _run(op, "forward_native", inp, output_final_state=False)
    pla_o, pla_state = _run(op, "forward_pla", inp, output_final_state=False)

    assert pla_state is None
    assert tri_state is None
    _assert_parity(pla_o, tri_o, None, None)


def test_prefill_pla_output_layout() -> None:
    """Pin the output contract the GDN core relies on.

    The caller does ``core_attn_out_non_spec.squeeze(0)`` and stitches the
    peeled decode outputs with ``torch.cat(..., dim=1)``, so forward_pla must
    return o as 4D ``[1, T, H, DV]`` and the state as ``[N, H, DV, DK]`` --
    the layouts forward_native returns, with FlashQLA's ``[N, H, DK, DV]``
    state transposed back.
    """
    _requires_ppu_pla()
    op = _bare_op()
    seq_lens = [64, 96]
    inp = _make_prefill_inputs(seq_lens, seed=61)

    pla_o, pla_state = _run(op, "forward_pla", inp)

    assert pla_o.shape == (1, inp.total, HV, DV)
    assert pla_o.dtype is inp.q.dtype
    assert pla_state.shape == (len(seq_lens), HV, DV, DK)
    assert pla_state.dtype is torch.float32
    assert pla_state[0].is_contiguous()

    # MRv2 scatters complete contiguous rows rather than assigning a strided
    # view. Exercise that actual consumer and preserve untouched cache slots.
    from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states

    cache = torch.zeros((3, HV, DV, DK), device=DEVICE, dtype=torch.float32)
    slots = torch.tensor([2, 0], device=DEVICE, dtype=torch.int32)
    scatter_states(cache, pla_state, slots)
    torch.cuda.synchronize()
    torch.testing.assert_close(cache[slots.long()], pla_state, rtol=0, atol=0)
    assert torch.count_nonzero(cache[1]).item() == 0


def test_prefill_pla_core_attn_out_buffer() -> None:
    """core_attn_out must receive exactly the values returned as output.

    Compared within a single call: FlashQLA's CP path does cross-segment
    reductions, so run-to-run bitwise determinism is not assumed.
    """
    _requires_ppu_pla()
    op = _bare_op()
    inp = _make_prefill_inputs([256], seed=71)
    buf = torch.zeros(inp.total, HV, DV, device=DEVICE, dtype=inp.q.dtype)

    pla_o, _ = op.forward_pla(
        q=inp.q,
        k=inp.k,
        v=inp.v,
        g=inp.g,
        beta=inp.beta,
        initial_state=inp.initial_state,
        output_final_state=True,
        cu_seqlens=inp.cu_seqlens,
        chunk_indices=None,
        chunk_offsets=None,
        use_qk_l2norm_in_kernel=False,
        core_attn_out=buf,
    )

    torch.testing.assert_close(buf, pla_o.squeeze(0), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# performance benchmarks (recorded, not asserted)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total", [4096, 16384])
def test_prefill_perf(total: int) -> None:
    _requires_ppu_pla()
    op = _bare_op()
    inp = _make_prefill_inputs([total], seed=81)

    t_ms, t_mem = _bench(lambda: _run(op, "forward_native", inp))
    c_ms, c_mem = _bench(lambda: _run(op, "forward_pla", inp))

    print(
        f"\n[pla prefill perf] T={total}: "
        f"triton {t_ms:.3f} ms / {t_mem:.1f} MiB  vs  "
        f"pla {c_ms:.3f} ms / {c_mem:.1f} MiB  (speedup {t_ms / c_ms:.2f}x)"
    )
