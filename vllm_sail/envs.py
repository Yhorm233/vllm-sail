# SPDX-License-Identifier: Apache-2.0
"""SAIL runtime environment variables — the single source of truth.

Rules (enforced in review):

1. Every SAIL-specific runtime environment variable is declared here. Do **not**
   read ``VLLM_SAIL_*`` or legacy ``VLLM_PPU_*`` runtime names elsewhere.
2. Access is lazy, via module ``__getattr__``, so a variable changed after
   import is still observed. Use :func:`is_set` to distinguish "explicitly set
   by the user" from "left at the default" — several PPU backend-selection paths
   depend on that difference.
3. ``VLLM_SAIL_*`` names take precedence over their ``VLLM_PPU_*`` aliases,
   including explicit false or empty values. Legacy Python attributes use the
   same resolution. Defaults are unchanged from the in-tree PPU fork.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Declarations for static analysis and IDE completion only.
#
# These MUST stay inside `if TYPE_CHECKING`. An annotated assignment at module
# scope creates a real module attribute, and a real attribute shadows the module
# `__getattr__` below — so the lazy getters would never run and every variable
# would read back as its literal default regardless of the environment. (This is
# also why vllm's own envs.py guards its declarations the same way.)
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    PPU_SDK: str = "/usr/local/PPU_SDK"
    VLLM_SAIL_FUSED_RMSNORM_QUANT: bool = False
    VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT: bool = False
    VLLM_SAIL_DENSE_BF16_DEEPGEMM: bool = False
    VLLM_SAIL_USE_PLA: bool = True
    VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED: bool = False
    VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT: bool = False
    VLLM_SAIL_USE_DEEPSELECT: bool = True
    VLLM_SAIL_MOE_BACKEND: str | None = None
    VLLM_SAIL_DENSE_BACKEND: str | None = None
    VLLM_SAIL_FUSED_GDN_DECODE: bool = True
    VLLM_SAIL_DISABLE_MOE_WNA16_CUDA: bool = False
    VLLM_SAIL_FORCE_MOE_WNA16_CUDA: bool = False
    VLLM_SAIL_ENABLE_MOE_MARLIN: bool = False
    VLLM_SAIL_MEGA_MOE_MMA_TYPE: str = "fp8xfp4"
    VLLM_SAIL_DISABLE_MEGA_MHC: bool = False
    VLLM_SAIL_USE_TRITON_INT8_QUANT: bool = True
    VLLM_SAIL_FP8_LOAD_POLICY: str | None = None
    VLLM_SAIL_ENABLE_FP8_TO_INT8: bool = False
    VLLM_SAIL_KPOOL_DECODE_LOGITS: str = "non_paged"
    VLLM_SAIL_KPOOL_MAX_DENSE_POOLS: int = 1 << 21
    VLLM_SAIL_QSA_KERNEL: str = "auto"
    VLLM_SAIL_STEP5_DSA_SCORES: bool = False
    VLLM_SAIL_STEP5_DSA_SELECTOR_PPU: bool = False
    VLLM_SAIL_STEP5_DSA_PREFILL_PPU: bool = False
    VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND: str = "flashinfer"
    VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION: bool = False
    VLLM_STEP5_INDEXER_FP8_AS_FP16: bool = False
    VLLM_SAIL_MINIMAX_M3_MSA: bool = True
    VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_TRITON: bool = False
    VLLM_SAIL_MINIMAX_M3_MSA_ATTEND_TRITON: bool = False
    VLLM_SAIL_NVTX_PROFILE: bool = False
    VLLM_SAIL_NVTX_DUMP_DICT: bool = False
    VLLM_SAIL_NVTX_DUMP_TOPK: bool = False
    VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN: bool = False
    VLLM_SAIL_FUSED_NORM_ROPE_ATTN: bool = False
    VLLM_SAIL_PPU_KV_MXFP8: bool = False
    VLLM_SAIL_PPU_KV_NVFP4: bool = False
    VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE: bool = False
    VLLM_DEEPEPLL_RECV_HOOK: bool = True
    VLLM_GDN_DECODE_KERNEL: str = "auto"

    # Backward-compatible Python attributes; all reads resolve SAIL first.
    VLLM_PPU_FUSED_RMSNORM_QUANT: bool = False
    VLLM_PPU_USE_OPT_TOKEN_GROUP_QUANT: bool = False
    VLLM_PPU_DENSE_BF16_DEEPGEMM: bool = False
    VLLM_PPU_USE_PLA: bool = True
    VLLM_PPU_DEEPGEMM_MOE_TP_FUSED: bool = False
    VLLM_PPU_DEEPGEMM_MOE_FUSED_ACT_QUANT: bool = False
    VLLM_PPU_USE_DEEPSELECT: bool = True
    VLLM_PPU_MOE_BACKEND: str | None = None
    VLLM_PPU_DENSE_BACKEND: str | None = None
    VLLM_PPU_FUSED_GDN_DECODE: bool = True
    VLLM_PPU_DISABLE_MOE_WNA16_CUDA: bool = False
    VLLM_PPU_FORCE_MOE_WNA16_CUDA: bool = False
    VLLM_PPU_ENABLE_MOE_MARLIN: bool = False
    VLLM_PPU_USE_TRITON_INT8_QUANT: bool = True
    VLLM_PPU_FP8_LOAD_POLICY: str | None = None
    VLLM_PPU_ENABLE_FP8_TO_INT8: bool = False
    VLLM_PPU_STEP5_DSA_SCORES: bool = False
    VLLM_PPU_STEP5_DSA_SELECTOR_PPU: bool = False
    VLLM_PPU_STEP5_DSA_PREFILL_PPU: bool = False
    VLLM_PPU_STEP5_DSA_PREFILL_BACKEND: str = "flashinfer"
    VLLM_PPU_STEP5_DSA_PREFILL_BATCH_ATTENTION: bool = False
    VLLM_PPU_NVTX_PROFILE: bool = False
    VLLM_PPU_NVTX_DUMP_DICT: bool = False
    VLLM_PPU_NVTX_DUMP_TOPK: bool = False
    VLLM_PPU_NVTX_VFA_DUMP_SEQLEN: bool = False
    VLLM_PPU_FUSED_NORM_ROPE_ATTN: bool = False
    VLLM_PPU_PPU_KV_MXFP8: bool = False
    VLLM_PPU_PPU_KV_NVFP4: bool = False
    VLLM_PPU_FUSED_ATTN_PER_TOKEN_SCALE: bool = False

_TRUTHY = ("true", "1", "yes", "on")

#: Backend names accepted by both MoE and dense GEMM selection.
GEMM_BACKENDS = ("deepgemm", "acext", "triton")
GDN_DECODE_KERNELS = ("auto", "cuda", "pla", "triton")
KPOOL_DECODE_LOGITS_BACKENDS = ("non_paged",)
#: Qwen4Exp QSA sparse-attention kernels. ``cuda`` is FA3's QSA path, which only
#: exists on PPU builds carrying the ``qsa_allow_aiu`` argument.
QSA_KERNELS = ("auto", "cuda", "triton")
MEGA_MOE_MMA_TYPES = ("fp4xfp4", "fp8xfp4", "bf16xfp4", "int8xint8")
STEP5_PREFILL_BACKENDS = ("flashinfer", "native")


def _env_names(name: str) -> tuple[str, ...]:
    """Canonical name followed by its aliases, in precedence order."""
    name = _canonical_names.get(name, name)
    return (name, *_aliases.get(name, ()))


def _read_env(name: str) -> tuple[str, str | None]:
    for candidate in _env_names(name):
        raw = os.getenv(candidate)
        if raw is not None:
            return candidate, raw
    return name, None


def _bool(name: str, default: bool) -> Callable[[], bool]:
    """Read a boolean env var, falling back to aliases only when unset."""

    def _get() -> bool:
        _, raw = _read_env(name)
        if raw is None:
            return default
        return raw.strip().lower() in _TRUTHY

    return _get


def _int(name: str, default: int) -> Callable[[], int]:
    """Read an integer env var, falling back to aliases only when unset."""

    def _get() -> int:
        source, raw = _read_env(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw.strip())
        except ValueError:
            raise ValueError(
                f"Invalid value {raw!r} for {source}. Expected an integer."
            ) from None

    return _get


def _choice(
    name: str, default: str | None, choices: tuple[str, ...]
) -> Callable[[], str | None]:
    """Read a string env var constrained to ``choices`` (case-insensitive)."""

    def _get() -> str | None:
        source, raw = _read_env(name)
        if raw is None or raw.strip() == "":
            return default
        value = raw.strip().lower()
        if value not in choices:
            raise ValueError(
                f"Invalid value {raw!r} for {source}. Expected one of "
                f"{', '.join(choices)}."
            )
        return value

    return _get


def _positive_int(name: str, default: int) -> Callable[[], int]:
    """Read a positive integer env var."""

    def _get() -> int:
        source, raw = _read_env(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Expected an integer for {source}, got {raw!r}.") from exc
        if value <= 0:
            raise ValueError(f"{source} must be positive, got {value}.")
        return value

    return _get


environment_variables: dict[str, Callable[[], Any]] = {
    "PPU_SDK": lambda: os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"),
    # Shared upstream name; PPU extends the choices through its general plugin.
    "VLLM_GDN_DECODE_KERNEL": _choice(
        "VLLM_GDN_DECODE_KERNEL", "auto", GDN_DECODE_KERNELS
    ),
    "VLLM_SAIL_FUSED_RMSNORM_QUANT": _bool("VLLM_SAIL_FUSED_RMSNORM_QUANT", False),
    "VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT": _bool(
        "VLLM_SAIL_USE_OPT_TOKEN_GROUP_QUANT", False
    ),
    "VLLM_SAIL_DENSE_BF16_DEEPGEMM": _bool("VLLM_SAIL_DENSE_BF16_DEEPGEMM", False),
    "VLLM_SAIL_USE_PLA": _bool("VLLM_SAIL_USE_PLA", True),
    "VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED": _bool("VLLM_SAIL_DEEPGEMM_MOE_TP_FUSED", False),
    "VLLM_SAIL_USE_DEEPSELECT": _bool("VLLM_SAIL_USE_DEEPSELECT", True),
    # Fuse the second GEMM's input quantization into DeepGEMM's activation
    # kernel (`silu_and_mul` / `situ_and_mul`) instead of running the activation
    # and a separate per-token/per-block quantize pass. The BF16 (w4a16 or
    # unquantized) experts use the same kernel's values-only epilogue. The same
    # epilogue also replaces the standalone `vllm::silu_and_mul` /
    # `vllm::situ_and_mul` that unquantized BF16 dense MLPs and shared experts
    # run on, through SiluAndMul / SituAndMul.
    "VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT": _bool(
        "VLLM_SAIL_DEEPGEMM_MOE_FUSED_ACT_QUANT", False
    ),
    # -- MoE ---------------------------------------------------------------
    # PPU MoE group-GEMM backend. Unset means "pick the best available",
    # which is why `is_set()` is meaningful here.
    "VLLM_SAIL_MOE_BACKEND": _choice("VLLM_SAIL_MOE_BACKEND", None, GEMM_BACKENDS),
    # PPU dense-GEMM backend for quantized linear layers.
    "VLLM_SAIL_DENSE_BACKEND": _choice("VLLM_SAIL_DENSE_BACKEND", None, GEMM_BACKENDS),
    # Disable / force the wna16 CUDA MoE kernel on PPU. Both exist because the
    # heuristic that picks between the CUDA and Triton kernels is shape
    # dependent and occasionally needs a manual override in either direction.
    "VLLM_SAIL_DISABLE_MOE_WNA16_CUDA": _bool(
        "VLLM_SAIL_DISABLE_MOE_WNA16_CUDA", False
    ),
    "VLLM_SAIL_FORCE_MOE_WNA16_CUDA": _bool("VLLM_SAIL_FORCE_MOE_WNA16_CUDA", False),
    # Legacy Marlin selection gate. Native Marlin execution remains excluded
    # by the CUDA-free manifest; this flag cannot supply those kernels.
    "VLLM_SAIL_ENABLE_MOE_MARLIN": _bool("VLLM_SAIL_ENABLE_MOE_MARLIN", False),
    "VLLM_SAIL_MEGA_MOE_MMA_TYPE": _choice(
        "VLLM_SAIL_MEGA_MOE_MMA_TYPE", "fp8xfp4", MEGA_MOE_MMA_TYPES
    ),
    # -- DSV4.1 mHC ----------------------------------------------------------
    # Disable the PPU Mega mHC kernel so the upstream TileLang mHC path takes
    # over. The A/B baseline for bisecting mHC regressions: the two paths
    # compute the same shifted post/pre block, so a difference in the model's
    # output points at the Mega mHC kernel.
    "VLLM_SAIL_DISABLE_MEGA_MHC": _bool("VLLM_SAIL_DISABLE_MEGA_MHC", False),
    # -- Quantization -------------------------------------------------------
    # Triton implementation of dynamic per-token int8 quant; faster than the
    # compiled kernel on PPU.
    "VLLM_SAIL_USE_TRITON_INT8_QUANT": _bool("VLLM_SAIL_USE_TRITON_INT8_QUANT", True),
    "VLLM_SAIL_FP8_LOAD_POLICY": _choice(
        "VLLM_SAIL_FP8_LOAD_POLICY", None, ("native", "bf16", "int8")
    ),
    "VLLM_SAIL_ENABLE_FP8_TO_INT8": _bool("VLLM_SAIL_ENABLE_FP8_TO_INT8", False),
    # Decode-time logits backend for the glm5_next kpool indexer.
    # ``non_paged`` (default) gathers the pool K cache and uses the non-paged
    # int8 kernel (same one as prefill). The paged INT8 kernel writes only
    # part of the logits on PPU, so selecting it is rejected until fixed.
    "VLLM_SAIL_KPOOL_DECODE_LOGITS": _choice(
        "VLLM_SAIL_KPOOL_DECODE_LOGITS", "non_paged", KPOOL_DECODE_LOGITS_BACKENDS
    ),
    # Dense decode fallback bound: batch * ceil(max_seq_len / kpool) pools.
    # Above it the fallback would materialise an O(batch * context) logits
    # matrix (the 1M-context warmup needs ~256 GiB), so fail loudly instead.
    "VLLM_SAIL_KPOOL_MAX_DENSE_POOLS": _positive_int(
        "VLLM_SAIL_KPOOL_MAX_DENSE_POOLS", 1 << 21
    ),
    # -- Step5 DSA ----------------------------------------------------------
    # Route the Step5 DSA indexer scoring kernel to the resident-block Triton
    # implementation in vllm_sail/models/step5/ops/. Read on every call. Off by
    # default: upstream's kernel stays the reference until validated per model.
    "VLLM_SAIL_STEP5_DSA_SCORES": _bool("VLLM_SAIL_STEP5_DSA_SCORES", False),
    # per_row stays the default; the new selector is opt-in only.
    "VLLM_SAIL_STEP5_DSA_SELECTOR_PPU": _bool(
        "VLLM_SAIL_STEP5_DSA_SELECTOR_PPU", False
    ),
    # Selects native paged attention during prefill metadata planning.
    "VLLM_SAIL_STEP5_DSA_PREFILL_PPU": _bool("VLLM_SAIL_STEP5_DSA_PREFILL_PPU", False),
    # Defaults to FlashInfer; native opts into the new prefill/decode and the old
    # v023 values raise.
    "VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND": _choice(
        "VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND", "flashinfer", STEP5_PREFILL_BACKENDS
    ),
    # The indexer still runs in chunks; only the explicit native route may batch
    # attention at finish.
    "VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION": _bool(
        "VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION", False
    ),
    # 0: E4M3 bytes; 1: E4M3-quantized values stored as FP16, using FP16 MMA.
    # The FP16 summary cache takes twice the memory; keeps the Step5 fork's name.
    # Read lazily; set it before the server starts and keep it fixed at runtime.
    "VLLM_STEP5_INDEXER_FP8_AS_FP16": _bool("VLLM_STEP5_INDEXER_FP8_AS_FP16", False),
    # -- MiniMax M3 SAIL MSA sparse attention -------------------------------
    # Enabled by default: the SAIL path is selected whenever the platform, the
    # model shape and the installed `msa_for_sail` allow it; the dtypes follow
    # the platform and the KV cache dtype (see models/minimax_m3_msa/gating.py).
    # 0 keeps the PPU Triton implementation for both halves.
    "VLLM_SAIL_MINIMAX_M3_MSA": _bool("VLLM_SAIL_MINIMAX_M3_MSA", True),
    # Bisection switches: keep one half on the PPU Triton implementation while
    # the other half runs the SAIL kernels.
    "VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_TRITON": _bool(
        "VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_TRITON", False
    ),
    "VLLM_SAIL_MINIMAX_M3_MSA_ATTEND_TRITON": _bool(
        "VLLM_SAIL_MINIMAX_M3_MSA_ATTEND_TRITON", False
    ),
    # -- Attention / linear attention --------------------------------------
    # Legacy getter only. Current PLA/Triton dispatch uses VLLM_SAIL_USE_PLA.
    "VLLM_SAIL_FUSED_GDN_DECODE": _bool("VLLM_SAIL_FUSED_GDN_DECODE", True),
    # Qwen4Exp QSA sparse-attention kernel. ``auto`` (default) takes FA3's QSA
    # path when the shape gate, the platform gate and the wheel probe all pass,
    # and falls back to the upstream Triton kernel otherwise; ``cuda`` and
    # ``triton`` pin either side for A/B measurement. The fork hard-wires
    # ``Qwen4ExpQSAImpl`` on the layer, so this is the only selection point.
    "VLLM_SAIL_QSA_KERNEL": _choice("VLLM_SAIL_QSA_KERNEL", "auto", QSA_KERNELS),
    # -- DeepSeek V4.1 fused norm-rope attention (FlashMLA PR#54 wheel) -----
    # Master switch for the PPU fused norm-rope sparse-attention path. Default
    # OFF (opt-in): the baseline attention class runs unless the user sets 1.
    # Enabling also requires the PR#54 wheel (capability probe), PPU SM80/SM89,
    # the 528B KV record switch below, and a compatible V4.1 head/group
    # topology; when any gate is closed the selector falls back to baseline
    # and logs the reason (A-B-A control: set 0 to force baseline even with
    # the fused wheel installed).
    "VLLM_SAIL_FUSED_NORM_ROPE_ATTN": _bool("VLLM_SAIL_FUSED_NORM_ROPE_ATTN", False),
    # KV record format for DeepSeek V4.1 on PPU: True -> 528B V4.1 MXFP8
    # record (required by the fused decode kernel; opt-in together with the
    # master switch above), False (default) -> 584B V4 record, identical to
    # the upstream non-SM100 default, so an unconfigured process behaves
    # exactly like baseline.
    # Read once per process by the dsv41_kv_record patch, which also prints
    # the forensic line "[VLLM_SAIL] kv_mxfp8=<bool> kv_bytes_per_token=<n>".
    "VLLM_SAIL_PPU_KV_MXFP8": _bool("VLLM_SAIL_PPU_KV_MXFP8", False),
    # Compressed-KV record for the fused V4.1 path: True -> 288B NVFP4 record
    # (256B e2m1 + 32 per-16 e4m3 scales) for the *compressed* cache only, while
    # the sliding-window cache keeps the 528B MXFP8 record the kernel requires.
    # Default OFF (opt-in): both caches stay 528B MXFP8, byte-identical to the
    # behaviour without this variable. Requires the master switch and
    # VLLM_SAIL_PPU_KV_MXFP8=1, capability 8.9 (ZW-M890P; the e2m1 PTX conversion
    # is unavailable on 8.0) and sliding_window % 8 == 0; the fused gate closes
    # with a named reason otherwise. Lossy: accuracy must be re-measured.
    "VLLM_SAIL_PPU_KV_NVFP4": _bool("VLLM_SAIL_PPU_KV_NVFP4", False),
    # Output FP8 scale format of the fused V4.1 norm-rope attention kernel
    # (FlashMLA-for-sail PR#59, "channel-wise output and runtime format
    # selection"). False (default) -> the PR#54 contract: num_per_channels=32
    # with TMA-aligned, rounded, packed UE8M0 scales, i.e. an int32
    # [tokens, n_wv_group, D/128] buffer. True -> one exact FP32 dequantization
    # scale per token and WV group: num_per_channels=4096 with all three format
    # flags cleared, i.e. a contiguous float32 [tokens, n_wv_group, 1] buffer.
    # Both formats live in the same PR#59 wheel; the kernel hard-requires
    # capability 8.9 (ZW-M890P) for the per-token format and rejects any mixed
    # flag combination, so the fused gate closes with a named reason on 8.0.
    # Opt-in and lossless-relative-to-block32: the FP32 scale avoids the UE8M0
    # power-of-two rounding, so accuracy must be re-measured against the
    # default arm rather than assumed. Requires a PR#59 (or newer) flash_mla
    # wheel; older wheels reject num_per_channels=4096.
    "VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE": _bool(
        "VLLM_SAIL_FUSED_ATTN_PER_TOKEN_SCALE", False
    ),
    # -- Distributed MoE ---------------------------------------------------
    # DeepEP low-latency deferred recv hook. The hook only pays off on backends
    # that can overlap the deferred receive with other compute; setting this to
    # 0 selects an event-based fallback that is also a simpler debugging
    # baseline for overlap issues.
    "VLLM_DEEPEPLL_RECV_HOOK": _bool("VLLM_DEEPEPLL_RECV_HOOK", True),
    # -- Profiling (opt-in; see vllm_sail/profiling/) ------------------------
    "VLLM_SAIL_NVTX_PROFILE": _bool(
        "VLLM_SAIL_NVTX_PROFILE",
        False,
    ),
    # Step5 forward dict arguments expand only tensor shape/dtype; no device reads.
    "VLLM_SAIL_NVTX_DUMP_DICT": _bool("VLLM_SAIL_NVTX_DUMP_DICT", False),
    "VLLM_SAIL_NVTX_DUMP_TOPK": _bool("VLLM_SAIL_NVTX_DUMP_TOPK", False),
    "VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN": _bool(
        "VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN",
        False,
    ),
}

# All plugin variables retain the corresponding PPU spelling. The shared
# upstream VLLM_DEEPEPLL_RECV_HOOK and VLLM_GDN_DECODE_KERNEL, and the Step5
# fork's VLLM_STEP5_INDEXER_FP8_AS_FP16, keep their names and have no plugin
# aliases.
_aliases: dict[str, tuple[str, ...]] = {
    name: (name.replace("VLLM_SAIL_", "VLLM_PPU_", 1),)
    for name in environment_variables
    if name.startswith("VLLM_SAIL_")
}
_aliases["VLLM_SAIL_NVTX_PROFILE"] += ("SAIL_NVTX_PROFILE",)
_aliases["VLLM_SAIL_DISABLE_MEGA_MHC"] += ("SAIL_DISABLE_MEGA_MHC",)
_canonical_names = {
    alias: name for name, aliases in _aliases.items() for alias in aliases
}


def __getattr__(name: str) -> Any:
    name = _canonical_names.get(name, name)
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return [*environment_variables, *_canonical_names]


def is_set(name: str) -> bool:
    """Whether ``name`` or any of its aliases was explicitly set.

    Backend selection distinguishes "user asked for X" from "user did not
    choose", so this is not the same as comparing against the default.
    """
    canonical = _canonical_names.get(name, name)
    return canonical in environment_variables and any(
        candidate in os.environ for candidate in _env_names(canonical)
    )


def snapshot() -> dict[str, Any]:
    """Effective runtime values under canonical names, for ``collect_env``."""
    return {name: getter() for name, getter in environment_variables.items()}
