# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.v1.attention.ops.flashmla``.

Upstream's FlashMLA op shims assume the compiled vLLM extensions
(``vllm._flashmla_C`` / ``vllm._flashmla_extension_C``). PPU ships the upstream
``flash_mla`` package instead, so the fork:

* makes ``_is_flashmla_available`` check for ``flash_mla`` on PPU;
* makes ``is_flashmla_dense_supported`` / ``is_flashmla_sparse_supported``
  accept any device capability on PPU once available;
* binds ``FlashMLASchedMeta``, ``flash_mla_sparse_fwd``, ``flash_mla_with_kvcache``
  and ``get_mla_metadata`` from ``flash_mla`` on PPU;
* wraps the dense decode kernel call in an ``[FM_FMHA]`` NVTX range when
  ``VLLM_SAIL_NVTX_PROFILE`` is set. The fork builds that range inside
  ``FlashMLAImpl.forward_mqa`` around ``flash_mla_with_kvcache``; annotating
  the op here keeps the range on the kernel itself and covers every caller;
* routes ``get_mla_metadata_dense_fp8`` / ``flash_mla_with_kvcache_fp8`` on PPU
  to the sail dense FP8 kernel. PPU does not build ``_flashmla_extension_C``, so
  the metadata call forwards to the sail ``get_mla_metadata`` (returning a lazy
  ``FlashMLASchedMeta`` exactly like the non-FP8 path) and the FP8 decode is
  dispatched through ``flash_mla.flash_mla_with_kvcache`` (qkv_fp8), which fills
  its scheduling metadata lazily from that ``FlashMLASchedMeta``. The paired
  ``_build_decode`` patch (see ``flashmla_backend``) keeps upstream's dense-FP8
  int32/CUDA-graph metadata path off PPU. The sail kernel exposes no descale
  parameters, so the caller quantizes q with ``descale_q``, folds
  ``q_scale * k_scale`` into ``softmax_scale`` (QK part) and rescales the output
  by ``k_scale`` (V part); validated by ``tools/validate_ppu_dense_fp8.py``.

The callables are patched with lazy wrappers (importing ``flash_mla`` on first
PPU call) because availability is only knowable on the target host; on non-PPU
platforms they delegate to the upstream bindings untouched.

``FlashMLASchedMeta`` is *not* patched: upstream consumers only use it in type
annotations bound at their own import time, and the real class is what the
patched ``get_mla_metadata`` returns at runtime.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.v1.attention.ops import flashmla as _flashmla_ops

import vllm_sail.envs as ppu_envs
from vllm_sail.patch.utils import PATCH_MARKER, patch

_AFFECTED = ">=0.32.0,<0.33.0"
_MODULE = "vllm.v1.attention.ops.flashmla"

try:  # torch.cuda.nvtx is absent on CPU-only torch builds.
    from torch.cuda.nvtx import range_pop as _range_pop
    from torch.cuda.nvtx import range_push as _range_push
except ImportError:  # pragma: no cover
    _range_push = None
    _range_pop = None

_upstream_is_flashmla_available = _flashmla_ops._is_flashmla_available
_upstream_is_flashmla_dense_supported = _flashmla_ops.is_flashmla_dense_supported
_upstream_is_flashmla_sparse_supported = _flashmla_ops.is_flashmla_sparse_supported
_upstream_get_mla_metadata_dense_fp8 = _flashmla_ops.get_mla_metadata_dense_fp8
_upstream_flash_mla_with_kvcache_fp8 = _flashmla_ops.flash_mla_with_kvcache_fp8
_upstream_flash_mla_sparse_fwd = _flashmla_ops.flash_mla_sparse_fwd
_upstream_flash_mla_with_kvcache = _flashmla_ops.flash_mla_with_kvcache
_upstream_get_mla_metadata = _flashmla_ops.get_mla_metadata

_FLASHMLA_ALIAS_CONSUMERS = {
    "flash_mla_sparse_fwd": (
        "vllm.models.deepseek_v4.nvidia.flashmla",
        "vllm.models.deepseek_v41.nvidia.flashmla",
        "vllm.models.hy_v4.nvidia.flashmla_sparse",
        "vllm.v1.attention.backends.mla.flashmla_sparse",
    ),
    "flash_mla_with_kvcache": (
        "vllm.models.deepseek_v4.nvidia.flashmla",
        "vllm.models.deepseek_v41.nvidia.flashmla",
        "vllm.models.hy_v4.nvidia.flashmla_sparse",
        "vllm.v1.attention.backends.mla.flashmla",
        "vllm.v1.attention.backends.mla.flashmla_sparse",
    ),
    "flash_mla_with_kvcache_fp8": ("vllm.v1.attention.backends.mla.flashmla",),
    "get_mla_metadata": (
        "vllm.models.hy_v4.nvidia.flashmla_sparse",
        "vllm.v1.attention.backends.mla.flashmla",
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        "vllm.v1.attention.backends.mla.sparse_swa",
    ),
    "get_mla_metadata_dense_fp8": ("vllm.v1.attention.backends.mla.flashmla",),
    "is_flashmla_dense_supported": ("vllm.v1.attention.backends.mla.flashmla",),
    "is_flashmla_sparse_supported": (
        "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn",
    ),
}

_ALIAS_REASON = (
    "vLLM 0.31 captured the upstream FlashMLA op before PPU patch "
    "installation, so the consumer would bypass the PPU provider or retain "
    "CUDA-only capability checks."
)
_ALIAS_REMOVE_WHEN = (
    "vLLM resolves FlashMLA operations through a provider registry or module "
    "lookup instead of capturing them by value."
)


def _ppu_flashmla_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("flash_mla") is not None


def _nvtx_push(label: str) -> bool:
    if _range_push is None:
        return False
    _range_push(label)
    return True


def _nvtx_pop() -> None:
    _range_pop()


def _fm_fmha_nvtx_push(call_kwargs: dict) -> bool:
    """Push the fork's ``[FM_FMHA]`` range for one dense FlashMLA decode call."""
    if not ppu_envs.VLLM_SAIL_NVTX_PROFILE:
        return False

    if "indices" in call_kwargs:
        # Sparse / absorbed decode variants reuse this op; the fork's [FM_FMHA]
        # range only brackets the dense kernel.
        return False

    q = call_kwargs.get("q")
    k_cache = call_kwargs.get("k_cache")
    if q is None or k_cache is None:
        # Called positionally: skip the range instead of guessing the layout.
        return False

    # q is already concatenated and reshaped, k_cache already carries the head
    # dim, so the label matches the fork's without any tuple handling.
    max_seqlen_q = q.shape[-3] if len(q.shape) == 4 else 1
    label = (
        f"[FM_FMHA] --format=MLA,Forward,type:D,seqlen_q:{max_seqlen_q},"
        f"head_dim:{q.shape[-1]},head_dim_v:{call_kwargs.get('head_dim_v')},"
        f"num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},"
        f"batch_size:{q.shape[0]},data_type:{q.dtype},"
        f"causal:{call_kwargs.get('causal', True)}"
    )
    if (
        k_cache.dim() >= 4
        and not torch.cuda.is_current_stream_capturing()
        and ppu_envs.VLLM_SAIL_NVTX_DUMP_TOPK
    ):
        cache_seqlens = call_kwargs.get("cache_seqlens")
        cu_seqlens_k_list = (
            cache_seqlens.flatten().cpu().tolist()
            if cache_seqlens is not None
            else "[]"
        )
        label += (
            f",num_blocks:{k_cache.shape[-4]},page_block_size:{k_cache.shape[-3]},"
            f"cu_seqlens_k:{cu_seqlens_k_list}"
        )
    return _nvtx_push(label)


@patch(
    _MODULE,
    "_is_flashmla_available",
    reason=(
        "PPU ships the upstream flash_mla package instead of vLLM's compiled "
        "_flashmla_C/_flashmla_extension_C extensions. Upstream reports "
        "FlashMLA unavailable unless those extensions imported, so on PPU the "
        "check must look for flash_mla instead."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU builds vLLM's FlashMLA extensions, or upstream grows a pluggable FlashMLA provider.",
)
def _is_flashmla_available() -> tuple[bool, str | None]:
    if current_platform.is_ppu():
        if _ppu_flashmla_available():
            return True, None
        return (
            False,
            "ppu flashmla is not available, please install flashmla. ",
        )
    return _upstream_is_flashmla_available()


@patch(
    _MODULE,
    "is_flashmla_dense_supported",
    reason=(
        "PPU supports dense FlashMLA on its own capability; upstream restricts "
        "it to Hopper (family 90). The fork returns (True, None) on PPU once "
        "flash_mla is available."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream's capability check accepts PPU for dense FlashMLA.",
)
def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    if current_platform.is_ppu():
        is_available, maybe_reason = _is_flashmla_available()
        if not is_available:
            return False, maybe_reason
        return True, None
    return _upstream_is_flashmla_dense_supported()


@patch(
    _MODULE,
    "is_flashmla_sparse_supported",
    reason=(
        "PPU supports sparse FlashMLA on its own capability; upstream restricts "
        "it to Hopper and Blackwell datacenter devices. The fork returns "
        "(True, None) on PPU once flash_mla is available."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream's capability check accepts PPU for sparse FlashMLA.",
)
def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    if current_platform.is_ppu():
        is_available, maybe_reason = _is_flashmla_available()
        if not is_available:
            return False, maybe_reason
        return True, None
    return _upstream_is_flashmla_sparse_supported()


@patch(
    _MODULE,
    "flash_mla_sparse_fwd",
    reason=(
        "On PPU the sparse FlashMLA entry point comes from the upstream "
        "flash_mla package, not from vllm.third_party.flashmla (which wraps the "
        "compiled CUDA extensions PPU does not build). Lazy import keeps non-PPU "
        "hosts and hosts without flash_mla unaffected."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU builds vLLM's FlashMLA extensions, or upstream grows a pluggable FlashMLA provider.",
)
def flash_mla_sparse_fwd(*args, **kwargs):
    if current_platform.is_ppu():
        from flash_mla import flash_mla_sparse_fwd as _ppu_impl

        return _ppu_impl(*args, **kwargs)
    return _upstream_flash_mla_sparse_fwd(*args, **kwargs)


@patch(
    _MODULE,
    "flash_mla_with_kvcache",
    reason=(
        "On PPU flash_mla_with_kvcache comes from the upstream flash_mla "
        "package rather than vllm.third_party.flashmla. Same reasoning as "
        "flash_mla_sparse_fwd. The PPU branch also emits the fork's [FM_FMHA] "
        "NVTX range when VLLM_SAIL_NVTX_PROFILE is set; the fork pushes it in "
        "FlashMLAImpl.forward_mqa around this very call."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU builds vLLM's FlashMLA extensions, or upstream grows a pluggable FlashMLA provider.",
)
def flash_mla_with_kvcache(*args, **kwargs):
    if current_platform.is_ppu():
        from flash_mla import flash_mla_with_kvcache as _ppu_impl

        pushed = _fm_fmha_nvtx_push(kwargs)
        try:
            return _ppu_impl(*args, **kwargs)
        finally:
            if pushed:
                _nvtx_pop()
    return _upstream_flash_mla_with_kvcache(*args, **kwargs)


@patch(
    _MODULE,
    "get_mla_metadata",
    reason=(
        "On PPU get_mla_metadata comes from the upstream flash_mla package "
        "rather than vllm.third_party.flashmla. Same reasoning as "
        "flash_mla_sparse_fwd."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU builds vLLM's FlashMLA extensions, or upstream grows a pluggable FlashMLA provider.",
)
def get_mla_metadata(*args, **kwargs):
    if current_platform.is_ppu():
        from flash_mla import get_mla_metadata as _ppu_impl

        return _ppu_impl(*args, **kwargs)
    return _upstream_get_mla_metadata(*args, **kwargs)


@patch(
    _MODULE,
    "get_mla_metadata_dense_fp8",
    reason=(
        "PPU does not build the _flashmla_extension_C op upstream uses for the "
        "FP8 dense FlashMLA metadata. The PPU FP8 decode routes to the sail "
        "dense kernel, which fills its scheduling metadata lazily from a "
        "FlashMLASchedMeta -- the same object the non-FP8 get_mla_metadata "
        "returns -- so this forwards to the sail get_mla_metadata instead of "
        "producing upstream's int32 tensors. The paired _build_decode patch "
        "(flashmla_backend) attaches this sched meta to the field forward_mqa's "
        "FP8 branch reads, keeping upstream's tensor/CUDA-graph copies off PPU."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU builds the FP8 dense FlashMLA metadata extension natively.",
)
def get_mla_metadata_dense_fp8(*args, **kwargs):
    if current_platform.is_ppu():
        from flash_mla import get_mla_metadata as _ppu_impl

        # The sail get_mla_metadata ignores its arguments and returns
        # (FlashMLASchedMeta(), None); the kernel fills the metadata lazily on
        # the first flash_mla_with_kvcache call. Forwarding *args/**kwargs keeps
        # this agnostic to upstream's positional signature, like get_mla_metadata.
        return _ppu_impl(*args, **kwargs)
    return _upstream_get_mla_metadata_dense_fp8(*args, **kwargs)


#: Host-side cache for the folded FP8 dense softmax scale, keyed by the scale
#: tensors' data_ptrs. The sail kernel takes ``softmax_scale`` as a host float,
#: but vLLM passes ``descale_q``/``descale_k`` as device tensors; reading them
#: with ``.item()`` is illegal during CUDA-graph capture, so the value is read
#: once (vLLM runs an eager warmup before capture) and reused thereafter.
_PPU_DENSE_FP8_SCALE_CACHE: dict = {}


def _ppu_dense_fp8_folded_scale(softmax_scale, descale_q, descale_k) -> float:
    """Return ``softmax_scale * q_scale * k_scale`` as a host float (cached)."""
    dq_is_t = isinstance(descale_q, torch.Tensor)
    dk_is_t = isinstance(descale_k, torch.Tensor)
    if dq_is_t or dk_is_t:
        key = (
            float(softmax_scale),
            descale_q.data_ptr() if dq_is_t else 0,
            descale_k.data_ptr() if dk_is_t else 0,
        )
        cached = _PPU_DENSE_FP8_SCALE_CACHE.get(key)
        if cached is not None:
            return cached
        q_scale = float(descale_q) if dq_is_t else float(descale_q or 1.0)
        k_scale = float(descale_k) if dk_is_t else float(descale_k or 1.0)
        val = float(softmax_scale) * q_scale * k_scale
        _PPU_DENSE_FP8_SCALE_CACHE[key] = val
        return val
    return float(softmax_scale) * float(descale_q or 1.0) * float(descale_k or 1.0)


@patch(
    _MODULE,
    "flash_mla_with_kvcache_fp8",
    reason=(
        "PPU does not build the _flashmla_extension_C op upstream uses for the "
        "FP8 dense FlashMLA decode kernel. Route it to the sail dense FP8 kernel "
        "(flash_mla.flash_mla_with_kvcache, qkv_fp8). The sail kernel exposes no "
        "descale parameters: it treats the FP8 q/k as scale-1 and returns BF16 "
        "output without dequantizing V, so the caller quantizes q with descale_q, "
        "folds q_scale*k_scale into softmax_scale (QK part) and rescales the "
        "output by k_scale (V part). Validated by tools/validate_ppu_dense_fp8.py."
    ),
    affected_versions=_AFFECTED,
    remove_when="PPU builds the FP8 dense FlashMLA decode extension natively.",
)
def flash_mla_with_kvcache_fp8(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits,
    softmax_scale=None,
    causal=False,
    descale_q=None,
    descale_k=None,
):
    if current_platform.is_ppu():
        from flash_mla import FlashMLASchedMeta
        from flash_mla import flash_mla_with_kvcache as _ppu_dense_fwd

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        folded_scale = _ppu_dense_fp8_folded_scale(softmax_scale, descale_q, descale_k)

        # The sail dense FP8 kernel requires q and k_cache to share the FP8
        # dtype; quantize q with descale_q so that q_real == q_fp8 * descale_q.
        if descale_q is not None:
            q = q.to(torch.float32) / descale_q
        q_fp8 = q.to(torch.float8_e4m3fn)

        # tile_scheduler_metadata is the lazy sail FlashMLASchedMeta built by the
        # patched _build_decode (via get_mla_metadata_dense_fp8); the kernel fills
        # it on first use and asserts shape consistency on reuse. The sail kernel
        # asserts this same contract internally, so assert here to fail fast when
        # the paired patches are missing -- a raw upstream int32 tensor would
        # otherwise be silently masked by a freshly built sched meta.
        sched_meta = tile_scheduler_metadata
        assert isinstance(sched_meta, FlashMLASchedMeta), (
            "PPU FP8 dense decode expects tile_scheduler_metadata to be a sail "
            "FlashMLASchedMeta from the patched get_mla_metadata_dense_fp8/"
            f"_build_decode, got {type(sched_meta).__name__}; the paired "
            "patches are likely not installed."
        )
        o, lse = _ppu_dense_fwd(
            q_fp8,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            sched_meta,
            None,
            softmax_scale=folded_scale,
            causal=causal,
        )
        # V is not dequantized inside the kernel: rescale the output by k_scale.
        if descale_k is not None:
            o = (o.to(torch.float32) * descale_k).to(o.dtype)
        return o, lse
    return _upstream_flash_mla_with_kvcache_fp8(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        descale_q=descale_q,
        descale_k=descale_k,
    )


def _rebind_loaded_flashmla_aliases() -> None:
    """Update consumers imported before the PPU FlashMLA provider patches."""
    import sys

    for alias_name, consumer_names in _FLASHMLA_ALIAS_CONSUMERS.items():
        replacement = getattr(_flashmla_ops, alias_name)
        target = f"{_MODULE}.{alias_name}"
        original = getattr(replacement, PATCH_MARKER, {}).get(target)

        for consumer_name in consumer_names:
            consumer = sys.modules.get(consumer_name)
            if consumer is None:
                continue
            captured = getattr(consumer, alias_name)
            if captured is replacement:
                continue
            if original is None or captured is not original:
                raise RuntimeError(
                    f"{consumer_name}.{alias_name} is not the expected "
                    "vLLM 0.30 FlashMLA alias"
                )
            patch(
                consumer_name,
                alias_name,
                reason=_ALIAS_REASON,
                affected_versions=_AFFECTED,
                remove_when=_ALIAS_REMOVE_WHEN,
            )(replacement)


_rebind_loaded_flashmla_aliases()
