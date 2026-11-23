# SPDX-License-Identifier: Apache-2.0
"""Dispatch between the new BF16 native prefill/decode and default FlashInfer."""

from __future__ import annotations

import functools
import importlib
import sys
from contextvars import ContextVar

import torch
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.models.step5.ops import dsa_native as modern
from vllm_sail.patch.utils import patch

_ADAPTER = "vllm.models.step5.nvidia.flashinfer_sparse"
_ATTENTION = "vllm.models.step5.nvidia.sparse_attention"
_REFERENCE = ContextVar("step5_prefill_reference", default=False)
_NATIVE_PLAN = ContextVar("step5_prefill_native_plan", default=False)
_ROUTE_ATTRIBUTE = "step5_sail_prefill_route"
_REASON = (
    "Step5 prefill reads the selected paged KV directly in the plugin, avoiding "
    "full BF16 staging. "
    "VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND=native explicitly enables the 20261110 890P BF16 "
    "prefill/decode; the PREFILL_PPU bool can also select the backend, and the "
    "default stays flashinfer. "
    "Unsupported native inputs restore the original FlashInfer runtime before "
    "launch; real launch errors still propagate. "
    "VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION explicitly enables chunked "
    "selection with attention batched at the end."
)
_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = "Upstream prefill attention natively supports PPU BF16/E4M3 paged KV without staging."
_installed = False
_PROFILE_REASON = (
    "When VLLM_SAIL_NVTX_PROFILE is enabled, the Step5 DSA forward entry records "
    "FW_SPARSE prefill/decode ranges with the actual input shape/dtype; it neither "
    "changes compute nor adds device syncs."
)
_PROFILE_REMOVE = "Upstream Step5 DSA forward entry provides the same shape-aware NVTX ranges."


def _profile_forward(original, pd_type, selected_backend="default"):
    # Combined with native prefill into one patch so the method is not wrapped twice.
    @functools.wraps(original, updated=())
    def traced(*args, **kwargs):
        if not envs.VLLM_SAIL_NVTX_PROFILE:
            return original(*args, **kwargs)
        from vllm_sail.profiling import nvtx

        try:
            label = nvtx.step5_sparse_label(
                pd_type, args, kwargs, dump_dict=envs.VLLM_SAIL_NVTX_DUMP_DICT
            ).replace("backend:default,", f"backend:{selected_backend},", 1)
        except (AttributeError, IndexError, TypeError, ValueError):
            # Invalid metadata is left to the original entry; instrumentation
            # never masks its exception.
            return original(*args, **kwargs)
        nvtx.range_push(label)
        try:
            return original(*args, **kwargs)
        finally:
            nvtx.range_pop()

    return traced


def backend():
    if envs.is_set("VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND"):
        return envs.VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND
    if envs.is_set("VLLM_SAIL_STEP5_DSA_PREFILL_PPU"):
        return "native" if envs.VLLM_SAIL_STEP5_DSA_PREFILL_PPU else "flashinfer"
    return envs.VLLM_SAIL_STEP5_DSA_PREFILL_BACKEND


def enabled():
    return (not _REFERENCE.get() and current_platform.is_ppu()
            and backend() != "flashinfer")


def _runtime_type(paged):
    cls = type(paged)
    return f"{cls.__module__}.{cls.__qualname__}"


def _record_route(metadata, requested, paged, stage, reason=None, *, layout=None,
                  positions=None, cache=None):
    """Diagnostics follow this batch's metadata.

    Only config and types are logged; no device reads and no tensor is kept.
    """
    effective = "native" if isinstance(paged, modern.NativePrefill) else "flashinfer"
    route = dict(getattr(metadata, _ROUTE_ATTRIBUTE, {}))
    route.update(requested=requested, effective=effective, stage=stage, reason=reason,
                 runtime_type=_runtime_type(paged), source=__file__)
    tile = getattr(paged, "query_tile", None)
    route["bq"] = tile if effective == "native" else None
    route["branch"] = "native_20261110" if effective == "native" else effective
    route["counter_scope"] = "metadata routing only, not device hits"
    if layout is not None:
        starts = metadata.query_start_loc_cpu[:metadata.num_actual_reqs + 1].tolist()
        route.update(query_dtype=str(layout.dtype), heads_per_group=layout.heads_per_group,
                     head_size=layout.head_size, quantized=layout.quantized,
                     positions_provided=positions is not None, request_starts=starts,
                     query_shape=(starts[-1], layout.heads_per_group, layout.head_size),
                     groups=layout.groups, block_size=layout.block_size,
                     page_stride=layout.page_stride)
    if isinstance(paged, modern.NativePrefill):
        route.update(slots=paged.region_count, query_shape=paged.query_shape,
                     requested_bq=paged.query_tile, batch_attention=paged.batch_attention,
                     indexer_chunks=len(getattr(paged, "chunk_rows", ())))
    if cache is not None:
        route.update(modern.cache_context(cache))
    extension = sys.modules.get("vllm_sail._C")
    route["extension"] = getattr(extension, "__file__", None)
    setattr(metadata, _ROUTE_ATTRIBUTE, route)
    if requested == "native":
        modern.record_route("prefill", route["branch"], reason, stage, route)
    return route


def _install_native_decode(adapter_module):
    target = "vllm.models.step5.common.ops.sparse_decode_attention"
    try:
        module = importlib.import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name and (target == exc.name or target.startswith(exc.name + ".")):
            return
        raise
    original = module.sparse_decode_attention
    replacement = functools.wraps(original, updated=())(
        modern.wrap_decode(original, backend, current_platform))
    # Also update consumer aliases already imported; later imports get the new
    # wrapper anyway.
    aliases = [(name, attr) for name, loaded in list(sys.modules.items())
               if (name == target or name.startswith("vllm.models.step5.nvidia.")) and loaded is not None
               for attr, value in list(vars(loaded).items()) if value is original]
    for name, attr in aliases:
        patch(name, attr, reason="Explicit native tries the new precompiled BF16 decode; unsupported inputs fall back before launch.",
              affected_versions=_AFFECTED, remove_when=_REMOVE_WHEN)(replacement)


def install():
    global _installed
    if _installed:
        return
    try:
        adapter_module = importlib.import_module(_ADAPTER)
        attention_module = importlib.import_module(_ATTENTION)
    except ModuleNotFoundError as exc:
        if exc.name and any(m == exc.name or m.startswith(exc.name + ".")
                            for m in (_ADAPTER, _ATTENTION)):
            return
        raise
    original_prepare = adapter_module.Step5FlashInferSparse.prepare_prefill_attention
    original_plan = attention_module.Step5DSAAttentionImpl.plan_prefill_step
    original_forward = attention_module.Step5DSAAttentionImpl._forward_prefill
    original_decode = attention_module.Step5DSAAttentionImpl._forward_decode
    traced_prefill = {
        name: _profile_forward(original_forward, "prefill", name)
        for name in ("flashinfer", "native")
    }

    @functools.wraps(original_prepare)
    def prepare(self, query_shape, dtype, device, block_size, positions,
                region_count, scale, *, contiguous_query=True):
        # Accept only the planner's one-shot decision; a fallback replan must not pick
        # native again.
        if _NATIVE_PLAN.get() and not _REFERENCE.get():
            reason = modern.candidate_reason(query_shape, dtype, device, block_size,
                                             region_count, scale)
            if reason is None:
                return modern.NativePrefill(query_shape, device, positions,
                                            region_count, scale, 0,
                                            batch_attention=envs.VLLM_SAIL_STEP5_DSA_PREFILL_BATCH_ATTENTION)
            modern.reject(reason)
        return original_prepare(self, query_shape, dtype, device, block_size,
                                positions, region_count, scale,
                                contiguous_query=contiguous_query)

    @functools.wraps(original_plan)
    def plan(self, attn_metadata, n, lengths, layout, positions=None):
        selected_backend = backend()
        if not enabled():
            reference_call = _REFERENCE.get()
            token = _REFERENCE.set(True)
            try:
                runtime = original_plan(self, attn_metadata, n, lengths, layout, positions)
            finally:
                _REFERENCE.reset(token)
            if not reference_call:
                _record_route(attn_metadata, selected_backend,
                              runtime[5] if runtime is not None else None, "plan",
                              "backend_flashinfer" if selected_backend == "flashinfer" else "non_ppu",
                              layout=layout, positions=positions)
            return runtime
        slots = min((int(lengths.max()) + 7) // 8, self.sparse_topk) + 1
        plan_reject = modern.plan_reason(attn_metadata, n, lengths, layout, positions, slots, self.scale)
        use_native = plan_reject is None
        if plan_reject is not None:
            modern.reject(plan_reject)
        # If an inexact MRv2 upper bound exceeds the staging budget, the original
        # planner has no sync-free fallback plan. Keep the original behavior then
        # and do not build a native runtime that may not fall back before launch.
        staging_bytes = int(((lengths + layout.block_size - 1)
                             // layout.block_size).sum()) * layout.block_size * layout.head_size * 4
        if (positions is not None and (
                layout.block_size not in (16, 32, 64, 128)
                or ((layout.quantized or layout.groups > 1)
                    and staging_bytes > attention_module._PREFILL_STAGING_BYTES))):
            if use_native:
                modern.reject("inexact_reference_budget")
                plan_reject = "inexact_reference_budget"
            use_native = False
        mode = _NATIVE_PLAN.set(use_native)
        reference_mode = _REFERENCE.set(not use_native)
        try:
            runtime = original_plan(self, attn_metadata, n, lengths,
                                    layout._replace(quantized=False, groups=1)
                                    if use_native else layout, positions)
        finally:
            _REFERENCE.reset(reference_mode)
            _NATIVE_PLAN.reset(mode)
        if runtime is not None and isinstance(runtime[5], modern.NativePrefill):
            paged = runtime[5]
            paged.fixup_positions(runtime[1])
            paged.owners = runtime[0].reshape(-1).contiguous()
            paged.chunk_rows = tuple(meta.current.numel() for meta in runtime[4])
            # Do not capture metadata, avoiding a reference cycle between the runtime
            # and the scheduler metadata.
            saved_lengths = lengths.copy()

            def reference(impl, metadata):
                token = _REFERENCE.set(True)
                try:
                    return original_plan(impl, metadata, n, saved_lengths, layout, positions)
                finally:
                    _REFERENCE.reset(token)

            paged.reference = reference
        paged = runtime[5] if runtime is not None else None
        if use_native and not isinstance(paged, modern.NativePrefill):
            plan_reject = "planner_runtime_mismatch"
        _record_route(attn_metadata, selected_backend, paged, "plan", plan_reject,
                      layout=layout, positions=positions)
        return runtime

    @functools.wraps(original_forward)
    def forward(self, query, kv_cache, attn_metadata, output, dsa_proxy_query,
                dsa_proxy_key, dsa_proxy_weights, dsa_proxy_z, *,
                summary_updated=False, kv_scales=(None, None)):
        runtime = attn_metadata.step5_runtime
        paged = runtime[5] if runtime is not None else None
        args = (self, query, kv_cache, attn_metadata, output, dsa_proxy_query,
                dsa_proxy_key, dsa_proxy_weights, dsa_proxy_z)
        kwargs = dict(summary_updated=summary_updated, kv_scales=kv_scales)
        if not isinstance(paged, modern.NativePrefill):
            route = getattr(attn_metadata, _ROUTE_ATTRIBUTE, None)
            requested = backend()
            if requested == "native":
                if route is None:
                    _record_route(attn_metadata, requested, paged, "forward", "untracked_runtime")
                elif route["requested"] != requested:
                    _record_route(attn_metadata, requested, paged, "forward", "backend_changed_after_plan")
                elif route["effective"] != "flashinfer" or route["runtime_type"] != _runtime_type(paged):
                    _record_route(attn_metadata, requested, paged, "forward", "runtime_replaced")
            return traced_prefill["flashinfer"](*args, **kwargs)
        count = attn_metadata.num_actual_reqs
        starts = attn_metadata.query_start_loc_cpu[:count + 1].tolist()
        try:
            reason = paged.bind(query, kv_cache, attn_metadata.block_table[:count],
                                starts, output, *kv_scales)
        except BaseException:
            paged.release()
            raise
        if reason is not None:
            paged.release()
            modern.reject(reason)
            reference = paged.reference(self, attn_metadata)
            if reference is None:
                raise RuntimeError("Step5 plugin prefill has no usable fallback plan for the original path")
            attn_metadata.step5_runtime = reference
            _record_route(attn_metadata, "native", reference[5], "bind", reason, cache=kv_cache)
            try:
                return traced_prefill["flashinfer"](*args, **kwargs)
            finally:
                attn_metadata.step5_runtime = runtime
        _record_route(attn_metadata, "native", paged, "forward", cache=kv_cache)
        try:
            # Real kernel execution errors must propagate; never fall back after
            # output was partially written.
            return traced_prefill["native"](*args, **kwargs)
        finally:
            paged.release()

    for module, name, replacement in (
        (_ADAPTER, "Step5FlashInferSparse.prepare_prefill_attention", prepare),
        (_ATTENTION, "Step5DSAAttentionImpl.plan_prefill_step", plan),
        (_ATTENTION, "Step5DSAAttentionImpl._forward_prefill", forward),
    ):
        forward_entry = name.endswith("._forward_prefill")
        patch(module, name, reason=_REASON + (" " + _PROFILE_REASON if forward_entry else ""),
              affected_versions=_AFFECTED,
              remove_when=_REMOVE_WHEN + (" " + _PROFILE_REMOVE if forward_entry else ""))(replacement)
    _install_native_decode(adapter_module)
    patch(_ATTENTION, "Step5DSAAttentionImpl._forward_decode",
          reason=_PROFILE_REASON, affected_versions=_AFFECTED,
          remove_when=_PROFILE_REMOVE)(_profile_forward(original_decode, "decode"))
    _installed = True


install()
