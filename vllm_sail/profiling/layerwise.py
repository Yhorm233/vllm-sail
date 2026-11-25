# SPDX-License-Identifier: Apache-2.0
"""Complete V2 Layer NVTX and draft/MTP module traversal in the plugin.

Reuses --enable-layerwise-nvtx-tracing and upstream's tensor marker format and does
not depend on VLLM_SAIL_NVTX_PROFILE or PyPI nvtx. Hooks register only after
the first target/draft tracing finishes and change no model, sampling,
compilation or CUDA Graph configuration.
"""

from __future__ import annotations

import functools
import importlib

from vllm.logger import init_logger

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.32.0,<0.33.0"
_REMOVE_WHEN = (
    "Upstream V2 natively supports draft/MTP Layer NVTX and deduplicate shared modules."
)
_INSTALLED: set[str] = set()


def _make_hooks():
    """Reuse v1 input/output labels; only dedupe registration and close on errors."""
    import torch
    from vllm.utils.nvtx_pytorch_hooks import construct_marker_dict_and_push

    class SharedModuleHooks:
        def __init__(self):
            self.module_to_name_map = {}

        def module_fwd_pre_hook(self, module, args, kwargs):
            construct_marker_dict_and_push(
                self.module_to_name_map[module], module, args, kwargs=kwargs
            )

        def module_fwd_hook(self, module, args, output):
            torch.cuda.nvtx.range_pop()
            construct_marker_dict_and_push(
                self.module_to_name_map[module], module, None, out_tensor=output
            )
            torch.cuda.nvtx.range_pop()

        def register_hooks(self, network_model, module_prefix="top"):
            skip_types = (
                torch.nn.Identity,
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            )
            for name, module in network_model.named_modules(prefix=module_prefix):
                if isinstance(module, skip_types) or module in self.module_to_name_map:
                    continue
                # Register target first, then walk draft; shared layers keep the
                # first name.
                self.module_to_name_map[module] = name
                module.register_forward_pre_hook(
                    self.module_fwd_pre_hook, with_kwargs=True
                )
                module.register_forward_hook(self.module_fwd_hook, always_call=True)

    return SharedModuleHooks()


def register_layerwise_nvtx_hooks(self) -> None:
    """Register where the runner finished its first tracing; no model walk when off."""
    config = self.vllm_config
    if not config.observability_config.enable_layerwise_nvtx_tracing:
        return
    if getattr(self, "layerwise_nvtx_hooks_registered", False):
        return

    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    compilation = self.compilation_config
    if compilation.mode == CompilationMode.STOCK_TORCH_COMPILE:
        logger.warning_once(
            "STOCK_TORCH_COMPILE does not support per-layer NVTX hooks; skipping registration. "
            "Use --enforce-eager to see per-layer markers."
        )
        return
    if (
        compilation.mode != CompilationMode.NONE
        or compilation.cudagraph_mode != CUDAGraphMode.NONE
    ):
        logger.warning_once(
            "Compiled paths may bypass module hooks, and CUDA Graph replay does not rerun "
            "per-layer NVTX hooks. Use --enforce-eager to see per-layer markers."
        )

    # Resolve the draft type before registering so a failed optional import does
    # not leave half of the target hooks. Only the last PP rank holds the
    # drafter/speculator; other ranks register target only.
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    proposer = self.speculator
    draft = proposer.model if isinstance(proposer, DraftModelSpeculator) else None
    hooks = _make_hooks()
    hooks.register_hooks(self.model, self.model.__class__.__name__)
    if draft is not None:
        hooks.register_hooks(draft, f"draft.{draft.__class__.__name__}")
    self._sail_layerwise_nvtx_hooks = hooks
    self.layerwise_nvtx_hooks_registered = True


def install() -> None:
    """Install the config-gated compat entries; idempotent, independent of the
    outer profiling env var.
    """
    module_path = "vllm.v1.worker.gpu.model_runner"
    if module_path in _INSTALLED:
        return
    try:
        runner = importlib.import_module(module_path).GPUModelRunner
        original_dummy_run = runner._dummy_run

        @functools.wraps(register_layerwise_nvtx_hooks)
        def register(self):
            return register_layerwise_nvtx_hooks(self)

        patch(
            "vllm.v1.worker.gpu.model_runner",
            "GPUModelRunner._register_layerwise_nvtx_hooks",
            reason="[profiling, config-gated] Per-layer markers for target and draft with shared-module deduplication.",
            affected_versions=_AFFECTED,
            remove_when=_REMOVE_WHEN,
            allow_missing=True,
        )(register)
        _install_v2_dummy_run(module_path, original_dummy_run)
        _INSTALLED.add(module_path)
    except Exception as exc:
        logger.warning(
            "Layer NVTX compat entry not installed on %s: %s", module_path, exc
        )


def _install_v2_dummy_run(module_path, original):
    @patch(
        module_path,
        "GPUModelRunner._dummy_run",
        reason="[profiling, config-gated] Register Layer NVTX after the first V2 target/draft dummy forwards finish.",
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )
    @functools.wraps(original)
    def dummy_run(self, *args, **kwargs):
        # Do not register when execute_model(dummy_run=True) returns: draft has not
        # traced yet.
        result = original(self, *args, **kwargs)
        self._register_layerwise_nvtx_hooks()
        return result
