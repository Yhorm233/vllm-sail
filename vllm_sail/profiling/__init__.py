# SPDX-License-Identifier: Apache-2.0
"""NVTX profiling instrumentation for PPU — strictly opt-in.

The in-tree PPU fork threads NVTX ranges through
``v1/worker/gpu_model_runner.py`` (+86 lines / 14 hunks),
``v1/engine/core.py`` (+59), ``v1/core/sched/scheduler.py`` (+34) and
``vllm/benchmarks/serve.py`` (+28). Phase-0 triage established that **all** of
those changes are instrumentation: not one line is needed for PPU to compute
correctly.

Keeping them in the always-on patch set would put three of the five largest-hunk
patches on the production path, for observability nobody uses in production. So
they live here instead and install **nothing** unless ``VLLM_SAIL_NVTX_PROFILE`` is
set and the optional dependencies import.

Consequences, all good:

* zero overhead and zero patch risk when profiling is off;
* the profiling patches may lag an upstream bump without blocking a release;
* a missing ``nvtx`` / ``model_prof`` degrades to a warning, not a crash.
"""

from __future__ import annotations

from vllm.logger import init_logger

import vllm_sail.envs as envs

logger = init_logger(__name__)

_installed = False


def is_enabled() -> bool:
    """Whether NVTX profiling was requested via the environment."""
    return bool(envs.VLLM_SAIL_NVTX_PROFILE)


def install() -> None:
    """Install NVTX instrumentation patches if requested. Idempotent."""
    global _installed
    if _installed or not is_enabled():
        return

    try:
        import nvtx  # noqa: F401
    except ImportError:
        logger.warning(
            "VLLM_SAIL_NVTX_PROFILE is set but the `nvtx` package is not "
            "installed, so NVTX profiling is disabled. Install it with "
            "`pip install vllm-sail[profiling]`."
        )
        return

    # `model_prof` ships with the PPU SDK rather than PyPI. Its absence only
    # disables per-iteration profiler hooks, not NVTX ranges, so it is optional
    # within an already-optional feature.
    try:
        import model_prof  # noqa: F401

        has_model_prof = True
    except ImportError:
        has_model_prof = False
        logger.info(
            "NVTX profiling enabled without `model_prof` (PPU SDK): NVTX ranges "
            "will be emitted but per-iteration profiler hooks are unavailable."
        )

    # Imported lazily so a profiling patch that breaks on a new upstream version
    # cannot affect a non-profiling run. Each submodule applies its @patch
    # decorations on import, so a failure here means the instrumentation drifted
    # from upstream; observability must never block engine startup, so degrade
    # to a warning instead of propagating.
    #
    # Both model runners are patched: vLLM 0.31 defaults to V2 while some
    # configs fall back to V1, and which class runs is only known once the
    # engine config exists. The unused one costs nothing.
    try:
        from vllm_sail.profiling import (  # noqa: F401
            engine_core,
            linear,
            model_runner,
            model_runner_v2,
            modular_moe,
            scheduler,
        )
    except Exception as exc:
        logger.warning(
            "NVTX profiling requested but its instrumentation could not be "
            "installed (%s). Profiling is disabled; this usually means the "
            "profiling patches drifted from the installed vLLM version.",
            exc,
        )
        return

    _installed = True
    logger.info("vllm-sail NVTX profiling installed (model_prof=%s).", has_model_prof)
