# SPDX-License-Identifier: Apache-2.0
"""NVTX instrumentation for ``vllm/v1/worker/gpu_model_runner.py``.

Port of the fork's model-runner hunks (+86 lines, 14 hunks) as delegating
patches, per ``docs/developer_guide/architecture.md``. The fork spliced
push/pop calls mid-body into the ~660-line ``execute_model``; instead each
range is reattached at the closest enclosing method boundary so no upstream
body is copied:

* ``[prof_range]: iter N`` + ``prof_iter`` + ``sche_mark`` → wrapper around
  ``execute_model`` (the fork opened it after the early-return checks; the
  wrapper spans the whole method, so no-forward iterations get a range too).
* ``total bs=..., P bs=...`` → wrapper around ``_model_forward``, reading
  the scheduler output that ``execute_model`` stashes on the runner.
* ``[FW_NATIVE] op:sampler,...`` → wrapper around ``_sample``; the fork's
  push/pop already aligned with the method boundaries, so this is exact.
* ``@annotate("prepare_inputs")`` on ``_prepare_inputs`` — exact port.

The compute_logits method of each loaded model is instrumented lazily before
execution, preserving the per-call shape label without copying execute_model.
"""

from __future__ import annotations

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_sail.patch.utils import patch
from vllm_sail.profiling import nvtx

_AFFECTED = ">=0.31.0,<0.32.0"

#: Where the execute_model wrapper stashes the scheduler output so the
#: _model_forward wrapper can label the forward range like the fork did.
_STASH = "_ppu_nvtx_prof_scheduler_output"

_upstream_init = GPUModelRunner.__init__
_upstream_prepare_inputs = GPUModelRunner._prepare_inputs
_upstream_sample = GPUModelRunner._sample
_upstream_model_forward = GPUModelRunner._model_forward
_upstream_execute_model = GPUModelRunner.execute_model


@patch(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner.__init__",
    reason=(
        "[profiling, opt-in] Adds the self.iteration counter that labels the "
        "worker-level NVTX iteration ranges (fork gpu_model_runner.py hunk "
        "@@ -469). Installed only when VLLM_SAIL_NVTX_PROFILE is set."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "model runner."
    ),
)
def runner_init(self, *args, **kwargs):
    _upstream_init(self, *args, **kwargs)
    self.iteration = 0


@patch(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner._prepare_inputs",
    reason=(
        "[profiling, opt-in] NVTX `prepare_inputs` range from the in-tree "
        "PPU fork (fork gpu_model_runner.py hunk @@ -1957). The wrapper "
        "delegates to upstream unchanged."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "model runner."
    ),
)
@nvtx.annotate("prepare_inputs")
def _prepare_inputs(self, *args, **kwargs):
    return _upstream_prepare_inputs(self, *args, **kwargs)


@patch(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner._sample",
    reason=(
        "[profiling, opt-in] Wraps sampling in the fork's "
        "`[FW_NATIVE] op:sampler,logits:<shape>` range (fork "
        "gpu_model_runner.py hunk @@ -3695). The fork's push/pop already "
        "aligned with this method's boundaries, so the range is exact."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "model runner."
    ),
)
def _sample(self, logits, spec_decode_metadata):
    nvtx.range_push(f"[FW_NATIVE] op:sampler,logits:{getattr(logits, 'shape', None)}")
    try:
        return _upstream_sample(self, logits, spec_decode_metadata)
    finally:
        nvtx.range_pop()


@patch(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner._model_forward",
    reason=(
        "[profiling, opt-in] Wraps the forward pass in the fork's "
        "`total bs=..., P bs=...` range (fork gpu_model_runner.py hunks "
        "@@ -4429/@@ -4455). The batch statistics come from the scheduler "
        "output that the execute_model wrapper stashes on the runner; "
        "without it (dummy runs, other callers) the call is uninstrumented."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "model runner."
    ),
)
def _model_forward(self, *args, **kwargs):
    scheduler_output = getattr(self, _STASH, None)
    if scheduler_output is None:
        return _upstream_model_forward(self, *args, **kwargs)

    nvtx.range_push(nvtx.batch_label(scheduler_output))
    try:
        return _upstream_model_forward(self, *args, **kwargs)
    finally:
        nvtx.range_pop()


@patch(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner.execute_model",
    reason=(
        "[profiling, opt-in] Wraps one worker iteration in the fork's "
        "`[prof_range]: iter N` range, prof_iter hook and sche_mark (fork "
        "gpu_model_runner.py hunks @@ -4207/@@ -4233/@@ -4779). The fork "
        "opened the range after the no-forward early returns; the wrapper "
        "spans the whole method, so those iterations get a range too. The "
        "model compute_logits calls receive their own shape-aware range."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "model runner."
    ),
)
def execute_model(self, *args, **kwargs):
    nvtx.instrument_compute_logits(getattr(self, "model", None))
    scheduler_output = args[0] if args else kwargs.get("scheduler_output")
    if scheduler_output is not None:
        nvtx.sche_mark(scheduler_output)
    nvtx.prof_iter(self.iteration)
    nvtx.range_push(f"[prof_range]: iter {self.iteration}")
    self.iteration += 1
    setattr(self, _STASH, scheduler_output)
    try:
        return _upstream_execute_model(self, *args, **kwargs)
    finally:
        nvtx.range_pop()
        setattr(self, _STASH, None)
