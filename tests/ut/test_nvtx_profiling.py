# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in NVTX profiling subpackage (Phase-1 decision Q6).

Everything here runs on a bare CPU runner: no vLLM, no ``nvtx`` and no
``model_prof`` are required. The fork's engine/worker/scheduler NVTX hunks
became delegating ``@patch`` modules under ``vllm_sail/profiling/`` that
install nothing unless ``VLLM_PPU_NVTX_PROFILE`` is set, so the tests cover:

* gate off -> ``install()`` patches nothing;
* gate on  -> every patch lands on its (faked) upstream target;
* the shared helpers degrade to no-ops without the optional dependencies.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY, original_of

GATE_VARS = ("VLLM_SAIL_NVTX_PROFILE", "VLLM_PPU_NVTX_PROFILE", "SAIL_NVTX_PROFILE")

PROFILING_TARGETS = (
    "vllm.model_executor.layers.linear.ReplicatedLinear.forward",
    "vllm.model_executor.layers.linear.ColumnParallelLinear.forward",
    "vllm.model_executor.layers.linear.RowParallelLinear.forward",
    "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernelModularImpl.apply",
    "vllm.v1.core.sched.scheduler.Scheduler.schedule",
    "vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule",
    "vllm.v1.engine.core.EngineCore.__init__",
    "vllm.v1.engine.core.EngineCore.step",
    "vllm.v1.engine.core.EngineCore.step_with_batch_queue",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_inputs",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model",
)

PROFILING_SUBMODULES = (
    "vllm_sail.profiling.linear",
    "vllm_sail.profiling.modular_moe",
    "vllm_sail.profiling.moe",
    "vllm_sail.profiling.nvtx",
    "vllm_sail.profiling.engine_core",
    "vllm_sail.profiling.model_runner",
    "vllm_sail.profiling.model_runner_v2",
    "vllm_sail.profiling.scheduler",
)


def _sched_output(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "scheduled_new_reqs": [],
        "total_num_scheduled_tokens": 5,
        "num_scheduled_tokens": {"r1": 5},
        "scheduled_spec_decode_tokens": {},
        "finished_req_ids": set(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def cleared_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in GATE_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A minimal fake vLLM tree carrying the three patch targets."""

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def plain(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    vllm = package("vllm")
    vllm_logger = plain("vllm.logger")
    vllm_logger.init_logger = logging.getLogger
    vllm.logger = vllm_logger

    package("vllm.v1")
    package("vllm.v1.worker")
    package("vllm.v1.worker.gpu")
    package("vllm.v1.engine")
    package("vllm.v1.core")
    package("vllm.v1.core.sched")

    class GPUModelRunner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
            return ("prepare_inputs", scheduler_output, num_scheduled_tokens)

        def _sample(self, logits, spec_decode_metadata):
            return ("sample", logits, spec_decode_metadata)

        def _model_forward(self, *args: Any, **kwargs: Any):
            return ("forward", args, kwargs)

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return ("execute_model", scheduler_output, intermediate_tensors)

    class GPUModelRunnerV2:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def prepare_inputs(
            self, scheduler_output, batch_req_state, batch_desc, num_active_loras
        ):
            return ("prepare_inputs", scheduler_output, batch_req_state)

        def sample(self, hidden_states, input_batch, grammar_output):
            return ("sample", hidden_states, input_batch)

        def execute_model(
            self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kw
        ):
            callback = getattr(self, "execute_callback", None)
            if callback is not None:
                return callback()
            return ("execute_model", scheduler_output, dummy_run)

    class EngineCore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_args = (args, kwargs)

        def step(self):
            return ("step",)

        def step_with_batch_queue(self):
            return ("step_with_batch_queue",)

    class Scheduler:
        def schedule(self, throttle_prefills: bool = False):
            return ("schedule", throttle_prefills)

        def _update_after_schedule(self, scheduler_output):
            self.last_update = scheduler_output

    package("vllm.model_executor")
    package("vllm.model_executor.layers")
    package("vllm.model_executor.layers.fused_moe")
    linear = plain("vllm.model_executor.layers.linear")
    for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
        setattr(linear, name, type(name, (), {"forward": lambda *a: None}))
    modular = plain("vllm.model_executor.layers.fused_moe.modular_kernel")
    modular.FusedMoEKernelModularImpl = type(
        "FusedMoEKernelModularImpl", (), {"apply": lambda *a: None}
    )

    runner_module = plain("vllm.v1.worker.gpu_model_runner")
    runner_module.GPUModelRunner = GPUModelRunner
    runner_v2_module = plain("vllm.v1.worker.gpu.model_runner")
    runner_v2_module.GPUModelRunner = GPUModelRunnerV2
    core_module = plain("vllm.v1.engine.core")
    core_module.EngineCore = EngineCore
    scheduler_module = plain("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = Scheduler

    return SimpleNamespace(
        GPUModelRunner=GPUModelRunner,
        GPUModelRunnerV2=GPUModelRunnerV2,
        EngineCore=EngineCore,
        Scheduler=Scheduler,
        originals={
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner.__init__": GPUModelRunner.__init__,
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs": GPUModelRunner._prepare_inputs,
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner._sample": GPUModelRunner._sample,
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner._model_forward": GPUModelRunner._model_forward,
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model": GPUModelRunner.execute_model,
            "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_inputs": (
                GPUModelRunnerV2.prepare_inputs
            ),
            "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample": GPUModelRunnerV2.sample,
            "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model": (
                GPUModelRunnerV2.execute_model
            ),
            "vllm.v1.engine.core.EngineCore.__init__": EngineCore.__init__,
            "vllm.v1.engine.core.EngineCore.step": EngineCore.step,
            "vllm.v1.engine.core.EngineCore.step_with_batch_queue": EngineCore.step_with_batch_queue,
            "vllm.v1.core.sched.scheduler.Scheduler.schedule": Scheduler.schedule,
            "vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule": Scheduler._update_after_schedule,
        },
    )


@pytest.fixture
def nvtx_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Fake `nvtx` package recording mark/annotate usage."""
    calls: dict[str, list] = {"marks": [], "annotated": []}
    stub = types.ModuleType("nvtx")

    def mark(message: str, **kwargs: Any) -> None:
        calls["marks"].append(message)

    def annotate(name: str, **kwargs: Any):
        calls["annotated"].append(name)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                return func(*args, **kw)

            return wrapper

        return decorator

    stub.mark = mark
    stub.annotate = annotate
    monkeypatch.setitem(sys.modules, "nvtx", stub)
    return calls


@pytest.fixture
def profiling_pkg(fake_vllm: SimpleNamespace, monkeypatch: pytest.MonkeyPatch):
    """Import vllm_sail.profiling fresh, with the gate flag reset."""
    module = importlib.import_module("vllm_sail.profiling")
    for name in PROFILING_SUBMODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # `from vllm_sail.profiling import engine_core, ...` resolves via these
    # attributes first; drop them too so install() re-imports the submodules
    # against this test's fakes instead of reusing a stale cached copy.
    for attr in (
        "nvtx",
        "engine_core",
        "model_runner",
        "model_runner_v2",
        "scheduler",
        "linear",
        "modular_moe",
        "moe",
    ):
        monkeypatch.delattr(module, attr, raising=False)
    # Step5 Norm instrumentation is covered by test_step5_norm_profiling; stub it
    # here so the real upstream Step5RMSNorm is not patched into PATCH_REGISTRY.
    norm = types.ModuleType("vllm_sail.profiling.step5_norm")
    norm.install = lambda: False
    monkeypatch.setitem(sys.modules, norm.__name__, norm)
    monkeypatch.setattr(module, "step5_norm", norm, raising=False)
    monkeypatch.setattr(module, "_installed", False)
    return module


@pytest.fixture(params=[False, True], ids=["plain", "dp-scope-first"])
def installed(
    profiling_pkg,
    nvtx_stub: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    fake_vllm: SimpleNamespace,
    request: pytest.FixtureRequest,
):
    """Run install() with the gate on and return the faked targets."""
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    saved_registry = PATCH_REGISTRY[:]
    PATCH_REGISTRY.clear()
    try:
        if request.param:
            from vllm_sail.patch.bugfix import dp_idle_mtp_dummy_batch

            dp_idle_mtp_dummy_batch._install_execute_model_scope()
            fake_vllm.originals[
                "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model"
            ] = fake_vllm.GPUModelRunnerV2.execute_model
        registry_before = len(PATCH_REGISTRY)
        profiling_pkg.install()
        yield SimpleNamespace(
            pkg=profiling_pkg,
            fakes=fake_vllm,
            nvtx_calls=nvtx_stub,
            registry_before=registry_before,
        )
    finally:
        PATCH_REGISTRY[:] = saved_registry


def _records_for(targets) -> dict[str, Any]:
    return {r.target: r for r in PATCH_REGISTRY if r.target in targets}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_gate_off_installs_nothing(
    profiling_pkg, fake_vllm: SimpleNamespace, cleared_gate: None
) -> None:
    records_before = _records_for(PROFILING_TARGETS)
    profiling_pkg.install()

    assert profiling_pkg._installed is False
    for name in PROFILING_SUBMODULES:
        assert name not in sys.modules
    for original in fake_vllm.originals.values():
        assert getattr(original, PATCH_MARKER, None) is None
    assert _records_for(PROFILING_TARGETS) == records_before


def test_gate_on_installs_every_patch(installed) -> None:
    assert installed.pkg._installed is True

    records = _records_for(PROFILING_TARGETS)
    assert set(records) == set(PROFILING_TARGETS)
    assert len(PATCH_REGISTRY) == installed.registry_before + len(PROFILING_TARGETS)

    fakes = installed.fakes
    replacements = {
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.prepare_inputs": (
            fakes.GPUModelRunnerV2.prepare_inputs
        ),
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample": fakes.GPUModelRunnerV2.sample,
        "vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model": (
            fakes.GPUModelRunnerV2.execute_model
        ),
        "vllm.v1.engine.core.EngineCore.__init__": fakes.EngineCore.__init__,
        "vllm.v1.engine.core.EngineCore.step": fakes.EngineCore.step,
        "vllm.v1.engine.core.EngineCore.step_with_batch_queue": fakes.EngineCore.step_with_batch_queue,
        "vllm.v1.core.sched.scheduler.Scheduler.schedule": fakes.Scheduler.schedule,
        "vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule": fakes.Scheduler._update_after_schedule,
    }
    for target, replacement in replacements.items():
        assert target in records
        assert replacement is not fakes.originals[target], target
        assert original_of(replacement, target) is fakes.originals[target], target


def test_install_is_idempotent(installed) -> None:
    count = len(PATCH_REGISTRY)
    installed.pkg.install()
    installed.pkg.install()
    assert len(PATCH_REGISTRY) == count


@pytest.mark.parametrize("gate_var", GATE_VARS)
def test_gate_aliases_enable_profiling(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    gate_var: str,
) -> None:
    monkeypatch.setenv(gate_var, "1")
    profiling_pkg.install()
    assert profiling_pkg._installed is True


@pytest.mark.parametrize("value", ["0", ""])
def test_sail_false_prevents_install_with_truthy_legacy_aliases(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    value: str,
) -> None:
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", value)
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setenv("SAIL_NVTX_PROFILE", "1")
    profiling_pkg.install()
    assert profiling_pkg._installed is False


def test_missing_nvtx_package_disables_with_warning(
    profiling_pkg,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setitem(sys.modules, "nvtx", None)
    with caplog.at_level(logging.WARNING):
        profiling_pkg.install()
    assert profiling_pkg._installed is False
    assert any("nvtx" in message for message in caplog.messages)


def test_missing_model_prof_still_installs(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    cleared_gate: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setitem(sys.modules, "model_prof", None)
    with caplog.at_level(logging.INFO):
        profiling_pkg.install()
    assert profiling_pkg._installed is True
    assert any(
        "without `model_prof`" in record.getMessage() for record in caplog.records
    )


def test_drifted_target_degrades_to_warning(
    profiling_pkg,
    nvtx_stub,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.delitem(sys.modules, "vllm.v1.worker.gpu.model_runner")
    with caplog.at_level(logging.WARNING):
        profiling_pkg.install()
    assert profiling_pkg._installed is False
    assert any("could not be" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# Wrapper behaviour against the faked targets
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_nvtx(installed, monkeypatch: pytest.MonkeyPatch):
    """Swap the helper backends for recorders after install imported them."""
    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    pushes: list[str] = []
    pops: list[int] = []
    iters: list[int] = []
    monkeypatch.setattr(
        prof_nvtx, "_torch_range_push", lambda label: pushes.append(label)
    )
    monkeypatch.setattr(prof_nvtx, "_torch_range_pop", lambda: pops.append(1))
    monkeypatch.setattr(prof_nvtx, "_model_prof_iter", lambda it: iters.append(it))
    return SimpleNamespace(pushes=pushes, pops=pops, iters=iters)


def test_v2_execute_model_wraps_iteration_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    # No __init__ patch: the counter is created on first use, like the engine
    # core does it after the profiling patches load mid-construction.
    assert not hasattr(runner, "iteration")

    output = _sched_output()
    assert runner.execute_model(output) == ("execute_model", output, False)

    assert recording_nvtx.pushes == ["[prof_range]: iter 0", "total bs=1, P bs=1"]
    assert len(recording_nvtx.pops) == 2
    assert recording_nvtx.iters == [0]
    assert runner.iteration == 1
    assert any(m.startswith("total_tokens=5") for m in installed.nvtx_calls["marks"])

    runner.execute_model(output)
    assert recording_nvtx.pushes[2:] == [
        "[prof_range]: iter 1",
        "total bs=1, P bs=1",
    ]


def test_v2_execute_model_forwards_dummy_runs_unprofiled(
    installed, recording_nvtx
) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    output = _sched_output()

    assert runner.execute_model(output, dummy_run=True) == (
        "execute_model",
        output,
        True,
    )
    assert recording_nvtx.pushes == []
    assert recording_nvtx.iters == []
    # Memory profiling and CUDA-graph capture must not consume iteration numbers.
    assert not hasattr(runner, "iteration")


@pytest.mark.parametrize("installed", [True], indirect=True, ids=["dp-scope-first"])
@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_v2_profiling_preserves_dp_scope(
    installed, recording_nvtx, dummy_run, raises
) -> None:
    from vllm_sail.patch.bugfix import dp_idle_mtp_dummy_batch as dp

    runner = installed.fakes.GPUModelRunnerV2()

    def execute():
        assert dp._get_uniform_token_count() is None
        dp._set_uniform_token_count(3)
        if raises:
            raise ValueError("model failed")
        return "done"

    runner.execute_callback = execute
    dp._set_uniform_token_count(32)
    try:
        if raises:
            with pytest.raises(ValueError, match="model failed"):
                runner.execute_model(_sched_output(), dummy_run=dummy_run)
        else:
            assert runner.execute_model(_sched_output(), dummy_run=dummy_run) == "done"
        assert dp._get_uniform_token_count() is None
        expected_ranges = 0 if dummy_run else 2
        assert len(recording_nvtx.pushes) == len(recording_nvtx.pops) == expected_ranges
    finally:
        dp._set_uniform_token_count(None)


def test_v2_sample_wraps_sampler_range(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    hidden_states = SimpleNamespace(shape=(7, 16))

    assert runner.sample(hidden_states, "batch", None) == (
        "sample",
        hidden_states,
        "batch",
    )
    assert recording_nvtx.pushes == ["[FW_NATIVE] op:sampler,hidden_states:(7, 16)"]
    assert len(recording_nvtx.pops) == 1


def test_v2_prepare_inputs_is_annotated(installed) -> None:
    runner = installed.fakes.GPUModelRunnerV2()

    assert runner.prepare_inputs("so", "state", "desc", 0) == (
        "prepare_inputs",
        "so",
        "state",
    )
    assert "prepare_inputs" in installed.nvtx_calls["annotated"]


def test_v2_instruments_compute_logits_once(installed, recording_nvtx) -> None:
    runner = installed.fakes.GPUModelRunnerV2()
    model = SimpleNamespace(compute_logits=lambda hidden_states: hidden_states)
    runner.model = model

    runner.execute_model(_sched_output())
    traced = model.compute_logits
    runner.execute_model(_sched_output())
    assert model.compute_logits is traced, "compute_logits must be wrapped once"

    hidden_states = SimpleNamespace(shape=(2, 4))
    assert model.compute_logits(hidden_states) is hidden_states
    assert recording_nvtx.pushes[-1] == (
        "[FW_GEMM] op:compute_logits,hidden_states:(2, 4)"
    )


def test_engine_core_steps_carry_iteration_ranges(installed, recording_nvtx) -> None:
    core = installed.fakes.EngineCore("arg")
    assert core.iteration == 0
    assert core.init_args == (("arg",), {})

    assert core.step() == ("step",)
    assert core.step_with_batch_queue() == ("step_with_batch_queue",)

    assert recording_nvtx.pushes == [
        "[prof_range]: iter 0",
        "[prof_range]: iter 1",
    ]
    assert len(recording_nvtx.pops) == 2
    assert recording_nvtx.iters == [0, 1]
    assert core.iteration == 2


@pytest.mark.parametrize("step_name", ["step", "step_with_batch_queue"])
def test_engine_core_loads_profiling_during_init(
    profiling_pkg, nvtx_stub, fake_vllm, cleared_gate, monkeypatch, step_name
) -> None:
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")

    def upstream_init(self):
        profiling_pkg.install()
        self.step_fn = getattr(self, step_name)

    monkeypatch.setattr(fake_vllm.EngineCore, "__init__", upstream_init)

    class EngineCoreProc(fake_vllm.EngineCore):
        def __init__(self):
            super().__init__()

    core = EngineCoreProc()
    assert profiling_pkg._installed
    assert not hasattr(core, "iteration")

    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    events = []
    monkeypatch.setattr(
        prof_nvtx, "_model_prof_iter", lambda i: events.append(("iter", i))
    )
    monkeypatch.setattr(
        prof_nvtx, "_torch_range_push", lambda label: events.append(("push", label))
    )
    monkeypatch.setattr(prof_nvtx, "_torch_range_pop", lambda: events.append(("pop",)))

    for iteration in range(2):
        assert core.step_fn() == (step_name,)
        assert core.iteration == iteration + 1

    assert events == [
        ("iter", 0),
        ("push", "[prof_range]: iter 0"),
        ("pop",),
        ("iter", 1),
        ("push", "[prof_range]: iter 1"),
        ("pop",),
    ]


def test_scheduler_schedule_is_annotated_and_update_marks(installed) -> None:
    scheduler = installed.fakes.Scheduler()

    assert scheduler.schedule(True) == ("schedule", True)
    assert "schedule" in installed.nvtx_calls["annotated"]

    output = _sched_output(
        scheduled_new_reqs=[SimpleNamespace(req_id="r1")],
        finished_req_ids={"r9"},
    )
    scheduler._update_after_schedule(output)
    assert scheduler.last_update is output
    marks = installed.nvtx_calls["marks"]
    assert "new_reqs: ['r1']" in marks
    assert "finish_req: {'r9'}" in marks
    assert any(m.startswith("total_tokens=5") for m in marks)


# ---------------------------------------------------------------------------
# Shared helpers without the optional dependencies
# ---------------------------------------------------------------------------


@pytest.fixture(params=[False, True], ids=["bare", "installed-profiler"])
def bare_nvtx_helpers(
    fake_vllm: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Import profiling/nvtx.py fresh with nvtx/model_prof genuinely absent."""
    if request.param:
        # Deleting sys.modules is insufficient: installed packages can be
        # imported again from disk. Reproduce an identity NVTX decorator.
        (tmp_path / "nvtx.py").write_text(
            "def annotate(name):\n    return lambda func: func\n"
            "def mark(message):\n    pass\n"
        )
        (tmp_path / "model_prof.py").write_text("def prof_iter(i):\n    pass\n")
        monkeypatch.syspath_prepend(str(tmp_path))
    # None blocks import even if the SDK/PyPI package is installed on disk.
    monkeypatch.setitem(sys.modules, "nvtx", None)
    monkeypatch.setitem(sys.modules, "model_prof", None)
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)
    module = importlib.import_module("vllm_sail.profiling.nvtx")
    yield module
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)


def test_helpers_import_cleanly_without_optional_deps(bare_nvtx_helpers) -> None:
    assert bare_nvtx_helpers.has_nvtx is False
    assert bare_nvtx_helpers.has_model_prof is False

    bare_nvtx_helpers.mark("ignored")
    bare_nvtx_helpers.range_push("ignored")
    bare_nvtx_helpers.range_pop()
    bare_nvtx_helpers.prof_iter(3)
    bare_nvtx_helpers.sche_mark(_sched_output())


def test_annotate_fallback_is_identity(bare_nvtx_helpers) -> None:
    @bare_nvtx_helpers.annotate("whatever")
    def documented(x: int) -> int:
        """Keeps its name."""
        return x + 1

    assert documented(1) == 2
    assert documented.__name__ == "documented"
    assert documented.__wrapped__ is not None


def test_range_helpers_swallow_backend_errors_and_warn_once(
    bare_nvtx_helpers, monkeypatch
) -> None:
    calls = []
    warnings = []

    def unavailable(*args):
        calls.append(args)
        raise RuntimeError("NVTX functions not installed")

    monkeypatch.setattr(bare_nvtx_helpers, "_torch_range_push", unavailable)
    monkeypatch.setattr(bare_nvtx_helpers, "_torch_range_pop", unavailable)
    monkeypatch.setattr(
        bare_nvtx_helpers.logger, "warning", lambda *args: warnings.append(args)
    )
    for _ in range(2):
        bare_nvtx_helpers.range_push("probe")
        bare_nvtx_helpers.range_pop()
    assert calls == [("probe",), (), ("probe",), ()]
    assert len(warnings) == 1


def test_sche_mark_format_with_nvtx(
    fake_vllm: SimpleNamespace,
    nvtx_stub: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "vllm_sail.profiling.nvtx", raising=False)
    prof_nvtx = importlib.import_module("vllm_sail.profiling.nvtx")
    output = _sched_output(
        scheduled_new_reqs=[SimpleNamespace(req_id="a"), SimpleNamespace(req_id="b")],
        finished_req_ids={"z"},
    )
    prof_nvtx.sche_mark(output)
    assert nvtx_stub["marks"] == [
        "new_reqs: ['a', 'b']",
        "total_tokens=5,req_id:num_tokens={'r1': 5}",
        "finish_req: {'z'}",
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("outcome", ["output", "sample", "failure"])
def test_real_engine_steps_keep_aborts_after_wait_and_close_ranges(
    upstream_source_root,
    profiling_pkg,
    fake_vllm,
    nvtx_stub,
    monkeypatch,
    queued,
    outcome,
):
    """执行完整真实 step；仅在 executor、scheduler 和诊断边界替身。"""
    from collections import deque
    from concurrent.futures import Future
    from contextlib import contextmanager
    from typing import cast

    from tests.support.source import function

    path = upstream_source_root / "vllm/v1/engine/core.py"
    name = "step_with_batch_queue" if queued else "step"
    provider = function(path, "EngineCore." + name, {"Future": Future, "cast": cast})
    monkeypatch.setattr(fake_vllm.EngineCore, name, provider)
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    profiling_pkg.install()
    tracing = importlib.import_module("vllm_sail.profiling.nvtx")
    pushes, pops, calls = [], [], []
    monkeypatch.setattr(tracing, "_torch_range_push", pushes.append)
    monkeypatch.setattr(tracing, "_torch_range_pop", lambda: pops.append(1))
    core = fake_vllm.EngineCore()
    core.num_executed_steps = 0
    core.deferred_step = None
    output, model_output, details, grammar = object(), object(), object(), object()
    scheduled = _sched_output()
    deferred = queued and outcome == "sample"
    next_scheduled = _sched_output(
        pending_structured_output_tokens=True, has_structured_output_spec_decode=True
    )
    drafts = object()
    sampled_future = Future()
    sampled_future.set_result(model_output)
    error = RuntimeError("owned executor failed")
    future, exec_future = Future(), Future()
    if outcome == "failure":
        (exec_future if queued else future).set_exception(error)
        if queued:
            future.set_result(None)
    else:
        future.set_result(None if outcome == "sample" and not queued else model_output)
        exec_future.set_result(None)

    @contextmanager
    def capture(actual):
        assert actual is scheduled
        yield details

    @contextmanager
    def diagnostic(actual):
        assert actual is scheduled or (deferred and actual is next_scheduled)
        calls.append("diagnostic-enter")
        try:
            yield
        finally:
            calls.append("diagnostic-exit")

    def execute(actual, *, non_block):
        assert actual is (next_scheduled if deferred else scheduled)
        assert non_block is True
        calls.append("execute")
        return exec_future if deferred else future

    def sample(actual, *, non_block=False):
        assert actual is grammar and non_block is deferred
        calls.append("sample")
        return sampled_future if deferred else model_output

    def grammar_bitmask(actual):
        assert actual is (next_scheduled if deferred else scheduled)
        if deferred:
            calls.append(("grammar", actual))
        return grammar

    def update_drafts(value, actual):
        assert value is drafts and actual is next_scheduled
        calls.append(("drafts", value, actual))

    def update(actual, value):
        assert actual is scheduled and value is model_output
        assert calls[-1] == "aborts"
        calls.append("update")
        return output

    core.scheduler = SimpleNamespace(
        has_requests=lambda: not queued or deferred,
        schedule=lambda throttle: (next_scheduled if deferred else scheduled)
        if throttle is True
        else pytest.fail("throttle lost"),
        get_grammar_bitmask=grammar_bitmask,
        update_from_output=update,
        update_draft_token_ids_in_output=update_drafts,
    )

    def take_drafts():
        calls.append("take-drafts")
        return drafts

    core.model_executor = SimpleNamespace(
        execute_model=execute,
        sample_tokens=sample,
        take_draft_token_ids=take_drafts,
    )
    core.is_mm_encoder_only = core.is_pooling_model = False
    core.check_for_draft_tokens = deferred
    core._should_throttle_prefills = lambda: True
    core._untimed_iteration_details = (
        lambda actual: details
        if actual is scheduled
        else pytest.fail("unexpected finished schedule")
    )
    core.log_error_detail = diagnostic
    core._wait_for_output = types.MethodType(
        function(path, "EngineCore._wait_for_output", {}), core
    )
    core._process_aborts_queue = lambda: calls.append("aborts")
    core._attach_iteration_details = lambda actual, value: calls.append((actual, value))
    core._attach_weight_version = lambda actual: calls.append(("weight", actual))
    core.batch_queue = deque([(future, scheduled, exec_future)])
    core.batch_queue_size = 2
    if outcome == "failure":
        with pytest.raises(RuntimeError) as raised:
            getattr(core, name)()
        assert raised.value is error
        assert calls[-1] == "diagnostic-exit" and "aborts" not in calls
    else:
        result, executed = getattr(core, name)()
        assert result is output and executed is (not queued or deferred)
        abort_index = calls.index("aborts")
        assert calls[abort_index : abort_index + 4] == [
            "aborts",
            "update",
            (output, details),
            ("weight", output),
        ]
        assert ("sample" in calls) is (outcome == "sample")
        if deferred:
            assert calls[abort_index + 4 :] == [
                "take-drafts",
                ("drafts", drafts, next_scheduled),
                ("grammar", next_scheduled),
                "sample",
            ]
            assert list(core.batch_queue) == [
                (sampled_future, next_scheduled, exec_future)
            ]
        elif queued:
            assert not core.batch_queue
    assert pushes == ["[prof_range]: iter 0"] and pops == [1]
    assert core.iteration == 1
    assert (
        original_of(
            getattr(fake_vllm.EngineCore, name),
            "vllm.v1.engine.core.EngineCore." + name,
        )
        is provider
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("fail", [False, True])
def test_real_no_forward_runner_keeps_state_updates_and_exception_identity(
    upstream_source_root,
    profiling_pkg,
    fake_vllm,
    nvtx_stub,
    monkeypatch,
    fail,
):
    from contextlib import nullcontext

    from tests.support.source import function

    calls, pushes, pops = [], [], []
    result, error = object(), RuntimeError("state update failed")
    source = "gpu/model_runner.py"
    provider = function(
        upstream_source_root / "vllm/v1/worker" / source,
        "GPUModelRunner.execute_model",
        {
            "has_kv_transfer_group": lambda: False,
            "has_ec_transfer": lambda: False,
            "record_function_or_nullcontext": lambda label: nullcontext(),
            "EMPTY_MODEL_RUNNER_OUTPUT": result,
        },
    )
    cls = fake_vllm.GPUModelRunnerV2
    monkeypatch.setattr(cls, "execute_model", provider)
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    profiling_pkg.install()
    tracing = importlib.import_module("vllm_sail.profiling.nvtx")
    monkeypatch.setattr(tracing, "_torch_range_push", pushes.append)
    monkeypatch.setattr(tracing, "_torch_range_pop", lambda: pops.append(1))
    owner = cls()
    scheduled = _sched_output(total_num_scheduled_tokens=0, num_scheduled_tokens={})

    def update(actual):
        assert actual is scheduled
        calls.append("update")
        if fail:
            raise error

    owner.draft_tokens_handler = SimpleNamespace(
        begin_step=lambda actual: calls.append(("begin", actual))
    )
    owner.update_pp_decode_requests = lambda: calls.append("pp")
    owner.finish_requests = update
    owner.free_states = lambda actual: calls.append(("free", actual))
    owner.add_requests = lambda actual: calls.append(("add", actual))
    owner.update_requests = lambda actual: calls.append(("requests", actual))
    owner.encoder_cache = owner.aux_output_connector = None
    owner.block_tables = SimpleNamespace(
        apply_staged_writes=lambda: calls.append("writes")
    )
    owner.kv_connector = SimpleNamespace(
        no_forward=lambda actual: result
        if actual is scheduled
        else pytest.fail("schedule lost")
    )
    owner._merge_ec_connector_no_forward = (
        lambda actual, value: value
        if actual is scheduled
        else pytest.fail("schedule lost")
    )
    if fail:
        with pytest.raises(RuntimeError) as raised:
            owner.execute_model(scheduled)
        assert raised.value is error
    else:
        assert owner.execute_model(scheduled) is result
    expected = [("begin", scheduled), "pp", "update"]
    if not fail:
        expected += [
            ("free", scheduled),
            ("add", scheduled),
            ("requests", scheduled),
            "writes",
        ]
    assert calls == expected
    expected_pushes = ["[prof_range]: iter 0"]
    expected_pushes.append("total bs=0, P bs=0")
    assert pushes == expected_pushes
    assert len(pops) == 2 and owner.iteration == 1


@pytest.mark.upstream_source
@pytest.mark.parametrize("deferred", [False, True])
def test_real_scheduler_update_preserves_output_sets_and_inflight_fence(
    upstream_source_root,
    profiling_pkg,
    fake_vllm,
    nvtx_stub,
    monkeypatch,
    deferred,
):
    from tests.support.source import function

    provider = function(
        upstream_source_root / "vllm/v1/core/sched/scheduler.py",
        "Scheduler._update_after_schedule",
        {},
    )
    monkeypatch.setattr(fake_vllm.Scheduler, "_update_after_schedule", provider)
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    profiling_pkg.install()
    owner = fake_vllm.Scheduler()

    class RequestBoundary:
        def __init__(self):
            self.num_computed_tokens = 3
            self.has_encoder_inputs = False
            self.num_in_flight_tokens = 1
            self.num_tokens = 5
            self.num_output_placeholders = 0
            self.use_structured_output = True
            self.last_sched_seq = 7

    request = RequestBoundary()
    owner.requests = {"r1": request}
    owner.defer_block_free, owner.sched_step_seq = deferred, 11
    owner._inflight_prefills = {request}
    owner.finished_req_ids, owner.reset_preempted_req_ids = {"r9"}, {"r8"}
    output = _sched_output(
        num_scheduled_tokens={"r1": 2},
        finished_req_ids=owner.finished_req_ids,
        preempted_req_ids=owner.reset_preempted_req_ids,
        has_structured_output_requests=False,
    )
    assert owner._update_after_schedule(output) is None
    assert (request.num_computed_tokens, request.num_in_flight_tokens) == (5, 3)
    assert request.last_sched_seq == (11 if deferred else 7)
    assert not request.is_prefill_chunk and not owner._inflight_prefills
    assert output.has_structured_output_requests is True
    assert output.finished_req_ids == {"r9"} and output.preempted_req_ids == {"r8"}
    assert not owner.finished_req_ids and not owner.reset_preempted_req_ids
    assert output.finished_req_ids is not owner.finished_req_ids
    assert output.preempted_req_ids is not owner.reset_preempted_req_ids
    assert "finish_req: {'r9'}" in nvtx_stub["marks"]


def test_layerwise_hooks_balance_ranges_and_deduplicate_shared_draft_layers(modules):
    from tests.support.source import function

    events = []
    skip = type("Dropout", (), {})
    modules(
        "torch",
        nn=SimpleNamespace(
            **{
                n: skip
                for n in ("Identity", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d")
            }
        ),
        cuda=SimpleNamespace(
            nvtx=SimpleNamespace(range_pop=lambda: events.append("pop"))
        ),
    )
    modules(
        "vllm.utils.nvtx_pytorch_hooks",
        construct_marker_dict_and_push=lambda name, layer, args, **kw: events.append(
            (name, args, kw)
        ),
    )

    class Layer:
        def register_forward_pre_hook(self, hook, *, with_kwargs):
            assert with_kwargs and not hasattr(self, "pre")
            self.pre = hook

        def register_forward_hook(self, hook, *, always_call):
            assert always_call and not hasattr(self, "post")
            self.post = hook

    shared = Layer()
    network = SimpleNamespace(
        named_modules=lambda prefix: [(prefix + ".shared", shared)]
    )
    hooks = function("vllm_sail/profiling/layerwise.py", "_make_hooks", {})()
    hooks.register_hooks(network, "target")
    hooks.register_hooks(network, "draft")
    shared.pre(shared, (2,), {"key": 3})
    shared.post(shared, (2,), None)
    assert events == [
        ("target.shared", (2,), {"kwargs": {"key": 3}}),
        "pop",
        ("target.shared", None, {"out_tensor": None}),
        "pop",
    ]
