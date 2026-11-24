# SPDX-License-Identifier: Apache-2.0
"""A cold artifact must be complete before use; a kernel must not hold flock."""

import importlib.util
import logging
import multiprocessing
import sys
from pathlib import Path
from types import ModuleType


def test_shared_cold_build_then_concurrent_kernel_execution(monkeypatch, tmp_path):
    import pytest

    fcntl = pytest.importorskip("fcntl")
    from vllm_sail.models import minimax_m3_msa

    logger = ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    name = "_msa_adapter_lock_test"
    spec = importlib.util.spec_from_file_location(
        name, Path(minimax_m3_msa.__file__).with_name("adapter.py")
    )
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, adapter)
    spec.loader.exec_module(adapter)
    pkg = ModuleType("msa_for_sail")
    api = pkg.api = ModuleType("msa_for_sail.api")
    jit = ModuleType("msa_for_sail.jit")
    jit.CACHE_BASE = tmp_path
    monkeypatch.setitem(sys.modules, "msa_for_sail", pkg)
    monkeypatch.setitem(sys.modules, "msa_for_sail.jit", jit)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2, timeout=10)
    building, peer_started = ctx.Event(), ctx.Event()
    artifact = tmp_path / "variant.so"

    def load():
        if not artifact.exists():
            artifact.write_text("building")
            building.set()
            assert peer_started.wait(5)
            artifact.write_text("ready")
        assert artifact.read_text() == "ready"
        return barrier.wait

    def op():
        api.get_plan_fn()()  # kernel needs both ranks after obtaining its handle

    for attr in ("fmha_sail_plan", "fmha_sail", "sparse_topk_select"):
        setattr(pkg, attr, op)
    for attr in adapter._MODULE_GETTERS:
        setattr(api, attr, load)

    def rank(peer):
        ops = adapter.sail_msa_ops()
        if peer:
            assert building.wait(5)
            with (tmp_path / adapter._LOCK_NAME).open() as fd:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            peer_started.set()
        ops.fmha_sail_plan()

    procs = [ctx.Process(target=rank, args=(peer,)) for peer in (False, True)]
    for proc in procs:
        proc.start()
    try:
        for proc in procs:
            proc.join(15)
        assert [proc.exitcode for proc in procs] == [0, 0]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join()
