# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for GSM8K readiness identity binding.

Covers the behaviours required so a stale vLLM instance lingering on the
accuracy port cannot pass readiness even if it answers HTTP 200 with the
catalog base name:

* Each run derives a unique served model name from a secure nonce.
* ``/v1/models`` probing rejects responses that do not advertise the
  per-run identity -- including malformed JSON, missing ``data``, or a
  non-200 status -- and keeps waiting instead of mis-attributing readiness.
* ``wait_server_ready`` fails cleanly on fatal log patterns and when the
  launched process has exited.
* ``_ensure_launched_server_alive`` guards the post-readiness window so
  evalscope and the golden write path never run against a dead server.

The tests mock ``urlopen`` and ``subprocess.Popen`` so no real network or
PPU device is required.  ``scripts/eval/ppu_gsm8k_eval.py`` is loaded by
file path (same pattern as :mod:`test_model_golden_identity`) so this
module stays importable on hosts without ``evalscope`` or ``vllm``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

EVAL_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "ppu_gsm8k_eval.py"
)


def _load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("ppu_gsm8k_eval", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    _sentinel = "GSM8K_DATASET_PATH" not in os.environ
    if _sentinel:
        os.environ["GSM8K_DATASET_PATH"] = "/unused/gsm8k"
    try:
        spec.loader.exec_module(module)
    finally:
        if _sentinel:
            del os.environ["GSM8K_DATASET_PATH"]
    return module


@pytest.fixture(scope="module")
def evaluator() -> Any:
    return _load_evaluator()


# ---------------------------------------------------------------------------
# Fake urlopen responses + fake subprocess
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _ok_body(ids: list[str]) -> bytes:
    return json.dumps(
        {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
    ).encode()


class _FakeProc:
    """Minimal :class:`subprocess.Popen` double for readiness tests."""

    def __init__(self, exit_after: int | None = None, returncode: int = 0) -> None:
        self._poll_calls = 0
        self._exit_after = exit_after
        self._pending_rc = returncode
        self.returncode: int | None = None
        self.pid = 424242

    def poll(self) -> int | None:
        self._poll_calls += 1
        if self._exit_after is not None and self._poll_calls > self._exit_after:
            self.returncode = self._pending_rc
            return self._pending_rc
        return None


# ---------------------------------------------------------------------------
# Unique served name
# ---------------------------------------------------------------------------


def test_unique_name_contains_base_and_distinct_nonces(evaluator: Any) -> None:
    a = evaluator.unique_served_model_name("MiniMax-M3-FP8")
    b = evaluator.unique_served_model_name("MiniMax-M3-FP8")
    assert a.startswith("MiniMax-M3-FP8-run-")
    assert b.startswith("MiniMax-M3-FP8-run-")
    # Each run must have its own nonce so a cached name cannot be replayed.
    assert a != b
    # Shape: 8 hex chars from secrets.token_hex(4).
    suffix = a.rsplit("-run-", 1)[1]
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


# ---------------------------------------------------------------------------
# _probe_models_endpoint
# ---------------------------------------------------------------------------


def test_probe_rejects_old_models_response_without_nonce(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """200 OK that only advertises the catalog base name must not pass."""

    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(200, _ok_body(["deepseek-v4-mxfp4"]))

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    assert (
        evaluator._probe_models_endpoint(
            "http://127.0.0.1:9985/v1/models",
            "deepseek-v4-mxfp4-run-cafebabe",
        )
        is False
    )


def test_probe_accepts_matching_served_name(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(
            200,
            _ok_body(
                [
                    "deepseek-v4-mxfp4",
                    "deepseek-v4-mxfp4-run-cafebabe",
                ]
            ),
        )

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    assert (
        evaluator._probe_models_endpoint(
            "http://127.0.0.1:9985/v1/models",
            "deepseek-v4-mxfp4-run-cafebabe",
        )
        is True
    )


def test_probe_tolerates_malformed_json(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(200, b"not-json <<<")

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    assert (
        evaluator._probe_models_endpoint(
            "http://127.0.0.1:9985/v1/models",
            "qwen3-run-deadbeef",
        )
        is False
    )


def test_probe_tolerates_missing_data_field(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(200, json.dumps({"object": "list"}).encode())

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    assert (
        evaluator._probe_models_endpoint(
            "http://127.0.0.1:9985/v1/models",
            "qwen3-run-deadbeef",
        )
        is False
    )


def test_probe_tolerates_non_200(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(503, b"")

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    assert (
        evaluator._probe_models_endpoint(
            "http://127.0.0.1:9985/v1/models",
            "qwen3-run-deadbeef",
        )
        is False
    )


def test_probe_tolerates_connection_error(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        raise OSError("connection refused")

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    assert (
        evaluator._probe_models_endpoint(
            "http://127.0.0.1:9985/v1/models",
            "qwen3-run-deadbeef",
        )
        is False
    )


def test_probe_rejects_data_entry_without_dict_id(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(
            200,
            json.dumps({"data": ["qwen3-run-deadbeef"]}).encode(),
        )

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    assert (
        evaluator._probe_models_endpoint(
            "http://127.0.0.1:9985/v1/models",
            "qwen3-run-deadbeef",
        )
        is False
    )


# ---------------------------------------------------------------------------
# wait_server_ready
# ---------------------------------------------------------------------------


def test_wait_server_ready_fails_when_old_service_lacks_nonce(
    evaluator: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A lingering old service returning 200 with the base name must not pass."""
    log = tmp_path / "serve.log"
    log.write_text("")

    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(200, _ok_body(["deepseek-v4-mxfp4"]))

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    monkeypatch.setattr(evaluator.time, "sleep", lambda *_: None)

    proc = _FakeProc()
    with pytest.raises(RuntimeError, match="server not ready"):
        evaluator.wait_server_ready(
            proc,
            log,
            expected_served_name="deepseek-v4-mxfp4-run-abc12345",
            port=9985,
            timeout=1,
        )


def test_wait_server_ready_returns_when_identity_matches(
    evaluator: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log = tmp_path / "serve.log"
    log.write_text("")
    expected = "qwen3-run-abc12345"

    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(200, _ok_body([expected]))

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    monkeypatch.setattr(evaluator.time, "sleep", lambda *_: None)

    evaluator.wait_server_ready(
        _FakeProc(),
        log,
        expected_served_name=expected,
        port=9985,
        timeout=5,
    )  # must not raise


def test_wait_server_ready_detects_subprocess_exit(
    evaluator: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log = tmp_path / "serve.log"
    log.write_text("boom")
    monkeypatch.setattr(
        evaluator,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("conn refused")),
    )
    monkeypatch.setattr(evaluator.time, "sleep", lambda *_: None)

    proc = _FakeProc(exit_after=0, returncode=42)
    with pytest.raises(RuntimeError, match="exited with code 42"):
        evaluator.wait_server_ready(
            proc,
            log,
            expected_served_name="qwen3-run-dead",
            port=9985,
            timeout=5,
        )


def test_wait_server_ready_detects_fatal_log(
    evaluator: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log = tmp_path / "serve.log"
    log.write_text("...\nEngineCore failed to start\n")
    monkeypatch.setattr(
        evaluator,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("conn refused")),
    )
    monkeypatch.setattr(evaluator.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="fatal error detected"):
        evaluator.wait_server_ready(
            _FakeProc(),
            log,
            expected_served_name="qwen3-run-dead",
            port=9985,
            timeout=5,
        )


def test_wait_server_ready_keeps_waiting_on_malformed_json(
    evaluator: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Malformed JSON must not short-circuit readiness as either success or
    failure -- the loop should keep polling until it times out.
    """
    log = tmp_path / "serve.log"
    log.write_text("")

    def fake_urlopen(url: str, timeout: int = 5) -> _FakeResp:
        return _FakeResp(200, b"{not: json")

    monkeypatch.setattr(evaluator, "urlopen", fake_urlopen)
    monkeypatch.setattr(evaluator.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="server not ready"):
        evaluator.wait_server_ready(
            _FakeProc(),
            log,
            expected_served_name="qwen3-run-dead",
            port=9985,
            timeout=1,
        )


# ---------------------------------------------------------------------------
# _ensure_launched_server_alive
# ---------------------------------------------------------------------------


def test_ensure_server_alive_returns_when_running(
    evaluator: Any, tmp_path: Path
) -> None:
    log = tmp_path / "serve.log"
    log.write_text("ok")
    evaluator._ensure_launched_server_alive(
        _FakeProc(), log, phase="post-readiness warm-up"
    )  # must not raise


def test_ensure_server_alive_raises_when_exited(evaluator: Any, tmp_path: Path) -> None:
    log = tmp_path / "serve.log"
    log.write_text("crashed")
    proc = _FakeProc(exit_after=0, returncode=7)
    with pytest.raises(RuntimeError, match="exited with code 7"):
        evaluator._ensure_launched_server_alive(
            proc, log, phase="post-readiness warm-up"
        )
    # Guarded phase name surfaces in the error message so callers can tell
    # which post-readiness window the crash happened in.
    with pytest.raises(RuntimeError, match="post-readiness warm-up"):
        evaluator._ensure_launched_server_alive(
            _FakeProc(exit_after=0, returncode=9),
            log,
            phase="post-readiness warm-up",
        )


# ---------------------------------------------------------------------------
# Static source-level invariants
# ---------------------------------------------------------------------------


def test_source_wires_unique_name_through_server_and_eval() -> None:
    """``run_accuracy_eval`` must route the per-run name through launch,
    readiness, the post-readiness liveness recheck, and the evalscope
    call.  Checked at the source level so a future refactor cannot
    silently drop any of the four touch points.
    """
    text = EVAL_SCRIPT.read_text()
    assert "unique_served_model_name(served_base)" in text
    assert "served_model_name=unique_served" in text
    assert "expected_served_name=unique_served" in text
    assert "_ensure_launched_server_alive(" in text
    # evalscope must be called with the per-run name, not the catalog base.
    assert "run_evalscope(\n            unique_served," in text


def test_golden_identity_not_polluted_by_unique_name() -> None:
    """The persisted golden identity must not include the per-run nonce.

    ``build_accuracy_golden`` keys identity by chip + checkpoint_tail +
    benchmark; a textual check guards against anyone later threading the
    unique served name into the golden record.
    """
    text = EVAL_SCRIPT.read_text()
    # The record passed to build_accuracy_golden has no served_name key.
    build_call = text.split("build_accuracy_golden(", 1)[1].split(")", 1)[0]
    assert "unique_served" not in build_call
    assert "served_base" not in build_call


# ---------------------------------------------------------------------------
# Smoke: _FakeResp body sanity check
# ---------------------------------------------------------------------------


def test_fake_resp_round_trips_json() -> None:
    """Guard the test helper itself so a broken fixture cannot silently
    turn every readiness assertion into a false pass.
    """
    body = _ok_body(["a", "b"])
    assert json.loads(io.BytesIO(body).read())["data"][0]["id"] == "a"
