#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Temporary location in vllm-sail; migrates to centralized hub executor/ in Phase 2.
"""Execute ``fast_sanity`` and ``model_sanity`` from ``ppu-serving/v1`` recipes."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from http.client import HTTPConnection, HTTPException, HTTPResponse, IncompleteRead
from pathlib import Path
from threading import Timer
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ppu_recipe import (  # noqa: E402
    redact_and_verify_public_payload,
    resolve_recipe,
    validate_paths,
)

STREAM_PROBE_TIMEOUT_S = 60.0
STREAM_PROBE_MAX_BYTES = 64 * 1024

FATAL_RE = re.compile(
    r"Traceback \(most recent"
    r"|EngineCore failed"
    r"|Engine core initialization failed"
    r"|core dumped"
    r"|Failed to infer device type"
    r"|Error: model architectures"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return data


def _resolve_dataset_spec(dataset_ref: str) -> dict[str, Any]:
    if "@" not in dataset_ref:
        raise ValueError(
            f"dataset reference must be <id>@<version>, got {dataset_ref!r}"
        )
    ds_id, ds_ver = dataset_ref.split("@", 1)
    ds_path = REPO_ROOT / "tests" / "ppu" / "policy" / "datasets" / f"{ds_id}.yaml"
    spec = _load_yaml(ds_path)
    if spec.get("id") != ds_id or str(spec.get("version")) != ds_ver:
        raise ValueError(
            f"{ds_path}: expected id={ds_id!r} version={ds_ver!r}, "
            f"got id={spec.get('id')!r} version={spec.get('version')!r}"
        )
    return spec


def _sanitize_error_text(text: str, *, max_chars: int = 1200) -> str:
    """Strip host paths, IPs, and internal hosts from failure excerpts."""
    cleaned = re.sub(
        r"/(?:nas_aisw|ppusw|mnt|Users|home|root|data)/[^\s'\"]+",
        "<redacted_path>",
        text,
    )
    cleaned = re.sub(
        r"[A-Za-z0-9._-]+\.(?:alibaba-inc\.com|eng\.t-head\.cn)",
        "<redacted_host>",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b", "<redacted_ip>", cleaned
    )
    if len(cleaned) > max_chars:
        cleaned = cleaned[-max_chars:]
    return cleaned


def _http_get_json(url: str, *, timeout: float = 5.0) -> tuple[int, Any]:
    with urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _probe_ready(port: int, expected_served_name: str) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            if resp.status != 200:
                return False
        status, payload = _http_get_json(
            f"http://127.0.0.1:{port}/v1/models", timeout=5
        )
        if status != 200 or not isinstance(payload, dict):
            return False
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id") == expected_served_name:
                return True
    except (URLError, OSError, ValueError):
        return False
    return False


def _stop_server(
    proc: subprocess.Popen[bytes] | None, *, grace_s: float = 30.0
) -> None:
    """Stop vLLM server and all surviving workers in its process group."""
    if proc is None:
        return
    # start_new_session=True sets pgid == proc.pid; workers may outlive the leader.
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass

    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass

    # Unconditionally SIGKILL the process group even if leader already exited.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _record_phase(phase: str) -> None:
    if phase_file := os.environ.get("PPU_RECIPE_PHASE_FILE"):
        Path(phase_file).write_text(phase + "\n", encoding="utf-8")


def _wait_server_ready(
    proc: subprocess.Popen[bytes],
    log_path: Path,
    *,
    port: int,
    expected_served_name: str,
    timeout: int,
) -> float:
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = (
                log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
                if log_path.exists()
                else ""
            )
            raise RuntimeError(
                f"vllm serve exited with code {proc.returncode} before ready.\n{tail}"
            )
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if FATAL_RE.search(text):
                raise RuntimeError(f"fatal error in vllm serve log:\n{text[-2500:]}")
        if _probe_ready(port, expected_served_name):
            return round(time.monotonic() - t0, 3)
        time.sleep(3)
    tail = (
        log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
        if log_path.exists()
        else ""
    )
    _record_phase("server_readiness_timeout")
    raise RuntimeError(f"server not ready within {timeout}s.\n{tail}")


def _chat_probe_body(served_name: str, probe: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": served_name,
        "messages": [{"role": "user", "content": probe["prompt"]}],
    }
    if "seed" in probe:
        body["seed"] = int(probe["seed"])
    return body


def _send_chat_probe(
    *,
    port: int,
    served_name: str,
    probe: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:
    body = _chat_probe_body(served_name, probe)
    body["max_completion_tokens"] = int(probe.get("max_tokens", 128))
    body["temperature"] = float(probe.get("temperature", 0.0))
    req = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    elapsed = round(time.monotonic() - t0, 3)
    choice = raw["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    combined = f"{content}\n{reasoning}".strip()
    completion_tokens = int((raw.get("usage") or {}).get("completion_tokens") or 0)
    expected = probe.get("expected_substring")
    if expected:
        if not combined:
            raise RuntimeError(f"empty completion from chat probe: {raw}")
        if expected not in combined:
            raise RuntimeError(
                f"probe response missing expected substring {expected!r}: {combined[:200]!r}"
            )
    elif completion_tokens < 1 and not combined:
        raise RuntimeError(f"empty dummy completion from chat probe: {raw}")
    return {
        "status": "pass",
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(content),
        "completion_tokens": completion_tokens,
    }


class _StreamProbeError(RuntimeError):
    """Carry an allowlisted failure code without response or exception text."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _stream_data_events(response: HTTPResponse) -> Iterator[bytes]:
    remaining = STREAM_PROBE_MAX_BYTES
    fields: list[bytes] = []
    while line := response.readline(remaining + 1):
        remaining -= len(line)
        if remaining < 0:
            raise _StreamProbeError("stream_response_too_large")
        if not line.endswith(b"\n"):
            break
        line = line.rstrip(b"\r\n")
        if not line:
            if fields:
                yield b"\n".join(fields)
                fields.clear()
            continue
        field, _, value = line.partition(b":")
        if field == b"data":
            fields.append(value.removeprefix(b" "))


def _reject_json_constant(value: str) -> None:
    raise ValueError("unexpected JSON constant")


def _check_stream_completion(response: HTTPResponse) -> int:
    content_chars = 0
    has_content = False
    finished = False
    try:
        for event in _stream_data_events(response):
            if event == b"[DONE]":
                if not finished:
                    raise _StreamProbeError("truncated_stream")
                if not has_content:
                    raise _StreamProbeError("empty_stream_content")
                return content_chars
            try:
                chunk = json.loads(event, parse_constant=_reject_json_constant)
                choices = chunk["choices"]
            except (ValueError, TypeError, KeyError, RecursionError):
                raise _StreamProbeError("invalid_stream_chunk") from None
            if finished and choices == [] and isinstance(chunk.get("usage"), dict):
                continue
            if finished or not isinstance(choices, list) or len(choices) != 1:
                raise _StreamProbeError("invalid_stream_chunk")
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(
                choice.get("delta"), dict
            ):
                raise _StreamProbeError("invalid_stream_chunk")
            delta = choice["delta"]
            if delta.get("role") not in (None, "assistant"):
                raise _StreamProbeError("invalid_stream_chunk")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise _StreamProbeError("invalid_stream_chunk")
                content_chars += len(content)
                has_content |= bool(content.strip())
            finish = choice.get("finish_reason")
            if finish is not None:
                if not isinstance(finish, str) or not finish.strip():
                    raise _StreamProbeError("invalid_stream_chunk")
                finished = True
    except (IncompleteRead, ConnectionResetError):
        pass
    raise _StreamProbeError("missing_done" if finished else "truncated_stream")


def _send_stream_chat_probe(
    *,
    port: int,
    served_name: str,
    probe: dict[str, Any],
    timeout: float = STREAM_PROBE_TIMEOUT_S,
) -> dict[str, Any]:
    stream = probe["stream"]
    body = _chat_probe_body(served_name, probe)
    body.update(stream=True, temperature=0.0, max_tokens=stream["max_tokens"])
    if "enable_thinking" in stream:
        body["chat_template_kwargs"] = {"enable_thinking": stream["enable_thinking"]}

    started = time.monotonic()
    deadline = started + timeout
    connection = HTTPConnection("127.0.0.1", port=port, timeout=timeout)
    timer = None
    try:
        connection.connect()
        connected_socket = connection.sock
        assert connected_socket is not None

        def interrupt_read() -> None:
            try:
                connected_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        # Socket timeouts alone allow a peer to keep a request alive by sending
        # small fragments. Shutdown also interrupts a blocked header or body read.
        timer = Timer(max(0.0, deadline - time.monotonic()), interrupt_read)
        timer.daemon = True
        timer.start()
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with connection.getresponse() as response:
            if response.status != 200:
                raise _StreamProbeError("stream_http_error")
            if response.headers.get_content_type() != "text/event-stream":
                raise _StreamProbeError("invalid_stream_chunk")
            content_chars = _check_stream_completion(response)
        if time.monotonic() >= deadline:
            raise _StreamProbeError("stream_timeout")
    except (TimeoutError, OSError, HTTPException, _StreamProbeError) as exc:
        if isinstance(exc, TimeoutError) or time.monotonic() >= deadline:
            raise _StreamProbeError("stream_timeout") from None
        if isinstance(exc, _StreamProbeError):
            raise
        raise _StreamProbeError("stream_transport_error") from None
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()
    return {
        "status": "pass",
        "elapsed_s": round(time.monotonic() - started, 3),
        "content_chars": content_chars,
    }


def _parse_prometheus_metric(metrics_text: str, metric_name: str) -> float | None:
    total = 0.0
    found = False
    prefix = metric_name
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(prefix + " ") or line.startswith(prefix + "{"):
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                    found = True
                except ValueError:
                    continue
    return total if found else None


def _drain_and_reset_cache(port: int, *, timeout: int = 120) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        running = _parse_prometheus_metric(text, "vllm:num_requests_running") or 0.0
        waiting = _parse_prometheus_metric(text, "vllm:num_requests_waiting") or 0.0
        if running == 0.0 and waiting == 0.0:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"server queues failed to drain within {timeout}s")

    req = Request(
        f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"
    )
    with urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"/reset_prefix_cache returned HTTP {resp.status}")
        body_text = resp.read().decode("utf-8", errors="replace").strip()
    if body_text:
        try:
            body_json = json.loads(body_text)
        except ValueError:
            body_json = None
        if isinstance(body_json, dict) and body_json.get("success") is False:
            raise RuntimeError(f"/reset_prefix_cache reported failure: {body_json}")
    return {"cache_reset_evidence": "endpoint_ack_only", "http_status": 200}


def assess_per_question_baseline(
    records: list[dict[str, Any]],
    *,
    expected_count: int,
    aggregate_score: float,
    aggregate_num: int,
    unscored: int = 0,
) -> dict[str, Any]:
    count = len(records)
    ids = [
        str(item["question_id"])
        for item in records
        if isinstance(item["question_id"], str | int)
        and not isinstance(item["question_id"], bool)
        and str(item["question_id"]).strip()
    ]
    unique_ids = len(set(ids))
    unscored += sum(not isinstance(item["correct"], bool) for item in records)
    correct_sum = sum(item["correct"] is True for item in records)
    reasons = []
    if len(ids) != count:
        reasons.append("缺少稳定题目 ID")
    if unique_ids != len(ids):
        reasons.append("重复 ID")
    if count != expected_count:
        reasons.append(f"数量不符：预期 {expected_count}，实际 {count}")
    if unscored:
        reasons.append(f"存在未评分题：{unscored}")
    if (
        not count
        or aggregate_num != count
        or not math.isfinite(aggregate_score)
        or not 0.0 <= aggregate_score <= 1.0
        or correct_sum != round(aggregate_score * aggregate_num)
        or abs(correct_sum / count - aggregate_score) > 1 / count
    ):
        reasons.append("与汇总不一致")
    return {
        "status": "baseline_incomplete" if reasons else "baseline_established",
        "reasons": reasons,
        "count": count,
        "unique_ids": unique_ids,
        "unscored": unscored,
        "correct_sum": correct_sum,
    }


def _extract_per_question_evalscope(
    eval_work_dir: Path,
    output_jsonl: Path,
    *,
    expected_count: int,
    aggregate_score: float,
    aggregate_num: int,
    dataset: str = "gsm8k",
    subset: str = "main",
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    unscored = 0
    missing_ids = 0
    source_reasons = []
    candidates = sorted(
        path
        for path in eval_work_dir.rglob(f"{dataset}_{subset}.jsonl")
        if path.parent.parent.name == "reviews"
    )
    source = candidates[0] if len(candidates) == 1 else None
    if source is None:
        source_reasons.append(f"评分文件数量不符：预期 1，实际 {len(candidates)}")
    else:
        try:
            lines = source.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeError):
            lines = []
            source_reasons.append("评分文件读取失败")
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                unscored += 1
                continue
            sample_score = row.get("sample_score") if isinstance(row, dict) else None
            score = (
                sample_score.get("score") if isinstance(sample_score, dict) else None
            )
            if (
                not isinstance(score, dict)
                or score.get("status") != "success"
                or score.get("main_score_name") not in {None, "accuracy", "acc"}
            ):
                unscored += 1
                continue
            value = score.get("value")
            if isinstance(value, dict):
                value = value.get("accuracy", value.get("acc"))
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or value not in (0.0, 1.0)
            ):
                unscored += 1
                continue
            question_id = sample_score.get("sample_id")
            if question_id is None:
                question_id = row.get("index")
            if (
                isinstance(question_id, bool)
                or not isinstance(question_id, str | int)
                or not str(question_id).strip()
            ):
                missing_ids += 1
                continue
            records.append(
                {
                    "question_id": str(question_id),
                    "correct": value == 1.0,
                    "score": float(value),
                }
            )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for item in records:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    baseline = assess_per_question_baseline(
        records,
        expected_count=expected_count,
        aggregate_score=aggregate_score,
        aggregate_num=aggregate_num,
        unscored=unscored,
    )
    if missing_ids:
        source_reasons.append(f"缺少稳定题目 ID：{missing_ids}")
    if source_reasons:
        baseline["status"] = "baseline_incomplete"
        baseline["reasons"].extend(source_reasons)
    try:
        evalscope_version = importlib.metadata.version("evalscope")
    except importlib.metadata.PackageNotFoundError:
        evalscope_version = "unknown"
    return {
        **baseline,
        "path": output_jsonl.name,
        "source_file": str(source.relative_to(eval_work_dir)) if source else None,
        "evalscope_version": evalscope_version,
        "dataset": dataset,
        "subset": subset,
    }


def _launch_resolved_server(
    resolved: dict[str, Any],
    *,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in resolved.get("server_env", {}).items()})
    env.update({k: str(v) for k, v in resolved.get("runtime_env", {}).items()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            resolved["argv"],
            start_new_session=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    finally:
        log_fh.close()


def _attempt_fast_sanity_once(
    *,
    recipe_path: Path,
    policy_path: Path,
    variant: str | None = None,
    preset: str | None,
    assets_map: Path | None,
    port: int,
    ready_timeout: int,
    layer_override: int | None,
    probe: dict[str, Any],
    results_dir: Path,
    label: str,
) -> dict[str, Any]:
    base_name = f"recipe-{label}-{secrets.token_hex(4)}"
    resolved = resolve_recipe(
        recipe_path,
        policy_path=policy_path,
        tier="fast_sanity",
        variant=variant,
        preset=preset,
        layer_override=layer_override,
        assets_map_path=assets_map,
        port=port,
        served_model_name=base_name,
    )
    log_path = results_dir / f"vllm_serve_{label}.log"
    proc: subprocess.Popen[bytes] | None = None
    t_start = time.monotonic()
    try:
        _record_phase("server_startup")
        proc = _launch_resolved_server(resolved, log_path=log_path)
        cold_start_s = _wait_server_ready(
            proc,
            log_path,
            port=port,
            expected_served_name=base_name,
            timeout=ready_timeout,
        )
        _record_phase("chat_probe")
        t_eval0 = time.monotonic()
        probe_res = _send_chat_probe(port=port, served_name=base_name, probe=probe)
        attempt = {
            "mode": label,
            "layer_override": layer_override,
            "status": "pass",
            "cold_start_s": cold_start_s,
            "eval_s": round(time.monotonic() - t_eval0, 3),
            "config_digest": resolved["config_digest"],
            "ppu_count": resolved["ppu_count"],
            "probe": probe_res,
        }
        if "stream" in probe:
            _record_phase("stream_probe")
            t_stream = time.monotonic()
            try:
                attempt["stream_probe"] = _send_stream_chat_probe(
                    port=port, served_name=base_name, probe=probe
                )
            except _StreamProbeError as exc:
                attempt["status"] = "fail"
                attempt["error"] = exc.reason
                attempt["stream_probe"] = {
                    "status": "fail",
                    "error": exc.reason,
                    "elapsed_s": round(time.monotonic() - t_stream, 3),
                }
            attempt["eval_s"] = round(time.monotonic() - t_eval0, 3)
        return attempt
    except Exception as exc:
        elapsed = round(time.monotonic() - t_start, 3)
        return {
            "mode": label,
            "layer_override": layer_override,
            "status": "fail",
            "cold_start_s": elapsed,
            "eval_s": 0.0,
            "config_digest": resolved["config_digest"],
            "ppu_count": resolved["ppu_count"],
            "error": _sanitize_error_text(str(exc)),
        }
    finally:
        _stop_server(proc)


def run_fast_sanity(
    *,
    recipe_path: Path,
    policy_path: Path,
    results_dir: Path,
    assets_map: Path | None = None,
    variant: str | None = None,
    preset: str | None = None,
    port: int = 9985,
    ready_timeout: int = 1800,
    layer_mode: str = "auto",
    parser_mode: str = "required",
    queue_s: float = 0.0,
) -> dict[str, Any]:
    """Run ``fast_sanity`` (2-layer dummy first, optional fallback to full-layer dummy)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    val_report = validate_paths(
        [recipe_path, policy_path], vllm_parser_mode=parser_mode
    )
    policy = _load_yaml(policy_path)
    fs_cfg = policy["tiers"]["fast_sanity"]
    probe = fs_cfg["probe"]
    fallback_full = bool(fs_cfg.get("fallback_full_layers", True))
    eff_variant = variant if variant is not None else policy.get("variant", "default")
    configured_layers = fs_cfg.get("layer_override", 2)

    attempts: list[dict[str, Any]] = []
    if layer_mode in {"auto", "2l", "both"} and (
        configured_layers is not None or layer_mode in {"2l", "both"}
    ):
        two_l = int(configured_layers if configured_layers is not None else 2)
        res_2l = _attempt_fast_sanity_once(
            recipe_path=recipe_path,
            policy_path=policy_path,
            variant=eff_variant,
            preset=preset,
            assets_map=assets_map,
            port=port,
            ready_timeout=ready_timeout,
            layer_override=two_l,
            probe=probe,
            results_dir=results_dir,
            label="fast_2l",
        )
        attempts.append(res_2l)
    if (
        layer_mode in {"full", "both"}
        or not attempts
        or (
            layer_mode == "auto"
            and fallback_full
            and attempts[-1]["status"] != "pass"
            and "stream_probe" not in attempts[-1]
        )
    ):
        res_full = _attempt_fast_sanity_once(
            recipe_path=recipe_path,
            policy_path=policy_path,
            variant=eff_variant,
            preset=preset,
            assets_map=assets_map,
            port=port + (1 if attempts else 0),
            ready_timeout=ready_timeout,
            layer_override=None,
            probe=probe,
            results_dir=results_dir,
            label="fast_full",
        )
        attempts.append(res_full)

    stream_failure = next(
        (
            item
            for item in attempts
            if item.get("stream_probe", {}).get("status") == "fail"
        ),
        None,
    )
    final_attempt = stream_failure or next(
        (item for item in attempts if item["status"] == "pass"), attempts[-1]
    )
    if stream_failure is not None:
        _record_phase(stream_failure["error"])
    summary = {
        "tier": "fast_sanity",
        "model_id": policy["model_id"],
        "variant": eff_variant,
        "preset": preset,
        "status": final_attempt["status"],
        "parser_checked": val_report["parser_checked"],
        "config_digest": final_attempt["config_digest"],
        "timings": {
            "queue_s": round(float(queue_s), 3),
            "cold_start_s": final_attempt["cold_start_s"],
            "eval_s": final_attempt["eval_s"],
            "bench_s": 0.0,
        },
        "attempts": attempts,
    }
    if "probe" in final_attempt:
        summary["timings"]["nonstream_s"] = final_attempt["probe"]["elapsed_s"]
    if "stream_probe" in final_attempt:
        summary["timings"]["stream_s"] = final_attempt["stream_probe"]["elapsed_s"]
    redact_and_verify_public_payload(summary, allow_loopback=True)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _run_vllm_bench(
    *,
    served_name: str,
    model_path: str,
    port: int,
    concurrency: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    trust_remote_code: bool = False,
    result_dir: Path,
    result_filename: str,
) -> dict[str, Any]:
    if not (0.0 <= range_ratio < 1.0):
        raise ValueError(f"range_ratio must be in [0, 1), got {range_ratio!r}")
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--model",
        served_name,
        "--tokenizer",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(range_ratio),
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(num_prompts),
        "--ignore-eos",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    out_file = result_dir / result_filename
    out_file.unlink(missing_ok=True)
    subprocess.run(cmd, check=True)
    return json.loads(out_file.read_text(encoding="utf-8"))


def _bench_completed(result: Any, expected: int) -> bool:
    return (
        expected > 0
        and isinstance(result, dict)
        and type(result.get("completed")) is int
        and result["completed"] == expected
        and type(result.get("failed")) is int
        and result["failed"] == 0
    )


def run_model_sanity(
    *,
    recipe_path: Path,
    policy_path: Path,
    results_dir: Path,
    assets_map: Path | None = None,
    variant: str | None = None,
    preset: str | None = None,
    port: int = 9985,
    ready_timeout: int = 3600,
    parser_mode: str = "required",
    queue_s: float = 0.0,
    limit: int | None = None,
    post_ready_sleep: int = 30,
) -> dict[str, Any]:
    """Run ``model_sanity``: 1 server -> probe -> GSM8K 1319 -> drain/reset -> bench [1,16]."""
    results_dir.mkdir(parents=True, exist_ok=True)
    val_report = validate_paths(
        [recipe_path, policy_path], vllm_parser_mode=parser_mode
    )
    policy = _load_yaml(policy_path)
    stages = {st["kind"]: st for st in policy["tiers"]["model_sanity"]["stages"]}
    eval_stage = stages["evalscope"]
    dataset_spec = _resolve_dataset_spec(eval_stage["dataset"])
    if (dataset_spec["id"], dataset_spec["subset_list"], dataset_spec["scoring"]) != (
        "gsm8k",
        ["main"],
        "evalscope-default",
    ):
        raise ValueError("model_sanity supports only GSM8K main with EvalScope scoring")
    dataset_options = {
        key: dataset_spec[key]
        for key in ("subset_list", "few_shot_num", "few_shot_random", "filters")
    }
    eval_seed = eval_stage["seed"]
    eval_batch_size = eval_stage["eval_batch_size"]
    if (
        type(eval_seed) is not int
        or type(eval_batch_size) is not int
        or eval_batch_size <= 0
    ):
        raise ValueError(
            "EvalScope requires an integer seed and a positive integer batch size"
        )

    resolved = resolve_recipe(
        recipe_path,
        policy_path=policy_path,
        tier="model_sanity",
        variant=variant,
        preset=preset,
        assets_map_path=assets_map,
        port=port,
        served_model_name=f"recipe-ms-{secrets.token_hex(4)}",
    )
    unique_served = resolved["served_name"]
    gsm8k_path = os.environ.get("GSM8K_DATASET_PATH", "")
    if not gsm8k_path:
        raise OSError("GSM8K_DATASET_PATH must be set for model_sanity")

    ppu_gsm8k_eval = importlib.import_module("scripts.eval.ppu_gsm8k_eval")
    accuracy_gate = importlib.import_module("tests.e2e.models._accuracy_gate")

    log_path = results_dir / "vllm_serve_model_sanity.log"
    proc: subprocess.Popen[bytes] | None = None
    try:
        _record_phase("server_startup")
        proc = _launch_resolved_server(resolved, log_path=log_path)
        ready_budget = max(
            ready_timeout,
            int(
                resolved.get("server_env", {}).get("VLLM_ENGINE_READY_TIMEOUT_S", 0)
                or 0
            ),
        )
        cold_start_s = _wait_server_ready(
            proc,
            log_path,
            port=port,
            expected_served_name=unique_served,
            timeout=ready_budget,
        )
        if post_ready_sleep > 0:
            time.sleep(post_ready_sleep)
        ppu_gsm8k_eval._ensure_launched_server_alive(
            proc, log_path, phase="post-readiness warm-up"
        )

        # 1. Deterministic sanity probe
        _record_phase("chat_probe")
        probe_res = _send_chat_probe(
            port=port,
            served_name=unique_served,
            probe=stages["sanity"]["probe"],
        )

        # 2. Full GSM8K via evalscope
        _record_phase("gsm8k_eval")
        eval_t0 = time.monotonic()
        eval_work_dir = results_dir / "evalscope"
        gen_cfg = dict(eval_stage["generation"])
        result_raw = ppu_gsm8k_eval.run_evalscope(
            unique_served,
            port,
            gsm8k_path,
            generation_config=gen_cfg,
            limit=limit if limit is not None else eval_stage.get("limit"),
            work_dir=eval_work_dir,
            seed=eval_seed,
            eval_batch_size=eval_batch_size,
            dataset_options=dataset_options,
        )
        score, num = ppu_gsm8k_eval.extract_score(result_raw)
        eval_s = round(time.monotonic() - eval_t0, 3)

        per_q_info = _extract_per_question_evalscope(
            eval_work_dir,
            results_dir / "gsm8k_per_question.jsonl",
            expected_count=dataset_spec["num_expected"],
            aggregate_score=score,
            aggregate_num=num,
            dataset=dataset_spec["id"],
            subset=dataset_spec["subset_list"][0],
        )

        golden_file = (
            REPO_ROOT
            / "tests"
            / "e2e"
            / "models"
            / "_golden"
            / accuracy_gate.golden_filename(resolved["golden_model_key"], "890p")
        )
        golden = accuracy_gate.read_golden(golden_file)
        ok_id, id_err = accuracy_gate.check_golden_identity(
            golden, chip="890p", checkpoint_tail=resolved["hf_model_id"]
        )
        ok_cfg, cfg_err = accuracy_gate.check_generation_config(golden, gen_cfg)
        ok_acc, acc_err = accuracy_gate.check_accuracy_golden(score, num, golden)
        ok_dataset = all(
            golden["dataset"].get(key) == dataset_spec[key]
            for key in (
                "subset_list",
                "few_shot_num",
                "few_shot_random",
                "num_expected",
            )
        )
        acc_ok = ok_id and ok_cfg and ok_acc and ok_dataset
        acc_error = (
            id_err
            or cfg_err
            or acc_err
            or (
                None
                if ok_dataset
                else "Dataset sampling contract does not match golden"
            )
        )

        # 3. Drain & clear prefix cache on the SAME server before bench
        reset_before_bench = _drain_and_reset_cache(port)

        # 4. Micro-perf bench (C=1, C=16) marked sanity_throughput_guard
        bench_stage = stages["bench"]
        bench_input = bench_stage["input"]
        _record_phase("micro_e2e")
        bench_dir = results_dir / "bench"
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_t0 = time.monotonic()
        bench_results: dict[str, Any] = {}
        trust_remote = "--trust-remote-code" in resolved["flags"]
        warmup_map = bench_stage["warmup_count_by_concurrency"]
        req_map = bench_stage["request_count_by_concurrency"]
        for conc in bench_stage["concurrencies"]:
            warmup_raw = warmup_map.get(conc, warmup_map.get(str(conc)))
            req_raw = req_map.get(conc, req_map.get(str(conc)))
            if warmup_raw is None or req_raw is None:
                raise KeyError(f"missing warmup/request count for concurrency {conc!r}")
            warmup_n = int(warmup_raw)
            req_n = int(req_raw)
            bench_kwargs = dict(
                served_name=unique_served,
                model_path=resolved["resolved_model"],
                port=port,
                concurrency=int(conc),
                input_len=int(bench_input["input_length"]),
                output_len=int(bench_input["output_length"]),
                range_ratio=float(bench_input.get("range_ratio", 0.0)),
                trust_remote_code=trust_remote,
                result_dir=bench_dir,
            )
            raw_warmup = _run_vllm_bench(
                **bench_kwargs,
                num_prompts=warmup_n,
                result_filename=f"warmup_c{conc}.json",
            )
            _drain_and_reset_cache(port)
            raw_bench = None
            failed_phase = "warmup"
            if _bench_completed(raw_warmup, warmup_n):
                raw_bench = _run_vllm_bench(
                    **bench_kwargs,
                    num_prompts=req_n,
                    result_filename=f"bench_c{conc}.json",
                )
                _drain_and_reset_cache(port)
                failed_phase = None if _bench_completed(raw_bench, req_n) else "measure"
            bench_results[f"c{conc}"] = {
                "concurrency": int(conc),
                "warmup_prompts": warmup_n,
                "num_prompts": req_n,
                "cache_mode": bench_stage.get("cache_mode", "sanity_throughput_guard"),
                "status": "fail" if failed_phase else "pass",
                "failed_phase": failed_phase,
                "warmup_metrics": raw_warmup,
                "metrics": raw_bench,
            }
        bench_s = round(time.monotonic() - bench_t0, 3)
        bench_ok = bool(bench_results) and all(
            result["status"] == "pass" for result in bench_results.values()
        )

        golden_score = float(golden["score"])
        minimum_score, maximum_score = accuracy_gate.golden_score_bounds(golden)
        summary = {
            "tier": "model_sanity",
            "model_id": policy["model_id"],
            "variant": resolved["variant"],
            "hf_model_id": resolved["hf_model_id"],
            "golden_model_key": resolved["golden_model_key"],
            "preset": preset,
            "status": "pass" if acc_ok and bench_ok else "fail",
            "parser_checked": val_report["parser_checked"],
            "config_digest": resolved["config_digest"],
            "dataset": {
                "id": dataset_spec["id"],
                "version": dataset_spec["version"],
                **dataset_options,
            },
            "eval_seed": eval_seed,
            "eval_batch_size": eval_batch_size,
            "timings": {
                "queue_s": round(float(queue_s), 3),
                "cold_start_s": cold_start_s,
                "eval_s": eval_s,
                "bench_s": bench_s,
            },
            "sanity_probe": probe_res,
            "accuracy": {
                "score": score,
                "num": num,
                "expected_num": golden["dataset"].get("num_expected"),
                "golden_score": golden_score,
                "minimum_score": minimum_score,
                "maximum_score": maximum_score,
                "delta": round(score - golden_score, 6),
                "within_band": acc_ok,
                "error": acc_error,
                "per_question": per_q_info,
            },
            "cache_reset_before_bench": {
                **reset_before_bench,
                "prior_workload_ids": ["sanity", "eval-gsm8k"],
            },
            "micro_e2e_passed": bench_ok,
            "bench": bench_results,
        }
        (results_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        _stop_server(proc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run PPU recipe fast_sanity / model_sanity"
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--tier", choices=["fast_sanity", "model_sanity"], required=True
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--assets-map", type=Path, default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--port", type=int, default=9985)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument(
        "--layer-mode", choices=["auto", "2l", "full", "both"], default="auto"
    )
    parser.add_argument(
        "--vllm-parser", choices=["auto", "required", "skip"], default="required"
    )
    parser.add_argument("--queue-s", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--post-ready-sleep", type=int, default=30)
    args = parser.parse_args(argv)

    if args.tier == "fast_sanity":
        res = run_fast_sanity(
            recipe_path=args.recipe,
            policy_path=args.policy,
            results_dir=args.results_dir,
            assets_map=args.assets_map,
            variant=args.variant,
            preset=args.preset,
            port=args.port,
            ready_timeout=args.ready_timeout,
            layer_mode=args.layer_mode,
            parser_mode=args.vllm_parser,
            queue_s=args.queue_s,
        )
    else:
        res = run_model_sanity(
            recipe_path=args.recipe,
            policy_path=args.policy,
            results_dir=args.results_dir,
            assets_map=args.assets_map,
            variant=args.variant,
            preset=args.preset,
            port=args.port,
            ready_timeout=args.ready_timeout,
            parser_mode=args.vllm_parser,
            queue_s=args.queue_s,
            limit=args.limit,
            post_ready_sleep=args.post_ready_sleep,
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
