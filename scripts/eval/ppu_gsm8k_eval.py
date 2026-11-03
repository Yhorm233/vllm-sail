#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PPU GSM8K accuracy E2E test via evalscope (server-client mode).

Single-file, standalone runnable::

    python scripts/eval/ppu_gsm8k_eval.py \\
        --model-key deepseek_v4_mxfp4 [--update-golden] [--limit 20]

Steps
-----
1. Parse arguments
2. Resolve model_id + chip → TP + guards
3. Start ``vllm serve`` (model resolved via HF_HOME + HF_HUB_OFFLINE)
4. Wait for readiness (poll ``/v1/models`` + scan fatal regex)
5. Run evalscope ``TaskConfig(eval_type='openai_api', datasets=['gsm8k'])``
6. Extract accuracy score (version-tolerant)
7. Compare golden / write pending / stop server
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

# --- Project root for cross-module imports ---------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.e2e.models._accuracy_gate import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_TOLERANCE,
    GSM8K_FULL_SIZE,
    build_accuracy_golden,
    check_accuracy_golden,
    check_golden_identity,
    check_score_threshold,
    golden_filename,
    pending_golden_filename,
    read_golden,
    write_golden,
)
from tests.e2e.models._offline_runner import (  # noqa: E402
    CAPABILITY_TO_CHIP,
    DEFAULT_CHIP,
    DEFAULT_DTYPE,
    DEFAULT_ENFORCE_EAGER,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_SEED,
    REPORTS_ROOT,
    _checkpoint_tail,
    tensor_parallel_for_chip,
)

# --- Server readiness / fatal detection ------------------------------------
SERVER_READY_TIMEOUT = 1800
SERVER_POST_READY_SLEEP = 30
FATAL_RE = re.compile(
    r"Traceback \(most recent"
    r"|EngineCore failed"
    r"|Engine core initialization failed"
    r"|core dumped"
    r"|Failed to infer device type"
    r"|Error: model architectures"
)

# --- GSM8K dataset path (from environment) --------------------------------
GSM8K_DATASET_PATH = os.environ.get("GSM8K_DATASET_PATH", "")
if not GSM8K_DATASET_PATH:
    raise OSError(
        "GSM8K_DATASET_PATH environment variable is required but not set. "
        "Set it to the local path of the GSM8K dataset "
        "(e.g. /path/to/EvalScope/data/gsm8k/v2/gsm8k)."
    )

# --- Golden / reports directories ------------------------------------------
GOLDEN_DIR = _PROJECT_ROOT / "tests" / "e2e" / "models" / "_golden"

# --- Model configurations --------------------------------------------------
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "deepseek_v4_int8": {
        "model_id": (
            "T-HEAD/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken"
        ),
        "served_name": "deepseek-v4-int8",
        "extra_flags": ["--trust-remote-code"],
        "chip": "810e",
        # o_groups=8 in the model config requires TP <= 8; the
        # upstream FlashMLA _o_proj computes
        #   heads_per_group = n_local_heads // n_local_groups
        # which divides by zero when TP > o_groups because
        #   n_local_groups = o_groups // tp_size = 8 // 16 = 0.
        "tp": 8,
    },
    "deepseek_v4_mxfp4": {
        "model_id": (
            "T-HEAD/DeepSeek-V4-Flash-0731-MoE-"
            "Quant-W-MXFP4-A-MXFP4-Dense-"
            "Quant-W-FP8-PerChannel-A-FP8-PerToken"
        ),
        "served_name": "deepseek-v4-mxfp4",
        "extra_flags": ["--trust-remote-code"],
        "chip": "890p",
    },
    "minimax-m3-fp8": {
        "model_id": ("T-HEAD/MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken"),
        "served_name": "MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken",
        "extra_flags": ["--trust-remote-code"],
    },
    "qwen3.8-flash-next": {
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "served_name": "Qwen3.8-Flash-Next",
        "extra_flags": ["--trust-remote-code"],
    },
}


# ===================================================================
# GSM8K dataset resolution
# ===================================================================


def resolve_gsm8k_dataset() -> str:
    """Return the GSM8K dataset path from ``GSM8K_DATASET_PATH``."""
    return GSM8K_DATASET_PATH


# ===================================================================
# Server lifecycle
# ===================================================================


def start_vllm_server(
    model: str,
    *,
    served_model_name: str,
    port: int = 9985,
    tp: int = 8,
    extra_flags: list[str] | None = None,
    log_path: Path | None = None,
) -> tuple[subprocess.Popen[bytes], Path]:
    """Start ``vllm serve`` in a new session; return (proc, log)."""
    if log_path is None:
        log_path = Path("/tmp/vllm_serve_gsm8k.log")
    cmd: list[str] = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        str(DEFAULT_GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(DEFAULT_MAX_MODEL_LEN),
        "--enforce-eager",
        "--dtype",
        DEFAULT_DTYPE,
        "--trust-remote-code",
        "--seed",
        str(DEFAULT_SEED),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    log_fh = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    return proc, log_path


def unique_served_model_name(base: str) -> str:
    """Return a run-unique served model name derived from ``base``.

    The suffix is drawn from :func:`secrets.token_hex` so a lingering vLLM
    process from a previous run cannot forge the identity even if it keeps
    answering on the same port with the same catalog name.
    """
    return f"{base}-run-{secrets.token_hex(4)}"


def _probe_models_endpoint(url: str, expected_served_name: str) -> bool:
    """Return True iff ``/v1/models`` lists ``expected_served_name`` as ``id``.

    HTTP 200 alone is not sufficient: a stale vLLM instance left running on
    the same port -- even one serving the same catalog base name -- will
    happily answer 200, so readiness is bound to the per-run nonce by
    inspecting the JSON ``data`` list for an exact ``id`` match.
    Malformed JSON, a missing ``data`` field, or any other mismatch returns
    False so the caller keeps waiting instead of mis-attributing readiness.
    """
    try:
        with urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                return False
            raw = resp.read()
    except (URLError, OSError):
        return False
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if isinstance(item, dict) and item.get("id") == expected_served_name:
            return True
    return False


def wait_server_ready(
    proc: subprocess.Popen[bytes],
    log_path: Path,
    *,
    expected_served_name: str,
    port: int = 9985,
    timeout: int = SERVER_READY_TIMEOUT,
) -> None:
    """Poll ``/v1/models`` + scan the log for fatal patterns.

    Readiness is bound to the launched process *and* its unique served model
    name: the loop keeps waiting until ``/v1/models`` returns 200 **and** the
    JSON ``data[*].id`` list contains ``expected_served_name``.  A stale
    service on the same port that still returns 200 (or returns the catalog
    base name only) cannot pass this gate.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _log_tail(log_path)
            raise RuntimeError(
                "vllm serve exited with code "
                f"{proc.returncode} before becoming ready."
                f"\nLog tail:\n{tail}"
            )
        try:
            log_text = log_path.read_text()
            if FATAL_RE.search(log_text):
                raise RuntimeError(
                    f"fatal error detected in vllm serve log:\n{log_text[-2000:]}"
                )
        except OSError:
            pass
        if _probe_models_endpoint(url, expected_served_name):
            return
        time.sleep(5)
    raise RuntimeError(
        f"server not ready within {timeout}s (expected served model "
        f"{expected_served_name!r}).\nLog tail:\n{_log_tail(log_path)}"
    )


def _ensure_launched_server_alive(
    proc: subprocess.Popen[bytes],
    log_path: Path,
    phase: str,
) -> None:
    """Fail loudly if the launched vLLM process has exited.

    Called after readiness and the warm-up sleep, immediately before the
    first evalscope request, so a crash during the post-ready window is
    turned into an explicit error instead of silently falling through to
    whatever lingering service happens to answer the port.
    """
    if proc.poll() is None:
        return
    tail = _log_tail(log_path)
    raise RuntimeError(
        f"vllm serve exited with code {proc.returncode} "
        f"after readiness ({phase}); refusing to evaluate."
        f"\nLog tail:\n{tail}"
    )


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    """Gracefully stop the vLLM server and all workers in its process group."""
    # start_new_session=True makes the server PID its process-group ID.
    # Workers may outlive the leader, including after proc.poll() reaps it.
    pgid = proc.pid

    # Always try SIGTERM on the process group first
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass  # Group already gone

    # Wait for graceful shutdown
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass

    # SIGKILL any survivors
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass  # Group already gone

    # Final reap
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _log_tail(path: Path, chars: int = 2000) -> str:
    try:
        text = path.read_text()
        return text[-chars:]
    except Exception:
        return "<unreadable>"


# ===================================================================
# Evalscope runner
# ===================================================================


def run_evalscope(
    served_name: str,
    port: int,
    gsm8k_path: str,
    *,
    limit: int | None = None,
    work_dir: Path | None = None,
) -> Any:
    """Run evalscope GSM8K (runtime import)."""
    from evalscope import TaskConfig, run_task  # type: ignore[import-untyped]

    cfg = TaskConfig(
        model=served_name,
        api_url=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=["gsm8k"],
        dataset_args={
            "gsm8k": {
                "dataset_id": gsm8k_path,
                "few_shot_num": 4,
                "few_shot_random": False,
                "subset_list": ["main"],
            }
        },
        generation_config={
            "temperature": 0,
            "max_tokens": 1024,
        },
        eval_batch_size=64,
        limit=limit,
        seed=42,
        work_dir=str(work_dir) if work_dir else None,
    )
    return run_task(task_cfg=cfg)


def extract_score(result: Any) -> tuple[float, int]:
    """Version-tolerant extraction of (accuracy, num_samples)."""
    rep = result["gsm8k"]
    score = getattr(rep, "score", None)
    if score is None:
        score = rep.metrics[0].score
    num = getattr(rep, "num", None)
    if num is None:
        num = rep.metrics[0].num
    return float(score), int(num)


# ===================================================================
# Main pipeline
# ===================================================================


def run_accuracy_eval(  # noqa: C901
    model_key: str,
    *,
    model_override: str | None = None,
    chip: str = "auto",
    update_golden: bool = False,
    limit: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    work_dir: Path | None = None,
    port: int = 9985,
    served_model_name: str | None = None,
    extra_serve_flags: list[str] | None = None,
    golden_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full GSM8K accuracy eval pipeline.

    Returns a dict with keys: ``status``, ``score``, ``num``,
    ``golden_path``, ``error``, ``record``.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"unknown model key {model_key!r}; available: {sorted(MODEL_CONFIGS)}"
        )
    model_cfg = MODEL_CONFIGS[model_key]

    # -- resolve model ------------------------------------------------
    model_id = model_override or model_cfg["model_id"]

    # -- resolve chip → TP --------------------------------------------
    if chip == "auto":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device available")
        cap = torch.cuda.get_device_capability(0)
        chip = CAPABILITY_TO_CHIP.get((cap[0], cap[1]), DEFAULT_CHIP)
    # Per-model TP override (e.g. DeepSeek-V4 o_groups=8 caps at TP=8).
    tp = model_cfg.get("tp") or tensor_parallel_for_chip(chip)

    # -- guard: enough GPUs? ------------------------------------------
    import torch

    if torch.cuda.device_count() < tp:
        raise RuntimeError(
            f"need {tp} GPUs for chip {chip!r} but only "
            f"{torch.cuda.device_count()} available"
        )

    # -- resolve GSM8K dataset ----------------------------------------
    gsm8k_path = resolve_gsm8k_dataset()

    # -- paths --------------------------------------------------------
    golden_root = golden_dir or GOLDEN_DIR
    gf = golden_root / golden_filename(model_key, chip)
    pending_gf = golden_root / pending_golden_filename(model_key, chip)
    wd = Path(work_dir) if work_dir else REPORTS_ROOT
    wd.mkdir(parents=True, exist_ok=True)

    served_base = served_model_name or model_cfg["served_name"]
    # Bind this run to a unique served-model identity so readiness and the
    # eval client both reject any stale vLLM instance that lingers on the
    # same port with a stable catalog name.  The golden identity below is
    # keyed by chip + checkpoint_tail + benchmark, so this per-run suffix
    # never leaks into the persisted golden contract.
    unique_served = unique_served_model_name(served_base)
    flags = list(extra_serve_flags or model_cfg.get("extra_flags", []))
    log_path = wd / f"vllm_serve_{model_key}.log"

    # -- start server -------------------------------------------------
    server, log_path = start_vllm_server(
        model_id,
        served_model_name=unique_served,
        port=port,
        tp=tp,
        extra_flags=flags,
        log_path=log_path,
    )
    try:
        wait_server_ready(
            server,
            log_path,
            port=port,
            expected_served_name=unique_served,
        )
        time.sleep(SERVER_POST_READY_SLEEP)
        # A readiness probe only proves the server was alive at the time it
        # answered; re-check right before the first eval request so a crash
        # during the warm-up window fails the run instead of silently
        # routing traffic to whatever still answers the port.
        _ensure_launched_server_alive(server, log_path, phase="post-readiness warm-up")

        # -- run evalscope --------------------------------------------
        result_raw = run_evalscope(
            unique_served,
            port,
            gsm8k_path,
            limit=limit,
            work_dir=wd / "evalscope",
        )
        score, num = extract_score(result_raw)

        # -- evalscope version ----------------------------------------
        try:
            import evalscope as _es  # type: ignore[import-untyped]

            es_ver = getattr(_es, "__version__", "unknown")
        except ImportError:
            es_ver = "unknown"

        # -- build golden record --------------------------------------
        ckpt_tail = _checkpoint_tail(model_id)
        ds_cfg: dict[str, Any] = {
            "dataset_id": gsm8k_path,
            "few_shot_num": 4,
            "few_shot_random": False,
            "subset_list": ["main"],
            "num_expected": GSM8K_FULL_SIZE,
        }
        srv_cfg: dict[str, Any] = {
            "tensor_parallel_size": tp,
            "max_model_len": DEFAULT_MAX_MODEL_LEN,
            "dtype": DEFAULT_DTYPE,
            "gpu_memory_utilization": (DEFAULT_GPU_MEMORY_UTILIZATION),
            "enforce_eager": DEFAULT_ENFORCE_EAGER,
            "seed": DEFAULT_SEED,
        }
        tol = dict(DEFAULT_TOLERANCE)
        tol["hard_floor"] = threshold
        record = build_accuracy_golden(
            model_key=model_key,
            chip=chip,
            checkpoint=model_id,
            evalscope_version=es_ver,
            dataset_cfg=ds_cfg,
            serving_cfg=srv_cfg,
            score=score,
            score_num=num,
            tolerance=tol,
        )

        # -- golden comparison ----------------------------------------
        return _golden_compare(
            record=record,
            score=score,
            num=num,
            chip=chip,
            ckpt_tail=ckpt_tail,
            gf=gf,
            pending_gf=pending_gf,
            golden_root=golden_root,
            update_golden=update_golden,
            model_key=model_key,
            threshold=threshold,
        )
    finally:
        stop_server(server)


def _golden_compare(
    *,
    record: dict[str, Any],
    score: float,
    num: int,
    chip: str,
    ckpt_tail: str,
    gf: Path,
    pending_gf: Path,
    golden_root: Path,
    update_golden: bool,
    model_key: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Perform threshold check + optional golden comparison."""
    status: str
    error_msg: str | None = None

    # --- Completeness always precedes creation, update, or comparison.
    expected_num = record["dataset"]["num_expected"]
    if num != expected_num:
        return {
            "status": "fail_completeness",
            "score": score,
            "num": num,
            "golden_path": str(gf),
            "error": (
                f"completeness gate failed: evaluated {num} samples, "
                f"expected {expected_num}"
            ),
            "record": record,
        }

    # --- Primary gate: score >= threshold --------------------------
    ok_thr, thr_err = check_score_threshold(score, threshold)

    if update_golden:
        if not ok_thr:
            status = "fail_threshold"
            error_msg = thr_err
        else:
            write_golden(record, gf)
            if pending_gf.exists():
                pending_gf.unlink()
            status = "updated"
    elif gf.exists():
        golden = read_golden(gf)
        ok_id, id_err = check_golden_identity(
            golden, chip=chip, checkpoint_tail=ckpt_tail
        )
        if not ok_id:
            status = "fail_identity"
            error_msg = id_err
        elif not ok_thr:
            status = "fail_threshold"
            error_msg = thr_err
        else:
            ok_g, acc_err = check_accuracy_golden(score, num, golden)
            status = "pass" if ok_g else "fail_accuracy"
            error_msg = acc_err
    elif not ok_thr:
        # No golden yet; threshold is the only gate.
        write_golden(record, pending_gf)
        status = "fail_threshold"
        error_msg = thr_err
    else:
        # Passes threshold but no golden — write pending.
        write_golden(record, pending_gf)
        status = "created"
        warnings.warn(
            f"no golden for {model_key}/{chip}; "
            f"wrote pending to {pending_gf.name}. "
            f"score={score:.4f} passes threshold "
            f"{threshold}.",
            stacklevel=3,
        )

    return {
        "status": status,
        "score": score,
        "num": num,
        "golden_path": str(gf),
        "error": error_msg,
        "record": record,
    }


# ===================================================================
# CLI entry point
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="PPU GSM8K accuracy E2E test")
    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_CONFIGS),
        help="Model configuration key",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default HF model_id or local path",
    )
    parser.add_argument(
        "--chip",
        default="auto",
        choices=["auto", "810e", "890p"],
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Rewrite the formal golden baseline",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Score >= threshold to pass (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation samples",
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--golden-dir",
        default=None,
        help="Writable directory for formal or pending golden records",
    )
    parser.add_argument("--port", type=int, default=9985)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--extra-serve-flags", default=None)
    args = parser.parse_args()

    extra = args.extra_serve_flags.split() if args.extra_serve_flags else None
    wd = Path(args.work_dir) if args.work_dir else None

    result = run_accuracy_eval(
        args.model_key,
        model_override=args.model,
        chip=args.chip,
        update_golden=args.update_golden,
        limit=args.limit,
        threshold=args.threshold,
        work_dir=wd,
        port=args.port,
        served_model_name=args.served_model_name,
        extra_serve_flags=extra,
        golden_dir=Path(args.golden_dir) if args.golden_dir else None,
    )

    out = {k: v for k, v in result.items() if k != "record"}
    print(json.dumps(out, indent=2))
    if result["status"].startswith("fail"):
        sys.exit(1)


if __name__ == "__main__":
    main()
