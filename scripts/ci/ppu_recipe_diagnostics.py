import json
import os
import re
import stat
from pathlib import Path

LOG_TAIL_LIMIT = 256 * 1024
SUMMARY_LIMIT = 1024 * 1024
PHASES = {
    "wheel_verification": "wheel_verification_failed",
    "runtime_dependencies": "runtime_dependencies_failed",
    "platform_check": "platform_check_failed",
    "model_cache_check": "model_cache_check_failed",
    "recipe_validation": "recipe_validation_failed",
    "server_startup": "server_startup_failed",
    "server_readiness_timeout": "server_readiness_timeout",
    "chat_probe": "chat_probe_failed",
    "gsm8k_eval": "gsm8k_eval_failed",
    "micro_e2e": "micro_e2e_failed",
}


def _read_file(path: Path, limit: int, *, tail: bool = False) -> str:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or (not tail and info.st_size > limit):
            return ""
        with path.open("rb") as stream:
            if tail:
                stream.seek(max(0, info.st_size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def failure_reason(source: Path, tier: str) -> str:
    phase = _read_file(source / "recipe-phase.txt", 64).strip()
    results = source / "results"
    try:
        has_results = stat.S_ISDIR(results.lstat().st_mode)
    except OSError:
        has_results = False
    if has_results and tier == "model_sanity":
        try:
            summary = json.loads(_read_file(results / "summary.json", SUMMARY_LIMIT))
            if isinstance(summary, dict):
                accuracy = summary.get("accuracy")
                if isinstance(accuracy, dict) and accuracy.get("within_band") is False:
                    return "gsm8k_gate_failed"
                if summary.get("micro_e2e_passed") is False:
                    return "micro_e2e_failed"
        except (ValueError, RecursionError):
            pass
    if phase == "server_startup":
        if has_results:
            for label in (
                ("model_sanity",)
                if tier == "model_sanity"
                else ("fast_full", "fast_2l")
            ):
                server = _read_file(
                    results / f"vllm_serve_{label}.log", LOG_TAIL_LIMIT, tail=True
                )
                if not server:
                    continue
                if (
                    "mla_indexer.py" in server
                    and re.search(r"schedul(?:e|er)_metadata", server)
                    and re.search(
                        r"(?:tensor a \(2\).*tensor b \(4\)|tensor a \(4\).*tensor b \(2\)"
                        r"|expanded size of the tensor \(2\).*existing size \(4\))"
                        r" at non-singleton dimension 1",
                        server,
                    )
                ):
                    return "mla_metadata_shape_mismatch"
                break
    if phase in PHASES:
        return PHASES[phase]
    runtime = _read_file(source / "runtime.log", 1, tail=True)
    return "unknown" if runtime else "runtime_setup_failed"


if __name__ == "__main__":
    print(failure_reason(Path(os.environ["PRIVATE_DIR"]), os.environ["TIER"]))
