import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

RESULT_FILES = ("results/summary.json", "provenance.json")
OPTIONAL_FILES = (
    "runtime.log",
    "recipe-phase.txt",
    "results/gsm8k_per_question.jsonl",
    "results/vllm_serve_fast_2l.log",
    "results/vllm_serve_fast_full.log",
    "results/vllm_serve_model_sanity.log",
)


def _check_dir_mode(directory: Path, *allowed_modes: int) -> None:
    mode = directory.lstat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) not in allowed_modes:
        raise ValueError("unexpected retention directory type or mode")


def retain(environ) -> None:
    index = {
        "checkout_sha": environ["PPU_CI_SOURCE_SHA"],
        "tier": environ["TIER"],
        "model_key": environ["MODEL_KEY"],
        "run_id": environ["GITHUB_RUN_ID"],
        "attempt": environ["GITHUB_RUN_ATTEMPT"],
    }
    if (
        index["tier"] not in {"fast_sanity", "model_sanity"}
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", index["model_key"])
        or not re.fullmatch(r"[0-9a-f]{40}", index["checkout_sha"])
        or not all(re.fullmatch(r"[0-9]+", index[key]) for key in ("run_id", "attempt"))
    ):
        raise ValueError("invalid retention metadata")

    root = Path(environ["PPU_CI_PRIVATE_RESULTS_ROOT"])
    if not root.is_absolute():
        raise ValueError("expected an absolute retention root")
    _check_dir_mode(root, 0o700, 0o755)
    root = root.resolve(strict=True)
    for key in ("RUNNER_TEMP", "GITHUB_WORKSPACE"):
        ephemeral = Path(environ[key]).resolve()
        if ephemeral == root or ephemeral in root.parents:
            raise ValueError("retention root must not be job-local")

    parent = root
    for component in (
        "vllm-sail",
        f"{index['run_id']}-{index['attempt']}",
        index["tier"],
    ):
        parent /= component
        try:
            _check_dir_mode(parent, 0o700)
        except FileNotFoundError:
            parent.mkdir(mode=0o700, exist_ok=True)
            _check_dir_mode(parent, 0o700)

    destination = parent / index["model_key"]
    if os.path.lexists(destination):
        raise FileExistsError("retention destination already exists")

    source_dir = Path(environ["PPU_CI_ARTIFACT_DIR"])
    partial = environ.get("RECIPE_OUTCOME") in ("failure", "cancelled")
    try:
        if not stat.S_ISDIR((source_dir / "results").lstat().st_mode):
            raise ValueError("expected a recipe results directory")
    except FileNotFoundError:
        if not partial:
            raise
    required = {"launcher.log"}
    if not partial:
        required.update(RESULT_FILES)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as staging_dir:
        staging = Path(staging_dir)
        published = staging / index["model_key"]
        published.mkdir(mode=0o700)
        files = {}
        for relative_path in ("launcher.log", *RESULT_FILES, *OPTIONAL_FILES):
            source = source_dir / relative_path
            name = source.name
            try:
                mode = source.lstat().st_mode
            except FileNotFoundError:
                if relative_path not in required:
                    continue
                raise
            if not stat.S_ISREG(mode):
                raise ValueError("expected a regular recipe output")
            copied = published / name
            shutil.copyfile(source, copied)
            copied.chmod(0o600)
            digest = hashlib.sha256()
            with copied.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[name] = digest.hexdigest()
        index_path = published / "index.json"
        index_path.write_text(
            json.dumps({**index, "files": files}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_path.chmod(0o600)
        published.rename(destination)


def main() -> int:
    os.umask(0o077)
    status = "not_configured"
    exit_code = 0
    if os.environ.get("PPU_CI_PRIVATE_RESULTS_ROOT"):
        try:
            retain(os.environ)
        except (OSError, ValueError, KeyError, RuntimeError):
            status = "failed"
            exit_code = 1
        else:
            status = "retained"
    else:
        print("::warning::private results retention not configured")

    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as outputs:
        outputs.write(f"retention={status}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
