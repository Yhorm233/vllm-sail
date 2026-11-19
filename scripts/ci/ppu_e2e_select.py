#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build smoke and accuracy matrices from the checked-in model catalog.

Catalog entries are partitioned by ``category``. Missing categories retain the
legacy ``smoke`` meaning. A requested subset is validated against the complete
catalog, then each output contains only entries belonging to its own category.

Usage:
    ppu_e2e_select.py [MODELS]

``MODELS`` is a comma-separated list of catalog keys. An empty value selects all
models in each category. Unknown keys are rejected so a typo fails fast. The
script emits ``matrix=<json>`` for smoke and ``accuracy_matrix=<json>`` for the
GSM8K accuracy workflow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CATALOG = Path(__file__).resolve().parent / "ppu_e2e_models.json"


def _category(entry: dict) -> str:
    return entry.get("category", "smoke")


def _select(catalog: list[dict], models: str, category: str) -> list[dict]:
    by_key = {entry["key"]: entry for entry in catalog}
    requested = [item.strip() for item in models.split(",") if item.strip()]
    unknown = [key for key in requested if key not in by_key]
    if unknown:
        available = ", ".join(by_key)
        raise ValueError(
            f"unknown model key(s): {', '.join(unknown)}; available: {available}"
        )
    entries = [by_key[key] for key in requested] if requested else list(catalog)
    return [entry for entry in entries if _category(entry) == category]


def build_matrix(catalog: list[dict], models: str) -> dict:
    """Return smoke entries plus the aliases default-branch callers read."""
    include = []
    for entry in _select(catalog, models, "smoke"):
        selected = dict(entry)
        # Default-branch workflow_dispatch callers may still read `checkpoint`.
        # Its value is a model ID, never a runner-specific filesystem path.
        if "model_id" in selected:
            selected.setdefault("checkpoint", selected["model_id"])
        # A PR run takes its workflow files from the default branch and its
        # catalog from the PR head, so the two disagree on the card count's
        # name until the rename reaches the default branch. Answering both
        # keeps a PR from failing on its own catalog; the alias can go once
        # every caller reads `ppu_count`.
        if "ppu_count" in selected:
            selected.setdefault("tp", selected["ppu_count"])
        include.append(selected)
    return {"include": include}


def build_accuracy_matrix(catalog: list[dict], models: str) -> dict:
    """Return the evaluator matrix for accuracy catalog entries only.

    Two identifiers travel with every entry:

    * ``case_id`` — the catalog ``key``. Several cases may point at the same
      checkpoint with different serve arguments; this is what the job name,
      the artifact and the serve log are labelled with.
    * ``model_key`` — the evaluator's model-config key, which also keys the
      golden file. Cases sharing a checkpoint share this value and therefore
      compare against one golden.
    """
    include = []
    for entry in _select(catalog, models, "accuracy"):
        case_id = entry["key"]
        model_key = entry.get("model_key")
        if not model_key:
            raise ValueError(
                f"accuracy catalog entry {case_id!r} is missing 'model_key'"
            )
        include.append(
            {
                "case_id": case_id,
                "model_key": model_key,
                # The container wrapper resolves this id against the offline HF
                # cache to mount the weights; the evaluator keys the rest of
                # its configuration off model_key instead.
                "model_id": entry["model_id"],
                "board": entry["board"],
                "ppu_count": int(entry["ppu_count"]),
                # Empty (or absent) means "use the evaluator's built-in argv".
                "server_cmd": entry.get("server_cmd") or [],
                # Extra environment for the server process only; merged over
                # the inherited environment by the evaluator.
                "server_env": entry.get("server_env") or {},
            }
        )
    return {"include": include}


def build_recipe_matrix(models: str, tier: str = "fast_sanity") -> dict:
    """Build a matrix from ``tests/ppu/policy/models/*.yaml`` using ``ppu_recipe``."""
    repo_root = Path(__file__).resolve().parents[2]
    ci_dir = Path(__file__).resolve().parent
    if str(ci_dir) not in sys.path:
        sys.path.insert(0, str(ci_dir))
    import ppu_recipe

    policy_dir = repo_root / "tests" / "ppu" / "policy" / "models"
    policy_files = sorted(policy_dir.glob("*.yaml"))
    by_slug = {path.stem: path for path in policy_files}
    requested = [item.strip() for item in models.split(",") if item.strip()]
    unknown = [key for key in requested if key not in by_slug]
    if unknown:
        available = ", ".join(by_slug)
        raise ValueError(
            f"unknown recipe policy key(s): {', '.join(unknown)}; available: {available}"
        )
    selected_keys = requested if requested else list(by_slug)
    include = []
    for slug in selected_keys:
        pol_path = by_slug[slug]
        pol_doc = ppu_recipe._load_yaml(pol_path)
        ppu_recipe.validate_policy_structure(
            pol_doc, context=str(pol_path), base_dir=repo_root
        )
        rec_rel = Path(pol_doc["recipe"])
        rec_path = rec_rel if rec_rel.is_absolute() else (repo_root / rec_rel)
        resolved = ppu_recipe.resolve_recipe(rec_path, policy_path=pol_path, tier=tier)
        include.append(
            {
                "key": slug,
                "policy_path": pol_path.relative_to(repo_root).as_posix(),
                "recipe_path": rec_path.relative_to(repo_root).as_posix(),
                "model_id": resolved["hf_model_id"],
                "golden_model_key": resolved["golden_model_key"],
                "ppu_count": int(resolved["ppu_count"]),
                "config_digest": resolved["config_digest"],
            }
        )
    return {"include": include}


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--recipes":
        models = argv[1] if len(argv) > 1 else ""
        tier = argv[2] if len(argv) > 2 else "fast_sanity"
        recipe_matrix = build_recipe_matrix(models, tier=tier)
        print(f"recipe_matrix={json.dumps(recipe_matrix, separators=(',', ':'))}")
        return 0
    models = argv[0] if argv else ""
    catalog = json.loads(CATALOG.read_text())
    matrix = build_matrix(catalog, models)
    accuracy_matrix = build_accuracy_matrix(catalog, models)
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
    print(f"accuracy_matrix={json.dumps(accuracy_matrix, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
