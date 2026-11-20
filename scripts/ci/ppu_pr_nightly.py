# SPDX-License-Identifier: Apache-2.0
"""Authorize and report one Nightly accuracy case for an immutable PR head."""

import json
import math
import os
import re
import subprocess
import sys
import urllib.error
from pathlib import Path

from scripts.ci import ppu_e2e_select, ppu_recipe_diagnostics
from scripts.ci.ppu_ci_command import Controller, Rejected, marker, parse_command
from scripts.ci.ppu_ci_command import main as ci_main

COMMAND = "/ci run nightly"
WORKFLOW = "ppu-pr-nightly.yaml"
CASE = r"[a-z0-9][a-z0-9.-]{0,63}"
USAGE = (
    f"Use `{COMMAND} <case>` with one Recipe policy slug or legacy accuracy case ID."
)
REASONS = {
    *ppu_recipe_diagnostics.PHASES.values(),
    "completeness_failed",
    "eval_incomplete",
    "golden_config_mismatch",
    "golden_identity_mismatch",
    "gsm8k_gate_failed",
    "mla_metadata_shape_mismatch",
    "runtime_setup_failed",
    "server_not_ready",
    "unknown",
}


def valid_case(case):
    return case != "all" and re.fullmatch(CASE, case) is not None


def nightly_command(event):
    try:
        return parse_command(event)
    except Rejected:
        body = event["comment"]["body"].strip()
        if body == COMMAND or body.startswith(COMMAND + " "):
            return body
        raise


def resolve_case(source, case, recipe_enabled):
    if not valid_case(case):
        raise Rejected(USAGE)
    catalog = json.loads((source / "scripts/ci/ppu_e2e_models.json").read_text())
    catalog_matches = [entry for entry in catalog if entry.get("key") == case]
    if len(catalog_matches) > 1:
        raise Rejected(f"Duplicate catalog key: {case}.")
    legacy = bool(catalog_matches and catalog_matches[0].get("category") == "accuracy")
    policy = source / f"tests/ppu/policy/models/{case}.yaml"
    if policy.is_file():
        if legacy:
            raise Rejected(
                f"Case {case} is ambiguous across Recipe and legacy accuracy."
            )
        if not recipe_enabled:
            raise Rejected("Recipe model sanity is disabled.")
        process = subprocess.run(
            [
                sys.executable,
                str(source / "scripts/ci/ppu_e2e_select.py"),
                "--recipes",
                case,
                "model_sanity",
            ],
            capture_output=True,
            text=True,
            env={"PATH": os.environ["PATH"]},
            timeout=60,
        )
        raw = process.stdout.removeprefix("recipe_matrix=")
        try:
            selected = json.loads(raw)["include"]
        except (ValueError, KeyError, TypeError):
            selected = []
        if process.returncode:
            raise Rejected(f"Recipe policy {case} failed model_sanity selection.")
        if (
            not isinstance(selected, list)
            or len(selected) != 1
            or not isinstance(selected[0], dict)
            or selected[0].get("key") != case
        ):
            raise Rejected(f"Expected one Recipe model sanity case: {case}.")
        return "recipe", {"include": selected}
    if legacy:
        selected = ppu_e2e_select.build_accuracy_matrix(catalog, case)["include"]
        if len(selected) == 1:
            return "legacy", None
    raise Rejected(f"Unknown Nightly accuracy case: {case}. See tests/ppu/README.md.")


def public_result(path):
    try:
        summary = json.loads(ppu_recipe_diagnostics._read_file(path, 16 * 1024))
    except (ValueError, RecursionError):
        return None, None, None, None
    if not isinstance(summary, dict):
        return None, None, None, None
    gsm8k = summary.get("gsm8k")
    if not isinstance(gsm8k, dict):
        gsm8k = {}
    reason = summary.get("failure_reason")
    score = gsm8k.get("score")
    samples = gsm8k.get("evaluated_samples")
    micro = summary.get("micro_e2e")
    return (
        reason if isinstance(reason, str) and reason in REASONS else None,
        score
        if type(score) in (int, float) and 0 <= score <= 1 and math.isfinite(score)
        else None,
        samples if type(samples) is int and 0 < samples <= 100_000 else None,
        micro if micro in ("pass", "fail") else None,
    )


class NightlyController(Controller):
    def command(self):
        super().command(parse=nightly_command)

    def authorize_nightly(self, pr, login):
        if pr["state"] != "open" or pr["base"]["repo"]["full_name"] != self.repo:
            raise Rejected("PR must be open and target this repository.")
        if not self.can_write(login):
            raise Rejected("Repository write permission is required for PR Nightly.")

    def execute(self, action, pr, login, cid):
        if action != COMMAND and not action.startswith(COMMAND + " "):
            return super().execute(action, pr, login, cid)
        if action == COMMAND:
            matrix = f"{self.url}/blob/{self.default_branch}/tests/ppu/README.md"
            return f"{USAGE} [List of available Nightly cases]({matrix})."
        case = action.removeprefix(COMMAND + " ")
        if case == "all":
            raise Rejected(f"Full PR Nightly is not available yet. {USAGE}")
        if not valid_case(case):
            raise Rejected(USAGE)
        self.authorize_nightly(pr, login)
        sha, number = pr["head"]["sha"], pr["number"]
        result = self.api("GET", f"/pulls/{number}")
        if result["head"]["sha"] != sha or result["state"] != "open":
            raise Rejected("PR changed; request Nightly again for the current head.")
        prefix = f"PPU PR #{number} Nightly {case} @ {sha} (comment "
        # GitHub caps filtered run listings at 1,000 results. Filter the
        # workflow's unfiltered pages locally so older active runs are visible.
        for run in self.pages(f"/actions/workflows/{WORKFLOW}/runs", "workflow_runs"):
            title = run.get("display_title", "")
            if (
                title.startswith(prefix)
                and (title == f"{prefix}{cid})" or run.get("status") != "completed")
                and run.get("event") == "workflow_dispatch"
                and run.get("path") == f".github/workflows/{WORKFLOW}"
                and run.get("head_branch") == self.default_branch
                and (run.get("repository") or {}).get("full_name") == self.repo
            ):
                return f"PR Nightly already requested: {run['html_url']}."
        inputs = {
            "pr_number": str(number),
            "head_sha": sha,
            "requester": login,
            "comment_id": str(cid),
            "case_id": case,
        }
        try:
            self.api(
                "POST",
                f"/actions/workflows/{WORKFLOW}/dispatches",
                {"ref": self.default_branch, "inputs": inputs},
            )
        except urllib.error.HTTPError as exc:
            raise Rejected(f"PR Nightly dispatch failed (HTTP {exc.code}).") from exc
        return (
            f"PR Nightly requested for `{case}` at `{sha}`. "
            f"[Actions]({self.url}/actions/workflows/{WORKFLOW})"
        )

    def inputs(self):
        inputs = super().inputs(case_id=CASE)
        if not valid_case(inputs["case_id"]):
            raise Rejected(USAGE)
        return inputs

    def validate(self):
        inputs = self.inputs()
        pr = self.api("GET", f"/pulls/{inputs['pr_number']}")
        login, case = inputs["requester"], inputs["case_id"]
        self.authorize_nightly(pr, login)
        if pr["head"]["sha"] != inputs["head_sha"]:
            raise Rejected("PR head changed; request Nightly again.")
        comment = self.api("GET", f"/issues/comments/{inputs['comment_id']}")
        author = comment.get("user") or {}
        api_url = self.env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        if (
            comment.get("issue_url")
            != f"{api_url}{self.root}/issues/{inputs['pr_number']}"
            or author.get("type") != "User"
            or author.get("login", "").lower() != login.lower()
            or (comment.get("body") or "").strip() != f"{COMMAND} {case}"
        ):
            raise Rejected("No matching PR Nightly request comment.")

    def report(self):
        inputs = self.inputs()
        needs = json.loads(self.env["NEEDS"])
        lane = needs["select"].get("outputs", {}).get("lane")
        result = (
            needs.get(lane, {}).get("result") if lane in ("recipe", "legacy") else None
        )
        failure = {
            "select": "Case selection failed",
            "build": "Wheel build failed",
        }
        for job, text in failure.items():
            stage = needs[job]["result"]
            if stage != "success":
                outcome = {
                    "failure": text,
                    "cancelled": "Cancelled",
                }.get(stage, f"{job.capitalize()} did not run")
                break
        else:
            outcome = {
                "success": "Passed",
                "failure": "Failed",
                "cancelled": "Cancelled",
            }.get(result, "Case did not run")
        case, sha = inputs["case_id"], inputs["head_sha"]
        reason, score, samples, micro = (None,) * 4
        if outcome in ("Passed", "Failed"):
            reason, score, samples, micro = public_result(
                Path(self.env["SUMMARY_PATH"])
            )
        lines = [
            f"PR Nightly `{case}` at `{sha}` — **{outcome}**",
            f"[Run, case logs and summary]({self.execution_url})",
        ]
        if reason and result == "failure":
            lines.append(f"- Failure reason: `{reason}`")
        elif result == "failure":
            lines.append("- Failure reason: unavailable")
        if score is not None:
            lines.append(
                f"- GSM8K: {score:.2%}" + (f" ({samples} samples)" if samples else "")
            )
        if micro:
            lines.append(f"- Micro-E2E: {micro}")
        number, cid = inputs["pr_number"], inputs["comment_id"]
        tag = marker(cid).replace("ppu-ci-command:", "ppu-pr-nightly-result:")
        body = "\n".join((tag, *lines))
        existing = self.find_comment(number, tag)
        if existing:
            self.api("PATCH", f"/issues/comments/{existing['id']}", {"body": body})
        else:
            self.api("POST", f"/issues/{number}/comments", {"body": body})
        if outcome != "Passed":
            raise Rejected(outcome)


if __name__ == "__main__":
    if sys.argv[1:2] == ["select"]:
        lane, matrix = resolve_case(
            Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "true"
        )
        print(f"lane={lane}")
        if matrix is not None:
            print("recipe_matrix=" + json.dumps(matrix, separators=(",", ":")))
    else:
        raise SystemExit(ci_main(controller=NightlyController))
