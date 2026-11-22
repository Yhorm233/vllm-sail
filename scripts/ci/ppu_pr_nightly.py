# SPDX-License-Identifier: Apache-2.0
"""Authorize and report on-demand Nightly accuracy cases for an immutable PR head."""

import json
import math
import os
import re
import subprocess
import sys
import urllib.error
from pathlib import Path

from scripts.ci import ppu_e2e_select, ppu_recipe_diagnostics
from scripts.ci.ppu_ci_command import (
    HELP,
    Controller,
    Rejected,
    best_effort,
    marker,
    parse_command,
)
from scripts.ci.ppu_ci_command import main as ci_main

COMMAND = "/ci run nightly"
WORKFLOW = "ppu-pr-nightly.yaml"
CASE = r"[a-z0-9][a-z0-9.-]{0,63}"
MAX_CASES = 16
MAX_CASE_SPEC_LEN = 150
USAGE = (
    f"Use `{COMMAND} <case>`, `{COMMAND} <case1>,<case2>`, or `{COMMAND} all` "
    "with Recipe policy slugs or legacy accuracy case IDs."
)
NIGHTLY_HELP = f"{HELP}, `{COMMAND} <case[,case...]|all>`"
TITLE = re.compile(
    r"PPU PR #([1-9][0-9]*) Nightly ([a-z0-9.,-]+) "
    r"@ ([0-9a-f]{40}) \(comment ([1-9][0-9]*)\)"
)
CASE_STATUSES = {"pass", "fail", "cancelled", "not_run"}
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
    return (
        isinstance(case, str) and case != "all" and re.fullmatch(CASE, case) is not None
    )


def parse_case_spec(raw):
    if not isinstance(raw, str) or re.search(r"[\r\n\t]", raw):
        raise Rejected(USAGE)
    spec = raw.strip()
    if not spec or re.search(r"(^,|,$|,\s*,)", spec):
        raise Rejected(USAGE)
    tokens = re.split(r"\s*,\s*|\s+", spec)
    if tokens == ["all"]:
        return ("all",)
    if "all" in tokens:
        raise Rejected(f"Cannot combine `all` with individual cases. {USAGE}")
    seen = set()
    for token in tokens:
        if not valid_case(token):
            raise Rejected(USAGE)
        if token in seen:
            raise Rejected(f"Duplicate Nightly case: `{token}`. {USAGE}")
        seen.add(token)
    canonical = ",".join(tokens)
    if len(tokens) > MAX_CASES or len(canonical) > MAX_CASE_SPEC_LEN:
        raise Rejected(
            f"Too many Nightly cases in one request; split requests or use `{COMMAND} all`."
        )
    return tuple(tokens)


def canonical_case_spec(raw):
    return ",".join(parse_case_spec(raw))


def valid_case_spec(spec):
    if not isinstance(spec, str) or re.search(r"\s", spec):
        return False
    try:
        return canonical_case_spec(spec) == spec
    except Rejected:
        return False


def nightly_command(event):
    try:
        return parse_command(event)
    except Rejected:
        body = event["comment"]["body"].strip()
        if body == COMMAND or body.startswith(COMMAND + " "):
            return body
        raise


def _run_candidate_recipe_select(source, models_csv):
    process = subprocess.run(
        [
            sys.executable,
            str(source / "scripts/ci/ppu_e2e_select.py"),
            "--recipes",
            models_csv,
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
        selected = None
    if process.returncode or not isinstance(selected, list):
        label = models_csv or "all"
        raise Rejected(f"Recipe policy selection failed for `{label}` (model_sanity).")
    return selected


def resolve_cases(source, case_spec, recipe_enabled):
    cases = parse_case_spec(case_spec)
    if cases == ("all",):
        if not recipe_enabled:
            raise Rejected("Recipe model sanity is disabled.")
        selected = _run_candidate_recipe_select(source, "")
        recipe_cases = []
        seen_recipe = set()
        for row in selected:
            if not isinstance(row, dict):
                raise Rejected("Malformed Recipe model sanity matrix entry.")
            key = row.get("key")
            policy = source / f"tests/ppu/policy/models/{key}.yaml"
            if not valid_case(key) or key in seen_recipe or not policy.is_file():
                raise Rejected(f"Invalid Recipe model sanity case: {key}.")
            seen_recipe.add(key)
            recipe_cases.append(key)
        if not recipe_cases:
            raise Rejected("No Recipe model sanity cases available.")
        return recipe_cases, {"include": selected}, []

    catalog = json.loads((source / "scripts/ci/ppu_e2e_models.json").read_text())
    recipe_cases = []
    legacy_cases = []
    for case in cases:
        catalog_matches = [entry for entry in catalog if entry.get("key") == case]
        if len(catalog_matches) > 1:
            raise Rejected(f"Duplicate catalog key: {case}.")
        legacy = bool(
            catalog_matches and catalog_matches[0].get("category") == "accuracy"
        )
        policy = source / f"tests/ppu/policy/models/{case}.yaml"
        if policy.is_file():
            if legacy:
                raise Rejected(
                    f"Case {case} is ambiguous across Recipe and legacy accuracy."
                )
            if not recipe_enabled:
                raise Rejected("Recipe model sanity is disabled.")
            recipe_cases.append(case)
        elif legacy:
            legacy_cases.append(case)
        else:
            raise Rejected(
                f"Unknown Nightly accuracy case: {case}. See tests/ppu/README.md."
            )

    recipe_matrix = None
    if recipe_cases:
        selected = _run_candidate_recipe_select(source, ",".join(recipe_cases))
        if (
            len(selected) != len(recipe_cases)
            or not all(isinstance(row, dict) for row in selected)
            or [row.get("key") for row in selected] != recipe_cases
        ):
            raise Rejected(
                f"Expected exact Recipe model sanity cases: {','.join(recipe_cases)}."
            )
        recipe_matrix = {"include": selected}

    if legacy_cases:
        try:
            selected_legacy = ppu_e2e_select.build_accuracy_matrix(
                catalog, ",".join(legacy_cases)
            )["include"]
        except ValueError as exc:
            raise Rejected(str(exc)) from exc
        if [row["case_id"] for row in selected_legacy] != legacy_cases:
            raise Rejected(
                f"Expected exact legacy accuracy cases: {','.join(legacy_cases)}."
            )

    return recipe_cases, recipe_matrix, legacy_cases


def public_result(path):
    try:
        summary = json.loads(ppu_recipe_diagnostics._read_file(path, 16 * 1024))
    except (ValueError, RecursionError):
        return None, None, None, None, None
    if not isinstance(summary, dict):
        return None, None, None, None, None
    gsm8k = summary.get("gsm8k")
    if not isinstance(gsm8k, dict):
        gsm8k = {}
    status = summary.get("status")
    reason = summary.get("failure_reason")
    score = gsm8k.get("score")
    samples = gsm8k.get("evaluated_samples")
    micro = summary.get("micro_e2e")
    return (
        status if isinstance(status, str) and status in CASE_STATUSES else None,
        reason if isinstance(reason, str) and reason in REASONS else None,
        score
        if type(score) in (int, float) and 0 <= score <= 1 and math.isfinite(score)
        else None,
        samples if type(samples) is int and 0 < samples <= 100_000 else None,
        micro if micro in ("pass", "fail") else None,
    )


def _split_output_cases(value):
    if not isinstance(value, str) or not value.strip():
        return []
    cases = [item.strip() for item in value.split(",") if item.strip()]
    return [case for case in cases if valid_case(case)]


def _format_gsm8k(score, samples):
    if score is None:
        return "—"
    return f"{score:.2%}" + (f" ({samples})" if samples else "")


class NightlyController(Controller):
    command_help = NIGHTLY_HELP

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
            raise Rejected(f"{USAGE} [List of available Nightly cases]({matrix}).")
        requested_cases = parse_case_spec(action.removeprefix(COMMAND + " "))
        case_spec = ",".join(requested_cases)
        requested_set = frozenset(requested_cases)
        self.authorize_nightly(pr, login)
        sha, number = pr["head"]["sha"], pr["number"]
        result = self.api("GET", f"/pulls/{number}")
        if result["head"]["sha"] != sha or result["state"] != "open":
            raise Rejected("PR changed; request Nightly again for the current head.")
        # GitHub caps filtered run listings at 1,000 results. Filter the
        # workflow's unfiltered pages locally so older active runs are visible.
        for run in self.pages(f"/actions/workflows/{WORKFLOW}/runs", "workflow_runs"):
            match = TITLE.fullmatch(run.get("display_title", ""))
            if (
                match
                and match[1] == str(number)
                and match[3] == sha
                and frozenset(match[2].split(",")) == requested_set
                and (match[4] == str(cid) or run.get("status") != "completed")
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
            "case_id": case_spec,
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
            f"PR Nightly requested for `{case_spec}` at `{sha}`. "
            f"[Actions]({self.url}/actions/workflows/{WORKFLOW})"
        )

    def inputs(self):
        inputs = super().inputs(case_id=r"[a-z0-9.,-]+")
        if not valid_case_spec(inputs["case_id"]):
            raise Rejected(USAGE)
        return inputs

    def validate(self):
        inputs = self.inputs()
        pr = self.api("GET", f"/pulls/{inputs['pr_number']}")
        login, case_spec = inputs["requester"], inputs["case_id"]
        self.authorize_nightly(pr, login)
        if pr["head"]["sha"] != inputs["head_sha"]:
            raise Rejected("PR head changed; request Nightly again.")
        comment = self.api("GET", f"/issues/comments/{inputs['comment_id']}")
        author = comment.get("user") or {}
        api_url = self.env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        raw_body = (comment.get("body") or "").strip()
        matched_spec = None
        if raw_body.startswith(COMMAND + " "):
            try:
                matched_spec = canonical_case_spec(raw_body.removeprefix(COMMAND + " "))
            except Rejected:
                matched_spec = None
        if (
            comment.get("issue_url")
            != f"{api_url}{self.root}/issues/{inputs['pr_number']}"
            or author.get("type") != "User"
            or author.get("login", "").lower() != login.lower()
            or matched_spec != case_spec
        ):
            raise Rejected("No matching PR Nightly request comment.")

    def _summary_path(self, lane, case, attempt=None):
        summary_dir = self.env.get("SUMMARY_DIR")
        if summary_dir:
            prefix = (
                "ppu-recipe-model_sanity"
                if lane == "recipe"
                else "ppu-pr-nightly-legacy"
            )
            run_id = self.env["GITHUB_RUN_ID"]
            resolved_attempt = (
                str(attempt)
                if attempt is not None
                else self.env.get("GITHUB_RUN_ATTEMPT", "1")
            )
            return (
                Path(summary_dir)
                / f"{prefix}-{case}-{run_id}-{resolved_attempt}"
                / "summary.json"
            )
        return Path(self.env["SUMMARY_PATH"])

    def _current_attempt(self):
        raw = self.env.get("GITHUB_RUN_ATTEMPT", "1")
        return (
            int(raw) if isinstance(raw, str) and raw.isdigit() and int(raw) > 0 else 1
        )

    def _latest_case_jobs(self, current_attempt):
        if current_attempt <= 1:
            return {}
        run_id = self.env.get("GITHUB_RUN_ID", "")
        if not run_id:
            return {}
        jobs = best_effort(
            lambda: list(
                self.pages(f"/actions/runs/{run_id}/jobs", "jobs", filter="latest")
            )
        )
        if not jobs:
            return {}
        mapping = {}
        for job in jobs:
            if not isinstance(job, dict):
                continue
            name = job.get("name")
            attempt = job.get("run_attempt")
            if (
                not isinstance(name, str)
                or type(attempt) is not int
                or not 1 <= attempt <= current_attempt
            ):
                continue
            leaf = name.rsplit("/", 1)[-1].strip()
            if leaf.startswith("Accuracy "):
                case = leaf.removeprefix("Accuracy ").strip()
                lane = "legacy"
            elif leaf.startswith("Model sanity & micro-E2E "):
                case = leaf.removeprefix("Model sanity & micro-E2E ").strip()
                lane = "recipe"
            else:
                continue
            if valid_case(case):
                key = (lane, case)
                prev = mapping.get(key)
                if prev is None or attempt >= prev[0]:
                    mapping[key] = (attempt, job.get("conclusion"))
        return mapping

    def _resolve_case_summary(
        self, lane, case, lane_result, current_attempt, latest_jobs
    ):
        job_info = latest_jobs.get((lane, case))
        if job_info is not None:
            attempt, conclusion = job_info
            result = public_result(self._summary_path(lane, case, attempt=attempt))
            return result, conclusion
        result = public_result(self._summary_path(lane, case, attempt=current_attempt))
        if (
            any(item is not None for item in result)
            or current_attempt <= 1
            or lane_result != "success"
            or not self.env.get("SUMMARY_DIR")
        ):
            return result, None
        # Fallback only when job metadata is unavailable and the lane succeeded:
        # accept only the latest prior passing summary, never a stale failure.
        for attempt in range(current_attempt - 1, 0, -1):
            candidate = public_result(self._summary_path(lane, case, attempt=attempt))
            if candidate[0] == "pass":
                return candidate, None
        return result, None

    def report(self):
        inputs = self.inputs()
        needs = json.loads(self.env["NEEDS"])
        select_outputs = needs["select"].get("outputs", {})
        legacy_cases = _split_output_cases(select_outputs.get("legacy_models", ""))
        recipe_cases = _split_output_cases(select_outputs.get("recipe_models", ""))
        selected_cases = [
            *((case, "legacy") for case in legacy_cases),
            *((case, "recipe") for case in recipe_cases),
        ]
        failure = {
            "select": "Case selection failed",
            "build": "Wheel build failed",
        }
        setup_failed = False
        for job, text in failure.items():
            stage = needs[job]["result"]
            if stage != "success":
                setup_failed = True
                outcome = {
                    "failure": text,
                    "cancelled": "Cancelled",
                }.get(stage, f"{job.capitalize()} did not run")
                break
        else:
            lane_results = []
            if legacy_cases:
                lane_results.append(needs.get("legacy", {}).get("result"))
            if recipe_cases:
                lane_results.append(needs.get("recipe", {}).get("result"))
            if not lane_results:
                outcome = "Case did not run"
            elif "failure" in lane_results:
                outcome = "Failed"
            elif "cancelled" in lane_results:
                outcome = "Cancelled"
            elif all(res == "success" for res in lane_results):
                outcome = "Passed"
            else:
                outcome = "Case did not run"

        case_spec, sha = inputs["case_id"], inputs["head_sha"]
        single_mode = (
            not setup_failed and len(selected_cases) == 1 and case_spec != "all"
        )
        current_attempt = self._current_attempt()
        latest_jobs = (
            self._latest_case_jobs(current_attempt)
            if not setup_failed and selected_cases
            else {}
        )
        if setup_failed or not selected_cases:
            lines = [
                f"PR Nightly `{case_spec}` at `{sha}` — **{outcome}**",
                f"[Run, case logs and summary]({self.execution_url})",
            ]
        elif single_mode:
            case, lane = selected_cases[0]
            lane_result = needs.get(lane, {}).get("result")
            _, reason, score, samples, micro = (None,) * 5
            if outcome in ("Passed", "Failed"):
                (_, reason, score, samples, micro), _ = self._resolve_case_summary(
                    lane, case, lane_result, current_attempt, latest_jobs
                )
            lines = [
                f"PR Nightly `{case}` at `{sha}` — **{outcome}**",
                f"[Run, case logs and summary]({self.execution_url})",
            ]
            if reason and lane_result == "failure":
                lines.append(f"- Failure reason: `{reason}`")
            elif lane_result == "failure":
                lines.append("- Failure reason: unavailable")
            if score is not None:
                lines.append(
                    f"- GSM8K: {score:.2%}"
                    + (f" ({samples} samples)" if samples else "")
                )
            if micro:
                lines.append(f"- Micro-E2E: {micro}")
        else:
            rows = []
            passed = 0
            for case, lane in selected_cases:
                lane_result = needs.get(lane, {}).get("result")
                conclusion = None
                if lane_result in ("success", "failure", "cancelled"):
                    (
                        (status, reason, score, samples, micro),
                        conclusion,
                    ) = self._resolve_case_summary(
                        lane, case, lane_result, current_attempt, latest_jobs
                    )
                else:
                    status, reason, score, samples, micro = (None,) * 5
                if status is None:
                    status = {
                        "cancelled": "cancelled",
                        "skipped": "not_run",
                    }.get(conclusion or lane_result, "unavailable")
                if status == "pass":
                    passed += 1
                reason_cell = (
                    f"`{reason}`"
                    if status == "fail" and reason
                    else ("unavailable" if status == "fail" else "—")
                )
                rows.append(
                    f"| `{case}` | {lane} | {status} | "
                    f"{_format_gsm8k(score, samples)} | {micro or '—'} | {reason_cell} |"
                )
            lines = [
                f"PR Nightly `{case_spec}` at `{sha}` — **{outcome}** "
                f"({passed}/{len(selected_cases)} passed)",
                f"[Run, case logs and summary]({self.execution_url})",
                "",
                "| Case | Lane | Result | GSM8K | Micro-E2E | Failure reason |",
                "| --- | --- | --- | --- | --- | --- |",
                *rows,
            ]

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
        recipe_cases, matrix, legacy_cases = resolve_cases(
            Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "true"
        )
        print("recipe_models=" + ",".join(recipe_cases))
        print("legacy_models=" + ",".join(legacy_cases))
        print(
            "recipe_matrix="
            + (json.dumps(matrix, separators=(",", ":")) if matrix is not None else "")
        )
    else:
        raise SystemExit(ci_main(controller=NightlyController))
