# SPDX-License-Identifier: Apache-2.0
"""Find a scheduled Nightly wheel donor for the PPU Preview build."""

import json
import os
import re
from itertools import islice

from scripts.ci.ppu_ci_command import Controller, GitHub, best_effort
from scripts.ci.ppu_wheel_manifest import NATIVE_INPUTS

NIGHTLY_WHEELS = "nightly-wheels.yaml"


def find_donor(controller):
    def native_input(path):
        return path in NATIVE_INPUTS or path.startswith(("cmake/", "csrc/"))

    runs = controller.pages(
        f"/actions/workflows/{NIGHTLY_WHEELS}/runs",
        "workflow_runs",
        limit=1,
        branch=controller.default_branch,
        event="schedule",
        status="completed",
    )
    for run in islice(runs, 5):
        run_id, source = run.get("id"), run.get("head_sha")
        if (
            type(run_id) is not int
            or not 0 < run_id < int(controller.env["GITHUB_RUN_ID"])
            or run.get("event") != "schedule"
            or run.get("status") != "completed"
            or run.get("path") != f".github/workflows/{NIGHTLY_WHEELS}"
            or run.get("head_branch") != controller.default_branch
            or (run.get("head_repository") or {}).get("full_name") != controller.repo
            or (run.get("repository") or {}).get("full_name") != controller.repo
            or not re.fullmatch(r"[0-9a-f]{40}", source or "")
        ):
            continue
        if not controller.trees_match(
            source, controller.env["GITHUB_SHA"], native_input
        ):
            continue
        # Subsequent publication outside Actions is independent of the build.
        if not any(
            job.get("name") == "build / PPU wheel build"
            and job.get("conclusion") == "success"
            for job in controller.pages(
                f"/actions/runs/{run_id}/jobs", "jobs", limit=1, filter="latest"
            )
        ):
            continue
        if any(
            artifact.get("name") == f"ppu-wheels-{run_id}"
            and artifact.get("expired") is False
            for artifact in controller.pages(
                f"/actions/runs/{run_id}/artifacts", "artifacts", limit=1
            )
        ):
            return str(run_id)
    return ""


def select_donor(controller):
    run_id = best_effort(lambda: find_donor(controller)) or ""
    print(f"Preview native wheel donor: {run_id or 'none; building from source'}")
    with open(controller.env["GITHUB_OUTPUT"], "a") as output:
        output.write(f"run_id={run_id}\n")


def main():
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    github = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_API_URL"])
    select_donor(Controller(github, event))


if __name__ == "__main__":
    main()
