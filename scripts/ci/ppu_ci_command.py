# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only PPU CI control plane; executor runs are the sole run database."""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import cached_property

COMMANDS = ("/ci run", "/ci run --allow-stale", "/ci retry", "/ci cancel")
HELP = "Commands: " + ", ".join(f"`{command}`" for command in COMMANDS)
READY = {"ready", "ready-run-all-tests"}
WRITE = {"write", "maintain", "admin"}
WORKFLOW = "ppu-pr-ci.yaml"
SKIP_PPU_PREFIXES = ("docs/", ".github/ISSUE_TEMPLATE/")
SKIP_PPU_FILES = {
    ".github/pull_request_template.md",
    ".gitignore",
    ".pre-commit-config.yaml",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "codecov.yml",
}
LOGIN = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?"
TITLE = re.compile(
    rf"PPU PR #([1-9][0-9]*) @ ([0-9a-f]{{40}}) by ({LOGIN}) \(comment ([1-9][0-9]*)\)"
)


def bot_owned(item):
    user = item.get("user") or {}
    return user.get("type") == "Bot" and user.get("login") == "github-actions[bot]"


def marker(comment_id):
    return f"<!-- ppu-ci-command:{comment_id} -->"


def best_effort(operation):
    try:
        return operation()
    except Exception as exc:
        print(f"Best-effort GitHub operation failed: {exc}", file=sys.stderr)


class Rejected(ValueError):
    """An expected refusal that should be explained to the requester."""


def parse_command(event):
    comment = event.get("comment", {})
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or comment.get("user", {}).get("type") != "User"
    ):
        return None
    body = comment.get("body", "").strip()
    if not body or body.split()[0] != "/ci":
        return None
    if body not in COMMANDS:
        raise Rejected(HELP)
    return body


class GitHub:
    """REST/GraphQL JSON transport. HTTP failures retain their status code."""

    def __init__(self, token, api_url="https://api.github.com"):
        self.token, self.api_url = token, api_url.rstrip("/")

    def request(self, method, path, data=None, params=None):
        base = self.api_url.removesuffix("/v3") if path == "/graphql" else self.api_url
        url = base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode() if data is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
        return json.loads(body) if body else None

    def graphql(self, query, variables):
        result = self.request(
            "POST", "/graphql", {"query": query, "variables": variables}
        )
        if result.get("errors"):
            raise RuntimeError("GitHub GraphQL request failed")
        return result["data"]


class Controller:
    """Inject request/GraphQL client, event, environment and sleep for CPU tests."""

    def __init__(self, github, event, env=None, sleep=time.sleep):
        self.gh, self.event = github, event
        self.env = os.environ if env is None else env
        self.sleep = sleep
        self.repo = self.env["GITHUB_REPOSITORY"]
        self.root = f"/repos/{self.repo}"
        server = self.env.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        self.url = f"{server}/{self.repo}"
        self.trusted = set(
            re.split(r"[,\s]+", self.env.get("CI_TRUSTED_USERS", "").lower())
        )

    def api(self, method, path, data=None, params=None):
        return self.gh.request(method, self.root + path, data, params)

    def pages(self, path, key=None, limit=None, **params):
        page = 1
        while limit is None or page <= limit:
            result = self.api(
                "GET", path, params={**params, "per_page": 100, "page": page}
            )
            items = result[key] if key else result
            yield from items
            if len(items) < 100:
                break
            page += 1

    def can_write(self, login):
        try:
            result = self.api(
                "GET", f"/collaborators/{urllib.parse.quote(login, safe='')}/permission"
            )
            return result.get("permission") in WRITE
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            return False

    def authorize(self, pr, login):
        if pr["state"] != "open" or pr["base"]["repo"]["full_name"] != self.repo:
            raise Rejected("PR must be open and target this repository.")
        writer = self.can_write(login)
        if writer or login.lower() in self.trusted:
            return writer
        if login.lower() != pr["user"]["login"].lower() or pr.get("draft"):
            raise Rejected(
                "Only maintainers, trusted users, or an authorized non-draft PR author may run CI."
            )
        if READY.intersection(label["name"] for label in pr.get("labels", [])):
            return False
        owner, name = self.repo.split("/")
        result = self.gh.graphql(
            "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){pullRequest(number:$number){reviewDecision}}}",
            {"owner": owner, "name": name, "number": pr["number"]},
        )
        decision = ((result.get("repository") or {}).get("pullRequest") or {}).get(
            "reviewDecision"
        )
        if decision == "APPROVED":
            latest = {}
            for review in sorted(
                self.pages(f"/pulls/{pr['number']}/reviews"),
                key=lambda r: (r.get("submitted_at") or "", r["id"]),
            ):
                if review.get("user") and review["state"] in {
                    "APPROVED",
                    "CHANGES_REQUESTED",
                    "DISMISSED",
                }:
                    latest[review["user"]["login"].lower()] = review["state"]
            for reviewer, state in latest.items():
                if state == "APPROVED" and (
                    reviewer in self.trusted or self.can_write(reviewer)
                ):
                    return False
        raise Rejected(
            "PR author needs ready/ready-run-all-tests or approval from a trusted reviewer."
        )

    @cached_property
    def default_branch(self):
        return (
            self.event.get("repository", {}).get("default_branch")
            or self.api("GET", "")["default_branch"]
        )

    @property
    def execution_url(self):
        return f"{self.url}/actions/runs/{self.env['GITHUB_RUN_ID']}/attempts/{self.env.get('GITHUB_RUN_ATTEMPT', '1')}"

    def skips_ppu(self, pr, head_sha):
        """Return true only for a small, complete allowlist of non-executable files."""
        changed = pr.get("changed_files")
        if not isinstance(changed, int) or not 0 < changed <= 300:
            return False
        files = list(self.pages(f"/pulls/{pr['number']}/files", limit=3))
        if len(files) != changed:
            return False
        paths = []
        for file in files:
            filename = file.get("filename")
            if not isinstance(filename, str):
                return False
            paths.append(filename)
            if file.get("status") == "renamed":
                previous = file.get("previous_filename")
                if not isinstance(previous, str):
                    return False
                paths.append(previous)
        unchanged = self.api("GET", f"/pulls/{pr['number']}")["head"]["sha"] == head_sha
        return unchanged and all(
            path in SKIP_PPU_FILES or path.startswith(SKIP_PPU_PREFIXES)
            for path in paths
        )

    def reusable_wheels(self, sha):
        no_wheel_prefixes = (
            "docs/",
            "tests/",
            "benchmarks/",
            ".github/workflows/",
            "scripts/",
            "tools/",
        )
        no_wheel_files = {
            "AGENTS.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "MANIFEST.in",
            "README.md",
            "requirements/dev.txt",
            ".gitignore",
            ".pre-commit-config.yaml",
        }
        rebuild_prefixes = ("cmake/", "csrc/")
        rebuild_files = {
            ".github/vllm-main-verified.commit",
            ".github/workflows/_ppu-build.yaml",
            "CMakeLists.txt",
            "pyproject.toml",
            "requirements/build.txt",
            "setup.py",
            "scripts/ci/build_ppu_wheels.sh",
            "scripts/ci/ppu_image_stack.py",
            "scripts/ci/ppu_wheel_manifest.py",
            "vllm_sail/native/manifest.py",
            "vllm_sail/native/plugin_manifest.py",
            "vllm_sail/native/resources.py",
            "vllm_sail/native/toolchain.py",
        }
        repack_files = {"requirements/ppu.txt", "requirements/profiling.txt"}

        def no_wheel(path):
            return path in no_wheel_files or path.startswith(no_wheel_prefixes)

        def native_safe(path):
            return (
                (
                    no_wheel(path)
                    or path in repack_files
                    or path.startswith("vllm_sail/")
                )
                and path not in rebuild_files
                and not path.startswith(rebuild_prefixes)
            )

        for run in self.find_runs()[:20]:
            if (
                run.get("status") != "completed"
                or run.get("conclusion") != "success"
                or not re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha") or "")
                or not 0 < run["id"] < int(self.env["GITHUB_RUN_ID"])
            ):
                continue
            if not any(
                a["name"] == f"ppu-wheels-{run['id']}" and a.get("expired") is False
                for a in self.pages(f"/actions/runs/{run['id']}/artifacts", "artifacts")
            ):
                continue
            source = TITLE.fullmatch(run["display_title"])[2]
            same_workflow = run["head_sha"] == self.env["GITHUB_SHA"]
            if source == sha and same_workflow:
                return "full", str(run["id"]), source
            diff = self.api("GET", f"/compare/{source}...{sha}")
            files = diff.get("files")
            if not (
                diff.get("status") in {"ahead", "identical"}
                and (diff.get("merge_base_commit") or {}).get("sha") == source
                and isinstance(files, list)
                and len(files) < 300
            ):
                continue
            paths = []
            for file in files:
                if not isinstance(file.get("filename"), str):
                    paths = []
                    break
                paths.append(file["filename"])
                if file.get("status") == "renamed":
                    if not isinstance(file.get("previous_filename"), str):
                        paths = []
                        break
                    paths.append(file["previous_filename"])
            if files and not paths:
                continue
            if same_workflow and all(
                no_wheel(path) and path not in rebuild_files for path in paths
            ):
                return "full", str(run["id"]), source
            if all(native_safe(path) for path in paths):
                return "native", str(run["id"]), sha
        return "none", "", ""

    def find_runs(self, pr=None, head_sha=None):
        query = (
            {"created": ">=" + pr["created_at"][:10]} if pr else {"status": "success"}
        )
        found = []
        for run in self.pages(
            f"/actions/workflows/{WORKFLOW}/runs",
            "workflow_runs",
            limit=10,
            event="workflow_dispatch",
            **query,
        ):
            title = TITLE.fullmatch(run.get("display_title", ""))
            if (
                title
                and (pr is None or int(title[1]) == pr["number"])
                and (head_sha is None or title[2] == head_sha)
                and run.get("event") == "workflow_dispatch"
                and run.get("path") == f".github/workflows/{WORKFLOW}"
                and run.get("head_branch") == self.default_branch
                and run.get("repository", {}).get("full_name") == self.repo
            ):
                found.append(run)
        return sorted(found, key=lambda run: run["id"], reverse=True)

    def find_comment(self, number, tag):
        for comment in self.pages(f"/issues/{number}/comments"):
            if bot_owned(comment) and tag in (comment.get("body") or ""):
                return comment

    def reply(self, number, comment_id, text):
        body = f"{marker(comment_id)}\n{text}"
        self.api("POST", f"/issues/{number}/comments", {"body": body})

    def react(self, comment_id, content):
        path = f"/issues/comments/{comment_id}/reactions"
        best_effort(lambda: self.api("POST", path, {"content": content}))

    def status(self, sha, state, description):
        payload = dict(
            state=state,
            context="PPU CI",
            description=description[:140],
            target_url=self.execution_url,
        )
        self.api("POST", f"/statuses/{sha}", payload)

    def launch(self, sha, description, path, data=None, fallback=None):
        self.status(sha, "pending", description)
        try:
            try:
                self.api("POST", path, data)
            except urllib.error.HTTPError as exc:
                if not fallback or not 400 <= exc.code < 500:
                    raise
                self.api("POST", fallback)
        except Exception as exc:
            self.status(sha, "error", "CI request failed")
            if isinstance(exc, urllib.error.HTTPError):
                raise Rejected(f"CI request failed (HTTP {exc.code}).") from exc
            raise

    def command(self):
        try:
            action = parse_command(self.event)
        except Rejected:
            action = "help"
        if action is None:
            return
        number, comment = self.event["issue"]["number"], self.event["comment"]
        cid, login = comment["id"], comment["user"]["login"]
        try:
            path = f"/issues/comments/{cid}/reactions"
            reactions = best_effort(lambda: list(self.pages(path))) or []
            if any(
                bot_owned(r) and r["content"] in {"rocket", "-1"} for r in reactions
            ) or self.find_comment(number, marker(cid)):
                return
            self.react(cid, "eyes")
            reaction = "rocket"
            try:
                if action == "help":
                    raise Rejected(HELP)
                pr = self.api("GET", f"/pulls/{number}")
                self.authorize(pr, login)
                if action == "/ci cancel":
                    text = self.cancel(pr, login)
                elif action == "/ci retry":
                    text = self.retry(pr, login)
                else:
                    text = self.run(pr, login, cid, action.endswith("--allow-stale"))
            except Rejected as exc:
                text, reaction = str(exc), "-1"
            self.reply(number, cid, text)
            self.react(cid, reaction)
        except Exception:
            self.react(cid, "confused")
            raise

    def run(self, pr, login, cid, allow_stale):
        sha = pr["head"]["sha"]
        runs = self.find_runs(pr, sha)
        active = next((r for r in runs if r["status"] != "completed"), None)
        if active:
            return f"PPU CI already running: {active['html_url']}"
        completed = next((r for r in runs if r["status"] == "completed"), None)
        base = urllib.parse.quote(pr["base"]["ref"], safe="")
        behind = self.api("GET", f"/compare/{base}...{sha}")["behind_by"]
        if behind and not allow_stale:
            raise Rejected(
                f"PR is {behind} commits behind base. Rebase or use `/ci run --allow-stale`."
            )
        fresh = self.api("GET", f"/pulls/{pr['number']}")
        if fresh["head"]["sha"] != sha:
            raise Rejected("PR head changed; use /ci run again.")
        writer = self.authorize(fresh, login)
        if completed and not writer:
            raise Rejected(
                f"Already tested: {completed['html_url']}. Use `/ci retry` for failed "
                "jobs or push a new commit; write permission is required to dispatch again."
            )
        dispatch_inputs = {
            "pr_number": str(pr["number"]),
            "head_sha": sha,
            "requester": login,
            "comment_id": str(cid),
        }
        self.launch(
            sha,
            f"Queued by @{login}",
            f"/actions/workflows/{WORKFLOW}/dispatches",
            {"ref": self.default_branch, "inputs": dispatch_inputs},
        )
        warning = f"Warning: running {behind} commits behind base. " if behind else ""
        deadline = time.monotonic() + 30
        for _ in range(10):
            if time.monotonic() + 3 > deadline:
                break
            self.sleep(3)
            for run in best_effort(lambda: self.find_runs(pr, sha)) or []:
                if TITLE.fullmatch(run["display_title"])[4] == str(cid):
                    return warning + f"PPU CI queued for {sha[:12]}: {run['html_url']}"
        return warning + (
            f"Dispatch accepted; run link pending: {self.url}/actions/workflows/{WORKFLOW}"
        )

    def cancel(self, pr, login):
        links = []
        for run in self.find_runs(pr):
            if run["status"] != "completed":
                self.api("POST", f"/actions/runs/{run['id']}/cancel")
                sha = TITLE.fullmatch(run["display_title"])[2]
                self.status(sha, "error", f"Cancelled by @{login}")
                links.append(run["html_url"])
        return (
            f"Cancellation requested for {len(links)} runs: " + " ".join(links)
            if links
            else "No active PPU CI runs (none to cancel)."
        )

    def retry(self, pr, login):
        runs = self.find_runs(pr, pr["head"]["sha"])
        if not runs:
            raise Rejected("No run for this head; use /ci run.")
        run = runs[0]
        if run["status"] != "completed":
            return f"PPU CI already running: {run['html_url']}"
        if run["conclusion"] == "success":
            return f"Nothing to rerun: {run['html_url']}"
        path = f"/actions/runs/{run['id']}"
        jobs = self.pages(path + "/jobs", "jobs", filter="latest")
        built = any(
            j["name"].rsplit("/", 1)[-1].strip() == "PPU wheel build"
            and j["conclusion"] == "success"
            for j in jobs
        )
        if built and not any(
            a["name"] == f"ppu-wheels-{run['id']}" and not a.get("expired", True)
            for a in self.pages(path + "/artifacts", "artifacts")
        ):
            raise Rejected("Wheel artifact missing or expired; use /ci run.")
        # Native retry can reuse successful validate; report depends on failed ppu.
        self.launch(
            pr["head"]["sha"],
            f"Retry requested by @{login}",
            path + "/rerun-failed-jobs",
            fallback=path + "/rerun",
        )
        return f"PPU CI retry requested: {run['html_url']}/attempts/{run['run_attempt'] + 1}"

    def notify(self):
        pr = self.event.get("pull_request", {})
        if (
            self.event.get("action") != "labeled"
            or self.event.get("label", {}).get("name") not in READY
            or pr.get("state") != "open"
            or pr.get("draft")
        ):
            return
        login = pr["user"]["login"]
        tag = "<!-- ppu-ci-authorized -->"
        if (
            login.lower() in self.trusted
            or self.can_write(login)
            or self.find_comment(pr["number"], tag)
        ):
            return
        body = f"{tag}\n@{login}: {HELP}. New commits require `/ci run`. `--allow-stale` permits running behind the base branch. CPU checks remain automatic."
        self.api("POST", f"/issues/{pr['number']}/comments", {"body": body})

    def inputs(self):
        inputs = self.event.get("inputs", {})
        patterns = dict(
            pr_number=r"[1-9][0-9]*",
            head_sha=r"[0-9a-f]{40}",
            requester=LOGIN,
            comment_id=r"[1-9][0-9]*",
        )
        for key, pattern in patterns.items():
            value = inputs.get(key)
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise Rejected(f"Invalid input: {key}")
        if inputs.keys() - patterns.keys():
            raise Rejected("Unexpected executor inputs")
        return inputs

    def validate(self):
        inputs = self.inputs()
        pr = self.api("GET", f"/pulls/{inputs['pr_number']}")
        if pr["head"]["sha"] != inputs["head_sha"]:
            raise Rejected("PR head changed; use /ci run again.")
        self.authorize(pr, inputs["requester"])
        self.status(
            inputs["head_sha"],
            "pending",
            f"Running (attempt {self.env.get('GITHUB_RUN_ATTEMPT', '1')})",
        )
        skip = best_effort(lambda: self.skips_ppu(pr, inputs["head_sha"])) is True
        mode, build, source = ("skip", "", "")
        if not skip:
            mode, build, source = best_effort(
                lambda: self.reusable_wheels(inputs["head_sha"])
            ) or ("none", "", "")
        print(f"Wheel reuse: mode={mode}, run={build}, source={source}")
        existing = build if mode == "full" else ""
        native = build if mode == "native" else ""
        with open(self.env["GITHUB_OUTPUT"], "a") as output:
            output.write(
                f"skip_ppu={'true' if skip else 'false'}\n"
                f"existing_build_run_id={existing}\n"
                f"native_reuse_run_id={native}\nwheel_source_sha={source}\n"
            )

    def report(self):
        inputs = self.inputs()
        needs = json.loads(self.env["NEEDS"])
        results = {
            stage: needs.get(stage, {}).get("result") for stage in ("validate", "ppu")
        }
        skipped = needs.get("validate", {}).get("outputs", {}).get("skip_ppu") == "true"
        state, description = "success", "Passed"
        if skipped and all(result == "success" for result in results.values()):
            description = "Skipped (no PPU-impacting changes)"
        if "cancelled" in results.values():
            state, description = "error", "Cancelled"
        elif any(result != "success" for result in results.values()):
            stage = next(
                stage for stage, result in results.items() if result != "success"
            )
            state, description = "failure", f"{stage} failed"
        description += f" (attempt {self.env.get('GITHUB_RUN_ATTEMPT', '1')})"
        self.status(inputs["head_sha"], state, description)

        def comment():
            number, cid = inputs["pr_number"], inputs["comment_id"]
            previous = self.find_comment(number, marker(cid))
            text = f"PPU CI: {description}: {self.execution_url}"
            if previous:
                self.api(
                    "PATCH",
                    f"/issues/comments/{previous['id']}",
                    {"body": f"{marker(cid)}\n{text}"},
                )
            else:
                self.reply(number, cid, text)

        best_effort(comment)


def main(argv=None, env=None, github=None, sleep=time.sleep):
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("command", "notify", "validate", "report"))
    mode = parser.parse_args(argv).mode
    try:
        with open(env["GITHUB_EVENT_PATH"]) as stream:
            event = json.load(stream)
        github = github or GitHub(
            env["GITHUB_TOKEN"], env.get("GITHUB_API_URL", "https://api.github.com")
        )
        getattr(Controller(github, event, env, sleep), mode)()
    except Exception as exc:
        if mode != "validate":
            raise
        message = f"PPU CI validation failed: {exc}\n"
        print(message, file=sys.stderr)
        if env.get("GITHUB_STEP_SUMMARY"):
            with open(env["GITHUB_STEP_SUMMARY"], "a") as stream:
                stream.write(message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
