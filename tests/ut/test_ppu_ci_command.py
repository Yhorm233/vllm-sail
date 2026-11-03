# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the PPU command control plane."""

import copy
import importlib
import io
import json
from urllib.error import HTTPError

import pytest

REPO = "t-head/vllm-sail"
ROOT = f"/repos/{REPO}"
SHA = "a" * 40
OLD = "b" * 40
BOT = {"login": "github-actions[bot]", "type": "Bot"}
USER = {"login": "author", "type": "User"}
ENV = {
    "GITHUB_REPOSITORY": REPO,
    "GITHUB_SERVER_URL": "https://github.example",
    "GITHUB_RUN_ID": "900",
    "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_SHA": "c" * 40,
    "CI_TRUSTED_USERS": "Trusted, OTHER\nthird",
}
PR = {
    "number": 7,
    "created_at": "2026-10-01T12:34:56Z",
    "state": "open",
    "draft": False,
    "user": USER,
    "head": {"sha": SHA},
    "base": {"ref": "release/next", "repo": {"full_name": REPO}},
    "changed_files": 1,
    "labels": [{"name": "ready"}],
}
EVENT = {
    "action": "created",
    "repository": {"default_branch": "main"},
    "issue": {"number": 7, "pull_request": {}},
    "comment": {"id": 42, "body": "/ci run", "user": USER},
}


def module():
    return importlib.import_module("scripts.ci.ppu_ci_command")


@pytest.fixture(autouse=True)
def github_output(tmp_path, monkeypatch):
    output = tmp_path / "github-output"
    monkeypatch.setitem(ENV, "GITHUB_OUTPUT", str(output))
    return output


def http_error(code):
    return HTTPError("https://github.example", code, "test rejection", {}, None)


class FakeGitHub:
    """Predetermined route responses; unexpected requests fail the test."""

    def __init__(self):
        self.calls = []
        self.routes = {}

    def add(self, method, path, *responses, params=None):
        key = (method, path, tuple(sorted((params or {}).items())))
        self.routes[key] = list(responses)
        return self

    def request(self, method, path, data=None, params=None):
        self.calls.append((method, path, copy.deepcopy(data), params or {}))
        key = (method, path, tuple(sorted((params or {}).items())))
        assert key in self.routes, f"Unexpected request: {self.calls[-1]}"
        responses = self.routes[key]
        response = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)

    def graphql(self, query, variables):
        return module().GitHub.graphql(self, query, variables)

    def page(self, path, response, key=None, page=1, **params):
        return self.add(
            "GET",
            ROOT + path,
            {key: response} if key else response,
            params={"per_page": 100, "page": page, **params},
        )

    def writes(self, ending):
        return [c for c in self.calls if c[0] != "GET" and c[1].endswith(ending)]


def make_controller(event=None, env=None):
    gh = FakeGitHub()
    sleeps = []
    control = module().Controller(
        gh, copy.deepcopy(event or EVENT), {**ENV, **(env or {})}, sleeps.append
    )
    return control, gh, sleeps


def permission(gh, login="author", value="read"):
    gh.add("GET", ROOT + f"/collaborators/{login}/permission", {"permission": value})


def review(ident, login, state):
    return {"id": ident, "user": {"login": login}, "state": state}


@pytest.mark.parametrize(
    "body",
    [
        "/ci run",
        "/ci run --allow-stale",
        "/ci retry",
        "/ci cancel",
        " \n/ci run\t ",
    ],
)
def test_exact_commands(body):
    event = copy.deepcopy(EVENT)
    event["comment"]["body"] = body
    assert module().parse_command(event) == body.strip()


@pytest.mark.parametrize(
    "body",
    [
        "/ci",
        "/ci RUN",
        "/ci run extra",
        "/ci  run",
        "/ci\trun",
        "/ci run\n/ci cancel",
        "/ci\nrun",
        "/ci retry --allow-stale",
    ],
)
def test_malformed_commands_request_help(body):
    event = copy.deepcopy(EVENT)
    event["comment"]["body"] = body
    with pytest.raises(module().Rejected) as exc:
        module().parse_command(event)
    for command in ("/ci run", "/ci run --allow-stale", "/ci retry", "/ci cancel"):
        assert command in str(exc.value)


@pytest.mark.parametrize(
    "change",
    [
        {"comment": {**EVENT["comment"], "body": "hello /ci run"}},
        {"comment": {**EVENT["comment"], "body": "/circuit"}},
        {"comment": {**EVENT["comment"], "body": ""}},
        {"comment": {**EVENT["comment"], "user": BOT}},
        {"action": "edited"},
        {"issue": {"number": 7}},
    ],
)
def test_ignore_noncommands_and_ineligible_events(change):
    assert module().parse_command({**EVENT, **change}) is None


@pytest.mark.parametrize("value", ["write", "maintain", "admin"])
def test_live_write_permission_allows_drafts(value):
    control, gh, _ = make_controller()
    permission(gh, value=value)
    assert control.authorize({**PR, "draft": True}, "author") is True


@pytest.mark.parametrize("login", ["trusted", "OTHER", "ThIrD"])
def test_trusted_list_is_case_insensitive_but_not_write(login):
    control, gh, _ = make_controller()
    permission(gh, login)
    assert control.authorize({**PR, "draft": True}, login) is False


@pytest.mark.parametrize("label", ["ready", "ready-run-all-tests"])
def test_author_with_ready_label(label):
    control, gh, _ = make_controller()
    permission(gh)
    assert control.authorize({**PR, "labels": [{"name": label}]}, "author") is False


@pytest.mark.parametrize(
    "change,login",
    [
        ({"state": "closed"}, "trusted"),
        ({"base": {"repo": {"full_name": "foreign/repo"}}}, "trusted"),
        ({"draft": True}, "author"),
        ({}, "outsider"),
    ],
)
def test_authorization_universal_and_delegation_guards(change, login):
    control, gh, _ = make_controller()
    permission(gh, login)
    with pytest.raises(module().Rejected):
        control.authorize({**PR, **change}, login)


@pytest.mark.parametrize(
    "aggregate,reviews,allowed",
    [
        ("APPROVED", [review(1, "trusted", "APPROVED")], True),
        (
            "APPROVED",
            [review(1, "trusted", "APPROVED"), review(2, "TRUSTED", "COMMENTED")],
            True,
        ),
        ("APPROVED", [review(1, "writer", "APPROVED")], True),
        ("APPROVED", [review(1, "outsider", "APPROVED")], False),
        (
            "APPROVED",
            [
                review(1, "trusted", "APPROVED"),
                review(2, "TRUSTED", "CHANGES_REQUESTED"),
            ],
            False,
        ),
        (
            "APPROVED",
            [review(2, "trusted", "DISMISSED"), review(1, "TRUSTED", "APPROVED")],
            False,
        ),
        (None, [review(1, "trusted", "APPROVED")], False),
        ("CHANGES_REQUESTED", [review(1, "trusted", "APPROVED")], False),
    ],
)
def test_aggregate_and_latest_meaningful_trusted_review(aggregate, reviews, allowed):
    control, gh, _ = make_controller()
    for login in ("author", "trusted", "writer", "outsider"):
        permission(gh, login, "write" if login == "writer" else "read")
    gh.add(
        "POST",
        "/graphql",
        {"data": {"repository": {"pullRequest": {"reviewDecision": aggregate}}}},
    )
    gh.page("/pulls/7/reviews", reviews)
    pr = {**PR, "labels": []}
    if allowed:
        assert control.authorize(pr, "author") is False
    else:
        with pytest.raises(module().Rejected):
            control.authorize(pr, "author")
    call = gh.writes("/graphql")[0]
    assert call[2]["variables"] == {"owner": "t-head", "name": "vllm-sail", "number": 7}


def test_permission_404_means_none_other_errors_propagate():
    control, gh, _ = make_controller()
    path = ROOT + "/collaborators/author/permission"
    gh.add("GET", path, http_error(404), http_error(403))
    assert control.authorize(PR, "author") is False
    with pytest.raises(HTTPError, match="403"):
        control.authorize(PR, "author")


@pytest.mark.parametrize(
    "api,url",
    [
        ("https://api.github.com", "https://api.github.com/graphql"),
        ("https://enterprise.example/api/v3", "https://enterprise.example/api/graphql"),
    ],
)
def test_stdlib_rest_graphql_transport(monkeypatch, api, url):
    seen = []

    def open_request(request, timeout):
        seen.append((request, timeout))
        return io.BytesIO(b'{"data":{"ok":true}}')

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    gh = module().GitHub("secret", api)
    assert gh.graphql("query", {"x": 1}) == {"ok": True}
    request, timeout = seen[0]
    assert request.full_url == url
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout > 0
    assert json.loads(request.data) == {"query": "query", "variables": {"x": 1}}
    gh.request("GET", ROOT + "/test", params={"created": ">=2026-10-01", "page": 2})
    assert "created=%3E%3D2026-10-01&page=2" in seen[-1][0].full_url


def test_graphql_errors_fail_closed():
    gh = FakeGitHub().add("POST", "/graphql", {"errors": [{"message": "denied"}]})
    with pytest.raises(RuntimeError):
        gh.graphql("query", {})


RUNS = "/actions/workflows/ppu-pr-ci.yaml/runs"
DISPATCH = "/actions/workflows/ppu-pr-ci.yaml/dispatches"
REACTIONS = "/issues/comments/42/reactions"
COMMENTS = "/issues/7/comments"
MARKER = "<!-- ppu-ci-command:42 -->"


def run(ident=100, sha=SHA, status="completed", conclusion="failure", **extra):
    return {
        "id": ident,
        "display_title": f"PPU PR #7 @ {sha} by author (comment 42)",
        "head_sha": "c" * 40,
        "head_branch": "main",
        "path": ".github/workflows/ppu-pr-ci.yaml",
        "repository": {"full_name": REPO},
        "event": "workflow_dispatch",
        "status": status,
        "conclusion": conclusion,
        "run_attempt": 2,
        "html_url": f"https://github.example/{REPO}/actions/runs/{ident}",
        **extra,
    }


def runs(gh, *responses, page=1):
    gh.add(
        "GET",
        ROOT + RUNS,
        *({"workflow_runs": r} for r in responses),
        params={
            "event": "workflow_dispatch",
            "created": ">=2026-10-01",
            "per_page": 100,
            "page": page,
        },
    )


def command_setup(body="/ci run", login="author", value="read", existing=None):
    event = copy.deepcopy(EVENT)
    event["comment"].update(body=body, user={"login": login, "type": "User"})
    control, gh, sleeps = make_controller(event)
    gh.page(REACTIONS, []).page(COMMENTS, [])
    gh.add("POST", ROOT + REACTIONS, {})
    gh.add("POST", ROOT + COMMENTS, {"id": 60})
    gh.add("GET", ROOT + "/pulls/7", PR)
    permission(gh, login, value)
    runs(gh, existing or [])
    gh.add("GET", ROOT + f"/compare/release%2Fnext...{SHA}", {"behind_by": 0})
    gh.add("POST", ROOT + f"/statuses/{SHA}", {})
    gh.add("POST", ROOT + DISPATCH, None)
    return control, gh, sleeps


def reply(gh):
    return gh.writes(COMMENTS)[-1][2]["body"]


def states(gh):
    return [c[2]["state"] for c in gh.writes(f"/statuses/{SHA}")]


def reactions(gh):
    return [c[2]["content"] for c in gh.writes(REACTIONS)]


@pytest.mark.parametrize(
    "extra",
    [
        {"display_title": f"PPU PR #8 @ {SHA} by author (comment 42)"},
        {"display_title": f"PPU PR #7 @ {SHA.upper()} by author (comment 42)"},
        {"display_title": f"PPU PR #7 @ {SHA} by author (comment 42) suffix"},
        {"display_title": "ppu / arbitrary"},
        {"path": ".github/workflows/evil.yaml"},
        {"head_branch": "feature"},
        {"repository": {"full_name": "fork/repo"}},
    ],
)
def test_run_discovery_filters_title_path_branch_repository(extra):
    control, gh, _ = make_controller()
    runs(gh, [run(200, **extra), run(99), run(101), run(105, sha=OLD)])
    assert [r["id"] for r in control.find_runs(PR, SHA)] == [101, 99]
    assert [r["id"] for r in control.find_runs(PR)] == [105, 101, 99]
    assert all(c[1] == ROOT + RUNS for c in gh.calls)


def test_find_runs_paginates_at_most_ten_pages_and_defaults_via_repository():
    control, gh, _ = make_controller({**EVENT, "repository": {}})
    gh.add("GET", ROOT, {"default_branch": "main"})
    for page in range(1, 11):
        runs(gh, [run(page * 100 + i) for i in range(100)], page=page)
    result = control.find_runs(PR, SHA)
    assert len(result) == 1000
    assert result[0]["id"] == 1099
    assert len([c for c in gh.calls if c[1] == ROOT + RUNS]) == 10


@pytest.mark.parametrize(
    "user,content,terminal",
    [
        (BOT, "rocket", True),
        (BOT, "-1", True),
        (BOT, "eyes", False),
        (BOT, "confused", False),
        (USER, "rocket", False),
        (USER, "-1", False),
        ({"login": "github-actions[bot]", "type": "User"}, "rocket", False),
        ({"login": "other[bot]", "type": "Bot"}, "rocket", False),
    ],
)
def test_terminal_reactions_must_be_owned_by_actions_bot(user, content, terminal):
    control, gh, _ = command_setup("/ci cancel")
    gh.page(REACTIONS, [{"user": user, "content": content}])
    control.command()
    assert bool(gh.writes(COMMENTS)) is not terminal
    assert not states(gh)


@pytest.mark.parametrize(
    "user,terminal",
    [
        (BOT, True),
        (USER, False),
        ({"login": "github-actions[bot]", "type": "User"}, False),
        ({"login": "someone[bot]", "type": "Bot"}, False),
    ],
)
def test_command_marker_is_not_user_forgeable(user, terminal):
    control, gh, _ = command_setup("/ci cancel")
    gh.page(COMMENTS, [{"user": user, "body": MARKER, "id": 66}])
    control.command()
    assert bool(gh.writes(COMMENTS)) is not terminal


def test_paginated_terminal_marker_and_reaction():
    for path, entry in (
        (COMMENTS, {"user": BOT, "body": MARKER, "id": 66}),
        (REACTIONS, {"user": BOT, "content": "rocket"}),
    ):
        control, gh, _ = command_setup()
        gh.page(path, [{"user": USER, "body": "hello", "content": "eyes"}] * 100)
        gh.page(path, [entry], page=2)
        control.command()
        assert not gh.writes(COMMENTS)


def test_malformed_untrusted_comment_gets_marked_help_without_authorizing():
    control, gh, _ = command_setup("/ci launch", "outsider")
    control.command()
    assert MARKER in reply(gh)
    assert "/ci run --allow-stale" in reply(gh)
    assert reactions(gh) == ["eyes", "-1"]
    assert not any("/collaborators/" in c[1] for c in gh.calls)


def test_untrusted_command_denial_explains_authorization():
    control, gh, _ = command_setup("/ci run", "outsider")
    control.command()
    assert "authorized" in reply(gh)
    assert MARKER in reply(gh)
    assert reactions(gh) == ["eyes", "-1"]


def test_unrelated_comment_does_not_call_github():
    control, gh, _ = command_setup("hello")
    control.command()
    assert gh.calls == []


@pytest.mark.parametrize(
    "status", ["queued", "in_progress", "pending", "waiting", "requested"]
)
def test_run_already_active(status):
    control, gh, _ = command_setup(existing=[run(status=status, conclusion=None)])
    control.command()
    assert "already running" in reply(gh).lower()
    assert run()["html_url"] in reply(gh)
    assert not states(gh) and not gh.writes(DISPATCH)


@pytest.mark.parametrize("login", ["author", "trusted"])
def test_completed_run_blocks_nonwriters_including_trusted_list(login):
    control, gh, _ = command_setup(login=login, existing=[run()])
    control.command()
    assert "already tested" in reply(gh).lower()
    assert run()["html_url"] in reply(gh)
    assert "/ci retry" in reply(gh)
    assert "new commit" in reply(gh)
    assert reactions(gh) == ["eyes", "-1"]
    assert not gh.writes(DISPATCH)
    assert len([c for c in gh.calls if c[1] == ROOT + "/pulls/7"]) == 2


def test_newly_granted_write_permission_can_retest_completed_head():
    control, gh, _ = command_setup(existing=[run()])
    gh.add(
        "GET",
        ROOT + "/collaborators/author/permission",
        {"permission": "read"},
        {"permission": "write"},
    )
    control.command()
    assert gh.writes(DISPATCH)
    assert states(gh) == ["pending"]
    assert reactions(gh)[-1] == "rocket"


def test_successful_dispatch_uses_default_ref_and_exact_inputs():
    control, gh, _ = command_setup(login="maintainer", value="write", existing=[run()])
    control.command()
    dispatch = gh.writes(DISPATCH)[0]
    assert dispatch[2] == {
        "ref": "main",
        "inputs": {
            "pr_number": "7",
            "head_sha": SHA,
            "requester": "maintainer",
            "comment_id": "42",
        },
    }
    status = gh.writes(f"/statuses/{SHA}")[0][2]
    assert status == {
        "state": "pending",
        "context": "PPU CI",
        "description": "Queued by @maintainer",
        "target_url": f"https://github.example/{REPO}/actions/runs/900/attempts/2",
    }
    assert gh.calls.index(gh.writes(f"/statuses/{SHA}")[0]) < gh.calls.index(dispatch)
    assert "PPU CI queued for " + SHA[:12] in reply(gh)
    assert run()["html_url"] in reply(gh)
    assert reactions(gh) == ["eyes", "rocket"]


def test_dispatch_poll_requires_exact_comment_id_and_accepts_timeout():
    control, gh, sleeps = command_setup()
    foreign = run(display_title=f"PPU PR #7 @ {SHA} by author (comment 43)")
    runs(gh, [], [foreign])
    control.command()
    assert "accepted" in reply(gh).lower()
    assert "/actions/workflows/ppu-pr-ci.yaml" in reply(gh)
    assert "pending" in reply(gh).lower()
    assert sum(sleeps) == 30 and set(sleeps) == {3}
    assert reactions(gh)[-1] == "rocket"
    assert states(gh) == ["pending"]


def test_dispatch_poll_finds_run_after_delayed_visibility():
    control, gh, sleeps = command_setup()
    runs(gh, [], [], [run(status="queued")])
    control.command()
    assert "queued for " + SHA[:12] in reply(gh)
    assert run()["html_url"] in reply(gh)
    assert sum(sleeps) <= 30


@pytest.mark.parametrize("allow", [False, True])
def test_stale_head_requires_explicit_override_and_warns(allow):
    body = "/ci run --allow-stale" if allow else "/ci run"
    control, gh, _ = command_setup(body)
    gh.add("GET", ROOT + f"/compare/release%2Fnext...{SHA}", {"behind_by": 3})
    control.command()
    assert bool(gh.writes(DISPATCH)) is allow
    assert "behind" in reply(gh).lower()
    assert "--allow-stale" in reply(gh) if not allow else "warning" in reply(gh).lower()


@pytest.mark.parametrize(
    "change",
    [
        {"head": {"sha": OLD}},
        {"state": "closed"},
        {"base": {"repo": {"full_name": "other/repo"}}},
        {"draft": True},
    ],
)
def test_dispatch_rechecks_head_and_live_authorization(change):
    control, gh, _ = command_setup()
    gh.add("GET", ROOT + "/pulls/7", PR, {**PR, **change})
    control.command()
    assert not gh.writes(DISPATCH) and not states(gh)
    assert reactions(gh)[-1] == "-1"


@pytest.mark.parametrize("code", [403, 422, 500])
def test_dispatch_http_failure_clears_pending_and_replies(code):
    control, gh, _ = command_setup()
    gh.add("POST", ROOT + DISPATCH, http_error(code))
    control.command()
    assert states(gh) == ["pending", "error"]
    assert "failed" in reply(gh).lower()
    assert reactions(gh)[-1] == "-1"


def test_unexpected_exception_reacts_confused_and_propagates():
    control, gh, _ = command_setup()
    gh.add("GET", ROOT + "/pulls/7", RuntimeError("unavailable"))
    with pytest.raises(RuntimeError, match="unavailable"):
        control.command()
    assert reactions(gh) == ["eyes", "confused"]
    assert not gh.writes(COMMENTS)


def test_reaction_api_failures_are_best_effort():
    control, gh, _ = command_setup("/ci cancel")
    gh.add(
        "GET", ROOT + REACTIONS, http_error(403), params={"per_page": 100, "page": 1}
    )
    gh.add("POST", ROOT + REACTIONS, http_error(403))
    control.command()
    assert "none" in reply(gh).lower()


@pytest.mark.parametrize(
    "status", ["queued", "in_progress", "pending", "waiting", "requested"]
)
def test_cancel_all_active_heads_writes_terminal_status_on_each_title_sha(status):
    control, gh, _ = command_setup(
        "/ci cancel",
        login="maintainer",
        value="write",
        existing=[
            run(101, status=status, conclusion=None),
            run(102, sha=OLD, status=status, conclusion=None),
            run(103),
        ],
    )
    gh.add("POST", ROOT + f"/statuses/{OLD}", {})
    for ident in (101, 102):
        gh.add("POST", ROOT + f"/actions/runs/{ident}/cancel", None)
    control.command()
    assert len(gh.writes("/cancel")) == 2
    assert "2" in reply(gh)
    assert run(101)["html_url"] in reply(gh) and run(102)["html_url"] in reply(gh)
    statuses = [c for c in gh.calls if "/statuses/" in c[1]]
    assert {c[1] for c in statuses} == {
        ROOT + f"/statuses/{SHA}",
        ROOT + f"/statuses/{OLD}",
    }
    assert len(statuses) == 2
    for ident, sha in ((101, SHA), (102, OLD)):
        status = gh.writes(f"/statuses/{sha}")[0]
        assert status[2] == {
            "state": "error",
            "context": "PPU CI",
            "description": "Cancelled by @maintainer",
            "target_url": control.execution_url,
        }
        cancel = gh.writes(f"/actions/runs/{ident}/cancel")[0]
        assert gh.calls.index(cancel) < gh.calls.index(status)


@pytest.mark.parametrize("code", [403, 409, 500])
def test_rejected_cancellation_does_not_write_cancelled_status(code):
    control, gh, _ = command_setup("/ci cancel", existing=[run(status="queued")])
    gh.add("POST", ROOT + "/actions/runs/100/cancel", http_error(code))
    with pytest.raises(HTTPError):
        control.command()
    assert not states(gh)
    assert reactions(gh)[-1] == "confused"


def test_cancellation_status_failure_is_not_reported_as_success():
    control, gh, _ = command_setup("/ci cancel", existing=[run(status="queued")])
    gh.add("POST", ROOT + "/actions/runs/100/cancel", None)
    gh.add("POST", ROOT + f"/statuses/{SHA}", http_error(403))
    with pytest.raises(HTTPError):
        control.command()
    assert states(gh) == ["error"]
    assert not gh.writes(COMMENTS)
    assert reactions(gh)[-1] == "confused"


def retry_setup(existing=None, jobs=None, artifacts=None):
    control, gh, sleeps = command_setup("/ci retry", existing=existing or [run()])
    gh.page("/actions/runs/100/jobs", jobs or [], "jobs", filter="latest")
    gh.page("/actions/runs/100/artifacts", artifacts or [], "artifacts")
    gh.add("POST", ROOT + "/actions/runs/100/rerun-failed-jobs", None)
    gh.add("POST", ROOT + "/actions/runs/100/rerun", None)
    return control, gh, sleeps


@pytest.mark.parametrize(
    "existing,hint",
    [
        ([], "/ci run"),
        ([run(sha=OLD)], "/ci run"),
        ([run(conclusion="success")], "nothing to rerun"),
    ],
)
def test_retry_absent_or_successful_current_head(existing, hint):
    control, gh, _ = command_setup("/ci retry", existing=existing)
    control.command()
    assert hint in reply(gh).lower()
    assert not states(gh)
    assert not gh.writes(DISPATCH)


@pytest.mark.parametrize(
    "status", ["queued", "in_progress", "pending", "waiting", "requested"]
)
def test_retry_active_run_does_not_rerun_or_change_status(status):
    control, gh, _ = retry_setup(existing=[run(status=status, conclusion=None)])
    control.command()
    assert "already running" in reply(gh).lower()
    assert run()["html_url"] in reply(gh)
    assert not states(gh)
    assert not gh.writes("/rerun-failed-jobs")
    assert not gh.writes("/rerun")
    assert not gh.writes(DISPATCH)


@pytest.mark.parametrize(
    "jobs,artifacts,allowed",
    [
        ([], [], True),
        ([{"name": "PPU wheel build", "conclusion": "failure"}], [], True),
        ([{"name": "PPU wheel build", "conclusion": "success"}], [], False),
        ([{"name": "ppu / PPU wheel build", "conclusion": "success"}], [], False),
        (
            [{"name": "ppu / PPU wheel build", "conclusion": "success"}],
            [{"name": "ppu-wheels-100", "expired": True}],
            False,
        ),
        (
            [{"name": "ppu / PPU wheel build", "conclusion": "success"}],
            [{"name": "ppu-wheels-100", "expired": False}],
            True,
        ),
        (
            [{"name": "PPU wheel build", "conclusion": "success"}],
            [{"name": "ppu-wheels-99", "expired": False}],
            False,
        ),
    ],
)
def test_retry_failed_run_requires_preserved_successful_build_artifact(
    jobs, artifacts, allowed
):
    control, gh, _ = retry_setup(jobs=jobs, artifacts=artifacts)
    control.command()
    assert bool(gh.writes("/rerun-failed-jobs")) is allowed
    assert not any("/compare/" in c[1] for c in gh.calls)
    if allowed:
        assert states(gh) == ["pending"]
        assert (
            gh.writes(f"/statuses/{SHA}")[0][2]["description"]
            == "Retry requested by @author"
        )
        assert run()["html_url"] + "/attempts/3" in reply(gh)
    else:
        assert "/ci run" in reply(gh)
        assert not states(gh)


@pytest.mark.parametrize(
    "code,fallback", [(400, True), (403, True), (422, True), (500, False), (503, False)]
)
def test_retry_falls_back_only_on_client_errors(code, fallback):
    control, gh, _ = retry_setup()
    gh.add("POST", ROOT + "/actions/runs/100/rerun-failed-jobs", http_error(code))
    control.command()
    assert bool(gh.writes("/rerun")) is fallback
    assert states(gh) == (["pending"] if fallback else ["pending", "error"])
    assert reactions(gh)[-1] == ("rocket" if fallback else "-1")


def test_retry_rejected_fallback_clears_pending():
    control, gh, _ = retry_setup()
    gh.add("POST", ROOT + "/actions/runs/100/rerun-failed-jobs", http_error(422))
    gh.add("POST", ROOT + "/actions/runs/100/rerun", http_error(403))
    control.command()
    assert states(gh) == ["pending", "error"]
    assert "failed" in reply(gh).lower()


def test_retry_uses_newest_run_and_paginates_jobs_and_artifacts():
    control, gh, _ = retry_setup(existing=[run(99), run(100)])
    gh.page(
        "/actions/runs/100/jobs",
        [{"name": "irrelevant", "conclusion": "success"}] * 100,
        "jobs",
        filter="latest",
    )
    gh.page(
        "/actions/runs/100/jobs",
        [{"name": "ppu / PPU wheel build", "conclusion": "success"}],
        "jobs",
        page=2,
        filter="latest",
    )
    gh.page(
        "/actions/runs/100/artifacts",
        [{"name": "wrong", "expired": False}] * 100,
        "artifacts",
    )
    gh.page(
        "/actions/runs/100/artifacts",
        [{"name": "ppu-wheels-100", "expired": False}],
        "artifacts",
        page=2,
    )
    control.command()
    assert len(gh.writes("/rerun-failed-jobs")) == 1
    assert not any("/actions/runs/99/" in c[1] for c in gh.calls)


INPUTS = {"pr_number": "7", "head_sha": SHA, "requester": "author", "comment_id": "42"}


def executor_setup(inputs=None, needs=None):
    control, gh, sleeps = make_controller(
        {"inputs": INPUTS if inputs is None else inputs},
        {
            "NEEDS": json.dumps(
                needs
                or {"validate": {"result": "success"}, "ppu": {"result": "success"}}
            ),
        },
    )
    gh.add("GET", ROOT + "/pulls/7", PR)
    gh.page(
        "/pulls/7/files",
        [{"filename": "vllm_sail/platform.py", "status": "modified"}],
    )
    permission(gh)
    gh.add("POST", ROOT + f"/statuses/{SHA}", {})
    gh.page(COMMENTS, [])
    gh.add("POST", ROOT + COMMENTS, {})
    gh.add("GET", ROOT, {"default_branch": "main"})
    gh.page(RUNS, [], "workflow_runs", event="workflow_dispatch", status="success")
    return control, gh, sleeps


@pytest.mark.parametrize(
    "field,value",
    [
        ("pr_number", "0"),
        ("pr_number", "-1"),
        ("pr_number", 7),
        ("pr_number", "7/../../issues"),
        ("pr_number", "7\n"),
        ("head_sha", SHA.upper()),
        ("head_sha", "a" * 39),
        ("head_sha", SHA + "\n"),
        ("comment_id", "0"),
        ("comment_id", "42?evil"),
        ("requester", "../../bad"),
        ("requester", "-bad"),
        ("requester", "x" * 40),
        ("requester", "user\n"),
        ("requester", ""),
    ],
)
@pytest.mark.parametrize("mode", ["validate", "report"])
def test_executor_rejects_malformed_inputs_before_any_api(field, value, mode):
    control, gh, _ = executor_setup({**INPUTS, field: value})
    with pytest.raises(module().Rejected):
        getattr(control, mode)()
    assert gh.calls == []


def test_validate_sets_running_for_expected_sha_and_attempt():
    control, gh, _ = executor_setup()
    control.validate()
    status = gh.writes(f"/statuses/{SHA}")[0][2]
    assert status["state"] == "pending"
    assert status["context"] == "PPU CI"
    assert status["description"] == "Running (attempt 2)"
    assert status["target_url"].endswith("/900/attempts/2")
    assert all(c[0] == "GET" or "/statuses/" in c[1] for c in gh.calls)


@pytest.mark.parametrize(
    "change",
    [
        {"head": {"sha": OLD}},
        {"state": "closed"},
        {"draft": True},
        {"base": {"repo": {"full_name": "wrong/repo"}}},
        {"labels": []},
    ],
)
def test_validate_fails_for_changed_head_closed_or_revoked_authorization(change):
    control, gh, _ = executor_setup()
    gh.add("GET", ROOT + "/pulls/7", {**PR, **change})
    gh.add(
        "POST",
        "/graphql",
        {"data": {"repository": {"pullRequest": {"reviewDecision": None}}}},
    )
    with pytest.raises(module().Rejected):
        control.validate()
    assert not states(gh)


@pytest.mark.parametrize(
    "validate,ppu,state,description",
    [
        ("success", "success", "success", "Passed"),
        ("failure", "skipped", "failure", "validate"),
        ("success", "failure", "failure", "ppu"),
        ("success", "skipped", "failure", "ppu"),
        ("cancelled", "skipped", "error", "Cancelled"),
        ("failure", "cancelled", "error", "Cancelled"),
        ("skipped", "success", "failure", "validate"),
    ],
)
def test_report_mapping_and_finishes_old_sha_without_reauthorization(
    validate, ppu, state, description
):
    control, gh, _ = executor_setup(
        needs={"validate": {"result": validate}, "ppu": {"result": ppu}}
    )
    control.report()
    status = gh.writes(f"/statuses/{SHA}")[0][2]
    assert status["state"] == state
    assert status["context"] == "PPU CI"
    assert description in status["description"] and "attempt 2" in status["description"]
    assert status["target_url"].endswith("/900/attempts/2")
    assert MARKER in reply(gh)
    assert status["description"] in reply(gh) and status["target_url"] in reply(gh)
    assert not any("/pulls/" in c[1] or "/collaborators/" in c[1] for c in gh.calls)


def test_report_marks_docs_only_execution_as_skipped_success():
    control, gh, _ = executor_setup(
        needs={
            "validate": {"result": "success", "outputs": {"skip_ppu": "true"}},
            "ppu": {"result": "success"},
        }
    )
    control.report()
    status = gh.writes(f"/statuses/{SHA}")[0][2]
    assert status["state"] == "success"
    assert status["description"] == ("Skipped (no PPU-impacting changes) (attempt 2)")
    assert status["description"] in reply(gh)


@pytest.mark.parametrize("owner,patch", [(BOT, True), (USER, False)])
def test_report_updates_only_bot_owned_command_comment(owner, patch):
    control, gh, _ = executor_setup()
    gh.page(COMMENTS, [{"id": 66, "body": MARKER, "user": owner}])
    gh.add("PATCH", ROOT + "/issues/comments/66", {})
    control.report()
    assert bool(gh.writes("/issues/comments/66")) is patch
    assert bool(gh.writes(COMMENTS)) is not patch


@pytest.mark.parametrize("where", ["list", "post", "patch"])
def test_report_comment_failure_is_best_effort_after_status(where):
    control, gh, _ = executor_setup()
    if where == "list":
        gh.add(
            "GET", ROOT + COMMENTS, http_error(403), params={"per_page": 100, "page": 1}
        )
    elif where == "post":
        gh.add("POST", ROOT + COMMENTS, http_error(403))
    else:
        gh.page(COMMENTS, [{"id": 66, "body": MARKER, "user": BOT}])
        gh.add("PATCH", ROOT + "/issues/comments/66", http_error(403))
    control.report()
    assert states(gh) == ["success"]


def test_report_status_error_must_fail_job():
    control, gh, _ = executor_setup()
    gh.add("POST", ROOT + f"/statuses/{SHA}", http_error(403))
    with pytest.raises(HTTPError):
        control.report()
    assert not gh.writes(COMMENTS)


@pytest.mark.parametrize(
    "change,notify",
    [
        ({}, True),
        ({"action": "unlabeled"}, False),
        ({"label": {"name": "other"}}, False),
        ({"label": {"name": "ready-run-all-tests"}}, True),
        ({"pull_request": {**PR, "state": "closed"}}, False),
        ({"pull_request": {**PR, "draft": True}}, False),
        ({"pull_request": {**PR, "user": {"login": "TRUSTED"}}}, False),
    ],
)
def test_notify_filters_event_author_and_labels(change, notify):
    event = {
        "action": "labeled",
        "label": {"name": "ready"},
        "pull_request": PR,
        **change,
    }
    control, gh, _ = make_controller(event)
    permission(gh)
    permission(gh, "TRUSTED")
    gh.page(COMMENTS, [])
    gh.add("POST", ROOT + COMMENTS, {})
    control.notify()
    assert bool(gh.writes(COMMENTS)) is notify
    if notify:
        text = reply(gh)
        for command in module().COMMANDS:
            assert command in text
        assert "<!-- ppu-ci-authorized -->" in text
        assert "new commits" in text.lower()
        assert "CPU" in text and "automatic" in text and "behind" in text


@pytest.mark.parametrize("user,sent", [(BOT, False), (USER, True)])
def test_notify_bot_marker_deduplication(user, sent):
    control, gh, _ = make_controller(
        {"action": "labeled", "label": {"name": "ready"}, "pull_request": PR}
    )
    permission(gh)
    gh.page(COMMENTS, [{"user": user, "body": "<!-- ppu-ci-authorized -->"}])
    gh.add("POST", ROOT + COMMENTS, {})
    control.notify()
    assert bool(gh.writes(COMMENTS)) is sent


def test_notify_skips_actual_writer():
    control, gh, _ = make_controller(
        {"action": "labeled", "label": {"name": "ready"}, "pull_request": PR}
    )
    permission(gh, value="write")
    control.notify()
    assert not gh.writes(COMMENTS)


def test_cli_reads_event_inputs_and_reports_validation_failure(tmp_path):
    event_path = tmp_path / "event.json"
    summary = tmp_path / "summary.txt"
    event_path.write_text(json.dumps({"inputs": {**INPUTS, "head_sha": "bad"}}))
    env = {
        **ENV,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    assert module().main(["validate"], env=env, github=FakeGitHub()) == 1
    assert "validation failed" in summary.read_text().lower()
    assert "head_sha" in summary.read_text()


@pytest.mark.parametrize("mode", ["validate", "report", "command", "notify"])
def test_cli_modes_use_injected_client_and_environment(tmp_path, mode):
    if mode in {"validate", "report"}:
        control, gh, _ = executor_setup()
    else:
        control, gh, _ = make_controller({"action": "ignored"})
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(control.event))
    env = {**control.env, "GITHUB_EVENT_PATH": str(event_path)}
    assert module().main([mode], env=env, github=gh, sleep=lambda _: None) == 0


def test_cli_module_starts_without_torch_or_vllm(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    event_path = tmp_path / "event.json"
    event_path.write_text('{"action":"ignored"}')
    code = (
        "import runpy,sys; "
        "sys.modules.update({k:None for k in ('torch','vllm','vllm_sail')}); "
        "sys.argv=['ppu_ci_command','command']; "
        "runpy.run_module('scripts.ci.ppu_ci_command',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            **ENV,
            "GITHUB_TOKEN": "fixture",
            "GITHUB_EVENT_PATH": str(event_path),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_review_latest_is_submission_time_not_review_creation_id():
    control, gh, _ = make_controller()
    permission(gh)
    gh.add(
        "POST",
        "/graphql",
        {"data": {"repository": {"pullRequest": {"reviewDecision": "APPROVED"}}}},
    )
    gh.page(
        "/pulls/7/reviews",
        [
            {
                **review(1, "trusted", "CHANGES_REQUESTED"),
                "submitted_at": "2026-10-03T12:00:00Z",
            },
            {
                **review(2, "trusted", "APPROVED"),
                "submitted_at": "2026-10-02T12:00:00Z",
            },
        ],
    )
    with pytest.raises(module().Rejected):
        control.authorize({**PR, "labels": []}, "author")


def test_dispatch_poll_stops_after_slow_lookup_and_still_accepts(monkeypatch):
    control, gh, sleeps = command_setup()
    clock = iter([0, 0, 31])
    monkeypatch.setattr(module().time, "monotonic", lambda: next(clock))
    control.command()
    assert sleeps == [3]
    assert "accepted" in reply(gh).lower()
    assert reactions(gh)[-1] == "rocket"


def test_review_approval_and_comment_are_paginated():
    control, gh, _ = make_controller()
    permission(gh)
    gh.add(
        "POST",
        "/graphql",
        {"data": {"repository": {"pullRequest": {"reviewDecision": "APPROVED"}}}},
    )
    gh.page("/pulls/7/reviews", [review(i, "trusted", "COMMENTED") for i in range(100)])
    gh.page("/pulls/7/reviews", [review(101, "trusted", "APPROVED")], page=2)
    assert control.authorize({**PR, "labels": []}, "author") is False
    assert any(c[3].get("page") == 2 for c in gh.calls)


def test_permission_revoked_between_run_checks_cannot_retest_completed_head():
    control, gh, _ = command_setup(value="write", existing=[run()])
    gh.add(
        "GET",
        ROOT + "/collaborators/author/permission",
        {"permission": "write"},
        {"permission": "read"},
    )
    control.command()
    assert "write permission" in reply(gh)
    assert not gh.writes(DISPATCH)


def test_accepted_dispatch_lookup_failure_does_not_clear_pending():
    control, gh, _ = command_setup()
    gh.add(
        "GET",
        ROOT + RUNS,
        {"workflow_runs": []},
        http_error(503),
        params={
            "event": "workflow_dispatch",
            "created": ">=2026-10-01",
            "per_page": 100,
            "page": 1,
        },
    )
    control.command()
    assert "accepted" in reply(gh).lower()
    assert states(gh) == ["pending"]
    assert reactions(gh)[-1] == "rocket"


def test_dispatch_network_failure_marks_error_and_reraises():
    control, gh, _ = command_setup()
    gh.add("POST", ROOT + DISPATCH, OSError("network offline"))
    with pytest.raises(OSError, match="network offline"):
        control.command()
    assert states(gh) == ["pending", "error"]
    assert reactions(gh)[-1] == "confused"


def test_pending_status_failure_prevents_dispatch():
    control, gh, _ = command_setup()
    gh.add("POST", ROOT + f"/statuses/{SHA}", http_error(403))
    with pytest.raises(HTTPError):
        control.command()
    assert not gh.writes(DISPATCH)
    assert reactions(gh)[-1] == "confused"


def test_reply_failure_is_unexpected_and_retriable():
    control, gh, _ = command_setup("/ci cancel")
    gh.add("POST", ROOT + COMMENTS, http_error(503))
    with pytest.raises(HTTPError):
        control.command()
    assert reactions(gh) == ["eyes", "confused"]


def test_report_uses_older_input_sha_even_when_pr_has_moved():
    control, gh, _ = executor_setup({**INPUTS, "head_sha": OLD})
    gh.add("POST", ROOT + f"/statuses/{OLD}", {})
    control.report()
    assert gh.writes(f"/statuses/{OLD}")[0][2]["state"] == "success"
    assert not states(gh)


@pytest.mark.parametrize("field", list(INPUTS))
def test_missing_inputs_rejected_without_api(field):
    inputs = {k: v for k, v in INPUTS.items() if k != field}
    control, gh, _ = executor_setup(inputs)
    with pytest.raises(module().Rejected, match=field):
        control.validate()
    assert gh.calls == []


@pytest.mark.parametrize("needs", ["not json", "[]", "null", '{"ppu": null}'])
def test_report_malformed_needs_fails_without_success_status(needs):
    control, gh, _ = executor_setup()
    control.env["NEEDS"] = needs
    with pytest.raises((ValueError, AttributeError)):
        control.report()
    assert not states(gh)


def test_client_empty_response_and_http_errors(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b""))
    gh = module().GitHub("fixture")
    assert gh.request("POST", "/dispatch", {}) is None

    def fail(*args, **kwargs):
        raise http_error(500)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(HTTPError):
        gh.request("POST", "/dispatch", {})


def test_cli_uses_env_transport_and_propagates_report_status_error(
    tmp_path, monkeypatch
):
    control, gh, _ = executor_setup()
    path = tmp_path / "event.json"
    path.write_text(json.dumps(control.event))
    constructed = []

    def client(token, api):
        constructed.append((token, api))
        return gh

    monkeypatch.setattr(module(), "GitHub", client)
    gh.add("POST", ROOT + f"/statuses/{SHA}", http_error(403))
    env = {
        **control.env,
        "GITHUB_TOKEN": "fixture",
        "GITHUB_API_URL": "https://host/api/v3",
        "GITHUB_EVENT_PATH": str(path),
    }
    with pytest.raises(HTTPError):
        module().main(["report"], env=env)
    assert constructed == [("fixture", "https://host/api/v3")]


def test_cli_real_module_validation_exit_and_summary_without_device_imports(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    path, summary = tmp_path / "event.json", tmp_path / "summary.txt"
    path.write_text(json.dumps({"inputs": {**INPUTS, "head_sha": "bad"}}))
    code = (
        "import runpy,sys; "
        "sys.modules.update({k:None for k in ('torch','vllm','vllm_sail')}); "
        "sys.argv=['ppu_ci_command','validate']; "
        "runpy.run_module('scripts.ci.ppu_ci_command',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            **ENV,
            "GITHUB_TOKEN": "fixture",
            "GITHUB_EVENT_PATH": str(path),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "head_sha" in summary.read_text()
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ci.ppu_ci_command", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0 and "command,notify,validate,report" in result.stdout


def test_native_retry_can_reuse_successful_validate_and_report_new_attempt():
    jobs = [
        {"name": "validate", "conclusion": "success"},
        {"name": "ppu / PPU wheel build", "conclusion": "success"},
        {"name": "ppu / PPU operator tests", "conclusion": "failure"},
    ]
    control, gh, _ = retry_setup(
        jobs=jobs, artifacts=[{"name": "ppu-wheels-100", "expired": False}]
    )
    control.command()
    assert len(gh.writes("/rerun-failed-jobs")) == 1
    assert not gh.writes("/rerun")
    # Model reused validate success; real dependency scheduling needs remote proof.
    control.event = {"inputs": INPUTS}
    control.env.update(
        GITHUB_RUN_ID="100",
        GITHUB_RUN_ATTEMPT="3",
        NEEDS=json.dumps(
            {"validate": {"result": "success"}, "ppu": {"result": "success"}}
        ),
    )
    calls_before_report = len(gh.calls)
    control.report()
    assert states(gh) == ["pending", "success"]
    assert "attempt 3" in reply(gh)
    assert run()["html_url"] + "/attempts/3" in reply(gh)
    assert not any("/pulls/" in c[1] for c in gh.calls[calls_before_report:])


def test_author_association_never_grants_authority():
    control, gh, _ = command_setup(login="outsider")
    control.event["comment"]["author_association"] = "OWNER"
    control.command()
    assert reactions(gh)[-1] == "-1"
    assert not gh.writes(DISPATCH)


def test_stdlib_only_python310_syntax_and_physical_line_budget():
    import ast
    import sys
    from pathlib import Path

    source = Path(module().__file__).read_text()
    assert len(source.splitlines()) <= 650
    tree = ast.parse(source, feature_version=(3, 10))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] in sys.stdlib_module_names
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in sys.stdlib_module_names


@pytest.mark.parametrize(
    "files",
    [
        [{"filename": "docs/guide.md", "status": "modified"}],
        [{"filename": "README.md", "status": "modified"}],
        [{"filename": ".github/ISSUE_TEMPLATE/bug_report.yml", "status": "modified"}],
        [
            {
                "filename": "docs/new.md",
                "previous_filename": "docs/old.md",
                "status": "renamed",
            }
        ],
    ],
)
def test_validate_skips_ppu_for_non_executable_changes(files, github_output, capsys):
    control, gh, _ = executor_setup()
    gh.add("GET", ROOT + "/pulls/7", {**PR, "changed_files": len(files)})
    gh.page("/pulls/7/files", files)
    control.validate()
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs == {
        "skip_ppu": "true",
        "existing_build_run_id": "",
        "native_reuse_run_id": "",
        "wheel_source_sha": "",
    }
    assert not any(
        c[1] == ROOT + RUNS or "/artifacts" in c[1] or "/compare/" in c[1]
        for c in gh.calls
    )
    assert "mode=skip" in capsys.readouterr().out


@pytest.mark.parametrize(
    "files",
    [
        [
            {"filename": "docs/guide.md", "status": "modified"},
            {"filename": "vllm_sail/platform.py", "status": "modified"},
        ],
        [{"filename": ".github/workflows/ut.yaml", "status": "modified"}],
        [{"filename": "tests/e2e/test_model.py", "status": "modified"}],
        [{"filename": "MANIFEST.in", "status": "modified"}],
        [{"filename": "requirements/ppu.txt", "status": "modified"}],
        [{"filename": "docs/new.md", "status": "renamed"}],
    ],
)
def test_validate_does_not_skip_ppu_for_executable_or_incomplete_changes(
    files, github_output
):
    control, gh, _ = executor_setup()
    gh.add("GET", ROOT + "/pulls/7", {**PR, "changed_files": len(files)})
    gh.page("/pulls/7/files", files)
    control.validate()
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs["skip_ppu"] == "false"


@pytest.mark.parametrize(
    "changed,files",
    [
        (0, []),
        (301, []),
        (2, [{"filename": "docs/guide.md", "status": "modified"}]),
    ],
)
def test_skip_detection_fails_closed_for_incomplete_file_inventory(
    changed, files, github_output
):
    control, gh, _ = executor_setup()
    gh.add("GET", ROOT + "/pulls/7", {**PR, "changed_files": changed})
    if 0 < changed <= 300:
        gh.page("/pulls/7/files", files)
    control.validate()
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs["skip_ppu"] == "false"


def test_skip_detection_api_failure_runs_ppu_instead(github_output, capsys):
    control, gh, _ = executor_setup()
    gh.page("/pulls/7/files", http_error(503))
    control.validate()
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs["skip_ppu"] == "false"
    assert "503" in capsys.readouterr().err


# Reuse is a trusted executor optimization, never a command/dispatch decision.


def reuse_setup(files=None, candidates=None, compare=None):
    control, gh, _ = executor_setup()
    candidates = (
        candidates if candidates is not None else [run(555, OLD, conclusion="success")]
    )
    gh.page(
        RUNS, candidates, "workflow_runs", event="workflow_dispatch", status="success"
    )
    for candidate in candidates:
        gh.page(
            f"/actions/runs/{candidate['id']}/artifacts",
            [{"name": f"ppu-wheels-{candidate['id']}", "expired": False}],
            "artifacts",
        )
    gh.add(
        "GET",
        ROOT + f"/compare/{OLD}...{SHA}",
        compare
        if compare is not None
        else {
            "status": "ahead",
            "merge_base_commit": {"sha": OLD},
            "files": files if files is not None else [{"filename": "docs/guide.rst"}],
        },
    )
    return control, gh


def validate_pair(control, gh, github_output):
    control.validate()
    assert states(gh) == ["pending"]
    assert len([c for c in gh.calls if c[1].endswith("/pulls/7/files")]) == 1
    assert not any(
        "/pulls/" in c[1] and not c[1].endswith(("/pulls/7", "/pulls/7/files"))
        for c in gh.calls
    )
    assert all(c[0] == "GET" or "/statuses/" in c[1] for c in gh.calls)
    assert github_output.exists(), "validate must emit the trusted reuse pair"
    return dict(line.split("=", 1) for line in github_output.read_text().splitlines())


def test_command_does_not_inspect_files_or_choose_reuse():
    control, gh, _ = command_setup(value="write", existing=[run()])
    control.command()
    assert gh.writes(DISPATCH)[0][2]["inputs"] == INPUTS
    assert not any(
        "/files" in c[1] or c[3].get("status") == "success" for c in gh.calls
    )
    assert "reused" not in reply(gh).lower()


@pytest.mark.parametrize(
    "extra",
    [
        {"skip_ppu": "true"},
        {"existing_build_run_id": "555"},
        {"native_reuse_run_id": "555"},
        {"wheel_source_sha": OLD},
        {"unexpected": ""},
    ],
)
@pytest.mark.parametrize("mode", ["validate", "report"])
def test_executor_rejects_forged_extra_inputs_before_apis(extra, mode, github_output):
    control, gh, _ = executor_setup({**INPUTS, **extra})
    with pytest.raises(module().Rejected):
        getattr(control, mode)()
    assert gh.calls == []
    assert not github_output.exists()


def reuse_result(mode, run_id="555", source=OLD):
    return {
        "skip_ppu": "false",
        "existing_build_run_id": run_id if mode == "full" else "",
        "native_reuse_run_id": run_id if mode == "native" else "",
        "wheel_source_sha": source if mode != "none" else "",
    }


@pytest.mark.parametrize(
    "files",
    [
        [],
        [{"filename": "docs/guide.rst"}],
        [{"filename": "tests/ut/test.py"}],
        [{"filename": ".github/workflows/build-ppu-wheels.yaml"}],
        [{"filename": "scripts/ci/ppu_ci_command.py"}],
        [
            {
                "filename": "docs/new",
                "previous_filename": "tests/old",
                "status": "renamed",
            }
        ],
    ],
)
def test_validate_reuses_complete_wheels_for_non_package_changes(
    files, github_output, capsys
):
    control, gh = reuse_setup(files=files)
    assert validate_pair(control, gh, github_output) == reuse_result("full")
    calls = [c[1] for c in gh.calls]
    assert ROOT + f"/compare/{OLD}...{SHA}" in calls
    assert calls.index(ROOT + "/collaborators/author/permission") < calls.index(
        ROOT + RUNS
    )
    logs = capsys.readouterr().out
    assert "mode=full" in logs and "555" in logs and OLD in logs


@pytest.mark.parametrize(
    "path,mode",
    [
        ("README.md", "full"),
        ("LICENSE", "full"),
        ("MANIFEST.in", "full"),
        ("AGENTS.md", "full"),
        ("CONTRIBUTING.md", "full"),
        ("requirements/dev.txt", "full"),
        ("scripts/ci/build_ppu_wheels.sh", "none"),
        ("scripts/ci/ppu_image_stack.py", "none"),
        ("scripts/ci/ppu_wheel_manifest.py", "none"),
        ("setup.py", "none"),
        ("pyproject.toml", "none"),
        ("CMakeLists.txt", "none"),
        ("csrc/kernel.cu", "none"),
        ("vllm_sail/file.py", "native"),
        ("plugin/hooks.py", "none"),
        ("cmake/build.cmake", "none"),
        (".github/vllm-main-verified.commit", "none"),
        ("unknown/file", "none"),
        ("scripts/ci/ppu_ci_command.py.bak", "full"),
        (".github/workflows/_ppu-build.yaml", "none"),
    ],
)
@pytest.mark.parametrize("rename", [False, True])
def test_package_changes_use_a_verified_native_donor(path, mode, rename, github_output):
    file = {"filename": path}
    if rename:
        file = {"filename": "docs/new", "previous_filename": path, "status": "renamed"}
    control, gh = reuse_setup(files=[file])
    assert validate_pair(control, gh, github_output) == reuse_result(
        mode, source=SHA if mode == "native" else OLD
    )


@pytest.mark.parametrize(
    "compare",
    [
        {},
        {"status": "ahead", "files": []},
        {"status": "diverged", "merge_base_commit": {"sha": OLD}, "files": []},
        {"status": "behind", "merge_base_commit": {"sha": OLD}, "files": []},
        {"status": "ahead", "merge_base_commit": {"sha": SHA}, "files": []},
        {"status": "ahead", "merge_base_commit": {"sha": OLD}},
        {"status": "ahead", "merge_base_commit": {"sha": OLD}, "files": None},
        {"status": "ahead", "merge_base_commit": {"sha": OLD}, "files": "docs/a"},
        {
            "status": "ahead",
            "merge_base_commit": {"sha": OLD},
            "files": [{"filename": "docs/new", "status": "renamed"}],
        },
        {
            "status": "ahead",
            "merge_base_commit": {"sha": OLD},
            "files": [{"filename": "docs/a"}] * 300,
        },
    ],
)
def test_unusable_truncated_or_nonancestor_compare_rejects_donor(
    compare, github_output
):
    control, gh = reuse_setup(compare=compare)
    assert validate_pair(control, gh, github_output) == reuse_result("none")


def test_malformed_compare_falls_back_to_full_build(github_output):
    control, gh = reuse_setup(
        compare={
            "status": "ahead",
            "merge_base_commit": {"sha": OLD},
            "files": [{}],
        }
    )
    assert validate_pair(control, gh, github_output) == reuse_result("none")


def test_compare_api_failure_falls_back_to_full_build(github_output):
    control, gh = reuse_setup(compare=http_error(500))
    assert validate_pair(control, gh, github_output) == reuse_result("none")


@pytest.mark.parametrize(
    "extra",
    [
        {"head_sha": None},
        {"head_branch": "feature"},
        {"event": "pull_request"},
        {"path": ".github/workflows/other.yaml"},
        {"repository": {"full_name": "fork/repo"}},
        {"display_title": "bad"},
        {"status": "in_progress"},
        {"conclusion": "failure"},
    ],
)
def test_validate_only_reuses_successful_trusted_executors(extra, github_output):
    control, gh = reuse_setup(
        candidates=[run(555, OLD, **{"conclusion": "success", **extra})]
    )
    assert validate_pair(control, gh, github_output) == reuse_result("none")
    assert not any("/compare/" in c[1] or "/artifacts" in c[1] for c in gh.calls)


def test_validate_accepts_artifacts_from_an_older_trusted_workflow_revision(
    github_output,
):
    control, gh = reuse_setup(
        candidates=[run(555, OLD, conclusion="success", head_sha="d" * 40)]
    )
    assert validate_pair(control, gh, github_output) == reuse_result(
        "native", source=SHA
    )


@pytest.mark.parametrize(
    "same_source,code_changed", [(False, False), (False, True), (True, False)]
)
def test_other_pr_needs_actual_immutable_source_equivalence(
    same_source, code_changed, github_output
):
    source = SHA if same_source else OLD
    candidate = run(
        555,
        conclusion="success",
        display_title=f"PPU PR #8 @ {source} by other (comment 43)",
    )
    control, gh = reuse_setup(
        candidates=[candidate],
        files=[{"filename": "csrc/new.cu" if code_changed else "tests/new.py"}],
    )
    pair = validate_pair(control, gh, github_output)
    mode = "full" if same_source or not code_changed else "none"
    expected_source = source
    assert pair == reuse_result(mode, source=expected_source)


@pytest.mark.parametrize(
    "artifacts",
    [
        [],
        [{"name": "ppu-wheels-555", "expired": True}],
        [{"name": "ppu-wheels-554", "expired": False}],
        [{"name": "ppu-wheels-555"}],
    ],
)
def test_missing_deleted_expired_or_reuse_only_run_falls_back_to_real_producer(
    artifacts, github_output
):
    control, gh = reuse_setup(
        candidates=[
            run(555, OLD, conclusion="success"),
            run(554, OLD, conclusion="success"),
        ]
    )
    gh.page("/actions/runs/555/artifacts", artifacts, "artifacts")
    assert validate_pair(control, gh, github_output) == reuse_result(
        "full", run_id="554"
    )


def test_reuse_artifacts_are_paginated(github_output):
    control, gh = reuse_setup()
    gh.page(
        "/actions/runs/555/artifacts",
        [{"name": "irrelevant", "expired": False}] * 100,
        "artifacts",
    )
    gh.page(
        "/actions/runs/555/artifacts",
        [{"name": "ppu-wheels-555", "expired": False}],
        "artifacts",
        page=2,
    )
    assert validate_pair(control, gh, github_output)["existing_build_run_id"] == "555"
    assert any(c[1].endswith("/artifacts") and c[3]["page"] == 2 for c in gh.calls)


@pytest.mark.parametrize("stage", ["runs", "artifacts", "compare"])
def test_reuse_api_failure_logs_and_builds_not_optimistic_reuse(
    stage, github_output, capsys
):
    control, gh = reuse_setup()
    if stage == "runs":
        gh.page(RUNS, http_error(503), event="workflow_dispatch", status="success")
    elif stage == "artifacts":
        gh.page("/actions/runs/555/artifacts", http_error(503))
    else:
        gh.add("GET", ROOT + f"/compare/{OLD}...{SHA}", http_error(503))
    assert validate_pair(control, gh, github_output) == reuse_result("none")
    logs = capsys.readouterr()
    assert "503" in logs.err
    assert "mode=none" in logs.out.lower()


def test_reuse_candidates_paginate_and_bound_expensive_selection(github_output):
    control, gh = reuse_setup()
    gh.page(
        RUNS,
        [
            run(i, OLD, conclusion="success", display_title="unrelated")
            for i in range(100)
        ],
        "workflow_runs",
        event="workflow_dispatch",
        status="success",
    )
    gh.page(
        RUNS,
        [run(555, OLD, conclusion="success")],
        "workflow_runs",
        page=2,
        event="workflow_dispatch",
        status="success",
    )
    assert validate_pair(control, gh, github_output)["existing_build_run_id"] == "555"

    github_output.unlink()
    control, gh = reuse_setup(
        candidates=[run(i, OLD, conclusion="success") for i in range(530, 551)]
    )
    for i in range(531, 551):
        gh.page(f"/actions/runs/{i}/artifacts", [], "artifacts")
    assert validate_pair(control, gh, github_output)["existing_build_run_id"] == ""
    assert len([c for c in gh.calls if c[1].endswith("/artifacts")]) == 20


def test_reuse_discovery_has_a_finite_page_budget(github_output):
    control, gh = reuse_setup(candidates=[])
    for page in range(1, 11):
        gh.page(
            RUNS,
            [run(i, display_title="unrelated") for i in range(100)],
            "workflow_runs",
            page=page,
            event="workflow_dispatch",
            status="success",
        )
    assert validate_pair(control, gh, github_output)["existing_build_run_id"] == ""
    assert len([c for c in gh.calls if c[1] == ROOT + RUNS]) <= 10
