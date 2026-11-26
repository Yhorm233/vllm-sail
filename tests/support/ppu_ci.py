# SPDX-License-Identifier: Apache-2.0
"""Installed-package and per-node result guards loaded explicitly by device CI only."""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath

import pytest

EXTENSIONS = ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C", "_cumem_allocator")
BENCHMARKS = {"test_packed_decode_perf", "test_update_perf", "test_prefill_perf"}
RUNTIME_MODULES = {
    "torch",
    "vllm",
    "vllm_sail",
    "vllm_sail.envs",
    "vllm.platforms",
    "vllm_sail.native.extensions",
} | {"vllm_sail." + name for name in EXTENSIONS}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_report(path, report):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def inspect_runtime(isolated, checkout, requires):
    isolated, checkout = Path(isolated).resolve(), Path(checkout).resolve()
    require(Path.cwd().resolve() == isolated, "cwd is not the isolated root")
    require(
        not isolated.is_relative_to(checkout), "isolated root lives inside the checkout"
    )
    require(
        Path(__file__).resolve() == isolated / "tests/support/ppu_ci.py",
        "CI plugin is shadowed",
    )
    paths = {"plugin": str(Path(__file__).resolve())}
    modules = {}
    for name in sorted(set(requires) | RUNTIME_MODULES):
        module = importlib.import_module(name)
        file = Path(module.__file__).resolve()
        require(
            file.is_file()
            and not any(file.is_relative_to(p) for p in (isolated, checkout)),
            f"installed package shadowed by source: {name}: {file}",
        )
        paths[name] = str(file)
        modules[name] = module
    envs = modules["vllm_sail.envs"]
    require(
        envs.VLLM_SAIL_USE_PLA is True and envs.VLLM_PPU_USE_PLA is True,
        "PLA launch effective value is not True",
    )
    torch = modules["torch"]
    require(modules["vllm.platforms"].current_platform.is_ppu(), "not a PPU platform")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "exactly one available PPU required",
    )
    providers = modules["vllm_sail.native.extensions"].import_kernels(strict=True)
    return {
        "isolated_root": str(isolated),
        "checkout_root": str(checkout),
        "paths": paths,
        "native_providers": list(providers),
        "capability": list(torch.cuda.get_device_capability(0)),
        "runtime_env": {"VLLM_SAIL_USE_PLA": True},
    }


def validate_node_report(report, entry, board):
    require(
        report.get("schema") == 1 and report.get("board") == board,
        "node report identity mismatch",
    )
    require(report.get("file") == entry["test"], "node report file mismatch")
    require(
        report.get("status") == "success"
        and report.get("exitstatus") == 0
        and not report.get("errors"),
        "node report not successful",
    )
    collected, applicable, excluded = (
        report[k] for k in ("collected", "applicable", "hardware_excluded")
    )
    for values in (collected, applicable, excluded):
        require(
            isinstance(values, list) and values == sorted(set(values)),
            "node list duplicated or unsorted",
        )
    require(
        bool(applicable)
        and set(collected) == set(applicable) | set(excluded)
        and not set(applicable) & set(excluded),
        "node sets incomplete",
    )
    functions = set()
    for node in collected:
        require(node.startswith(entry["test"] + "::"), "node from unknown file")
        function = node.split("::")[-1].split("[", 1)[0]
        functions.add(function)
        supported = board in entry["board_overrides"].get(function, entry["boards"])
        require((node in applicable) == supported, "hardware-applicable node mismatch")
    require(set(entry["board_overrides"]) <= functions, "stale hardware exception")
    require(
        set(report["nodes"]) == set(applicable), "node phase reports missing or extra"
    )
    for phases in report["nodes"].values():
        require(
            set(phases) == {"setup", "call", "teardown"},
            "missing node execution phases",
        )
        for phase in phases.values():
            require(
                phase["outcome"] == "passed" and phase["wasxfail"] is None,
                "node did not pass normally",
            )
            duration = phase["duration_seconds"]
            require(
                isinstance(duration, int | float)
                and math.isfinite(duration)
                and duration >= 0,
                "invalid node duration",
            )
    runtime = report.get("runtime", {})
    paths = runtime.get("paths", {})
    required = set(entry["requires"]) | RUNTIME_MODULES
    require(
        required | {"plugin", "conftest"} <= paths.keys(),
        "missing installed-state subprocess paths",
    )
    # Aggregation runs on another CPU host: validate resolved absolute paths without requiring Pod files to still exist.
    isolated = PurePosixPath(runtime.get("isolated_root", ""))
    checkout = PurePosixPath(runtime.get("checkout_root", ""))
    require(
        isolated.is_absolute()
        and checkout.is_absolute()
        and not isolated.is_relative_to(checkout),
        "invalid isolated root",
    )
    for name in required:
        path = PurePosixPath(paths[name])
        require(
            path.is_absolute()
            and ".." not in path.parts
            and not any(path.is_relative_to(p) for p in (isolated, checkout)),
            "package under test is not in installed state",
        )
    for name, suffix in (
        ("plugin", "tests/support/ppu_ci.py"),
        ("conftest", "tests/conftest.py"),
    ):
        require(paths[name] == str(isolated / suffix), "test helper package shadowed")
    require(
        runtime.get("runtime_env") == {"VLLM_SAIL_USE_PLA": True},
        "PLA launch evidence mismatch",
    )


def pytest_addoption(parser):
    group = parser.getgroup("ppu-ci")
    group.addoption("--ppu-ci-config", required=True)
    group.addoption("--ppu-ci-board", choices=("ppu10", "ppu15"), required=True)
    group.addoption("--ppu-ci-report", required=True)


def pytest_configure(config):
    config.pluginmanager.register(NodeGuard(config), "ppu-node-guard")


class NodeGuard:
    def __init__(self, config):
        self.config = config
        self.destination = config.getoption("--ppu-ci-report")
        self.board = config.getoption("--ppu-ci-board")
        catalog = json.loads(Path(config.getoption("--ppu-ci-config")).read_bytes())[
            "test_catalog"
        ]
        self.catalog = {t["test"]: t for t in catalog}
        self.report = {
            "schema": 1,
            "board": self.board,
            "file": None,
            "collected": [],
            "applicable": [],
            "hardware_excluded": [],
            "nodes": {},
            "errors": [],
            "runtime": {},
            "benchmark_seconds": {},
            "status": "failure",
            "exitstatus": 1,
        }

    def pytest_sessionstart(self, session):
        try:
            root = Path(os.environ["PPU_CI_ISOLATED_ROOT"]).resolve()
            require(
                len(self.config.args) == 1, "each process must run exactly one file"
            )
            file = Path(self.config.args[0]).resolve().relative_to(root).as_posix()
            self.report["file"] = file
            require(
                file in self.catalog and self.board in self.catalog[file]["boards"],
                "file not routed to the current board",
            )
            require(
                self.config.inipath.resolve() == root / "pyproject.toml",
                "pytest config not in the isolated tree",
            )
            require(
                [Path(p).resolve() for p in self.config.getini("pythonpath")] == [root],
                "pytest pythonpath is not the isolated root",
            )
            self.report["runtime"] = inspect_runtime(
                root, os.environ["PPU_CI_CHECKOUT"], self.catalog[file]["requires"]
            )
            conftest = importlib.import_module("tests.conftest")
            require(
                Path(conftest.__file__).resolve() == root / "tests/conftest.py",
                "conftest shadowed",
            )
            require(
                conftest in self.config.pluginmanager.get_plugins(),
                "conftest not actually loaded",
            )
            require(
                conftest.HAS_REAL_DEVICE is True,
                "conftest did not detect a real device",
            )
            self.report["runtime"]["paths"]["conftest"] = str(
                Path(conftest.__file__).resolve()
            )
        except Exception as error:
            self.report["errors"].append(f"startup: {type(error).__name__}: {error}")
            write_report(self.destination, self.report)
            pytest.exit(str(error), returncode=1)

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.report["errors"].append(
                f"collection: {report.nodeid}: {report.longrepr}"
            )

    def pytest_deselected(self, items):
        self.report["errors"].append(
            "unexpected deselection: " + ", ".join(i.nodeid for i in items)
        )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items):
        original = list(items)
        yield
        if set(items) != set(original):
            self.report["errors"].append("another plugin changed the node set")
        entry = self.catalog[self.report["file"]]
        self.report["collected"] = sorted(i.nodeid for i in original)
        functions = {i.originalname or i.name.split("[", 1)[0] for i in original}
        if not set(entry["board_overrides"]) <= functions:
            self.report["errors"].append("stale hardware exception")
        kept = []
        for item in items:
            function = item.originalname or item.name.split("[", 1)[0]
            if self.board not in entry["board_overrides"].get(
                function, entry["boards"]
            ):
                self.report["hardware_excluded"].append(item.nodeid)
            else:
                kept.append(item)
                if (
                    "patch_utils_module" in item.fixturenames
                    or "upstream_source_root" in item.fixturenames
                ):
                    self.report["errors"].append(
                        "device node requests source-tree fixture"
                    )
        items[:] = kept
        self.report["applicable"] = sorted(i.nodeid for i in kept)
        self.report["hardware_excluded"].sort()
        if self.report["errors"]:
            items[:] = []

    def pytest_runtest_logreport(self, report):
        phases = self.report["nodes"].setdefault(report.nodeid, {})
        if report.when in phases:
            self.report["errors"].append(
                "duplicate node phase execution: " + report.nodeid
            )
        phases[report.when] = {
            "outcome": report.outcome,
            "wasxfail": getattr(report, "wasxfail", None),
            "duration_seconds": report.duration,
            "reason": str(report.longrepr) if report.longrepr else None,
        }
        function = report.nodeid.split("::")[-1].split("[", 1)[0]
        if function in BENCHMARKS:
            durations = self.report["benchmark_seconds"]
            durations[report.nodeid] = durations.get(report.nodeid, 0) + report.duration

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.report.update(exitstatus=int(exitstatus), status="success")
        try:
            validate_node_report(
                self.report, self.catalog[self.report["file"]], self.board
            )
        except (ValueError, KeyError, TypeError) as error:
            self.report["errors"].append(str(error))
            self.report.update(status="failure", exitstatus=1)
            session.exitstatus = 1
        write_report(self.destination, self.report)
