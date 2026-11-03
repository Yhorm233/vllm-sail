# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for collecting CPU and device tests together."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_same_basename_modules_collect_together(tmp_path: Path) -> None:
    files = []
    for tier in ("ut", "e2e"):
        path = tmp_path / tier / "test_shared_name.py"
        path.parent.mkdir()
        path.write_text(
            "from pathlib import Path\n\n"
            f"def test_origin():\n    assert Path(__file__).parent.name == {tier!r}\n",
            encoding="utf-8",
        )
        files.append(str(path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROOT / "pyproject.toml"),
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-q",
            *files,
        ],
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
