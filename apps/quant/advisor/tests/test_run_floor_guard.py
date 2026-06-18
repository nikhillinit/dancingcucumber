from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def _run_floor(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "tools/run-floor.mjs", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_holdout_is_not_reachable_through_wrapper() -> None:
    result = _run_floor("--holdout")
    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "not reachable through this wrapper" in combined_output


def test_unknown_arg_is_rejected_by_wrapper() -> None:
    result = _run_floor("--bogus")

    assert result.returncode != 0


def test_enforce_holdout_combo_is_rejected_by_wrapper() -> None:
    result = _run_floor("--enforce", "--holdout")

    assert result.returncode != 0
