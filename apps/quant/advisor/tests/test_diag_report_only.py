from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    root = Path.cwd() / "apps" / "quant" / "advisor" / "tests" / f".tmp_diag_report_only_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    app_path = str(Path.cwd() / "apps" / "quant")
    env["PYTHONPATH"] = app_path if not env.get("PYTHONPATH") else f"{app_path}{os.pathsep}{env['PYTHONPATH']}"
    return env


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    positions = tmp_path / "positions.csv"
    prices = tmp_path / "prices.csv"
    positions.write_text("ticker,qty\nAAA,2\nBBB,1\n", encoding="utf-8", newline="\n")
    lines = ["Date,AAA,BBB"]
    for i in range(30):
        lines.append(f"2024-01-{i + 1:02d},{100 + i * 0.2:.2f},{50 + i * 0.1:.2f}")
    prices.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return positions, prices


def test_report_import_pulls_no_data_or_network_providers():
    script = """
import json
import sys

before = set(sys.modules)
import advisor.diagnostics.report  # noqa: F401
after = set(sys.modules) - before
bad = sorted(
    name for name in after
    if name.startswith("advisor.data.") or name in {"yfinance", "requests"}
)
print(json.dumps(bad))
raise SystemExit(1 if bad else 0)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == []


def test_cli_out_writes_only_requested_output_file(tmp_path: Path):
    positions, prices = _write_inputs(tmp_path)
    out = tmp_path / "diagnostics.txt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "advisor.diagnostics",
            "--positions",
            str(positions),
            "--prices",
            str(prices),
            "--out",
            str(out),
        ],
        cwd=Path.cwd(),
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert out.read_text(encoding="utf-8").endswith("\n")
    assert {path.name for path in tmp_path.iterdir()} == {
        "positions.csv",
        "prices.csv",
        "diagnostics.txt",
    }
