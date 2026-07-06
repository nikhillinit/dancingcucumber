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
    root = Path.cwd() / "apps" / "quant" / "advisor" / "tests" / f".tmp_diag_cli_{uuid.uuid4().hex}"
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
    positions.write_text("ticker,qty\nAAA,2\nBBB,1\nCASH,100\n", encoding="utf-8", newline="\n")
    lines = ["Date,AAA,BBB"]
    for i in range(30):
        lines.append(f"2024-01-{i + 1:02d},{100 + i * 0.2:.2f},{50 + i * 0.1:.2f}")
    prices.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return positions, prices


def _run_diag(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "advisor.diagnostics", *args],
        cwd=Path.cwd(),
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_diagnostics_cli_happy_path_prints_text(tmp_path: Path):
    positions, prices = _write_inputs(tmp_path)

    result = _run_diag("--positions", str(positions), "--prices", str(prices))

    assert result.returncode == 0
    assert result.stderr == ""
    assert "DISCLOSURES:" in result.stdout
    assert "Diagnostics report-only: no signal, direction, or sizing." in result.stdout


def test_diagnostics_cli_missing_file_exits_nonzero_with_clear_message(tmp_path: Path):
    _positions, prices = _write_inputs(tmp_path)
    missing = tmp_path / "missing.csv"

    result = _run_diag("--positions", str(missing), "--prices", str(prices))

    assert result.returncode != 0
    assert "positions file not found" in result.stderr
    assert str(missing) in result.stderr


def test_diagnostics_cli_json_output_parses(tmp_path: Path):
    positions, prices = _write_inputs(tmp_path)

    result = _run_diag("--positions", str(positions), "--prices", str(prices), "--json")

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "disclosures" in data
    assert "metrics" in data
