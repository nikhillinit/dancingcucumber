from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

RETIRED_PATHS = {
    "production_ready_system.py",
    "robust_trading_system.py",
    "automated_trading_system.py",
    "enhanced_training_system.py",
    "personalized_portfolio_system.py",
    "single_user_ai_system.py",
    "apps/quant/finrl_trading_agent.py",
    "apps/quant/qlib_factor_generator.py",
    "apps/quant/autogluon_ensemble.py",
    "AIHedgeFund/apps/quant/finrl_trading_agent.py",
    "AIHedgeFund/apps/quant/qlib_factor_generator.py",
    "AIHedgeFund/apps/quant/autogluon_ensemble.py",
}

RETIRED_MODULES = {
    Path(path).stem
    for path in RETIRED_PATHS
}


def _tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        REPO_ROOT / line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


IMPORT_RE = re.compile(r"^\s*import\s+(?P<modules>.+)$")
FROM_RE = re.compile(r"^\s*from\s+(?P<module>[\.\w]+)\s+import\s+")


def _imported_roots(source: str) -> set[str]:
    imported: set[str] = set()
    for line in source.splitlines():
        import_match = IMPORT_RE.match(line)
        if import_match:
            imported.update(
                module.strip().split()[0].split(".", maxsplit=1)[0]
                for module in import_match.group("modules").split(",")
            )
            continue

        from_match = FROM_RE.match(line)
        if from_match:
            imported.add(from_match.group("module").lstrip(".").split(".", maxsplit=1)[0])

    return imported


def test_retired_cleanup_targets_are_absent() -> None:
    remaining = sorted(
        path
        for path in RETIRED_PATHS
        if (REPO_ROOT / path).exists()
    )

    assert remaining == []


def test_tracked_python_modules_do_not_import_retired_modules() -> None:
    offenders: list[str] = []
    retired_paths = {REPO_ROOT / path for path in RETIRED_PATHS}

    for py_file in _tracked_python_files():
        if py_file in retired_paths or not py_file.exists():
            continue

        imported = _imported_roots(py_file.read_text(encoding="utf-8"))
        blocked = sorted(imported & RETIRED_MODULES)
        if blocked:
            rel_path = py_file.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel_path}: {', '.join(blocked)}")

    assert offenders == []
