from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

# Vendored stefan-jansen-ml/, docs/superpowers/ specs+plans, .omx/ plans, and
# FLOOR_RESULT.md legitimately discuss these terms or are third-party/meta, so an
# explicit operator-doc set is required to avoid false positives.
REQUIRED_OPERATOR_DOCS = {
    "README.md",
    "START_HERE.md",
    "SOPHISTICATION_ROADMAP.md",
    "COMPLETE_SYSTEM_STATUS.md",
    "IMPLEMENTATION_GUIDE.md",
    "OPTIMIZED_SYSTEM_SUMMARY.md",
    "OPTIONS_FLOW_SYSTEM_SUMMARY.md",
    "bt_integration_summary.md",
}

TRUTH_HEADER_START = "<!-- AIHF_TRUTH_HEADER_START -->"
TRUTH_HEADER_END = "<!-- AIHF_TRUTH_HEADER_END -->"
ARCHIVE_START_TOKEN = "AIHF_TRUTH_ARCHIVE_START"
ARCHIVE_END = "<!-- AIHF_TRUTH_ARCHIVE_END -->"
ARCHIVE_START_RE = re.compile(
    r'^<!-- AIHF_TRUTH_ARCHIVE_START '
    r'superseded_by="apps/quant/advisor/backtest/FLOOR_RESULT\.md" '
    r'reason="(?P<reason>[^"]+)" -->$'
)

HEADER_ANCHORS = {
    "DEV_FAILED": re.compile(r"DEV_FAILED", re.IGNORECASE),
    "not authorized/not production/research qualifier": re.compile(
        r"not\s+authorized|not\s+production|research",
        re.IGNORECASE,
    ),
    "FLOOR_RESULT.md": re.compile(r"FLOOR_RESULT\.md", re.IGNORECASE),
}

DENYLIST_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"production ready|production-ready",
        r"live trading",
        r"go live",
        r"real money|real-money",
        r"paper trading operational",
        r"automated order execution",
        r"Fidelity automation|Fidelity automated",
        r"expected annual alpha",
        r"50-70%|50-60% annual|28-35% annual(ly)?",
        r"92% accuracy",
        r"95%\+",
        r"Sharpe\s?>\s?2\.5",
        r"immediate deployment",
    ]
]


def _tracked_required_docs() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *sorted(REQUIRED_OPERATOR_DOCS)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }


def _read_doc(rel_path: str) -> list[str]:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8").splitlines()


def _truth_header_regions(lines: list[str], rel_path: str) -> tuple[list[tuple[int, int]], list[str]]:
    regions: list[tuple[int, int]] = []
    errors: list[str] = []
    active_start: int | None = None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if "AIHF_TRUTH_HEADER_START" in stripped and stripped != TRUTH_HEADER_START:
            errors.append(f"{rel_path}:{line_no}: malformed truth-header START marker")
            continue
        if "AIHF_TRUTH_HEADER_END" in stripped and stripped != TRUTH_HEADER_END:
            errors.append(f"{rel_path}:{line_no}: malformed truth-header END marker")
            continue

        if stripped == TRUTH_HEADER_START:
            if active_start is not None:
                errors.append(f"{rel_path}:{line_no}: nested truth-header START marker")
            else:
                active_start = line_no
            continue

        if stripped == TRUTH_HEADER_END:
            if active_start is None:
                errors.append(f"{rel_path}:{line_no}: truth-header END without START")
            else:
                regions.append((active_start, line_no))
                active_start = None

    if active_start is not None:
        errors.append(f"{rel_path}:{active_start}: truth-header START without END")

    return regions, errors


def _archive_regions(lines: list[str], rel_path: str) -> tuple[list[tuple[int, int]], list[str]]:
    regions: list[tuple[int, int]] = []
    errors: list[str] = []
    active_start: int | None = None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if ARCHIVE_START_TOKEN in stripped:
            match = ARCHIVE_START_RE.fullmatch(stripped)
            if match is None:
                errors.append(
                    f"{rel_path}:{line_no}: archive START must carry exact "
                    'superseded_by="apps/quant/advisor/backtest/FLOOR_RESULT.md" '
                    'and non-empty reason="..."'
                )
            elif not match.group("reason").strip():
                errors.append(f"{rel_path}:{line_no}: archive START reason is empty")

            if active_start is not None:
                errors.append(f"{rel_path}:{line_no}: nested archive START marker")
            else:
                active_start = line_no
            continue

        if "AIHF_TRUTH_ARCHIVE_END" in stripped and stripped != ARCHIVE_END:
            errors.append(f"{rel_path}:{line_no}: malformed archive END marker")
            continue

        if stripped == ARCHIVE_END:
            if active_start is None:
                errors.append(f"{rel_path}:{line_no}: archive END without START")
            else:
                regions.append((active_start, line_no))
                active_start = None

    if active_start is not None:
        errors.append(f"{rel_path}:{active_start}: archive START without END")

    return regions, errors


def _line_numbers_in(regions: list[tuple[int, int]]) -> set[int]:
    exempt: set[int] = set()
    for start, end in regions:
        exempt.update(range(start, end + 1))
    return exempt


def test_required_docs_exist() -> None:
    tracked = _tracked_required_docs()
    missing_from_git = sorted(REQUIRED_OPERATOR_DOCS - tracked)
    missing_from_disk = sorted(
        rel_path
        for rel_path in REQUIRED_OPERATOR_DOCS
        if not (REPO_ROOT / rel_path).exists()
    )

    assert missing_from_git == []
    assert missing_from_disk == []


def test_truth_header_and_archive_markers_well_formed() -> None:
    errors: list[str] = []

    for rel_path in sorted(REQUIRED_OPERATOR_DOCS):
        lines = _read_doc(rel_path)
        _, header_errors = _truth_header_regions(lines, rel_path)
        _, archive_errors = _archive_regions(lines, rel_path)
        errors.extend(header_errors)
        errors.extend(archive_errors)

    assert errors == []


def test_no_archive_markers_in_first_40_lines() -> None:
    offenders: list[str] = []

    for rel_path in sorted(REQUIRED_OPERATOR_DOCS):
        for line_no, line in enumerate(_read_doc(rel_path)[:40], start=1):
            if "AIHF_TRUTH_ARCHIVE_" in line:
                offenders.append(f"{rel_path}:{line_no}")

    assert offenders == []


def test_each_required_doc_leads_with_truth_header() -> None:
    errors: list[str] = []

    for rel_path in sorted(REQUIRED_OPERATOR_DOCS):
        lines = _read_doc(rel_path)
        header_regions, _ = _truth_header_regions(lines, rel_path)
        archive_regions, _ = _archive_regions(lines, rel_path)

        if not header_regions:
            errors.append(f"{rel_path}: missing truth-header region")
            continue

        header_start, header_end = header_regions[0]
        if header_start > 40:
            errors.append(f"{rel_path}:{header_start}: truth-header must start within first 40 lines")

        if archive_regions:
            first_archive_start = min(start for start, _ in archive_regions)
            if header_end >= first_archive_start:
                errors.append(f"{rel_path}:{header_start}: truth-header must appear before archive block")

        header_text = "\n".join(lines[header_start - 1 : header_end])
        for label, pattern in HEADER_ANCHORS.items():
            if not pattern.search(header_text):
                errors.append(f"{rel_path}:{header_start}: truth-header missing {label}")

    assert errors == []


def test_no_unqualified_denylist_claims_outside_exempt_zones() -> None:
    offenders: list[str] = []

    for rel_path in sorted(REQUIRED_OPERATOR_DOCS):
        lines = _read_doc(rel_path)
        header_regions, _ = _truth_header_regions(lines, rel_path)
        archive_regions, _ = _archive_regions(lines, rel_path)
        exempt_lines = _line_numbers_in(header_regions) | _line_numbers_in(archive_regions)

        for line_no, line in enumerate(lines, start=1):
            if line_no in exempt_lines:
                continue

            for pattern in DENYLIST_PATTERNS:
                if pattern.search(line):
                    offenders.append(f"{rel_path}:{line_no}: {pattern.pattern}: {line.strip()}")
                    break

    assert offenders == []
