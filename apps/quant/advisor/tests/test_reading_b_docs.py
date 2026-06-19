from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PREREG = REPO_ROOT / "apps/quant/advisor/research/READING_B_PREREG.md"
RESULT = REPO_ROOT / "apps/quant/advisor/research/READING_B_RESULT.md"

FIXTURE_SHA = "fd2e574c43a78a096b459404f051f120ae03e02bf5ee73e8b20a174e710f1943"
PINNED_HASHES = (
    "4a481623a9776c111c65ca3ab176722d3a6e7305da045af2c78b2ba4529f6613",  # candidate
    "5f5254d3025a0ab3e4de3151825e77d1aee813cf69d802250b82a8811cb4ca8b",  # validation
    "408eebf70bf4bdefeb01e8975c93274fc641a0b1f384a002fab817b7052849e9",  # run hash (holdout-unlock)
)

# go-live denylist (mirrors the spirit of test_docs_truth.py for these research docs)
DENY = [
    r"production ready|production-ready", r"live trading", r"go live",
    r"real money|real-money", r"ready for production", r"deploy to production|go to production",
    r"live deployment", r"automated order execution", r"order placement",
]


def test_reading_b_docs_exist():
    assert PREREG.exists() and RESULT.exists()


def test_prereg_pins_fixture_and_all_hashes():
    text = PREREG.read_text(encoding="utf-8")
    assert FIXTURE_SHA in text
    for h in PINNED_HASHES:
        assert h in text, h


def test_result_records_blinded_holdout_and_pending_dev():
    text = RESULT.read_text(encoding="utf-8")
    assert FIXTURE_SHA in text
    assert "NOT YET RUN" in text       # dev-gate run is the next step, honestly not claimed done
    assert "UNTOUCHED" in text         # holdout blinded


def test_both_docs_carry_truth_anchors():
    for path in (PREREG, RESULT):
        text = path.read_text(encoding="utf-8")
        assert "DEV_FAILED" in text, path.name
        assert re.search(r"holdout", text, re.IGNORECASE), path.name
        assert re.search(r"report-only|research|does not authorize", text, re.IGNORECASE), path.name


def test_no_go_live_denylist_claims():
    offenders: list[str] = []
    for path in (PREREG, RESULT):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pat in DENY:
                if re.search(pat, line, re.IGNORECASE):
                    offenders.append(f"{path.name}:{line_no}: {pat}")
    assert offenders == [], offenders
