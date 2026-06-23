"""B5 threshold counting and report rendering for the source diagnostic."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from advisor.source_integrity.bridge import BridgeRow, MATCHED
from advisor.source_integrity.edgar import ReasonClass


CONFIDENT_REASONS = {
    ReasonClass.ACQUISITION,
    ReasonClass.BANKRUPTCY,
    ReasonClass.PERFORMANCE,
}
ADVERSE_REASONS = {
    ReasonClass.BANKRUPTCY,
    ReasonClass.PERFORMANCE,
}

BLOCKED_REPORT_PATTERNS = tuple(
    re.compile("".join(parts), re.IGNORECASE)
    for parts in (
        ("ret", "urn", "s?"),
        ("sha", "rpe"),
        ("p", "n", "l"),
        ("family", r"\s+comparisons?"),
        ("portfolio", r"\s+weights?"),
        ("prod", "uction"),
        ("bro", "ker"),
        ("paper", r"\s+", "trading"),
    )
)


@dataclass(frozen=True)
class DiagnosticThresholds:
    adverse_mass: int = 50
    mappability: float = 0.85
    concentration: float = 2.0


@dataclass(frozen=True)
class DiagnosticResult:
    total_delistings: int
    matched_classified: int
    mappability: float
    adverse_mass: int
    adverse_by_momentum_decile: dict[int, int]
    concentration_ratio: float
    failed_thresholds: tuple[str, ...]
    outcome: str

    @property
    def passed(self) -> bool:
        return not self.failed_thresholds


def evaluate_thresholds(
    rows: list[BridgeRow],
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
) -> DiagnosticResult:
    total = len(rows)
    matched_classified = sum(
        1
        for row in rows
        if row.bridge_status == MATCHED and row.reason in CONFIDENT_REASONS
    )
    mappability = matched_classified / total if total else 0.0

    adverse_by_decile = {decile: 0 for decile in range(1, 11)}
    for row in rows:
        if row.bridge_status == MATCHED and row.reason in ADVERSE_REASONS:
            adverse_by_decile[row.momentum_decile] += 1

    adverse_mass = sum(adverse_by_decile.values())
    loser_tail = adverse_by_decile[1]
    winner_tail = adverse_by_decile[10]
    if winner_tail == 0:
        concentration_ratio = math.inf if loser_tail > 0 else 0.0
    else:
        concentration_ratio = loser_tail / winner_tail

    failed: list[str] = []
    if adverse_mass < thresholds.adverse_mass:
        failed.append("mass")
    if mappability < thresholds.mappability:
        failed.append("mappability")
    if concentration_ratio < thresholds.concentration:
        failed.append("concentration")

    return DiagnosticResult(
        total_delistings=total,
        matched_classified=matched_classified,
        mappability=mappability,
        adverse_mass=adverse_mass,
        adverse_by_momentum_decile=adverse_by_decile,
        concentration_ratio=concentration_ratio,
        failed_thresholds=tuple(failed),
        outcome="STOP" if failed else "NEXT_PREREG_REVIEW",
    )


def render_report(result: DiagnosticResult) -> str:
    ratio = (
        "inf"
        if math.isinf(result.concentration_ratio)
        else f"{result.concentration_ratio:.2f}"
    )
    failed = ", ".join(result.failed_thresholds) if result.failed_thresholds else "none"
    text = "\n".join(
        [
            "# QC plus EDGAR source-integrity diagnostic",
            "",
            f"Outcome: {result.outcome}",
            f"Total QC eligible delistings: {result.total_delistings}",
            f"Matched and classified: {result.matched_classified}",
            f"Mappability: {result.mappability:.1%}",
            f"Confident adverse count: {result.adverse_mass}",
            f"Momentum decile 1 adverse count: {result.adverse_by_momentum_decile[1]}",
            f"Momentum decile 10 adverse count: {result.adverse_by_momentum_decile[10]}",
            f"Decile concentration ratio: {ratio}",
            f"Failed thresholds: {failed}",
        ]
    )
    assert_source_only_report(text)
    return text


def assert_source_only_report(text: str) -> None:
    offenders = [
        pattern.pattern
        for pattern in BLOCKED_REPORT_PATTERNS
        if pattern.search(text)
    ]
    if offenders:
        raise ValueError(f"diagnostic report contains out-of-scope terms: {offenders}")
