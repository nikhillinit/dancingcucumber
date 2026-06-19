from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta

from advisor.data.provider import REPORTING_LAG_DAYS

# WS3A source-agnostic contract: 15 fields per (asset, concept, as-of) record.
# snapshot_date is intentionally ABSENT here: SEC EDGAR is a FILING-BACKED source
# (WS3A spec lines 54, 70-72), so availability is driven by accepted/filed dates,
# never a global snapshot/fetch date.
FIELDS = (
    "asset", "cik", "accession", "form", "report_period_end", "filing_date",
    "accepted_datetime", "concept", "unit", "value", "available_asof",
    "superseded_by", "amended_flag", "missingness_reason", "denominator_policy",
)


@dataclass(frozen=True)
class EdgarXbrlRecord:
    asset: str
    cik: str
    accession: str
    form: str
    report_period_end: date
    filing_date: date
    accepted_datetime: date  # date granularity suffices: available_asof's +90d lag
                             # dominates, so the intraday component never binds.
    concept: str
    unit: str
    value: float
    available_asof: date
    superseded_by: str        # "" when not superseded
    amended_flag: bool
    missingness_reason: str    # "" when the datum is present
    denominator_policy: str


def _parse_date(s: str) -> date:
    # tolerate full ISO timestamps ("2016-10-26T16:31:00") and plain dates
    return date.fromisoformat(s.strip().split("T")[0].split(" ")[0])


def compute_available_asof(
    report_period_end: date,
    filing_date: date,
    accepted_datetime: date,
    lag_days: int = REPORTING_LAG_DAYS,
) -> date:
    """WS3A availability rule for a FILING-BACKED source: snapshot_date is omitted.
    Implementing snapshot_date as a fetch date would push every row past the backtest
    window and silently zero the signal -> a false-negative DEV_FAILED in WS4."""
    return max(report_period_end + timedelta(days=lag_days), filing_date, accepted_datetime)


def audit_available_asof(rec: EdgarXbrlRecord, lag_days: int = REPORTING_LAG_DAYS) -> bool:
    """T7 writes available_asof canonically into the fixture; this re-derives it for
    an audit equality check."""
    return rec.available_asof == compute_available_asof(
        rec.report_period_end, rec.filing_date, rec.accepted_datetime, lag_days
    )


def _is_complete(row: dict) -> bool:
    # missing->excluded (DEV_BRAIN rail #5): a usable record needs a numeric value and
    # the dates the availability rule depends on. Never zero-/median-/future-fill.
    if not (row.get("value") or "").strip():
        return False
    for key in ("report_period_end", "filing_date", "accepted_datetime", "available_asof"):
        if not (row.get(key) or "").strip():
            return False
    try:
        float(row["value"])
    except (TypeError, ValueError):
        return False
    return True


def load_fixture(path) -> list[EdgarXbrlRecord]:
    """Deterministic, network-free loader. Drops incomplete rows (never fills)."""
    records: list[EdgarXbrlRecord] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not _is_complete(row):
                continue
            records.append(EdgarXbrlRecord(
                asset=row["asset"],
                cik=row["cik"],
                accession=row["accession"],
                form=row["form"],
                report_period_end=_parse_date(row["report_period_end"]),
                filing_date=_parse_date(row["filing_date"]),
                accepted_datetime=_parse_date(row["accepted_datetime"]),
                concept=row["concept"],
                unit=row.get("unit", ""),
                value=float(row["value"]),
                available_asof=_parse_date(row["available_asof"]),
                superseded_by=row.get("superseded_by", "") or "",
                amended_flag=str(row.get("amended_flag", "")).strip().lower() in ("1", "true", "yes"),
                missingness_reason=row.get("missingness_reason", "") or "",
                denominator_policy=row.get("denominator_policy", "") or "",
            ))
    return records


def coverage_in_window(records: list[EdgarXbrlRecord], start: date, end: date) -> float:
    """Non-degeneracy stat: fraction of records usable inside [start, end]. A snapshot/
    fetch-date bug pushes available_asof past `end` -> coverage 0 -> this fails, not WS4."""
    if not records:
        return 0.0
    inwin = [r for r in records if start <= r.available_asof <= end]
    return len(inwin) / len(records)
