from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from advisor.data.provider import Fundamentals, REPORTING_LAG_DAYS, is_available_asof

_NUMERIC_FIELDS = (
    "net_income",
    "total_equity",
    "revenue",
    "operating_income",
    "total_debt",
    "depreciation",
    "capex",
    "shares_outstanding",
    "market_cap",
)

_REQUIRED_FIELDS = ("ticker", "period_end", "snapshot_date", "source", *_NUMERIC_FIELDS)


@dataclass(frozen=True)
class FundamentalSnapshot:
    """One source snapshot of fundamentals, forward-filled only after it is knowable."""

    ticker: str
    snapshot_date: date
    source: str
    fundamentals: Fundamentals

    def is_available_asof(self, as_of: date, lag_days: int = REPORTING_LAG_DAYS) -> bool:
        return (
            self.snapshot_date <= as_of
            and is_available_asof(self.fundamentals.period_end, as_of, lag_days)
        )


def _parse_date(value: str | None, field: str) -> date:
    if value is None or not value.strip():
        raise ValueError(f"missing {field}")
    return date.fromisoformat(value.strip())


def _parse_float(row: dict[str, str], field: str) -> float:
    value = (row.get(field) or "").strip()
    if not value or value.lower() in {"none", "nan"}:
        raise ValueError(f"missing {field}")
    return float(value)


def _parse_snapshot(row: dict[str, str]) -> FundamentalSnapshot:
    ticker = (row.get("ticker") or "").strip().upper()
    source = (row.get("source") or "").strip()
    if not ticker:
        raise ValueError("missing ticker")
    if not source:
        raise ValueError("missing source")

    values = {field: _parse_float(row, field) for field in _NUMERIC_FIELDS}
    fundamentals = Fundamentals(
        period_end=_parse_date(row.get("period_end"), "period_end"),
        **values,
    )
    return FundamentalSnapshot(
        ticker=ticker,
        snapshot_date=_parse_date(row.get("snapshot_date"), "snapshot_date"),
        source=source,
        fundamentals=fundamentals,
    )


def load_fundamental_snapshots(path: str | Path) -> list[FundamentalSnapshot]:
    """Load a small point-in-time fundamentals fixture.

    Incomplete data rows are skipped instead of defaulted; a missing datum must be
    unavailable to the advisor seam, never fabricated as zero.
    """

    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(_REQUIRED_FIELDS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")

        records: list[FundamentalSnapshot] = []
        for row in reader:
            try:
                records.append(_parse_snapshot(row))
            except ValueError:
                continue
        return records


class SnapshotForwardFundamentalsProvider:
    """Fixture-backed provider for the advisor fundamentals seam."""

    def __init__(
        self,
        records: Iterable[FundamentalSnapshot],
        lag_days: int = REPORTING_LAG_DAYS,
    ) -> None:
        self.lag_days = lag_days
        self._records_by_ticker: dict[str, list[FundamentalSnapshot]] = {}
        for record in records:
            self._records_by_ticker.setdefault(record.ticker.upper(), []).append(record)
        for records_for_ticker in self._records_by_ticker.values():
            records_for_ticker.sort(
                key=lambda r: (r.fundamentals.period_end, r.snapshot_date)
            )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        lag_days: int = REPORTING_LAG_DAYS,
    ) -> "SnapshotForwardFundamentalsProvider":
        return cls(load_fundamental_snapshots(path), lag_days=lag_days)

    def get_prices(self, ticker: str, start: date, end: date):
        raise NotImplementedError(
            "SnapshotForwardFundamentalsProvider only supplies fundamentals"
        )

    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None:
        eligible = [
            record
            for record in self._records_by_ticker.get(ticker.upper(), [])
            if record.is_available_asof(as_of, self.lag_days)
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda r: (r.fundamentals.period_end, r.snapshot_date),
        ).fundamentals
