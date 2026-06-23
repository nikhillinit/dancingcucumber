"""QC/LEAN export schema for PIT eligible delistings."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping


QC_EXPORT_FIELDS = (
    "qc_symbol",
    "map_file_symbol",
    "company_name",
    "delisting_date",
    "last_eligible_date",
    "pit_eligible",
    "momentum_decile",
    "value_bucket",
    "negative_book",
)

BLOCKED_COLUMN_TOKENS = tuple(
    "".join(parts)
    for parts in (
        ("ret", "urn"),
        ("sha", "rpe"),
        ("p", "nl"),
        ("weight",),
        ("alloc", "ation"),
        ("family",),
        ("gate",),
        ("alpha",),
    )
)


@dataclass(frozen=True)
class QCDelisting:
    qc_symbol: str
    map_file_symbol: str
    company_name: str
    delisting_date: date
    last_eligible_date: date
    momentum_decile: int
    value_bucket: str
    negative_book: bool


def _parse_bool(value: str, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{field} must be boolean-like, got {value!r}")


def _reject_blocked_columns(columns: Iterable[str]) -> None:
    blocked: list[str] = []
    for column in columns:
        normalized = column.strip().lower().replace("_", "")
        if any(token in normalized for token in BLOCKED_COLUMN_TOKENS):
            blocked.append(column)
    if blocked:
        raise ValueError(f"QC export contains out-of-scope columns: {sorted(blocked)}")


def _strict_columns(row: Mapping[str, str]) -> None:
    columns = set(row)
    expected = set(QC_EXPORT_FIELDS)
    _reject_blocked_columns(columns)
    missing = sorted(expected - columns)
    extra = sorted(columns - expected)
    if missing or extra:
        raise ValueError(f"QC export schema mismatch; missing={missing}, extra={extra}")


def _parse_qc_row(row: Mapping[str, str]) -> QCDelisting:
    _strict_columns(row)
    pit_eligible = _parse_bool(row["pit_eligible"], "pit_eligible")
    if not pit_eligible:
        raise ValueError("QC export may contain only PIT-eligible delistings")

    delisting_date = date.fromisoformat(row["delisting_date"].strip())
    last_eligible_date = date.fromisoformat(row["last_eligible_date"].strip())
    days = (delisting_date - last_eligible_date).days
    if not 0 <= days <= 366:
        raise ValueError(
            "last_eligible_date must be on/before delisting_date and within the 12-month lookback"
        )

    momentum_decile = int(row["momentum_decile"])
    if not 1 <= momentum_decile <= 10:
        raise ValueError(f"momentum_decile must be 1..10, got {momentum_decile}")

    value_bucket = row["value_bucket"].strip()
    if not value_bucket:
        raise ValueError("value_bucket is required")

    return QCDelisting(
        qc_symbol=row["qc_symbol"].strip(),
        map_file_symbol=row["map_file_symbol"].strip(),
        company_name=row["company_name"].strip(),
        delisting_date=delisting_date,
        last_eligible_date=last_eligible_date,
        momentum_decile=momentum_decile,
        value_bucket=value_bucket,
        negative_book=_parse_bool(row["negative_book"], "negative_book"),
    )


def validate_qc_delisting_rows(rows: Iterable[Mapping[str, str]]) -> list[QCDelisting]:
    return [_parse_qc_row(row) for row in rows]


def load_qc_delistings_csv(path: str | Path) -> list[QCDelisting]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("QC export is missing a header")
        _strict_columns({field: "" for field in reader.fieldnames})
        return validate_qc_delisting_rows(reader)
