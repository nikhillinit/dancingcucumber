"""Deterministic QC-to-EDGAR identifier bridge."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

from advisor.source_integrity.edgar import EdgarDelisting, ReasonClass
from advisor.source_integrity.qc_export import QCDelisting


MATCHED = "matched"
AMBIGUOUS = "ambiguous"
UNMATCHED = "unmatched"


@dataclass(frozen=True)
class BridgeRow:
    qc_symbol: str
    company_name: str
    delisting_date: date
    momentum_decile: int
    value_bucket: str
    negative_book: bool
    bridge_status: str
    cik: str = ""
    form25_date: date | None = None
    reason: ReasonClass = ReasonClass.UNKNOWN


_NAME_SUFFIXES = {
    "adr",
    "class",
    "co",
    "common",
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "plc",
    "sa",
    "stock",
    "the",
}


def normalize_company_name(name: str) -> str:
    text = re.sub(r"&", " and ", name.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in _NAME_SUFFIXES]
    return " ".join(tokens)


def _token_score(a: str, b: str) -> float:
    left = set(a.split())
    right = set(b.split())
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def _date_close(a: date, b: date, days: int) -> bool:
    return abs((a - b).days) <= days


def _candidate_score(qc: QCDelisting, edgar: EdgarDelisting,
                     max_date_gap_days: int) -> float | None:
    if not _date_close(qc.delisting_date, edgar.filing_date, max_date_gap_days):
        return None
    qc_name = normalize_company_name(qc.company_name)
    edgar_name = normalize_company_name(edgar.company)
    if qc_name == edgar_name:
        return 1.0
    score = _token_score(qc_name, edgar_name)
    return score if score >= 0.82 else None


def bridge_qc_to_edgar(
    qc_rows: Sequence[QCDelisting],
    edgar_rows: Sequence[EdgarDelisting],
    reason_by_accession: Mapping[str, ReasonClass],
    *,
    max_date_gap_days: int = 15,
) -> list[BridgeRow]:
    out: list[BridgeRow] = []
    for qc in qc_rows:
        scored: list[tuple[float, EdgarDelisting]] = []
        for edgar in edgar_rows:
            score = _candidate_score(qc, edgar, max_date_gap_days)
            if score is not None:
                scored.append((score, edgar))

        scored.sort(key=lambda pair: (-pair[0], pair[1].filing_date, pair[1].accession))
        best_score = scored[0][0] if scored else None
        best = [edgar for score, edgar in scored if score == best_score]

        if len(best) == 1:
            match = best[0]
            out.append(
                BridgeRow(
                    qc_symbol=qc.qc_symbol,
                    company_name=qc.company_name,
                    delisting_date=qc.delisting_date,
                    momentum_decile=qc.momentum_decile,
                    value_bucket=qc.value_bucket,
                    negative_book=qc.negative_book,
                    bridge_status=MATCHED,
                    cik=match.cik,
                    form25_date=match.filing_date,
                    reason=reason_by_accession.get(match.accession, ReasonClass.UNKNOWN),
                )
            )
        elif len(best) > 1:
            out.append(_unknown_bridge_row(qc, AMBIGUOUS))
        else:
            out.append(_unknown_bridge_row(qc, UNMATCHED))
    return out


def _unknown_bridge_row(qc: QCDelisting, status: str) -> BridgeRow:
    return BridgeRow(
        qc_symbol=qc.qc_symbol,
        company_name=qc.company_name,
        delisting_date=qc.delisting_date,
        momentum_decile=qc.momentum_decile,
        value_bucket=qc.value_bucket,
        negative_book=qc.negative_book,
        bridge_status=status,
    )
