from __future__ import annotations

from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord  # reuse 15-field schema

LAZY_PRICES = "lazy_prices"
SIMILARITY_CONCEPT = "FilingSimilarity"


def compute_text_available_asof(filing_date: date, accepted_datetime: date) -> date:
    """Availability for a FILING-TEXT signal: the document IS the disclosure, knowable
    the instant the filing is public -> max(filing_date, accepted_datetime). NO +90d
    reporting lag (that lag is for XBRL financial VALUES tied to a report period). The
    similarity needs only that the current filing exist on EDGAR. snapshot_date stays
    None (filing-backed) — implementing it as a fetch date would zero the signal in WS4."""
    return max(filing_date, accepted_datetime)


def audit_text_available_asof(rec: EdgarXbrlRecord) -> bool:
    """D6 writes available_asof canonically; this re-derives it for an audit equality
    check. Distinct from edgar_xbrl_fixture.audit_available_asof (which adds +90)."""
    return rec.available_asof == compute_text_available_asof(
        rec.filing_date, rec.accepted_datetime
    )
