"""EDGAR-side parsers for the frozen QC plus EDGAR source diagnostic.

The live fetch helpers enforce SEC fair-access mechanics, but tests exercise
only deterministic parsing and request construction.
"""
from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import PurePosixPath
from typing import Callable, Iterable


FORM_25_TYPES = frozenset({"25", "25-NSE"})
SEC_MAX_REQUESTS_PER_SECOND = 10
SEC_MIN_INTERVAL_SECONDS = 1 / SEC_MAX_REQUESTS_PER_SECOND


class ReasonClass(str, Enum):
    BANKRUPTCY = "bankruptcy"
    ACQUISITION = "acquisition"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EdgarDelisting:
    cik: str
    company: str
    form: str
    filing_date: date
    accession: str
    filename: str
    cited_rule: str = ""


@dataclass(frozen=True)
class FilingEvent:
    cik: str
    form: str
    filing_date: date
    accession: str = ""
    items: frozenset[str] = frozenset()
    text: str = ""


_ITEM_RE = re.compile(r"\bitem\s+([0-9]{1,2}\.[0-9]{2})\b", re.IGNORECASE)


def _norm_cik(cik: str) -> str:
    digits = re.sub(r"\D", "", cik)
    if not digits:
        raise ValueError(f"invalid CIK: {cik!r}")
    return digits.zfill(10)


def _accession_from_filename(filename: str) -> str:
    stem = PurePosixPath(filename).name.rsplit(".", maxsplit=1)[0]
    return stem.strip()


def parse_master_index(text: str) -> list[EdgarDelisting]:
    """Parse SEC master.idx content and keep Form 25 / 25-NSE delisting rows."""
    out: list[EdgarDelisting] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 5 or parts[0].lower() == "cik":
            continue
        cik, company, form, filed, filename = parts
        form = form.upper()
        if form not in FORM_25_TYPES:
            continue
        out.append(
            EdgarDelisting(
                cik=_norm_cik(cik),
                company=company,
                form=form,
                filing_date=date.fromisoformat(filed),
                accession=_accession_from_filename(filename),
                filename=filename,
            )
        )
    return out


def build_master_index_url(year: int, quarter: int) -> str:
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1..4, got {quarter!r}")
    return f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"


def _declared_user_agent(user_agent: str) -> str:
    ua = user_agent.strip()
    if len(ua) < 12 or "@" not in ua:
        raise ValueError("SEC EDGAR fetches require a declared User-Agent with contact email")
    return ua


class SecRateLimiter:
    def __init__(self, min_interval: float = SEC_MIN_INTERVAL_SECONDS):
        self.min_interval = float(min_interval)
        self._last_request = 0.0

    def wait(self, now: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep) -> None:
        current = now()
        elapsed = current - self._last_request
        if self._last_request and elapsed < self.min_interval:
            sleep(self.min_interval - elapsed)
        self._last_request = now()


def fetch_text(url: str, *, user_agent: str, limiter: SecRateLimiter | None = None,
               opener: Callable = urllib.request.urlopen) -> str:
    """Fetch a SEC text resource with declared UA and optional 10 req/s limiter."""
    ua = _declared_user_agent(user_agent)
    if limiter is not None:
        limiter.wait()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": ua,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        },
    )
    with opener(req, timeout=60) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def extract_8k_items(text: str) -> frozenset[str]:
    return frozenset(match.group(1) for match in _ITEM_RE.finditer(text))


def _event_items(event: FilingEvent) -> frozenset[str]:
    if event.items:
        return event.items
    return extract_8k_items(event.text)


def _within_days(target: date, observed: date, days: int) -> bool:
    return abs((observed - target).days) <= days


def _defm14a_in_prior_year(target: date, observed: date) -> bool:
    delta = (target - observed).days
    return 0 <= delta <= 365


def _listing_standard_signal(cited_rule: str) -> bool:
    text = cited_rule.lower()
    return any(
        token in text
        for token in (
            "continued listing",
            "listing standard",
            "12d2-2(b)",
            "12d2-2(c)",
            "12d2-2(d)",
        )
    )


def classify_reason(delisting: EdgarDelisting,
                    events: Iterable[FilingEvent]) -> ReasonClass:
    """Classify terminal reason with bankruptcy > acquisition > performance > unknown."""
    cik = _norm_cik(delisting.cik)
    nearby_8k: list[FilingEvent] = []
    prior_defm14a = False

    for event in events:
        if _norm_cik(event.cik) != cik:
            continue
        form = event.form.upper()
        if form in {"8-K", "8-K/A"} and _within_days(delisting.filing_date, event.filing_date, 90):
            nearby_8k.append(event)
        elif form == "DEFM14A" and _defm14a_in_prior_year(delisting.filing_date, event.filing_date):
            prior_defm14a = True

    item_sets = [_event_items(event) for event in nearby_8k]
    if any("1.03" in items for items in item_sets):
        return ReasonClass.BANKRUPTCY
    if prior_defm14a or any("2.01" in items for items in item_sets):
        return ReasonClass.ACQUISITION
    if any("3.01" in items for items in item_sets) or _listing_standard_signal(delisting.cited_rule):
        return ReasonClass.PERFORMANCE
    return ReasonClass.UNKNOWN
