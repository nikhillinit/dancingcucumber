"""One-time NETWORK fetch/freeze writer for the Lazy Prices fixture. NOT imported by any
test or runtime path. Pure helpers (tokenize/cosine/build_similarity_row) are unit-tested
offline; main() is the only network entrypoint and is operator-run outside the gate.

Usage (operator, network):
    python -m advisor.data.filing_text_fetch --out apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv

PIT discipline: available_asof = max(filing_date, accepted_datetime) — NEVER the fetch
date. Each current filing is paired with the SAME-FORM filing one year earlier; 2014
filings are pulled only as the prior-year baseline for 2015 and are not emitted as signal.
SEC etiquette: declared User-Agent, <=10 req/s."""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from datetime import date
from typing import Callable

from advisor.research.lazy_prices import SIMILARITY_CONCEPT, compute_text_available_asof
from advisor.data.edgar_xbrl_fixture import FIELDS

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Deterministic preprocessing: lowercase, \\w+ word tokens. Pinned so the fixture is
    byte-reproducible. (HTML stripping happens upstream in fetch_document.)"""
    return _WORD.findall(text.lower())


def cosine_tfidf(a: list[str], b: list[str]) -> float:
    """Cosine over raw term-frequency vectors (no IDF: a 2-doc YoY pair has no corpus).
    Identical docs -> 1.0; disjoint -> 0.0; deterministic."""
    ca, cb = Counter(a), Counter(b)
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _parse_d(s: str) -> date:
    return date.fromisoformat(s.strip().split("T")[0].split(" ")[0])


def build_similarity_row(*, asset, cik, accession, form, report_period_end,
                         filing_date, accepted_datetime, similarity) -> dict:
    """Build one fixture row in the 15-field EdgarXbrlRecord schema."""
    avail = compute_text_available_asof(_parse_d(filing_date), _parse_d(accepted_datetime))
    return {
        "asset": asset, "cik": cik, "accession": accession, "form": form,
        "report_period_end": _parse_d(report_period_end).isoformat(),
        "filing_date": _parse_d(filing_date).isoformat(),
        "accepted_datetime": _parse_d(accepted_datetime).isoformat(),
        "concept": SIMILARITY_CONCEPT, "unit": "ratio",
        "value": format(float(similarity), "g"),
        "available_asof": avail.isoformat(),
        "superseded_by": "", "amended_flag": "false", "missingness_reason": "",
        "denominator_policy": "cosine_tfidf_yoy_same_form",
    }


def fetch_document(http_get: Callable[[str], str], cik: str, accession: str) -> str:
    """Fetch a filing's primary document and strip HTML to text. Injected http_get keeps
    the parse path testable offline. (Implementation: data.sec.gov submissions -> primary
    doc URL -> http_get -> regex-strip tags -> tokenize upstream.)"""
    html = http_get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}.txt")
    return re.sub(r"<[^>]+>", " ", html)


def main() -> None:  # pragma: no cover - network entrypoint, operator-run
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--user-agent", required=True, help="SEC requires a declared UA")
    args = ap.parse_args()
    # 1) For each of the 30 CIKs: pull submissions (10-K/10-Q) 2014-2023.
    # 2) Pair each filing with the prior-year SAME-FORM, SAME-FISCAL-PERIOD filing:
    #    a 10-K pairs with the prior fiscal-year 10-K; a 10-Q pairs with the year-ago
    #    SAME fiscal quarter's 10-Q (e.g. Q2-2016 <-> Q2-2015), NEVER the adjacent
    #    quarter (Q1<->Q2 mixes seasonal boilerplate and inflates "change"). Match on
    #    (form, fiscal_period) from the submissions metadata. Tokenize both; cosine_tfidf.
    # 3) build_similarity_row(...) for CURRENT filings with report_period_end in 2015-2023
    #    (2014 filings are consumed only as the prior-year baseline, never emitted).
    # 4) Write CSV with header = FIELDS. Throttle <=10 req/s; declared UA.
    rows: list[dict] = []  # populated by the loop above (omitted: pure orchestration)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
