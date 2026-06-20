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
import html as _html
import json
import math
import re
import time
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable

from advisor.research.lazy_prices import SIMILARITY_CONCEPT, compute_text_available_asof
from advisor.data.edgar_xbrl_fixture import FIELDS

_WORD = re.compile(r"\w+")
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_IXBRL_META = re.compile(r"<ix:(header|hidden)[^>]*>.*?</ix:\1>", re.S | re.I)

# Fetch/pairing parameters (operator network path; NOT in the pytest gate).
THROTTLE_S = 0.12                          # <=10 req/s SEC etiquette
BASE_FORMS = {"10-K", "10-Q"}             # base periodics; amendments excluded for clean pairing
PAIR_MIN_DAYS, PAIR_MAX_DAYS = 320, 410   # YoY same-fiscal-period window (~365 +/- 45)
EMIT_MIN_YEAR, EMIT_MAX_YEAR = 2015, 2023 # 2014 filings are baseline-only, never emitted


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


# ----------------------------- live network getters (NOT in the gate) -----------------------------


def _http_get(url: str, ua: str) -> str:  # pragma: no cover - network
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def _http_json(url: str, ua: str) -> dict:  # pragma: no cover - network
    return json.loads(_http_get(url, ua))


def collect_filings(cik: str, ua: str) -> list[dict]:  # pragma: no cover - network
    """All base 10-K/10-Q filings for a CIK from submissions `recent` + the paginated older
    files. Returns dicts: accession (dashed), form, report_end, filed, accepted (date), and
    primary_doc. Deduped by accession for a deterministic fixture."""
    top = _http_json(f"https://data.sec.gov/submissions/CIK{cik}.json", ua)
    blobs = [top.get("filings", {}).get("recent", {})]
    for f in top.get("filings", {}).get("files", []):
        time.sleep(THROTTLE_S)
        blobs.append(_http_json(f"https://data.sec.gov/submissions/{f['name']}", ua))
    out: list[dict] = []
    seen: set[str] = set()
    for rec in blobs:
        accns = rec.get("accessionNumber", [])
        forms = rec.get("form", [])
        reports = rec.get("reportDate", [])
        fileds = rec.get("filingDate", [])
        accepteds = rec.get("acceptanceDateTime", [])
        primaries = rec.get("primaryDocument", [])
        for i, accn in enumerate(accns):
            form = forms[i] if i < len(forms) else ""
            rep = reports[i] if i < len(reports) else ""
            if form not in BASE_FORMS or not rep or accn in seen:
                continue
            seen.add(accn)
            out.append({
                "accession": accn, "form": form, "report_end": rep,
                "filed": fileds[i] if i < len(fileds) else "",
                "accepted": (accepteds[i][:10] if i < len(accepteds) and accepteds[i] else ""),
                "primary_doc": primaries[i] if i < len(primaries) else "",
            })
    return out


def _doc_url(cik: str, accession: str, primary_doc: str) -> str:
    """URL of the filing's PRIMARY document (the 10-K/10-Q body) — NOT the full-submission
    .txt (which concatenates exhibits + XBRL + binary blobs and would bias YoY cosine)."""
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{primary_doc}")


def _pair_baseline(cur: dict, same_form: list[dict]) -> dict | None:
    """Prior-year SAME-FORM, SAME-FISCAL-PERIOD filing: report_end ~365d earlier (within
    [PAIR_MIN_DAYS, PAIR_MAX_DAYS], closest to 365). Maps 10-K<->prior-FY 10-K and
    Q2<->year-ago Q2; NEVER the adjacent quarter."""
    rc = _parse_d(cur["report_end"])
    best, best_gap = None, None
    for b in same_form:
        gap = (rc - _parse_d(b["report_end"])).days
        if PAIR_MIN_DAYS <= gap <= PAIR_MAX_DAYS:
            d = abs(gap - 365)
            if best_gap is None or d < best_gap:
                best, best_gap = b, d
    return best


def _html_to_text(doc: str) -> str:
    """Faithful HTML->text for the cosine. Decodes entities FIRST so `&nbsp;` and `&#160;`
    collapse to the same char (else they tokenize as 'nbsp' vs '160' and a filing-agent
    encoding flip craters YoY cosine), then drops non-prose <script>/<style> and the
    inline-XBRL <ix:header>/<ix:hidden> metadata (CIK/context-ref junk that otherwise
    depresses the whole 2019-2020 large-filer iXBRL-transition cohort), then strips tags.
    NOT a methodology change: tokenize() stays frozen; this only fixes text extraction."""
    txt = _html.unescape(doc)
    txt = _SCRIPT_STYLE.sub(" ", txt)
    txt = _IXBRL_META.sub(" ", txt)
    return _TAG.sub(" ", txt)


def _tokens_for(cik: str, fl: dict, ua: str, cache: dict) -> list[str] | None:  # pragma: no cover - network
    """Fetch + strip + tokenize a filing's primary document, memoized by accession so each
    filing (used as both a current and the next year's baseline) is fetched at most once."""
    accn = fl["accession"]
    if accn in cache:
        return cache[accn]
    if not fl["primary_doc"]:
        cache[accn] = None
        return None
    time.sleep(THROTTLE_S)
    try:
        doc = _http_get(_doc_url(cik, accn, fl["primary_doc"]), ua)
    except Exception as e:  # noqa: BLE001 - one-time script; log + skip the filing
        print(f"    {accn}: fetch failed ({e})")
        cache[accn] = None
        return None
    toks = tokenize(_html_to_text(doc))
    cache[accn] = toks
    return toks


def main() -> None:  # pragma: no cover - network entrypoint, operator-run
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--user-agent", required=True, help="SEC requires a declared UA")
    ap.add_argument("--prices", default="apps/quant/advisor/tests/fixtures/floor_prices.csv")
    ap.add_argument("--only", default="", help="comma-separated tickers for a dry-run subset")
    args = ap.parse_args()
    ua = args.user_agent

    universe = [c for c in pd.read_csv(args.prices, nrows=0).columns
                if c not in ("Date", "SPY")]
    if args.only:
        want = {t.strip() for t in args.only.split(",") if t.strip()}
        universe = [t for t in universe if t in want]
    tk = _http_json("https://www.sec.gov/files/company_tickers.json", ua)
    cik_map = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in tk.values()}
    missing = [t for t in universe if t not in cik_map]
    assert not missing, f"no CIK for {missing}"

    rows: list[dict] = []
    for ticker in universe:
        cik = cik_map[ticker]
        time.sleep(THROTTLE_S)
        by_form: dict[str, list[dict]] = {}
        for fl in collect_filings(cik, ua):
            by_form.setdefault(fl["form"], []).append(fl)
        cache: dict[str, list[str] | None] = {}
        emitted = 0
        for form, group in by_form.items():
            group_sorted = sorted(group, key=lambda x: x["report_end"])
            for cur in group_sorted:
                rc = _parse_d(cur["report_end"])
                if not (EMIT_MIN_YEAR <= rc.year <= EMIT_MAX_YEAR):
                    continue  # 2014 currents are baseline-only; tail FY filed past window dropped
                base = _pair_baseline(cur, group_sorted)
                if base is None:
                    continue  # no prior-year same-period filing -> drop (coverage dips, not error)
                tc = _tokens_for(cik, cur, ua, cache)
                tb = _tokens_for(cik, base, ua, cache)
                if not tc or not tb:
                    continue
                rows.append(build_similarity_row(
                    asset=ticker, cik=cik, accession=cur["accession"], form=form,
                    report_period_end=cur["report_end"], filing_date=cur["filed"],
                    accepted_datetime=cur["accepted"] or cur["filed"],
                    similarity=cosine_tfidf(tc, tb),
                ))
                emitted += 1
        print(f"  {ticker}: {emitted} similarity rows")

    # Deterministic order so the committed fixture's SHA (the holdout-unlock key) is stable.
    rows.sort(key=lambda r: (r["asset"], r["report_period_end"], r["accession"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print(f"DONE. {len(rows)} rows -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
