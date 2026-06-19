"""WS3C T7 — one-time fetch/freeze of the committed SEC EDGAR XBRL fundamentals fixture.

NOT imported by any test/runtime path except the offline parse test. The live fetch
(`build_fixture` with the default network getters) is operator/Bash-run, NOT in the pytest
gate. The pure parse helpers below are unit-tested offline against captured JSON shapes.

mktcap_anchor is REAL market cap at the filing's availability:
    mktcap_anchor = shares_asof * split_factor(period_end -> now) * C_sa(t0)
where C_sa is yfinance Close (auto_adjust=False, split-adjusted) and split_factor is the
product of yfinance splits after period_end. Aligning shares (report basis) and price
(current split basis) to the same basis makes the product real dollars -> split-invariant,
with NO spurious step at split dates (the reason the naive shares*split_adj_close fails).
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from advisor.data.provider import REPORTING_LAG_DAYS

SEC_UA = "AIHedgeFund research nikhil@narralytics.ai"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
FIELDS = (
    "asset", "cik", "accession", "form", "report_period_end", "filing_date",
    "accepted_datetime", "concept", "unit", "value", "available_asof",
    "superseded_by", "amended_flag", "missingness_reason", "denominator_policy",
)
ANCHOR_POLICY = "as_reported_shares_x_split_factor_x_yf_close_at_avail"

# ----------------------------- pure parse helpers (offline-testable) -----------------------------


def latest_facts_by_accn(units_facts: list[dict]) -> dict[str, dict]:
    """From a companyconcept units list, keep the LATEST `end` per accession (drops the
    prior-year comparatives a 10-K carries). Skips rows missing accn/end/val."""
    out: dict[str, dict] = {}
    for f in units_facts:
        accn, end, val = f.get("accn"), f.get("end"), f.get("val")
        if not accn or not end or val is None:
            continue
        if accn not in out or end > out[accn]["end"]:
            out[accn] = {"end": end, "val": val, "form": f.get("form", ""), "filed": f.get("filed", "")}
    return out


def merge_concept_facts(primary: list[dict], fallback: list[dict]) -> list[dict]:
    """Prefer `primary` facts; include `fallback` facts only for accessions absent from
    primary. Used to backfill names that report equity under the alternate tag
    `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`."""
    primary_accns = {f.get("accn") for f in primary}
    return list(primary) + [f for f in fallback if f.get("accn") not in primary_accns]


def accepted_map(submission_blobs: list[dict]) -> dict[str, str]:
    """Merge accn -> acceptanceDateTime (date part) across the `recent` block and the older
    paginated submission files. Older files are flat dicts; `recent` is nested."""
    m: dict[str, str] = {}
    for blob in submission_blobs:
        rec = blob.get("filings", {}).get("recent") if "filings" in blob else blob
        if not rec:
            continue
        for a, dt in zip(rec.get("accessionNumber", []), rec.get("acceptanceDateTime", [])):
            if a and dt:
                m[a] = dt[:10]
    return m


def compute_available_asof(period_end: date, filed: date | None, accepted: date | None,
                           lag_days: int = REPORTING_LAG_DAYS) -> date:
    """WS3A availability rule, snapshot_date omitted (filing-backed). Missing accepted
    falls back to max(end+lag, filed) — end+lag dominates anyway."""
    cands = [period_end + timedelta(days=lag_days)]
    if filed:
        cands.append(filed)
    if accepted:
        cands.append(accepted)
    return max(cands)


def cumulative_split_factor(splits: list[tuple[date, float]], after: date) -> float:
    """Product of split ratios strictly after `after` (period_end). 1.0 if none."""
    f = 1.0
    for d, r in splits:
        if d > after:
            f *= float(r)
    return f


def close_asof(closes: list[tuple[date, float]], t0: date) -> float | None:
    """Last close on/<= t0 from a date-sorted (date, close) list. None if all are later."""
    val = None
    for d, c in closes:
        if d <= t0:
            val = c
        else:
            break
    return val


def _row(asset, cik, accn, form, period_end, filed, accepted, avail, concept, unit, value,
         denom=""):
    amended = form.endswith("/A")
    return {
        "asset": asset, "cik": cik, "accession": accn, "form": form,
        "report_period_end": period_end.isoformat(),
        "filing_date": filed.isoformat() if filed else "",
        "accepted_datetime": accepted.isoformat() if accepted else "",
        "concept": concept, "unit": unit, "value": repr(float(value)),
        "available_asof": avail.isoformat(), "superseded_by": "",
        "amended_flag": "true" if amended else "false", "missingness_reason": "",
        "denominator_policy": denom,
    }


def build_asset_rows(asset: str, cik: str, equity_facts: list[dict], shares_facts: list[dict],
                     accepted: dict[str, str], splits: list[tuple[date, float]],
                     closes: list[tuple[date, float]], lag_days: int = REPORTING_LAG_DAYS,
                     min_year: int = 2014, max_year: int = 2023) -> list[dict]:
    """Pure: join equity<->shares by (accn, same period_end), compute available_asof and the
    real-dollar MarketCapAnchor. Emits StockholdersEquity + CommonStockSharesOutstanding +
    MarketCapAnchor rows per periodic filing within [min_year, max_year]."""
    eq = latest_facts_by_accn(equity_facts)
    sh = latest_facts_by_accn(shares_facts)
    rows: list[dict] = []
    for accn, e in sorted(eq.items()):
        if e["form"] not in FORMS:
            continue
        period_end = date.fromisoformat(e["end"])
        if not (min_year <= period_end.year <= max_year):
            continue
        filed = date.fromisoformat(e["filed"]) if e.get("filed") else None
        acc = date.fromisoformat(accepted[accn]) if accepted.get(accn) else None
        avail = compute_available_asof(period_end, filed, acc, lag_days)
        rows.append(_row(asset, cik, accn, e["form"], period_end, filed, acc, avail,
                         "StockholdersEquity", "USD", e["val"]))
        s = sh.get(accn)
        # current-period shares = period_end (us-gaap CommonStockSharesOutstanding) OR a cover
        # date a few weeks later (dei EntityCommonStockSharesOutstanding); reject prior-year
        # comparatives (~ -1yr).
        s_end = date.fromisoformat(s["end"]) if s else None
        if s is not None and -15 <= (s_end - period_end).days <= 125:
            rows.append(_row(asset, cik, accn, s["form"], period_end, filed, acc, avail,
                             "CommonStockSharesOutstanding", "shares", s["val"]))
            c0 = close_asof(closes, avail)
            if c0 is not None and c0 > 0:
                factor = cumulative_split_factor(splits, period_end)
                anchor = float(s["val"]) * factor * c0
                rows.append(_row(asset, cik, accn, s["form"], period_end, filed, acc, avail,
                                 "MarketCapAnchor", "USD", anchor, denom=ANCHOR_POLICY))
    return rows


# ----------------------------- live network getters (NOT in the gate) -----------------------------


def _http_json(url: str, ua: str = SEC_UA) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _sec_concept(cik: str, taxonomy: str, tag: str) -> list[dict]:
    try:
        d = _http_json(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json")
    except Exception:
        return []
    units = d.get("units", {})
    key = next(iter(units), None)
    return units.get(key, []) if key else []


def _sec_accepted(cik: str) -> dict[str, str]:
    blobs = [_http_json(f"https://data.sec.gov/submissions/CIK{cik}.json")]
    for f in blobs[0].get("filings", {}).get("files", []):
        blobs.append(_http_json(f"https://data.sec.gov/submissions/{f['name']}"))
        time.sleep(0.15)
    return accepted_map(blobs)


def _yf_closes_splits(ticker: str):
    import yfinance as yf
    t = yf.Ticker(ticker)
    hist = yf.download(ticker, start="2014-06-01", end="2024-07-01", auto_adjust=False,
                       progress=False)
    close = hist["Close"]
    if hasattr(close, "columns"):  # MultiIndex single-ticker frame
        close = close.iloc[:, 0]
    closes = [(d.date(), float(v)) for d, v in close.items() if v == v]
    splits = [(d.date(), float(r)) for d, r in t.splits.items() if r]
    return sorted(closes), sorted(splits)


def build_fixture(universe: list[str], cik_map: dict[str, str], out_path: str | Path) -> dict:
    """Live fetch for the universe; writes the 15-field fixture CSV. Returns per-name anchor
    coverage counts. Operator/Bash-run; needs network."""
    rows: list[dict] = []
    coverage: dict[str, int] = {}
    for ticker in universe:
        cik = cik_map[ticker]
        equity = _sec_concept(cik, "us-gaap", "StockholdersEquity")
        time.sleep(0.15)
        if True:  # backfill names that report only the with-NCI equity tag (PG/T/VZ/JNJ)
            incl = _sec_concept(cik, "us-gaap",
                                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
            time.sleep(0.15)
            equity = merge_concept_facts(equity, incl)
        # MERGE us-gaap (period_end, exact) with dei (cover date, near-universal coverage):
        # prefer us-gaap per accession, backfill dei for accns it misses. The us-gaap tag is
        # sparse for many names (banks/WMT have ~6-24 facts vs ~70 in dei), so either-or
        # under-covers; the cover-date window match in build_asset_rows accepts both.
        shares = _sec_concept(cik, "us-gaap", "CommonStockSharesOutstanding")
        time.sleep(0.15)
        dei = _sec_concept(cik, "dei", "EntityCommonStockSharesOutstanding")
        time.sleep(0.15)
        shares = merge_concept_facts(shares, dei)
        accepted = _sec_accepted(cik)
        closes, splits = _yf_closes_splits(ticker)
        ar = build_asset_rows(ticker, cik, equity, shares, accepted, splits, closes)
        coverage[ticker] = sum(1 for r in ar if r["concept"] == "MarketCapAnchor")
        rows.extend(ar)
        print(f"  {ticker}: {coverage[ticker]} anchor-quarters, {len(ar)} rows")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return coverage


if __name__ == "__main__":
    import pandas as pd

    UNIVERSE = [c for c in pd.read_csv(
        "apps/quant/advisor/tests/fixtures/floor_prices.csv", nrows=0).columns
        if c not in ("Date", "SPY")]
    tk = _http_json("https://www.sec.gov/files/company_tickers.json")
    cik_map = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in tk.values()}
    missing = [t for t in UNIVERSE if t not in cik_map]
    assert not missing, f"no CIK for {missing}"
    cov = build_fixture(UNIVERSE, cik_map,
                        "apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv")
    thin = {t: n for t, n in cov.items() if n < 20}
    print(f"\nDONE. names {len(cov)}; thin (<20 anchor-quarters): {thin or 'none'}")
