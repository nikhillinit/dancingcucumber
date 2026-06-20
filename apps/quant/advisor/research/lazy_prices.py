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

import pandas as pd

from advisor.research.fundamental_value import select_asof  # generic PIT selector


def build_lazy_prices_panel(
    records: list[EdgarXbrlRecord],
    panel: pd.DataFrame,
    assets: list[str] | None = None,
    *,
    warmup: int = 0,
    concept: str = SIMILARITY_CONCEPT,
) -> pd.DataFrame:
    """Date-indexed similarity panel aligned row-for-row to `panel` (floor_prices, a
    DatetimeIndex of adjusted closes), sliced [warmup:] and reset to a positional
    RangeIndex so it shares candidate_pipeline's
    `prices_all = panel[assets].iloc[warmup:].reset_index(drop=True)` basis. Per row:
    the latest FilingSimilarity available as-of that date (select_asof; a STEP FUNCTION
    held between filings). Unavailable -> NaN (the percentile transform maps NaN -> 0 ->
    flat). NO price recompute and NO split handling: similarity is a dimensionless ratio,
    basis-free. Date threading is done HERE, never inside raw_fn."""
    if assets is None:
        assets = [c for c in panel.columns if c != "SPY"]
    dates = list(panel.index)
    cols: dict[str, list[float]] = {}
    for a in assets:
        a_records = [r for r in records if r.asset == a]  # shrink select_asof's scan
        col: list[float] = []
        for t in dates:
            as_of = t.date() if hasattr(t, "date") else t
            rec = select_asof(a_records, a, concept, as_of)
            col.append(float(rec.value) if rec is not None else float("nan"))
        cols[a] = col
    funda = pd.DataFrame(cols, index=range(len(dates)))
    return funda.iloc[warmup:].reset_index(drop=True)

from typing import Callable

from advisor.backtest.continuous_signals import raw_metric


def make_lazy_prices_raw(
    panel_lp: pd.DataFrame,
    base_raw_fn: Callable[[str, pd.Series], pd.Series] = raw_metric,
) -> Callable[[str, pd.Series], pd.Series]:
    """Build the `raw_fn` candidate_pipeline expects: `(family, prices) -> Series`.

    Dispatch: `lazy_prices` -> precomputed panel lookup; any other family -> base_raw_fn
    (the frozen price raw_metric, which handles momentum). SIGN: raw = the similarity
    itself — HIGH similarity = 'non-changer' = the long leg, so the percentile transform
    ranks high-similarity names long. Using `1 - similarity` would invert to the short
    leg. The pipeline passes a POSITIONAL Series (RangeIndex, `.name`=asset); panel_lp is
    aligned to the same warmup-sliced basis, so `.reindex(prices.index)` aligns for both
    the dev sweep and the holdout. Unknown asset -> all-NaN (neutral). A panel shorter
    than prices means a warmup mismatch -> raise loudly rather than silently misalign."""
    def _raw(family: str, prices: pd.Series) -> pd.Series:
        if family == LAZY_PRICES:
            name = prices.name
            if name not in panel_lp.columns:
                return pd.Series([float("nan")] * len(prices), index=prices.index, name=name)
            if len(panel_lp) < len(prices):
                raise ValueError(
                    f"panel_lp rows ({len(panel_lp)}) < prices rows ({len(prices)}); "
                    "lazy-prices panel and prices_all must share cfg.warmup"
                )
            return panel_lp[name].reindex(prices.index)
        return base_raw_fn(family, prices)
    return _raw
