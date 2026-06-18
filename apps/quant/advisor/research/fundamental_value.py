from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd

from advisor.backtest.continuous_signals import raw_metric
from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord

FUNDAMENTAL_VALUE = "fundamental_value"


def select_asof(
    records: list[EdgarXbrlRecord],
    asset: str,
    concept: str,
    as_of: date,
) -> EdgarXbrlRecord | None:
    """Point-in-time selection honoring the WS3A named rules (spec lines 74-83):

    - strict-lag != PIT / no lookahead: a record is usable only once its canonical
      `available_asof` has arrived (available_asof <= as_of). available_asof already
      folds in the conservative +90d reporting lag, the filing date, and the accepted
      datetime, so this is the honest as-of gate.
    - amendments-are-separate: an amendment/restatement is its own record with its own
      (later) available_asof. Before that date the ORIGINAL is returned; a later
      restatement is never backfilled to an earlier as_of.

    Returns the latest-available record for (asset, concept), or None when nothing for
    that key is knowable yet (missing -> None, never fabricated).
    """
    eligible = [
        r for r in records
        if r.asset == asset and r.concept == concept and r.available_asof <= as_of
    ]
    if not eligible:
        return None
    # latest available wins; deterministic accession tie-break for same-day availability
    return max(eligible, key=lambda r: (r.available_asof, r.accession))


def bp_timely(
    equity: float | None,
    mktcap_anchor: float | None,
    price_adj_t0: float | None,
    price_adj_t: float | None,
) -> float | None:
    """Split-INVARIANT timely book-to-price.

        bp_timely(t) = (equity / mktcap_anchor) * price_adj(t0) / price_adj(t)

    `equity` (StockholdersEquity) and `mktcap_anchor` (= shares_asof * RAW close at the
    filing's availability t0) are real-dollar aggregates -> split-invariant. The
    `price_adj(t0)/price_adj(t)` ratio uses ONE consistent adjusted series -> split-
    neutral. There is NO per-share division across a split date, so no split-basis
    bridge is needed. t0 = the anchor's available_asof; select_asof re-anchors at each
    new filing, so the rescale window is < ~1 quarter and auto_adjust dividend drift in
    the ratio is negligible (intentional approximation; do not gold-plate).

    Returns None on any missing input or degenerate denominator (mktcap_anchor <= 0 or
    price_adj_t <= 0) -> neutral, never fabricated.
    """
    if equity is None or mktcap_anchor is None or price_adj_t0 is None or price_adj_t is None:
        return None
    if mktcap_anchor <= 0 or price_adj_t <= 0:
        return None
    return (equity / mktcap_anchor) * (price_adj_t0 / price_adj_t)


def build_fundamental_panel(
    records: list[EdgarXbrlRecord],
    panel: pd.DataFrame,
    assets: list[str] | None = None,
    *,
    warmup: int = 0,
    equity_concept: str = "StockholdersEquity",
    anchor_concept: str = "MarketCapAnchor",
) -> pd.DataFrame:
    """Date-indexed `bp_timely` panel aligned row-for-row to `panel` (floor_prices, a
    DatetimeIndex of adjusted closes), then sliced `[warmup:]` and reset to a positional
    RangeIndex so it shares candidate_pipeline's `prices_all = panel[assets].iloc[warmup:]
    .reset_index(drop=True)` basis. As-of per row (re-anchored each quarter by select_asof);
    unavailable -> NaN (the percentile transform maps NaN -> 0 -> flat). Date threading is
    done HERE, never inside raw_fn.
    """
    if assets is None:
        assets = [c for c in panel.columns if c != "SPY"]
    dates = list(panel.index)
    cols: dict[str, list[float]] = {}
    for a in assets:
        # pre-filter once per asset (keeps select_asof's scan small; precompute a step
        # function if this O(assets*dates*records) dominates at full WS4 scale).
        a_records = [r for r in records if r.asset == a]
        price = panel[a]
        col: list[float] = []
        for t in dates:
            as_of = t.date() if hasattr(t, "date") else t
            eq = select_asof(a_records, a, equity_concept, as_of)
            anc = select_asof(a_records, a, anchor_concept, as_of)
            if eq is None or anc is None:
                col.append(float("nan"))
                continue
            t0_ts = pd.Timestamp(anc.available_asof)
            prior = price.loc[price.index <= t0_ts]
            if prior.empty:
                col.append(float("nan"))
                continue
            bp = bp_timely(eq.value, anc.value, float(prior.iloc[-1]), float(price.loc[t]))
            col.append(bp if bp is not None else float("nan"))
        cols[a] = col
    funda = pd.DataFrame(cols, index=range(len(dates)))
    return funda.iloc[warmup:].reset_index(drop=True)


def make_fundamental_raw(
    panel_funda: pd.DataFrame,
    base_raw_fn: Callable[[str, pd.Series], pd.Series] = raw_metric,
) -> Callable[[str, pd.Series], pd.Series]:
    """Build the `raw_fn` candidate_pipeline expects: `(family, prices) -> Series`.

    The candidate has TWO families (`fundamental_value`, `momentum`), and the pipeline
    calls this for BOTH. Dispatch: `fundamental_value` -> precomputed panel lookup; any
    other family -> `base_raw_fn` (the frozen price `raw_metric`, which handles momentum).
    The pipeline passes a POSITIONAL Series (RangeIndex, `.name`=asset); `dev` is a prefix
    of `prices_all`, so `panel_funda[name].reindex(prices.index)` aligns for both the dev
    sweep and the holdout call. Unknown asset -> all-NaN (neutral). A panel shorter than
    `prices` means a warmup mismatch -> raise loudly rather than silently misalign.
    """
    def _raw(family: str, prices: pd.Series) -> pd.Series:
        if family == FUNDAMENTAL_VALUE:
            name = prices.name
            if name not in panel_funda.columns:
                return pd.Series([float("nan")] * len(prices), index=prices.index, name=name)
            if len(panel_funda) < len(prices):
                raise ValueError(
                    f"panel_funda rows ({len(panel_funda)}) < prices rows ({len(prices)}); "
                    "fundamental panel and prices_all must share cfg.warmup"
                )
            return panel_funda[name].reindex(prices.index)
        return base_raw_fn(family, prices)
    return _raw
