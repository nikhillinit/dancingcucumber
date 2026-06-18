from __future__ import annotations

from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord


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
