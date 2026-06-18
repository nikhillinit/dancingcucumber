from datetime import date

import pandas as pd
import pytest

from advisor.backtest.continuous_signals import raw_metric
from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from advisor.research.fundamental_value import (
    bp_timely, build_fundamental_panel, make_fundamental_raw, select_asof,
)


def _rec(accession, value, accepted, available, amended=False, superseded_by=""):
    return EdgarXbrlRecord(
        asset="AAPL", cik="0000320193", accession=accession,
        form="10-K/A" if amended else "10-K",
        report_period_end=date(2016, 9, 24),
        filing_date=date.fromisoformat(accepted),
        accepted_datetime=date.fromisoformat(accepted),
        concept="StockholdersEquity", unit="USD", value=value,
        available_asof=date.fromisoformat(available),
        superseded_by=superseded_by, amended_flag=amended,
        missingness_reason="", denominator_policy="as_reported_then_split_adjusted",
    )


ORIGINAL = _rec("0000320193-16-000178", 128249000000.0, "2016-10-26", "2016-12-23",
                superseded_by="0000320193-17-000009")
AMENDMENT = _rec("0000320193-17-000009", 130000000000.0, "2017-03-15", "2017-03-15",
                 amended=True)
RECORDS = [ORIGINAL, AMENDMENT]


def test_returns_none_before_anything_available():
    # strict lag: filing was accepted 2016-10-26, but available_asof (2016-12-23) has
    # not arrived at 2016-11-01 -> not usable yet (no lookahead).
    assert select_asof(RECORDS, "AAPL", "StockholdersEquity", date(2016, 11, 1)) is None


def test_returns_original_before_amendment_is_available():
    # amendments-are-separate: between the original's and the amendment's availability,
    # the ORIGINAL is returned; the later restatement is NOT backfilled.
    r = select_asof(RECORDS, "AAPL", "StockholdersEquity", date(2017, 1, 1))
    assert r is not None and r.value == 128249000000.0
    assert r.accession == "0000320193-16-000178"


def test_returns_amendment_once_it_is_available():
    r = select_asof(RECORDS, "AAPL", "StockholdersEquity", date(2017, 6, 1))
    assert r is not None and r.value == 130000000000.0
    assert r.accession == "0000320193-17-000009"


def test_missing_key_returns_none():
    assert select_asof(RECORDS, "MSFT", "StockholdersEquity", date(2017, 6, 1)) is None
    assert select_asof(RECORDS, "AAPL", "Revenues", date(2017, 6, 1)) is None


def test_bp_timely_is_split_invariant():
    # Advisor-verified scenario. At t0: equity=1000, shares=10 (pre-split),
    # raw_close(t0)=50 -> mktcap_anchor=500, book/price(t0)=1000/500=2.0.
    # Then a 2:1 split + 20% rise: true shares=20, raw_close(t)=30 -> true mktcap(t)=600,
    # so the TRUE timely book/price = 1000/600 = 1.6667. The adjusted series (ref=present)
    # has price_adj(t)=30 and price_adj(t0)=25 (the pre-split 50 halved for the 2:1).
    # The split-invariant formula must reproduce the true value WITHOUT any share bridge:
    bp = bp_timely(equity=1000.0, mktcap_anchor=500.0, price_adj_t0=25.0, price_adj_t=30.0)
    assert bp is not None
    assert abs(bp - (1000.0 / 600.0)) < 1e-9


def test_bp_timely_neutral_on_missing_or_degenerate():
    assert bp_timely(1000.0, 500.0, 25.0, 30.0) is not None
    assert bp_timely(None, 500.0, 25.0, 30.0) is None
    assert bp_timely(1000.0, None, 25.0, 30.0) is None
    assert bp_timely(1000.0, 0.0, 25.0, 30.0) is None      # anchor <= 0
    assert bp_timely(1000.0, 500.0, 25.0, 0.0) is None      # price_adj_t <= 0


def test_bp_timely_higher_when_price_is_lower():
    # cheaper today (lower price_adj_t) => higher book-to-price (more "value")
    expensive = bp_timely(1000.0, 500.0, 25.0, 50.0)
    cheap = bp_timely(1000.0, 500.0, 25.0, 10.0)
    assert cheap > expensive


# --- T4: panel + closure -------------------------------------------------------

def _panel():
    idx = pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 11)])  # 10 trading rows
    return pd.DataFrame(
        {
            "AAA": [10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # rising
            "BBB": [50.0] * 10,
            "SPY": [100.0] * 10,
        },
        index=idx,
    )


# AAA equity + anchor both available 2020-01-05 (row index 4); BBB has no fundamentals.
PANEL_RECORDS = [
    EdgarXbrlRecord(
        asset="AAA", cik="x", accession="acc-eq", form="10-K",
        report_period_end=date(2019, 12, 31), filing_date=date(2020, 1, 5),
        accepted_datetime=date(2020, 1, 5), concept="StockholdersEquity", unit="USD",
        value=1000.0, available_asof=date(2020, 1, 5), superseded_by="",
        amended_flag=False, missingness_reason="", denominator_policy="",
    ),
    EdgarXbrlRecord(
        asset="AAA", cik="x", accession="acc-anchor", form="10-K",
        report_period_end=date(2019, 12, 31), filing_date=date(2020, 1, 5),
        accepted_datetime=date(2020, 1, 5), concept="MarketCapAnchor", unit="USD",
        value=500.0, available_asof=date(2020, 1, 5), superseded_by="",
        amended_flag=False, missingness_reason="",
        denominator_policy="as_reported_shares_x_raw_close_at_avail",
    ),
]


def test_panel_leakage_and_neutral():
    funda = build_fundamental_panel(PANEL_RECORDS, _panel(), ["AAA", "BBB"])
    # AAA: NaN before 2020-01-05 (rows 0..3), values from row 4 onward (no lookahead)
    assert funda["AAA"].iloc[:4].isna().all()
    assert funda["AAA"].iloc[4:].notna().all()
    # BBB: no fundamentals -> all NaN (neutral, never fabricated)
    assert funda["BBB"].isna().all()


def test_panel_bp_value_uses_anchor_and_adjusted_ratio():
    funda = build_fundamental_panel(PANEL_RECORDS, _panel(), ["AAA", "BBB"])
    # row 4 (2020-01-05): t0 == t, price ratio 14/14 -> bp = (1000/500)*1 = 2.0
    assert abs(funda["AAA"].iloc[4] - 2.0) < 1e-9
    # row 9 (2020-01-10): price_adj(t)=19, price_adj(t0)=14 -> bp = 2.0 * 14/19
    assert abs(funda["AAA"].iloc[9] - 2.0 * 14.0 / 19.0) < 1e-9


def test_panel_warmup_slice_alignment():
    f0 = build_fundamental_panel(PANEL_RECORDS, _panel(), ["AAA", "BBB"], warmup=0)
    f2 = build_fundamental_panel(PANEL_RECORDS, _panel(), ["AAA", "BBB"], warmup=2)
    assert len(f2) == len(f0) - 2
    # row i of the warmup=2 panel is the same calendar date as row i+2 of warmup=0
    pd.testing.assert_series_equal(
        f2["AAA"].reset_index(drop=True),
        f0["AAA"].iloc[2:].reset_index(drop=True),
        check_names=False,
    )


def test_make_fundamental_raw_positional_alignment():
    funda = build_fundamental_panel(PANEL_RECORDS, _panel(), ["AAA", "BBB"])
    raw_fn = make_fundamental_raw(funda)
    # holdout-style: full-length positional series
    full = pd.Series(range(len(funda)), name="AAA")
    out = raw_fn("fundamental_value", full)
    assert list(out.index) == list(full.index)
    assert list(out.fillna(-1).values) == list(funda["AAA"].fillna(-1).values)
    # dev-style: prefix-length positional series (dev is a prefix of prices_all)
    pref = pd.Series(range(5), name="AAA")
    outp = raw_fn("fundamental_value", pref)
    assert len(outp) == 5
    assert list(outp.fillna(-1).values) == list(funda["AAA"].iloc[:5].fillna(-1).values)
    # unknown asset -> all NaN (neutral)
    assert raw_fn("fundamental_value", pd.Series(range(5), name="ZZZ")).isna().all()


def test_raw_fn_routes_momentum_to_frozen_price_metric():
    # the candidate is (fundamental_value, momentum); raw_fn must serve BOTH. momentum
    # must delegate to the frozen price metric, NOT return book-to-price.
    funda = build_fundamental_panel(PANEL_RECORDS, _panel(), ["AAA", "BBB"])
    raw_fn = make_fundamental_raw(funda)
    s = pd.Series([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19], name="AAA")
    pd.testing.assert_series_equal(raw_fn("momentum", s), raw_metric("momentum", s))


def test_raw_fn_raises_on_warmup_mismatch():
    funda = build_fundamental_panel(PANEL_RECORDS, _panel(), ["AAA", "BBB"], warmup=6)  # 4 rows
    raw_fn = make_fundamental_raw(funda)
    with pytest.raises(ValueError):
        raw_fn("fundamental_value", pd.Series(range(8), name="AAA"))  # longer than panel
