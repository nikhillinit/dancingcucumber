from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from advisor.research.fundamental_value import bp_timely, select_asof


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
