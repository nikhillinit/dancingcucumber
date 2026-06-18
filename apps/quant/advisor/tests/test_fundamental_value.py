from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from advisor.research.fundamental_value import select_asof


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
