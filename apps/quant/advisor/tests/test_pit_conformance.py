from datetime import date, timedelta

from advisor.data import provider
from advisor.data.provider import Fundamentals, select_latest_available
from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from advisor.research.fundamental_value import select_asof


def _fund(period_end: date) -> Fundamentals:
    return Fundamentals(
        period_end=period_end,
        net_income=100,
        total_equity=500,
        revenue=1000,
        operating_income=200,
        total_debt=100,
        depreciation=50,
        capex=40,
        shares_outstanding=10,
        market_cap=900,
    )


def _edgar_record(accession: str, available_asof: date) -> EdgarXbrlRecord:
    return EdgarXbrlRecord(
        asset="AAPL",
        cik="0000320193",
        accession=accession,
        form="10-K",
        report_period_end=date(2024, 1, 1),
        filing_date=date(2024, 3, 1),
        accepted_datetime=date(2024, 3, 1),
        concept="StockholdersEquity",
        unit="USD",
        value=1000.0,
        available_asof=available_asof,
        superseded_by="",
        amended_flag=False,
        missingness_reason="",
        denominator_policy="as_reported_then_split_adjusted",
    )


def test_reporting_lag_covers_sec_filing_deadlines():
    # The lag must cover the latest SEC deadline for large accelerated filers
    # (10-K: 60 days; 10-Q: 40 days); below 60 reintroduces filing-date lookahead.
    assert provider.REPORTING_LAG_DAYS >= 60


def test_is_available_asof_exact_boundary_day():
    period_end = date(2024, 1, 1)

    assert provider.is_available_asof(period_end, date(2024, 3, 31)) is True
    assert provider.is_available_asof(period_end, date(2024, 3, 30)) is False


def test_select_latest_available_never_leaks():
    as_of = date(2024, 3, 31)
    records = [
        _fund(date(2023, 12, 15)),
        _fund(date(2024, 1, 1)),
        _fund(date(2024, 1, 2)),
    ]

    chosen = select_latest_available(records, as_of)

    assert chosen is not None
    assert chosen.period_end + timedelta(days=provider.REPORTING_LAG_DAYS) <= as_of
    assert chosen.period_end == date(2024, 1, 1)
    assert select_latest_available([_fund(date(2024, 1, 2))], as_of) is None


def test_edgar_select_asof_exact_boundary_day():
    as_of = date(2024, 4, 1)

    selected = select_asof(
        [_edgar_record("0000320193-24-000001", as_of)],
        "AAPL",
        "StockholdersEquity",
        as_of,
    )
    not_yet_available = select_asof(
        [_edgar_record("0000320193-24-000002", as_of + timedelta(days=1))],
        "AAPL",
        "StockholdersEquity",
        as_of,
    )

    assert selected is not None
    assert selected.available_asof == as_of
    assert not_yet_available is None
