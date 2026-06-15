from datetime import date

from advisor.data.provider import (
    Fundamentals,
    is_available_asof,
    select_latest_available,
)
from advisor.data.fakes import FakeProvider


def _fund(period_end: date) -> Fundamentals:
    return Fundamentals(period_end=period_end, net_income=100, total_equity=500,
                        revenue=1000, operating_income=200, total_debt=100,
                        depreciation=50, capex=40, shares_outstanding=10, market_cap=900)


def test_availability_guard_blocks_too_recent():
    # period ends 2024-01-01; with 90-day lag it is only available from ~2024-03-31
    assert is_available_asof(date(2024, 1, 1), date(2024, 2, 1)) is False
    assert is_available_asof(date(2024, 1, 1), date(2024, 5, 1)) is True


def test_select_latest_available_picks_newest_lagged_record():
    records = [_fund(date(2023, 9, 30)), _fund(date(2023, 12, 31)), _fund(date(2024, 3, 31))]
    # as of 2024-04-15 only the first two are >90 days old
    chosen = select_latest_available(records, date(2024, 4, 15))
    assert chosen is not None
    assert chosen.period_end == date(2023, 12, 31)


def test_fake_provider_returns_configured_fundamentals():
    p = FakeProvider(fundamentals={"AAPL": _fund(date(2023, 12, 31))})
    f = p.get_fundamentals_asof("AAPL", date(2024, 5, 1))
    assert f is not None and f.revenue == 1000
    assert p.get_fundamentals_asof("MSFT", date(2024, 5, 1)) is None
