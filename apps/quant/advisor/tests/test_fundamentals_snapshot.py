from datetime import date
from pathlib import Path

from advisor.data.fundamentals_snapshot import (
    SnapshotForwardFundamentalsProvider,
    load_fundamental_snapshots,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fundamentals_snapshot.csv"


def _provider() -> SnapshotForwardFundamentalsProvider:
    return SnapshotForwardFundamentalsProvider.from_csv(FIXTURE)


def test_loader_preserves_point_in_time_provenance():
    records = load_fundamental_snapshots(FIXTURE)
    record = next(
        r
        for r in records
        if r.ticker == "AAPL" and r.fundamentals.period_end == date(2023, 12, 31)
    )
    assert record.snapshot_date == date(2024, 4, 15)
    assert record.source == "prototype-alpha-vantage-reference"
    assert record.fundamentals.shares_outstanding == 10


def test_report_period_is_not_available_before_reporting_lag():
    provider = _provider()
    assert provider.get_fundamentals_asof("EARLY", date(2024, 3, 1)) is None

    f = provider.get_fundamentals_asof("EARLY", date(2024, 3, 30))
    assert f is not None
    assert f.period_end == date(2023, 12, 31)


def test_future_source_snapshot_does_not_leak_even_after_lag():
    provider = _provider()

    before_snapshot = provider.get_fundamentals_asof("AAPL", date(2024, 3, 31))
    assert before_snapshot is not None
    assert before_snapshot.period_end == date(2023, 9, 30)

    after_snapshot = provider.get_fundamentals_asof("AAPL", date(2024, 4, 15))
    assert after_snapshot is not None
    assert after_snapshot.period_end == date(2023, 12, 31)


def test_future_restatement_snapshot_does_not_leak():
    provider = _provider()

    first_snapshot = provider.get_fundamentals_asof("AAPL", date(2024, 5, 1))
    assert first_snapshot is not None
    assert first_snapshot.net_income == 120

    restated_snapshot = provider.get_fundamentals_asof("AAPL", date(2024, 5, 15))
    assert restated_snapshot is not None
    assert restated_snapshot.net_income == 125


def test_newer_quarter_is_snapshot_forward_only_after_its_snapshot_date():
    provider = _provider()

    before_snapshot = provider.get_fundamentals_asof("AAPL", date(2024, 7, 1))
    assert before_snapshot is not None
    assert before_snapshot.period_end == date(2023, 12, 31)

    after_snapshot = provider.get_fundamentals_asof("AAPL", date(2024, 7, 15))
    assert after_snapshot is not None
    assert after_snapshot.period_end == date(2024, 3, 31)


def test_missing_required_fundamental_is_unavailable_not_zero_filled():
    provider = _provider()
    assert provider.get_fundamentals_asof("MSFT", date(2024, 5, 1)) is None
