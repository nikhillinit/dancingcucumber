from datetime import date

from advisor.cli import run
from advisor.data.fakes import FakeProvider
from advisor.data.provider import Fundamentals


def test_run_emits_direction_and_disclosures():
    f = Fundamentals(period_end=date(2023, 12, 31), net_income=200, total_equity=1000,
                     revenue=800, operating_income=200, total_debt=200, depreciation=60,
                     capex=40, shares_outstanding=10, market_cap=500)
    provider = FakeProvider(fundamentals={"AAPL": f})
    out = run(provider, "AAPL", date(2024, 5, 1))
    assert "AAPL" in out
    assert "bullish" in out
    assert "DISCLOSURES:" in out


def test_run_handles_missing_fundamentals():
    out = run(FakeProvider(), "ZZZZ", date(2024, 5, 1))
    assert "no point-in-time fundamentals" in out
    assert "DISCLOSURES:" in out
