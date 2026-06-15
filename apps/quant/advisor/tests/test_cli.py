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


def test_run_with_explain_appends_persona_line():
    from datetime import date

    from advisor.cli import run
    from advisor.data.fakes import FakeProvider
    from advisor.data.provider import Fundamentals
    from advisor.personas.overlay import PersonaVerdict

    f = Fundamentals(period_end=date(2023, 12, 31), net_income=10.0, total_equity=100.0,
                     revenue=200.0, operating_income=30.0, total_debt=20.0, depreciation=5.0,
                     capex=4.0, shares_outstanding=10.0, market_cap=500.0)
    provider = FakeProvider(fundamentals={"AAPL": f})

    out = run(provider, "AAPL", date(2024, 6, 1),
              critic=lambda sig: PersonaVerdict(1.0, "value+quality concur"))
    assert "value+quality concur" in out
