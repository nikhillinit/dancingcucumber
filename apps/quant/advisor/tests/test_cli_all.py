from datetime import date

from advisor.analysis.news_scorer import lexicon_score
from advisor.cli import run_all
from advisor.personas.overlay import PersonaVerdict
from advisor.tests.fakes import FakeFredProvider, FakeMarketDataProvider, FakeNewsProvider, steep_curve

AS_OF = date(2024, 5, 1)


def _providers():
    return (
        FakeMarketDataProvider(fundamentals=None),  # value/quality -> neutral; prices rise -> trend/mom bullish
        FakeFredProvider(steep_curve()),            # macro bullish
        FakeNewsProvider(["Earnings beat estimates", "Record profit"]),  # sentiment bullish
    )


def test_run_all_produces_nonzero_buy_with_disclosure():
    market, fred, news = _providers()
    out = run_all(market, fred, news, lexicon_score, "AAPL", AS_OF,
                  net_liq=100_000.0, vol=0.20, correlation=0.50)
    assert "AAPL" in out
    assert "buy qty=" in out
    qty = int(out.split("qty=")[1].split(" ")[0])
    assert qty > 0  # ensemble bullish -> allocate exercised, not vacuous
    assert "DISCLOSURES" in out


def test_run_all_explain_appends_persona():
    market, fred, news = _providers()
    critic = lambda d: PersonaVerdict(1.0, f"{d.bundle_direction} per 5-family ensemble")
    out = run_all(market, fred, news, lexicon_score, "AAPL", AS_OF,
                  net_liq=100_000.0, vol=0.20, correlation=0.50, persona_critic=critic)
    assert "persona" in out.lower()
