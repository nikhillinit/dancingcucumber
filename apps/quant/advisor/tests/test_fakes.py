from datetime import date

import pandas as pd

from advisor.tests.fakes import (
    FakeFredProvider, FakeMarketDataProvider, FakeNewsProvider,
    inverted_curve, rising_prices, steep_curve,
)


def test_market_fake_records_calls_and_returns_prices():
    p = FakeMarketDataProvider()
    df = p.get_prices("AAPL", date(2024, 1, 1), date(2024, 5, 1))
    assert "Close" in df.columns and len(df) == 260
    assert p.calls[-1] == ("prices", "AAPL", date(2024, 1, 1), date(2024, 5, 1))


def test_market_fake_fundamentals_default_none():
    assert FakeMarketDataProvider().get_fundamentals_asof("AAPL", date(2024, 5, 1)) is None


def test_fred_fake_records_and_returns_series():
    f = FakeFredProvider(steep_curve())
    s = f.get_series("T10Y2Y", date(2024, 4, 1), date(2024, 5, 1))
    assert isinstance(s, pd.Series) and float(s.iloc[-1]) > 0.5
    assert f.calls[-1] == ("T10Y2Y", date(2024, 4, 1), date(2024, 5, 1))


def test_news_fake_returns_and_can_raise():
    assert FakeNewsProvider(["a"]).get_headlines("AAPL", date(2024, 5, 1)) == ["a"]
    raiser = FakeNewsProvider(raises=True)
    try:
        raiser.get_headlines("AAPL", date(2024, 5, 1))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_curve_fixtures():
    assert float(inverted_curve().iloc[-1]) < 0
    assert float(steep_curve().iloc[-1]) > 0.5
