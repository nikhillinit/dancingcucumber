import asyncio
from datetime import date

from advisor.analysis.news_scorer import lexicon_score
from advisor.pipeline.families import (
    make_macro_coro, make_momentum_coro, make_sentiment_coro,
    make_trend_coro, make_value_quality_coro,
)
from advisor.schemas import Direction
from advisor.tests.fakes import (
    FakeFredProvider, FakeMarketDataProvider, FakeNewsProvider,
    inverted_curve, rising_prices, steep_curve,
)

AS_OF = date(2024, 5, 1)


def _run(coro):
    return asyncio.run(coro(AS_OF))


def test_trend_coro_bullish_and_bounds_end_at_as_of():
    p = FakeMarketDataProvider(prices=rising_prices())
    sig = _run(make_trend_coro(p, "AAPL"))
    assert sig.direction is Direction.BULLISH
    assert p.calls[-1][0] == "prices" and p.calls[-1][3] == AS_OF  # end == as_of


def test_momentum_coro_bullish():
    sig = _run(make_momentum_coro(FakeMarketDataProvider(prices=rising_prices()), "AAPL"))
    assert sig.direction is Direction.BULLISH


def test_value_quality_coro_none_fundamentals_is_neutral():
    sig = _run(make_value_quality_coro(FakeMarketDataProvider(fundamentals=None), "AAPL"))
    assert sig.direction is Direction.NEUTRAL


def test_macro_coro_bounds_end_at_as_of_and_reads_curve():
    fred = FakeFredProvider(inverted_curve())
    sig = _run(make_macro_coro(fred))
    assert sig.direction is Direction.BEARISH
    assert fred.calls[-1][0] == "T10Y2Y" and fred.calls[-1][2] == AS_OF  # end == as_of


def test_macro_coro_empty_series_is_neutral():
    assert _run(make_macro_coro(FakeFredProvider())).direction is Direction.NEUTRAL


def test_steep_curve_macro_is_bullish():
    assert _run(make_macro_coro(FakeFredProvider(steep_curve()))).direction is Direction.BULLISH


def test_sentiment_coro_empty_is_neutral_not_fabricated():
    news = FakeNewsProvider([])
    sig = _run(make_sentiment_coro(news, lexicon_score, "AAPL"))
    assert sig.direction is Direction.NEUTRAL
    assert news.calls[-1] == ("AAPL", AS_OF)  # as_of passed through for bounding


def test_sentiment_coro_positive_news_is_bullish():
    news = FakeNewsProvider(["Earnings beat estimates", "Record profit"])
    assert _run(make_sentiment_coro(news, lexicon_score, "AAPL")).direction is Direction.BULLISH


def test_coro_swallows_exception_into_neutral():
    sig = _run(make_sentiment_coro(FakeNewsProvider(raises=True), lexicon_score, "AAPL"))
    assert sig.direction is Direction.NEUTRAL
