from __future__ import annotations

from datetime import date, timedelta
from typing import Awaitable, Callable

import pandas as pd

from advisor.analysis import macro, momentum, sentiment, trend, value_quality
from advisor.data.fred_provider import FRED_SERIES_T10Y2Y, FredProvider
from advisor.data.news_provider import NewsProvider
from advisor.data.provider import MarketDataProvider
from advisor.schemas import FamilySignal

FamilyCoro = Callable[[date], Awaitable[FamilySignal]]

PRICE_LOOKBACK_DAYS = 420   # > trend MIN_HISTORY (201 trading days)
MACRO_LOOKBACK_DAYS = 30    # daily T10Y2Y; macro needs only 2 points


def close_series(df) -> pd.Series:
    """Extract a 1-D close price Series from a provider price frame."""
    if isinstance(df, pd.Series):
        return df
    col = df["Close"] if "Close" in getattr(df, "columns", []) else df.iloc[:, 0]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col


def make_value_quality_coro(provider: MarketDataProvider, ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            f = provider.get_fundamentals_asof(ticker, as_of)
            if f is None:
                return FamilySignal.neutral(value_quality.FAMILY, as_of, "no fundamentals available")
            return value_quality.evaluate(f, as_of)
        except Exception as e:
            return FamilySignal.neutral(value_quality.FAMILY, as_of, f"value_quality unavailable: {e!s}")
    return coro


def make_trend_coro(provider: MarketDataProvider, ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            df = provider.get_prices(ticker, as_of - timedelta(days=PRICE_LOOKBACK_DAYS), as_of)
            return trend.evaluate(close_series(df), as_of)
        except Exception as e:
            return FamilySignal.neutral(trend.FAMILY, as_of, f"trend unavailable: {e!s}")
    return coro


def make_momentum_coro(provider: MarketDataProvider, ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            df = provider.get_prices(ticker, as_of - timedelta(days=PRICE_LOOKBACK_DAYS), as_of)
            return momentum.evaluate(close_series(df), as_of)
        except Exception as e:
            return FamilySignal.neutral(momentum.FAMILY, as_of, f"momentum unavailable: {e!s}")
    return coro


def make_macro_coro(fred: FredProvider, series_id: str = FRED_SERIES_T10Y2Y) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            s = fred.get_series(series_id, as_of - timedelta(days=MACRO_LOOKBACK_DAYS), as_of)
            return macro.evaluate(s, as_of)
        except Exception as e:
            return FamilySignal.neutral(macro.FAMILY, as_of, f"macro unavailable: {e!s}")
    return coro


def make_sentiment_coro(news: NewsProvider, scorer: Callable[[str], float], ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            headlines = news.get_headlines(ticker, as_of)
            return sentiment.evaluate(headlines, as_of, scorer)
        except Exception as e:
            return FamilySignal.neutral(sentiment.FAMILY, as_of, f"sentiment unavailable: {e!s}")
    return coro
