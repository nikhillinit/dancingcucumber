from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.data.provider import Fundamentals


def rising_prices(n: int = 260, start: float = 50.0, step: float = 0.5) -> pd.DataFrame:
    closes = [start + step * i for i in range(n)]
    idx = pd.date_range(end="2024-05-01", periods=n, freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


def steep_curve() -> pd.Series:
    idx = pd.date_range(end="2024-05-01", periods=5, freq="D")
    return pd.Series([0.8, 0.9, 1.0, 1.1, 1.2], index=idx)  # latest > 0.5 -> bullish


def inverted_curve() -> pd.Series:
    idx = pd.date_range(end="2024-05-01", periods=5, freq="D")
    return pd.Series([-0.1, -0.2, -0.3, -0.4, -0.5], index=idx)  # latest < 0 -> bearish


class FakeMarketDataProvider:
    def __init__(self, prices: pd.DataFrame | None = None, fundamentals: Fundamentals | None = None) -> None:
        self._prices = prices if prices is not None else rising_prices()
        self._fundamentals = fundamentals
        self.calls: list[tuple] = []

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append(("prices", ticker, start, end))
        return self._prices

    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None:
        self.calls.append(("fundamentals", ticker, as_of))
        return self._fundamentals


class FakeFredProvider:
    def __init__(self, series: pd.Series | None = None) -> None:
        self._series = series if series is not None else pd.Series(dtype="float64")
        self.calls: list[tuple] = []

    def get_series(self, series_id: str, start: date, end: date) -> pd.Series:
        self.calls.append((series_id, start, end))
        return self._series


class FakeNewsProvider:
    def __init__(self, headlines: list[str] | None = None, raises: bool = False) -> None:
        self._headlines = list(headlines or [])
        self._raises = raises
        self.calls: list[tuple] = []

    def get_headlines(self, ticker: str, as_of: date) -> list[str]:
        self.calls.append((ticker, as_of))
        if self._raises:
            raise RuntimeError("boom")
        return list(self._headlines)
