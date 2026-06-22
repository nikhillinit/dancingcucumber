from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from advisor.data.provider import YFinanceProvider

PriceGetter = Callable[[str, date, date], pd.DataFrame]

_DEFAULT_PROVIDER = YFinanceProvider()
DEFAULT_GETTER: PriceGetter = _DEFAULT_PROVIDER.get_prices


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _price_column(frame: pd.DataFrame, name: str) -> pd.Series | pd.DataFrame:
    if name in frame.columns:
        return frame[name]
    if isinstance(frame.columns, pd.MultiIndex):
        matches = [col for col in frame.columns if name in col]
        if matches:
            return frame.loc[:, matches[0]]
    raise ValueError(f"price frame missing {name!r} column")


def _adjusted_close(frame: pd.DataFrame) -> pd.Series:
    try:
        column = _price_column(frame, "Adj Close")
    except ValueError:
        column = _price_column(frame, "Close")
    if isinstance(column, pd.DataFrame):
        column = column.iloc[:, 0]
    series = pd.Series(column).astype("float64").dropna()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def build_price_fixture(
    universe: list[str],
    out_path: str | Path,
    start: str | date = "2015-01-01",
    end: str | date = "2024-01-01",
    getter: PriceGetter = DEFAULT_GETTER,
) -> dict:
    """Fetch adjusted closes and write a floor_prices-shaped CSV.

    Network behavior is isolated behind `getter`; tests inject a fake getter.
    """
    start_date = _as_date(start)
    end_date = _as_date(end)
    names = [ticker for ticker in universe if ticker != "SPY"]
    tickers = names + ["SPY"]
    prices = {
        ticker: _adjusted_close(getter(ticker, start_date, end_date))
        for ticker in tickers
    }

    calendar = prices["SPY"].index
    if calendar.empty:
        raise ValueError("SPY returned no price rows")

    coverage: dict[str, int] = {}
    dropped: list[str] = []
    kept: list[str] = []
    columns: dict[str, pd.Series] = {}
    for ticker in names:
        aligned = prices[ticker].reindex(calendar)
        coverage[ticker] = int(aligned.notna().sum())
        if aligned.isna().any():
            dropped.append(ticker)
            continue
        kept.append(ticker)
        columns[ticker] = aligned.astype("float64")

    spy = prices["SPY"].reindex(calendar)
    coverage["SPY"] = int(spy.notna().sum())
    if spy.isna().any():
        raise ValueError("SPY lacks full-window price coverage")
    columns["SPY"] = spy.astype("float64")

    panel = pd.DataFrame(columns, index=calendar).loc[:, kept + ["SPY"]]
    panel.index = pd.Index(pd.to_datetime(panel.index).date, name="Date")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out)
    return {"coverage": coverage, "dropped": dropped, "kept": kept + ["SPY"], "rows": len(panel)}
