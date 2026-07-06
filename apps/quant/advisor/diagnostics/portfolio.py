from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAX_DAILY_MOVE = 0.40


@dataclass(frozen=True)
class LoadedPortfolio:
    returns: pd.Series
    weights_book: pd.DataFrame
    n_obs: int
    dropped_dates: int
    tickers: list[str]
    cash_dollars: float | None
    start: date
    end: date


class DiagnosticsInputError(ValueError):
    pass


def load_portfolio(positions_path: str | Path, prices_path: str | Path) -> LoadedPortfolio:
    positions_raw = _read_csv(positions_path, "positions")
    prices_raw = _read_csv(prices_path, "prices")

    price_frame, price_columns = _parse_prices(prices_raw)
    tickers, qty, cash_dollars = _parse_positions(positions_raw)

    missing = [ticker for ticker in tickers if ticker not in price_columns]
    if missing:
        raise DiagnosticsInputError(f"unknown ticker in positions: {missing[0]} missing from prices")

    prices = pd.DataFrame(
        {
            ticker: pd.to_numeric(price_frame[price_columns[ticker]], errors="coerce")
            for ticker in tickers
        },
        index=price_frame.index,
    )
    prices = prices.sort_index()

    _reject_partial_history(prices)
    common_prices = _common_price_window(prices)
    _reject_bad_frequency(common_prices.index)
    _reject_large_daily_moves(common_prices)

    market_values = common_prices.multiply(qty, axis="columns")
    cash_value = 0.0 if cash_dollars is None else cash_dollars
    equity = market_values.sum(axis=1) + cash_value

    # intentional duplication of the pct_change one-liner; backtest/** is frozen
    returns = equity.pct_change().dropna()
    returns.name = "returns"

    weights_book = market_values.divide(equity, axis="index")
    weights_book = weights_book.loc[:, tickers]

    return LoadedPortfolio(
        returns=returns,
        weights_book=weights_book,
        n_obs=len(returns),
        dropped_dates=len(prices.loc[common_prices.index[0] : common_prices.index[-1]]) - len(common_prices),
        tickers=tickers,
        cash_dollars=cash_dollars,
        start=common_prices.index[0].date(),
        end=common_prices.index[-1].date(),
    )


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DiagnosticsInputError(f"{label} CSV parse error: {exc}") from exc


def _column_map(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in columns:
            raise DiagnosticsInputError(f"{label} CSV parse error: duplicate column {key}")
        columns[key] = column
    return columns


def _parse_positions(frame: pd.DataFrame) -> tuple[list[str], pd.Series, float | None]:
    columns = _column_map(frame, "positions")
    if "ticker" not in columns:
        raise DiagnosticsInputError("positions CSV parse error: missing ticker column")
    for weight_column in ("weight", "weights"):
        if weight_column in columns:
            raise DiagnosticsInputError(
                f"weights input rejected: column {weight_column} is not supported; use qty"
            )
    if "qty" not in columns:
        raise DiagnosticsInputError("weights input rejected: missing qty column")

    positions = pd.DataFrame(
        {
            "ticker": frame[columns["ticker"]].astype(str).str.strip().str.upper(),
            "qty": pd.to_numeric(frame[columns["qty"]], errors="coerce"),
        }
    )
    if positions.empty or positions["ticker"].eq("").any() or positions["qty"].isna().any():
        raise DiagnosticsInputError("positions CSV parse error: malformed ticker or qty")

    bad_qty = positions[positions["qty"] <= 0]
    if not bad_qty.empty:
        ticker = str(bad_qty.iloc[0]["ticker"])
        qty = float(bad_qty.iloc[0]["qty"])
        if qty < 0:
            raise DiagnosticsInputError(f"shorts rejected: {ticker} qty must be positive")
        raise DiagnosticsInputError(f"non-positive qty rejected: {ticker} qty must be positive")

    duplicate = positions[positions["ticker"].duplicated()]["ticker"]
    if not duplicate.empty:
        raise DiagnosticsInputError(f"duplicate ticker rejected: {duplicate.iloc[0]}")

    cash_rows = positions[positions["ticker"] == "CASH"]
    cash_dollars = None if cash_rows.empty else float(cash_rows.iloc[0]["qty"])

    non_cash = positions[positions["ticker"] != "CASH"].copy()
    if non_cash.empty:
        raise DiagnosticsInputError("positions CSV parse error: at least one non-CASH ticker is required")
    non_cash = non_cash.sort_values("ticker")

    tickers = non_cash["ticker"].tolist()
    qty = pd.Series(non_cash["qty"].to_numpy(dtype=float), index=tickers)
    return tickers, qty, cash_dollars


def _parse_prices(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty or len(frame.columns) < 2:
        raise DiagnosticsInputError("prices CSV parse error: expected Date plus ticker columns")

    first_column = frame.columns[0]
    if str(first_column).strip().lower() != "date":
        raise DiagnosticsInputError("prices CSV parse error: first column must be Date")

    parsed_dates = pd.to_datetime(frame[first_column], errors="coerce")
    if parsed_dates.isna().any():
        raise DiagnosticsInputError("prices CSV parse error: malformed Date")

    prices = frame.drop(columns=[first_column]).copy()
    prices.index = parsed_dates
    if prices.index.has_duplicates:
        raise DiagnosticsInputError("prices CSV parse error: duplicate Date")

    columns: dict[str, Any] = {}
    for column in prices.columns:
        ticker = str(column).strip().upper()
        if not ticker:
            raise DiagnosticsInputError("prices CSV parse error: empty ticker column")
        if ticker in columns:
            raise DiagnosticsInputError(f"prices CSV parse error: duplicate ticker column {ticker}")
        columns[ticker] = column
    return prices, columns


def _reject_partial_history(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise DiagnosticsInputError("empty common price window")

    first_date = prices.index[0]
    last_date = prices.index[-1]
    for ticker in prices.columns:
        series = prices[ticker]
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        if first_valid is None or last_valid is None:
            raise DiagnosticsInputError(f"partial history for {ticker}: no adjusted price data")
        if first_valid != first_date or last_valid != last_date:
            raise DiagnosticsInputError(
                f"partial history for {ticker}: adjusted prices must span full window"
            )


def _common_price_window(prices: pd.DataFrame) -> pd.DataFrame:
    common = prices.dropna(how="any")
    if common.empty:
        raise DiagnosticsInputError("empty common price window")
    start = common.index[0]
    end = common.index[-1]
    return prices.loc[start:end].dropna(how="any")


def _reject_bad_frequency(index: pd.DatetimeIndex) -> None:
    if len(index) < 2:
        return
    gaps = pd.Series(index).diff().dropna().dt.days
    median_gap = float(gaps.median())
    if median_gap < 1 or median_gap > 4:
        raise DiagnosticsInputError(
            f"prices frequency error: median calendar-day gap {median_gap:g} outside [1, 4]"
        )


def _reject_large_daily_moves(prices: pd.DataFrame) -> None:
    moves = prices.pct_change().abs()
    violations = np.argwhere((moves > MAX_DAILY_MOVE).to_numpy())
    if violations.size == 0:
        return
    row, col = violations[0]
    ticker = prices.columns[col]
    move_date = prices.index[row].date().isoformat()
    move = moves.iloc[row, col]
    raise DiagnosticsInputError(
        f"phantom split guard tripped for {ticker} on {move_date}: "
        f"daily move {move:.2%} exceeds 40%; prices must be split+dividend-adjusted"
    )
