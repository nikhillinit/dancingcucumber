from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

REPORTING_LAG_DAYS = 90


@dataclass(frozen=True)
class Fundamentals:
    period_end: date
    net_income: float
    total_equity: float
    revenue: float
    operating_income: float
    total_debt: float
    depreciation: float
    capex: float
    shares_outstanding: float
    market_cap: float


def is_available_asof(period_end: date, as_of: date, lag_days: int = REPORTING_LAG_DAYS) -> bool:
    """A report is only knowable after period_end + a conservative reporting lag."""
    return period_end + timedelta(days=lag_days) <= as_of


def select_latest_available(records: list[Fundamentals], as_of: date,
                            lag_days: int = REPORTING_LAG_DAYS) -> Fundamentals | None:
    eligible = [r for r in records if is_available_asof(r.period_end, as_of, lag_days)]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r.period_end)


class MarketDataProvider(Protocol):
    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...
    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None: ...


class YFinanceProvider:
    """Thin yfinance adapter. Network-bound; verified manually, not in unit tests.

    yfinance fundamentals are RESTATED, not as-reported - the point-in-time guard
    above only approximates availability. See spec section 6 disclosures.
    """

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=True)
        return df

    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        fin = t.financials  # columns are period-end dates
        records: list[Fundamentals] = []
        for col in getattr(fin, "columns", []):
            period_end = col.date() if hasattr(col, "date") else col
            try:
                records.append(Fundamentals(
                    period_end=period_end,
                    net_income=float(fin.loc["Net Income", col]),
                    total_equity=float(info.get("totalStockholderEquity") or 0) or 1.0,
                    revenue=float(fin.loc["Total Revenue", col]),
                    operating_income=float(fin.loc["Operating Income", col]),
                    total_debt=float(info.get("totalDebt") or 0),
                    depreciation=float(info.get("ebitda", 0)) - float(info.get("operatingCashflow", 0) or 0),
                    capex=float(info.get("capitalExpenditures") or 0),
                    shares_outstanding=float(info.get("sharesOutstanding") or 0),
                    market_cap=float(info.get("marketCap") or 0),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return select_latest_available(records, as_of)
