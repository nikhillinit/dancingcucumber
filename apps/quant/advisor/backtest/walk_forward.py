from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISCLOSURES = [
    "Fundamentals are restated, not as-reported.",
    "Point-in-time lag approximated (~90-day proxy); results indicative, not as-reported.",
    "yfinance is survivorship-biased (delisted names absent); long-side results upward-biased.",
    "Any LLM/news-derived feature may carry look-ahead from pretraining that cannot be purged.",
]


def disclosure_header() -> str:
    return "DISCLOSURES:\n" + "\n".join(f"  - {d}" for d in DISCLOSURES)


@dataclass(frozen=True)
class BacktestResult:
    sharpe: float
    total_return: float
    n_periods: int
    disclosures: list[str]


def _sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    std = returns.std(ddof=0)
    if std == 0 or len(returns) == 0:
        return 0.0
    excess = returns - rf / periods_per_year
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def walk_forward(prices: pd.Series, signal: pd.Series, cost_per_turn: float = 0.0005) -> BacktestResult:
    """Position is YESTERDAY's signal (no look-ahead). Cost charged on position change."""
    position = signal.shift(1).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * asset_ret - turnover * cost_per_turn
    total = float((1 + strat_ret).prod() - 1)
    return BacktestResult(sharpe=_sharpe(strat_ret), total_return=total,
                          n_periods=int(len(strat_ret)), disclosures=list(DISCLOSURES))


def passes_floor(strategy_sharpe: float, benchmark_sharpe: float, margin: float) -> bool:
    """Spec section 7 floor: beat the benchmark by a pre-registered margin."""
    return (strategy_sharpe - benchmark_sharpe) >= margin
