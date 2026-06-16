from __future__ import annotations

import pandas as pd


def book_returns(weights: pd.DataFrame, prices: pd.DataFrame,
                 cost_per_turn: float = 0.0005) -> pd.Series:
    """Daily book return. Position held = yesterday's weights (no look-ahead);
    transaction cost charged on the day's one-way weight change."""
    w = weights.reindex(columns=prices.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    gross = (held * asset_ret).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    return gross - turnover * cost_per_turn
