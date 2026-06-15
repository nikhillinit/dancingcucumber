from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.backtest.walk_forward import walk_forward


def purged_walk_forward_sharpe(
    prices: pd.Series,
    signal: pd.Series,
    folds: int = 5,
    embargo: int = 5,
) -> float:
    """Average out-of-sample Sharpe across purged walk-forward folds."""
    if folds < 2 or embargo < 0:
        return 0.0

    n = min(len(prices), len(signal))
    if n < folds * 10:
        return 0.0

    fold_size = n // folds
    sharpes = []
    for fold in range(1, folds):
        test_start = fold * fold_size + embargo
        test_end = (fold + 1) * fold_size if fold < folds - 1 else n
        if test_start >= test_end:
            continue

        fold_prices = prices.iloc[test_start:test_end].reset_index(drop=True)
        fold_signal = signal.iloc[test_start:test_end].reset_index(drop=True)
        sharpes.append(walk_forward(fold_prices, fold_signal).sharpe)

    return float(np.mean(sharpes)) if sharpes else 0.0


def beats_floor(ensemble: float, spy: float, best_family: float, margin: float) -> bool:
    """Require ensemble Sharpe to beat SPY by margin and beat its best family."""
    values = (ensemble, spy, best_family, margin)
    if not all(np.isfinite(value) for value in values):
        return False
    return (ensemble - spy) >= margin and ensemble > best_family
