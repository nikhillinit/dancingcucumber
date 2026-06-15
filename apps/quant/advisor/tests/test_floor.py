import numpy as np
import pandas as pd

from advisor.backtest.floor import beats_floor, purged_walk_forward_sharpe


def _series(slope):
    return pd.Series(np.linspace(100, 100 + slope, 300))


def test_purged_walk_forward_returns_finite_sharpe():
    prices = _series(50)
    signal = pd.Series(1.0, index=prices.index)
    s = purged_walk_forward_sharpe(prices, signal, folds=3, embargo=5)
    assert np.isfinite(s)


def test_beats_floor_requires_both_conditions():
    # ensemble 1.2 beats spy 0.8 (margin 0.3) AND best family 1.0
    assert beats_floor(ensemble=1.2, spy=0.8, best_family=1.0, margin=0.3) is True
    # fails benchmark margin
    assert beats_floor(ensemble=1.0, spy=0.8, best_family=0.5, margin=0.3) is False
    # beats benchmark but not its own best family
    assert beats_floor(ensemble=1.2, spy=0.8, best_family=1.3, margin=0.3) is False
    # ties its own best family, which is not a beat
    assert beats_floor(ensemble=1.2, spy=0.8, best_family=1.2, margin=0.3) is False
