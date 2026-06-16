import numpy as np
import pandas as pd

from advisor.backtest.dev_gate import dev_gate
from advisor.backtest.prereg import DEFAULT_CONFIG


def _series(mean, n=1000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.01, n))


def test_clear_winner_passes():
    deltas = [0.3, 0.25, 0.4, 0.2]
    ens, best = _series(0.0015), _series(0.0002, seed=1)
    res = dev_gate(deltas, ens, best, DEFAULT_CONFIG)
    assert res.passed is True and not res.reasons


def test_majority_negative_folds_fail():
    deltas = [0.3, -0.1, -0.2, -0.05]                  # < 70% positive
    ens, best = _series(0.0008), _series(0.0007, seed=1)
    res = dev_gate(deltas, ens, best, DEFAULT_CONFIG)
    assert res.passed is False
    assert any("70%" in r or "positive folds" in r for r in res.reasons)


def test_single_fold_concentration_fails():
    deltas = [0.5, 0.01, 0.01, 0.01]                   # one fold dominates excess
    ens, best = _series(0.0009), _series(0.0006, seed=2)
    res = dev_gate(deltas, ens, best, DEFAULT_CONFIG, fold_excess=[0.9, 0.03, 0.03, 0.04])
    assert res.passed is False
    assert any("concentration" in r for r in res.reasons)
