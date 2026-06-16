import numpy as np
import pandas as pd

from advisor.backtest.stats import book_sharpe
from advisor.backtest.validation import per_obs_sharpe, sharpe_moments


def _r(mean, sd=0.01, n=1250, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n))


def test_per_obs_sharpe_is_annualized_over_sqrt_252():
    r = _r(0.0005)
    assert abs(per_obs_sharpe(r) - book_sharpe(r) / np.sqrt(252)) < 1e-12


def test_per_obs_sharpe_zero_on_zero_vol():
    assert per_obs_sharpe(pd.Series([0.001, 0.001, 0.001])) == 0.0


def test_sharpe_moments_returns_T_skew_kurt():
    r = _r(0.0005, seed=3)
    T, skew, kurt = sharpe_moments(r)
    assert T == 1250
    assert abs(kurt - 3.0) < 0.5        # ~normal -> Pearson kurtosis near 3
    assert abs(skew) < 0.5
