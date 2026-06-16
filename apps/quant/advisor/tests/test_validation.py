from statistics import NormalDist

import numpy as np
import pandas as pd

from advisor.backtest.stats import book_sharpe
from advisor.backtest.validation import per_obs_sharpe, psr, sharpe_moments


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


def test_psr_matches_closed_form():
    # SR_hat=0.10 (per-obs), SR*=0.0, T=1250, skew=0, kurt=3 (normal)
    sr, T = 0.10, 1250
    denom = (1 - 0.0 * sr + ((3.0 - 1) / 4) * sr ** 2) ** 0.5
    expected = NormalDist().cdf((sr - 0.0) * (T - 1) ** 0.5 / denom)
    assert abs(psr(sr_hat=sr, sr_benchmark=0.0, T=T, skew=0.0, kurt=3.0) - expected) < 1e-12


def test_psr_rises_with_more_observations():
    base = dict(sr_hat=0.08, sr_benchmark=0.0, skew=0.0, kurt=3.0)
    assert psr(T=300, **base) < psr(T=3000, **base)


def test_psr_negative_skew_lowers_confidence():
    base = dict(sr_hat=0.08, sr_benchmark=0.0, T=1250, kurt=6.0)
    assert psr(skew=-2.0, **base) < psr(skew=0.0, **base)
