from __future__ import annotations

from math import e
from statistics import NormalDist

import numpy as np
import pandas as pd

_Z = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def per_obs_sharpe(returns: pd.Series) -> float:
    """Per-observation Sharpe (mean/std). NOT annualized -- DSR/PSR require this.
    Distinct from stats.book_sharpe, which multiplies by sqrt(252)."""
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=0)
    if len(r) == 0 or sd == 0:
        return 0.0
    return float(r.mean() / sd)


def sharpe_moments(returns: pd.Series) -> tuple[int, float, float]:
    """Return (T, skewness, Pearson-kurtosis[normal=3]) of the returns series."""
    r = pd.Series(returns).dropna().to_numpy()
    T = len(r)
    if T < 2:
        return T, 0.0, 3.0
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return T, 0.0, 3.0
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))      # Pearson (normal -> 3.0)
    return T, skew, kurt


def psr(sr_hat: float, sr_benchmark: float, T: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark) given the estimate's
    moments. sr_hat is PER-OBSERVATION (see per_obs_sharpe). Bailey & LdP 2014."""
    if T < 2:
        return 0.0
    var = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    if var <= 0:
        return 0.0
    return float(_Z.cdf((sr_hat - sr_benchmark) * (T - 1) ** 0.5 / var ** 0.5))


def _sr0(N: int, var_sr: float) -> float:
    """Deflated benchmark = expected max of N trial Sharpes (LdP). var_sr is the
    CROSS-TRIAL dispersion V[{SR_n}], NOT a single strategy's sampling variance.
    N<=1 means no multiple testing -> no deflation (and inv_cdf(0) would raise)."""
    if N <= 1 or var_sr <= 0:
        return 0.0
    g = _EULER_GAMMA
    term = (1 - g) * _Z.inv_cdf(1 - 1.0 / N) + g * _Z.inv_cdf(1 - 1.0 / (N * e))
    return float(var_sr ** 0.5 * term)


def var_sr_trials(trial_sharpes) -> float:
    """Cross-trial dispersion V[{SR_n}] = sample variance of the per-trial Sharpe
    estimates. This is the correct variance for _sr0 (see Task 4 var_sr note)."""
    s = [float(x) for x in trial_sharpes]
    if len(s) < 2:
        return 0.0
    return float(np.var(s, ddof=1))


def deflated_sharpe(returns: pd.Series, n_trials: int, var_sr: float,
                    sr_benchmark: float = 0.0) -> float:
    """DSR = PSR evaluated at the multiple-testing-deflated benchmark SR0.
    var_sr is the CROSS-TRIAL Sharpe dispersion (caller supplies it, floored)."""
    sr = per_obs_sharpe(returns)
    T, skew, kurt = sharpe_moments(returns)
    sr0 = max(sr_benchmark, _sr0(n_trials, var_sr))
    return psr(sr_hat=sr, sr_benchmark=sr0, T=T, skew=skew, kurt=kurt)
