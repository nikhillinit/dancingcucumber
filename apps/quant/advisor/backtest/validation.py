from __future__ import annotations

import numpy as np
import pandas as pd


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
