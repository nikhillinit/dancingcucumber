from __future__ import annotations

from math import ceil, e
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


def effective_n_pca(family_returns: dict[str, pd.Series]) -> float:
    """PCA participation ratio on the family return-series correlation matrix.
    Neff = (sum eigenvalues)^2 / sum(eigenvalues^2). Correlated families -> small Neff."""
    series = [pd.Series(s).reset_index(drop=True) for s in family_returns.values()]
    if len(series) <= 1:
        return float(len(series))
    mat = pd.concat(series, axis=1).dropna()
    if mat.shape[0] < 2:
        return float(mat.shape[1])
    corr = np.corrcoef(mat.to_numpy(), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    lam = np.linalg.eigvalsh(corr)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 1.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def n_for_dsr(family_returns: dict[str, pd.Series], declared_trials_N: int) -> int:
    """Trials used for DSR: max(declared integer, ceil(effective-N)), floored at 1.
    Effective-N may never drop the penalty below the pre-registered declared count."""
    eff = ceil(effective_n_pca(family_returns)) if family_returns else 0
    return max(1, declared_trials_N, eff)


def minbtl_exceeded(n_trials: int, max_trials: int) -> bool:
    """True when the search budget is over-spent (gate is meaningless past this)."""
    return n_trials > max_trials


def tstat_meets_hurdle(tstat: float | None, hurdle: float) -> bool:
    """Harvey-Liu-Zhu selection hurdle. None (no t-stat claim) does not meet it."""
    return tstat is not None and tstat >= hurdle
