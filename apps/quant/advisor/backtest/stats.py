from __future__ import annotations

import numpy as np
import pandas as pd


def book_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(returns).dropna()
    std = r.std(ddof=0)
    if len(r) == 0 or std == 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, periods_per_year: int = 252,
                       target: float = 0.0) -> float:
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return 0.0
    d = np.minimum(r - target, 0.0)
    return float(np.sqrt((d ** 2).mean()) * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252,
            target: float = 0.0) -> float:
    """Annualized Sortino ratio.

    Returns the 0.0 sentinel BOTH for an empty series and for a series with no
    downside (dd == 0, e.g. all-winning): ambiguous by design — returning inf
    would break strict-JSON report output. Callers must not read 0.0 as "dead".
    """
    r = pd.Series(returns).dropna()
    dd = np.sqrt((np.minimum(r - target, 0.0) ** 2).mean()) if len(r) else 0.0
    if len(r) == 0 or dd == 0:
        return 0.0
    return float((r - target).mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return 0.0
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index sample of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])
    return idx[:n]


def block_bootstrap_lcb(returns: pd.Series, block: int, draws: int,
                        seed: int, level: float = 0.95) -> float:
    r = pd.Series(returns).dropna().to_numpy()
    if len(r) < block:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        sample = r[_block_indices(len(r), block, rng)]
        samples[i] = book_sharpe(pd.Series(sample))
    return float(np.quantile(samples, 1.0 - level))


def block_bootstrap_diff_lcb(a: pd.Series, b: pd.Series, block: int, draws: int,
                             seed: int, level: float = 0.95) -> float:
    """One-sided lower CI of book_sharpe(a) - book_sharpe(b), paired by index."""
    df = pd.concat([pd.Series(a).reset_index(drop=True),
                    pd.Series(b).reset_index(drop=True)], axis=1).dropna()
    av, bv = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    n = len(av)
    if n < block:
        return 0.0
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws)
    for i in range(draws):
        idx = _block_indices(n, block, rng)
        diffs[i] = book_sharpe(pd.Series(av[idx])) - book_sharpe(pd.Series(bv[idx]))
    return float(np.quantile(diffs, 1.0 - level))
