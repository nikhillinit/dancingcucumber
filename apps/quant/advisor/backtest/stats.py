from __future__ import annotations

import numpy as np
import pandas as pd


def book_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(returns).dropna()
    std = r.std(ddof=0)
    if len(r) == 0 or std == 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


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
