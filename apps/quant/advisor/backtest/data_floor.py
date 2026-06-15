from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.backtest.floor import beats_floor, purged_walk_forward_sharpe


def _momentum_signal(prices: pd.Series, lookback: int = 126) -> pd.Series:
    return np.sign(prices / prices.shift(lookback) - 1.0).fillna(0.0)


def _trend_signal(prices: pd.Series, short: int = 50, long: int = 200) -> pd.Series:
    return np.sign(prices.rolling(short).mean() - prices.rolling(long).mean()).fillna(0.0)


def _long_flat(signal: pd.Series) -> pd.Series:
    """Mirror the deployed allocator: it sizes a positive (long) position or holds.

    There are no shorting rails (allocator caps with a single positive dollar limit),
    so a bearish signal means FLAT (exit), not short. The floor must backtest what
    actually ships, not a more-penalized long/short variant.
    """
    return (signal > 0).astype(float)


def floor_metrics(
    panel: pd.DataFrame,
    benchmark: str = "SPY",
    margin: float = 0.0,
    folds: int = 5,
    embargo: int = 5,
) -> dict:
    """Backtest the deployed long-flat price-only ensemble vs SPY (spec sections 6-7).

    Returns OOS purged-walk-forward Sharpes for the ensemble, SPY buy-and-hold, and the
    best single price family, plus whether the floor is cleared. All three strategy legs
    use identical long-flat position construction so "beat the parts" is a fair comparison.
    """
    tickers = [c for c in panel.columns if c != benchmark]
    ens, mom, tr = [], [], []
    for t in tickers:
        p = panel[t].dropna()
        ensemble_sig = np.sign(_momentum_signal(p) + _trend_signal(p))
        ens.append(purged_walk_forward_sharpe(p, _long_flat(ensemble_sig), folds, embargo))
        mom.append(purged_walk_forward_sharpe(p, _long_flat(_momentum_signal(p)), folds, embargo))
        tr.append(purged_walk_forward_sharpe(p, _long_flat(_trend_signal(p)), folds, embargo))

    ensemble = float(np.mean(ens)) if ens else 0.0
    best_family = max(float(np.mean(mom)) if mom else 0.0, float(np.mean(tr)) if tr else 0.0)

    spy = panel[benchmark].dropna()
    spy_sharpe = purged_walk_forward_sharpe(spy, pd.Series(1.0, index=spy.index), folds, embargo)

    return {
        "ensemble": ensemble,
        "spy": spy_sharpe,
        "best_family": best_family,
        "margin": float(margin),
        "passes": bool(beats_floor(ensemble, spy_sharpe, best_family, margin)),
    }
