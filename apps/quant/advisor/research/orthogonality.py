from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.backtest.continuous_signals import apply_transform, fit_percentile_transform
from advisor.backtest.splits import purged_splits
from advisor.research.candidate_signals import VALUE, candidate_raw


def _dev_frame(panel: pd.DataFrame, warmup: int, holdout_frac: float):
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    return assets, prices_all.iloc[:dev_end]


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if a.size == 0 or b.size == 0 or a.std() == 0 or b.std() == 0:
        return 0.0
    if method == "spearman":
        a = pd.Series(a).rank().to_numpy()
        b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def dev_fold_raw_corr(panel: pd.DataFrame, *, warmup: int, holdout_frac: float,
                      value_skip: int, value_lookback: int,
                      neighbors: tuple[str, ...],
                      folds: int = 5, embargo: int = 5,
                      method: str = "pearson") -> dict[str, float]:
    """Correlation of `value` raw vs each neighbor raw, pooled over the dev test
    folds and all assets (holdout strictly excluded). NaN rows dropped pairwise.
    method='pearson' (default) or 'spearman' (rank corr -- a scale-robust cross-check,
    Amendment F4). This is a pre-registered DIAGNOSTIC, not the gate: the kill-gate
    reads the post-transform surface (dev_fold_post_transform_corr)."""
    assets, dev = _dev_frame(panel, warmup, holdout_frac)
    test_rows = sorted({i for _, te in purged_splits(len(dev), folds, embargo) for i in te})
    out: dict[str, float] = {}
    for nb in neighbors:
        vv, nn = [], []
        for c in assets:
            v = candidate_raw(VALUE, dev[c], value_skip=value_skip,
                              value_lookback=value_lookback).iloc[test_rows]
            n = candidate_raw(nb, dev[c]).iloc[test_rows]
            df = pd.concat([v, n], axis=1).dropna()
            vv.append(df.iloc[:, 0].to_numpy())
            nn.append(df.iloc[:, 1].to_numpy())
        out[nb] = _corr(np.concatenate(vv), np.concatenate(nn), method)
    return out


def dev_fold_post_transform_corr(panel: pd.DataFrame, *, warmup: int, holdout_frac: float,
                                 value_skip: int, value_lookback: int,
                                 neighbors: tuple[str, ...],
                                 folds: int = 5, embargo: int = 5,
                                 clip: tuple[float, float] = (0.05, 0.95)) -> dict[str, float]:
    """THE GATE SURFACE (Amendment F4): correlation of the long-flat CONVICTION scores
    that actually enter the blend. Per dev fold, fit fit_percentile_transform on the
    fold's TRAIN rows (per asset, exactly as pipeline._family_scores), apply_transform,
    then correlate `value` vs each neighbor on the fold TEST rows. Pooled across folds
    and assets (mirrors dev_fold_raw_corr; more stable than per-fold averaging on the
    thin fold-1 fit). Holdout strictly excluded. Because the transform clamps raw<=0 to
    flat, a signal and its negation fire on DISJOINT days here -> their post-transform
    corr is far weaker than the raw -1; that disjointness is the blend-relevant
    diversification the section-7.2 gate rewards."""
    assets, dev = _dev_frame(panel, warmup, holdout_frac)
    splits = purged_splits(len(dev), folds, embargo)
    out: dict[str, float] = {}
    for nb in neighbors:
        vv, nn = [], []
        for c in assets:
            v_raw = candidate_raw(VALUE, dev[c], value_skip=value_skip,
                                  value_lookback=value_lookback)
            n_raw = candidate_raw(nb, dev[c])
            for train_idx, test_idx in splits:
                v_par = fit_percentile_transform(v_raw.iloc[train_idx], clip=clip)
                n_par = fit_percentile_transform(n_raw.iloc[train_idx], clip=clip)
                v_sc = apply_transform(v_par, v_raw).iloc[test_idx]
                n_sc = apply_transform(n_par, n_raw).iloc[test_idx]
                df = pd.concat([v_sc, n_sc], axis=1).dropna()
                vv.append(df.iloc[:, 0].to_numpy())
                nn.append(df.iloc[:, 1].to_numpy())
        out[nb] = _corr(np.concatenate(vv), np.concatenate(nn), "pearson")
    return out
