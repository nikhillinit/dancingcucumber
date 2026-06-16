# apps/quant/advisor/backtest/pipeline.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from advisor.backtest.blend import select_weights
from advisor.backtest.book import book_returns
from advisor.backtest.continuous_signals import apply_transform, fit_percentile_transform, raw_metric
from advisor.backtest.portfolio import build_long_flat_book
from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import book_sharpe


@dataclass(frozen=True)
class SweepResult:
    fold_deltas: list[float]
    ensemble_test_returns: pd.Series
    best_family_test_returns: pd.Series
    chosen_weights: dict


@dataclass(frozen=True)
class HoldoutReturns:
    ensemble: pd.Series
    best_family: pd.Series
    spy: pd.Series


def _family_scores(family: str, prices: pd.DataFrame, train_idx, all_idx, clip) -> pd.DataFrame:
    """Fit the transform on TRAIN rows per column, apply to all rows. Long-flat."""
    cols = {}
    for c in prices.columns:
        raw = raw_metric(family, prices[c])
        params = fit_percentile_transform(raw.iloc[train_idx], clip=clip)
        cols[c] = apply_transform(params, raw)
    return pd.DataFrame(cols).iloc[all_idx].reset_index(drop=True)


def run_dev_sweep(panel: pd.DataFrame, families: tuple, cfg: PreRegConfig,
                  holdout_frac: float = 0.2) -> SweepResult:
    """Dev folds only: hold out the final holdout_frac, purge-walk-forward the rest."""
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)   # warmup excluded
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    dev = prices_all.iloc[:dev_end]

    caps = (cfg.max_asset_weight, cfg.gross_cap, cfg.turnover_cap)
    deltas, ens_parts, best_parts = [], [], []
    chosen = {f: 1.0 / len(families) for f in families}
    for train_idx, test_idx in purged_splits(len(dev), cfg.folds, cfg.embargo):
        all_idx = list(range(min(train_idx), max(test_idx) + 1))
        scores = {f: _family_scores(f, dev, train_idx,
                                    [i for i in all_idx], cfg.pct_clip) for f in families}
        # map global train/test indices into the all_idx-local frame
        local = {g: i for i, g in enumerate(all_idx)}
        tr = [local[i] for i in train_idx]
        te = [local[i] for i in test_idx]
        train_scores = {f: scores[f].iloc[tr].reset_index(drop=True) for f in families}
        chosen = select_weights(train_scores, dev.iloc[train_idx].reset_index(drop=True),
                                families, cfg.weight_grid, cfg.train_lift_threshold,
                                cfg.cost_per_turn, caps)
        test_prices = dev.iloc[test_idx].reset_index(drop=True)
        blended = sum(chosen[f] * scores[f].iloc[te].reset_index(drop=True) for f in families)
        blended = pd.DataFrame(blended, columns=dev.columns)
        ens_w = build_long_flat_book(blended, *caps, cfg.cost_per_turn)
        ens_r = book_returns(ens_w, test_prices, cfg.cost_per_turn)

        fam_sharpes = {}
        fam_rets = {}
        for f in families:
            fs = pd.DataFrame(scores[f].iloc[te].reset_index(drop=True), columns=dev.columns)
            fw = build_long_flat_book(fs, *caps, cfg.cost_per_turn)
            fr = book_returns(fw, test_prices, cfg.cost_per_turn)
            fam_sharpes[f] = book_sharpe(fr)
            fam_rets[f] = fr
        best_f = max(fam_sharpes, key=fam_sharpes.get)
        deltas.append(book_sharpe(ens_r) - fam_sharpes[best_f])
        ens_parts.append(ens_r)
        best_parts.append(fam_rets[best_f])

    # Freeze the blend weights on the FULL dev portion (leakage-safe: dev excludes
    # the holdout), NOT the arbitrary last fold (debate finding #3). run_holdout
    # fits its transform on the full dev portion too -- keep the two symmetric so
    # the held-out tail is scored with a single pre-registered final model.
    full_idx = list(range(len(dev)))
    full_scores = {f: _family_scores(f, dev, full_idx, full_idx, cfg.pct_clip)
                   for f in families}
    frozen = select_weights(full_scores, dev.reset_index(drop=True), families,
                            cfg.weight_grid, cfg.train_lift_threshold,
                            cfg.cost_per_turn, caps) if full_idx else chosen

    return SweepResult(
        fold_deltas=deltas,
        ensemble_test_returns=pd.concat(ens_parts, ignore_index=True) if ens_parts else pd.Series(dtype=float),
        best_family_test_returns=pd.concat(best_parts, ignore_index=True) if best_parts else pd.Series(dtype=float),
        chosen_weights=frozen,
    )


def run_holdout(panel: pd.DataFrame, families: tuple, cfg: PreRegConfig,
                frozen_weights: dict, holdout_frac: float = 0.2) -> HoldoutReturns:
    """Evaluate the dev-FROZEN construction on the held-out tail ONCE. Transform is
    fit on the full dev portion; frozen_weights come from run_dev_sweep. Returns
    ensemble, best-standalone-family, and SPY book returns over the holdout window."""
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)
    spy_all = panel["SPY"].iloc[cfg.warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    caps = (cfg.max_asset_weight, cfg.gross_cap, cfg.turnover_cap)

    # raw over the full series (continuity for rolling/shift); transform fit on dev only
    scores = {}
    for f in families:
        cols = {}
        for c in assets:
            raw = raw_metric(f, prices_all[c])
            params = fit_percentile_transform(raw.iloc[:dev_end], clip=cfg.pct_clip)
            cols[c] = apply_transform(params, raw)
        scores[f] = pd.DataFrame(cols)

    hold = slice(dev_end, len(prices_all))
    hold_prices = prices_all.iloc[hold].reset_index(drop=True)
    blended = sum(frozen_weights[f] * scores[f].iloc[hold].reset_index(drop=True) for f in families)
    blended = pd.DataFrame(blended, columns=assets)
    ens_r = book_returns(build_long_flat_book(blended, *caps, cfg.cost_per_turn),
                         hold_prices, cfg.cost_per_turn)

    fam_rets = {}
    for f in families:
        fs = pd.DataFrame(scores[f].iloc[hold].reset_index(drop=True), columns=assets)
        fam_rets[f] = book_returns(build_long_flat_book(fs, *caps, cfg.cost_per_turn),
                                   hold_prices, cfg.cost_per_turn)
    best_f = max(fam_rets, key=lambda f: book_sharpe(fam_rets[f]))
    spy_r = spy_all.iloc[hold].reset_index(drop=True).pct_change().fillna(0.0)
    return HoldoutReturns(ensemble=ens_r, best_family=fam_rets[best_f], spy=spy_r)
