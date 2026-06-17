from __future__ import annotations

import pandas as pd

from advisor.backtest.continuous_signals import apply_transform, fit_percentile_transform
from advisor.backtest.dev_gate import dev_gate
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import block_bootstrap_diff_lcb, book_sharpe
from advisor.backtest.universe import classify_universe
from advisor.backtest.validation import validation_report
from advisor.research.candidate_prereg import CandidatePreReg
from advisor.research.candidate_pipeline import run_dev_sweep_ext, run_holdout_ext
from advisor.research.candidate_signals import VALUE, candidate_raw
from advisor.research.candidate_validation_prereg import (
    CandidateValidationPreReg, DEFAULT_CANDIDATE_VALIDATION,
)


def _value_power_report(panel: pd.DataFrame, cfg: CandidatePreReg, holdout_frac: float,
                        raw_fn, positive_floor: int = 25) -> dict:
    """Amendment F6 — sufficiency of the (thin) value percentile fit per dev fold. Reports
    the per-fold positive-raw TRAIN count (min/median across assets) and the nonzero
    transformed-score coverage on TEST rows. power_limited iff the MEDIAN positive-train
    count drops below positive_floor in any fold -> a DEV_FAILED there may be a power
    artifact, not a signal verdict (Task 8 labels it power-limited, not 'Reading A
    exhausted'). Report-only; never changes the machine verdict."""
    if VALUE not in cfg.families:
        return {"folds": [], "power_limited": False, "positive_floor": positive_floor}
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)
    dev = prices_all.iloc[:int(len(prices_all) * (1 - holdout_frac))]
    folds, power_limited = [], False
    for train_idx, test_idx in purged_splits(len(dev), cfg.folds, cfg.embargo):
        pos_counts, cover_num, cover_den = [], 0, 0
        for c in assets:
            raw = raw_fn(VALUE, dev[c])
            pos_counts.append(int((raw.iloc[train_idx].dropna() > 0).sum()))
            params = fit_percentile_transform(raw.iloc[train_idx], clip=cfg.pct_clip)
            sc = apply_transform(params, raw).iloc[test_idx]
            cover_num += int((sc > 0).sum())
            cover_den += len(sc)
        sp = sorted(pos_counts)
        median_pos = sp[len(sp) // 2] if sp else 0
        folds.append({
            "min_positive_train": sp[0] if sp else 0,
            "median_positive_train": median_pos,
            "nonzero_transformed_coverage": (cover_num / cover_den) if cover_den else 0.0,
            "test_obs": cover_den,
        })
        if median_pos < positive_floor:
            power_limited = True
    return {"folds": folds, "power_limited": power_limited, "positive_floor": positive_floor}


def candidate_metrics(panel: pd.DataFrame, cfg: CandidatePreReg,
                      prereg_hash: str | None = None, holdout_frac: float = 0.2,
                      vcfg: CandidateValidationPreReg = DEFAULT_CANDIDATE_VALIDATION) -> dict:
    """Candidate bench mirror of floor_metrics: dev gate first, one blinded holdout.
    Report-only; never authorizes sizing; frozen floor untouched. Validation uses the
    CANDIDATE's own N/var_sr surface (Amendment F1, NOT the floor's DEFAULT_VALIDATION),
    stays report-only (never folded into `passes`); a F6 power block labels thin fits."""
    families = cfg.families

    def raw_fn(family: str, prices: pd.Series) -> pd.Series:
        return candidate_raw(family, prices, value_skip=cfg.value_skip,
                             value_lookback=cfg.value_lookback)

    assets = panel.drop(columns=["SPY"]).iloc[cfg.warmup:].reset_index(drop=True)
    sweep = run_dev_sweep_ext(panel, families, cfg, raw_fn=raw_fn, holdout_frac=holdout_frac)

    dev_end = int(len(assets) * (1 - holdout_frac))
    n_active = [int(assets.iloc[te].notna().all(axis=0).sum())
                for _, te in purged_splits(dev_end, cfg.folds, cfg.embargo)] or [assets.shape[1]]
    universe = classify_universe(n_active, cfg)
    gate = dev_gate(sweep.fold_deltas, sweep.ensemble_test_returns,
                    sweep.best_family_test_returns, cfg)

    holdout = None
    verdict = "DEV_FAILED"
    legacy_spy = book_sharpe(panel["SPY"].iloc[cfg.warmup:].pct_change().fillna(0.0))
    if universe == "do_not_run":
        verdict = "UNSUPPORTED"
    elif gate.passed and prereg_hash is not None:
        h = run_holdout_ext(panel, families, cfg, sweep.chosen_weights,
                            raw_fn=raw_fn, holdout_frac=holdout_frac)
        delta_lcb = block_bootstrap_diff_lcb(h.ensemble, h.best_family, cfg.bootstrap_block,
                                             cfg.bootstrap_draws, cfg.bootstrap_seed,
                                             level=cfg.final_lcb)
        spy_lcb = block_bootstrap_diff_lcb(h.ensemble, h.spy, cfg.bootstrap_block,
                                           cfg.bootstrap_draws, cfg.bootstrap_seed,
                                           level=cfg.final_lcb)
        beats_parts = delta_lcb > 0
        beats_spy = spy_lcb > cfg.margin
        holdout = {
            "delta_lcb": delta_lcb, "spy_lcb": spy_lcb,
            "beats_parts": beats_parts, "beats_spy": beats_spy,
            "ensemble_sharpe": book_sharpe(h.ensemble), "spy_sharpe": book_sharpe(h.spy),
            "best_family_sharpe": book_sharpe(h.best_family),
            "label": "diagnostic" if universe == "micro" else "formal",
        }
        legacy_spy = book_sharpe(h.spy)
        verdict = "PASSED" if (beats_parts and beats_spy) else "INCONCLUSIVE"

    validation = validation_report(
        sweep.ensemble_test_returns,
        {"ensemble": sweep.ensemble_test_returns,
         "best_family": sweep.best_family_test_returns},
        vcfg,   # Amendment F1: candidate's own N/var_sr (report-only; never sets `passes`)
    )
    power = _value_power_report(panel, cfg, holdout_frac, raw_fn)
    return {
        "verdict": verdict, "universe": universe,
        "dev": {"passed": gate.passed, "reasons": gate.reasons, "fold_deltas": sweep.fold_deltas},
        "weights": sweep.chosen_weights, "holdout": holdout,
        "ensemble": book_sharpe(sweep.ensemble_test_returns), "spy": legacy_spy,
        "best_family": book_sharpe(sweep.best_family_test_returns),
        "margin": float(cfg.margin), "passes": verdict == "PASSED",
        "validation": validation, "power": power,
    }
