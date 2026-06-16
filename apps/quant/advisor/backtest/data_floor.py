from __future__ import annotations

import pandas as pd

from advisor.backtest.dev_gate import dev_gate
from advisor.backtest.pipeline import run_dev_sweep, run_holdout
from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import block_bootstrap_diff_lcb, book_sharpe
from advisor.backtest.universe import classify_universe
from advisor.backtest.validation import validation_report
from advisor.backtest.validation_prereg import DEFAULT_VALIDATION


def floor_metrics(panel: pd.DataFrame, cfg: PreRegConfig, prereg_hash: str | None = None,
                  families: tuple | None = None, holdout_frac: float = 0.2) -> dict:
    """Run the v2 floor: dev gate first, then one blinded holdout evaluation."""
    families = families or cfg.families
    assets = panel.drop(columns=["SPY"]).iloc[cfg.warmup:].reset_index(drop=True)
    sweep = run_dev_sweep(panel, families, cfg, holdout_frac=holdout_frac)

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
        h = run_holdout(panel, families, cfg, sweep.chosen_weights, holdout_frac=holdout_frac)
        delta_lcb = block_bootstrap_diff_lcb(h.ensemble, h.best_family, cfg.bootstrap_block,
                                             cfg.bootstrap_draws, cfg.bootstrap_seed,
                                             level=cfg.final_lcb)
        spy_lcb = block_bootstrap_diff_lcb(h.ensemble, h.spy, cfg.bootstrap_block,
                                           cfg.bootstrap_draws, cfg.bootstrap_seed,
                                           level=cfg.final_lcb)
        beats_parts = delta_lcb > 0
        beats_spy = spy_lcb > cfg.margin
        holdout = {
            "delta_lcb": delta_lcb,
            "spy_lcb": spy_lcb,
            "beats_parts": beats_parts,
            "beats_spy": beats_spy,
            "ensemble_sharpe": book_sharpe(h.ensemble),
            "spy_sharpe": book_sharpe(h.spy),
            "best_family_sharpe": book_sharpe(h.best_family),
            "label": "diagnostic" if universe == "micro" else "formal",
        }
        legacy_spy = book_sharpe(h.spy)
        verdict = "PASSED" if (beats_parts and beats_spy) else "INCONCLUSIVE"

    validation = validation_report(
        sweep.ensemble_test_returns,
        {"ensemble": sweep.ensemble_test_returns,
         "best_family": sweep.best_family_test_returns},
        DEFAULT_VALIDATION,
    )

    return {
        "verdict": verdict,
        "universe": universe,
        "dev": {"passed": gate.passed, "reasons": gate.reasons, "fold_deltas": sweep.fold_deltas},
        "weights": sweep.chosen_weights,
        "holdout": holdout,
        "ensemble": book_sharpe(sweep.ensemble_test_returns),
        "spy": legacy_spy,
        "best_family": book_sharpe(sweep.best_family_test_returns),
        "margin": float(cfg.margin),
        "passes": verdict == "PASSED",
        "validation": validation,
    }
