from __future__ import annotations

from typing import Callable

import pandas as pd

from advisor.backtest.continuous_signals import apply_transform, fit_percentile_transform
from advisor.backtest.dev_gate import dev_gate
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import block_bootstrap_diff_lcb, book_sharpe
from advisor.backtest.universe import classify_universe
from advisor.backtest.validation import validation_report
from advisor.research.candidate_prereg import CandidatePreReg, candidate_run_hash
from advisor.research.candidate_pipeline import run_dev_sweep_ext, run_holdout_ext
from advisor.research.candidate_signals import VALUE, candidate_raw
from advisor.research.candidate_validation_prereg import (
    CandidateValidationPreReg, DEFAULT_CANDIDATE_VALIDATION,
)
from advisor.research.candidate_prereg_fundamental import (
    FundamentalCandidatePreReg, DEFAULT_FUNDAMENTAL_CANDIDATE,
    fundamental_candidate_run_hash,
)
from advisor.research.candidate_validation_prereg_fundamental import (
    FundamentalCandidateValidationPreReg, DEFAULT_FUNDAMENTAL_CANDIDATE_VALIDATION,
)
from advisor.research.fundamental_value import FUNDAMENTAL_VALUE, make_fundamental_raw
from advisor.research.candidate_prereg_lazy_prices import (
    LazyPricesCandidatePreReg, DEFAULT_LAZY_PRICES_CANDIDATE,
    lazy_prices_candidate_run_hash,
)
from advisor.research.candidate_validation_prereg_lazy_prices import (
    LazyPricesCandidateValidationPreReg, DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION,
)
from advisor.research.lazy_prices import LAZY_PRICES, make_lazy_prices_raw

RawFn = Callable[[str, pd.Series], pd.Series]


def _raw_power_report(panel: pd.DataFrame, cfg, holdout_frac: float,
                      raw_fn: RawFn, family: str, positive_floor: int) -> dict:
    """Amendment F6 — sufficiency of the (thin) value percentile fit per dev fold. Reports
    the per-fold positive-raw TRAIN count (min/median across assets) and the nonzero
    transformed-score coverage on TEST rows. power_limited iff the MEDIAN positive-train
    count drops below positive_floor in any fold -> a DEV_FAILED there may be a power
    artifact, not a signal verdict (Task 8 labels it power-limited, not 'Reading A
    exhausted'). Report-only; never changes the machine verdict."""
    if family not in cfg.families:
        return {"folds": [], "power_limited": False, "positive_floor": positive_floor}
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)
    dev = prices_all.iloc[:int(len(prices_all) * (1 - holdout_frac))]
    folds, power_limited = [], False
    for train_idx, test_idx in purged_splits(len(dev), cfg.folds, cfg.embargo):
        pos_counts, cover_num, cover_den = [], 0, 0
        for c in assets:
            raw = raw_fn(family, dev[c])
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


def _value_power_report(panel: pd.DataFrame, cfg: CandidatePreReg, holdout_frac: float,
                        raw_fn: RawFn, positive_floor: int = 25) -> dict:
    return _raw_power_report(panel, cfg, holdout_frac, raw_fn, VALUE, positive_floor)


def _fundamental_power_report(panel: pd.DataFrame, cfg: FundamentalCandidatePreReg,
                              holdout_frac: float, raw_fn: RawFn) -> dict:
    return _raw_power_report(panel, cfg, holdout_frac, raw_fn, FUNDAMENTAL_VALUE, 1)


def _verify_run_hash_unlock(cfg, prereg_hash: str | None, fixture_path,
                            run_hash_fn, hash_name: str) -> bool:
    if prereg_hash is None:
        return False
    if fixture_path is None or prereg_hash != run_hash_fn(cfg, fixture_path):
        raise ValueError(
            f"holdout unlock requires prereg_hash == {hash_name}(cfg, fixture_path); "
            "refusing to evaluate the reserved tail with an unverified run-hash"
        )
    return True


def _verify_holdout_unlock(cfg: CandidatePreReg, prereg_hash: str | None,
                           fixture_path) -> bool:
    """F2 (review hardening, STRICTER than frozen data_floor.py:34): the reserved tail
    unlocks ONLY when `prereg_hash` equals `candidate_run_hash(cfg, fixture_path)` (config +
    fixture bytes) — a non-null string is NOT enough, matching the `HOLDOUT_LEDGER.md`
    contract. Returns True iff a verified run-hash was supplied; RAISES on a supplied-but-wrong
    hash so a careless caller cannot silently touch (and burn) the shared reserved tail."""
    return _verify_run_hash_unlock(cfg, prereg_hash, fixture_path, candidate_run_hash,
                                   "candidate_run_hash")


def _candidate_metrics_with_raw_fn(panel: pd.DataFrame, cfg, families: tuple[str, ...],
                                   raw_fn: RawFn, prereg_hash: str | None,
                                   holdout_frac: float, vcfg, fixture_path,
                                   run_hash_fn, run_hash_name: str,
                                   power_fn) -> dict:
    """Candidate bench mirror of floor_metrics: dev gate first, one blinded holdout.
    Report-only; never authorizes sizing; frozen floor untouched. Validation uses the
    CANDIDATE's own N/var_sr surface (Amendment F1, NOT the floor's DEFAULT_VALIDATION),
    stays report-only (never folded into `passes`); a F6 power block labels thin fits.
    The reserved tail is touched ONLY iff the dev gate passes AND `prereg_hash` is a verified
    `candidate_run_hash(cfg, fixture_path)` (review F2); when blinded, even the SPY benchmark is
    computed over the DEV window only so the tail is genuinely never read (review F1)."""
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
    # Review F1 (STRICTER than frozen data_floor.py:31, which uses the full post-warmup
    # series): when the holdout is blinded, compute the SPY benchmark over the DEV window
    # ONLY, so the reserved tail is never read on the no-holdout path. If the holdout is
    # unlocked below, legacy_spy is reassigned to the holdout SPY.
    spy_all = panel["SPY"].iloc[cfg.warmup:].reset_index(drop=True)
    spy_dev_end = int(len(spy_all) * (1 - holdout_frac))
    legacy_spy = book_sharpe(spy_all.iloc[:spy_dev_end].pct_change().fillna(0.0))
    if universe == "do_not_run":
        verdict = "UNSUPPORTED"
    elif gate.passed and _verify_run_hash_unlock(
        cfg, prereg_hash, fixture_path, run_hash_fn, run_hash_name
    ):
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
    power = power_fn(panel, cfg, holdout_frac, raw_fn)
    return {
        "verdict": verdict, "universe": universe,
        "dev": {"passed": gate.passed, "reasons": gate.reasons, "fold_deltas": sweep.fold_deltas},
        "weights": sweep.chosen_weights, "holdout": holdout,
        "ensemble": book_sharpe(sweep.ensemble_test_returns), "spy": legacy_spy,
        "best_family": book_sharpe(sweep.best_family_test_returns),
        "margin": float(cfg.margin), "passes": verdict == "PASSED",
        "validation": validation, "power": power,
    }


def candidate_metrics(panel: pd.DataFrame, cfg: CandidatePreReg,
                      prereg_hash: str | None = None, holdout_frac: float = 0.2,
                      vcfg: CandidateValidationPreReg = DEFAULT_CANDIDATE_VALIDATION,
                      fixture_path=None) -> dict:
    families = cfg.families

    def raw_fn(family: str, prices: pd.Series) -> pd.Series:
        return candidate_raw(family, prices, value_skip=cfg.value_skip,
                             value_lookback=cfg.value_lookback)

    return _candidate_metrics_with_raw_fn(
        panel, cfg, families, raw_fn, prereg_hash, holdout_frac, vcfg, fixture_path,
        candidate_run_hash, "candidate_run_hash", _value_power_report,
    )


def fundamental_candidate_metrics(
    panel: pd.DataFrame,
    panel_funda: pd.DataFrame,
    cfg: FundamentalCandidatePreReg = DEFAULT_FUNDAMENTAL_CANDIDATE,
    prereg_hash: str | None = None,
    holdout_frac: float = 0.2,
    vcfg: FundamentalCandidateValidationPreReg = DEFAULT_FUNDAMENTAL_CANDIDATE_VALIDATION,
    fixture_path=None,
) -> dict:
    """Reading-B candidate floor mirror using the precomputed PIT fundamentals panel.

    `panel_funda` must be built with `build_fundamental_panel(..., warmup=cfg.warmup)`
    so it shares the positional basis used by candidate_pipeline. The holdout remains
    blinded unless a verified `fundamental_candidate_run_hash` is supplied.
    """
    expected_rows = max(0, len(panel) - cfg.warmup)
    if len(panel_funda) != expected_rows:
        raise ValueError(
            f"panel_funda rows ({len(panel_funda)}) must equal len(panel)-cfg.warmup "
            f"({expected_rows}); build it with build_fundamental_panel(..., warmup=cfg.warmup)"
        )
    raw_fn = make_fundamental_raw(panel_funda)
    families = cfg.families
    return _candidate_metrics_with_raw_fn(
        panel, cfg, families, raw_fn, prereg_hash, holdout_frac, vcfg, fixture_path,
        fundamental_candidate_run_hash, "fundamental_candidate_run_hash",
        _fundamental_power_report,
    )


def _lazy_prices_power_report(panel: pd.DataFrame, cfg: LazyPricesCandidatePreReg,
                              holdout_frac: float, raw_fn: RawFn) -> dict:
    # lazy_prices is a low-frequency step function: positive_floor=1 (like fundamentals)
    return _raw_power_report(panel, cfg, holdout_frac, raw_fn, LAZY_PRICES, 1)


def lazy_prices_candidate_metrics(
    panel: pd.DataFrame,
    panel_lp: pd.DataFrame,
    cfg: LazyPricesCandidatePreReg = DEFAULT_LAZY_PRICES_CANDIDATE,
    prereg_hash: str | None = None,
    holdout_frac: float = 0.2,
    vcfg: LazyPricesCandidateValidationPreReg = DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION,
    fixture_path=None,
) -> dict:
    """Reading-C candidate floor mirror using the precomputed PIT similarity panel.
    `panel_lp` must be built with `build_lazy_prices_panel(..., warmup=cfg.warmup)` so it
    shares candidate_pipeline's positional basis. Holdout stays blinded unless a verified
    `lazy_prices_candidate_run_hash` is supplied. Reuses the generic injection helper."""
    expected_rows = max(0, len(panel) - cfg.warmup)
    if len(panel_lp) != expected_rows:
        raise ValueError(
            f"panel_lp rows ({len(panel_lp)}) must equal len(panel)-cfg.warmup "
            f"({expected_rows}); build it with build_lazy_prices_panel(..., warmup=cfg.warmup)"
        )
    raw_fn = make_lazy_prices_raw(panel_lp)
    return _candidate_metrics_with_raw_fn(
        panel, cfg, cfg.families, raw_fn, prereg_hash, holdout_frac, vcfg, fixture_path,
        lazy_prices_candidate_run_hash, "lazy_prices_candidate_run_hash",
        _lazy_prices_power_report,
    )
