import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from advisor.backtest.continuous_signals import (
    apply_transform, fit_percentile_transform, raw_metric,
)
from advisor.backtest.pipeline import run_dev_sweep, run_holdout          # frozen (Amendment F3)
from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.stats import book_sharpe
from advisor.research.candidate_pipeline import run_dev_sweep_ext, run_holdout_ext


def load_floor_panel():
    # Amendment F7 — concrete loader; the floor's fixture is script-local (not importable).
    return pd.read_csv(Path("apps/quant/advisor/tests/fixtures/floor_prices.csv"),
                       index_col=0, parse_dates=True)


def _synth_panel(n=900, k=12, seed=7):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)


def test_ext_pipeline_equals_frozen_pipeline_elementwise():
    # Amendment F3 — the REAL trust anchor. raw_fn=raw_metric -> the mirror must be the
    # frozen pipeline exactly on shared families; any drift in EITHER goes red. Dev-only
    # (run_dev_sweep never touches the reserved tail) -> holdout stays blinded.
    panel = load_floor_panel()
    cfg = PreRegConfig()
    ext = run_dev_sweep_ext(panel, ("momentum", "trend"), cfg, raw_fn=raw_metric, holdout_frac=0.2)
    ref = run_dev_sweep(panel, ("momentum", "trend"), cfg, holdout_frac=0.2)
    assert ext.fold_deltas == ref.fold_deltas
    pd.testing.assert_series_equal(ext.ensemble_test_returns, ref.ensemble_test_returns)
    pd.testing.assert_series_equal(ext.best_family_test_returns, ref.best_family_test_returns)
    assert ext.chosen_weights == ref.chosen_weights


def test_holdout_ext_equals_frozen_holdout_on_synthetic():
    # Amendment F2/F3: prove the HOLDOUT mirror faithful too, but on SYNTHETIC data ONLY —
    # an equality check that calls run_holdout on floor_prices.csv would touch the real tail.
    panel = _synth_panel()
    cfg = PreRegConfig()
    fams = ("momentum", "trend")
    ref_sweep = run_dev_sweep(panel, fams, cfg, holdout_frac=0.2)
    ext_sweep = run_dev_sweep_ext(panel, fams, cfg, raw_fn=raw_metric, holdout_frac=0.2)
    ref_h = run_holdout(panel, fams, cfg, ref_sweep.chosen_weights, holdout_frac=0.2)
    ext_h = run_holdout_ext(panel, fams, cfg, ext_sweep.chosen_weights,
                            raw_fn=raw_metric, holdout_frac=0.2)
    pd.testing.assert_series_equal(ext_h.ensemble, ref_h.ensemble)
    pd.testing.assert_series_equal(ext_h.best_family, ref_h.best_family)
    pd.testing.assert_series_equal(ext_h.spy, ref_h.spy)


def test_bench_reproduces_floor_construction_C():
    # Documentation check of the published numbers (no longer the trust anchor).
    panel = load_floor_panel()                  # the real 2015-2023 fixture
    cfg = PreRegConfig()                         # floor's own config (warmup=200)
    res = run_dev_sweep_ext(panel, ("momentum", "trend"), cfg,
                            raw_fn=raw_metric, holdout_frac=0.2)
    ens = book_sharpe(res.ensemble_test_returns)
    best = book_sharpe(res.best_family_test_returns)
    assert ens == pytest.approx(0.732, abs=0.01)     # FLOOR_RESULT.md C ensemble
    assert best == pytest.approx(0.828, abs=0.01)    # FLOOR_RESULT.md C best family
    assert all(d < 0 for d in res.fold_deltas)       # every fold delta negative


def test_value_leg_is_live_in_every_dev_fold():
    # Feasibility guard (rail #5): the pre-registered value_lookback must leave a
    # non-degenerate positive-value training set in EVERY dev fold on the real fixture,
    # else the dev gate is rigged to fail for non-signal reasons.
    from advisor.backtest.splits import purged_splits
    from advisor.research.candidate_prereg import DEFAULT_CANDIDATE as C
    from advisor.research.candidate_signals import VALUE, candidate_raw
    panel = load_floor_panel()
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[C.warmup:].reset_index(drop=True)
    dev = prices_all.iloc[:int(len(prices_all) * 0.8)]
    val = candidate_raw(VALUE, dev[assets[0]], value_skip=C.value_skip,
                        value_lookback=C.value_lookback)
    for train_idx, _ in purged_splits(len(dev), C.folds, C.embargo):
        live_pos = val.iloc[train_idx].dropna()
        assert (live_pos > 0).sum() >= 10   # non-degenerate percentile fit per fold


def test_value_transform_handles_empty_pos_without_crash():
    # Amendment F3: a price window where the formation return is >= 0 everywhere ->
    # value raw <= 0 everywhere -> fit_percentile_transform gets empty `pos` -> value
    # scores flat (0), no crash. Exercises the degenerate path the golden families never hit.
    from advisor.research.candidate_signals import VALUE, candidate_raw
    p = pd.Series(100.0 * np.exp(np.cumsum(np.full(600, 0.004))))      # strictly rising
    raw = candidate_raw(VALUE, p, value_skip=126, value_lookback=270)  # formation>=0 -> value<=0
    params = fit_percentile_transform(raw.iloc[:400])
    assert params["pos"] == []                        # empty positive distribution
    scores = apply_transform(params, raw)
    assert (scores == 0).all()                        # all flat, no NaN/crash
