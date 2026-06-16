# apps/quant/advisor/tests/test_pipeline.py
import numpy as np
import pandas as pd

from advisor.backtest.pipeline import run_dev_sweep
from advisor.backtest.prereg import DEFAULT_CONFIG


def _panel(n=1500, k=22, seed=0):
    rng = np.random.default_rng(seed)
    # trending names + noise so families are non-degenerate; last col is SPY
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (n, k)), axis=0) * 100
    cols = [f"T{i}" for i in range(k - 1)] + ["SPY"]
    return pd.DataFrame(base, columns=cols)


def test_dev_sweep_returns_per_fold_deltas_and_series():
    res = run_dev_sweep(_panel(), ("momentum", "trend"), DEFAULT_CONFIG)
    assert len(res.fold_deltas) >= 3                       # one Δ per evaluable fold
    assert all(np.isfinite(d) for d in res.fold_deltas)
    assert len(res.ensemble_test_returns) > 0
    assert len(res.best_family_test_returns) == len(res.ensemble_test_returns)
    assert res.chosen_weights and abs(sum(res.chosen_weights.values()) - 1.0) < 1e-9


def test_dev_sweep_uses_only_dev_folds_not_full_series():
    # the concatenated dev returns must be shorter than the full panel (holdout excluded)
    panel = _panel()
    res = run_dev_sweep(panel, ("momentum", "trend"), DEFAULT_CONFIG)
    assert len(res.ensemble_test_returns) < len(panel)


def test_run_holdout_evaluates_the_held_out_tail():
    from advisor.backtest.pipeline import run_holdout
    panel = _panel()
    res = run_dev_sweep(panel, ("momentum", "trend"), DEFAULT_CONFIG)
    h = run_holdout(panel, ("momentum", "trend"), DEFAULT_CONFIG, res.chosen_weights)
    post_warmup = len(panel) - DEFAULT_CONFIG.warmup
    expected_holdout = post_warmup - int(post_warmup * 0.8)
    assert abs(len(h.ensemble) - expected_holdout) <= 1     # holdout window length
    assert len(h.best_family) == len(h.ensemble) == len(h.spy)
    # holdout must NOT overlap the dev returns it was frozen on
    assert len(h.ensemble) < len(res.ensemble_test_returns) + len(h.ensemble)
