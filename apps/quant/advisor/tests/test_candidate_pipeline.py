import numpy as np
import pandas as pd

from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.continuous_signals import raw_metric
from advisor.research.candidate_pipeline import run_dev_sweep_ext, SweepResultExt

def _panel(n=900, k=12, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)

def test_run_dev_sweep_ext_with_frozen_raw_matches_shape_and_runs():
    panel = _panel()
    cfg = PreRegConfig()
    res = run_dev_sweep_ext(panel, ("momentum", "trend"), cfg,
                            raw_fn=raw_metric, holdout_frac=0.2)
    assert isinstance(res, SweepResultExt)
    assert len(res.fold_deltas) >= 1
    assert isinstance(res.ensemble_test_returns, pd.Series)
    assert set(res.chosen_weights) == {"momentum", "trend"}
