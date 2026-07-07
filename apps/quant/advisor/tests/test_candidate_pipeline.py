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

def test_sweep_exposes_turnover_and_gross_exposure():
    # single-family sweep, same _panel()/raw_metric setup as the test above
    res = run_dev_sweep_ext(_panel(), ("momentum",), PreRegConfig(),
                            raw_fn=raw_metric, holdout_frac=0.2)
    assert len(res.ensemble_test_turnover) == len(res.ensemble_test_returns)
    assert len(res.ensemble_test_gross) == len(res.ensemble_test_returns)
    assert (res.ensemble_test_turnover >= 0).all()
    assert (res.ensemble_test_gross >= 0).all()


def test_turnover_reconstructs_gross_from_net():
    # net + turnover*c must equal the zero-cost book on identical weights (book.py mirror)
    import pandas as pd
    from advisor.backtest.book import book_returns
    idx = pd.RangeIndex(6)
    prices = pd.DataFrame({"A": [10, 11, 12, 11, 12, 13], "B": [20, 19, 21, 22, 21, 23]}, index=idx)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.0, 0.5, 0.5, 0.5], "B": [0.5, 0.0, 0.5, 0.5, 0.5, 0.0]}, index=idx)
    c = 0.0005
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    pd.testing.assert_series_equal(
        book_returns(w, prices, c) + turnover * c,
        book_returns(w, prices, 0.0),
        atol=1e-12, rtol=0.0, check_names=False,
    )
