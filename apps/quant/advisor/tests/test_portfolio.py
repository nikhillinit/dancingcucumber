import numpy as np
import pandas as pd

from advisor.backtest.portfolio import build_long_flat_book


def _scores(n=50, k=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(0, 1, (n, k)),
                        columns=[f"T{i}" for i in range(k)])


def test_no_shorts_and_caps_respected():
    w = build_long_flat_book(_scores(), max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=0.20, cost_per_turn=0.0005)
    assert (w.to_numpy() >= -1e-12).all()                 # long-flat only
    assert w.to_numpy().max() <= 0.20 + 1e-9              # per-name cap
    assert (w.sum(axis=1).to_numpy() <= 1.0 + 1e-9).all() # gross cap


def test_turnover_cap_enforced_each_rebalance():
    s = _scores()
    s.iloc[10:] = s.iloc[10:].values[::-1]                # force a big target swing
    w = build_long_flat_book(s, max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=0.10, cost_per_turn=0.0)
    one_way = w.diff().abs().sum(axis=1).iloc[1:]
    assert (one_way.to_numpy() <= 0.10 + 1e-9).all()


def test_all_zero_scores_go_to_cash():
    s = _scores()
    s.iloc[20] = 0.0
    w = build_long_flat_book(s, max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=1.0, cost_per_turn=0.0)
    assert w.iloc[20].sum() == 0.0                        # fully in cash


def test_hysteresis_skips_subcost_trades():
    s = _scores()
    s.iloc[5] = s.iloc[4] + 1e-6                          # negligible target change
    w = build_long_flat_book(s, max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=1.0, cost_per_turn=0.01)
    assert w.iloc[5].equals(w.iloc[4])                    # no churn below cost
