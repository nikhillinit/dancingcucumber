import numpy as np
import pandas as pd
import pytest

from advisor.backtest.blend import select_weights


def _train_scores(families, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return {f: pd.DataFrame(rng.uniform(0, 1, (n, 12))) for f in families}


def test_defaults_to_equal_weight_when_no_clear_lift():
    fam = ("momentum", "trend")
    prices = pd.DataFrame(np.linspace(100, 120, 300).repeat(12).reshape(300, 12))
    w = select_weights(_train_scores(fam), prices, fam, grid=(0.25, 0.5, 0.75),
                       lift_threshold=0.05, cost_per_turn=0.0005, caps=(0.2, 1.0, 0.2))
    assert w == {"momentum": 0.5, "trend": 0.5}          # Rule A wins by default


def test_weights_sum_to_one_and_exclude_endpoints():
    fam = ("momentum", "trend")
    prices = pd.DataFrame(np.linspace(100, 120, 300).repeat(12).reshape(300, 12))
    w = select_weights(_train_scores(fam), prices, fam, grid=(0.25, 0.5, 0.75),
                       lift_threshold=0.05, cost_per_turn=0.0005, caps=(0.2, 1.0, 0.2))
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(0 < v < 1 for v in w.values())            # never collapse to a single family


def test_rejects_non_price_family():
    with pytest.raises(ValueError):
        select_weights({"sentiment": pd.DataFrame()}, pd.DataFrame(), ("sentiment",),
                       grid=(0.25, 0.5, 0.75), lift_threshold=0.05,
                       cost_per_turn=0.0005, caps=(0.2, 1.0, 0.2))
