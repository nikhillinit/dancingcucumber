import numpy as np
import pandas as pd

from advisor.backtest.continuous_signals import (
    RAW_METRICS, raw_metric, fit_percentile_transform, apply_transform,
)


def _prices(n=600):
    return pd.Series(np.linspace(100, 200, n))


def test_registry_has_expected_families():
    assert set(RAW_METRICS) >= {"momentum", "trend", "mean_reversion", "breakout", "long_momentum"}


def test_uptrend_gives_positive_bull_for_momentum():
    raw = raw_metric("momentum", _prices())
    assert (raw.dropna() > 0).mean() > 0.5          # mostly bullish in an uptrend


def test_transform_is_long_flat_and_clipped():
    raw = raw_metric("trend", _prices())
    params = fit_percentile_transform(raw.iloc[:300], clip=(0.05, 0.95))
    score = apply_transform(params, raw)
    assert score.min() >= 0.0 and score.max() <= 1.0    # in [0,1], no shorts
    # a raw value <= 0 must map to exactly 0 (flat, not short)
    neg = pd.Series([-1.0, 0.0, 5.0])
    assert apply_transform(params, neg).iloc[0] == 0.0
    assert apply_transform(params, neg).iloc[1] == 0.0


def test_transform_fit_is_train_only_deterministic():
    raw = raw_metric("momentum", _prices())
    p1 = fit_percentile_transform(raw.iloc[:300])
    p2 = fit_percentile_transform(raw.iloc[:300])
    assert p1 == p2                                    # pure function of train slice
