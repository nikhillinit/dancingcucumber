import numpy as np
import pandas as pd

from advisor.backtest.data_floor import floor_metrics
from advisor.backtest.prereg import DEFAULT_CONFIG


def test_floor_metrics_back_compat_keys_and_types():
    rng = np.random.default_rng(0)
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (1600, 22)), axis=0) * 100
    panel = pd.DataFrame(base, columns=[f"T{i}" for i in range(21)] + ["SPY"])
    m = floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    for key in ("ensemble", "spy", "best_family", "margin", "passes"):
        assert key in m
    assert np.isfinite(m["ensemble"]) and isinstance(m["passes"], bool)
