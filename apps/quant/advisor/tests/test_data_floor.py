import numpy as np
import pandas as pd

from advisor.backtest.data_floor import floor_metrics


def _panel():
    # synthetic 2-regime panel: uptrend then drawdown, plus a benchmark column
    n = 600
    up = np.linspace(100, 180, n // 2)
    down = np.linspace(180, 130, n - n // 2)
    series = np.concatenate([up, down])
    return pd.DataFrame({"AAA": series, "BBB": series * 1.01, "SPY": series * 0.99})


def test_floor_metrics_has_expected_keys_and_types():
    m = floor_metrics(_panel(), benchmark="SPY", margin=0.0)
    for key in ("ensemble", "spy", "best_family", "margin", "passes"):
        assert key in m
    assert np.isfinite(m["ensemble"])
    assert np.isfinite(m["spy"])
    assert np.isfinite(m["best_family"])
    assert isinstance(m["passes"], bool)


def test_floor_metrics_passes_is_consistent_with_beats_floor():
    from advisor.backtest.floor import beats_floor

    m = floor_metrics(_panel(), benchmark="SPY", margin=0.0)
    assert m["passes"] == beats_floor(m["ensemble"], m["spy"], m["best_family"], m["margin"])
