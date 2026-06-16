import numpy as np
import pandas as pd

from advisor.backtest.data_floor import floor_metrics
from advisor.backtest.prereg import DEFAULT_CONFIG


def _panel(n=1600, k=22, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (n, k)), axis=0) * 100
    return pd.DataFrame(base, columns=[f"T{i}" for i in range(k - 1)] + ["SPY"])


def test_floor_metrics_back_compat_keys_and_types():
    rng = np.random.default_rng(0)
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (1600, 22)), axis=0) * 100
    panel = pd.DataFrame(base, columns=[f"T{i}" for i in range(21)] + ["SPY"])
    m = floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    for key in ("ensemble", "spy", "best_family", "margin", "passes"):
        assert key in m
    assert np.isfinite(m["ensemble"]) and isinstance(m["passes"], bool)


def test_floor_metrics_validation_is_additive_only(monkeypatch):
    """Report-only invariant: validation_report's result must NOT change verdict,
    passes, holdout, or any legacy metric. Forcing passes=False changes ONLY the
    additive 'validation' key."""
    from advisor.backtest import data_floor
    from advisor.backtest.prereg import DEFAULT_CONFIG
    panel = _panel()
    baseline = data_floor.floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    assert "validation" in baseline and baseline["validation"]["dsr_pass_bar"] == 0.95

    # Force the gate to "fail" and prove nothing else moves.
    monkeypatch.setattr(data_floor, "validation_report",
                        lambda *a, **k: {"passes": False, "sentinel": True})
    forced = data_floor.floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    for k in set(baseline) | set(forced):
        if k == "validation":
            continue
        assert forced[k] == baseline[k], f"validation leaked into {k!r}"
    assert forced["validation"] == {"passes": False, "sentinel": True}
