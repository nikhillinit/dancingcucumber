import numpy as np
import pandas as pd

from advisor.backtest.data_floor import floor_metrics
from advisor.backtest.prereg import DEFAULT_CONFIG


def _panel(n=1600, k=22, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (n, k)), axis=0) * 100
    return pd.DataFrame(base, columns=[f"T{i}" for i in range(k - 1)] + ["SPY"])


def test_floor_metrics_reports_without_touching_holdout_when_no_prereg():
    m = floor_metrics(_panel(), DEFAULT_CONFIG, prereg_hash=None)
    assert m["verdict"] in {"DEV_FAILED", "INCONCLUSIVE", "UNSUPPORTED", "PASSED"}
    assert "holdout" not in m or m["holdout"] is None     # holdout blinded without prereg
    for legacy in ("ensemble", "spy", "best_family", "margin", "passes"):
        assert legacy in m                                # back-compat keys present
    assert isinstance(m["passes"], bool)


def test_holdout_gates_on_both_parts_and_spy():
    m = floor_metrics(_panel(), DEFAULT_CONFIG, prereg_hash="deadbeef")
    if m["dev"]["passed"]:
        assert m["holdout"] is not None                   # gated holdout evaluated
        h = m["holdout"]
        assert {"beats_parts", "beats_spy", "delta_lcb", "spy_lcb"} <= set(h)
        # PASSED iff BOTH section 7.2 (beat the parts) AND section 7.1 (beat SPY) clear
        assert (m["verdict"] == "PASSED") == (h["beats_parts"] and h["beats_spy"])
    else:
        assert m["holdout"] is None                       # dev fail -> no holdout


def test_verdict_print_includes_validation_caveat(capsys):
    from tools.floor_data_check import _print_verdict
    m = {
        "verdict": "DEV_FAILED", "universe": "formal",
        "ensemble": 0.73, "spy": 0.85, "best_family": 0.83,
        "dev": {"reasons": ["median fold delta not > 0"]},
        "validation": {"dsr": 0.41, "dsr_pass_bar": 0.95, "passes": False,
                       "n_used": 45, "minbtl_exceeded": False},
    }
    _print_verdict(m)
    out = capsys.readouterr().out
    assert "DSR" in out and "0.41" in out and "report-only" in out.lower()
