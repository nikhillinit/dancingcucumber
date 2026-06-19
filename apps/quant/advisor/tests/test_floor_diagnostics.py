from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from advisor.backtest.data_floor import floor_metrics
from advisor.backtest.prereg import DEFAULT_CONFIG

FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")


@pytest.fixture(scope="module")
def metrics() -> dict:
    panel = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    return floor_metrics(panel, DEFAULT_CONFIG)


def test_diagnostics_block_is_present_and_well_formed(metrics):
    d = metrics["diagnostics"]
    for key in ("ensemble_sortino", "ensemble_max_drawdown", "best_family_sortino",
                "spy_sortino", "spy_max_drawdown", "concentration", "robust_family"):
        assert key in d
    # drawdowns are non-positive; sortinos are finite floats
    assert d["ensemble_max_drawdown"] <= 0.0 and d["spy_max_drawdown"] <= 0.0
    assert isinstance(d["ensemble_sortino"], float)


def test_concentration_report_shape(metrics):
    c = metrics["diagnostics"]["concentration"]
    assert "passes" in c and isinstance(c["passes"], bool)
    assert c["min_invested_breadth"] <= c["min_breadth"] or c["min_invested_breadth"] >= 0
    assert 0.0 <= c["max_single_name"] <= 1.0 + 1e-9
    assert c["thresholds"]["min_breadth"] == 9


def test_robust_family_is_a_real_family(metrics):
    rf = metrics["diagnostics"]["robust_family"]
    assert rf is not None
    assert rf["family"] in DEFAULT_CONFIG.families
    assert rf["min_fold_sharpe"] <= rf["mean_fold_sharpe"] + 1e-9


def test_diagnostics_do_not_change_the_verdict(metrics):
    # The wiring is report-only: the pre-registered floor numbers must be byte-stable.
    assert metrics["verdict"] == "DEV_FAILED"
    assert abs(metrics["ensemble"] - 0.7323) < 1e-3
    assert abs(metrics["spy"] - 0.7562) < 1e-3
    assert abs(metrics["best_family"] - 0.8277) < 1e-3
    assert metrics["passes"] is False
