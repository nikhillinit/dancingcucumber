from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from advisor.backtest.data_floor import build_diagnostics, floor_metrics
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
    assert c["min_invested_breadth"] >= c["min_breadth"] >= 0
    assert 0.0 <= c["max_single_name"] <= 1.0 + 1e-9
    assert c["thresholds"]["min_breadth"] == 9


def test_metrics_are_strict_json(metrics):
    json.loads(json.dumps(metrics, allow_nan=False))


def test_spy_diagnostics_use_dev_oos_window(metrics):
    assert metrics["diagnostics"]["spy_window"] == "dev_oos"
    assert metrics["diagnostics"]["spy_n_obs"] == metrics["validation"]["T"]


def test_print_verdict_covers_diagnostics_lines(metrics, capsys):
    from tools.floor_data_check import _print_verdict

    _print_verdict(metrics)
    out = capsys.readouterr().out
    assert "risk diagnostics (report-only)" in out
    assert "book concentration (report-only)" in out
    assert "minimax-robust family" in out


def test_print_verdict_labels_actual_thresholds(capsys):
    from tools.floor_data_check import _print_verdict

    m = {
        "verdict": "DEV_FAILED",
        "dev": {"reasons": ["median fold delta not > 0"]},
        "ensemble": 0.73,
        "spy": 0.85,
        "best_family": 0.83,
        "diagnostics": {
            "ensemble_sortino": 0.5,
            "ensemble_max_drawdown": -0.1,
            "spy_sortino": 0.4,
            "spy_max_drawdown": -0.2,
            "concentration": {
                "thresholds": {
                    "min_breadth": 5,
                    "max_single_name": 0.5,
                    "max_top_k": 0.9,
                    "k": 5,
                },
                "min_invested_breadth": 5,
                "median_invested_breadth": 6.0,
                "max_single_name": 0.4,
                "max_top_k": 0.8,
                "k": 5,
                "passes": True,
            },
            "robust_family": None,
        },
    }
    _print_verdict(m)
    out = capsys.readouterr().out
    assert "vs 5/50%/90%" in out


def test_empty_book_concentration_marked_not_computed(metrics):
    stub = SimpleNamespace(
        ensemble_test_returns=pd.Series([0.01, -0.02, 0.005]),
        best_family_test_returns=pd.Series([0.01, 0.0, 0.01]),
        ensemble_book=pd.DataFrame(),
        family_fold_sharpes={},
    )
    d = build_diagnostics(stub, pd.Series([0.0, 0.01, -0.01]))
    assert d["concentration"]["computed"] is False
    assert d["concentration"]["passes"] is False
    assert d["robust_family"] is None
    assert metrics["diagnostics"]["concentration"]["computed"] is True


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
