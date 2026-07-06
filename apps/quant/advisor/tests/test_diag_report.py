from __future__ import annotations

import json
from datetime import date

import pandas as pd

from advisor.backtest.concentration import concentration_report
from advisor.backtest.stats import (
    block_bootstrap_lcb,
    book_sharpe,
    downside_deviation,
    max_drawdown,
    sortino,
)
from advisor.diagnostics.portfolio import LoadedPortfolio
from advisor.diagnostics.report import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    build_report,
    render_json,
    render_text,
)


def _lp(returns: list[float], weights: pd.DataFrame | None = None) -> LoadedPortfolio:
    r = pd.Series(returns, dtype=float)
    book = weights if weights is not None else pd.DataFrame(
        [[0.6, 0.4] for _ in returns],
        columns=["AAA", "BBB"],
    )
    return LoadedPortfolio(
        returns=r,
        weights_book=book,
        n_obs=len(r),
        dropped_dates=2,
        tickers=list(book.columns),
        cash_dollars=1000.0,
        start=date(2024, 1, 2),
        end=date(2024, 12, 31),
    )


def test_report_metrics_match_stats_and_concentration_calls():
    returns = [0.01, -0.004, 0.006, -0.003, 0.002] * 60
    weights = pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.45, 0.35, 0.2]] * 150,
        columns=["AAA", "BBB", "CCC"],
    )

    report = build_report(_lp(returns, weights))
    metrics = report["metrics"]

    assert metrics["book_sharpe"]["value"] == round(book_sharpe(pd.Series(returns)), 6)
    assert metrics["sortino"]["value"] == round(sortino(pd.Series(returns)), 6)
    assert metrics["downside_deviation"]["value"] == round(downside_deviation(pd.Series(returns)), 6)
    assert metrics["max_drawdown"]["value"] == round(max_drawdown(pd.Series(returns)), 6)
    assert metrics["block_bootstrap_lcb"]["value"] == round(
        block_bootstrap_lcb(
            pd.Series(returns),
            block=BOOTSTRAP_BLOCK,
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        ),
        6,
    )
    assert report["concentration"] == concentration_report(weights)


def test_report_uses_zero_variance_sentinel_instead_of_zero_sharpe():
    report = build_report(_lp([0.01] * 30))

    assert report["metrics"]["book_sharpe"] == {
        "value": None,
        "note": "n/a \u2014 zero variance",
    }
    text = render_text(report)
    assert "book_sharpe: n/a \u2014 zero variance" in text
    assert "book_sharpe: 0.0000" not in text


def test_report_uses_no_downside_sentinel_instead_of_zero_sortino():
    report = build_report(_lp([0.001, 0.002, 0.003] * 10))

    assert report["metrics"]["sortino"] == {
        "value": None,
        "note": "n/a \u2014 no downside observed",
    }
    assert "sortino: n/a \u2014 no downside observed" in render_text(report)


def test_report_uses_short_window_sentinel_instead_of_zero_lcb():
    report = build_report(_lp([0.01, -0.004, 0.006, -0.002]))

    assert report["metrics"]["block_bootstrap_lcb"] == {
        "value": None,
        "note": "n/a \u2014 window < block",
    }
    assert "block_bootstrap_lcb: n/a \u2014 window < block" in render_text(report)


def test_report_banner_is_present_in_text_and_json():
    report = build_report(_lp([0.01, -0.004, 0.006, -0.003, 0.002] * 60))

    text = render_text(report)
    data = json.loads(render_json(report))

    assert "DISCLOSURES:" in text
    assert "Diagnostics report-only: no signal, direction, or sizing." in text
    assert "Advisor floor is DEV_FAILED; advisor has no validated alpha." in text
    assert "Diagnostics report-only: no signal, direction, or sizing." in data["disclosures"]
    assert "Advisor floor is DEV_FAILED; advisor has no validated alpha." in data["disclosures"]


def test_report_basis_statement_is_present_in_text_and_json():
    report = build_report(_lp([0.01, -0.004, 0.006, -0.003, 0.002] * 60))

    assert "split+dividend-adjusted total-return prices" in render_text(report)
    assert "split+dividend-adjusted total-return prices" in render_json(report)


def test_report_bootstrap_metadata_and_short_history_advisory():
    report = build_report(_lp([0.01, -0.004, 0.006, -0.003, 0.002] * 6))

    assert report["bootstrap"] == {
        "seed": BOOTSTRAP_SEED,
        "block": BOOTSTRAP_BLOCK,
        "draws": BOOTSTRAP_DRAWS,
        "n_obs": 30,
    }
    assert report["advisories"] == ["window under one year; LCB unstable"]
    assert "window under one year; LCB unstable" in render_text(report)


def test_report_rendering_is_byte_identical_and_uses_lf_newlines():
    report = build_report(_lp([0.01, -0.004, 0.006, -0.003, 0.002] * 60))

    text_a = render_text(report).encode("utf-8")
    text_b = render_text(report).encode("utf-8")
    json_a = render_json(report).encode("utf-8")
    json_b = render_json(report).encode("utf-8")

    assert text_a == text_b
    assert json_a == json_b
    assert b"\r\n" not in text_a
    assert b"\r\n" not in json_a
    assert text_a.endswith(b"\n")
    assert json_a.endswith(b"\n")


def test_report_json_is_strict_parseable():
    report = build_report(_lp([0.01, -0.004, 0.006, -0.003, 0.002] * 60))

    parsed = json.loads(render_json(report))

    assert parsed == report
