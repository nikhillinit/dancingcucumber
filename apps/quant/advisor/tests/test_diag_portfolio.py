from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advisor.diagnostics.portfolio import DiagnosticsInputError, load_portfolio


FIXTURES = Path(__file__).parent / "fixtures"


def _csv(text: str) -> StringIO:
    return StringIO(text)


def test_qty_positions_load_from_fixtures():
    loaded = load_portfolio(
        FIXTURES / "diag_positions.csv",
        FIXTURES / "diag_prices_adjusted.csv",
    )

    assert loaded.tickers == ["AAA", "BBB", "CCC"]
    assert loaded.cash_dollars == 1000.0
    assert loaded.n_obs == 9
    assert loaded.start.isoformat() == "2026-01-02"
    assert loaded.end.isoformat() == "2026-01-15"
    assert loaded.dropped_dates == 0


def test_weights_input_rejected_by_column_name():
    with pytest.raises(DiagnosticsInputError, match="weights input rejected: column weight"):
        load_portfolio(_csv("ticker,weight\nAAA,1.0\n"), FIXTURES / "diag_prices_adjusted.csv")


def test_shorts_rejected_before_price_guards():
    with pytest.raises(DiagnosticsInputError, match="shorts rejected: AAA"):
        load_portfolio(_csv("ticker,qty\nAAA,-1\n"), FIXTURES / "diag_prices_adjusted.csv")


def test_cash_row_keeps_invested_weights_below_one():
    loaded = load_portfolio(
        FIXTURES / "diag_positions.csv",
        FIXTURES / "diag_prices_adjusted.csv",
    )

    first = loaded.weights_book.iloc[0]
    assert loaded.cash_dollars == 1000.0
    assert first["AAA"] == 1000 / 2650
    assert first["BBB"] == 250 / 2650
    assert first["CCC"] == 400 / 2650
    assert first.sum() == 1650 / 2650


def test_malformed_or_empty_csv_rejected():
    with pytest.raises(DiagnosticsInputError, match="positions CSV parse error"):
        load_portfolio(_csv(""), FIXTURES / "diag_prices_adjusted.csv")
    with pytest.raises(DiagnosticsInputError, match="positions CSV parse error"):
        load_portfolio(_csv("ticker,qty\nAAA,not-a-number\n"), FIXTURES / "diag_prices_adjusted.csv")


def test_unknown_position_ticker_rejected():
    with pytest.raises(DiagnosticsInputError, match="unknown ticker in positions: ZZZ"):
        load_portfolio(_csv("ticker,qty\nZZZ,1\n"), FIXTURES / "diag_prices_adjusted.csv")


def test_intersection_window_reports_dropped_dates():
    positions = _csv("ticker,qty\nAAA,1\nBBB,1\n")
    prices = _csv(
        "\n".join(
            [
                "Date,AAA,BBB",
                "2026-01-01,10,20",
                "2026-01-02,11,",
                "2026-01-05,12,21",
                "2026-01-06,13,22",
                "",
            ]
        ),
    )

    loaded = load_portfolio(positions, prices)

    assert loaded.start.isoformat() == "2026-01-01"
    assert loaded.end.isoformat() == "2026-01-06"
    assert loaded.dropped_dates == 1
    assert loaded.weights_book.index.strftime("%Y-%m-%d").tolist() == [
        "2026-01-01",
        "2026-01-05",
        "2026-01-06",
    ]


def test_partial_history_rejected_by_ticker():
    positions = _csv("ticker,qty\nAAA,1\nBBB,1\n")
    prices = _csv(
        "\n".join(
            [
                "Date,AAA,BBB",
                "2026-01-01,10,",
                "2026-01-02,11,20",
                "2026-01-05,12,21",
                "",
            ]
        ),
    )

    with pytest.raises(DiagnosticsInputError, match="partial history for BBB"):
        load_portfolio(positions, prices)


def test_weekly_frequency_rejected():
    positions = _csv("ticker,qty\nAAA,1\n")
    prices = _csv(
        "\n".join(
            [
                "Date,AAA",
                "2026-01-01,10",
                "2026-01-08,11",
                "2026-01-15,12",
                "",
            ]
        ),
    )

    with pytest.raises(DiagnosticsInputError, match="prices frequency error"):
        load_portfolio(positions, prices)


def test_jump_guard_rejects_unadjusted_split_fixture():
    with pytest.raises(DiagnosticsInputError, match="phantom split guard tripped for BBB on 2026-01-08"):
        load_portfolio(
            FIXTURES / "diag_positions.csv",
            FIXTURES / "diag_prices_unadjusted_split.csv",
        )


def test_single_name_book_loads_without_cash():
    loaded = load_portfolio(_csv("ticker,qty\nAAA,10\n"), FIXTURES / "diag_prices_adjusted.csv")

    assert loaded.tickers == ["AAA"]
    assert loaded.cash_dollars is None
    assert loaded.n_obs == 9
    pd.testing.assert_series_equal(
        loaded.weights_book["AAA"],
        pd.Series(np.ones(10), index=loaded.weights_book.index, name="AAA"),
    )


def test_equity_curve_oracle_uses_adjusted_prices_and_cash():
    loaded = load_portfolio(
        FIXTURES / "diag_positions.csv",
        FIXTURES / "diag_prices_adjusted.csv",
    )

    # Hand-computed total equity:
    # 2026-01-02: 10*100 + 5*50 + 2*200 + 1000 = 2650
    # 2026-01-05: 10*102 + 5*51 + 2*198 + 1000 = 2671
    # 2026-01-06: 10*101 + 5*52 + 2*202 + 1000 = 2674
    # 2026-01-07: 10*103 + 5*53 + 2*204 + 1000 = 2703
    # 2026-01-08: 10*104 + 5*54 + 2*206 + 1000 = 2722
    # 2026-01-09: 10*105 + 5*55 + 2*208 + 1000 = 2741
    # 2026-01-12: 10*107 + 5*56 + 2*210 + 1000 = 2770
    # 2026-01-13: 10*106 + 5*57 + 2*212 + 1000 = 2769
    # 2026-01-14: 10*108 + 5*58 + 2*214 + 1000 = 2798
    # 2026-01-15: 10*110 + 5*59 + 2*216 + 1000 = 2827
    expected_returns = np.array(
        [
            21 / 2650,
            3 / 2671,
            29 / 2674,
            19 / 2703,
            19 / 2722,
            29 / 2741,
            -1 / 2770,
            29 / 2769,
            29 / 2798,
        ]
    )

    np.testing.assert_allclose(loaded.returns.to_numpy(), expected_returns, rtol=0, atol=1e-15)


def test_duplicate_ticker_rejected_case_insensitive():
    with pytest.raises(DiagnosticsInputError, match="duplicate ticker rejected: AAA"):
        load_portfolio(_csv("ticker,qty\nAAA,1\naaa,2\n"), FIXTURES / "diag_prices_adjusted.csv")
