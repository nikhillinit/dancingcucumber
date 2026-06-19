from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.backtest.concentration import (
    breadth,
    max_name_weight,
    top_k_weight,
    concentration_report,
    passes_concentration,
)


def _equal_book(n_names: int, n_rows: int = 3) -> pd.DataFrame:
    cols = [f"T{i}" for i in range(n_names)]
    w = 1.0 / n_names
    return pd.DataFrame(np.full((n_rows, n_names), w), columns=cols)


def test_diversified_book_passes():
    book = _equal_book(10)
    ok, report = passes_concentration(book)
    assert ok is True
    assert report["min_breadth"] == 10
    assert report["max_single_name"] == 0.1
    assert abs(report["max_top_k"] - 0.5) < 1e-12
    assert report["n_rows"] == 3
    assert report["k"] == 5


def test_concentrated_book_fails_single_name_and_breadth():
    # one name at 0.9, rest flat -> breadth 1, max single 0.9
    book = pd.DataFrame([[0.9, 0.0, 0.0]], columns=["A", "B", "C"])
    ok, report = passes_concentration(book)
    assert ok is False
    assert report["max_single_name"] == 0.9        # breaches max_single_name (0.25)
    assert report["min_invested_breadth"] == 1     # breaches min_breadth (9)


def test_book_breaches_only_top_k():
    # 5 heavy names at 0.13 (top-5 = 0.65 > 0.60) plus a long diversified tail.
    heavy = [0.13] * 5
    tail = [0.35 / 10] * 10  # 10 tail names totalling 0.35, each 0.035
    weights = heavy + tail
    cols = [f"H{i}" for i in range(5)] + [f"L{i}" for i in range(10)]
    book = pd.DataFrame([weights], columns=cols)
    ok, report = passes_concentration(book)
    assert ok is False
    assert abs(report["max_top_k"] - 0.65) < 1e-12   # breaches max_top_k (0.60)
    assert report["max_single_name"] == 0.13         # single-name ok (<=0.25)
    assert report["min_invested_breadth"] == 15      # breadth ok (>=9)


def test_per_row_series_values():
    cols = ["A", "B", "C", "D"]
    book = pd.DataFrame(
        [
            [0.5, 0.3, 0.2, 0.0],
            [0.4, 0.4, 0.1, 0.1],
        ],
        columns=cols,
    )
    pd.testing.assert_series_equal(
        breadth(book), pd.Series([3, 4], dtype=int)
    )
    pd.testing.assert_series_equal(
        max_name_weight(book), pd.Series([0.5, 0.4])
    )
    tk = top_k_weight(book, k=2)
    assert abs(tk.iloc[0] - 0.8) < 1e-12   # 0.5 + 0.3
    assert abs(tk.iloc[1] - 0.8) < 1e-12   # 0.4 + 0.4


def test_warmup_flat_rows_ignored():
    body = _equal_book(10, n_rows=2)
    cols = list(body.columns)
    flat = pd.DataFrame(np.zeros((3, 10)), columns=cols)
    book = pd.concat([flat, body], ignore_index=True)
    ok, report = passes_concentration(book)
    assert ok is True
    # raw min_breadth over all rows includes the zero warmup rows...
    assert report["min_breadth"] == 0
    # ...but the invested-only breadth floor stays diversified.
    assert report["min_invested_breadth"] == 10
