from __future__ import annotations

import numpy as np
import pandas as pd

"""Breadth & anti-concentration gating for a long-flat weights book.

Pure, additive read-only diagnostics over a weights DataFrame `book`
(rows=time/rebalance, cols=tickers, values=weights). Mirrors the
anti-concentration discipline from the walk-forward options methodology:
a breadth floor, a single-name cap and a top-k cap. Nothing here mutates
the book or feeds the recorded floor evidence."""


def breadth(book: pd.DataFrame, eps: float = 1e-9) -> pd.Series:
    """Per-row count of names with weight > eps."""
    return (book > eps).sum(axis=1).astype(int)


def max_name_weight(book: pd.DataFrame) -> pd.Series:
    """Per-row maximum single-name weight."""
    return book.max(axis=1)


def top_k_weight(book: pd.DataFrame, k: int = 5) -> pd.Series:
    """Per-row sum of the k largest weights."""
    arr = np.sort(book.to_numpy(dtype=float), axis=1)[:, ::-1][:, :k]
    return pd.Series(arr.sum(axis=1), index=book.index)


def concentration_report(book: pd.DataFrame, k: int = 5) -> dict:
    """Aggregate summary across all rows of the book."""
    b = breadth(book)
    mnw = max_name_weight(book)
    tkw = top_k_weight(book, k=k)
    return {
        "min_breadth": int(b.min()),
        "median_breadth": float(b.median()),
        "max_single_name": float(mnw.max()),
        "max_top_k": float(tkw.max()),
        "n_rows": int(len(book)),
        "k": int(k),
    }


def passes_concentration(book: pd.DataFrame, *, min_breadth: int = 9,
                         max_single_name: float = 0.25, max_top_k: float = 0.60,
                         k: int = 5, eps: float = 1e-9) -> tuple[bool, dict]:
    """Gate the book on breadth floor, single-name cap and top-k cap.

    Defaults (9 names / 25% / top-5 60%) come from the reference methodology;
    callers may tune via keyword args. Only invested rows (weights sum > eps)
    count toward the breadth floor so all-flat warmup rows don't trip the gate.
    """
    invested = book[book.sum(axis=1) > eps]
    breadth_floor = int(breadth(invested).min()) if len(invested) else 0

    report = concentration_report(book, k=k)
    report["min_invested_breadth"] = breadth_floor
    report["median_invested_breadth"] = float(breadth(invested).median()) if len(invested) else 0.0
    report["thresholds"] = {
        "min_breadth": min_breadth,
        "max_single_name": max_single_name,
        "max_top_k": max_top_k,
        "k": k,
    }

    ok = bool(
        breadth_floor >= min_breadth
        and report["max_single_name"] <= max_single_name
        and report["max_top_k"] <= max_top_k
    )
    return ok, report
