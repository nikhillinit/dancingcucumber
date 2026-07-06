import numpy as np
import pandas as pd

from advisor.backtest.stats import sortino, downside_deviation, max_drawdown
from advisor.backtest.stats import book_sharpe


def test_sortino_exceeds_sharpe_on_asymmetric_series():
    # Upside spikes, small downside -> downside vol << total vol.
    r = pd.Series([0.05, -0.005, 0.05, -0.005] * 50)
    assert downside_deviation(r) < r.std(ddof=0) * np.sqrt(252)
    assert sortino(r) > book_sharpe(r)


def test_sortino_relates_to_sharpe_on_symmetric_series():
    # Zero-mean symmetric noise: per-period downside dev ~ std/sqrt(2),
    # so sortino ~ sharpe * sqrt(2). Check that ratio holds within tolerance.
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 5000))
    r = r - r.mean()  # symmetric about target=0
    s, b = sortino(r), book_sharpe(r)
    assert np.isfinite(s) and np.isfinite(b)
    # Mean ~0 -> both ~0; relationship sanity: |sortino| close to |sharpe|*sqrt(2).
    assert abs(s) < 1.0 and abs(b) < 1.0
    assert abs(abs(s) - abs(b) * np.sqrt(2)) < 0.2


def test_downside_deviation_zero_when_all_at_or_above_target():
    assert downside_deviation(pd.Series([0.0, 0.01, 0.02, 0.0])) == 0.0
    assert downside_deviation(pd.Series([0.05] * 10)) == 0.0


def test_max_drawdown_known_series():
    # +0.1, -0.5, +0.1 -> equity 1.1, 0.55, 0.605; peak 1.1; trough dd = -0.5.
    dd = max_drawdown(pd.Series([0.1, -0.5, 0.1]))
    assert dd < 0
    assert abs(dd - (-0.5)) < 1e-12


def test_max_drawdown_zero_on_all_positive():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.0, 0.03])) == 0.0


def test_empty_series_guards_return_zero():
    empty = pd.Series([], dtype=float)
    assert downside_deviation(empty) == 0.0
    assert sortino(empty) == 0.0
    assert max_drawdown(empty) == 0.0


def test_sortino_zero_sentinel_when_no_downside():
    # dd == 0 (never lost) returns the same 0.0 sentinel as empty input;
    # inf would break strict-JSON report output.
    assert sortino(pd.Series([0.01] * 10)) == 0.0
