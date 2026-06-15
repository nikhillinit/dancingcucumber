from datetime import date

import pandas as pd

from advisor.analysis.macro import FAMILY, evaluate
from advisor.schemas import Direction


def test_inverted_curve_is_bearish():
    spread = pd.Series([0.5, 0.1, -0.3])  # 10y-2y inverted at the latest point
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.family == FAMILY
    assert sig.direction is Direction.BEARISH


def test_steep_curve_is_bullish():
    spread = pd.Series([0.2, 0.6, 1.0])
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.direction is Direction.BULLISH


def test_flat_curve_is_neutral():
    spread = pd.Series([0.1, 0.2, 0.3])
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL


def test_insufficient_history_is_neutral():
    spread = pd.Series([0.3])
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL
