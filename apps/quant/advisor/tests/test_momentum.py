from datetime import date

import numpy as np
import pandas as pd

from advisor.analysis.momentum import FAMILY, evaluate
from advisor.schemas import Direction


def test_uptrend_is_bullish():
    prices = pd.Series(np.linspace(100, 200, 200))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.family == FAMILY
    assert sig.direction is Direction.BULLISH
    assert sig.confidence > 50


def test_downtrend_is_bearish():
    prices = pd.Series(np.linspace(200, 100, 200))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.direction is Direction.BEARISH


def test_insufficient_history_is_neutral():
    prices = pd.Series(np.linspace(100, 110, 20))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL
