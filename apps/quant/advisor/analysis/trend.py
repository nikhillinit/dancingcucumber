from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.schemas import Direction, FamilySignal

FAMILY = "trend"
SHORT, LONG = 50, 200
MIN_HISTORY = LONG + 1


def evaluate(prices: pd.Series, as_of: date) -> FamilySignal:
    prices = prices.dropna()
    if len(prices) < MIN_HISTORY:
        return FamilySignal.neutral(FAMILY, as_of, "insufficient price history")

    short_ma = float(prices.rolling(SHORT).mean().iloc[-1])
    long_ma = float(prices.rolling(LONG).mean().iloc[-1])
    if long_ma <= 0:
        return FamilySignal.neutral(FAMILY, as_of, "non-positive long moving average")

    gap = short_ma / long_ma - 1.0
    if gap > 0.01:
        direction = Direction.BULLISH
    elif gap < -0.01:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + min(abs(gap), 0.5) * 100.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"SMA{SHORT}/SMA{LONG} gap={gap:.1%}")
