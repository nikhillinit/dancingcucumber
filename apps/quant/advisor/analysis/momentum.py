from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.schemas import Direction, FamilySignal

FAMILY = "momentum"
HORIZONS = (63, 126)  # ~3m and ~6m trading days
MIN_HISTORY = max(HORIZONS) + 1


def evaluate(prices: pd.Series, as_of: date) -> FamilySignal:
    prices = prices.dropna()
    if len(prices) < MIN_HISTORY:
        return FamilySignal.neutral(FAMILY, as_of, "insufficient price history")

    returns = []
    for h in HORIZONS:
        past = prices.iloc[-(h + 1)]
        if past > 0:
            returns.append(prices.iloc[-1] / past - 1.0)
    if not returns:
        return FamilySignal.neutral(FAMILY, as_of, "no valid horizon returns")

    score = sum(returns) / len(returns)  # average multi-horizon momentum
    if score > 0.02:
        direction = Direction.BULLISH
    elif score < -0.02:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + min(abs(score), 1.0) * 100.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"avg multi-horizon momentum={score:.1%}")
