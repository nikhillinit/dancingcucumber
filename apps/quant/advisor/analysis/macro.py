from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.schemas import Direction, FamilySignal

FAMILY = "macro"
MIN_HISTORY = 2
STEEP_THRESHOLD = 0.5  # 10y-2y spread (pct points) above which the curve is risk-on


def evaluate(yield_curve_spread: pd.Series, as_of: date) -> FamilySignal:
    """Regime from the 10y-2y Treasury spread (FRED T10Y2Y). Inverted -> bearish."""
    s = yield_curve_spread.dropna()
    if len(s) < MIN_HISTORY:
        return FamilySignal.neutral(FAMILY, as_of, "insufficient macro history")

    latest = float(s.iloc[-1])
    if latest < 0:
        direction = Direction.BEARISH
    elif latest > STEEP_THRESHOLD:
        direction = Direction.BULLISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + min(abs(latest), 2.0) / 2.0 * 50.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"10y-2y spread={latest:.2f}")
