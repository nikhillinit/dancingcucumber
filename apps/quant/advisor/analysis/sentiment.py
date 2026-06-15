from __future__ import annotations

from datetime import date
from typing import Callable

from advisor.schemas import Direction, FamilySignal

FAMILY = "sentiment"
THRESHOLD = 0.2

NewsScorer = Callable[[str], float]  # headline -> score in [-1, 1]


def evaluate(headlines: list[str], as_of: date, scorer: NewsScorer) -> FamilySignal:
    """Average scored news surprise. Missing source -> neutral, never fabricated (spec section 10)."""
    if not headlines:
        return FamilySignal.neutral(FAMILY, as_of, "no news available")

    scores = [max(-1.0, min(1.0, float(scorer(h)))) for h in headlines]
    avg = sum(scores) / len(scores)
    if avg > THRESHOLD:
        direction = Direction.BULLISH
    elif avg < -THRESHOLD:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + abs(avg) * 50.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"avg news score={avg:+.2f} over {len(scores)} headlines")
