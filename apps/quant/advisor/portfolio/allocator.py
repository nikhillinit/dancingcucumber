from __future__ import annotations

from dataclasses import dataclass

from advisor.schemas import Direction, SignalBundle


@dataclass(frozen=True)
class Allocation:
    ticker: str
    action: str  # "buy" | "sell" | "hold"
    quantity: int
    reasoning: str


def ensemble_vote(bundle: SignalBundle) -> tuple[Direction, float]:
    """Skill-weighted vote across families. Net sign decides direction.

    This live seam must remain equivalent to parity.skill_weighted_vote, the
    independent mirror; the parity test suite enforces agreement.
    """
    score = 0.0
    weight = 0.0
    for s in bundle.signals:
        w = s.confidence * s.skill_weight
        if s.direction is Direction.BULLISH:
            score += w
        elif s.direction is Direction.BEARISH:
            score -= w
        weight += w
    if weight == 0 or score == 0:
        return Direction.NEUTRAL, 50.0
    direction = Direction.BULLISH if score > 0 else Direction.BEARISH
    confidence = min(100.0, 50.0 + abs(score) / weight * 50.0)
    return direction, confidence


def allocate(bundle: SignalBundle, price: float, position_limit_dollars: float) -> Allocation:
    direction, confidence = ensemble_vote(bundle)
    if direction is Direction.NEUTRAL or price <= 0 or position_limit_dollars <= 0:
        return Allocation(bundle.ticker, "hold", 0, "neutral or no capacity")

    max_shares = int(position_limit_dollars // price)
    # scale by conviction above the 50 baseline (0..1)
    conviction = max(0.0, (confidence - 50.0) / 50.0)
    quantity = int(max_shares * conviction)
    if quantity <= 0:
        return Allocation(bundle.ticker, "hold", 0, "conviction below threshold")

    action = "buy" if direction is Direction.BULLISH else "sell"
    return Allocation(bundle.ticker, action, quantity,
                      f"{direction.value} conviction={conviction:.0%}, max_shares={max_shares}")
