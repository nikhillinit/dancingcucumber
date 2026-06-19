from __future__ import annotations

from advisor.portfolio.allocator import ensemble_vote
from advisor.schemas import Direction, SignalBundle


def skill_weighted_vote(bundle: SignalBundle) -> tuple[Direction, float]:
    """Skill-weighted vote: same shape as ensemble_vote but each family's signed
    confidence is scaled by its skill_weight. This is the *intended* deployment
    seam implied by the backtest when skill_weight != 1."""
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


def vote_parity(bundle: SignalBundle) -> dict:
    """Field-level audit comparing the live ensemble_vote seam against the
    intended skill_weighted_vote seam. The two seams are guaranteed to agree
    only when all skill_weight values are uniform."""
    live_dir, live_conf = ensemble_vote(bundle)
    w_dir, w_conf = skill_weighted_vote(bundle)
    weights = [s.skill_weight for s in bundle.signals]
    uniform = len(set(weights)) <= 1
    return {
        "direction_match": live_dir is w_dir,
        "live_direction": live_dir,
        "weighted_direction": w_dir,
        "live_confidence": live_conf,
        "weighted_confidence": w_conf,
        "skill_weights_uniform": uniform,
    }
