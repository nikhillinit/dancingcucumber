from __future__ import annotations

from advisor.portfolio.allocator import ensemble_vote
from advisor.schemas import Direction, SignalBundle


def skill_weighted_vote(bundle: SignalBundle) -> tuple[Direction, float]:
    """Parity/audit helper for the live weighted seam.

    This independent mirror keeps the same shape as ensemble_vote but scales each
    family's signed confidence by skill_weight. Non-uniform weights require
    validated skill estimates; none exist as of 2026-07.
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


def vote_parity(bundle: SignalBundle) -> dict:
    """Field-level audit comparing the live ensemble_vote seam against the
    skill_weighted_vote independent mirror. Post-fix the seams must always
    agree; any divergence is a regression signal."""
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
