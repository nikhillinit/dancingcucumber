from datetime import date

import pytest

from advisor.portfolio.allocator import allocate, ensemble_vote
from advisor.portfolio.parity import skill_weighted_vote, vote_parity
from advisor.schemas import Direction, FamilySignal, SignalBundle

AS_OF = date(2024, 1, 2)


def _bundle(specs):
    """specs: list of (direction, confidence, skill_weight)."""
    sigs = [
        FamilySignal(family=f"f{i}", direction=d, confidence=c,
                     skill_weight=w, as_of=AS_OF)
        for i, (d, c, w) in enumerate(specs)
    ]
    return SignalBundle(ticker="AAPL", as_of=AS_OF, signals=sigs)


UNIFORM_BUNDLES = [
    _bundle([(Direction.BULLISH, 80.0, 1.0), (Direction.BULLISH, 70.0, 1.0),
             (Direction.BEARISH, 60.0, 1.0)]),
    _bundle([(Direction.BEARISH, 90.0, 2.0), (Direction.BULLISH, 50.0, 2.0)]),
    _bundle([(Direction.BULLISH, 65.0, 0.5), (Direction.BULLISH, 55.0, 0.5)]),
    _bundle([(Direction.BULLISH, 80.0, 1.0), (Direction.BEARISH, 80.0, 1.0)]),  # tie -> neutral
]


@pytest.mark.parametrize("bundle", UNIFORM_BUNDLES)
def test_uniform_skill_weights_are_parity(bundle):
    """When all skill_weight are equal, the two seams must agree exactly."""
    live = ensemble_vote(bundle)
    weighted = skill_weighted_vote(bundle)
    assert live[0] is weighted[0]
    assert live[1] == pytest.approx(weighted[1])

    audit = vote_parity(bundle)
    assert audit["skill_weights_uniform"] is True
    assert audit["direction_match"] is True
    assert audit["live_direction"] is audit["weighted_direction"]
    assert audit["live_confidence"] == pytest.approx(audit["weighted_confidence"])


def test_allocate_action_sign_agrees_with_vote_direction():
    """allocate's action sign must track ensemble_vote's direction."""
    for specs in (
        [(Direction.BULLISH, 80.0, 1.0), (Direction.BULLISH, 70.0, 1.0)],
        [(Direction.BEARISH, 80.0, 1.0), (Direction.BEARISH, 70.0, 1.0)],
        [(Direction.BULLISH, 80.0, 1.0), (Direction.BEARISH, 80.0, 1.0)],  # tie
    ):
        bundle = _bundle(specs)
        direction, _ = ensemble_vote(bundle)
        action = allocate(bundle, price=100.0, position_limit_dollars=25_000.0).action
        if direction is Direction.BULLISH:
            assert action == "buy"
        elif direction is Direction.BEARISH:
            assert action == "sell"
        else:
            assert action == "hold"


def _flip_bundle():
    # equal-weight: +80 +80 -60 = +100 -> BULLISH
    # skill-weighted: 80*0.1 + 80*0.1 - 60*5.0 = -284 -> BEARISH
    return _bundle([
        (Direction.BULLISH, 80.0, 0.1),
        (Direction.BULLISH, 80.0, 0.1),
        (Direction.BEARISH, 60.0, 5.0),
    ])


def test_nonuniform_flip_setup_is_a_real_divergence():
    """Guard: the flip bundle genuinely diverges between the two seams."""
    bundle = _flip_bundle()
    assert ensemble_vote(bundle)[0] is Direction.BULLISH
    assert skill_weighted_vote(bundle)[0] is Direction.BEARISH
    audit = vote_parity(bundle)
    assert audit["skill_weights_uniform"] is False
    assert audit["direction_match"] is False


@pytest.mark.xfail(
    reason="live ensemble_vote ignores skill_weight; tracked as future work at the live seam",
    strict=False,
)
def test_live_seam_honors_skill_weight():
    """DOCUMENTED GAP: when skill_weight is non-uniform the live seam should
    already match the intended skill-weighted seam. Turns green the day the
    live ensemble_vote honors skill_weight."""
    bundle = _flip_bundle()
    assert ensemble_vote(bundle) == skill_weighted_vote(bundle)
