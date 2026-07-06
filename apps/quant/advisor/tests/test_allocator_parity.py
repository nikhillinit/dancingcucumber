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


def _mute_bundle():
    live_signal = FamilySignal(
        family="live",
        direction=Direction.BULLISH,
        confidence=80.0,
        skill_weight=1.0,
        as_of=AS_OF,
    )
    muted_signal = FamilySignal(
        family="muted",
        direction=Direction.BEARISH,
        confidence=100.0,
        skill_weight=0.0,
        as_of=AS_OF,
    )
    neutral_signal = FamilySignal.neutral("neutral", AS_OF)
    return SignalBundle(
        ticker="AAPL",
        as_of=AS_OF,
        signals=[live_signal, muted_signal, neutral_signal],
    )


def test_nonuniform_flip_setup_is_a_real_divergence():
    """Guard: raw equal voting would diverge from the live weighted seam."""
    bundle = _flip_bundle()
    raw_signed_confidence = sum(
        s.confidence if s.direction is Direction.BULLISH
        else -s.confidence if s.direction is Direction.BEARISH
        else 0.0
        for s in bundle.signals
    )
    assert raw_signed_confidence > 0
    assert ensemble_vote(bundle)[0] is Direction.BEARISH

    audit = vote_parity(bundle)
    assert audit["skill_weights_uniform"] is False
    assert audit["direction_match"] is True

    allocation = allocate(bundle, price=100.0, position_limit_dollars=25_000.0)
    assert allocation.action == "sell"
    assert allocation.quantity > 0


@pytest.mark.parametrize("bundle", UNIFORM_BUNDLES + [_flip_bundle(), _mute_bundle()])
def test_live_vote_matches_skill_weighted_oracle(bundle):
    """The live seam and independent oracle must agree for every fixture bundle."""
    assert ensemble_vote(bundle) == skill_weighted_vote(bundle)


def test_zero_weight_signal_is_excluded_and_neutral_signal_dilutes():
    bundle = _mute_bundle()
    live_signal, muted_signal, neutral_signal = bundle.signals
    live_only = SignalBundle(ticker="AAPL", as_of=AS_OF, signals=[live_signal])
    live_and_muted = SignalBundle(
        ticker="AAPL",
        as_of=AS_OF,
        signals=[live_signal, muted_signal],
    )
    live_and_neutral = SignalBundle(
        ticker="AAPL",
        as_of=AS_OF,
        signals=[live_signal, neutral_signal],
    )

    assert ensemble_vote(live_and_muted) == ensemble_vote(live_only)

    direction, confidence = ensemble_vote(bundle)
    assert direction is Direction.BULLISH
    assert confidence == pytest.approx(80.76923076923077)
    assert ensemble_vote(bundle) == ensemble_vote(live_and_neutral)
    assert confidence < ensemble_vote(live_only)[1]
    assert ensemble_vote(bundle) == skill_weighted_vote(bundle)


def test_all_zero_skill_weights_are_neutral():
    bundle = _bundle([
        (Direction.BULLISH, 80.0, 0.0),
        (Direction.BEARISH, 60.0, 0.0),
        (Direction.NEUTRAL, 50.0, 0.0),
    ])
    assert ensemble_vote(bundle) == (Direction.NEUTRAL, 50.0)


def test_allocate_guard_branches_hold():
    bundle = _bundle([(Direction.BULLISH, 80.0, 1.0), (Direction.BULLISH, 70.0, 1.0)])
    assert allocate(bundle, price=0.0, position_limit_dollars=25_000.0).action == "hold"
    assert allocate(bundle, price=-1.0, position_limit_dollars=25_000.0).action == "hold"
    assert allocate(bundle, price=100.0, position_limit_dollars=0.0).action == "hold"
    assert allocate(bundle, price=100.0, position_limit_dollars=-1.0).action == "hold"

    tiny_edge = _bundle([(Direction.BULLISH, 50.1, 1.0), (Direction.BEARISH, 50.0, 1.0)])
    allocation = allocate(tiny_edge, price=100.0, position_limit_dollars=1_000.0)
    assert allocation.action == "hold"
    assert allocation.quantity == 0


def test_live_seam_honors_skill_weight():
    """Non-uniform skill_weight is enforced at the live weighted seam."""
    bundle = _flip_bundle()
    assert ensemble_vote(bundle) == skill_weighted_vote(bundle)
