from datetime import date

from advisor.portfolio.allocator import ensemble_vote, allocate
from advisor.schemas import Direction, FamilySignal, SignalBundle


def _bundle(dirs):
    sigs = [FamilySignal(family=f"f{i}", direction=d, confidence=80.0, as_of=date(2024, 5, 1))
            for i, d in enumerate(dirs)]
    return SignalBundle(ticker="AAPL", as_of=date(2024, 5, 1), signals=sigs)


def test_ensemble_vote_majority_bullish():
    d, conf = ensemble_vote(_bundle([Direction.BULLISH, Direction.BULLISH, Direction.BEARISH]))
    assert d is Direction.BULLISH
    assert 0 < conf <= 100


def test_ensemble_tie_is_neutral():
    d, _ = ensemble_vote(_bundle([Direction.BULLISH, Direction.BEARISH]))
    assert d is Direction.NEUTRAL


def test_allocate_buy_bounded_by_limit():
    # bullish, limit 25k, price 100 -> max 250 shares; target weight scaled by confidence
    a = allocate(_bundle([Direction.BULLISH, Direction.BULLISH]), price=100.0,
                 position_limit_dollars=25_000.0)
    assert a.action == "buy"
    assert 0 < a.quantity <= 250


def test_allocate_neutral_holds():
    a = allocate(_bundle([Direction.BULLISH, Direction.BEARISH]), price=100.0,
                 position_limit_dollars=25_000.0)
    assert a.action == "hold"
    assert a.quantity == 0
