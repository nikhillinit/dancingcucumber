from datetime import date

import pytest
from pydantic import ValidationError

from advisor.schemas import Direction, FamilySignal, SignalBundle


def test_family_signal_valid():
    s = FamilySignal(family="value_quality", direction=Direction.BULLISH,
                     confidence=80.0, as_of=date(2024, 1, 2))
    assert s.direction is Direction.BULLISH
    assert s.skill_weight == 1.0


def test_confidence_must_be_0_100():
    with pytest.raises(ValidationError):
        FamilySignal(family="x", direction=Direction.NEUTRAL,
                     confidence=150.0, as_of=date(2024, 1, 2))


def test_skill_weight_must_be_non_negative():
    with pytest.raises(ValidationError):
        FamilySignal(family="x", direction=Direction.BULLISH,
                     confidence=80.0, skill_weight=-0.1,
                     as_of=date(2024, 1, 2))


def test_skill_weight_must_not_be_nan():
    with pytest.raises(ValidationError):
        FamilySignal(family="x", direction=Direction.BULLISH,
                     confidence=80.0, skill_weight=float("nan"),
                     as_of=date(2024, 1, 2))


def test_skill_weight_must_not_be_inf():
    with pytest.raises(ValidationError):
        FamilySignal(family="x", direction=Direction.BULLISH,
                     confidence=80.0, skill_weight=float("inf"),
                     as_of=date(2024, 1, 2))


def test_skill_weight_must_not_exceed_upper_bound():
    with pytest.raises(ValidationError):
        FamilySignal(family="x", direction=Direction.BULLISH,
                     confidence=80.0, skill_weight=101.0,
                     as_of=date(2024, 1, 2))


def test_neutral_fallback():
    s = FamilySignal.neutral("value_quality", date(2024, 1, 2))
    assert s.direction is Direction.NEUTRAL
    assert s.confidence == 50.0


def test_bundle_collects_signals():
    s = FamilySignal.neutral("value_quality", date(2024, 1, 2))
    b = SignalBundle(ticker="AAPL", as_of=date(2024, 1, 2), signals=[s])
    assert b.signals[0].family == "value_quality"
