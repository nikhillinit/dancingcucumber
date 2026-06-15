from datetime import date

from advisor.analysis.sentiment import FAMILY, evaluate
from advisor.schemas import Direction


def test_positive_news_is_bullish():
    sig = evaluate(["earnings beat estimates"], date(2024, 5, 1), scorer=lambda h: 1.0)
    assert sig.family == FAMILY
    assert sig.direction is Direction.BULLISH


def test_negative_news_is_bearish():
    sig = evaluate(["guidance cut, probe opened"], date(2024, 5, 1), scorer=lambda h: -1.0)
    assert sig.direction is Direction.BEARISH


def test_no_news_is_neutral_not_fabricated():
    sig = evaluate([], date(2024, 5, 1), scorer=lambda h: 1.0)
    assert sig.direction is Direction.NEUTRAL
    assert "no news" in sig.reasoning.lower()


def test_mixed_news_near_zero_is_neutral():
    sig = evaluate(["a", "b"], date(2024, 5, 1), scorer=lambda h: 1.0 if h == "a" else -1.0)
    assert sig.direction is Direction.NEUTRAL
