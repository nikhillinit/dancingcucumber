from datetime import date

from advisor.analysis.value_quality import FAMILY, evaluate
from advisor.data.provider import Fundamentals
from advisor.schemas import Direction


def _fund(net_income, total_equity, revenue, operating_income, total_debt,
          depreciation, capex, market_cap) -> Fundamentals:
    return Fundamentals(period_end=date(2023, 12, 31), net_income=net_income,
                        total_equity=total_equity, revenue=revenue,
                        operating_income=operating_income, total_debt=total_debt,
                        depreciation=depreciation, capex=capex,
                        shares_outstanding=10, market_cap=market_cap)


def test_high_quality_cheap_is_bullish():
    # ROE 20%, op margin 25%, D/E 0.2 (quality 6/6); cheap vs DCF -> bullish
    f = _fund(net_income=200, total_equity=1000, revenue=800, operating_income=200,
              total_debt=200, depreciation=60, capex=40, market_cap=500)
    sig = evaluate(f, date(2024, 5, 1))
    assert sig.family == FAMILY
    assert sig.direction is Direction.BULLISH
    assert sig.confidence > 50


def test_expensive_is_bearish():
    f = _fund(net_income=50, total_equity=1000, revenue=800, operating_income=80,
              total_debt=900, depreciation=20, capex=60, market_cap=50_000)
    sig = evaluate(f, date(2024, 5, 1))
    assert sig.direction is Direction.BEARISH


def test_nonpositive_equity_is_neutral():
    f = _fund(net_income=10, total_equity=0, revenue=100, operating_income=5,
              total_debt=10, depreciation=5, capex=5, market_cap=100)
    sig = evaluate(f, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL
