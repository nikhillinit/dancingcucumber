from __future__ import annotations

from datetime import date

from advisor.analysis.valuation_primitives import intrinsic_value_dcf, owner_earnings
from advisor.data.provider import Fundamentals
from advisor.schemas import Direction, FamilySignal

FAMILY = "value_quality"


def _quality_score(f: Fundamentals) -> int:
    """0..6 from ROE, operating margin, and leverage."""
    roe = f.net_income / f.total_equity if f.total_equity else 0.0
    op_margin = f.operating_income / f.revenue if f.revenue else 0.0
    d_e = f.total_debt / f.total_equity if f.total_equity else float("inf")
    score = 0
    if roe > 0.15:
        score += 2
    if op_margin > 0.15:
        score += 2
    if d_e < 0.5:
        score += 2
    return score


def evaluate(f: Fundamentals, as_of: date) -> FamilySignal:
    if f.total_equity <= 0 or f.revenue <= 0 or f.market_cap <= 0:
        return FamilySignal.neutral(FAMILY, as_of, "non-positive equity/revenue/market cap")

    quality = _quality_score(f)
    iv = intrinsic_value_dcf(owner_earnings(f.net_income, f.depreciation, f.capex))
    margin_of_safety = (iv - f.market_cap) / f.market_cap

    if margin_of_safety > 0.15 and quality >= 4:
        direction = Direction.BULLISH
    elif margin_of_safety < -0.15:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + quality * 5.0 + min(abs(margin_of_safety), 1.0) * 30.0)
    reasoning = (f"quality={quality}/6, margin_of_safety={margin_of_safety:.0%}, "
                 f"intrinsic={iv:.0f} vs market_cap={f.market_cap:.0f}")
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=reasoning)
