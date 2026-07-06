import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

from advisor.pipeline.run import run_pipeline
from advisor.schemas import Direction, FamilySignal


def test_pipeline_produces_bounded_decision():
    prices = pd.Series(np.linspace(100, 200, 200))  # uptrend -> momentum bullish

    async def momentum_family(as_of):
        from advisor.analysis.momentum import evaluate
        return evaluate(prices, as_of)

    decision = asyncio.run(run_pipeline(
        ticker="AAPL", as_of=date(2024, 5, 1), price=100.0,
        net_liq=100_000.0, vol=0.10, correlation=0.5,
        family_coros=[momentum_family],
    ))
    assert decision.ticker == "AAPL"
    assert decision.action in {"buy", "sell", "hold"}
    # bounded by risk: vol 0.10 -> 25% of 100k = 25k -> <=250 shares at price 100
    assert decision.quantity <= 250
    assert decision.bundle_direction in {d.value for d in Direction}


def test_pipeline_applies_persona_veto():
    from advisor.analysis.momentum import evaluate
    from advisor.personas.overlay import PersonaVerdict

    prices = pd.Series(np.linspace(100, 200, 200))

    async def momentum_family(as_of):
        return evaluate(prices, as_of)

    decision = asyncio.run(run_pipeline(
        ticker="AAPL", as_of=date(2024, 5, 1), price=100.0,
        net_liq=100_000.0, vol=0.10, correlation=0.5,
        family_coros=[momentum_family],
        persona_critic=lambda d: PersonaVerdict(0.0, "forensic red flag"),
    ))

    assert decision.action == "hold"
    assert decision.quantity == 0
    assert "forensic red flag" in decision.reasoning


def test_pipeline_rejects_mixed_skill_weights():
    async def default_weight_family(as_of):
        return FamilySignal(
            family="default",
            direction=Direction.BULLISH,
            confidence=80.0,
            as_of=as_of,
        )

    async def weighted_family(as_of):
        return FamilySignal(
            family="weighted",
            direction=Direction.BEARISH,
            confidence=60.0,
            skill_weight=2.0,
            as_of=as_of,
        )

    with pytest.raises(ValueError, match="(?i)non-uniform skill weights.*validated calibration"):
        asyncio.run(run_pipeline(
            ticker="AAPL", as_of=date(2024, 5, 1), price=100.0,
            net_liq=100_000.0, vol=0.10, correlation=0.5,
            family_coros=[default_weight_family, weighted_family],
        ))


def test_pipeline_allows_uniform_skill_weights():
    async def bullish_family(as_of):
        return FamilySignal(
            family="bullish",
            direction=Direction.BULLISH,
            confidence=80.0,
            skill_weight=2.0,
            as_of=as_of,
        )

    async def bearish_family(as_of):
        return FamilySignal(
            family="bearish",
            direction=Direction.BEARISH,
            confidence=60.0,
            skill_weight=2.0,
            as_of=as_of,
        )

    decision = asyncio.run(run_pipeline(
        ticker="AAPL", as_of=date(2024, 5, 1), price=100.0,
        net_liq=100_000.0, vol=0.10, correlation=0.5,
        family_coros=[bullish_family, bearish_family],
    ))

    assert decision.ticker == "AAPL"
    assert decision.action in {"buy", "sell", "hold"}
    assert decision.bundle_direction in {d.value for d in Direction}
