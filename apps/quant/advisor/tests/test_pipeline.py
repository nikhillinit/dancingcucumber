import asyncio
from datetime import date

import numpy as np
import pandas as pd

from advisor.pipeline.run import run_pipeline
from advisor.schemas import Direction


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
