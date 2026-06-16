import numpy as np
import pandas as pd

from advisor.backtest.walk_forward import (
    DISCLOSURES,
    disclosure_header,
    passes_floor,
    walk_forward,
)


def test_disclosure_header_has_all_seven_lines():
    header = disclosure_header()
    assert len(DISCLOSURES) == 7
    assert any("price-only proxy" in line for line in DISCLOSURES)
    assert any("survivorship" in line for line in DISCLOSURES)
    assert any("does not prove the 5-family" in line for line in DISCLOSURES)
    assert any("labeled market regimes" in line for line in DISCLOSURES)
    for line in DISCLOSURES:
        assert line in header


def test_long_only_on_uptrend_is_profitable_and_carries_disclosures():
    prices = pd.Series(np.linspace(100, 200, 50))  # steady uptrend
    signal = pd.Series(1.0, index=prices.index)     # always long
    result = walk_forward(prices, signal, cost_per_turn=0.0)
    assert result.total_return > 0
    assert result.n_periods == 50
    assert len(result.disclosures) == 7


def test_costs_reduce_return():
    prices = pd.Series(np.linspace(100, 110, 30))
    flip = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(30)])
    no_cost = walk_forward(prices, flip, cost_per_turn=0.0).total_return
    with_cost = walk_forward(prices, flip, cost_per_turn=0.01).total_return
    assert with_cost < no_cost


def test_passes_floor():
    assert passes_floor(1.2, 0.8, margin=0.3) is True
    assert passes_floor(1.0, 0.8, margin=0.3) is False
