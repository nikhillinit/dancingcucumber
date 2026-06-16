import numpy as np
import pandas as pd

from advisor.backtest.book import book_returns
from advisor.backtest.stats import book_sharpe


def test_full_long_book_tracks_equal_weight_basket():
    prices = pd.DataFrame({"A": np.linspace(100, 200, 50), "B": np.linspace(100, 150, 50)})
    weights = pd.DataFrame(0.5, index=prices.index, columns=["A", "B"])
    r = book_returns(weights, prices, cost_per_turn=0.0)
    assert len(r) == 50 and book_sharpe(r) > 0           # uptrend -> positive


def test_no_lookahead_first_period_is_zero():
    prices = pd.DataFrame({"A": np.linspace(100, 200, 30)})
    weights = pd.DataFrame(1.0, index=prices.index, columns=["A"])
    r = book_returns(weights, prices, cost_per_turn=0.0)
    assert r.iloc[0] == 0.0                               # position is yesterday's weight


def test_costs_reduce_book_return():
    prices = pd.DataFrame({"A": np.linspace(100, 110, 30)})
    flip = pd.DataFrame({"A": [1.0 if i % 2 == 0 else 0.0 for i in range(30)]})
    no_cost = (1 + book_returns(flip, prices, 0.0)).prod()
    with_cost = (1 + book_returns(flip, prices, 0.02)).prod()
    assert with_cost < no_cost
