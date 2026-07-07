import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from advisor.backtest.ls_reversal_screen import decide, reversed_net_stream
from advisor.backtest.residual_screen import resid
from advisor.backtest.stats import book_sharpe
from advisor.research.ls_reversal_prereg import DEFAULT_LS_REVERSAL

RNG = np.random.default_rng(7)


def _streams(n=500):
    spy = pd.Series(RNG.normal(0.0004, 0.01, n))
    net = pd.Series(0.9 * spy.values + RNG.normal(-0.0002, 0.004, n))  # negative-alpha book
    turnover = pd.Series(RNG.uniform(0.0, 0.2, n))
    gross = pd.Series(RNG.uniform(0.8, 1.0, n))
    return net, turnover, gross, spy


def test_zero_cost_identity():
    net, turnover, gross, spy = _streams()
    cfg = replace(DEFAULT_LS_REVERSAL, cost_per_turn=0.0, borrow_rate_annual=0.0)
    rev, a = reversed_net_stream(net, turnover, gross, spy, cfg)
    res, a2 = resid(net, spy)
    assert a == a2
    assert book_sharpe(rev) == pytest.approx(-book_sharpe(res), abs=1e-12)


def test_hand_computed_costs():
    net = pd.Series([0.01, -0.01]); spy = pd.Series([0.0, 0.0])
    turnover = pd.Series([0.1, 0.2]); gross = pd.Series([1.0, 1.0])
    cfg = replace(DEFAULT_LS_REVERSAL, trading_days=252)
    rev, a = reversed_net_stream(net, turnover, gross, spy, cfg)
    b = 0.005 / 252
    # rev = -net - 2*turn*c - gross*b + a*spy (spy=0 kills the hedge term)
    assert rev.iloc[0] == pytest.approx(-0.01 - 2 * 0.1 * 0.0005 - b)
    assert rev.iloc[1] == pytest.approx(+0.01 - 2 * 0.2 * 0.0005 - b)


def test_costs_are_monotone_drag():
    net, turnover, gross, spy = _streams()
    lo, _ = reversed_net_stream(net, turnover, gross, spy, DEFAULT_LS_REVERSAL)
    hi, _ = reversed_net_stream(net, turnover, gross, spy,
                                replace(DEFAULT_LS_REVERSAL, borrow_rate_annual=0.05))
    assert book_sharpe(hi) < book_sharpe(lo)


def _family(precost, postcost):
    return {"precost_ir": precost, "postcost_reversed_ir": postcost, "beta": 0.9}


def test_tripwire_abort_suppresses_postcost():
    fams = {"value": _family(-0.10, 0.5),                       # drifted vs published -0.41
            "fundamental_value": _family(-0.32, 0.5),
            "lazy_prices": _family(-0.40, 0.5)}
    out = decide(fams, DEFAULT_LS_REVERSAL)
    assert out["verdict"] == "ABORT"
    assert all("postcost_reversed_ir" not in f for f in out["families"].values())


def test_pass_needs_two_of_three_at_tau():
    ok = {"value": _family(-0.41, 0.20), "fundamental_value": _family(-0.32, 0.20),
          "lazy_prices": _family(-0.40, 0.05)}
    assert decide(ok, DEFAULT_LS_REVERSAL)["verdict"] == "PASS"
    one = {"value": _family(-0.41, 0.35), "fundamental_value": _family(-0.32, 0.10),
           "lazy_prices": _family(-0.40, 0.10)}
    assert decide(one, DEFAULT_LS_REVERSAL)["verdict"] == "CLOSED"
