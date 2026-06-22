import numpy as np
import pandas as pd

from advisor.backtest.residual_screen import decide_verdict, resid
from advisor.backtest.stats import book_sharpe


def test_resid_removes_pure_beta():
    rng = np.random.default_rng(0)
    spy = pd.Series(rng.normal(size=500))
    stream = 3.0 * spy

    res, beta = resid(stream, spy)

    assert abs(beta - 3.0) < 1e-6
    assert np.allclose(res.values, 0.0, atol=1e-9)


def test_resid_keeps_orthogonal_alpha():
    rng = np.random.default_rng(1)
    spy = pd.Series(rng.normal(size=500))
    g0 = pd.Series(rng.normal(size=500))
    beta_g = np.cov(spy.values, g0.values, ddof=0)[0, 1] / np.var(spy.values)
    g = g0 - beta_g * spy
    g = g - g.mean() + 0.01
    stream = spy + g

    res, beta = resid(stream, spy)

    assert abs(beta - 1.0) < 1e-6
    assert book_sharpe(g) > 0
    assert abs(book_sharpe(res) - book_sharpe(g)) < 1e-6


def test_decide_verdict_uses_strict_tau_boundary():
    assert decide_verdict({"a": -0.1, "b": 0.2}, 0.0)[1] == "GREEN"
    assert decide_verdict({"a": -0.1, "b": -0.2}, 0.0)[1] == "RED"
    assert decide_verdict({"a": 0.0}, 0.0)[1] == "RED"
