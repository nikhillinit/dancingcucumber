import numpy as np
import pandas as pd

from advisor.backtest.stats import book_sharpe, block_bootstrap_lcb, block_bootstrap_diff_lcb


def test_book_sharpe_matches_annualized_formula():
    r = pd.Series([0.001] * 252)            # constant positive daily return
    s = book_sharpe(r)
    assert s > 0 and np.isfinite(s)
    assert book_sharpe(pd.Series([0.0] * 10)) == 0.0   # zero vol -> 0, no div0


def test_lcb_is_below_point_estimate_and_deterministic():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1000))
    lcb_a = block_bootstrap_lcb(r, block=21, draws=500, seed=7, level=0.95)
    lcb_b = block_bootstrap_lcb(r, block=21, draws=500, seed=7, level=0.95)
    assert lcb_a == lcb_b                     # seeded -> reproducible
    assert lcb_a < book_sharpe(r)             # one-sided lower bound


def test_diff_lcb_positive_when_a_clearly_better():
    # Realistic section 7.2 case: the ensemble and its constituent family are
    # CORRELATED (share market noise), so a's extra drift is statistically
    # detectable. Independent series with a modest mean gap correctly yield an
    # LCB straddling 0 (see the next test) -- right statistics, not a stronger
    # separation -- so a shared-noise fixture is the faithful "a clearly better".
    rng = np.random.default_rng(1)
    common = rng.normal(0.0, 0.01, 1500)            # shared market path
    a = pd.Series(common + 0.0015)                  # same path, clear extra drift
    b = pd.Series(common + 0.0002)
    lcb = block_bootstrap_diff_lcb(a, b, block=21, draws=500, seed=3, level=0.95)
    assert lcb > 0


def test_diff_lcb_straddles_zero_when_indistinguishable():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0.0005, 0.01, 1500))
    b = pd.Series(rng.normal(0.0005, 0.01, 1500))
    lcb = block_bootstrap_diff_lcb(a, b, block=21, draws=500, seed=4, level=0.95)
    assert lcb <= 0
