from math import e
from statistics import NormalDist

import numpy as np
import pandas as pd

from advisor.backtest.stats import book_sharpe
from advisor.backtest.validation import (
    deflated_sharpe, effective_n_pca, minbtl_exceeded, n_for_dsr, per_obs_sharpe,
    psr, sharpe_moments, tstat_meets_hurdle, validation_report, var_sr_trials,
)
from advisor.backtest.validation_prereg import DEFAULT_VALIDATION

_GAMMA = 0.5772156649015329


def _r(mean, sd=0.01, n=1250, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n))


def test_per_obs_sharpe_is_annualized_over_sqrt_252():
    r = _r(0.0005)
    assert abs(per_obs_sharpe(r) - book_sharpe(r) / np.sqrt(252)) < 1e-12


def test_per_obs_sharpe_zero_on_zero_vol():
    assert per_obs_sharpe(pd.Series([0.001, 0.001, 0.001])) == 0.0


def test_sharpe_moments_returns_T_skew_kurt():
    r = _r(0.0005, seed=3)
    T, skew, kurt = sharpe_moments(r)
    assert T == 1250
    assert abs(kurt - 3.0) < 0.5        # ~normal -> Pearson kurtosis near 3
    assert abs(skew) < 0.5


def test_psr_matches_closed_form():
    # SR_hat=0.10 (per-obs), SR*=0.0, T=1250, skew=0, kurt=3 (normal)
    sr, T = 0.10, 1250
    denom = (1 - 0.0 * sr + ((3.0 - 1) / 4) * sr ** 2) ** 0.5
    expected = NormalDist().cdf((sr - 0.0) * (T - 1) ** 0.5 / denom)
    assert abs(psr(sr_hat=sr, sr_benchmark=0.0, T=T, skew=0.0, kurt=3.0) - expected) < 1e-12


def test_psr_rises_with_more_observations():
    base = dict(sr_hat=0.08, sr_benchmark=0.0, skew=0.0, kurt=3.0)
    assert psr(T=300, **base) < psr(T=3000, **base)


def test_psr_negative_skew_lowers_confidence():
    base = dict(sr_hat=0.08, sr_benchmark=0.0, T=1250, kurt=6.0)
    assert psr(skew=-2.0, **base) < psr(skew=0.0, **base)


def _ref_sr0(N, var_sr):
    """Independent re-derivation of the deflated benchmark from the published formula."""
    z = NormalDist()
    return var_sr ** 0.5 * ((1 - _GAMMA) * z.inv_cdf(1 - 1.0 / N)
                            + _GAMMA * z.inv_cdf(1 - 1.0 / (N * e)))


def _ref_dsr(returns, N, var_sr, sr_benchmark=0.0):
    sr = per_obs_sharpe(returns)
    T, skew, kurt = sharpe_moments(returns)
    sr0 = max(sr_benchmark, _ref_sr0(N, var_sr))
    denom = (1 - skew * sr + ((kurt - 1) / 4) * sr ** 2) ** 0.5
    return NormalDist().cdf((sr - sr0) * (T - 1) ** 0.5 / denom)


def test_dsr_matches_independent_reference():
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    got = deflated_sharpe(r, n_trials=45, var_sr=4e-4, sr_benchmark=0.0)
    assert abs(got - _ref_dsr(r, 45, 4e-4)) < 1e-12


def test_dsr_units_handoff_example_reconciled():
    # Annualized 2.5 -> per-obs; verifies the module agrees with the hand formula
    # on the handoff's scenario. NOT a magic 0.90 literal; an independent re-derivation.
    import numpy as np
    r = _r(2.5 / np.sqrt(252) * 0.01, sd=0.01, n=1250, seed=11)  # ~per-obs SR 0.1575
    got = deflated_sharpe(r, n_trials=100, var_sr=3e-4)
    assert abs(got - _ref_dsr(r, 100, 3e-4)) < 1e-12


def test_dsr_falls_as_trial_count_rises():
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    assert deflated_sharpe(r, n_trials=200, var_sr=4e-4) < deflated_sharpe(r, n_trials=5, var_sr=4e-4)


def test_dsr_n1_is_undeflated_and_does_not_raise():
    # N=1 -> no multiple testing -> SR0=0 -> DSR == PSR at the benchmark. inv_cdf(0) must
    # never be hit (StatisticsError). Guards n_for_dsr's floor-of-1 path.
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    T, skew, kurt = sharpe_moments(r)
    assert abs(deflated_sharpe(r, n_trials=1, var_sr=4e-4)
               - psr(per_obs_sharpe(r), 0.0, T, skew, kurt)) < 1e-12


def test_larger_var_sr_is_stricter():
    # Direction of the dominant knob: larger cross-trial dispersion -> larger SR0 -> lower DSR.
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    assert deflated_sharpe(r, n_trials=45, var_sr=1e-3) < deflated_sharpe(r, n_trials=45, var_sr=1e-9)


def test_var_sr_trials_is_sample_variance_of_trial_sharpes():
    import numpy as np
    sharpes = [0.05, 0.10, 0.15, 0.20]
    assert abs(var_sr_trials(sharpes) - float(np.var(sharpes, ddof=1))) < 1e-12


def test_deflation_bites_at_production_var_sr():
    # Bind the "fails harder" property to the value that SHIPS, not a hand-picked strict one.
    from advisor.backtest.validation_prereg import DEFAULT_VALIDATION
    v = DEFAULT_VALIDATION.declared_var_sr
    r = _r(0.0005, sd=0.01, n=1250, seed=5)            # moderate Sharpe near the floor's level
    T, skew, kurt = sharpe_moments(r)
    undeflated = psr(per_obs_sharpe(r), 0.0, T, skew, kurt)   # N=1 equivalent
    deflated = deflated_sharpe(r, n_trials=45, var_sr=v)
    assert deflated < undeflated                       # multiple-testing penalty bites at ship config


def test_effective_n_collapses_correlated_series():
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 500)
    fams = {                                   # 3 near-identical -> Neff ~ 1
        "a": pd.Series(base + rng.normal(0, 1e-5, 500)),
        "b": pd.Series(base + rng.normal(0, 1e-5, 500)),
        "c": pd.Series(base + rng.normal(0, 1e-5, 500)),
    }
    assert effective_n_pca(fams) < 1.5


def test_effective_n_independent_series_near_count():
    import numpy as np
    rng = np.random.default_rng(0)
    fams = {k: pd.Series(rng.normal(0, 0.01, 500)) for k in ("a", "b", "c", "d")}
    assert effective_n_pca(fams) > 3.0         # ~4 independent


def test_n_for_dsr_never_below_declared():
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 500)
    fams = {k: pd.Series(base) for k in ("a", "b")}   # Neff ~ 1
    assert n_for_dsr(fams, declared_trials_N=45) == 45  # declared dominates
    assert n_for_dsr(fams, declared_trials_N=0) == 1    # floors at >=1


def test_minbtl_exceeded_above_budget():
    assert minbtl_exceeded(n_trials=60, max_trials=45) is True
    assert minbtl_exceeded(n_trials=45, max_trials=45) is False
    assert minbtl_exceeded(n_trials=10, max_trials=45) is False


def test_tstat_hurdle():
    assert tstat_meets_hurdle(3.5, hurdle=3.0) is True
    assert tstat_meets_hurdle(3.0, hurdle=3.0) is True
    assert tstat_meets_hurdle(2.9, hurdle=3.0) is False
    assert tstat_meets_hurdle(None, hurdle=3.0) is False   # no claim -> not met


def test_validation_report_shape_and_deflation():
    cand = _r(0.0012, sd=0.01, n=1250, seed=7)
    fams = {"ensemble": cand, "best_family": _r(0.0002, seed=8)}
    rep = validation_report(cand, fams, DEFAULT_VALIDATION)
    for k in ("per_obs_sharpe", "T", "skew", "kurt", "declared_N", "effective_N",
              "n_used", "var_sr", "dsr", "dsr_pass_bar", "dsr_passes", "tstat_met",
              "minbtl_exceeded", "passes"):
        assert k in rep
    assert rep["n_used"] == 45                 # declared dominates effective-N
    assert rep["dsr_pass_bar"] == 0.95
    # tstat is None here -> no-op; passes = dsr_passes and not over-budget
    assert rep["passes"] == (rep["dsr_passes"] and not rep["minbtl_exceeded"])


def test_minbtl_checks_actual_n_used_not_declared():
    # n_used (here = declared) above the budget must flag minbtl_exceeded.
    from advisor.backtest.validation_prereg import ValidationPreReg
    over = ValidationPreReg(declared_trials_N=50, minbtl_max_trials=45)
    cand = _r(0.0012, sd=0.01, n=1250, seed=7)
    fams = {"ensemble": cand, "best_family": _r(0.0002, seed=8)}
    rep = validation_report(cand, fams, over)
    assert rep["n_used"] == 50 and rep["minbtl_exceeded"] is True and rep["passes"] is False


def test_tstat_claim_below_hurdle_fails_otherwise_passing_candidate():
    # A strong (DSR-passing) candidate with a supplied weak t-stat must NOT pass;
    # the same candidate with no t-stat claim is unaffected by the hurdle.
    from advisor.backtest.validation_prereg import ValidationPreReg
    lax = ValidationPreReg(declared_trials_N=2, dsr_pass=0.0)   # force dsr_passes True
    cand = _r(0.0015, sd=0.01, n=1250, seed=9)
    fams = {"ensemble": cand, "best_family": _r(0.0002, seed=8)}
    assert validation_report(cand, fams, lax, tstat=None)["passes"] is True
    assert validation_report(cand, fams, lax, tstat=2.0)["passes"] is False
    assert validation_report(cand, fams, lax, tstat=3.5)["passes"] is True
