import numpy as np
import pandas as pd

from advisor.research.candidate_signals import VALUE, candidate_raw

def _prices(n=900, drift=0.0):
    return pd.Series(100.0 * np.exp(np.cumsum(np.full(n, drift))))

def _declining_then_flat(n_decline=374, n_flat=126, drift=-0.003):
    # ONE continuous series (no concat discontinuity): falls across the formation
    # window, then flat inside the recent skip window.
    dec = 100.0 * np.exp(np.cumsum(np.full(n_decline, drift)))
    flat = np.full(n_flat, dec[-1])
    return pd.Series(np.concatenate([dec, flat]))

def test_value_is_bullish_for_intermediate_term_losers():
    # Falls across [t-270, t-126], flat in the last 126 -> formation return < 0 -> value > 0.
    p = _declining_then_flat()
    v = candidate_raw(VALUE, p, value_skip=126, value_lookback=270)
    assert v.dropna().iloc[-1] > 0   # past loser -> bullish reversal

def test_value_excludes_recent_window_via_skip():
    # value at t depends on p[t-skip] and p[t-lookback], never on p[t-1..t-skip+1].
    p = _prices(900)
    base = candidate_raw(VALUE, p, value_skip=126, value_lookback=270)
    spiked = p.copy()
    spiked.iloc[-50:] *= 2.0          # perturb only the last 50 days (inside the skip)
    after = candidate_raw(VALUE, spiked, value_skip=126, value_lookback=270)
    assert np.isclose(base.iloc[-1], after.iloc[-1])   # last 50 days don't enter value(t)

def test_known_families_delegate_to_frozen_raw():
    from advisor.backtest.continuous_signals import raw_metric
    p = _prices(900)
    for fam in ("momentum", "trend", "long_momentum", "mean_reversion", "breakout"):
        pd.testing.assert_series_equal(candidate_raw(fam, p), raw_metric(fam, p))
