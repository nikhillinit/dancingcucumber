import numpy as np
import pandas as pd

from advisor.research.orthogonality import (
    dev_fold_post_transform_corr,
    dev_fold_raw_corr,
)


def _panel(n=900, k=12, seed=1):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)


def _firing_panel(n=2200, k=12, seed=1):
    # Random walk (drift 0), long enough that dev folds clear value_lookback, so the
    # `value` leg actually FIRES (intermediate-term losers exist). On a positive-drift
    # or too-short panel `value` is all-flat and the post-transform corr is
    # degenerately 0 -- which would make the transform tests vacuous.
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, n)))
    return pd.DataFrame(cols)


# --- raw diagnostic (plan Task 5) ---

def test_returns_corr_against_each_neighbor_on_dev_only():
    panel = _panel()
    corr = dev_fold_raw_corr(panel, warmup=200, holdout_frac=0.2,
                             value_skip=126, value_lookback=400,
                             neighbors=("long_momentum", "mean_reversion"))
    assert set(corr) == {"long_momentum", "mean_reversion"}
    assert all(-1.0 <= v <= 1.0 for v in corr.values())


def test_perfect_relabel_is_detected():
    # If "value" were literally -long_momentum, |raw corr| ~ 1 -> must be caught.
    panel = _panel()
    corr = dev_fold_raw_corr(panel, warmup=200, holdout_frac=0.2,
                             value_skip=0, value_lookback=252,           # == -long_momentum
                             neighbors=("long_momentum",))
    assert abs(corr["long_momentum"]) > 0.95


def test_raw_spearman_cross_check_runs():
    # Amendment F4: rank corr as a scale-robust cross-check on the raw Pearson.
    panel = _panel()
    corr = dev_fold_raw_corr(panel, warmup=200, holdout_frac=0.2,
                             value_skip=0, value_lookback=252,
                             neighbors=("long_momentum",), method="spearman")
    assert abs(corr["long_momentum"]) > 0.95     # monotone relabel -> rank corr ~ -1 too


# --- post-transform gate surface (Amendment F4) ---

def test_post_transform_corr_keys_and_range():
    panel = _firing_panel()
    pt = dev_fold_post_transform_corr(panel, warmup=200, holdout_frac=0.2,
                                      value_skip=126, value_lookback=270,
                                      neighbors=("momentum", "long_momentum", "mean_reversion"))
    assert set(pt) == {"momentum", "long_momentum", "mean_reversion"}
    assert all(-1.0 <= v <= 1.0 for v in pt.values())


def test_post_transform_decorrelates_a_negated_relabel():
    # value(skip=0, lb=252) == -long_momentum (raw). On the RAW surface this is a
    # near-perfect (-1) relabel; on the POST-TRANSFORM (blend) surface the long-flat
    # clamp (raw<=0 -> flat) makes value and long_momentum fire on DISJOINT days, so
    # the blend-relevant corr is far weaker (~-0.3). This proves the gate reads the
    # transformed surface, not the raw (Amendment F4) -- and is exactly why a negated
    # relabel can PASS the post-transform tau while the raw diagnostic flags it.
    panel = _firing_panel()
    raw = dev_fold_raw_corr(panel, warmup=200, holdout_frac=0.2,
                            value_skip=0, value_lookback=252,
                            neighbors=("long_momentum",))
    pt = dev_fold_post_transform_corr(panel, warmup=200, holdout_frac=0.2,
                                      value_skip=0, value_lookback=252,
                                      neighbors=("long_momentum",))
    assert raw["long_momentum"] < -0.95                       # raw: near-perfect relabel
    assert -0.6 < pt["long_momentum"] < -0.05                 # transform: weak, still negative
    assert abs(pt["long_momentum"]) < abs(raw["long_momentum"]) - 0.3   # transform decorrelates


def test_post_transform_differs_from_raw_on_live_horizon():
    # Transform-actually-applied guard: on a firing panel the post-transform corr is a
    # materially different number than the raw Pearson (not a pass-through of raw_fn).
    panel = _firing_panel()
    raw = dev_fold_raw_corr(panel, warmup=200, holdout_frac=0.2,
                            value_skip=126, value_lookback=270,
                            neighbors=("long_momentum",))
    pt = dev_fold_post_transform_corr(panel, warmup=200, holdout_frac=0.2,
                                      value_skip=126, value_lookback=270,
                                      neighbors=("long_momentum",))
    assert abs(raw["long_momentum"] - pt["long_momentum"]) > 0.2
