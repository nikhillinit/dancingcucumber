import inspect

import numpy as np
import pandas as pd
import pytest

from advisor.backtest import prereg, portfolio, splits, continuous_signals
from advisor.backtest.pipeline import run_dev_sweep
from advisor.backtest.prereg import DEFAULT_CONFIG, PreRegConfig


def test_rail_negative_margin_is_impossible():
    with pytest.raises(ValueError):
        PreRegConfig(margin=-0.0001)


def test_rail_no_shorts_in_allocator():
    rng = np.random.default_rng(0)
    s = pd.DataFrame(rng.uniform(-5, 5, (40, 8)))      # raw-looking negatives
    w = portfolio.build_long_flat_book(s, 0.2, 1.0, 0.5, 0.0005)
    assert (w.to_numpy() >= -1e-12).all()


def test_rail_transform_maps_nonpositive_to_flat():
    params = continuous_signals.fit_percentile_transform(pd.Series([1.0, 2.0, 3.0]))
    out = continuous_signals.apply_transform(params, pd.Series([-9.0, 0.0]))
    assert out.iloc[0] == 0.0 and out.iloc[1] == 0.0


def test_rail_splits_enforce_train_before_test():
    for tr, te in splits.purged_splits(1000, 5, 5):
        assert max(tr) < min(te) - 5


def test_rail_blend_rejects_non_price_family():
    with pytest.raises(ValueError):
        from advisor.backtest.blend import select_weights
        select_weights({"macro": pd.DataFrame()}, pd.DataFrame(), ("macro",),
                       (0.25, 0.5, 0.75), 0.05, 0.0005, (0.2, 1.0, 0.2))


def test_rail_floor_code_does_not_import_live_allocator():
    # floor-only scope: no backtest module may import portfolio.allocator (the live seam)
    import advisor.backtest.pipeline as pl
    src = inspect.getsource(pl)
    assert "portfolio.allocator" not in src and "ensemble_vote" not in src
