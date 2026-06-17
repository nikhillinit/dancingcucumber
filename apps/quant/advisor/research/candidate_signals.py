from __future__ import annotations

import pandas as pd

from advisor.backtest.continuous_signals import raw_metric as _frozen_raw

VALUE = "value"


def candidate_raw(family: str, prices: pd.Series, *, value_skip: int = 126,
                  value_lookback: int = 270) -> pd.Series:
    """Long-flat RAW strength (sign carries direction; transform clamps <=0 to flat).
    Known price families delegate to the frozen floor metric (read-only). The new
    'value' family is intermediate-term reversal: the NEGATIVE of the formation-window
    return from `value_lookback` ago to `value_skip` ago. Skipping the recent
    `value_skip` days excludes the momentum window. Whether this is decorrelated from
    momentum is the Task-6 kill-gate, NOT an assumption. NaN for the first
    `value_lookback` rows (handled downstream as flat); keep `value_lookback` below
    fold-1's train end so no dev fold is dead (see plan horizon note)."""
    if family == VALUE:
        assert value_lookback > value_skip, "value_lookback must exceed value_skip"
        p = pd.Series(prices).astype(float)
        formation = p.shift(value_skip) / p.shift(value_lookback) - 1.0
        return -formation
    return _frozen_raw(family, prices)
