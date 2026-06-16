from __future__ import annotations

import numpy as np
import pandas as pd

# Each metric returns a RAW continuous strength (sign carries direction); the
# transform later clamps <=0 to flat. All are price-only and purgeable.
RAW_METRICS = ("momentum", "trend", "mean_reversion", "breakout", "long_momentum")


def raw_metric(family: str, prices: pd.Series) -> pd.Series:
    p = pd.Series(prices).astype(float)
    if family == "momentum":
        return (p / p.shift(126) - 1.0)
    if family == "long_momentum":
        return (p / p.shift(252) - 1.0)
    if family == "trend":
        return (p.rolling(50).mean() - p.rolling(200).mean()) / p
    if family == "mean_reversion":            # bullish when below short MA (snap-back)
        return (p.rolling(10).mean() - p) / p
    if family == "breakout":                  # bullish above prior 50-day high
        return (p - p.rolling(50).max().shift(1)) / p
    raise ValueError(f"unknown family {family!r}")


def fit_percentile_transform(train_raw: pd.Series, clip: tuple[float, float] = (0.05, 0.95)) -> dict:
    """Fit the conviction transform on TRAIN raw values only. Stores the empirical
    distribution of POSITIVE raw values; negatives/zeros map to flat (0)."""
    pos = pd.Series(train_raw).dropna()
    pos = pos[pos > 0].sort_values().to_numpy()
    return {"pos": pos.tolist(), "lo": clip[0], "hi": clip[1]}


def apply_transform(params: dict, raw: pd.Series) -> pd.Series:
    pos = np.asarray(params["pos"], dtype=float)
    lo, hi = params["lo"], params["hi"]
    r = pd.Series(raw).astype(float).fillna(0.0)

    def _score(x: float) -> float:
        if x <= 0 or len(pos) == 0:
            return 0.0
        pct = np.searchsorted(pos, x, side="right") / len(pos)     # empirical CDF
        return float(np.clip((pct - lo) / (hi - lo), 0.0, 1.0))

    return r.map(_score)
