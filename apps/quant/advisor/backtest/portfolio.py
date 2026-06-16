from __future__ import annotations

import numpy as np
import pandas as pd


def _target_weights(scores_row: np.ndarray, max_asset_weight: float, gross_cap: float) -> np.ndarray:
    s = np.where(scores_row > 0, scores_row, 0.0)
    total = s.sum()
    if total <= 0:
        return np.zeros_like(s)
    w = s / total * gross_cap
    # iterative cap: clip to max, redistribute excess to uncapped names
    for _ in range(10):
        over = w > max_asset_weight
        if not over.any():
            break
        excess = (w[over] - max_asset_weight).sum()
        w[over] = max_asset_weight
        room = ~over & (w > 0)
        if not room.any():
            break
        w[room] += excess * (w[room] / w[room].sum())
    return np.minimum(w, max_asset_weight)


def build_long_flat_book(scores: pd.DataFrame, max_asset_weight: float, gross_cap: float,
                         turnover_cap: float, cost_per_turn: float) -> pd.DataFrame:
    """Deterministic long-flat weights with per-rebalance one-way turnover cap +
    hysteresis. Row t depends only on scores<=t and the prior row (no look-ahead)."""
    cols = list(scores.columns)
    prev = np.zeros(len(cols))
    rows = []
    for _, row in scores.iterrows():
        target = _target_weights(row.to_numpy(dtype=float), max_asset_weight, gross_cap)
        delta = target - prev
        gross_turn = np.abs(delta).sum()
        if gross_turn <= cost_per_turn:                  # hysteresis: skip churn
            new = prev.copy()
        elif gross_turn > turnover_cap and gross_turn > 0:
            new = prev + delta * (turnover_cap / gross_turn)   # scale move to cap
        else:
            new = target
        rows.append(new)
        prev = new
    return pd.DataFrame(rows, index=scores.index, columns=cols)
