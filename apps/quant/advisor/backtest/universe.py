from __future__ import annotations

import numpy as np

from advisor.backtest.prereg import PreRegConfig


def classify_universe(n_active_per_fold: list[int], cfg: PreRegConfig) -> str:
    n = np.asarray(n_active_per_fold, dtype=int)
    if n.size == 0 or n.min() < cfg.min_universe_floor:
        return "do_not_run"
    if np.median(n) >= cfg.min_universe_formal and n.min() >= cfg.min_universe_floor:
        return "formal"
    return "micro"
