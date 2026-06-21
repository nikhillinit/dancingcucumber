from __future__ import annotations

import pandas as pd

from advisor.backtest.book import book_returns
from advisor.backtest.continuous_signals import RAW_METRICS
from advisor.backtest.portfolio import build_long_flat_book
from advisor.backtest.splits import inner_blocks
from advisor.backtest.stats import book_sharpe
from advisor.research.candidate_signals import VALUE
from advisor.research.fundamental_value import FUNDAMENTAL_VALUE
from advisor.research.lazy_prices import LAZY_PRICES

# Bench allowlist: frozen RAW_METRICS plus the two pre-registered research families.
# The frozen selector guard is a registry-membership check, not selection math. This
# remains the only divergence from frozen blend.py and is proven equal on shared families
# by the golden test.
_ALLOWED = set(RAW_METRICS) | {VALUE, FUNDAMENTAL_VALUE, LAZY_PRICES}


def _ensemble_book_sharpe(scores_by_fam, weights, prices, caps, cost) -> float:
    max_w, gross, turn = caps          # caps = (max_asset_weight, gross_cap, turnover_cap)
    cols = next(iter(scores_by_fam.values())).columns
    blended = sum(weights[f] * scores_by_fam[f] for f in weights)
    blended = pd.DataFrame(blended, columns=cols)
    w = build_long_flat_book(blended, max_w, gross, turn, cost)
    return book_sharpe(book_returns(w, prices, cost))


def select_weights(train_scores: dict, train_prices: pd.DataFrame, families: tuple,
                   grid: tuple, lift_threshold: float, cost_per_turn: float,
                   caps: tuple) -> dict:
    """Bench mirror of backtest.blend.select_weights — IDENTICAL except the family guard
    also admits pre-registered research families (frozen RAW_METRICS predates them).
    Train-only weight selection. Rule A = equal; Rule B deviates onto the grid only with
    >= lift_threshold book-Sharpe gain in >=2 inner train blocks."""
    for f in families:
        if f not in _ALLOWED:
            raise ValueError(f"non-price / unknown family rejected: {f!r}")
    n = len(families)
    equal = {f: 1.0 / n for f in families}
    if n != 2:
        return equal                                   # Rule B defined for the 2-family case
    base = _ensemble_book_sharpe(train_scores, equal, train_prices, caps, cost_per_turn)
    blocks = inner_blocks(len(train_prices), n_blocks=2)
    best = equal
    best_lift = 0.0
    f0, f1 = families
    for w0 in grid:
        cand = {f0: w0, f1: round(1.0 - w0, 6)}
        if cand == equal:
            continue
        block_lifts = []
        for blk in blocks:
            ts = {f: train_scores[f].iloc[blk] for f in families}
            tp = train_prices.iloc[blk]
            block_lifts.append(_ensemble_book_sharpe(ts, cand, tp, caps, cost_per_turn)
                               - _ensemble_book_sharpe(ts, equal, tp, caps, cost_per_turn))
        full_lift = _ensemble_book_sharpe(train_scores, cand, train_prices, caps, cost_per_turn) - base
        clears = sum(1 for bl in block_lifts if bl >= lift_threshold) >= 2
        if clears and full_lift > best_lift:
            best, best_lift = cand, full_lift
    return best
