from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.stats import block_bootstrap_diff_lcb, book_sharpe


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def dev_gate(fold_deltas: list[float], ensemble_returns: pd.Series,
             best_family_returns: pd.Series, cfg: PreRegConfig,
             fold_excess: list[float] | None = None) -> GateResult:
    reasons: list[str] = []
    d = np.asarray(fold_deltas, dtype=float)
    if d.size == 0:
        return GateResult(False, ["no dev folds evaluated"])
    if np.median(d) <= 0:
        reasons.append("median fold delta not > 0")
    if (d > 0).mean() < 0.70:
        reasons.append("fewer than 70% positive folds")
    lcb = block_bootstrap_diff_lcb(ensemble_returns, best_family_returns,
                                   cfg.bootstrap_block, cfg.bootstrap_draws,
                                   cfg.bootstrap_seed, level=cfg.dev_lcb)
    if lcb <= 0:
        reasons.append(f"dev {int(cfg.dev_lcb*100)}% bootstrap LCB of delta not > 0")
    total_lift = book_sharpe(ensemble_returns) - book_sharpe(best_family_returns)
    if total_lift < cfg.train_lift_threshold:
        reasons.append(f"total dev book-Sharpe lift < {cfg.train_lift_threshold}")
    if fold_excess is not None and len(fold_excess) > 0:
        fe = np.asarray(fold_excess, dtype=float)
        if fe.sum() > 0 and fe.max() / fe.sum() > 0.60:
            reasons.append("single-fold concentration > 60% of excess")
    return GateResult(passed=not reasons, reasons=reasons)
