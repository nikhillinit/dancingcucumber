from __future__ import annotations

import pandas as pd

from advisor.backtest.residual_screen import resid
from advisor.backtest.stats import book_sharpe
from advisor.research.ls_reversal_prereg import LongShortReversalPreReg


def reversed_net_stream(net: pd.Series, turnover: pd.Series, gross_exposure: pd.Series,
                        spy: pd.Series, cfg: LongShortReversalPreReg):
    """Post-cost reversed hedged stream. book.py returns NET of one-way costs, so a
    bare sign flip turns costs into gains: recover gross (+turn*c), negate, then
    charge the mirrored trades' own costs — hence the 2x. Borrow accrues on the held
    short notional; hedge is static OLS beta (resid convention), zero rebalance cost."""
    assert len(net) == len(turnover) == len(gross_exposure) == len(spy)
    if spy.reset_index(drop=True).nunique(dropna=False) <= 1:
        a = 0.0
    else:
        _, a = resid(net, spy)
    carry = (cfg.borrow_rate_annual - cfg.short_rebate_annual) / cfg.trading_days
    rev = (-net.reset_index(drop=True)
           - 2.0 * turnover.reset_index(drop=True) * cfg.cost_per_turn
           - gross_exposure.reset_index(drop=True) * carry
           + a * spy.reset_index(drop=True))
    return rev, a


def decide(families: dict, cfg: LongShortReversalPreReg) -> dict:
    """Frozen rule. Tripwire first: every family's precost_ir must reproduce the
    published value within tolerance, else ABORT with post-cost outputs SUPPRESSED
    (the tripwire compares only to already-published numbers — no outcome peek)."""
    published = dict(zip(cfg.families, cfg.published_precost_ir))
    drifted = [f for f in cfg.families
               if abs(families[f]["precost_ir"] - published[f]) > cfg.reproduction_tolerance]
    if drifted:
        redacted = {f: {k: v for k, v in s.items() if k != "postcost_reversed_ir"}
                    for f, s in families.items()}
        return {"verdict": "ABORT", "drifted": drifted, "families": redacted,
                "tau_ls": cfg.tau_ls}
    survivors = [f for f in cfg.families
                 if families[f]["postcost_reversed_ir"] >= cfg.tau_ls]
    verdict = "PASS" if len(survivors) >= 2 else "CLOSED"
    return {"verdict": verdict, "survivors": survivors, "families": families,
            "tau_ls": cfg.tau_ls}
