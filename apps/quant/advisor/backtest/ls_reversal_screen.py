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


def _readings(cfg):
    """family -> (panel, family_cfg, raw_fn). Each family runs on its OWN frozen
    reading surface, single-family sweep, dev folds only (how the published numbers
    were measured — reading A via residual_screen conventions, B/C via their frozen
    candidate surfaces + fixtures; see the committed blend-futility note)."""
    import pandas as pd
    from advisor.backtest.residual_screen import _default_raw_fn
    from advisor.data.edgar_xbrl_fixture import load_fixture
    from advisor.research.candidate_prereg import DEFAULT_CANDIDATE
    from advisor.research.candidate_prereg_fundamental import DEFAULT_FUNDAMENTAL_CANDIDATE
    from advisor.research.candidate_prereg_lazy_prices import DEFAULT_LAZY_PRICES_CANDIDATE
    from advisor.research.fundamental_value import build_fundamental_panel, make_fundamental_raw
    from advisor.research.lazy_prices import build_lazy_prices_panel, make_lazy_prices_raw

    panel = pd.read_csv(cfg.panel, index_col=0, parse_dates=True)
    cfg_a = DEFAULT_CANDIDATE
    rb = load_fixture(cfg.fundamental_fixture)
    rc = load_fixture(cfg.lazy_prices_fixture)
    cfg_b, cfg_c = DEFAULT_FUNDAMENTAL_CANDIDATE, DEFAULT_LAZY_PRICES_CANDIDATE
    return {
        "value": (panel, cfg_a, _default_raw_fn(cfg_a)),   # reuse, don't duplicate (A3)
        "fundamental_value": (panel, cfg_b,
            make_fundamental_raw(build_fundamental_panel(rb, panel, warmup=cfg_b.warmup))),
        "lazy_prices": (panel, cfg_c,
            make_lazy_prices_raw(build_lazy_prices_panel(rc, panel, warmup=cfg_c.warmup))),
    }


def run_screen(cfg) -> dict:
    from advisor.backtest.residual_screen import spy_dev_stream
    from advisor.research.candidate_pipeline import run_dev_sweep_ext
    families = {}
    for family, (panel, fam_cfg, raw_fn) in _readings(cfg).items():
        sweep = run_dev_sweep_ext(panel, (family,), fam_cfg, raw_fn=raw_fn,
                                  holdout_frac=cfg.holdout_frac)
        spy = spy_dev_stream(panel, fam_cfg, cfg.holdout_frac)
        net = sweep.ensemble_test_returns
        res, _ = resid(net, spy)
        rev, a = reversed_net_stream(net, sweep.ensemble_test_turnover,
                                     sweep.ensemble_test_gross, spy, cfg)
        families[family] = {"precost_ir": book_sharpe(res), "beta": a,
                            "postcost_reversed_ir": book_sharpe(rev)}
    return decide(families, cfg)


def enforce_one_shot(result_path, rerun_reason) -> None:
    """Frozen one-shot rule (LS_REVERSAL_PREREG.md): PASS/CLOSED lock the outcome
    forever; ABORT permits a rerun only with an explicit operator reason."""
    import json
    from pathlib import Path
    rp = Path(result_path)
    if not rp.exists():
        return
    prior = json.loads(rp.read_text(encoding="utf-8"))
    if prior["verdict"] != "ABORT":
        raise SystemExit(
            f"REFUSED: recorded verdict={prior['verdict']} at {rp}; the L/S Gate-1 "
            "outcome stands — rerun is not permitted.")
    if not rerun_reason:
        raise SystemExit(
            "REFUSED: prior run ABORTed; rerun requires --rerun-after-abort "
            "\"<operator reason>\".")


def record_result(result_path, out: dict, rerun_reason) -> None:
    import json
    from pathlib import Path
    payload = {**out, "rerun_after_abort_reason": rerun_reason}
    Path(result_path).write_text(json.dumps(payload, indent=2, default=str),
                                 encoding="utf-8")


def main() -> None:
    import argparse
    from advisor.research.ls_reversal_prereg import (
        DEFAULT_LS_REVERSAL, ls_reversal_hash, ls_reversal_run_hash)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-after-abort", default=None,
                        help="operator reason; accepted ONLY when the recorded verdict is ABORT")
    args = parser.parse_args()
    cfg = DEFAULT_LS_REVERSAL
    enforce_one_shot(cfg.result_path, args.rerun_after_abort)
    m_hash = ls_reversal_hash(cfg)
    r_hash = ls_reversal_run_hash(cfg, cfg.panel, cfg.fundamental_fixture, cfg.lazy_prices_fixture)
    print(f"methodology_hash {m_hash}")
    print(f"run_hash {r_hash}")
    out = run_screen(cfg)
    for family, s in out["families"].items():
        post = s.get("postcost_reversed_ir")
        post_txt = f"{post:.4f}" if post is not None else "SUPPRESSED"
        print(f"{family} | precost {s['precost_ir']:.4f} | beta {s['beta']:.4f} | postcost_rev {post_txt}")
    print(f"VERDICT | {out['verdict']} (tau_ls={out['tau_ls']}, rule 2-of-3)")
    record_result(cfg.result_path,
                  {"methodology_hash": m_hash, "run_hash": r_hash, **out},
                  args.rerun_after_abort)


if __name__ == "__main__":
    main()
