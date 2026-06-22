from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import book_sharpe
from advisor.research.candidate_pipeline import run_dev_sweep_ext
from advisor.research.candidate_prereg import DEFAULT_CANDIDATE
from advisor.research.candidate_signals import candidate_raw

RawFn = Callable[[str, pd.Series], pd.Series]


def spy_dev_stream(panel: pd.DataFrame, cfg, holdout_frac: float) -> pd.Series:
    assets = [c for c in panel.columns if c != "SPY"]
    n = len(panel[assets].iloc[cfg.warmup:])
    dev_end = int(n * (1 - holdout_frac))
    spy_dev = panel["SPY"].iloc[cfg.warmup:].reset_index(drop=True).iloc[:dev_end]
    parts = []
    for _, test_idx in purged_splits(dev_end, cfg.folds, cfg.embargo):
        parts.append(
            spy_dev.iloc[list(test_idx)].reset_index(drop=True).pct_change().fillna(0.0)
        )
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, ignore_index=True)


def resid(stream: pd.Series, spy: pd.Series) -> tuple[pd.Series, float]:
    a, _b = np.polyfit(spy.values, stream.values, 1)
    return pd.Series(stream.values - a * spy.values), float(a)


def decide_verdict(residual_by_family: dict[str, float], tau: float) -> tuple[float, str]:
    max_residual = max(residual_by_family.values())
    return max_residual, ("GREEN" if max_residual > tau else "RED")


def residual_screen(
    panel: pd.DataFrame,
    cfg,
    raw_fn: RawFn,
    holdout_frac: float = 0.2,
    tau: float = 0.0,
) -> dict:
    spy = spy_dev_stream(panel, cfg, holdout_frac)
    families = {}
    for family in cfg.families:
        stream = run_dev_sweep_ext(
            panel, (family,), cfg, raw_fn=raw_fn, holdout_frac=holdout_frac
        ).ensemble_test_returns
        assert len(stream) == len(spy), (family, len(stream), len(spy))
        res, beta = resid(stream, spy)
        families[family] = {
            "standalone": book_sharpe(stream),
            "beta": beta,
            "residual": book_sharpe(res),
        }
    max_residual, verdict = decide_verdict(
        {family: stats["residual"] for family, stats in families.items()}, tau
    )
    return {
        "families": families,
        "max_residual": max_residual,
        "verdict": verdict,
        "tau": tau,
    }


def _default_raw_fn(cfg):
    return lambda family, prices: candidate_raw(
        family,
        prices,
        value_skip=cfg.value_skip,
        value_lookback=cfg.value_lookback,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="apps/quant/advisor/tests/fixtures/broad_prices.csv")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.0)
    args = parser.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} is missing; operator network follow-on: run build_price_fixture "
            "in advisor/data/price_fetch.py with a real S&P 500 list to produce broad_prices.csv"
        )

    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    cfg = DEFAULT_CANDIDATE
    result = residual_screen(
        panel,
        cfg,
        _default_raw_fn(cfg),
        holdout_frac=args.holdout_frac,
        tau=args.tau,
    )

    print("family | standalone | beta | residual")
    for family, stats in result["families"].items():
        print(
            f"{family} | {stats['standalone']:.4f} | "
            f"{stats['beta']:.4f} | {stats['residual']:.4f}"
        )
    print(f"max_residual | {result['max_residual']:.4f}")
    print(f"VERDICT | {result['verdict']}")


if __name__ == "__main__":
    main()
