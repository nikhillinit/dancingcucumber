from __future__ import annotations

import json
from typing import Any

import pandas as pd

from advisor.backtest.concentration import concentration_report
from advisor.backtest.stats import (
    block_bootstrap_lcb,
    book_sharpe,
    downside_deviation,
    max_drawdown,
    sortino,
)
from advisor.backtest.walk_forward import DISCLOSURES, disclosure_header
from advisor.diagnostics.portfolio import LoadedPortfolio

BOOTSTRAP_SEED = 42
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_DRAWS = 1000

SORTINO_SENTINEL = "n/a \u2014 no downside observed"
LCB_SENTINEL = "n/a \u2014 window < block"
SHARPE_SENTINEL = "n/a \u2014 zero variance"
MIN_N_ADVISORY = "window under one year; LCB unstable"

DIAGNOSTIC_DISCLOSURES = [
    "Diagnostics report-only: no signal, direction, or sizing.",
    "Advisor floor is DEV_FAILED; advisor has no validated alpha.",
    "One-ETF books report max_single_name = 1.0 by construction.",
    "Inputs assumed split+dividend-adjusted total-return prices.",
]


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _metric(value: float | None, note: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": None if value is None else _round_float(value)}
    if note is not None:
        out["note"] = note
    return out


def _jsonable_concentration(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(report):
        value = report[key]
        if isinstance(value, float):
            out[key] = _round_float(value)
        else:
            out[key] = value
    return out


def _zero_variance(returns: pd.Series) -> bool:
    return len(returns) == 0 or int(returns.nunique(dropna=True)) <= 1


def _no_downside(returns: pd.Series) -> bool:
    return len(returns) == 0 or not bool((returns < 0.0).any())


def build_report(lp: LoadedPortfolio) -> dict:
    returns = pd.Series(lp.returns, dtype=float).dropna()
    zero_variance = _zero_variance(returns)
    no_downside = _no_downside(returns)
    short_window = len(returns) < BOOTSTRAP_BLOCK

    metrics = {
        "book_sharpe": _metric(None, SHARPE_SENTINEL)
        if zero_variance else _metric(book_sharpe(returns)),
        "sortino": _metric(None, SORTINO_SENTINEL)
        if no_downside else _metric(sortino(returns)),
        "downside_deviation": _metric(downside_deviation(returns)),
        "max_drawdown": _metric(max_drawdown(returns)),
        "block_bootstrap_lcb": _metric(None, LCB_SENTINEL)
        if short_window else _metric(
            block_bootstrap_lcb(
                returns,
                block=BOOTSTRAP_BLOCK,
                draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
        ),
    }

    advisories = [MIN_N_ADVISORY] if len(returns) < 250 else []

    return {
        "disclosures": list(DISCLOSURES) + list(DIAGNOSTIC_DISCLOSURES),
        "basis": DIAGNOSTIC_DISCLOSURES[-1],
        "scope": "total book including cash"
        if lp.cash_dollars is not None else "invested sleeve only",
        "portfolio": {
            "tickers": list(lp.tickers),
            "cash_dollars": None if lp.cash_dollars is None else _round_float(lp.cash_dollars),
            "start": lp.start.isoformat(),
            "end": lp.end.isoformat(),
            "dropped_dates": int(lp.dropped_dates),
        },
        "metrics": metrics,
        "concentration": _jsonable_concentration(concentration_report(lp.weights_book)),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "block": BOOTSTRAP_BLOCK,
            "draws": BOOTSTRAP_DRAWS,
            "n_obs": int(lp.n_obs),
        },
        "advisories": advisories,
    }


def _format_metric(name: str, metric: dict[str, Any]) -> str:
    value = metric["value"]
    rendered = metric["note"] if value is None else f"{value:.4f}"
    return f"{name}: {rendered}"


def render_text(report: dict) -> str:
    lines = [
        disclosure_header(),
        *[f"  - {line}" for line in DIAGNOSTIC_DISCLOSURES],
        "",
        f"Basis: {report['basis']}",
        f"Scope: {report['scope']}",
        f"Window: {report['portfolio']['start']} to {report['portfolio']['end']} "
        f"({report['bootstrap']['n_obs']} observations, "
        f"{report['portfolio']['dropped_dates']} dropped dates)",
    ]
    cash = report["portfolio"]["cash_dollars"]
    if cash is not None:
        lines.append(f"Cash: ${cash:,.2f}")

    lines.extend([
        "",
        "Metrics:",
        _format_metric("  book_sharpe", report["metrics"]["book_sharpe"]),
        _format_metric("  sortino", report["metrics"]["sortino"]),
        _format_metric("  downside_deviation", report["metrics"]["downside_deviation"]),
        _format_metric("  max_drawdown", report["metrics"]["max_drawdown"]),
        _format_metric("  block_bootstrap_lcb", report["metrics"]["block_bootstrap_lcb"]),
        "",
        "Concentration:",
    ])
    for key in sorted(report["concentration"]):
        value = report["concentration"][key]
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"  {key}: {rendered}")

    lines.extend([
        "",
        "Bootstrap:",
        f"  seed: {report['bootstrap']['seed']}",
        f"  block: {report['bootstrap']['block']}",
        f"  draws: {report['bootstrap']['draws']}",
        f"  n_obs: {report['bootstrap']['n_obs']}",
    ])
    if report["advisories"]:
        lines.extend(["", "Advisories:"])
        lines.extend(f"  - {line}" for line in report["advisories"])
    return "\n".join(lines) + "\n"


def render_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"
