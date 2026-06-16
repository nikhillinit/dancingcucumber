# Floor fixture — universe selection rule (pre-registered)

**Recorded:** 2026-06-16, BEFORE pulling prices. Frozen thereafter — never re-selected after a floor number is seen.

## Window
`2015-01-01 .. 2023-12-31` (daily). Extends the prior 2018–2023 fixture earlier to span more regimes (2015–16 energy/China selloff, 2018 Q4 drawdown, 2020 COVID crash + recovery, 2022 bear).

## Selection rule (as-of-window-START, not as-of-today)
The universe is the fixed candidate list below — large-cap US names that were among the **most liquid (mega-cap, high dollar-volume) as of the window start, 2015-01-01** — filtered to those with **full continuous daily coverage across the whole window**, plus `SPY` as the benchmark. Selection uses **2015 liquidity only** (no 2023 hindsight on returns); the survival/coverage filter is purely mechanical (no return- or Sharpe-based pruning).

**Candidate list (30, pre-registered):**
AAPL, MSFT, XOM, JNJ, JPM, WFC, GE, PG, KO, PFE, T, VZ, CVX, MRK, INTC, CSCO, WMT, HD, BAC, ORCL, DIS, MCD, IBM, QCOM, C, GILD, AMGN, UNH, GOOGL, BA — plus benchmark **SPY**.

A formal floor claim requires ≥20 names surviving the coverage filter (spec: median N_active ≥ 20, min ≥ 12). If fewer than 20 survive, the run is labelled "micro-universe diagnostic only."

## Data & corporate actions
- **Source:** yfinance 0.2.66, pulled 2026-06-16.
- **Price:** adjusted close (`auto_adjust=True` — splits & dividends adjusted).
- **Missing-data policy:** restrict to dates where SPY is present; forward-fill gaps ≤ 1 trading day; any ticker still missing data over the window is dropped (coverage filter).

## Survivorship disclosure
Requiring full-window survival excludes names that delisted/merged after 2015 (residual survivorship bias). **Long-side results are upward-biased.** Selection is as-of-2015-liquidity, so there is no return look-ahead, but the survival requirement is a known, disclosed bias.
