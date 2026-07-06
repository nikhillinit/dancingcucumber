# External repo review — virattt/ai-hedge-fund v2026.7.3

**Date:** 2026-07-06 · **Method:** 4 parallel subagent scouts (external signals/data,
external backtester/risk, external infra + release notes, local seam map) →
roleplay/debate/synthesis (Research-Integrity Guardian vs Capability Advocate vs
Pragmatic Solo-Dev Engineer).
**Subject:** https://github.com/virattt/ai-hedge-fund at tag `v2026.7.3` (commit 3a18702).
**Question:** what to adopt directly, adopt with modification, or reject.

## Headline

The external repo is an LLM-agent trading demo. Its v2026.7.3 flagship change is a
**post-hoc lookahead-leak fix** ("query metrics by filing_date, fail-loud client") — the
bug class this repo prevents by construction (frozen prereg, `available_asof`,
operator-locked holdout). Its backtester has **zero transaction costs, zero slippage,
prior-day-close fills** — its performance numbers are structurally inflated. Read it as a
cautionary control group, not a benchmark.

## Verdicts

| # | Item | Verdict | Modification / gate |
|---|------|---------|---------------------|
| 1 | Filing-date PIT conformance test (their leak-fix class) | **Adopt directly** | Test-only tripwire: `REPORTING_LAG_DAYS >= 60` (SEC large-accelerated-filer deadlines: 10-K 60d, 10-Q 40d) + exact-boundary-day tests on `is_available_asof` and EDGAR `select_asof`. This slice. |
| 2 | Insider-trades candidate family | **Adapt — PROPOSED NEW LANE** | EDGAR Form 4 (keyless, native filing timestamps; reuses `source_integrity/` EDGAR plumbing), NOT their paid financialdatasets.ai API. Must exclude 10b5-1 plan sales at ingestion; full candidate prereg frozen before any run. **Operator: rank against the standing 5>3>4 queue at next program review.** Known risk: insider-signal literature is strongest in small caps; our universe is 30 mega-caps — step one of the lane is a cheap effect-size sanity check, freeze the kill rule before running. |
| 3 | L/S portfolio state machine + integration-test structure (their `src/backtesting/portfolio.py`, `tests/backtesting/integration/`) | **Adapt** | Weighted-avg cost basis, proportional margin release, realized-gain separation, long-only/long-short/short-only test triad — port pattern is sound. Must ADD what they lack: borrow costs, transaction costs, maintenance margin. Build INSIDE the L/S prereg surface. **Gated on Decision 5** (do not build speculatively). |
| 4 | Exposure metrics (gross/net/L-S ratio) in diagnostics | **Adapt (trivial)** | Already captured verbatim in TODOS.md "P3 — Long/short book diagnostics" (Decision-5 gated). No new work item. |
| 5 | Event-study engine (their v2, CAR around filings) | **Adapt, parked** | Only as a cheap pre-floor kill screen if an event-driven candidate lane opens. |
| 6 | LLM personas / LLM portfolio manager / LangGraph | **Reject** | Documented negative: LLM trading agents fail post-cutoff (leakage); non-preregisterable, non-reproducible. |
| 7 | financialdatasets.ai paid API | **Reject** | Violates keyless constraint; EDGAR/FRED/yfinance cover the ground. |
| 8 | Web app (React flow builder), Docker/Ollama, multi-provider config | **Reject** | Solo-dev maintenance tax; no research value. |
| 9 | Hardcoded fundamental/valuation/technical threshold recipes (ROE>15%, P/E<25, EMA 8/21/55, Hurst/ADX ensembles) | **Reject** | Uncalibrated folklore; more correlated price families is exactly what the blend-non-additivity record and the deep-research warning rule out. Vol-regime survives only as a diagnostic idea, never a signal. |
| 10 | Their metrics calculator / vol+corr risk manager | **Reject** | Duplicates `diagnostics/report.py` and `risk/limits.py`. |

## Debate cruxes (what would change these verdicts)

- **Insider lane goes ahead** iff Form 4 availability is provably filing-timestamped and
  10b5-1 sales are excluded at ingestion; **dies** if the mega-cap effect-size check shows
  near-zero effect for large-cap insider buys too.
- **Fail-loud vs fail-neutral:** their fix moved to fail-loud data clients; our
  missing→neutral is a live-path trust-boundary feature. Research loaders already carry
  coverage guards (e.g. lazy_prices triplet coverage). Residual gap = the conformance
  test in item 1, not a convention change.
- **State-machine port timing:** L/S mechanics (margin, borrow) belong inside the prereg
  surface, so the port waits for Decision 5's prereg design — porting first puts the cart
  before the horse.

## Executed from this review

- Item 1: `apps/quant/advisor/tests/test_pit_conformance.py` (this branch, via Hermes).
- Items 2/5: operator decision points recorded here; nothing built.
- Items 3/4: gated on Decision 5; nothing built.
