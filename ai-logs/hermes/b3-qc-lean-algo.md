# Hermes task B3 — write the QC LEAN/QuantBook diagnostic algorithm (CODE ONLY, no run)

## Hard constraints
- Write CODE ONLY. Do NOT run `npm` or `node`. Do NOT commit. You CANNOT run this on QuantConnect
  (no network in this sandbox) — your job is to produce a correct, heavily-commented LEAN **Python**
  algorithm the operator will paste into QuantConnect project 33255206 and run there.
- Write exactly ONE file: `ai-logs/hermes/runs/qc_b3_lean_diagnostic.py`. Touch nothing else.
- **NON-ALPHA:** it must NOT compute or output returns, Sharpe, PnL, or portfolio performance — only
  universe membership, momentum/value *ranks*, and delisting bookkeeping. Any performance readout
  violates the frozen prereg.

## What it implements
Read the frozen prereg `docs/superpowers/plans/2026-06-23-qc-edgar-diagnostic-v2-prereg.md` §3–§4.
Goal: for 2015-01-01..2023-12-31, emit a CSV of delisted small/mid-cap US common stocks with their
**pre-delisting cross-sectional ranks**, for later joining to EDGAR delisting reasons. NO trading.

**Eligible universe (point-in-time, monthly):**
- US common shares: `FineFundamental.SecurityReference.IsPrimaryShare == True`,
  `IsDepositaryReceipt == False`, common-stock `SecurityType` (`"ST00000001"`).
- 21-day **median** dollar-volume ≥ $1,000,000 (compute the rolling median from history — Coarse
  exposes only *daily* `DollarVolume`).
- market-cap percentile band **20–90** (rank `MarketCap` cross-sectionally each selection; keep pctile 20..90).

**Per eligible name each selection, record:** symbol, company name, date, `MarketCap`, trailing
**12-1 month price momentum** (return from t-12mo to t-1mo, used ONLY as a ranking label — a price
ratio for bucketing, NOT a strategy return), `BookValuePerShare` (reported value axis; flag negative book).

**Delistings:**
- Capture delisting **dates** for names eligible within the prior 12 months. **PREFER map-file
  delisting dates** over live `Delisting` events (live events undercount — Morningstar drops delisted
  names from fundamentals before the event fires). If map-file access isn't available from the algo,
  fall back to capturing `Delisting(Type=Delisted)` in `on_data` AND keep each delisting-candidate
  security subscribed through delisting so the event fires.
- For each delisted name, assign its momentum **decile** + value bucket as-of the **last date it was a
  valid eligible member** (guaranteed within the 12-month lookback). Momentum is price-only → computable
  even after fundamentals are dropped.

**Output CSV `qc_eligible_delistings.csv`** (write via QC ObjectStore, or Log the rows for export):
columns `symbol, company_name, last_eligible_date, delist_date, momentum_decile (1=worst..10=best),
value_bucket, negative_book_flag, market_cap`. **NO returns/PnL columns.**

## QC API facts to use (from prior cited research — mark any call you are UNSURE of with `# UNVERIFIED`)
- Research env: `qb.universe_history(qb.fundamental, start, end, flatten=True)` → historical universe
  snapshots (Fundamental objects per date). Good for the universe + ranking half.
- Coarse: `CoarseFundamental` has `DollarVolume`, `Price`, `HasFundamentalData`. Fine: `FineFundamental`
  has `MarketCap`, `SecurityReference.{IsPrimaryShare,IsDepositaryReceipt,SecurityType}`,
  `ValuationRatios.PBRatio` (NULLS when book ≤ 0 → use `ValuationRatios.BookValuePerShare` for the sign
  / negative-book detection).
- `Delisting` carries only `Type ∈ {Warning, Delisted}` + `Ticket` — NO reason field (reason comes
  from EDGAR, not here).
- Free tier: B-MICRO node, daily resolution, **chunk by year** to fit memory/time; 200 backtests/day.

## Structure (provide both halves)
1. A **QuantBook research-notebook** section: universe construction + momentum/value ranking +
   bucketing + storing per-(symbol,date) membership.
2. A thin **QCAlgorithm backtest** (or map-file enumeration) that captures delisting dates and joins
   them to the stored membership to emit the final CSV.
Comment heavily; mark `# UNVERIFIED` calls; keep it NON-ALPHA; runnable-as-written on QC free tier with
minimal operator edits (note any edits needed, e.g., ObjectStore keys, chunk boundaries).
