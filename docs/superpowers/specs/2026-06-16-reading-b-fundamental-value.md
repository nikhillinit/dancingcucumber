# Reading B — Fundamental Value with a Timely-Price Leg (spec stub)

> **Status:** PLANNING-ONLY stub (Lane B, Task 11). Do NOT build the fixture or signal until an
> operator greenlights the fundamentals data work. This is the recommended next investment after
> Lane B Reading A returned a **power-limited DEV_FAILED** (`research/CANDIDATE_RESULT.md`): the
> price-only intermediate-reversal value leg was too thinly fit on the 9-yr price fixture
> (median ~7 positive train points/fold vs a 25 floor) to be a genuine signal verdict.

## Why Reading B (the actual lead)

Memory `deep-research-orthogonal-signals`: the cheapest *genuinely orthogonal* lead is to make the
existing `value_quality` family use **timely prices**. The live `value_quality` family uses book
value with ~90-day-lagged fundamentals; refreshing only the **price leg** (market cap from a timely
price) yields a **fundamental** value signal that is genuinely orthogonal to price-momentum — a
stronger candidate than Reading A's pure-price reversal, and NOT power-starved the way a price-only
reversal horizon is on this fixture. Reading A's negative does **not** refute classic 36–60mo
LT-reversal or fundamental value; both are fixture-infeasible on price-only data and live here.

## What Reading B requires (NOT built in Lane B)

1. **A fundamentals-bearing fixture** — point-in-time book value / shares outstanding per asset,
   aligned to `floor_prices.csv`'s assets + dates, **as-of bounded** with the existing
   `REPORTING_LAG_DAYS = 90` discipline (restated-not-as-reported; disclose, never fabricate;
   snapshot-forward). Source + point-in-time provenance must be recorded; a missing datum →
   `unavailable` and excluded, never a default value (DEV_BRAIN rail #5).
2. **An as-of assertion / test** that no fundamental value is used before `report_date + 90d`
   (leakage guard), mirroring the floor's holdout discipline.
3. **The value-with-timely-price construction** — e.g. `value = book_value_per_share / price`
   (or an equivalent fundamental-to-price ratio) using the **timely** price for the denominator
   and the lagged, as-of-bounded fundamentals for the numerator. Long-flat, sign carries direction,
   same `fit_percentile_transform` conviction transform as every other family.
4. **A `CandidatePreReg` extension + its own pre-registration** (new `candidate_hash`, fixture SHA,
   frozen horizons/lag), and — because Reading A did NOT touch the shared reserved tail — Reading B
   may pre-register against the SAME reserved tail OR a fresh one; either way every holdout touch is
   logged to `research/HOLDOUT_LEDGER.md` (Amendment F2).

## Reuse of the Lane B bench (signal-agnostic)

The bench is signal-agnostic: only `candidate_raw` (a new `value` construction) and the fixture
change. `candidate_pipeline` / `candidate_blend` / `candidate_floor` / `orthogonality` are reused
unchanged. The same kill-gate (post-transform orthogonality vs `long_momentum`/`mean_reversion`,
τ=0.40) and the same acceptance bar apply.

## Acceptance bar (unchanged — the floor's gates verbatim)

§7.2 ensemble beats best family (LCB > 0), §7.1 beats SPY by margin, DSR ≥ 0.95 at the candidate's
`declared_trials_N`. No new outcome thresholds. The frozen floor stays `DEV_FAILED` / `--enforce`
exit 1; promotion remains out of scope (Plans 1b/3, with operator sign-off + a fresh holdout).

## Open questions for the operator before greenlighting

- Fundamentals data source + licensing for point-in-time book value / shares (the heavy item).
- Whether to extend the fixture's date range (longer history also rescues classic LT-reversal,
  separately fixture-infeasible here).
- Effective independent trial count `N` once Reading A + Reading B both draw on the shared tail
  (multiple-testing accounting via `HOLDOUT_LEDGER.md`).
