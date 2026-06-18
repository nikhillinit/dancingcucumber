# Reading B — Fundamental Value with a Timely-Price Leg (planning spec)

> **Status:** PLANNING-ONLY source-agnostic data contract (Lane B, Task 11). Do NOT build the
> fixture or signal until an operator greenlights the fundamentals data work. This is the
> recommended next investment after
> Lane B Reading A returned a **power-limited DEV_FAILED** (`research/CANDIDATE_RESULT.md`): the
> price-only intermediate-reversal value leg was too thinly fit on the 9-yr price fixture
> (median ~7 positive train points/fold vs a 25 floor) to be a genuine signal verdict.

Summary: Adds the source-agnostic fundamentals record contract, strict availability rule,
denominator bridge, preregistration hash/trial-count pins, and unchanged blocked-floor rails.

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
4. **A `CandidatePreReg` extension + `CandidateValidationPreReg` pre-registration** (new
   `candidate_hash`, fixture SHA, candidate validation hash, frozen horizons/lag), and — because
   Reading A did NOT touch the shared reserved tail — Reading B may pre-register against the SAME
   reserved tail OR a fresh one; either way every holdout touch is logged to
   `research/HOLDOUT_LEDGER.md` (Amendment F2).

## Data contract (source-agnostic)

Reading B does not choose a fundamentals source. Before any source-specific fixture work, the
fundamentals-bearing fixture must expose one immutable record per `(asset, concept, as-of)` with
the following schema:

- `asset` — fixture asset id, aligned to `floor_prices.csv` assets/dates.
- `cik` — issuer identifier, or the source-equivalent entity id when CIK is unavailable.
- `accession` — exact filing/record id the value came from.
- `form` — filing/form type, such as 10-Q, 10-K, or amendment.
- `report_period_end` — fiscal period the value describes.
- `filing_date` — date the record became publicly retrievable.
- `accepted_datetime` — precise acceptance timestamp, or `snapshot_date` if the source is a dated snapshot.
- `concept` — fundamental being recorded, such as book value or common shares outstanding.
- `unit` — unit of measure.
- `value` — numeric datum AS ORIGINALLY REPORTED in that record.
- `available_asof` — earliest date the value may be used under the availability rule below.
- `superseded_by` — accession of a later record that restates this one, if any.
- `amended_flag` — whether this record is an amendment/restatement.
- `missingness_reason` — why a datum is absent; kept `unavailable`, never filled.
- `denominator_policy` — how the price/share denominator is adjusted under the denominator bridge below.

Strict availability rule:

```
available_asof = max(report_period_end + REPORTING_LAG_DAYS, filing_date, accepted_datetime, snapshot_date)
```

`REPORTING_LAG_DAYS = 90` remains the existing discipline. For filing-backed records,
`snapshot_date` is the source-equivalent dated snapshot input only when such a snapshot exists;
otherwise the accepted/filing availability inputs must still be explicit and auditable.

Named availability rules:

- **strict-lag≠PIT** — A strict lag applied to restated/as-reported-latest data is NOT
  point-in-time evidence. PIT requires the value be reconstructable from the exact record
  (accession/form/accepted) knowable before `as_of`.
- **missing→excluded** — Missing fundamentals are `unavailable` and excluded; never zero-,
  median-, current-, nor future-restatement-filled (DEV_BRAIN rail #5).
- **amendments-are-separate** — Amendments/restatements are separate records with their own
  `accepted_datetime`; an earlier `as_of` uses the original record, never a later restatement
  backfilled to an earlier date.

### Denominator bridge

The value-with-timely-price construction divides an as-of-bounded fundamental numerator, such as
book value per share, by a TIMELY price denominator. The numerator share count and the price
denominator must use a consistent split/adjustment basis: share counts and prices are reconciled to
the same adjustment basis so the ratio is not corrupted by un-mirrored splits or adjustments.

### Pre-run pins and trial surface

Before any Reading B run, freeze the fixture SHA, candidate hash, and candidate validation hash.
The LIVE multiple-testing trial count is `CandidateValidationPreReg.declared_trials_N`.
`CandidatePreReg.declared_trials_N` is VESTIGIAL and must NOT be used as the DSR trial surface.

### Rails preserved

`DEV_FAILED` stays; holdout remains untouched; `node tools/run-floor.mjs --enforce` exits 1;
promotion and production/live/sizing remain OUT of scope. WS3C (source-specific fixture/prereg)
stays BLOCKED until the separate WS3B feasibility record confirms a point-in-time source.

## Reuse of the Lane B bench (signal-agnostic)

The bench is signal-agnostic: only `candidate_raw` (a new `value` construction) and the fixture
change. `candidate_pipeline` / `candidate_blend` / `candidate_floor` / `orthogonality` are reused
unchanged. The same kill-gate (post-transform orthogonality vs `long_momentum`/`mean_reversion`,
τ=0.40) and the same acceptance bar apply.

## Acceptance bar (unchanged — the floor's gates verbatim)

§7.2 ensemble beats best family (LCB > 0), §7.1 beats SPY by margin, DSR ≥ 0.95 at
`CandidateValidationPreReg.declared_trials_N`. No new outcome thresholds. The frozen floor stays
`DEV_FAILED` / `--enforce` exit 1; promotion remains out of scope (Plans 1b/3, with operator
sign-off + a fresh holdout).

## Open questions for the operator before greenlighting

- Fundamentals data source + licensing for point-in-time book value / shares (the heavy item).
- Whether to extend the fixture's date range (longer history also rescues classic LT-reversal,
  separately fixture-infeasible here).
- Effective independent trial count `N` once Reading A + Reading B both draw on the shared tail
  (multiple-testing accounting via `HOLDOUT_LEDGER.md`).
