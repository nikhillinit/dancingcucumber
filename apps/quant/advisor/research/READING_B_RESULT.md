# Reading B — Result Record (SEC EDGAR XBRL fundamentals)

Status: PRE-RUN READINESS. The fixture, adapter, and preregistration pin are complete and
frozen (see `READING_B_PREREG.md`). The candidate dev-gate bench run is Reading-B's NEXT
step and has NOT been executed in this record. Report-only research; the floor verdict stays
`DEV_FAILED` and the reserved holdout is blinded. Authoritative floor truth:
`apps/quant/advisor/backtest/FLOOR_RESULT.md`.

## What is complete
- Frozen fixture `apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv`
  (SHA-256 `fd2e574c43a78a096b459404f051f120ae03e02bf5ee73e8b20a174e710f1943`, 3527 rows).
- Deterministic adapter: `select_asof` + split-invariant `bp_timely` + `build_fundamental_panel`
  + `make_fundamental_raw` (`research/fundamental_value.py`).
- Bench wiring: `fundamental_candidate_metrics` (`research/candidate_floor.py`), holdout-blinded.
- Coverage: all 30 names >= 20 MarketCapAnchor quarters; no spurious step at split dates
  (AAPL 2020 4:1 ratio 1.09; GOOGL 2022 20:1 ratio 1.06).

## Dev-gate verdict
NOT YET RUN. The bench dev run is the next Reading-B step. When executed it must call
`fundamental_candidate_metrics(panel, panel_funda, DEFAULT_FUNDAMENTAL_CANDIDATE,
prereg_hash=None)` — `prereg_hash=None` keeps the reserved tail blinded — and this record is
then updated with: `dev.passed`, the per-fold deltas, and the per-fold positive-train counts
(so a `DEV_FAILED` is provably signal-driven, not a fixture artifact).

## Holdout
UNTOUCHED. `HOLDOUT_LEDGER.md` is empty. The reserved tail unlocks ONLY if the dev gate
passes AND the caller supplies `prereg_hash == fundamental_candidate_run_hash` =
`408eebf70bf4bdefeb01e8975c93274fc641a0b1f384a002fab817b7052849e9` (config + fixture bytes).
A dev pass is necessary but not sufficient; unlocking the shared tail is a separate operator
decision and burns the tail for promotion.

## Rails
`DEV_FAILED` stays; `backtest/` untouched; `node tools/run-floor.mjs --enforce` exits 1.
This is report-only research and does not authorize sizing or capital allocation.
