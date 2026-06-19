# Reading B — Result Record (SEC EDGAR XBRL fundamentals)

Status: DEV RUN COMPLETE. The pre-registered `fundamental_value + momentum` candidate was run
through the dev gate with the reserved holdout BLINDED (`prereg_hash=None`). Outcome: the
candidate did **not** clear the dev gate. This is report-only research; the floor verdict stays
`DEV_FAILED`, the reserved holdout is UNTOUCHED, and nothing here authorizes sizing or capital
allocation. Authoritative floor truth: `apps/quant/advisor/backtest/FLOOR_RESULT.md`.

## What is complete
- Frozen fixture `apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv`
  (SHA-256 `fd2e574c43a78a096b459404f051f120ae03e02bf5ee73e8b20a174e710f1943`, 3527 rows).
- Deterministic adapter: `select_asof` + split-invariant `bp_timely` + `build_fundamental_panel`
  + `make_fundamental_raw` (`research/fundamental_value.py`).
- Bench wiring: `fundamental_candidate_metrics` (`research/candidate_floor.py`), holdout-blinded.
- Coverage: all 30 names >= 20 MarketCapAnchor quarters; no spurious step at split dates
  (AAPL 2020 4:1 ratio 1.09; GOOGL 2022 20:1 ratio 1.06).

## Dev-gate verdict
RUN with `fundamental_candidate_metrics(panel, panel_funda, DEFAULT_FUNDAMENTAL_CANDIDATE,
prereg_hash=None)` (holdout blinded). Runner: `ai-logs/hermes/run_ws4_dev.py`.

- `dev.passed = False` — the candidate does NOT clear the §7.2 dev gate.
- `dev.reasons`:
  - median fold delta not > 0
  - fewer than 70% positive folds
  - dev 90% bootstrap LCB of delta not > 0
  - total dev book-Sharpe lift < 0.05
- Per-fold deltas (ensemble − best-family, purged folds): `[-0.3120, -0.0311, +0.0451, -0.1290]`
  — 1 of 4 folds positive (25% < the 70% bar); median ≈ -0.080.
- Dev book-Sharpe: ensemble `0.557` < best_family `0.665` < SPY (dev window) `0.752`. The
  orthogonal `fundamental_value + momentum` blend (weights 0.5/0.5) fails to beat its best
  single component — the same structural failure the price-only floor showed (a correlated /
  non-additive blend does not exceed its best member), now confirmed with an orthogonal
  fundamentals input rather than reweighted price families.
- Report-only validation (deflation guard, never folds into the gate): DSR `0.677` < `0.95`
  bar; effective_N `1.10`; consistent with the dev failure.

## Power (is the DEV_FAILED signal-driven or a fixture artifact?)
`power.power_limited = False`. Per-fold median positive-train counts are `[325, 330, 330, 330]`
(floor = 1), and nonzero transformed-score TEST coverage is ~30–44% per fold. The percentile
fit is well-populated every fold, so this is a REAL, non-degenerate signal verdict — NOT a
power-limited / thin-fit artifact and NOT "fundamentals exhausted". The orthogonal book-to-price
signal was tested with adequate power and did not produce a dev-passing candidate on this universe.

## Holdout
UNTOUCHED. `HOLDOUT_LEDGER.md` is empty and stays empty — the dev gate did not pass, and with
`prereg_hash=None` the reserved tail was never read on any path (verdict reads `DEV_FAILED` by
construction whenever the holdout is blinded). The reserved tail would unlock ONLY if the dev
gate passed AND the caller supplied `prereg_hash == fundamental_candidate_run_hash` =
`408eebf70bf4bdefeb01e8975c93274fc641a0b1f384a002fab817b7052849e9` (config + fixture bytes);
neither condition was met. A dev pass is necessary but not sufficient; unlocking the shared
tail is a separate operator decision that burns the tail for promotion.

## Rails
`DEV_FAILED` stays; `backtest/` untouched; `node tools/run-floor.mjs --enforce` exits 1.
This is report-only research and does not authorize sizing or capital allocation. A candidate
dev failure is news about the candidate, not a floor change.
