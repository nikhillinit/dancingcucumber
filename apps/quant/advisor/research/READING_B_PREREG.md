# Reading B — Candidate Preregistration (SEC EDGAR XBRL fundamentals)

Status: PREREGISTRATION pin, frozen BEFORE any Reading-B bench run. Research bench only —
report-only, the floor verdict stays `DEV_FAILED`, the reserved holdout is blinded, and
nothing here authorizes sizing or capital allocation. Authoritative floor truth remains
`apps/quant/advisor/backtest/FLOOR_RESULT.md`.

## Why Reading B
The price-only floor is `DEV_FAILED` and cannot be cleared by reweighting correlated price
families (Lane B / Reading A established this). Reading B introduces an ORTHOGONAL input —
as-reported SEC EDGAR XBRL fundamentals (point-in-time) — to test whether a
`fundamental_value + momentum` candidate clears the dev gate. WS3B established that
`SEC_EDGAR_XBRL` qualifies as a point-in-time source.

## Frozen fixture
- Path: `apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv`
- SHA-256: `fd2e574c43a78a096b459404f051f120ae03e02bf5ee73e8b20a174e710f1943`
- Rows: 3527 (StockholdersEquity + CommonStockSharesOutstanding + MarketCapAnchor per filing)
- Universe: the 30 floor names; SPY is the benchmark only. Window: 2015–2023.
- Built by `apps/quant/advisor/data/edgar_xbrl_fetch.py` (operator/Bash-run; network; NOT in
  the pytest gate). Parse logic is covered offline by `tests/test_edgar_xbrl_fetch.py`.

## Frozen methodology (the falsifiable choices)
- Signal `fundamental_value` = split-invariant timely book-to-price:
  `bp_timely(t) = (equity_asof / mktcap_anchor) x price_adj(t0) / price_adj(t)`,
  `t0 = available_asof` (re-anchored each filing by `select_asof`).
- `mktcap_anchor = shares_asof x split_factor(period_end->now) x yf_close(t0)` — real dollars,
  so the signal carries no spurious step at split dates (verified on AAPL 2020 4:1 and
  GOOGL 2022 20:1; consecutive-day ratios 1.09 and 1.06 across the splits).
- As-of availability: `available_asof = max(period_end + 90d, filing_date, accepted_datetime)`;
  `snapshot_date` omitted (filing-backed source). Amendments are separate records used only
  as-of their own later availability (no backfill). Missing data -> excluded, never filled.
- Candidate families: `fundamental_value + momentum` (`value_metric = book_to_price`). Floor
  params inherited verbatim (warmup 200, folds 5, embargo 5, margin 0.0, etc.).

## Pinned hashes
- `fundamental_candidate_hash`            : `4a481623a9776c111c65ca3ab176722d3a6e7305da045af2c78b2ba4529f6613`
- `fundamental_candidate_validation_hash` : `5f5254d3025a0ab3e4de3151825e77d1aee813cf69d802250b82a8811cb4ca8b`
- `fundamental_candidate_run_hash`        : `408eebf70bf4bdefeb01e8975c93274fc641a0b1f384a002fab817b7052849e9`
  (the holdout-unlock key — binds config JSON + fixture bytes; the ONLY string that may
  unlock the reserved tail, and only after a dev pass.)
- LIVE multiple-testing trial count `declared_trials_N = 45`; `dsr_pass = 0.95`;
  `declared_var_sr = 1e-4` (carried on `FundamentalCandidateValidationPreReg`, report-only).

## Coverage (fixture property; not a bench result)
All 30 names carry >= 20 MarketCapAnchor quarters across 2015–2023 (min DIS 20, max T 42).
This non-degeneracy floor exists so that any later `DEV_FAILED` reflects the signal, not a
zeroed/empty fixture. The bench dev run itself is Reading-B's NEXT step (not part of this
preregistration); when it runs it uses `prereg_hash=None` (holdout blinded) and records the
per-fold positive-train counts in `READING_B_RESULT.md`.

## Rails preserved
`DEV_FAILED` stays; `apps/quant/advisor/backtest/` is untouched; `node tools/run-floor.mjs
--enforce` exits 1; `HOLDOUT_LEDGER.md` is empty (holdout never read). This bench is
report-only research; it does not authorize sizing, capital allocation, or any go-to-market
step. Data caveats: as-reported EDGAR is point-in-time but the fixture universe requires
full-window survival (residual survivorship bias); the ~90-day availability lag is a
conservative proxy.
