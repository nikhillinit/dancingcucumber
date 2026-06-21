# Reading C — Candidate Preregistration (SEC EDGAR filing-text "Lazy Prices")

Status: PREREGISTRATION pin, frozen with the committed fixture. Research bench only —
report-only, the floor verdict stays `DEV_FAILED`, the reserved holdout is blinded, and
nothing here authorizes sizing or capital allocation. Authoritative floor truth remains
`apps/quant/advisor/backtest/FLOOR_RESULT.md`.

## Why Reading C
The price-only floor is `DEV_FAILED` and cannot be cleared by reweighting correlated price
families (Lane B / Reading A). Reading B (book-to-price fundamentals) introduced one ORTHOGONAL
input and still failed the dev gate. Reading C introduces a SECOND, independent orthogonal input
— year-over-year 10-K/10-Q **filing-text similarity** ("Lazy Prices", Cohen–Malloy–Nguyen) — to
test whether a `lazy_prices + momentum` candidate clears the dev gate. WS3B established that
`SEC_EDGAR` qualifies as a point-in-time source; the text *is* the disclosure, so it carries no
+90d reporting lag.

## Frozen fixture
- Path: `apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv`
- SHA-256: `372a8518bd4ce40dad41c9fb03ff64502a515bf0d6e9107c63e8815af13e09f1`
- Rows: 1053 (one `FilingSimilarity` ratio per current 10-K/10-Q). 30 floor names; SPY is the
  benchmark only. Window: 2015–2023 (2014 filings pulled only as the prior-year baseline).
- Built by `apps/quant/advisor/data/filing_text_fetch.py` (operator/network-run; NOT in the
  pytest gate). Pure helpers (tokenize / cosine_tfidf / build_similarity_row) covered offline by
  `tests/test_lazy_prices.py`; the committed fixture has an acceptance test
  (`test_committed_fixture_loads_audits_and_is_non_degenerate`).

## Frozen methodology (the falsifiable choices)
- Signal `lazy_prices` = year-over-year cosine similarity of consecutive SAME-FORM filings
  (10-K vs prior-fiscal-year 10-K; 10-Q vs year-ago SAME-fiscal-quarter 10-Q — never the adjacent
  quarter). HIGH similarity = "non-changer" → predicted higher return = the LONG leg. RAW signal =
  the similarity itself (NOT `1 − similarity`; D3 sign guard). `similarity_metric = "cosine_tfidf"`
  (cosine over raw term-frequency vectors, no IDF; a 2-doc YoY pair has no corpus).
- **Text extraction (load-bearing for reproducibility — `filing_text_fetch._html_to_text`):** the
  PRIMARY filing document (not the full-submission `.txt`) is `html.unescape`-d, then `<script>` /
  `<style>` and inline-XBRL `<ix:header>` / `<ix:hidden>` blocks are dropped, then tags are
  stripped, then tokenized (lowercase `\w+`). The unescape collapses `&nbsp;` vs `&#160;` (a
  filing-agent encoding flip that otherwise tokenizes as different words and craters YoY cosine);
  the ix-block drop removes the 2019–2020 inline-XBRL context/metadata that would systematically
  depress the large-filer transition cohort. This removes ONLY non-prose; visible body text
  (`<ix:nonNumeric>` / `<ix:nonFraction>`) is untouched, and `tokenize` / `cosine_tfidf` /
  `build_similarity_row` stay byte-frozen.
- As-of availability: `available_asof = max(filing_date, accepted_datetime)`; **no +90d lag**;
  `snapshot_date` omitted (filing-backed). `select_asof` never backfills; missing → excluded.
- Candidate families: `lazy_prices + momentum`. Floor params inherited verbatim (warmup 200,
  folds 5, embargo 5, margin 0.0, pct_clip (0.05, 0.95), weight_grid (0.25, 0.50, 0.75),
  bootstrap block 21 / draws 2000 / seed 12345, dev_lcb 0.90, final_lcb 0.95).
- `orthogonality_tau = 0.40` (max |corr| lazy_prices vs momentum on dev, report-only).

## Pinned hashes
- `lazy_prices_candidate_hash`            : `934365a1a06f7a6584b7f3cc0e8859cdd621d519dfe138ee5b3426a96a44d063`
  (methodology-only id; NOT the holdout-unlock key.)
- `lazy_prices_candidate_validation_hash` : `5f5254d3025a0ab3e4de3151825e77d1aee813cf69d802250b82a8811cb4ca8b`
  (identical to Reading B's validation surface — the deflation prereg is reused verbatim; see below.)
- `lazy_prices_candidate_run_hash`        : `27c2850bae6e53580548b71b495fe87383cb3e8c39982d50812443cba4819388`
  (the holdout-unlock key — binds config JSON + fixture bytes; the ONLY string that may unlock
  the reserved tail, and only after a dev pass.)
- LIVE multiple-testing trial count `declared_trials_N = 45`; `dsr_pass = 0.95`;
  `declared_var_sr = 1e-4` (carried on `LazyPricesCandidateValidationPreReg`, report-only).

### declared_var_sr — reuse of 1e-4 (not recalibrated)
`declared_var_sr` is reused at `1e-4` rather than recalibrated, justified NOT by a measured
comparison (the per-trial Sharpe variance was not separately measured here) but by **invariance of
the report-only conclusion**: (a) it is the floor's already-calibrated value and the Reading-C
validation surface is byte-identical to Reading B's (hash `5f5254d3…` matches); (b) the validation
report is report-only and never folds into the dev gate or `passes`; (c) the measured DSR (0.708)
fails the 0.95 bar by a wide margin, and DSR is monotone decreasing in `var_sr`, so a larger true
variance would only deflate further — the qualitative result (DSR fails) is robust to the choice.

## Coverage (fixture property; not a bench result)
`coverage_in_window(2015-01-01 … 2023-12-31) = 0.975`. The only coverage shortfalls are
corporate-restructuring **CIK discontinuities**, not bugs (missing → neutral handles them):
DIS CIK `0001744489` = the new Walt Disney holding co (Mar 2019) → 16 rows from 2020Q1; GOOGL
`0001652044` = Alphabet (Oct 2015) → 30 rows from 2016Q3; GE one pairing gap (35). All other names
carry the full 36 rows (9 years × [1×10-K + 3×10-Q]).

## Rails preserved
`DEV_FAILED` stays; `apps/quant/advisor/backtest/` untouched; `node tools/run-floor.mjs --enforce`
exits 1; `HOLDOUT_LEDGER.md` empty (holdout never read). Reading C lives on its OWN separate hash
surface; the frozen `PreRegConfig` / `CandidatePreReg` / `FundamentalCandidatePreReg` are not
touched. Report-only research; does not authorize sizing or any go-to-market step. Data caveats:
the fixture universe requires full-window survival (residual survivorship bias); full-document TF
cosine is dominated by stable financial-statement boilerplate, so it is insensitive to narrative
("lazy") change relative to the section-level reading (see RESULT, "what was NOT tested").
