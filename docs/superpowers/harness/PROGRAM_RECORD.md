# Program Record — append-only verdict table

One row per lane closeout, appended in the closeout commit. NEVER edit, reorder, or
delete a row — every row carries a chain hash over all prior rows, and the footer
pins the chain tip + row count; both are recomputed by `test_prereg_conformance.py`.
Any edit/reorder/interior-deletion fails the chain; tail deletion fails the tip
footer. (A deliberate rewrite of rows AND footer together defeats this — the guard
targets sloppy edits, the documented failure mode; deliberate tampering is a git-
history matter.) To append: write the row with chain cell `TBD`, run pytest, paste
the printed expected hash, update the tip footer the same way. Exactly ONE table
lives in this file — the test enforces that.
Seeded 2026-07-05 from `notes/2026-07-04-program-review-memo.md` §1; historic rows
point at the memo (not all eight have a dedicated closeout — stating that beats
faking it).

| # | Lane | Verdict | Key number | Record pointer | Chain |
|---|------|---------|-----------|----------------|-------|
| 1 | Advisor v1 equal-weight price ensemble (2026-06-14) | FAILS FLOOR | Sharpe 0.32 < SPY 0.85 | notes/2026-07-04-program-review-memo.md | 3152c74976c1 |
| 2 | Plan 4 v2 continuous long-flat ensemble (2026-06-16) | DEV_FAILED | ens 0.73 < best family 0.83 | notes/2026-07-04-program-review-memo.md | 59ca7bfbf7f0 |
| 3 | Lane B value+momentum candidate (2026-06-17) | DEV_FAILED (power-limited) | ens 0.662 < best 0.668 | notes/2026-07-04-program-review-memo.md | 7dd165c7db26 |
| 4 | WS4 Reading B fundamental_value+momentum (2026-06-19) | DEV_FAILED (not power-limited) | ens 0.557 < momentum 0.665 < SPY 0.752 | notes/2026-07-04-program-review-memo.md | 2582ca44b75a |
| 5 | WS3D Reading C lazy_prices (2026-06-21) | DEV_FAILED | 0/4 folds; ens 0.598 < momentum 0.681 | notes/2026-07-04-program-review-memo.md | f0fc855475a4 |
| 6 | Universe-change broad screen T0.2b (2026-06-23) | INCONCLUSIVE | GREEN non-discriminating; survivorship-confounded | notes/2026-07-04-program-review-memo.md | f8da74a0c549 |
| 7 | QC+EDGAR source-integrity diagnostic v2 (2026-06-23) | STOP | mappability 68.9% < 85% frozen gate | notes/2026-07-04-program-review-memo.md | cf0c6917e7b5 |
| 8 | SESTM Phase-0 corpus matrix (2026-07-04) | STOP | no keyless corpus clears the 7 frozen criteria | notes/2026-07-04-program-review-memo.md | 179a032bc60f |
| 9 | L/S reversal Gate-1 kill screen (2026-07-06) | CLOSED | post-cost reversed IRs 0.1787 / 0.1922 / 0.2389; need 2/3 >= 0.20 | notes/2026-07-06-ls-reversal-gate1-result.md | 591a2e3935bd |

Chain tip: 591a2e3935bd over 9 rows
