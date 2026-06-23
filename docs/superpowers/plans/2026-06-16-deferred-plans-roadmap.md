# Deferred Plans Roadmap (post validation gate)

Plan 1 (this slice): validation gate, REPORT-ONLY. DONE when Task 9 commits. The gate
computes DSR/MinBTL/t-stat diagnostics but does NOT influence `verdict` or the
`--enforce` exit code (proven by test_floor_metrics_validation_is_additive_only).

Plan 1b — Wire validation into the release gate (write after Plan 1 + first real
candidate). Make `node tools/run-floor.mjs --enforce` require BOTH verdict==PASSED AND
validation["passes"], so a deflation-failing candidate cannot be released. This is the
deliberate step that turns the report-only guard into a blocking one; deferred so the
current accepted DEV_FAILED floor is not disturbed.

Plan 2 — Workstream C completion. DONE (unit/fake path, report-only). The five-family
run_pipeline is wired into the CLI via `--families all` with FRED + composite/Alpha
Vantage adapters, a deterministic lexicon news scorer, and async family-coro factories;
every adapter is as-of bounded and degrades to FamilySignal.neutral on missing
input/error (no fabrication, spec section 10). Floor UNTOUCHED: advisor-gate exit 0,
`run-floor --enforce` exit 1. DEFERRED: the live multi-source smoke (real FRED_API_KEY +
ALPHAVANTAGE_API_KEY) and adding Finnhub/NewsAPI to the composite - both are operator
steps, not in the pytest gate. See docs/superpowers/plans/2026-06-16-workstream-c.md.

Plan 3 — Post-C signal program (write ONLY after C lands AND a candidate exists):
registry, append-only evidence ledger (separate from checkpoint upsert), report-only
candidate families (filing/accounting events; macro/credit expansion), orthogonality
diagnostics, dormant eligibility report, and a later separate skill_weight activation
plan. Do NOT pre-commit forward thresholds (windows, observation counts, Brier lifts)
until that plan is written against real constraints. SESTM/news sentiment stays
research-only/conditional for the large-cap book.

> **UPDATE 2026-06-23:** SESTM now has a dedicated plan —
> `docs/superpowers/plans/2026-06-23-sestm-news-lane-plan.md`. Reframed from a "large-cap book"
> add-on to a **small/mid-cap** lane (that is where its published alpha lives), gated on a Phase-0
> free-news-corpus capability matrix + a Phase-1 corpus-swap survival probe before any build, with
> default STOP at each phase. It is the remaining genuinely-open orthogonal lane after the
> residual-alpha / market-neutral price lane closed (`notes/2026-06-23-v2-STOP-closeout.md`).

Status note (post-C): Plan 1b (wire validation["passes"] into --enforce) and Plan 3
(post-C signal program) are now unblocked-by-prerequisite (C provides the live
five-family path) but remain gated on a real candidate that clears dev. Do not start
either until such a candidate exists; the accepted DEV_FAILED floor stays undisturbed.

## Orthogonal-input readings (WS3C/WS3D) — status

- **Reading B — fundamentals (book-to-price), DONE 2026-06-19.** `fundamental_value + momentum`
  dev gate, holdout BLINDED → genuine `DEV_FAILED`, NOT power-limited (1/4 positive folds; ens
  0.557 < momentum 0.665 < SPY 0.752). See `research/READING_B_{PREREG,RESULT}.md`.
- **Reading C — filing-text "Lazy Prices", DONE 2026-06-21 (WS3D).** `lazy_prices + momentum` dev
  gate, holdout BLINDED → genuine, faithfully-tested `DEV_FAILED` (mode 1). 0/4 positive folds;
  ens 0.598 < momentum 0.681 < SPY 0.752. Non-degeneracy triplet: coverage 0.975, cross-sectional
  dispersion median 0.373 (transform-collapse ruled out), orthogonality corr -0.018. Fixture SHA
  `372a8518…`. See `research/READING_C_{PREREG,RESULT}.md`. NOTE: the full-document TF-cosine +
  per-asset-percentile construction is in-scope; the canonical cross-sectional-level + section-text
  Lazy Prices anomaly remains UNTESTED (separate operator prereg decision).

**Load-bearing cross-reading insight:** two INDEPENDENT orthogonal inputs (B fundamentals, C
filing-text) now fail the dev gate the SAME way — the orthogonal signal dilutes momentum and the
0.5/0.5 ensemble underperforms the best single family. Reading C is confirmed orthogonal AND
cross-sectionally discriminating, so this is not "weak/correlated signal." It points at **blend
non-additivity on this 30-name universe**, not a per-signal defect. The next program should test the
BLENDING / weighting scheme and the UNIVERSE — not merely "add another orthogonal signal."

## WS3E — Days-to-Cover (short-interest) reading — SEQUENCED FOLLOW-ON (not started)
The next orthogonal-input reading: SEC/exchange short-interest "days to cover" as a separate
data subsystem (own plan, mirrors the WS3C/WS3D fixture+adapter+separate-prereg pattern). Given the
B/C blend-non-additivity finding, WS3E should be paired with a blend/weighting experiment rather than
run as another standalone equal-weight `dtc + momentum` candidate. A `lazy_prices + fundamental_value`
two-orthogonal-family candidate (both fixtures now exist) is also available as a WS4 test of whether
TWO orthogonal inputs together clear dev where each-plus-momentum did not.
