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

Status note (post-C): Plan 1b (wire validation["passes"] into --enforce) and Plan 3
(post-C signal program) are now unblocked-by-prerequisite (C provides the live
five-family path) but remain gated on a real candidate that clears dev. Do not start
either until such a candidate exists; the accepted DEV_FAILED floor stays undisturbed.
