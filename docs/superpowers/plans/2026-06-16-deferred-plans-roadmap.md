# Deferred Plans Roadmap (post validation gate)

Plan 1 (this slice): validation gate, REPORT-ONLY. DONE when Task 9 commits. The gate
computes DSR/MinBTL/t-stat diagnostics but does NOT influence `verdict` or the
`--enforce` exit code (proven by test_floor_metrics_validation_is_additive_only).

Plan 1b — Wire validation into the release gate (write after Plan 1 + first real
candidate). Make `node tools/run-floor.mjs --enforce` require BOTH verdict==PASSED AND
validation["passes"], so a deflation-failing candidate cannot be released. This is the
deliberate step that turns the report-only guard into a blocking one; deferred so the
current accepted DEV_FAILED floor is not disturbed.

Plan 2 — Workstream C completion. Wire the full five-family run_pipeline into the
CLI (value/quality, momentum, trend, macro, sentiment) with provider adapters and
fake-provider tests. Prerequisite for ALL candidate-family work. See
docs/superpowers/plans/2026-06-15-followups-handoff.md.

Plan 3 — Post-C signal program (write ONLY after C lands AND a candidate exists):
registry, append-only evidence ledger (separate from checkpoint upsert), report-only
candidate families (filing/accounting events; macro/credit expansion), orthogonality
diagnostics, dormant eligibility report, and a later separate skill_weight activation
plan. Do NOT pre-commit forward thresholds (windows, observation counts, Brier lifts)
until that plan is written against real constraints. SESTM/news sentiment stays
research-only/conditional for the large-cap book.
