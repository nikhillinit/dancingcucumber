# TODOS

Deferred work with context. Added by /plan-ceo-review 2026-07-06 (skill_weight seam plan).

## P2 — Surface skill weights in decision output (S)

- **What:** When a bundle carries non-uniform `skill_weight`s, surface them in human-facing output — append weights to `Allocation.reasoning` (`apps/quant/advisor/portfolio/allocator.py:46`) and/or print `vote_parity()` in the CLI `--families all` report.
- **Why:** Post seam-activation, a non-default weight shifts live votes with no human-visible trace (weights persist to the checkpoint DB only). This is the designated silent-failure surface once weights ever go non-uniform.
- **Pros:** Closes the last silent path around the weighted vote.
- **Cons:** Dead code until a validated calibration exists; the run_pipeline tripwire (same PR as seam activation) blocks non-uniform weights meanwhile.
- **Context:** The seam-activation PR adds a fail-fast in `pipeline/run.py` rejecting non-uniform weights until a validated calibration artifact exists (spec §8: shrinkage, rank-IC/IR, caps, min OOS window, Brier gating — `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md:119`). **When that tripwire is relaxed, this observability MUST land in the same change.**
- **Depends on / blocked by:** Nothing to build; activation blocked by a validated calibration source (none exists as of 2026-07).

## P3 — Upper-bound `FamilySignal.skill_weight` (S)

- **What:** Add a finite upper bound to `skill_weight` in `apps/quant/advisor/schemas.py:21` (e.g. `le=100` or `allow_inf_nan=False`) plus a rejection test.
- **Why:** It is the only schema field admitting `float('inf')` (`ge=0` passes inf, rejects NaN/negative). An inf weight yields `inf/inf → nan → min(100, nan) → 100.0`: a silently confident garbage vote.
- **Pros:** Kills the one unmitigated input pathology at the layer where it belongs.
- **Cons:** Schema change — deliberately excluded from the seam-activation PR's "no schema changes" scope promise.
- **Context:** Interim mitigation exists: the run_pipeline tripwire rejects non-uniform weights, so inf cannot reach a live vote except as uniform-inf (absurd input). Flagged independently by both Claude review and Codex outside voice, 2026-07-06.
- **Depends on / blocked by:** Land after the seam-activation PR merges.
