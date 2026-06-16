# Floor v2 — result

## Verdict: **UNSUPPORTED** (the floor is NOT cleared; advisor is not production-ready)

No pre-registered construction (C, D, or E) cleared the dev stability gate, so per the pre-registered decision rule the verdict is **UNSUPPORTED** and the **held-out tail was never evaluated** (no candidate earned a holdout run). This is the rail-honest outcome: a clean negative, reported as the lead finding.

**Recorded:** 2026-06-16. Pre-registration: `PREREG.md` (config hash `1ad2ed4a…`, candidate order C→E, margin 0.0, IMMUTABLE). Fixture: 30 large-caps + SPY, 2015-2023 (`UNIVERSE_RULE.md`, SHA-256 `d40b9959…`), universe = **formal** (≥20).

## Evidence (dev folds only — holdout blinded)
| Candidate | dev.passed | ensemble book-Sharpe | best standalone family | all 4 fold Δ |
|---|---|---|---|---|
| **C** (momentum, trend) | **False** | 0.732 | 0.828 | −0.21, −0.08, −0.10, −0.08 |
| **D** (+ mean_reversion) | **False** | 0.595 | 0.878 | −0.42, −0.39, −0.10, −0.53 |
| **E** (+ breakout) | **False** | 0.592 | 0.878 | −0.41, −0.41, −0.11, −0.49 |

Every fold delta is negative in every candidate. Rule B never deviated from equal weights (no ≥0.05 lift in ≥2 inner blocks). SPY book-Sharpe over the same window ≈ 0.76, so the ensemble (0.73) also fails §7.1 (beat SPY).

## Interpretation
- **§7.2 (beat the parts) fails decisively.** The continuous long-flat price-only ensemble does not beat its best standalone constituent family on any fold; the single best family (trend-type) dominates the blend.
- **Adding families makes it worse**, not better — diluting toward weaker price families drags the blend down while raising the best-of-parts bar. So "add 1–2 decorrelated families" does not rescue the 2-family failure here.
- **v2 calibration improved absolute level but not the floor.** v1 equal-weight integer-sign ensemble scored 0.32; v2 continuous long-flat scores 0.73 (≈ SPY's 0.76) — a real improvement — but the §7.2 "beat the parts" floor is structural: a fixed equal/near-equal blend of correlated long-only price families cannot beat its best member out-of-sample net of costs on this universe.
- **necessary-not-sufficient:** this is the price-only proxy only; it does not speak to the full 5-family advisor. But the proxy floor that gates production release is not cleared.

## Gate status
- `npm run advisor-gate` (report) → exit 0, prints `DEV_FAILED`/`UNSUPPORTED` + disclosures (per-commit, non-blocking).
- `node tools/run-floor.mjs --enforce` (release) → **exit 1** (verdict ≠ PASSED) — production release correctly blocked.
- The holdout (`--holdout`, hash `1ad2ed4a…`) remains untouched and is reserved; do NOT run it without a dev-passing candidate.

## What would change the verdict (future work, out of scope here)
Not reachable by reweighting these correlated long-only price families. Would require genuinely decorrelated/orthogonalized signals, a different aggregation than a fixed long-flat blend, or non-price families — all deferred to a post-Workstream-C plan and the live `ensemble_vote` seam (NOT this floor).
