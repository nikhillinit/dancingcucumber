# External repo review — dog23/park

**Date:** 2026-07-22 · **Method:** Direct read via GitHub web pages (repo not addable to
this session — cross-owner add blocked in v1; no clone, no code copied).
**Subject:** https://github.com/dog23/park (`AI Deep Learning Automation/` subtree) — a
solo, live NinjaTrader 8 futures system: PyTorch entry/exit/trend models behind FastAPI,
C#/NinjaScript strategies, PowerShell ops. ~30k LOC, 1 star, **no LICENSE file** (default
all-rights-reserved — everything below is pattern-level reimplementation, not code reuse).
**Question:** what operational patterns are worth adopting into this repo's runtime/ops
layer, distinct from its statistical/backtest governance which is already far ahead.

## Headline

Park is the mirror image of this repo: weak on statistical discipline (auto-tunes live
strategy constants from n=5 samples over 5-day windows — curve-fitting-to-noise this
repo's prereg/holdout regime exists to forbid), but its **runtime governance** — serve-time
readiness gates with fallback, poison detection as a running service, a single automation
registry, and a fully-worked safety scaffold around closed-loop parameter changes — is more
mature than anything in this repo's ops layer today. Adopt the scaffolding, never the
practice of auto-editing live/pinned parameters from thin evidence.

## Verdicts

| # | Item | Verdict | Modification / gate |
|---|------|---------|---------------------|
| 1 | Serve-time readiness gates + rule-based fallback (entry model needs base-rate+5pp accuracy; exit model needs AUC≥0.55 & ≥100 minority labels; else falls back, reason logged) | **Adapt** | Runtime analog of this repo's `dev_gate.py`/`adequacy.py`: a signal family should need to clear numeric criteria to contribute to an advisor report, else excluded with logged reason. Direct template for whenever paper-trading promotion gets its own gate (currently NOT authorized) — park's phase-escalation barriers (≥150 trades, AUC≥0.58, ≥4 distinct weeks) are a ready-made shape: counts + metric floors + calendar-diversity minimums, pinned in advance. |
| 2 | Auto-tuning safety scaffold (measure→tier-by-n→suggest→cross-bucket unanimity→average deltas→per-run step cap→hard range ceiling→re-simulate invariants→timestamped backup+append-only audit JSON; 5-day evidence aging; last-applied cutoffs preventing re-application; dry-run default; dashboard reads the same code path the automation decides from) | **Adapt (scaffold only)** | Maps to TODOS.md P2 (skill-weight seam) / spec §8 (shrinkage, rank-IC/IR, Brier gating, min OOS window) — spec pins the statistical half, park documents the operational half needed once the `pipeline/run.py` non-uniform-weight tripwire is ever relaxed. **Never** adopt the practice itself (n=5, 5-day windows) against pinned/prereg parameters. Their own incident is the cautionary tale: a pullback parameter ratcheted to ~0.012 over two days before they added a hard floor — PROGRAM_RECORD-style incident→invariant lesson worth citing verbatim if this seam ever activates. |
| 3 | Poison detection as a running service (six checks: cross_symbol, dup_scan, feature_psi, label_drift, empty_window, determinism; green/amber/red verdict naming implicated rows; quarantine-not-delete remediation) | **Adapt** | Extends this repo's gate-time integrity work (`source_integrity/`, PIT-conformance tests) to ingestion-time. Quarantine-not-delete is congruent with the existing "missing data means unavailable, not defaulted" rule. No current lane owns this — new small utility, not gated on anything. |
| 4 | Single automation registry (`TASKS.md` inventorying every scheduled task: watchdogs, circuit-breaker watchdog, naked-position guard, hardware monitor, log pruning, autopush, each with schedule/purpose/setup script) | **Adapt (trivial)** | This repo's automation is scattered across npm scripts, Claude hooks/skills, and Routines with no single inventory. Cheap win, no dependencies. |
| 5 | Circuit breaker as an independent watcher process (vs. this repo's DB-trigger kill switch, which is enforcement without a watcher) | **Adapt** | Pair the existing kill-switch trigger with a separate monitor/alerter — enforcement and detection as separate failure domains. |
| 6 | Off-site backup of model weights / result artifacts | **Adapt** | Extend to this repo's floor artifacts and checkpoint DB; same rationale as their daily weights backup. |
| 7 | Retrain/validation blocking window (validation refuses to run 13:50–14:10 while retraining writes artifacts) | **Adapt (small)** | Generalizes to "gates don't run while artifacts rebuild" — apply wherever the floor runner and checkpoint writes could race. |
| 8 | Feature-schema/layout guards (`feature_utils.py` refuses to mix incompatible feature orderings across model versions) | **Adapt (small)** | Cheap insurance if SignalBundle-consuming models ever version their inputs — a layout hash check. |
| 9 | Daily retraining on live trade data | **Reject** | Coherent for an intraday futures system; antithetical to this repo's frozen-pin, `DEV_FAILED`-floor research-advisor posture. |
| 10 | C#/NinjaScript strategies, Windows Task Scheduler/S4U, PowerShell ops, MEGA cloud upload | **Reject** | Platform-specific; patterns transfer (items 1-8 above), artifacts don't. |

## Debate cruxes (what would change these verdicts)

- **Item 2 stays scaffold-only** unless a validated skill-weight calibration source ever
  exists (per TODOS.md P2, none as of 2026-07) — until then there is nothing to safely
  auto-tune, so only the safety mechanics are worth pre-building, not a live instance.
- **Item 1's fallback semantics differ from park's:** park falls back to a cruder
  rule-based signal at serve time (continuity matters, it's live money). This repo's
  correct analog is exclusion-with-logged-reason, not a rule-based substitute signal —
  fabricating a fallback family would violate "missing data means unavailable, not
  defaulted."
- **Item 3 dies as proposed** if it duplicates existing `source_integrity/` coverage
  rather than extending it to ingestion-time — needs a quick check against current EDGAR/QC
  bridge scope before scoping the utility, not assumed net-new.

## Executed from this review

Nothing built. This is a review-only doc; all nine adapt/reject items above are unscoped
future work, not committed to any lane or priority order.
