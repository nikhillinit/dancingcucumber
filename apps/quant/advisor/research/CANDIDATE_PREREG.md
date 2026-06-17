# CANDIDATE_PREREG — Lane B (Reading A: price-only intermediate-term reversal)

**Status:** FROZEN before any evaluation result was observed. This document pre-registers
the methodology and the Task-6 orthogonality kill-gate decision rule. Outcomes are judged
by the floor's existing gates verbatim — no new outcome thresholds are invented here.

This is the bench's OWN hash surface; it never touches the floor's `PreRegConfig`
(`1ad2ed4a…`) or `ValidationPreReg`. The frozen floor stays `DEV_FAILED` / `--enforce`
exit 1 regardless of anything recorded here (promotion is out of scope — Plans 1b/3).

## Identity (immutable)

| Field | Value |
|---|---|
| `candidate_hash(DEFAULT_CANDIDATE)` | `578cce4bb6311e239adec23e5ce8062df0891562d2af322e8323950325e69d1d` |
| `candidate_validation_hash(DEFAULT_CANDIDATE_VALIDATION)` | `5f5254d3025a0ab3e4de3151825e77d1aee813cf69d802250b82a8811cb4ca8b` |
| Fixture path | `apps/quant/advisor/tests/fixtures/floor_prices.csv` |
| Fixture SHA-256 | `d40b9959ba34241a2ea3d60f45516c9f0781718de83f2d77e93de1e23830e2c1` |
| Fixture bytes / shape | 1,310,650 / (2264, 31) — SPY + 30 assets |

> `candidate_run_hash(DEFAULT_CANDIDATE, FIXTURE)` (config **+ fixture bytes**, Amendment F2)
> is the holdout-unlock key; it is added with Task 7 and recorded here only if Task 6 passes
> and a Task-8 holdout evaluation is actually run.

## Frozen methodology constants (`DEFAULT_CANDIDATE`)

- `families = ("value", "momentum")` (primary order)
- `value_skip = 126` (exclude the recent 6-month momentum window)
- `value_lookback = 270` (~13mo→6mo formation; FIXTURE-FEASIBLE, `< ~325` = fold-1 train end,
  so every dev fold has a live value leg — see plan horizon note / Amendment F6)
- `orthogonality_tau = 0.40`
- `declared_trials_N = 45` **on `CandidateValidationPreReg`** (the surface `validation_report`
  reads — Amendment F1); the secondary run bumps it.

## Task-6 orthogonality kill-gate — PRE-REGISTERED DECISION RULE (Amendment F4)

The blend uses the **post-transform** long-flat conviction scores (`apply_transform`; raw ≤0 →
flat), so the diversification §7.2 rewards lives in the transformed scores, not the raw. The
gate therefore decides on the **post-transform fold-level** correlation
(`dev_fold_post_transform_corr`), fit per dev fold on TRAIN rows, evaluated on TEST rows,
holdout excluded, pooled across folds + assets.

**Gate axis (the rejected-factor relabel check):** `long_momentum` AND `mean_reversion`.

- **PASS (proceed to B2):** `max(|corr_pt(value, long_momentum)|, |corr_pt(value, mean_reversion)|) < 0.40`.
- **FAIL (cheap kill — record negative, skip B2, go to Reading B / Task 11):** either gated
  post-transform `|corr| ≥ 0.40`.

**Diagnostics, NOT gates (recorded for lineage, never decide the kill):**
- Raw Pearson `dev_fold_raw_corr(..., method="pearson")` and Spearman
  `dev_fold_raw_corr(..., method="spearman")` for all three neighbors. A raw `|corr|` near 1 with
  a low post-transform `|corr|` is the expected signature of "same underlying signal, opposite
  tail, distinct long-flat book" — that is real diversification, not redundancy, so it does NOT
  kill (Amendment F4; verified mechanic: a perfect negated relabel fires on disjoint days under
  the ≤0→flat clamp → raw −1 but post-transform ≈ −0.3).

**`momentum` clause (blend partner, separate from the kill axis):** `momentum` is `value`'s
blend partner, reported as the most decision-relevant diagnostic. A HIGH post-transform
`|corr_pt(value, momentum)|` does NOT kill; it **demotes** a PASS from "clean orthogonality
PASS" to a "coarse pre-filter — §7.2 on the real fixture adjudicates," it is not a green light.

**Bias note (pre-committed):** the expected outcome is a kill, so the rule above is frozen
before the run specifically to prevent reading a high RAW `|corr|` as a kill when the gate
surface is post-transform. Let the post-transform numbers decide.

## Candidate order (frozen)

1. **Primary:** `value + momentum`.
2. **Secondary (only if primary dev-passes-but-holdout-fails):** `value + momentum + trend`.
   This increments `declared_trials_N` on `CandidateValidationPreReg` and re-runs DSR at the
   higher N. No other horizons or family sets are tried (no post-hoc p-hacking).

## Acceptance bar (Task 8, only if Task 6 passes) — the floor's gates VERBATIM

- §7.2: ensemble beats best single family — block-bootstrap `delta_lcb > 0` at `final_lcb=0.95`.
- §7.1: ensemble beats SPY by the margin — `spy_lcb > margin (0.0)` at `final_lcb=0.95`.
- DSR ≥ 0.95 at the candidate's `declared_trials_N` (report-only promotion-readiness, not the
  machine `passes`, mirroring floor parity — Amendment F1).

No new thresholds. A holdout touch is logged to `HOLDOUT_LEDGER.md` and BURNS the shared
reserved tail (promotion requires a FRESH holdout — Amendment F2).
