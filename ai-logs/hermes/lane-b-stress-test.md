# Lane B Candidate-Search Plan — Adversarial Design Review (DEBATE)

You are an adversarial design-review panel. The artifact under test is a **written
implementation plan**, not running code. There is NOTHING to execute. Do **NOT** edit
any files. Return your findings as **text only** (your final message IS the deliverable).

## What to read first (zero re-derivation)

1. The plan under test:
   `docs/superpowers/plans/2026-06-16-lane-b-candidate-search.md`
   (committed `9145249` on `main`, not pushed). Read it in full.
2. The frozen code the plan mirrors / imports read-only (verify every claim against these —
   a finding that contradicts the actual code is wrong and must be dropped):
   - `apps/quant/advisor/backtest/pipeline.py`   (`run_dev_sweep`, `run_holdout`, `_family_scores`)
   - `apps/quant/advisor/backtest/data_floor.py`  (`floor_metrics` — the verdict branch the bench mirrors)
   - `apps/quant/advisor/backtest/validation.py`  (`validation_report`, `deflated_sharpe`, `n_for_dsr`)
   - `apps/quant/advisor/backtest/validation_prereg.py` (`ValidationPreReg`, `DEFAULT_VALIDATION`)
   - `apps/quant/advisor/backtest/splits.py`      (`purged_splits` — the fold geometry)
   - `apps/quant/advisor/backtest/continuous_signals.py` (`raw_metric`, `fit_percentile_transform`, `apply_transform`)
   - `apps/quant/advisor/backtest/dev_gate.py`    (`dev_gate` pass conditions)

## SETTLED — not up for debate (attack WITHIN these rails; do not re-litigate them)

- **Report-only posture.** The bench reports a verdict; it never sizes capital.
- **Frozen floor + `allocator.py` / `ensemble_vote` / `backtest/` are untouched.**
  `node tools/run-floor.mjs --enforce` stays **exit 1**; `npm run advisor-gate` stays exit 0.
- **Rail-safe separate `apps/quant/advisor/research/` bench** that imports the frozen
  `backtest/` primitives **read-only** and re-orchestrates them (~40-line mirror).
- **Promotion is deferred to Plans 1b/3** — explicitly out of scope here.
- **No fabricated data.** Reading A uses only `floor_prices.csv`. Reading B (fundamentals
  fixture) is operator-gated and NOT built here.
- **As-of / leakage discipline** and the **slice-then-compute landmine** (a `shift(L)` signal
  is NaN→flat for the first L dev rows; classic 756–1260 LT-reversal is fixture-infeasible →
  the plan pre-registers an intermediate reversal at `value_lookback=270, value_skip=126`).

The debate attacks the plan's **correctness and completeness**, NOT its architecture or rails.

## Agenda — attack surfaces to target (seed these; avoid generic style nits)

1. **Golden-replication sufficiency.** Reproducing momentum/trend at 0.732/0.828 (`abs=0.01`)
   proves faithfulness for NaN-free families. Does it guarantee faithfulness for `value`, with
   its long NaN prefix + the empty-`pos` percentile-fit path the floor families never exercise?
   Is `abs=0.01` a strong enough drift anchor, or should the mirror be proven equal to the
   frozen pipeline ELEMENT-WISE for shared families?
2. **Orthogonality measure.** Pearson on pooled RAW values across heterogeneous assets — vs
   Spearman, or correlation on the POST-TRANSFORM conviction scores that actually enter the
   blend (negatives→0 flatten the relationship). Is `τ=0.40` justified? Is treating `momentum`
   as diagnostic-only (not gated) correct, given momentum is the blend partner?
3. **Is Reading A predetermined?** `lookback=270/skip=126` sits near `long_momentum` (252).
   Is it near-certain to fail the orthogonality gate (a near-rigged negative), making B0–B2
   wasted vs jumping to Reading B? If so, what is the minimum bench worth building, and in
   what ORDER (the plan builds Tasks 3/4 — the expensive pipeline mirror + golden — BEFORE the
   cheap kill-gate Task 6)?
4. **DSR validity.** The bench's `candidate_floor.py` passes `DEFAULT_VALIDATION` (the FLOOR's
   `declared_trials_N=45`, `declared_var_sr=1e-4`) to `validation_report`. Verify against
   `validation.py`/`validation_prereg.py`: is `CandidatePreReg.declared_trials_N` actually used,
   or dead? Is the rail-#4 promise "each secondary run increments `declared_trials_N` and re-runs
   DSR at higher N" implementable as written? Is reusing the floor's `var_sr` (calibrated from the
   floor's A–E return distribution) valid for a different value+momentum return distribution?
5. **Shared-holdout multiple testing.** The bench reuses `floor_prices.csv`, whose tail IS the
   floor's already-reserved holdout. Is touching it under a second pre-registration a holdout-reuse
   violation even with dev-only construction? If Reading A fails and Reading B later reuses the
   same tail, is that a second peek? What makes the holdout claim honest (a touch-ledger)?
6. **Promotion transfer.** If the bench verdict is PASSED, does that legitimately transfer to the
   frozen release gate (which runs momentum/trend on its OWN PreRegConfig/PREREG)? Or does
   promotion require re-running the candidate through a re-pre-registered floor on a FRESH holdout,
   not the side bench on the already-peeked tail?
7. **Power.** ~1654 dev rows / 4 purged test folds / block-bootstrap LCB, with `value_lookback=270`
   leaving fold-1's value-leg fit on only ~10–25 positive points. Realistic power to detect a true
   §7.2 edge, or so underpowered (compounded by plan4's structural finding that a fixed blend of
   correlated long-only price families cannot beat its best member OOS net of costs) that even a
   real candidate lands INCONCLUSIVE? Is a PASSED outcome even reachable for this candidate class?
8. **Mirror drift.** The ~40-line `candidate_pipeline.py` mirror diverges from frozen `pipeline.py`
   over time. Is the golden test a one-time gate or a per-change guard? Does asserting hardcoded
   0.732/0.828 conflate a floor CHANGE with a mirror DRIFT? What makes the duplication safer?

## Output contract (every agent)

Return a **numbered list** of findings. Each finding is either a `WEAKNESS` or an `IMPROVEMENT`
and MUST give:
- **(a) Plan hook** — the exact plan task / rail / section it hits (e.g. "Task 7 Step 3",
  "rail #4", "Design-decision note").
- **(b) Why it is a real defect, not a style nit** — grounded in the actual frozen code where
  relevant (cite the file/function). If your claim depends on code behavior, state what the code
  does.
- **(c) A concrete fix** — the specific plan edit (new/changed task step, new test, reordered
  build, new pre-registered field, tightened assertion, added ledger, etc.).

Prefer depth over breadth: 5–10 well-grounded findings beat 20 shallow ones. If you believe an
attack-surface concern is already correctly handled by the plan, SAY SO with the evidence
(which task handles it) rather than inventing a finding.

## Synthesis

The synthesis step ranks all findings by severity (Critical / High / Medium / Low), DE-DUPES
findings that converge (e.g. #1 and #8 likely converge on "replace the hardcoded-number golden
test with an element-wise mirror-equals-frozen equality check"; #3 and #7 likely converge on
"reorder build to fail-fast at the kill-gate + state the structural power ceiling"), and for each
keeps the single best concrete fix. Output the ranked, de-duped list as the final deliverable.
