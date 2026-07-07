# Hermes deliberation — RED TEAM voice (read-only; deliberation, not implementation)

You are the adversarial voice in a multi-model deliberation on the AIHedgeFund research
program's next step. Your ONLY job: try to refute the recommendation below. Find the
strongest case AGAINST it, hidden assumptions, cognitive biases, and cheaper or
higher-EV alternatives. Do not be agreeable. If a piece survives your attack, say so
explicitly — grudging survival is signal.

RULES (hard):
- READ-ONLY deliberation. Do NOT modify any tracked file. Do NOT run
  ls_reversal_screen.py, any backtest, pytest, npm, or any pipeline. No gates.
- Write your findings to exactly one NEW file: ai-logs/hermes/debate-redteam-2026-07-06.md
- The L/S reversal lane is CLOSED forever (frozen one-shot verdict, recorded today).
  Proposing reruns, threshold changes, or Gate-2 work is out of bounds and will be
  discarded — attack the recommendation, not the frozen record.

## Context (verified today, 2026-07-06)

- Program record: 9 consecutive pre-registered negatives (advisor-v1 floor fail, Plan-4
  DEV_FAILED, Lane-B kill-gate negative, WS4 Reading-B DEV_FAILED, WS3D Reading-C
  DEV_FAILED, corrected residual screen RED everywhere, QC source-integrity STOP,
  SESTM Phase-0 STOP, and today L/S Gate-1 CLOSED).
- Today's Gate-1 result (frozen, final): post-cost reversed IR value 0.1787,
  fundamental_value 0.1922, lazy_prices 0.2389 vs tau 0.20, rule needed 2-of-3;
  tripwire reproduced published pre-cost IRs cleanly. Lane CLOSED.
- Known structural confounder: the 30-mega-cap survivor universe. The keyless broad
  screen (T0.2b) was INCONCLUSIVE and survivorship-confounded direction-specifically;
  settling it requires paid delisting-aware PIT data (Decision 3).
- Queue: insider Form-4 lane (proposed from external repo review, ranked next),
  Decision-3 paid-data lever, Decision-4 disclosure tone (SESTM died on corpus
  availability), Decision-2 diagnostics kill-criterion evaluation ~2026-09-04
  (operator-reported real runs; <8 runs by then = kill).
- Constraints: solo operator, prereg + one-shot ceremony discipline, MinBTL
  multiple-testing budget, blinded holdout, no LLM in the decision path.

Read for full grounding (all committed):
- docs/superpowers/notes/2026-07-06-ls-reversal-gate1-result.md
- docs/superpowers/notes/2026-07-04-program-review-memo.md
- docs/superpowers/plans/2026-07-06-external-repo-review-ai-hedge-fund.md
- docs/superpowers/notes/2026-06-23-broad-universe-residual-screen-result.md
- docs/superpowers/notes/2026-06-23-phase1-direction-roleplay-debate-synthesis.md
- docs/superpowers/harness/PROGRAM_RECORD.md

## THE RECOMMENDATION UNDER ATTACK (from the Claude lane)

1. H1 now (core bet): open the insider Form-4 lane as a Gate-1-style kill-screen-first
   lane. Free keyless EDGAR data; PIT conformance tests already merged. Prereg must
   inherit two lessons at freeze time: (a) cost overlay frozen at design time, (b)
   benchmark chosen for the signal's structure (not reflexively "beat SPY").
   Include a pre-freeze contamination check: if >~90% of mega-cap Form-4 volume is
   10b5-1-scheduled, the lane's prior collapses and the paid-data lever jumps the queue.
   Claimed rationale: Form-4 is the first orthogonal INFORMATION SOURCE lane (not
   another price transform), so 8 of 9 negatives do not degrade its prior; low-turnover
   event-driven signals are structurally more cost-robust (the Gate-1 lesson).
2. H1 parallel (design-only, zero cost): Decision-3 scoping one-pager — exact vendor
   for delisting-aware PIT data, cost, and the single frozen question it settles.
   Scoping only, no purchase.
3. Fixed anchor: Decision-2 evaluation ~2026-09-04 proceeds untouched.
4. H2 conditional (new program-level exit ramp, frozen now while neutral): if Form-4
   CLOSES and the Sept-4 evaluation kills, the next ruling is program-level — either
   fund the Decision-3 purchase for one de-confounding diagnostic, or formally pivot
   the program's product to the validation harness + negative record.
5. Parked: disclosure tone (same corpus wall as SESTM, lowest rank).

## YOUR DELIVERABLE

Write ai-logs/hermes/debate-redteam-2026-07-06.md containing:
1. Numbered findings, each with: severity (HIGH/MED/LOW), the specific claim attacked,
   your counter-argument with evidence from the repo record, and the concrete change
   it implies (amend / reorder / drop / add).
2. The strongest single argument for a DIFFERENT next step than opening Form-4.
3. Bias audit: is the recommendation exhibiting lane-hopping, queue-order anchoring,
   free-data bias (picking what is free over what is informative), or premature
   pivot-to-meta (the "harness as product" reframe as escape hatch)?
4. Explicit list of which recommendation elements SURVIVE your attack.
Be terse and technical. No praise, no summary of what you were asked.
