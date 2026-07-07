# Deliberation voice: RED TEAM AUDITOR (read-only)

Adversarially audit the frozen governance plan AS AMENDED and the executed Gate-1
result. Your job is to find what is wrong, not to agree. If something survives your
attack, say so explicitly.

RULES (hard):
- READ-ONLY. Do not modify any tracked file. Do not run pytest/npm/node or any
  backtest/screen. No returns, CAR, Sharpe, IR, floor, holdout, or effect-size
  metrics may be computed or inspected.
- Write your findings to exactly one NEW file: ai-logs/hermes/plan-review-redteam-codex-2026-07-06.md
- The L/S lane is CLOSED forever; reruns/threshold changes are out of bounds.

READ (all committed):
- docs/superpowers/plans/2026-07-06-decision3-form4-no-alpha-ruling-plan.md (plan, amended through commit 33d44a4)
- docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md (freeze 1e87157)
- docs/superpowers/notes/2026-07-06-decision3-paid-data-source-matrix.md (Gate-1 STOP, commit 113a07b)
- docs/superpowers/harness/PROGRAM_RECORD.md
- ai-logs/hermes/debate-redteam-2026-07-06.md and debate-blind-2026-07-06.md (prior round — this is your do-not-repeat list; only NEW findings count)

ATTACK SURFACES (prioritize):
1. Gate-1 execution integrity: was STOP derived strictly from the frozen criteria?
   Any cell where evidence is weaker than claimed (e.g., "carried from Step-0" used
   where live proof was required; the CRSP cost/access FAIL resting on absence of
   public pricing rather than an actual quote)?
2. Gate-2 Form-4 criteria as amended: internal contradictions, remaining unfrozen
   degrees of freedom, gameable denominators, window-interaction problems
   (2023-04-01..2026-06-30; >=100/>=20/>=20/>=12; <=15% uncertain x4; <=20%/<=35%/<=10%).
3. Gate-3 ruling package: can the decision table produce an ambiguous or
   self-serving outcome? Is the "defer" option a drift hole?
4. Process: the plan was amended after its first freeze commit (operator-ruled,
   pre-evidence). Does anything in that sequence create a precedent that weakens
   the freeze discipline?

DELIVERABLE: numbered findings with severity HIGH/MED/LOW + concrete change each;
explicit list of what survives; final one-line position on the pending election
(RUN GATE 2 vs GO TO GATE 3 with NOT RUN).
