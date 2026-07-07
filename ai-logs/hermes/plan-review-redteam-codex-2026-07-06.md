# Codex red-team audit - frozen no-alpha plan and Gate-1 result

Scope: read-only adversarial audit of the amended no-alpha governance plan and executed Gate-1 source matrix. This audit does not rerun any screen, inspect any new performance output, or modify the frozen governance artifacts.

## Findings

1. **Severity: MED - Gate-1 CRSP/WRDS STOP is directionally right, but the cost/access proof is overstated.**

   The frozen criteria allow public pricing "if available" and otherwise require official documentation, samples, or written vendor answers. The executed matrix uses public CRSP/WRDS access pages to infer that CRSP cannot evidence the <= $1,000 one-time + <= $100/month ceiling and has no individual path. That supports a one-session STOP under the frozen evidence rule, but it is weaker than an actual quote or support refusal. The current wording drifts from "not evidenced inside the frozen session" into "no individual path" and "sample proof unreachable," which is stronger than the recorded evidence can carry.

   **Concrete change:** Keep the STOP, but narrow the Gate-3 wording to: "CRSP/WRDS did not provide public/sample-backed evidence clearing the frozen price/access ceiling in the one allowed session." Do not claim a definitive no-individual-access or no-sample path unless an official quote/support reply is added under a new operator-approved source inquiry.

2. **Severity: LOW - The optional documented-change vendor row was deleted with a universal claim that the matrix does not prove.**

   The matrix says no written evidence of any post-2026-06-23 capability change exists. The plan allowed deleting the optional row if no written evidence was found or claimed in the session, but the matrix does not preserve the actual searched universe, search terms, or vendor list behind "none exists." The result is probably harmless because Gate-1 already STOPs, but the phrasing creates a proof standard mismatch: absence from this session is not global absence.

   **Concrete change:** In the ruling package, phrase this as "no documented-change candidate was recorded in the one-session Gate-1 evidence set." If a future operator wants to re-open vendor-change checking, require a new filename and predeclared search list before evidence collection.

3. **Severity: MED - Gate-2 is still only a template in the amended plan, not a frozen standalone criteria artifact.**

   Commit `33d44a4` improved the Form-4 criteria before evidence by fixing the window, renaming the counted bucket, and recording STOP acceptance. That survives as a pre-evidence operator amendment. However, no `docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md` file exists yet. The Gate-2 thresholds therefore live inside an implementation-plan section, while Gate-1 had a standalone criteria file and blob identity before matrix execution. Running Gate-2 directly from the plan would be a weaker freeze pattern than Gate-1.

   **Concrete change:** If Gate-2 is ever run, first create and hash the standalone Form-4 Phase-0 criteria artifact, then treat that file as the frozen source of truth. For the current pending election, do not treat the amended plan text alone as sufficient execution authority.

4. **Severity: MED - The Gate-3 "defer" option is bounded, but it still needs STOP-by-default expiration semantics.**

   The decision table requires a review date and exact missing evidence, which is a real improvement over open-ended drift. The remaining hole is what happens if the review date arrives with no new source evidence. Without an explicit default, "defer" can become a parking lot that keeps signal-lane optionality alive without paying the cost of a new operator decision.

   **Concrete change:** Amend the Gate-3 ruling shell so any defer entry states: "If the named evidence is not obtained by the review date, the deferred branch resolves to STOP, and reopening requires a new operator decision under a new filename."

5. **Severity: LOW - Program-record language can accidentally imply Decision-3/Gate-1 is hash-chain state when it is not.**

   The plan correctly says Gate-3 should not append `PROGRAM_RECORD.md` unless a proper closeout workflow is opened, and the current program record remains at row 9. But the ruling package template asks for "Live Record" plus matrix verdicts in one artifact. That can invite future readers to treat the Decision-3 STOP as equivalent to a hash-chained row even before closeout.

   **Concrete change:** In Gate-3, add a sentence after the Decision-3 matrix row: "This is a governance-input verdict, not a `PROGRAM_RECORD.md` row unless separately appended by the closeout protocol."

## What Survives

- The no-alpha boundary survives: the plan and Gate-1 matrix keep returns, candidate metrics, floor, and holdout outside the source-scoping work.
- The Gate-1 STOP survives, with narrower wording: CRSP/WRDS fails the frozen one-session public/sample-backed cost/access proof, while QC/Sharadar/Norgate remain stopped on the carried delisting-return failure.
- The 2026-07-06 operator STOP acceptance survives as a useful freeze-discipline guard.
- The post-freeze amendment sequence survives because `33d44a4` records pre-evidence operator rulings, but it should not become precedent for changing criteria after source evidence is observed.
- The L/S lane remains closed forever; no reruns, threshold changes, or Gate-2 L/S work are justified.
- Form-4 remains optional no-alpha source diligence only, not an opened prereg or signal lane.

## Position

GO TO GATE 3 with NOT RUN. Gate-1 already STOPs under the frozen source criteria, and running Gate-2 now would spend another optional source-diligence branch before the operator ruling records whether Form-4 evidence is still worth a fresh slot after Decision-3 STOP.
