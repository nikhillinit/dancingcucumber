# Post-L/S operator ruling package

Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."

Governing plan: `docs/superpowers/plans/2026-07-06-decision3-form4-no-alpha-ruling-plan.md` (frozen `81c5276`, operator ratifications `33d44a4`). Operator election "go to gate 3" recorded 2026-07-06 after the round-2 three-model deliberation (`465932a`) returned a unanimous GO TO GATE 3 recommendation.

## Live Record

- Canonical `docs/superpowers/harness/PROGRAM_RECORD.md`: 9 hash-chained rows, chain tip `591a2e3935bd over 9 rows` (row 9 = L/S reversal Gate-1 CLOSED, appended `0977632`).
- L/S state: CLOSED by `docs/superpowers/notes/2026-07-06-ls-reversal-gate1-result.md`; write-once lock; no reruns or threshold changes, ever.
- Decision-2 state: report-only diagnostics lane remains governed by `docs/superpowers/plans/2026-07-06-decision2-diagnostics-cli.md`, usage evaluation ~2026-09-04 (untouched by this package).
- Form-4 state: proposed lane only; no prereg, no parser, no return measurement authorized.
- The verdicts below are governance-input verdicts, NOT `PROGRAM_RECORD.md` rows unless separately appended by the closeout protocol.

## Evidence Inputs

| Artifact | Verdict | Binding notes |
| --- | --- | --- |
| Decision-3 paid-data source matrix (`docs/superpowers/notes/2026-07-06-decision3-paid-data-source-matrix.md`, commit `113a07b`; criteria freeze `1e87157`, blob `345fc5de6c69…`) | **STOP** | CRSP/WRDS — the only known native-`DLRET` source — did not provide public or sample-backed evidence clearing the frozen <= $1,000 one-time + <= $100/month ceiling within the one allowed session (crsp.org subscription page and WRDS Terms of Use cited; no public pricing, institutional access model). QC/Sharadar/Norgate remain STOPped on the delisting-return omission carried from Step-0 by citation; no documented-change candidate was recorded in the one-session Gate-1 evidence set. |
| Form-4 Phase-0 source matrix | **NOT RUN** | Gate 2 was deliberately not opened (operator election 2026-07-06, on the unanimous panel recommendation): the frozen Gate-2 criteria were found non-reproducible as written (EDGAR dedup keys, Form 4/A amendment handling, transaction-code map, 10b5-1 schema element, and dollar-denominator rules all unpinned) and the concentration caps likely bind by construction on the 30-mega-cap universe. |

Definitions (panel improvement): **NOT RUN** means the gate was deliberately not opened; **STOP** means the gate was opened and failed a pinned criterion. Decision-3's STOP does not mechanically stop Form-4 — each matrix stands under its own frozen criteria; the recommended default below, however, treats STOP + NOT RUN together as evidence against opening a new signal lane now.

## Operator Decision Table

| Option | Allowed only if | Consequence |
| --- | --- | --- |
| Fund enterprise-data purchase (Decision 3 above ceiling) | Operator deliberately accepts cost/license/integration above the frozen <= $1,000 + <= $100/month ceiling (e.g., CRSP-grade access via a written quote). | Write a separate procurement/integration plan under a new filename. No alpha run until a future prereg is frozen. |
| Open future Form-4 prereg | A NEW standalone Form-4 Phase-0 criteria artifact is frozen first (fixing the ten recorded reproducibility defects), that matrix PASSes all pinned bars, AND the operator elects a fresh hypothesis slot. Continuation of this plan is not sufficient authority. | Write a new Form-4 criteria file and, on PASS, a new prereg under a new filename. No return metric until that prereg is frozen. |
| Stop opening signal lanes | Decision-3 STOPped and Form-4 was NOT RUN or is not worth a fresh slot, or operator chooses to preserve the negative record. | Continue the validation harness and Decision-2 diagnostics only. Reopening any signal lane requires a new deliberate operator decision under a new filename. |
| Defer | Operator names a review date (no more than 90 days out unless tied to a specific named external trigger, e.g. a vendor pricing reply or SEC schema release) and the exact missing evidence. | If the named evidence is not obtained by the review date, the deferred branch resolves to STOP by default, and reopening requires a new operator decision under a new filename. No drift into implementation. |

## Recommended Default

**Stop opening signal lanes.** This follows mechanically from the matrix verdicts: Decision-3 STOPped under its frozen criteria (the survivorship confound cannot be settled at the operator's ceiling — the only native-delisting-return source is access-locked, and every ceiling-fitting vendor omits the terminal loss), and Form-4 was deliberately NOT RUN after its screening criteria were shown to be non-reproducible as frozen and likely concentration-bound by construction on this universe. Per the operator's recorded STOP acceptance (2026-07-06), neither ceiling, window, denominator, threshold, universe, nor classification rules may be revised to salvage a lane. The program continues on its two live, already-frozen tracks: the validation harness asset and the Decision-2 diagnostics lane with its ~2026-09-04 usage evaluation. Enterprise-data funding and a properly re-frozen Form-4 Phase-0 both remain available as future deliberate operator decisions under new filenames.

## Recorded Ruling (operator, 2026-07-06)

**STOP OPENING SIGNAL LANES.** The operator adopted the recommended default, satisfying its allowed-only-if condition (Decision-3 STOPped under frozen criteria; Form-4 NOT RUN). Binding consequences, per the decision table and the operator's recorded STOP acceptance:

- The program continues on exactly two live tracks: the validation-harness asset and the Decision-2 diagnostics lane (usage evaluation ~2026-09-04, frozen kill criterion).
- No new signal lane may be proposed as a continuation of any existing plan. Reopening requires a new deliberate operator decision under a new filename — specifically: enterprise-data funding via a written quote (Decision-3 escalation path), or a re-frozen standalone Form-4 Phase-0 criteria artifact addressing the ten recorded reproducibility defects plus an explicit fresh hypothesis slot.
- No ceiling, window, denominator, threshold, universe, or classification rule may be revised to salvage a lane.
- Disclosure tone (Decision 4) remains parked; the L/S lane remains CLOSED forever; the QC, SESTM, and residual-alpha negatives stand as recorded.

## Non-Goals

- No parser implementation.
- No data purchase.
- No Form-4 prereg.
- No returns, CAR, Sharpe, IR, DSR, floor, holdout, candidate pipeline, or effect-size metric.
- No `PROGRAM_RECORD.md` append unless a proper closeout workflow is explicitly opened.
