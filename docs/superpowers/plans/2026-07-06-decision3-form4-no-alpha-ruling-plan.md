# Decision-3 / Form-4 No-Alpha Ruling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the post-L/S deliberation into frozen no-alpha evidence artifacts and one operator ruling, without opening Form-4 or any other signal lane.

**Architecture:** Gate 0 normalizes the live record and freezes the no-alpha boundary. Gate 1 produces a Decision-3 paid-data source-scope matrix. Gate 2 optionally produces a Form-4 Phase-0 source-capability matrix, still with no returns touched. Gate 3 packages a structured operator ruling: fund Decision-3, open a future Form-4 prereg, stop opening signal lanes, or defer with a review date and exact missing evidence.

**Tech Stack:** Markdown governance artifacts under `docs/superpowers/`; official vendor and SEC documentation; repo-local `rg`, `git`, and pytest conformance checks. No new runtime dependencies, no app code, no backtest code, no data purchase.

## Global Constraints

- No alpha lane opens in this plan.
- No return, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metric may be computed or inspected.
- Decision-2 diagnostics remains operationally separate until its +60-day usage evaluation on 2026-09-04; its outcome does not automatically authorize a signal lane.
- `docs/superpowers/harness/PROGRAM_RECORD.md` has nine hash-chained rows (row 9 = L/S CLOSED, appended 2026-07-06 in commit `0977632`, chain tip `591a2e3935bd`). Quote the live row count when executing.
- Form-4 Phase-0 PASS, if it happens, only authorizes a future operator decision on whether to write a prereg; it does not authorize alpha measurement.
- Decision-3 PASS, if it happens, only authorizes a budget/procurement decision; it does not authorize an alpha run.
- Budget cap exhaustion, session cap exhaustion, missing legal use, missing sample proof, or missing required fields all default to STOP.
- All source criteria must be frozen before evidence collection. Post-freeze denominator changes are new work under a new filename.
- Use repo-relative paths in artifacts. Keep `.claude/settings.local.json` and existing untracked debate logs out of any staged scope.
- Gates 1–2 evidence collection is web/vendor/SEC research: run it in a web-capable research lane (Claude-side), NOT via a sandboxed Codex/Hermes dispatch (QC-prereg precedent: networked research is operator/Claude-side; Hermes implements deterministic code only after a source contract is frozen).
- Artifact filenames below carry the 2026-07-06 stamp; if an artifact is created on a later date, substitute the actual creation date in its filename and update cross-references.
- Step-0 is binding precedent (operator ruling 2026-07-06): `docs/superpowers/plans/2026-06-23-qc-source-integrity-diagnostic-prereg.md` §1 already established that no affordable source natively supplies delisting returns (only CRSP `DLRET`, access-locked; QC/Sharadar/Norgate omit bankruptcy terminal loss). Gate 1 evaluates only the delta, never re-runs Step-0.
- Operator STOP acceptance (recorded 2026-07-06, before Gate-1 execution): "I accept STOP as the modal and valid outcome. If Gate 1 or Gate 2 fails under the frozen ceilings, window, count thresholds, uncertainty caps, or concentration caps, I will not revise the budget, window, denominator, threshold, universe, or classification rules to salvage a lane. A STOP means stop opening signal lanes unless a new operator decision deliberately funds enterprise data or opens a new hypothesis under a new filename."

---

## File Structure

- Create: `docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md`
  - Responsibility: frozen no-spend criteria for paid delisting-aware PIT data scoping.
- Create: `docs/superpowers/notes/2026-07-06-decision3-paid-data-source-matrix.md`
  - Responsibility: PASS/STOP result matrix for Decision-3 source scoping.
- Create only if Gate 1 is complete and operator still wants Form-4 evidence: `docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md`
  - Responsibility: frozen no-alpha criteria for Form-4 source capability.
- Create only if Gate 2 is authorized by the criteria file: `docs/superpowers/notes/2026-07-06-form4-phase0-source-matrix.md`
  - Responsibility: PASS/STOP source-capability matrix for Form-4 fields and event-density feasibility, with no returns.
- Create: `docs/superpowers/notes/2026-07-06-post-ls-operator-ruling.md`
  - Responsibility: final operator-facing ruling package with a structured decision table.

## Task 1: Gate 0 Record And Boundary Preflight

**Files:**
- Read: `docs/superpowers/harness/PROGRAM_RECORD.md`
- Read: `docs/superpowers/harness/LANE_LIFECYCLE.md`
- Read: `docs/superpowers/notes/2026-07-06-ls-reversal-gate1-result.md`
- Read: `docs/superpowers/plans/2026-07-06-decision2-diagnostics-cli.md`
- Read: `docs/superpowers/plans/2026-07-06-external-repo-review-ai-hedge-fund.md`
- Create: none
- Modify: none

**Interfaces:**
- Consumes: current repo truth and dirty-tree state.
- Produces: the exact language used in later artifacts for record count, lane state, and forbidden surfaces.

- [ ] **Step 1: Verify working tree scope**

Run:

```powershell
git status --short
```

Expected: existing local dirt may include `.claude/settings.local.json` and untracked `ai-logs/hermes/debate-*.md`. Do not modify, stage, or delete those files in this plan.

- [ ] **Step 2: Verify canonical record count**

Run:

```powershell
rg -n "Chain tip:|\\| [0-9]+ \\|" docs/superpowers/harness/PROGRAM_RECORD.md
```

Expected: nine table rows and `Chain tip: 591a2e3935bd over 9 rows`. If the record has later valid rows when this plan is executed, quote the live count instead of forcing nine.

- [ ] **Step 3: Verify no-lane-opening rule**

Run:

```powershell
rg -n "No lane opens|No further signal lanes|PROPOSED NEW LANE|2026-09-04" docs/superpowers/harness/LANE_LIFECYCLE.md docs/superpowers/notes/2026-07-04-program-review-memo.md docs/superpowers/plans/2026-07-06-decision2-diagnostics-cli.md docs/superpowers/plans/2026-07-06-external-repo-review-ai-hedge-fund.md
```

Expected: evidence that Form-4 is proposed only, no further signal lanes open without operator decision, and Decision-2 has a 2026-09-04 usage evaluation.

- [ ] **Step 4: Freeze language for downstream artifacts**

Use this exact boundary paragraph in every artifact created by later tasks:

```markdown
Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."
```

- [ ] **Step 5: Stop if a forbidden surface is already dirty**

Run:

```powershell
git diff --name-only -- apps/quant/advisor docs/superpowers/harness/PROGRAM_RECORD.md apps/quant/advisor/research/HOLDOUT_LEDGER.md
```

Expected: no output caused by this plan. If there is pre-existing unrelated dirt, record it in the final report and do not touch it.

## Task 2: Gate 1 Freeze Decision-3 Paid-Data Criteria

**Files:**
- Create: `docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md`
- Read: `docs/superpowers/notes/2026-06-23-broad-universe-residual-screen-result.md`
- Read: `docs/superpowers/notes/2026-06-23-phase1-direction-roleplay-debate-synthesis.md`
- Read: `docs/superpowers/notes/2026-07-04-program-review-memo.md`

**Interfaces:**
- Consumes: Gate 0 boundary paragraph and prior survivorship-confound record.
- Produces: a frozen source criteria file used by Task 3.

- [ ] **Step 1: Create the criteria file**

Create `docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md` with this content:

```markdown
# Decision-3 Paid-Data Source Criteria

Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."

## Frozen Question

Can any legally usable paid source mechanically settle the survivorship/data-fidelity confound from the broad-universe residual screen by providing delisting-aware, point-in-time equity data with native or explicitly documented delisting-return handling?

## Frozen Ceilings (operator ruling 2026-07-06)

- Scoping budget: $0 recurring, $250 one-time max, spent only on sample files, trial access, or documentation access. No production data subscription may be purchased inside the scoping task.
- Qualifying acquisition ceiling: a source PASSes cost/access only if <= $1,000 one-time plus <= $100/month recurring, cancellable without a multi-year lock, and licensed for this research use. Anything above returns to the operator as an explicit "enterprise-data purchase?" decision, not a PASS.
- Integration-time ceiling: <= 3 focused dev-days for sample proof plus loader integration; the scoping matrix itself is one session.
- These ceilings do not move to make a vendor fit.

## Delta Scope (Step-0 is binding)

- Consume `docs/superpowers/plans/2026-06-23-qc-source-integrity-diagnostic-prereg.md` §1 as settled: QC / Sharadar / Norgate delisting-return FAILs carry over by citation and are re-examined only on documented vendor change since 2026-06-23.
- Evaluate live: CRSP/WRDS access and cost at the frozen acquisition ceiling only.
- Disregard: S&P Capital IQ and other enterprise platforms (cannot clear the ceiling by construction).
- Honest prior, predeclared: at these ceilings the expected verdict is STOP unless a vendor can now prove native, legally usable delisting-return fidelity.

## Required Source Criteria

Every candidate source must be evaluated on these fields:

| Criterion | PASS requirement | STOP condition |
| --- | --- | --- |
| Legal use | License permits the intended solo-dev research use and a private normalized fixture or reproducible regeneration path. | License unclear, redistribution/regeneration forbidden for the needed workflow, or terms require a production/professional setup outside the operator budget. |
| Sample proof | Vendor provides sample file, trial, schema documentation, or official support reply proving the required fields. | Marketing page only, no field-level proof, or support refuses sample proof. |
| Delisted names retained | Historical universe keeps delisted securities visible after delisting. | Delisted names are absent, manually patched, or not auditable. |
| Delisting returns | Total-return stream includes delisting returns natively, or the vendor exposes a documented delisting-return field sufficient to compute them without self-imposed terminal losses. | Delisting returns are missing, vendor-caveated, self-built, or only inferable from disappearance. |
| Point-in-time membership | Tradable universe membership is knowable as of each historical date. | Only current membership, current symbol map, or restated history. |
| Small/mid coverage | Source reaches below the current 30 mega-cap floor enough to test the known size/survivorship confound. | Mega-cap-only or coverage unclear. |
| Corporate actions | Splits, dividends, ticker changes, mergers, and delistings have documented adjustment policy. | Adjustment policy absent or not PIT-auditable. |
| Cost and access | <= $1,000 one-time plus <= $100/month, cancellable without multi-year lock, licensed for this research use (Frozen Ceilings above). | Above ceiling, enterprise-sales-only pricing, or no public/sample-backed price evidence. |
| Integration estimate | One engineer can build a fixture/provenance loader without changing the frozen advisor floor or holdout. | Requires broad architecture rewrite, production credentials in tests, or touching frozen surfaces. |

## Decision Rule

PASS if at least one source clears every required criterion with official documentation, a sample, or a written vendor answer. STOP if no source clears every required criterion, if the session/budget cap is exhausted, or if evidence would require inspecting alpha performance.

## Allowed Evidence

- Official vendor documentation.
- Official sample files or schemas.
- Written vendor support answers.
- Existing repo notes cited by path.
- Cost quotes or public pricing, if available without purchase.

## Forbidden Evidence

- Any backtest output.
- Any return, CAR, Sharpe, IR, DSR, holdout, floor, or effect-size metric.
- Any self-filled terminal-loss assumption used as a substitute for native delisting-return proof.
- Any data purchase without a separate operator decision.
```

- [ ] **Step 2: Verify no placeholders or alpha metrics as executable work**

Run:

```powershell
rg -n "T[B]D|TO[D]O|fill\\s+in|implement\\s+later|run-floor|holdout ledger|candidate_run|backtest output" docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md
```

Expected: no output except allowed occurrences inside the explicit `Forbidden Evidence` section. If the command matches outside that section, edit the criteria file before proceeding.

- [ ] **Step 3: Capture freeze identity after commit-ready state**

Run:

```powershell
git hash-object docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md
```

Expected: one blob SHA. Paste this SHA into the Task 3 matrix header when Task 3 is executed. If the criteria file is later committed first, also record the commit SHA in the matrix header.

## Task 3: Gate 1 Execute Decision-3 Source Matrix

**Files:**
- Read: `docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md`
- Create: `docs/superpowers/notes/2026-07-06-decision3-paid-data-source-matrix.md`

**Interfaces:**
- Consumes: frozen Decision-3 criteria from Task 2.
- Produces: PASS/STOP matrix used by the operator ruling in Task 6.

- [ ] **Step 1: Create the matrix shell**

Create `docs/superpowers/notes/2026-07-06-decision3-paid-data-source-matrix.md` with this structure:

```markdown
# Decision-3 paid-data source matrix

Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."

Frozen criteria: docs/superpowers/plans/2026-07-06-decision3-paid-data-source-criteria.md | Freeze commit: write the live commit SHA, or UNCOMMITTED if this criteria file has not yet been committed | blob SHA: write the hash produced in Task 2 Step 3
Sessions used: 1 of 1

| Source | Legal use | Sample proof | Delisted names retained | Delisting returns | PIT membership | Small/mid coverage | Corporate actions | Cost/access | Integration estimate | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRSP / WRDS (live: cost/access ONLY) | Cite Step-0 carry-over. | Cite Step-0 carry-over. | Cite Step-0 carry-over (retained). | Cite Step-0 carry-over (native `DLRET` — the known PASS). | Cite Step-0 carry-over. | Cite Step-0 carry-over. | Cite Step-0 carry-over. | Write PASS or FAIL against the frozen <= $1,000 + <= $100/month ceiling, then cite official pricing/access evidence. | Write PASS or FAIL, then state one-engineer feasibility. | Write PASS or STOP and name binding failures. |
| QC / Sharadar / Norgate (carried from Step-0, 2026-06-23) | Carried by citation of `2026-06-23-qc-source-integrity-diagnostic-prereg.md` §1. | Carried. | Carried. | FAIL carried: bankruptcy/performance terminal loss omitted. Re-open ONLY on documented vendor change since 2026-06-23, cited. | Carried. | Carried. | Carried. | Carried. | Carried. | STOP carried unless a documented vendor change is cited. |
| Documented-change vendor (optional row) | Use only if written evidence shows a post-2026-06-23 capability change; otherwise delete this row. | Same. | Same. | Same. | Same. | Same. | Same. | Same. | Same. | Write PASS or STOP and name binding failures. |

## Verdict

Pre-committed decision rule from the frozen criteria:
> PASS if at least one source clears every required criterion with official documentation, a sample, or a written vendor answer. STOP if no source clears every required criterion, if the session/budget cap is exhausted, or if evidence would require inspecting alpha performance.

Write one paragraph beginning with either PASS or STOP. If PASS, name the source eligible for a later budget decision. If STOP, name every binding failure.
```

- [ ] **Step 2: Replace instruction cells with evidence**

Use official documentation, sample files, or written vendor answers only. If no official evidence exists for a cell, write `FAIL: no official field-level proof found in this session`.

- [ ] **Step 3: Enforce one-session cap**

Use one focused research session. Do not extend the matrix because a source is attractive but incomplete. If the session ends without a complete PASS, the matrix verdict is STOP or INCONCLUSIVE-STOP, not "continue shopping."

- [ ] **Step 4: Verify no unresolved placeholders**

Run:

```powershell
rg -n "ANGLE_TOKEN|T[B]D|TO[D]O|fill\\s+in" docs/superpowers/notes/2026-07-06-decision3-paid-data-source-matrix.md
```

Expected: no output.

## Task 4: Gate 2 Freeze Optional Form-4 Phase-0 Criteria

**Files:**
- Create only after Task 3 is complete: `docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md`
- Read: `docs/superpowers/plans/2026-07-06-external-repo-review-ai-hedge-fund.md`
- Read: `ai-logs/hermes/debate-redteam-2026-07-06.md`
- Read: `ai-logs/hermes/debate-blind-2026-07-06.md`
- Read: `apps/quant/advisor/source_integrity/edgar.py`

**Interfaces:**
- Consumes: Gate 0 boundary and Form-4 evidence from repo/external docs.
- Produces: frozen no-alpha criteria for a source capability matrix.

- [ ] **Step 1: Confirm Form-4 infrastructure gap**

Run:

```powershell
rg -n "FORM_25_TYPES|parse_master_index|Form 4|FORM4|ownership|10b5" apps/quant/advisor/source_integrity apps/quant/advisor/tests/test_source_integrity.py
```

Expected: current code proves reusable EDGAR mechanics for Form 25 / 25-NSE and related filings, not a Form-4 ownership parser or 10b5-1 exclusion path.

- [ ] **Step 2: Create the criteria file**

Create `docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md` with this content:

```markdown
# Form-4 Phase-0 Source Criteria

Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."

## Frozen Question

Can SEC Form 4 / ownership data support a future preregistered insider-trade lane on this repo's current advisor universe without building on missing event fields, weak 10b5-1/routine separation, or underpowered mega-cap event density?

## Required Source Criteria

| Criterion | PASS requirement | STOP condition |
| --- | --- | --- |
| Filing availability | Filing date is available from SEC data, and accepted timestamp is available from archive metadata if sub-day timing is ever needed. | Event date is the only usable date, or filing availability cannot be reconstructed. |
| Transaction fields | Transaction date, transaction code, shares/value, price, and acquired/disposed flag are typed or reliably parseable. | Any required transaction field is missing or only inferable from prose. |
| 10b5-1 handling | 2023+ plan flag or plan-adoption disclosure is usable; older data has a conservative `uncertain` bucket. | Unknown plan status would be treated as discretionary, or plan trades cannot be separated from discretionary trades. |
| Routine classification | The matrix defines `non-10b5-1`, `10b5-1`, and `uncertain` buckets before any count, and `uncertain` stays <= 15% in EACH primary denominator (purchase count, sale count, purchase dollar value, sale dollar value). | Routine/unknown events folded into a favorable bucket, or `uncertain` > 15% in any primary denominator (unknown plan status is unavailable, not neutral — do not infer discretion from silence). |
| Universe and dates | Universe is the current advisor floor for feasibility counts; filed-date keyed; no post-hoc small-cap expansion. | Universe expands after seeing counts, or transaction-date keyed availability is substituted. |
| Event density | >= 100 usable non-10b5-1 open-market events (dedup: same issuer + same insider + same accession + same direction = one event) AND >= 20 open-market purchases AND >= 20 open-market sales AND >= 12 distinct issuers, inside the frozen window. | Any count bar missed. If only one side (buys or sells) clears, the two-sided lane FAILS — a one-sided lane needs a new prereg. |
| Concentration | No single issuer > 20% of usable non-10b5-1 open-market event count or > 35% of usable non-10b5-1 dollar value; no single insider > 10% of event count. | Any concentration bar breached — underpowered/narrative-prone even if raw counts look acceptable. |
| Parser proof | At least three representative sample filings parse into the required typed fields outside app code, or official SEC flat files cover the same fields. | Parser proof requires changing `apps/quant/advisor` or does not cover representative filings. |
| No-alpha boundary | Matrix contains only availability, classification, and count evidence. | Any return, CAR, Sharpe, IR, effect-size, floor, holdout, or candidate metric is inspected. |

## Decision Rule

PASS only if every required criterion passes at its pinned threshold (>= 100 / >= 20 / >= 20 / >= 12 counts; <= 15% uncertain per primary denominator; <= 20% / <= 35% / <= 10% concentration). STOP if any required field is missing, any pinned bar is missed, denominator choice is ambiguous, or any alpha metric is touched. PASS authorizes only a later operator ruling on whether to write a Form-4 prereg.

## Frozen Evaluation Window (RATIFIED by operator 2026-07-06; immovable)

- Gating window: filed dates 2023-04-01 through 2026-06-30 (structured 10b5-1 checkbox era per the SEC amendments effective for reports filed on or after 2023-04-01 — plan status is typed, not inferred).
- Secondary, NON-GATING report: filed dates 2015-01-01 through 2023-03-31, with every unknown-plan event bucketed `uncertain` (expected to breach the 15% cap; reported for context only).
- FIXED end date: no roll-forward or backfill if Gate 2 runs later. A newer window requires a NEW criteria filename, adopted before any count is observed.
- Terminology (operator amendment): the checkbox proves 10b5-1 status only, NOT opportunistic intent (Cohen/Malloy/Pomorski sense). The counted bucket is `non-10b5-1 open-market`; the labels "discretionary"/"opportunistic" are reserved for a future prereg that freezes a routine/opportunistic classifier first.

## Predeclared Denominators

- Event definition: one event = same issuer + same insider + same accession + same direction (collapse duplicates before any count).
- Count denominator: number of usable open-market transactions after categorizing each event as `non-10b5-1`, `10b5-1`, or `uncertain`.
- Dollar denominator: gross disclosed transaction value where price and shares are available.
- Purchases and sales are reported separately, in counts and in dollars (four primary denominators).
- Filed date is the availability key.
- `uncertain` is unavailable for alpha design unless a later prereg explicitly excludes it.

## Forbidden Evidence

- Return, CAR, Sharpe, IR, DSR, floor, holdout, or effect-size metrics.
- Threshold tuning after counts are observed.
- Treating 10b5-1 or unknown-plan sales as discretionary.
- Modifying `apps/quant/advisor` parser/runtime code.
```

- [ ] **Step 3: Verify criteria file**

Run:

```powershell
rg -n "ANGLE_TOKEN|T[B]D|TO[D]O|fill\\s+in|run-floor|candidate_run" docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md
```

Expected: no output.

- [ ] **Step 4: Capture freeze identity**

Run:

```powershell
git hash-object docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md
```

Expected: one blob SHA. Paste this SHA into the Task 5 matrix header when Task 5 is executed.

## Task 5: Gate 2 Execute Optional Form-4 Source Matrix

**Files:**
- Read: `docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md`
- Create: `docs/superpowers/notes/2026-07-06-form4-phase0-source-matrix.md`
- Do not modify: `apps/quant/advisor/**`

**Interfaces:**
- Consumes: frozen Form-4 criteria from Task 4.
- Produces: PASS/STOP matrix used by Task 6.

- [ ] **Step 1: Create the matrix shell**

Create `docs/superpowers/notes/2026-07-06-form4-phase0-source-matrix.md` with this structure:

```markdown
# Form-4 Phase-0 source matrix

Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."

Frozen criteria: docs/superpowers/plans/2026-07-06-form4-phase0-source-criteria.md | Freeze commit: write the live commit SHA, or UNCOMMITTED if this criteria file has not yet been committed | blob SHA: write the hash produced in Task 4 Step 4
Sessions used: 1 of 1

| Source path | Filing availability | Transaction fields | 10b5-1 handling | Routine buckets (<=15% uncertain) | Universe/frozen window | Event density + concentration | Parser proof | No-alpha boundary | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEC ownership XML sample filings | Write PASS or FAIL, then cite the SEC evidence. | Write PASS or FAIL, then cite typed transaction-field evidence. | Write PASS or FAIL, then cite plan-disclosure evidence. | Write PASS or FAIL, then state bucket behavior. | Write PASS or FAIL, then state universe/date discipline. | Write PASS or FAIL, then state count evidence without returns. | Write PASS or FAIL, then cite sample parse proof. | Write PASS or FAIL, then confirm no alpha metric touched. | Write PASS or STOP and name binding failures. |
| SEC insider transaction flat files | Write PASS or FAIL, then cite the SEC evidence. | Write PASS or FAIL, then cite typed transaction-field evidence. | Write PASS or FAIL, then cite plan-disclosure evidence. | Write PASS or FAIL, then state bucket behavior. | Write PASS or FAIL, then state universe/date discipline. | Write PASS or FAIL, then state count evidence without returns. | Write PASS or FAIL, then cite sample parse proof. | Write PASS or FAIL, then confirm no alpha metric touched. | Write PASS or STOP and name binding failures. |
| Existing repo EDGAR plumbing | Write PASS or FAIL, then cite repo evidence. | Write PASS or FAIL, then cite repo evidence. | Write PASS or FAIL, then cite repo evidence. | Write PASS or FAIL, then cite repo evidence. | Write PASS or FAIL, then cite repo evidence. | Write PASS or FAIL, then cite repo evidence. | Write PASS or FAIL, then cite repo evidence. | Write PASS or FAIL, then confirm no app-code change. | Write PASS or STOP and name binding failures. |

## Verdict

Pre-committed decision rule from the frozen criteria:
> PASS only if every required criterion passes. STOP if any required field is missing, classification coverage is insufficient, event density is too sparse, denominator choice is ambiguous, or any alpha metric is touched. PASS authorizes only a later operator ruling on whether to write a Form-4 prereg.

Write one paragraph beginning with either PASS or STOP. If PASS, state limited future eligibility. If STOP, name every binding failure.
```

- [ ] **Step 2: Replace instruction cells with evidence**

Use official SEC docs, sample filings, SEC flat-file docs, and repo citations only. If representative samples cannot be parsed without app-code changes, record STOP.

- [ ] **Step 3: Verify app code unchanged**

Run:

```powershell
git diff --name-only -- apps/quant/advisor
```

Expected: no output.

- [ ] **Step 4: Verify no unresolved placeholders**

Run:

```powershell
rg -n "ANGLE_TOKEN|T[B]D|TO[D]O|fill\\s+in" docs/superpowers/notes/2026-07-06-form4-phase0-source-matrix.md
```

Expected: no output.

## Task 6: Gate 3 Write Operator Ruling Package

**Files:**
- Read: `docs/superpowers/notes/2026-07-06-decision3-paid-data-source-matrix.md`
- Read if present: `docs/superpowers/notes/2026-07-06-form4-phase0-source-matrix.md`
- Create: `docs/superpowers/notes/2026-07-06-post-ls-operator-ruling.md`

**Interfaces:**
- Consumes: Gate 1 matrix, optional Gate 2 matrix, and Gate 0 boundary.
- Produces: one operator-facing ruling package.

- [ ] **Step 1: Create the ruling package**

Create `docs/superpowers/notes/2026-07-06-post-ls-operator-ruling.md` with this structure:

```markdown
# Post-L/S operator ruling package

Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."

## Live Record

- Canonical `PROGRAM_RECORD.md`: write the live row count and chain tip from Task 1.
- Separate L/S state: CLOSED by `docs/superpowers/notes/2026-07-06-ls-reversal-gate1-result.md`; do not call it a hash-chained program-record row unless appended by the closeout protocol.
- Decision-2 state: report-only diagnostics lane remains governed by `docs/superpowers/plans/2026-07-06-decision2-diagnostics-cli.md`, with usage evaluation on 2026-09-04.
- Form-4 state: proposed lane only; no prereg and no return measurement authorized.

## Evidence Inputs

| Artifact | Verdict | Binding notes |
| --- | --- | --- |
| Decision-3 paid-data source matrix | Write PASS or STOP from the matrix verdict. | Name binding failures or the eligible source. |
| Form-4 Phase-0 source matrix | Write PASS, STOP, or NOT RUN from the matrix verdict. | Name binding failures, limited eligibility, or why it was not run. |

## Operator Decision Table

| Option | Allowed only if | Consequence |
| --- | --- | --- |
| Fund Decision-3 purchase | Decision-3 matrix PASSes and operator accepts cost/license/integration burden. | Write a separate procurement/integration plan. No alpha run until a future prereg is frozen. |
| Open future Form-4 prereg | Form-4 matrix PASSes all required criteria, Decision-3 is declined or deferred explicitly, and operator elects a fresh hypothesis slot. | Write a new Form-4 prereg under a new filename. No return metric until that prereg is frozen. |
| Stop opening signal lanes | Decision-3 STOPs, Form-4 STOPs or is not worth a fresh slot, or operator chooses to preserve the negative record. | Continue validation harness and Decision-2 diagnostics only. |
| Defer | Operator names a review date and the exact missing evidence. | No default drift into implementation; next review starts from this ruling package. |

## Recommended Default

Write one paragraph recommendation based only on the matrix verdicts. It must choose one of: fund Decision-3 purchase, open future Form-4 prereg, stop opening signal lanes, or defer.

## Non-Goals

- No parser implementation.
- No data purchase.
- No Form-4 prereg.
- No returns, CAR, Sharpe, IR, DSR, floor, holdout, candidate pipeline, or effect-size metric.
- No `PROGRAM_RECORD.md` append unless a proper closeout workflow is explicitly opened.
```

- [ ] **Step 2: Replace instruction cells with final ruling evidence**

Use the live matrix verdicts. If the Form-4 matrix was not run, write `NOT RUN` and explain why in one sentence.

- [ ] **Step 3: Verify no unresolved placeholders**

Run:

```powershell
rg -n "ANGLE_TOKEN|T[B]D|TO[D]O|fill\\s+in" docs/superpowers/notes/2026-07-06-post-ls-operator-ruling.md
```

Expected: no output.

## Task 7: Verification And Closeout

**Files:**
- Verify: all files created by this plan.
- Do not modify: `apps/quant/advisor/**`, `docs/superpowers/harness/PROGRAM_RECORD.md`, `apps/quant/advisor/research/HOLDOUT_LEDGER.md`

**Interfaces:**
- Consumes: all created artifacts.
- Produces: completion evidence.

- [ ] **Step 1: Verify edited file scope**

Run:

```powershell
git diff --name-only
```

Expected: only the planned `docs/superpowers/plans/2026-07-06-*.md` and `docs/superpowers/notes/2026-07-06-*.md` artifacts, plus any pre-existing unrelated dirt already present before Task 1.

- [ ] **Step 2: Verify frozen/runtime surfaces unchanged**

Run:

```powershell
git diff --name-only -- apps/quant/advisor docs/superpowers/harness/PROGRAM_RECORD.md apps/quant/advisor/research/HOLDOUT_LEDGER.md
```

Expected: no output caused by this plan.

- [ ] **Step 3: Run docs conformance if available**

Run:

```powershell
npm run check
```

Expected: full suite passes (`check` = `node tools/run-pytest.mjs apps/quant/advisor/tests`; docs-only artifacts must not change the count).

- [ ] **Step 4: Verify no holdout or floor command was needed**

Run:

```powershell
git diff -- apps/quant/advisor/research/HOLDOUT_LEDGER.md apps/quant/advisor/backtest/FLOOR_RESULT.md
```

Expected: no output.

- [ ] **Step 5: Final report**

Report:

```markdown
Mode: planning/governance only.
Changed files: list every changed file.
Decision-3 matrix verdict: write PASS or STOP.
Form-4 matrix verdict: write PASS, STOP, or NOT RUN.
Operator ruling recommendation: write fund Decision-3, future Form-4 prereg, stop signal lanes, or defer.
Verification: list commands and results.
Known limits: no purchase, no parser, no returns, no PROGRAM_RECORD append.
```

## Self-Review

- Spec coverage: the plan covers no-alpha boundary, Decision-3 primary scoping, optional Form-4 Phase-0 matrix, structured operator ruling, Decision-2 separation, L/S record-count normalization, and verification.
- Placeholder scan: the plan avoids unresolved evidence placeholders; each generated artifact task includes a command that fails if placeholder markers remain in created artifacts.
- Type/path consistency: every generated file path is declared in File Structure and reused consistently in tasks.
- Scope check: this is one governance plan. It intentionally avoids code, parser work, vendor purchase, alpha prereg, and program-record append.
