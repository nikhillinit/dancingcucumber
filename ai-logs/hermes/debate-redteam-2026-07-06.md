# Red-team findings against opening Form-4 next

1. **Severity: HIGH**
   - **Claim attacked:** "Open the insider Form-4 lane now as the core H1 bet."
   - **Counter-argument:** The repo record does not yet make Form-4 an elected lane. The external repo review calls it a "PROPOSED NEW LANE" and says the operator must rank it against the standing queue before program review (`docs/superpowers/plans/2026-07-06-external-repo-review-ai-hedge-fund.md`). The 2026-07-04 program memo says "No further signal lanes open without an operator decision on the reframes above" and recommends validation harness, then diagnostics, then paid-data lever. The Decision-2 diagnostics plan says the operator formally picked Decision 2 on 2026-07-06, with a +60 day kill criterion. Opening Form-4 now risks treating "queued 2nd after L/S" as equivalent to an operator-elected prereg.
   - **Concrete change:** **Amend/reorder.** Replace "open Form-4 now" with "operator ruling on whether Form-4 outranks Decision-3 and active Decision-2 follow-through." Do not begin prereg or fixture work until that ruling exists.

2. **Severity: HIGH**
   - **Claim attacked:** "8 of 9 negatives do not degrade the Form-4 prior because it is an orthogonal information source."
   - **Counter-argument:** Orthogonal input is not enough. The failed record is not only "price transforms"; it includes repeated evidence that the 30-mega-cap floor destroys additivity and that keyless/free data paths hit structural ceilings. The original indicator scan says opportunistic insider alpha decayed post-2008 to roughly 30-40 bps/mo OOS and flags decay/crowding. The external review itself says insider literature is strongest in small caps while this repo's universe is 30 mega-caps. That directly weakens Form-4 in this repo's actual testbed. Orthogonality raises the prior relative to another price blend, but it does not reset the program prior.
   - **Concrete change:** **Amend.** State a degraded-but-nonzero prior. Require a pre-prereg mega-cap relevance sanity memo before any lane freeze, not merely a contamination check.

3. **Severity: HIGH**
   - **Claim attacked:** "Free keyless EDGAR makes Form-4 a cheap first next lane."
   - **Counter-argument:** The repo has reusable SEC mechanics, not a Form-4 lane. `apps/quant/advisor/source_integrity/edgar.py` parses Form 25 / 25-NSE delisting rows, 8-K items, and fair-access fetches. Repo search shows no tracked Form-4 parser, no ownership-XML schema, and no 10b5-1 exclusion path. The external review makes 10b5-1 exclusion a go/no-go condition, but the recommendation turns that into an informal pre-freeze check. That is free-data bias: choosing the free source before proving it carries the needed typed fields at the needed coverage.
   - **Concrete change:** **Add/reorder.** Insert a Phase-0 source-capability matrix before Form-4 prereg: sample filings, parser proof, ownership transaction fields, filing timestamp, transaction code, acquisition/disposition flag, 10b5-1 plan indicator or disclosed proxy, and coverage rate. Any missing mandatory field kills or downgrades the lane before alpha design.

4. **Severity: MED**
   - **Claim attacked:** "Pre-freeze contamination check: if >~90% of mega-cap Form-4 volume is 10b5-1-scheduled, the lane's prior collapses."
   - **Counter-argument:** The threshold is directionally right but underspecified and too late. "Volume" can mean transaction count, dollar value, share value, sale-only value, or insider-person events. The evidence base for the lane depends on separating opportunistic from routine trades; the indicator scan explicitly says routine-vs-opportunistic classification is the point even though a stronger "classification strictly required" claim was refuted. A vague >~90% check invites post-hoc denominator choice.
   - **Concrete change:** **Amend.** Freeze the contamination denominator before looking: by dollar value and by count, purchases and sales separated, mega-cap universe only, filed-date keyed, no alpha metrics. Use conservative kill if either primary denominator is dominated by routine/scheduled activity or if classification coverage is insufficient.

5. **Severity: HIGH**
   - **Claim attacked:** "Decision-3 scoping can run in parallel, design-only, while Form-4 opens."
   - **Counter-argument:** Decision-3 is not just parallel paperwork; it attacks the known structural confound. The broad-universe residual screen says the keyless ladder is exhausted and only a delisting-aware, point-in-time universe can settle the survivorship axis. The Phase-1 synthesis says no affordable native delisting-return source was found and the remaining rung was a non-alpha source-integrity diagnostic, not another signal run. Form-4 is a new unknown; Decision-3 addresses the known blocker.
   - **Concrete change:** **Reorder.** Promote Decision-3 scoping to the primary H1 research action: exact vendor, sample proof, cost, license, and the single frozen question it settles. Keep Form-4 as conditional after the source-capability matrix.

6. **Severity: MED**
   - **Claim attacked:** "Benchmark chosen for the signal's structure, not reflexively 'beat SPY'."
   - **Counter-argument:** Correct in principle, dangerous in phrasing. The broad-screen note allowed residual/absolute Sharpe for a market-neutral design because the structure justified it. Form-4 is cross-sectional/event-driven but not automatically exempt from floor comparators. After multiple negatives, benchmark choice is a high-risk post-hoc flexibility point. The harness design requires pre-committed decision rules, STOP default, and no goalpost moves.
   - **Concrete change:** **Amend.** Freeze a benchmark rationale and kill rule in the prereg before any sample measurement. If using event-study CAR, define the null, costs, holding window, turnover, and family comparator. "Signal-appropriate" cannot mean easier after the fact.

7. **Severity: MED**
   - **Claim attacked:** "Low-turnover event-driven signals are structurally more cost-robust, so Gate-1's cost lesson favors Form-4."
   - **Counter-argument:** This imports a lesson from the L/S failure without proving it applies. Form-4 may be low turnover on 30 mega-caps, but if the effective opportunistic subset is sparse after 10b5-1/routine filtering, the binding problem becomes sample power and event concentration, not costs. The Decision-2 diagnostics plan already distinguishes report-only value from signal claims; sparse event studies can produce narrative-friendly but underpowered positives.
   - **Concrete change:** **Add.** Before any alpha prereg, require an event-density feasibility check with no returns: event counts by year, ticker concentration, buy/sell split, routine/scheduled exclusion coverage, and minimum effective events. If too sparse, close or park without alpha measurement.

8. **Severity: MED**
   - **Claim attacked:** "H2 conditional exit ramp: if Form-4 closes and Sept-4 evaluation kills, then decide paid data or pivot to validation harness + negative record."
   - **Counter-argument:** This frames the harness as an escape hatch after more failed signal searching, but the validation-harness-as-asset design was already elected on 2026-07-05 as an internal governance package. Making it conditional on two additional failures undervalues the asset already chosen and creates a psychological permission slip to take another signal shot first.
   - **Concrete change:** **Amend.** Decouple harness continuation from Form-4 and Decision-2 outcomes. The harness remains active infrastructure. The future binary should be "fund Decision-3 or stop opening signal lanes," not "discover harness value later."

9. **Severity: LOW**
   - **Claim attacked:** "Decision-2 evaluation proceeds untouched."
   - **Counter-argument:** This survives, but the recommendation should avoid crowding it out. Decision-2 has a frozen +60 day usage kill criterion and is a report-only product surface, not a research side quest. Opening Form-4 immediately risks splitting solo-operator attention before the diagnostics lane gets real usage evidence.
   - **Concrete change:** **Survives with guard.** Keep the Sept-4 evaluation untouched and explicitly cap Form-4 prework to design-only until Decision-2 has enough usage runway.

10. **Severity: LOW**
    - **Claim attacked:** "Park disclosure tone."
    - **Counter-argument:** This survives. The SESTM Phase-0 record says the corpus wall is structural for the published hypothesis, and issuer-disclosure tone is a cold new hypothesis, not a continuation. There is no reason to rank it above paid-data scoping or a Form-4 capability matrix.
    - **Concrete change:** **Survives.** Keep parked unless the operator deliberately elects it under a new prereg.

## Strongest different next step

Make **Decision-3 paid-data scoping** the next research action, not Form-4. No purchase yet. Produce one page that names the cheapest legally usable delisting-aware PIT vendor, sample-file proof, native delisting-return fidelity or disclosed absence, small/mid coverage, license constraints, integration estimate, and the single frozen question it can settle. This is higher EV than opening Form-4 because it targets the already-known confound from the broad residual screen and Phase-1 synthesis. Form-4 is free but still unproven on the repo's mega-cap universe and lacks existing parser/10b5-1 infrastructure.

## Bias audit

- **Lane-hopping:** Present. The recommendation jumps from a closed L/S futility screen into a new signal lane before forcing an operator ruling on whether signal lanes should remain open.
- **Queue-order anchoring:** Present. "Queued 2nd" from the L/S plan is being treated as stronger authority than the program memo's "no further signal lanes without operator decision" and the already-picked Decision-2 diagnostics lane.
- **Free-data bias:** Strong. Form-4 is attractive because EDGAR is free, but the binding uncertainties are classification coverage, 10b5-1/routine separation, mega-cap event density, and post-2008 decay.
- **Premature pivot-to-meta:** Mixed. The harness-as-product reframe is not premature because it was already elected as an internal asset. The bias is using it as a deferred escape hatch to justify one more signal lane first.

## Elements that survive

- L/S reversal lane remains CLOSED forever. No reruns, threshold changes, or Gate-2 work.
- Decision-2 September 2026 usage evaluation remains untouched.
- Disclosure tone stays parked.
- Form-4 remains a plausible cold candidate only if promoted through an operator ruling and a no-alpha source-capability matrix.
- The Form-4 prereg, if ever opened, must freeze cost model, benchmark/null, event-density rule, 10b5-1/routine handling, and STOP condition before any return measurement.
- Decision-3 scoping survives and should be promoted, not treated as secondary busywork.
