# 2026-06-23 — Phase-1 direction: roleplay → debate → synthesis (decision aid, report-only)

> Follows `2026-06-23-broad-universe-residual-screen-result.md` (verdict INCONCLUSIVE) and memory
> `[[blend-futility-residual-alpha]]`. This is a structured operator-decision aid — NOT a gate run.
> Holdout untouched, no prereg hash, floor still DEV_FAILED, `tools/run-floor.mjs --enforce` exit 1.

## Decision framed
**The keyless residual-screen ladder is exhausted at INCONCLUSIVE. What is the next move:**
**(A) acquire keyed delisting-aware data for one discriminating test, (B) pivot the design**
**(long-short / small-mid-cap / news), or (C) stop and write the negative result?**

Success = a choice that (i) is decoupled from the non-discriminating GREEN, (ii) protects the
blinded holdout + MinBTL multiple-testing budget + prereg discipline, (iii) takes at most one clean
decisive shot, (iv) has a pre-committed stop rule so a negative is a *conclusion*, not an invitation
to keep grinding universes.

## Roles (4)
**1. Empiricist — run the discriminating test.** INCONCLUSIVE is the worst state: neither go nor
stop. The cheap ladder ruled out pure noise but structurally cannot reach contrarian-survivorship,
deflation, holdout, or the unobservable short leg. Only a keyed, delisting-aware, point-in-time
panel breaks the tie. *Vulnerability:* assumes an affordable delisting-aware sample exists — if not,
the clean test cannot be run either.

**2. Rigor Guardian — protect the discipline.** Any next test must sit on its OWN immutable prereg
surface (mirror `FundamentalCandidatePreReg` / `ValidationPreReg` — NEVER bolt fields onto
`PreRegConfig`), with its OWN random-null floor + DSR bar + blinded holdout, frozen before any run.
The untouched holdout and the multiple-testing budget ARE the project's asset; a silent string of
universes is p-hacking. *Vulnerability:* a SINGLE, separately-surfaced, pre-registered test is clean
science — it does not touch the protected holdout or inflate the main MinBTL budget. The Guardian
can BLESS one such shot. Isolation + freeze, not refusal, is what the discipline requires.

**3. Alpha Strategist — pivot the design to where alpha actually is.** Don't re-run a long-only
screen on a nicer universe — the *instrument* was wrong. The structurally untested design is
market-neutral (beta + dollar-neutral) long-short on a small/mid-cap-inclusive, delisting-aware
panel — where the reversed in-sample IRs (+0.32..+0.41) and the SESTM news finding point. A long-only
proxy cannot see the short leg, so it will again be non-discriminating. *Vulnerability (must own):*
those reversed IRs come from the LEAST trustworthy numbers — survivorship-inflated contrarian
families (value worst→best is survivorship's fingerprint). Long-short trades exactly the extreme
deciles where delisting concentrates, so its data-quality bar is HIGHER, not lower. The design that
answers the right question is the one most sensitive to the data defect.

**4. Resource Steward / KISS — stop or timebox.** Five+ DEV_FAILED results (advisor-v1, Plan 4,
WS4-B, WS3D-C, the floor) are a body of evidence; P(real edge | five negatives) is low. The honest
move may be to STOP, write the negative, close the residual-alpha lane — unless ONE cheap, decisive,
timeboxed shot has clearly positive EV. A solo dev's scarcest resource is time; do not build the
full automated Phase-1 infra before a one-shot manual test even exists. *Vulnerability:* if a
genuinely decisive test is cheap and pre-committed with a stop rule, refusing it is *also* a bias
(premature closure). One clean shot can clear the EV bar.

## Debate
**Dimensions:** discriminating power · cost/EV · survivorship/data-quality exposure · discipline/budget.

**The hinge clash — Rigor Guardian vs Resource Steward are NOT a restraint bloc.**
- *Rigor Guardian:* "I can bless ONE pre-registered test on its own surface — clean, isolated, does
  not touch the holdout or the main budget. My discipline is satisfied by isolation + freeze."
- *Resource Steward:* "Clean ≠ worth it. Does one test clear the EV bar for a solo dev? The data
  costs money; even a pass is 'hypothesis, not edge.' Show me the EV."
- *Crux / resolution:* EV is positive ONLY if all three hold — (a) the spike finds an affordable
  delisting-aware PIT sample, (b) the test is designed to answer the RIGHT question (market-neutral
  long-short, so pass/fail is informative), (c) a hard stop rule converts a negative into a CLOSED
  lane (bounded cost, terminal value either way). If any fails → Resource Steward wins → stop.

**Empiricist vs Alpha Strategist crux:** Empiricist wants to run *a* test; Strategist insists it
must be long-short or it is another non-discriminating long-only result. Strategist wins on design —
but must own that long-short raises the delisting-aware data bar (extreme deciles = where delisting
concentrates), so "delisting-aware" must be STRICTER here than for a long-only screen.

**Irreducible tradeoff (named, not papered over):** the design that answers the right question
(market-neutral long-short on small/mid-cap) is precisely the design most sensitive to the
missing-delisted-names defect. There is no cheap middle — "long-short on the keyless survivor panel"
is the *worst* of both worlds: the appendix's +0.10 structural-null floor plus contrarian
survivorship inflation would dominate the result. Either get genuinely delisting-aware data (and pay
for it) or do not run it.

**The crux that decides everything:** *is a point-in-time, delisting-aware sample obtainable at a
cost a solo dev should pay?* This is empirical and cheap to check.

## Synthesis — Conditional Strategy; STOP is a first-class branch
**Do NOT commit to the full Phase-1 build, and do NOT silently stop. Run ONE cheap, in-lane,**
**keyless data-availability spike — with BOTH downstream branches and the stop rule pre-committed**
**BEFORE it runs.** (This is the skill's Conditional-Strategy pattern, NOT the "we need more info"
anti-pattern: both branches are specified now.)

**Step 0 — data-availability spike (now; in-lane; keyless network scoping; no spend).**
Scope the cheapest delisting-aware, point-in-time, small/mid-cap-reaching source for a solo dev:
- Norgate Data (survivorship-bias-free US equities incl. delisted; PIT index constituents)
- Sharadar SF1 + SEP/SFP via Nasdaq Data Link (delisted tickers retained; PIT)
- Tiingo / Polygon.io (delisted coverage; cheaper tiers)
- CRSP / WRDS = gold standard but academic/expensive → out of scope for a solo dev.
Output: the cheapest source delivering (i) delisted names present, (ii) point-in-time membership,
(iii) small/mid-cap reach — and its price.

**Branch A — affordable delisting-aware PIT sample exists (≤ operator-set threshold):**
take ONE pre-registered shot. Design FROZEN before any data touches the test:
- market-neutral (beta + dollar-neutral) long-short
- small/mid-cap-inclusive, delisting-aware, point-in-time panel
- its OWN immutable prereg surface (mirror `FundamentalCandidatePreReg`; do NOT touch `PreRegConfig`)
- blinded holdout + ledger; its OWN random-null floor AND DSR bar
- judged on residual / absolute Sharpe — NOT "beat SPY"
- one shot, timeboxed; the MinBTL N is spent on THAT surface only.
Pass = clears the null floor (necessary) AND DSR AND blinded holdout (sufficient) → a hypothesis
worth a Phase-1 build — STILL not a tradable edge. (Null-clear alone is the *weaker* bar; it already
failed to save trend's identical 0.828 / +0.41 under deflation + holdout.)

**Branch B — no affordable delisting-aware PIT sample:** STOP. Write the negative, close the
residual-alpha lane. The keyless ladder is exhausted and the discriminating test is out of reach at
acceptable cost — that is a conclusion, not a defeat.

**Pre-committed STOP rule (freeze WITH the design):** if the one shot does not clear its own null
floor AND DSR AND blinded holdout, the residual-alpha / long-short lane is CLOSED — write the
negative; do NOT iterate universes. This is what makes the spend EV-positive: bounded cost, terminal
value either way.

**What the synthesis explicitly refuses:**
- a "balanced GO" = long-short on the keyless survivor panel (worst of both worlds);
- folding SESTM news in now (separate lane: needs a news fixture + keys; orthogonal; decide later);
- building the automated Phase-1 infra before the one manual shot exists;
- treating a null-clear as an edge.

**Monitored assumptions:** (1) the spike's cost finding — if the cheapest delisting-aware source
exceeds the operator's threshold, Branch B fires automatically; (2) confirm the chosen source
actually contains delisted small/mid-cap names (the short leg), else the long-short test is still
blind.

## Validation (rubric)
- **All four perspectives honored, none dismissed:** Empiricist gets the test (conditionally);
  Rigor Guardian gets isolation + freeze and blesses one shot; Alpha Strategist gets the long-short
  design and owns the higher data bar; Resource Steward gets the EV gate + the stop rule + STOP as a
  first-class branch.
- **Real tension surfaced:** Rigor Guardian vs Resource Steward forced to clash on EV (not a
  restraint bloc); the irreducible tradeoff (right-question design = most data-sensitive) is named.
- **GO bias pre-empted:** STOP is a named branch with an automatic trigger.
- **Anti-manufacturing check:** the synthesis converges with the screen note's own §Decision
  (separate prereg surface, market-neutral long-short, blinded holdout, freeze-before-run, operator
  greenlight). Convergence = the framework *found* the answer, did not invent a new one.

## Two-voice deliberation (Claude × Hermes/Codex gpt-5.5)
A second, independent voice was dispatched via Hermes to adversarially pressure-test Claude's
synthesis (task: `ai-logs/hermes/phase1-deliberation.md`; response:
`ai-logs/hermes/runs/hermes-deliberation-response.md`; HEAD unchanged, suite green, no commit).
Verdict: **AGREE-WITH-AMENDMENTS.** The two voices converge on the spine (spike-first, both branches,
hard stop rule, one shot, separate prereg surface, no news-folding) — but Hermes sharpened three
things Claude under-weighted. All three strengthen the plan; none contradict it.

**Amendment 1 — the hinge is wrong.** Claude's hinge was "does an *affordable* source exist."
Hermes's hinge: "does a source enable a *mechanically and legally valid decision*." Affordability is
necessary, not sufficient. A cheap feed may retain delisted *symbols* yet still fail the actual
requirement: **delisting returns present in the tested total-return stream**, plus point-in-time
rankable membership, corporate-action adjustment, and licensing that permits the use. Symbol
retention ≠ defect repair.

**Amendment 2 — the cheaper discriminating test Claude missed is data-QC, not alpha.** Before
spending the one alpha shot, run a **source-integrity diagnostic**: prove the chosen source actually
repairs the survivor-panel defect — i.e., delisted losers *populate the bottom / cheap deciles*
in-sample — *before reading any alpha metric*. If the source doesn't repair the defect, KILL there;
don't burn the alpha attempt. This attacks the survivorship confound at the data layer, where it is
cheaper and more decisive than at the alpha layer.

**Amendment 3 — the spike is a kill-biased capability matrix, default STOP, one session.** Not a
vendor reading list: a one-page PASS/FAIL matrix with sample-file proof, price, license constraints,
required fields, and integration burden (≤ ~3 focused dev-days). The lane-kill rule expands to
include **source-integrity failure**, not just alpha-metric failure.

**Residual clash (Hermes's own steelman):** over-kill-bias could prematurely close the only
discriminating residual-alpha path (the panel *did* clear the random-null decisively). *Resolution:*
the data-QC diagnostic IS the safeguard — kill-bias is calibrated to **data validity**, not to
pessimism about alpha. A valid, defect-repairing source still earns the alpha shot; the bias only
bites when no source can make a valid decision possible.

**News lane:** both voices agree — do NOT fold SESTM news into this shot. If the operator is
allocating real spend broadly, news gets a SEPARATE EV comparison against the price long-short lane.

## Amended recommendation (cross-examined — supersedes the solo synthesis above)
1. **Step 0 — kill-biased source-capability spike** (now; in-lane; keyless; no spend; one session).
   Output = a one-page PASS/FAIL matrix, columns: vendor · cost · sample-file proof · delisted
   tickers retained · **delisting returns in the total-return series** · PIT tradable-universe /
   membership history · small/mid-cap coverage · corporate-action adjustment · PIT fundamentals (if
   value in scope) · short/borrow data (or declared short-proxy limitation) · license constraints ·
   integration estimate · PASS/FAIL. Default posture: STOP.
2. **Branch A** (≥1 source clears ALL must-have fields under the operator's cost threshold AND
   ≤ ~3 dev-days integration): (a) run the **source-integrity diagnostic** first (delisted losers
   populate bottom/cheap deciles); if it fails → kill. (b) If it passes → ONE pre-registered
   **market-neutral (beta + dollar-neutral) long-short** shot on its own immutable prereg surface
   (mirror `FundamentalCandidatePreReg`; never touch `PreRegConfig`), small/mid-cap-inclusive,
   blinded holdout + ledger, own null floor AND DSR bar, judged on residual/absolute Sharpe. Pass =
   Phase-1-build hypothesis after prereg + data-QC evidence — NOT a production/tradable claim.
3. **Branch B** (no source clears the matrix under threshold): **STOP**, write the negative, close
   the residual-alpha lane.
4. **Lane-kill (any one → close lane, write negative, NO universe iteration):** (i) no source clears
   the must-have matrix under cost + integration ceiling; (ii) source-integrity diagnostic fails;
   (iii) the one market-neutral shot fails its null floor / DSR / blinded holdout.
5. Do NOT favor value / mean_reversion (survivor-inflated reversal) as the lead evidence for the
   shot; prefer families less mechanically advantaged by missing delisted losers, OR gate on the
   source-integrity diagnostic first.

## The single operator input required
Two numbers + a go/stop: (1) the **cost threshold** for an acceptable delisting-aware PIT source
(the solo-dev wallet); (2) the **integration-effort ceiling** (Hermes proposes ≤ ~3 focused
dev-days); (3) run the Step-0 spike now, or stop on principle. Everything downstream — both
branches, the data-QC gate, and the three-way lane-kill — is pre-committed here.

## Step-0 result (EXECUTED 2026-06-23) — source-capability matrix
Three parallel keyless research sweeps, 8 vendors, cited to vendor docs (raw returns in session).
Decisive column = (b) **delisting RETURNS in the total-return stream** (terminal acquisition /
bankruptcy return), NOT mere symbol retention.

| Vendor | Cost (solo-dev) | (b) delisting returns in stream | (c) PIT membership | (d) small/mid | PASS/FAIL |
|---|---|---|---|---|---|
| **CRSP / WRDS** | $5k–70k/yr, academic-gated | **Y — only true YES (`DLRET`)** | Y | Y | DATA PASS / **ACCESS FAIL** |
| **QuantConnect** | **$0 cloud / ~$24/mo** | PARTIAL (last-price auto-liquidation, not `DLRET`) | PARTIAL (PIT universe Y; index-history N/verify) | Y | CONDITIONAL — cheapest valid-enough |
| **Sharadar** | low-hundreds/yr (verify) | PARTIAL (acq captured; bankruptcy stub not) | PARTIAL (S&P500 only; self-build small/mid) | Y | CONDITIONAL — best PIT value fundamentals |
| **Norgate** | ~$630/yr (verify) | PARTIAL (same) | **Y (Russell+S&P off-the-shelf)** | Y | CONDITIONAL — best ready PIT universe |
| **EODHD** | $59.99/mo | UNCLEAR→N (`DelistedDate`=last trade) | Y (S&P500 from 2000) | Y | FAIL |
| **Polygon** | $29–199/mo | UNCLEAR→N | **N** | Y | FAIL |
| **Tiingo** | $30/mo | UNCLEAR (CRSP-method claim = split/div only) | **N** | Y | FAIL |
| **Intrinio** | $150/mo | UNCLEAR (no method; delisted prices only 2007) | UNCLEAR | Y | FAIL |

**Decisive finding:** *no affordable source NATIVELY supplies verified delisting returns in the
return stream.* The only one that does — CRSP — is unreachable for a solo non-academic. Every
affordable source captures **acquisition** delistings (last traded price ≈ deal value) but omits the
**bankruptcy / performance terminal loss** (CRSP's ~−30%/−100% `DLRET`) — exactly the
downside-delisting events the survivorship axis (the short leg, contrarian longs → 0) most depends on.

**Cost is NOT the binding constraint** — QuantConnect clears it at ~$0. The binding constraints are
(b) delisting-return *fidelity* and PIT-fundamentals *quality* (QC's Morningstar PIT is
vendor-caveated). The pre-set cost threshold is therefore moot; the real fork is rigor, not wallet.

### The sharpened fork (supersedes the cost-threshold question)
- **Branch B — STOP (kill-biased default; our pre-committed rule fires).** "(b) native + verified" is
  cleared by NO affordable source → close the residual-alpha lane, write the negative. Honest,
  pre-committed, consistent with the 5+ DEV_FAILED body of evidence.
- **Branch A′ — eyes-open GO on a valid-ENOUGH source.** QuantConnect ($0–24, cheapest) or Sharadar
  (best value fundamentals) + a **self-engineered delisting-return overlay** (Shumway haircut ~−30%
  performance / −100% bankruptcy, keyed on the delist-reason field both carry). MANDATORY first: the
  **source-integrity diagnostic** (Amendment 2) — after the overlay, do delisted losers populate the
  bottom/cheap deciles? If not → kill (≈$0 spent, bounded dev-days). The overlay pushes toward/over
  the ~3-dev-day ceiling, and the test's validity then rests on the overlay being correct — a
  disclosed, self-built repair of the exact thing CRSP would supply.

**Claude's recommendation: lean Branch B (STOP).** The pre-committed kill-biased rule fires (no
affordable native (b)); Branch A′ asks me to hand-build, at the survivorship axis, the precise
fidelity CRSP exists to provide, then trust my own overlay — a thin, circular-risk edge on top of 5+
negatives. The single defensible GO is QuantConnect-free + overlay + the source-integrity diagnostic
as a hard pre-alpha kill: near-zero dollars, eyes-open on fidelity, bounded to ~3 dev-days. **This is
the operator's call** — rigor vs one cheap shot — and it is now concrete, not abstract.

## Fork deliberation — second Hermes voice (2026-06-23)
Dispatched Codex to weigh STOP vs the QC-overlay GO on the matrix result (task
`ai-logs/hermes/fork-deliberation.md`; response `…/runs/hermes-fork-deliberation-response.md`; HEAD
unchanged, suite green, no commit). Verdict: **GO-WITH-AMENDMENTS** — but the "GO" is NOT the
overlay+alpha shot; it is a **non-alpha data-falsification diagnostic ONLY.**

**The missing middle Claude collapsed.** Claude framed the fork as binary (STOP vs overlay-GO). Codex
found the rung between them: a QC-free **source-integrity diagnostic** answering "can this source even
EXPOSE the survivorship axis?" — separate from, and prior to, "can a self-built overlay PRICE it
correctly?" (the latter unverifiable without CRSP). The diagnostic is ~$0, one session, and **forbids
all alpha metrics**, so it cannot become a sunk-cost interpretation trap.

**Where the two voices land:**
- Both agree the overlay-dependent **alpha shot is NOT warranted now**; default conclusion = STOP.
- Improvement Claude absorbs: the diagnostic rung **dominates pure-STOP** — ~same cost, strictly more
  information. Fail → STOP is *evidentially stronger* (QC lacks the adverse-delisting mass, not merely
  "no native DLRET"). Pass → earns only a *separate* preregistration gate, never a direct alpha run.
- Codex's own steelman (preserved): the diagnostic can become a **loophole** that delays a warranted
  negative; even a pass leaves the alpha test overlay-dependent on the one field no affordable vendor
  verifies. Guardrail: pre-register the diagnostic + thresholds BEFORE running, forbid alpha readout,
  default = STOP. If the operator doubts the discipline to not rationalize a pass, **pure-STOP is
  cleaner.**

## Merged recommendation (cross-examined — supersedes "lean STOP")
**Run ONE pre-registered, QuantConnect-free, NON-ALPHA source-integrity diagnostic — or pure-STOP.**
Diagnostic spec (pre-commit BEFORE running; alpha metrics / Sharpe / PnL / family comparisons
FORBIDDEN in the artifact):
- Export the small/mid-cap eligible universe + delisting events over the dev period.
- Classify delistings: acquisition / bankruptcy-performance / unknown (QC event+reason fields only).
- Count delisting events by pre-delist rank bucket on the candidate tails (bottom/cheap deciles).
- Report coverage gaps: unknown-reason rate, missing-metadata symbols, names unmappable to PIT
  tradable-universe membership.
- **Pre-committed kill thresholds (ANY fail → STOP, write the negative):** ≥50 classified
  bankruptcy/performance/severe-loss delistings in the dev sample; ≥90% of events mappable to a
  terminal class; ≥2× concentration of adverse delistings in bottom/cheap tails vs top/expensive.
- **Pass ≠ alpha run.** A clean pass triggers only a SEPARATE preregistration of the Shumway-overlay
  spec + disclosed sensitivity bands + null floor + DSR + blinded holdout + no-iteration stop, for
  exactly one market-neutral shot — brought back to the operator as a fresh go/no-go.

Logistics if GO: needs a QuantConnect account (operator) + a LEAN diagnostic algorithm; Codex is
no-network so it cannot run QC's cloud — the LEAN run is operator/Claude-side, NOT a Hermes dispatch.
