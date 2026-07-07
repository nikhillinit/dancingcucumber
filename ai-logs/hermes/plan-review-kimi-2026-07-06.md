# Independent plan review — Kimi voice

Scope: Review the frozen Decision-3 / Form-4 no-alpha ruling plan and its Gate-1 result; recommend the pending operator election. No metrics computed; no tracked files modified.

---

## 1. Plan-quality findings (new issues not raised in red-team / blind round)

### Gate-2 Form-4 Phase-0 criteria

1. **HIGH — Deduplication rule is under-specified for a hard count gate.**
   The plan defines one event as "same issuer + same insider + same accession + same direction." It does not say which EDGAR identity keys map to "issuer" (issuer CIK? ticker? central index key of the subject company?) or "insider" (reporting owner CIK? RPTOwnerCIK? name?). It also omits Form 4/A amendments: an amendment cancels or replaces a prior filing, so naive counting will double-count or under-count events. Without a keying scheme and amendment handling, the >=100 / >=20 / >=20 / >=12 thresholds are not reproducible.
   **Concrete change:** Add a subsection that pins the exact key fields (e.g., `issuerCik` + `rptOwnerCik` + `accessionNumber` + `transactionCode` direction group) and mandates that Form 4/A filings replace their original accession before any count.

2. **HIGH — Concentration caps are likely breached by construction on the 30-mega-cap floor, but the plan treats them as empirical.**
   Prior debates noted mega-cap coverage weakness generally, but did not flag that the 20% issuer / 35% dollar / 10% insider caps are almost certain to bind on a small mega-cap universe. A few issuers (e.g., TSLA, AAPL, AMZN) and a handful of repeat-reporting insiders will dominate the event stream. Treating this as a discoverable matrix outcome invites a STOP that could have been pre-declared.
   **Concrete change:** Add a pre-matrix feasibility note: if the universe is fixed at ~30 mega-caps, the concentration caps should be evaluated against a simulated or prior-known event distribution before Gate 2 is authorized. If the caps are expected to bind, Gate 2 should default to STOP unless the operator explicitly widens the universe under a new prereg.

3. **HIGH — "Open-market" transaction filter is not pinned to Form-4 transaction codes.**
   The plan repeatedly references "open-market" purchases and sales but never lists the transaction codes that qualify (typically Form 4 code `P` for open-market or private purchase and `S` for open-market sale). Codes like `A` (grant), `G` (gift), `F` (payment of exercise price), `M` (conversion of derivative), `J` (other), and `W` (will or intestate acquisition) must be excluded or explicitly routed to `uncertain`. Without this filter, counts are not comparable across parsers.
   **Concrete change:** Add a mandatory code map in the criteria file: include `P`/`S` as non-10b5-1 open-market (subject to 10b5-1 flag), exclude `A/F/G/J/M/W`, and route ambiguous or missing codes to `uncertain`.

4. **MED — 10b5-1 flag field is not tied to a specific SEC schema element.**
   The plan says "2023+ plan flag or plan-adoption disclosure is usable" but does not name the EDGAR ownership XML element (e.g., `transactionCoding/transactionCode` with a footnote, or the newer `is10b5-1` indicator introduced by the 2023 amendments). This leaves the matrix open to inconsistent interpretation.
   **Concrete change:** Cite the specific SEC schema/element expected for the checkbox era and require that any parser proof demonstrate extraction of that exact element.

5. **MED — Dollar denominator is fragile to missing, zero, or estimated prices.**
   The plan defines dollar denominator as "gross disclosed transaction value where price and shares are available." Form 4 transactions are often reported with `transactionPricePerShare` of `$0` (grants), "deemed execution" prices, or footnoted ranges. The plan does not say how to treat these cases, yet the <=15% uncertain cap applies separately to purchase-dollar and sale-dollar denominators.
   **Concrete change:** Add a missing-price rule: if either price or shares is missing, zero, footnoted as estimated, or belongs to an excluded code, exclude the transaction from dollar denominators and count it toward the `uncertain` bucket. State the minimum precision required for a transaction to be "usable" in dollar terms.

6. **MED — "Representative sample filings" for parser proof is undefined.**
   The criteria require "at least three representative sample filings" but do not define what representation means (e.g., one purchase, one sale, one with a 10b5-1 flag; or coverage across years 2023–2026). This invites cherry-picking.
   **Concrete change:** Define a representative sample as: (a) one open-market purchase with a 10b5-1 flag, (b) one open-market sale without a 10b5-1 flag, and (c) one filing containing both purchase and sale transactions, all drawn from the 30-mega-cap advisor universe inside the frozen window.

7. **LOW — The fixed end date 2026-06-30 is ratified but creates an implicit staleness issue if Gate 2 is deferred.**
   The plan says no roll-forward, which is correct, but does not state how the matrix should label data freshness. If the matrix is produced weeks after 2026-06-30, the ruling package should disclose that the evidence is frozen as of that date.
   **Concrete change:** Add a metadata line to the matrix: "Data coverage frozen through 2026-06-30; no roll-forward without a new criteria file."

### Gate-3 ruling-package structure

8. **MED — Ruling table does not disambiguate "NOT RUN" from "STOP" for the Form-4 matrix.**
   The operator decision table allows Form-4 matrix verdicts of PASS, STOP, or NOT RUN, but the surrounding text does not state when NOT RUN is the correct label. If Gate 2 is skipped because Decision-3 already STOPped and the operator does not want to spend a fresh hypothesis slot, the matrix is NOT RUN, not STOP. If Gate 2 is attempted and fails a pinned criterion, it is STOP. Blurring the two weakens the negative record.
   **Concrete change:** Add a sentence in the ruling package: "NOT RUN means Gate 2 was deliberately not opened; STOP means Gate 2 was opened and failed a pinned criterion."

9. **MED — Independence of Decision-3 and Form-4 outcomes is not explicit in the ruling table.**
   The table lists "Stop opening signal lanes" as triggered when "Decision-3 STOPs, Form-4 STOPs or is not worth a fresh slot," which could be read as implying a Decision-3 STOP automatically closes Form-4. The two gates answer different questions (paid PIT data vs. insider-event feasibility) and should remain logically independent unless the operator explicitly couples them.
   **Concrete change:** Add a footnote to the evidence-inputs table: "Decision-3 STOP does not mechanically stop Form-4; each matrix is evaluated under its own frozen criteria. The default operator recommendation, however, treats both STOP/NOT RUN as evidence against opening a new signal lane."

10. **LOW — The "Defer" option lacks a maximum review horizon.**
    The plan allows deferral with a review date and exact missing evidence, but does not bound how far out that date may be. Without a horizon, deferral can become a soft STOP that avoids recording a clean decision.
    **Concrete change:** Add a constraint: deferral review date must be no more than 90 days from the ruling date unless the operator records a specific external trigger (e.g., vendor pricing response, SEC schema release).

---

## 2. Recommendation on the pending operator election

**GO TO GATE 3 with Form-4 = NOT RUN.**

Decision-3 already STOPped under frozen criteria: the only source with native delisting-return fidelity (CRSP/WRDS) is access-locked behind institutional licensing with no public pricing or individual path, while every ceiling-fitting vendor carries the same delisting-return omission cited in Step-0. That was the predeclared honest prior and it was confirmed. Running Gate 2 now would be a lateral move into a second unproven lane before the operator has formally ruled on the central implication of Decision-3 — that the survivorship confound cannot be settled at the current budget ceiling. Form-4 remains a cold hypothesis: it lacks parser infrastructure, has unresolved 10b5-1/routine classification mechanics, is likely to breach concentration caps on a 30-mega-cap universe, and was ranked below paid-data scoping by both the red-team and blind allocator. The correct next action is to package the Decision-3 STOP into a clean operator ruling that stops opening signal lanes, preserves the nine-plus-one negative record, and keeps Decision-2 diagnostics on its frozen 2026-09-04 evaluation track. Form-4 should be marked NOT RUN, not STOP, because Gate 2 was deliberately not opened. If the operator later wants to revive Form-4, that should require a fresh hypothesis slot and a new prereg, not a continuation of this plan.

---

## 3. Gate-1 STOP verdict derivation

The Gate-1 STOP verdict is **sound** given the frozen criteria.

- The matrix correctly applied the Step-0 carry-over: QC / Sharadar / Norgate delisting-return FAILs were not re-litigated without documented vendor change.
- CRSP/WRDS was evaluated only on cost/access, per the delta scope, and the evidence cited (no public pricing, institutional-only access, no individual subscription path) supports a FAIL against the <=$1,000 + <=$100/month ceiling.
- The decision rule was applied literally: PASS requires at least one source clearing every criterion; no source did.
- The verdict explicitly named the binding failures and did not drift into alpha metrics, threshold revision, or ceiling relaxation.

No unsound derivation was found. The only residual concern is that the matrix's "Carried" cells rely on prior repo notes rather than fresh official vendor documentation, but the criteria explicitly allow "Existing repo notes cited by path" as allowed evidence, and the 13-day interval since Step-0 makes silent vendor change implausible. The STOP acceptance recorded by the operator on 2026-07-06 is therefore consistent with the evidence.
