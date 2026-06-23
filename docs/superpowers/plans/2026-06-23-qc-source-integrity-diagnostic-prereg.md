# PRE-REGISTRATION — QuantConnect source-integrity diagnostic (NON-ALPHA, report-only)

> **Status: DRAFT — not yet frozen.** This document must be **committed (frozen) BEFORE** the
> diagnostic is run. The thresholds in §5 are immutable once frozen; any post-hoc change voids the
> run and forces STOP. Separate surface — does NOT touch `PreRegConfig`, the floor, or the holdout.
> Lineage: `2026-06-23-phase1-direction-roleplay-debate-synthesis.md` (two-voice fork → diagnostic rung).

## 0. One-line
Falsify-or-validate that QuantConnect's survivorship-bias-free US equity dataset carries **enough
auditable adverse-delisting mass on the small/mid-cap cross-sectional tails** to justify a later
self-built delisting-return overlay. **This is a data-QC test, NOT an alpha test.**

## 0a. Pre-freeze revisions (2026-06-23, before ANY data touched — domain review)
Three false-FAIL traps in the v1 draft, patched on domain reasoning (not on data), recorded for
transparency:
1. **Rank-timing dropout:** distressed names decay out of the cap/liquidity band *before* delisting →
   ranking "1 month before last trade" null-ranks/drops them, starving the mass count. **Fix:** rank
   as-of the last date the name was a valid eligible member (§4.4).
2. **Negative book-value contamination:** wiped-out equity → negative B/P → naive numeric sort places
   the worst bankruptcies in the *cheapest* bin, corrupting the concentration metric. **Fix:** gate on
   the momentum key only (clean/monotonic); book-to-price is reported, non-gating, with negative-book
   excluded + bucketed separately (§4.4–4.5, §5.3).
3. **Over-strict mappability:** ≤10% unknown is unrealistic for non-CRSP vendor metadata → false-FAIL
   on sloppiness. **Fix:** soften to ≤15% AND treat unknown-reason events as non-adverse (conservative)
   so unknowns can only hurt a PASS (§4.3, §5.1–5.2).

## 1. Why this exists (the binding fact)
The Step-0 matrix found **no affordable source natively supplies verified delisting *returns* in the
total-return stream**; only CRSP does (`DLRET`), and CRSP is access-locked for a solo non-academic.
Every affordable source (QC, Sharadar, Norgate) captures *acquisition* delistings but omits the
*bankruptcy/performance terminal loss* — exactly the survivorship axis a market-neutral long-short
depends on. Before spending dev-time hand-building a Shumway overlay (the unverifiable repair), this
diagnostic answers the prior, cheaper question: **can the source even EXPOSE the axis?**

## 2. Scope & hard exclusions
- **NON-ALPHA.** The artifact MUST NOT contain: returns, Sharpe, info ratio, PnL, portfolio
  construction, family/strategy comparisons, or any performance readout. Producing any of these voids
  the run. (This is the anti-sunk-cost guardrail: a "pass" cannot be rationalized into a result.)
- Report-only. No gate logic, no holdout touch, no prereg hash on `PreRegConfig`. Floor unchanged.

## 3. Data & universe (PIT, pre-committed)
- **Source:** QuantConnect US Equity Security Master + corporate-actions/delisting events + Morningstar
  market-cap for ranking. QC event/reason fields ONLY for classification (no external delist source).
- **Eligible universe (per date, point-in-time, no look-ahead):** US common equity; exclude ETFs/funds/
  ADRs; tradability filter ≥ $1M 21-day median dollar-volume; **small/mid-cap band = market-cap
  percentile 20–90** (drops mega-cap top-decile-ish and the most illiquid micro-cap noise).
- **Dev period:** 2015-01-01 → 2023-12-31 (matches prior fixture span). No period after 2023-12-31 is
  examined (PIT hygiene; the future window stays untouched for any later alpha prereg).

## 4. Procedure
1. Reconstruct the eligible universe per month over the dev period (PIT membership; a name enters only
   on dates it satisfies §3, using only data available as-of that date).
2. Enumerate every **delisting event** for names that were in the eligible universe **at any point in
   the 12 months prior** to delisting.
3. **Classify** each delisting using QC event+reason fields only → {acquisition/merger,
   bankruptcy-performance, unknown}. "Adverse" = bankruptcy-performance (or a documented severe-loss
   delist reason). **Unknown-reason events are treated as NON-adverse (conservative)** for the mass
   (§5.1) and concentration (§5.3) counts — unknowns can only ever hurt a PASS, never inflate it.
4. For each delisted name, assign its **pre-delist cross-sectional rank decile as-of the last date the
   name was a valid eligible-universe member (§3)** — guaranteed to exist within the 12-month lookback.
   (A dying name decays out of the cap/liquidity band *before* its last trade, so ranking "1 month
   before last trade" would null-rank or silently drop it; ranking at last-valid-membership fixes it.)
   Keys (ranking labels only, NO returns):
   - **Momentum key (GATING):** trailing 12-1 month price-change decile — decile 1 = biggest losers =
     short-leg / adverse tail; decile 10 = winners = opposite tail. Clean, monotonic, accounting-quirk
     free (terminal decay shows here regardless of book-value accounting).
   - **Value key (REPORTED, non-gating):** book-to-price decile, with **negative-book-equity names
     excluded from the decile sort and reported as a separate "negative-equity/distressed" bucket.**
     (Naive numeric B/P places wiped-out negative-book bankruptcies in the *cheapest* bin, contaminating
     any value-tail metric — so the value axis is descriptive, not gated.)
5. **Count confidently-adverse delisting events per momentum decile** (the gating metric); separately
   **report** the value-decile distribution and the negative-equity bucket count (descriptive only).
6. **Coverage report:** unknown-reason rate; % of delisting events with usable event metadata; % of
   delisted names mappable back to PIT eligible-universe membership; negative-book-equity count.

## 5. Pre-committed kill thresholds — ANY fail → STOP
1. **Mass:** ≥ **50** *confidently-classified* adverse (bankruptcy/performance/severe-loss) delistings
   in the dev sample's eligible universe (unknown-reason events NOT counted — conservative).
2. **Mappability:** ≥ **85%** of enumerated delisting events classifiable to a terminal class
   (acquisition / adverse / documented-other), i.e. unknown-reason rate ≤ 15%. *(Softened from 90%
   pre-freeze: ≤10% unknown is unrealistic for non-CRSP vendor metadata and would false-FAIL on
   sloppiness, not data inadequacy; the §4.3 conservative-unknown treatment already blocks unknowns
   from inflating mass/concentration, so this bar only checks "do we understand most events.")*
3. **Concentration:** ≥ **2×** confidently-adverse delistings in the **momentum-loser tail (decile 1)**
   versus the **momentum-winner tail (decile 10)** — momentum key only. (The value/book axis is
   reported descriptively per §4 but does NOT gate, owing to the negative-book contamination above.)

If ALL three pass → see §6 PASS. If ANY fails → §6 FAIL.

## 6. Outcomes (pre-committed; no third path)
- **FAIL (any threshold):** STOP. Write the negative closeout: *"No affordable native-`DLRET` source
  exists, AND QuantConnect did not prove enough auditable adverse-delisted-loser support on the
  small/mid-cap tails to justify a self-built overlay → residual-alpha / market-neutral lane CLOSED,
  no universe iteration."* This is a STRONGER negative than "no native DLRET" alone.
- **PASS (all three):** does **NOT** authorize an alpha run. It authorizes exactly ONE thing — drafting
  a **separate** pre-registration for the market-neutral shot (Shumway-overlay spec + disclosed
  haircut sensitivity bands + own null floor + DSR + blinded holdout + no-iteration stop), brought
  back to the operator as a fresh go/no-go. The overlay's terminal-loss magnitudes remain
  unverifiable without CRSP, so any future result there is labeled hypothesis-generating.

## 7. Freeze & logistics
- **Freeze:** commit this file (on a branch; never push shared `main`) BEFORE running the diagnostic;
  record the commit hash here at freeze time. Thresholds (§5) immutable thereafter.
- **Execution:** needs a QuantConnect account + a LEAN diagnostic algorithm (Python). QC cloud requires
  network → Codex/Hermes (sandboxed, no-network) CANNOT run it; the LEAN run is operator/Claude-side.
- **Deviation rule:** any change to §3–§5 after freeze, or any forbidden §2 metric appearing in the
  artifact, voids the run → STOP.

## 8. Freeze record
- **Freeze commit hash: `f863c56`** (`f863c566d2d1da3027d2e7c01405d4164ad386d6`) — this commit locks
  §3–§5; any later change to those sections voids the run.
- Date: 2026-06-23 · Branch: `exec/qc-source-integrity-diagnostic-prereg`
- This §8 record was added in the immediately-following commit (provenance only; §3–§5 byte-identical).
