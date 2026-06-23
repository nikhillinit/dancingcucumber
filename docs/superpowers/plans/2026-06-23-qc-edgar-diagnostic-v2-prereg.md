# PRE-REGISTRATION v2 — QC prices + EDGAR delisting reasons, source-integrity diagnostic (NON-ALPHA)

> **Status: DRAFT — freeze BEFORE any data run.** Supersedes v1
> (`2026-06-23-qc-source-integrity-diagnostic-prereg.md`, frozen `f863c56`), now **RETIRED**: v1
> required QC-native delisting classification, which QC API research proved impossible (no reason field
> in the `Delisting` object, `DelistingType` enum, or Security Master map files). v1 was never executed
> → superseding pre-run is clean (no results to bias). Thresholds (§5) lock on freeze; any later change
> to §3–§5 voids the run. Separate surface — does NOT touch `PreRegConfig`, the floor, or the holdout.
> Lineage: `2026-06-23-phase1-direction-roleplay-debate-synthesis.md` → `…-v2-PROPOSAL.md`.

## 0. One-line
Falsify-or-validate that **QC (survivorship-bias-free prices + delisting dates) joined to EDGAR
(delisting reasons)** carries enough auditable adverse-delisting mass on the small/mid-cap momentum
tails to justify a later self-built Shumway overlay. **Data-QC test, NOT an alpha test.**

## 1. Why v2 (what changed from v1)
QC cannot say *why* a stock delisted (confirmed: `Delisting`/`DelistingType`/map files carry no cause).
SEC EDGAR can. v2 keeps v1's universe, ranking, momentum-gating, conservative-unknown, and thresholds,
and changes ONE thing: the delisting **reason** comes from EDGAR filings, joined to QC's delisted
names. The eventual overlay needs these reasons anyway (−30% performance vs −100% bankruptcy), so the
EDGAR layer is load-bearing for the next stage, not throwaway.

## 2. Scope & hard exclusions (unchanged from v1)
- **NON-ALPHA.** Artifact MUST NOT contain returns / Sharpe / info ratio / PnL / portfolio
  construction / family comparisons. Any alpha readout voids the run (anti-sunk-cost guardrail).
- Report-only. No gate logic, no holdout touch, no `PreRegConfig` hash. Floor unchanged.

## 3. Data sources (both keyless)
- **QuantConnect (free tier, project 33255206):** survivorship-bias-free daily prices for delisted
  names (map files); delisting **dates** (map files — NOT live events, which undercount); PIT universe
  selection (coarse/fine); fundamentals (MarketCap, BookValuePerShare) for ranking.
- **SEC EDGAR (keyless, data.sec.gov + EDGAR full-text/daily index; 10 req/s, User-Agent required):**
  delisting reason via filing types (Form 25, 8-K Items 1.03/2.01/3.01, DEFM14A, Form 15).
- **Universe (PIT, per date):** US common equity (`SecurityReference.IsPrimaryShare==True`,
  `IsDepositaryReceipt==False`, SecurityType Common Stock); 21-day median dollar-volume ≥ $1M;
  market-cap percentile band 20–90. **Dev period 2015-01-01 → 2023-12-31**; nothing post-2023 examined.

## 4. Procedure
1. Reconstruct the eligible universe per month (PIT; a name enters only on dates it satisfies §3
   as-of that date). Record membership + momentum + BVPS at each selection.
2. Enumerate delisting **dates** from QC **map files** for names eligible at any point in the 12 months
   prior to delisting.
3. Rank each delisted name as-of **the last date it was a valid eligible member (§3)** — guaranteed
   within the 12-month lookback. Keys (labels only, NO returns):
   - **Momentum (GATING):** trailing 12-1 month price-change decile (decile 1 = losers = adverse tail;
     decile 10 = winners = opposite tail). Price-only → computable even after fundamentals are dropped.
   - **Value (REPORTED, non-gating):** book-to-price = BVPS/price; **negative-BVPS names excluded from
     the decile sort and bucketed separately** (negative book ≠ cheap).
4. **EDGAR classification (EDGAR-first join — the robust direction):**
   a. Enumerate all **Form 25 / 25-NSE** filings 2015–2023 → table (CIK, company name, filing date,
      cited Rule 12d2-2 paragraph). Every delisting files one → bounded authoritative set.
   b. **Match** each QC eligible-universe delisting to a Form 25 by company-name (normalized fuzzy) +
      delisting-date proximity (QC map-file date within ±15 calendar days of the Form 25 date). A QC
      delisting with no matched Form 25 = **unmatched** → unknown.
   c. **Classify** via the issuer's (CIK) 8-K items within ±90 days of the Form 25 date, precedence
      **bankruptcy > acquisition > performance > unknown**:
      - **8-K Item 1.03** (Bankruptcy/Receivership) present → **bankruptcy** (ADVERSE).
      - else **8-K Item 2.01** (Completion of Acquisition) or a **DEFM14A** in the prior 12 months →
        **acquisition** (non-adverse).
      - else **8-K Item 3.01** (Notice of Delisting / Failure to Satisfy a Continued Listing Standard)
        OR the Form 25 cites a listing-standard rule paragraph → **performance** (ADVERSE).
      - else → **unknown**.
   d. **Adverse = bankruptcy ∪ performance.** **Unknown (unmatched or unclassifiable) = conservatively
      NON-adverse** (can only hurt a PASS, never inflate it).
5. Count **confidently-adverse** delistings per momentum decile (GATING); separately report the value
   buckets, negative-equity count, and coverage gaps (unmatched rate, unknown-reason rate, CIK-bridge
   failures).

## 5. Pre-committed kill thresholds — ANY fail → STOP
1. **Mass:** ≥ **50** confidently-classified adverse (bankruptcy/performance) delistings in the dev
   sample's eligible universe (unknown = non-adverse).
2. **Mappability:** ≥ **85%** of QC eligible-universe delistings matched to a Form 25 **and** assigned a
   confident terminal class (acquisition / bankruptcy / performance), i.e. unknown ≤ 15%. *(This is
   also the identifier-bridge gate: a weak QC↔EDGAR join shows up here as low mappability → honest
   STOP, not a wrong answer.)*
3. **Concentration:** ≥ **2×** confidently-adverse delistings in the **momentum-loser tail (decile 1)**
   vs the **momentum-winner tail (decile 10)** — momentum key only (value reported, non-gating).

If ALL three pass → §6 PASS. If ANY fails → §6 FAIL.

## 6. Outcomes (pre-committed; no third path)
- **FAIL (any threshold):** STOP. Write the negative: *"No affordable native-`DLRET` source; QC+EDGAR
  did not prove enough auditable adverse-delisted-loser support on the small/mid-cap momentum tails (or
  the identifier bridge was too weak) → residual-alpha / market-neutral lane CLOSED, no iteration."*
- **PASS (all three):** authorizes ONLY drafting a **separate** pre-registration for the market-neutral
  shot (Shumway overlay using these EDGAR reasons + disclosed haircut sensitivity bands + own null
  floor + DSR + blinded holdout + no-iteration stop), back to the operator as a fresh go/no-go. The
  overlay's terminal-loss magnitudes remain unverifiable without CRSP → any future result is
  hypothesis-generating.

## 7. Freeze & logistics
- **Freeze:** commit this file (branch; never push `main`) BEFORE running; record the hash in §8.
  §3–§5 immutable thereafter; any change, or any forbidden §2 metric in the artifact, voids the run.
- **Build routing:** EDGAR network fetches (Form 25 enumeration, 8-K items) run DIRECTLY/Claude-side
  (Codex is no-network); deterministic parsers/bridge/counting logic → Hermes tasks + pytest vs
  fixtures; the QC universe/map-file run is operator/Claude-side on QC cloud (NOT a Hermes dispatch).
- **Deviation rule:** any §3–§5 change after freeze → STOP.

## 8. Freeze record (fill at commit time)
- Freeze commit hash: `__________` · Date: `__________` · Branch: `__________`
