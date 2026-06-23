# PROPOSAL (NOT frozen, no build) — v2 source-integrity diagnostic: QC prices + EDGAR delisting reasons

> **Status: SCOPING PROPOSAL for a GO/STOP decision.** Supersedes the never-executed frozen v1
> (`2026-06-23-qc-source-integrity-diagnostic-prereg.md`, `f863c56`) — v1 required QC-native delisting
> classification, which the QC API research proved IMPOSSIBLE (no reason field anywhere: `Delisting`
> object, `DelistingType` enum, Security Master map files). v1 as-frozen resolves to a hollow STOP
> (mappability 0% by construction). This proposal is **not** a prereg and **not** frozen; it exists so
> the operator can decide GO/STOP with real scope + cost in front of them. Nothing here runs until a
> v2 prereg is written AND frozen.

## 1. The fix (operator's insight): two keyless sources, joined
QC cannot say *why* a stock delisted; SEC EDGAR can. Split the data accordingly:
- **QuantConnect (free tier):** survivorship-bias-free daily **prices** for delisted names (via map
  files), **delisting dates** (map files), PIT **universe** selection (coarse/fine) for the cap-band +
  dollar-volume filter, and **fundamentals** (MarketCap, BookValuePerShare) as-of last-eligible date
  for ranking. Momentum (price-only) is computable even after Morningstar drops a delisted name's
  fundamentals — which is why the v1→v2 momentum-only gating also dodges QC's fundamental-undercount hole.
- **SEC EDGAR (keyless):** the delisting **reason**, from filing types around the delisting date:
  - **Form 25** (Rule 12d2-2 removal notice) — the delisting event + the rule paragraph it cites.
  - **8-K Item 1.03** (Bankruptcy or Receivership) → *bankruptcy* (adverse).
  - **8-K Item 3.01** (Notice of Delisting / Failure to Satisfy a Continued Listing Standard) →
    *performance/compliance* (adverse).
  - **8-K Item 2.01** (Completion of Acquisition) / **DEFM14A** merger proxy → *acquisition* (non-adverse).
  - **Form 15** (deregistration) — corroborating, not discriminating alone.
  - "Adverse" = bankruptcy ∪ performance. (Exact discriminating ruleset = a deliverable of the
    pretotype, §4 — must be sample-validated, not assumed.)

## 2. The hard part — the identifier bridge (TOP RISK)
Joining a QC delisted symbol to its EDGAR filings requires a **point-in-time** ticker/name → CIK map.
SEC `company_tickers.json` is **current-only**; a name delisted in 2017 may have had its ticker reused
since → naive ticker→CIK mis-joins. Two join directions (the pretotype tests which is reliable):
- **(a) QC-first:** QC symbol+date → CIK (via former-ticker history in QC map files + EDGAR submissions
  `formerNames`/filing history) → pull that CIK's filings → classify.
- **(b) EDGAR-first (likely more robust):** enumerate ALL **Form 25** filings 2015–2023 from EDGAR
  (bounded set — every delisting files one), build a (CIK, company-name, date, rule) reason table, then
  **fuzzy-join to QC's delisted names by company-name + delisting-date proximity** — sidesteps
  ticker reuse entirely. 8-K Item 1.03/3.01/2.01 add the reason refinement per CIK.
**Bridge quality IS a gate:** if < ~85% of QC delisted names map to a confident terminal class, §5.2
mappability fails → STOP. So a weak bridge produces an honest STOP, not a wrong answer.

## 3. Procedure (v2, if built — carries v1's frozen design where unchanged)
1. PIT eligible universe per month (QC coarse/fine: US common via `IsPrimaryShare` &
   `IsDepositaryReceipt==False` & SecurityType Common; 21d-median dollar-vol ≥ $1M; mkt-cap pctile
   20–90), 2015–2023. Record membership + momentum + BVPS at each selection.
2. Enumerate delisting **dates** from QC **map files** (survivorship-bias-free; NOT live events, which
   undercount) for names eligible within 12 months prior.
3. Rank each delisted name as-of **last-valid-eligible-membership**: momentum decile (GATING,
   price-only) + book-to-price bucket (REPORTED, negative-BVPS excluded + bucketed separately).
4. **EDGAR classify** each delisting via §1/§2 → {acquisition, bankruptcy, performance, unknown};
   adverse = bankruptcy ∪ performance; unknown = conservatively NON-adverse.
5. Count confidently-adverse delistings per momentum decile; report value buckets, negative-equity,
   and coverage gaps (unknown rate, unbridged names).

## 4. De-risk FIRST — identifier-bridge pretotype (cheap, ~0.5–1 day, keyless, NO freeze)
Before any prereg/freeze/full build, test the single riskiest assumption on a small sample:
- Hand-pick ~25 known delisted 2015–2023 small/mid-caps spanning all classes (a few bankruptcies, a few
  acquisitions, a few listing-standard delistings).
- For each, attempt: (a) QC delisting date, (b) CIK resolution as-of date (test both join directions),
  (c) find the Form 25 / 8-K 1.03 / 3.01 / 2.01 that classifies it.
- **Pre-set kill:** if < ~85% of the sample bridges to a confident terminal class, the join is
  unreliable → **STOP cheaply** (do not build the full adapter). If it clears, the full v2 build is
  justified and the working ruleset is now in hand.
This mirrors the project's pretotyping discipline (cf. the T0.2b structural-null probe) — spend ~1 day
to validate the assumption that would otherwise sink a 5–10 day build.

## 5. Honest cost estimate (the EV input)
| Component | Effort | Risk |
|---|---|---|
| Identifier-bridge **pretotype** (§4) | ~0.5–1 day | the decisive gate; cheap to run |
| EDGAR filing-type adapter (submissions API + 8-K **item-number** parsing — items are in the 8-K body/FTS, not the submissions list) | ~2–3 days | medium (8-K item extraction is fiddly) |
| Point-in-time identifier bridge (full, beyond pretotype) | ~1–2 days | **high** (ticker reuse, name matching) |
| QC side: map-file delisting dates + QuantBook PIT universe/ranking | ~1–2 days | medium (free-tier B-MICRO RAM/time on 8y daily) |
| Join + reconciliation + diagnostic counting | ~1 day | medium |
| **Total (post-pretotype)** | **~5–8 dev-days** | dominated by the bridge + 8-K parsing |
Exceeds the original ~3-day ceiling → the EV bar genuinely moves. Mitigant: the EDGAR reason layer is
**not throwaway** — the eventual Shumway overlay needs these exact reasons (to assign −30% performance
vs −100% bankruptcy), so this work feeds the next stage if the lane survives.

## 6. Thresholds (proposed — would be frozen in the v2 prereg, unchanged from v1 in spirit)
Mass ≥ 50 confidently-adverse delistings · mappability ≥ 85% (now achievable via EDGAR; unknowns
conservative) · momentum-loser-tail concentration ≥ 2× vs winner tail (value non-gating). Alpha
metrics remain FORBIDDEN; PASS ≠ alpha run (unlocks only the separate overlay prereg).

## 7. Recommended path
**Run the §4 pretotype next** (cheap, keyless, no freeze, no commitment). It is the smallest move that
makes the GO/STOP decision real: it either kills the lane for ~1 day of work (bridge unreliable) or
hands back a validated classification ruleset that justifies freezing + building v2. Only after the
pretotype passes do we write + freeze the v2 prereg and start the ~5–8 day build.

Default if the operator declines further spend: literal STOP on frozen v1 (QC-only) — write the
negative, close the residual-alpha lane. All work remains report-only; floor DEV_FAILED unchanged.
