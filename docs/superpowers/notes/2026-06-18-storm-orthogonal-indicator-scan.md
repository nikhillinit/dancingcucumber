# Orthogonal Indicator Scan — STORM-style research (2026-06-18)

**Method:** STORM-style multi-angle retrieval (deep-research harness): 5 search angles →
25 sources fetched → 113 claims extracted → 25 top claims adversarially verified (3-vote,
2-of-3 refutes kills) → 23 confirmed / 2 killed → 11 findings synthesized.
**Scope (operator-locked):** free + low-cost-API reproducible signals; families = news/text NLP,
macro/credit/cross-asset, positioning/flows. EXCLUDED: price-momentum/value (DEV_FAILED) and
SEC EDGAR XBRL fundamentals (already scoped in WS3C). Emphasis: reproducibility + leakage-avoidance
over raw Sharpe.

## Platform filter applied
Each candidate scored on: (a) orthogonality to price factors, (b) PIT-reproducibility under a
frozen prereg, (c) free/low-cost data feasibility, (d) fit for a WS3C-style fixture+prereg slice.

---

## TIER A — free, deterministic, cross-sectional → directly prereg-able (recommend)

### A1. Lazy Prices (10-K/10-Q YoY textual change) — TOP PICK
- **Signal:** low year-over-year document similarity ("changers") predicts low future returns.
- **Evidence:** Cohen, Malloy & Nguyen, *Lazy Prices*, J. Finance 75(3) 2020 (NBER w25084). Headline
  "up to 188bps/mo" L/S; **plan against the factor-adjusted 18–45bps/mo EW** (up to 58bps VW, t=3.59).
  Explicitly unaffected by FF3 + momentum + liquidity controls → orthogonal.
- **Data/PIT:** FREE SEC EDGAR full-text. Key on filing **acceptance date** (no look-ahead).
- **Failure modes:** small/micro-cap concentration; EW≫VW; post-2014 decay/crowding (now well-known).
- **Why top:** free, deterministic, cross-sectional, orthogonal, and **reuses WS3C EDGAR plumbing**
  (same source family + filing-date keying as the fundamentals slice). Lowest marginal build cost.

### A2. Days-to-Cover (DTC) short interest
- **Signal:** short-interest ratio / avg daily turnover; long-low/short-high DTC.
- **Evidence:** Hong, Li, Ni, Scheinkman & Yan, *Days to Cover and Stock Returns* (NBER w21166).
  EW L/S 1.19%/mo (t=6.67) vs plain short-ratio 0.71% (t=2.57); **DTC subsumes the short ratio**;
  survives DGTW, 4- and 5-factor alphas; not explained by Amihud illiquidity → not liquidity repackaging.
- **Data/PIT:** exchange short interest (bi-monthly/monthly), ~1–2wk settlement→publication lag.
- **Failure modes:** VW much weaker (0.67%); shorting cost; 2008-ban tail risk; 1988–2012 sample (decay).

### A3. Opportunistic insider trades (Form 4)
- **Signal:** insider trades classified opportunistic vs routine; trade only opportunistic.
- **Evidence:** Cohen, Malloy & Pomorski, *Decoding Inside Information*, J. Finance 67(3) 2012.
  Opportunistic VW abnormal 82bps/mo; routine ≈ 0. **Decays to ~30–40bps/mo OOS post-2008.**
  NOTE: the "classification strictly required / aggregate dilutes to zero" framing was **refuted** —
  only the 82bps opportunistic figure is confirmed.
- **Data/PIT:** FREE EDGAR; Form 4 filed ~2 business days after trade; key on filing date.

### A4. 13F holdings + confidential-treatment (CT) amendments
- **Signal:** institutional holdings changes; the CT "add-new-holdings" amendment is the higher-signal
  slice — delayed-disclosed positions earn significant abnormal returns over the confidential period.
- **Evidence:** Aragon, Hertzel & Shi, JFQA 48(5) 2013; Christoffersen, Danesh & Musto; SEC 13F FAQ.
- **PIT — strongest leakage lesson in the set:** key the panel on the **actual EDGAR File Date per
  filing** (avg lag 37d, SD 10; ~5% <2wk, ~30% at/over 45d, ~2% >49d) — **never a fixed 45-day assumption.**
- **Failure modes:** CT materially harder to obtain post-2019 (FOIA Exemption-4 + SEC 2020/2022 guidance)
  → shrinking prospective CT volume; highest build complexity in Tier A.

---

## TIER B — regime/timing, FRED-reproducible but REVISED + overlap risk (use as gate, not stock-picker)

### B1. Excess Bond Premium (EBP)
- GZ corporate-bond spread minus expected-default component = credit-market risk appetite.
  Favara, Gilchrist, Lewis & Zakrajsek (Fed FEDS Note 2016); Gilchrist & Zakrajsek, AER 2012 (NBER w22058).
  +50bp ⇒ +15pp 12-mo downturn probability; **all GZ predictive content is the EBP**; orthogonal to
  Treasury term structure and stock returns by construction.
- **PIT:** FREE FRED monthly, but **REVISED → requires ALFRED/vintage data** (latest series injects look-ahead).
  Aggregate recession-risk timing signal, NOT cross-sectional.

### B2. Variance Risk Premium (VRP)
- Option-implied variance − realized variance; predicts aggregate S&P 500 excess returns, strongest at
  **quarterly** horizon (Bollerslev, Tauchen & Zhou, RFS 2009: in-sample adj R² 6.82%, t=2.86).
- **CRITICAL:** 6.82%/t=2.86 is **in-sample**; OOS R² much weaker/insignificant in follow-up work.
  Aggregate timing only. Needs option-implied variance (VIX-style) — borderline vs the operator's
  "exclude options-implied" scope; flag before adopting.

### B3. Chicago Fed NFCI / ANFCI
- FREE FRED weekly composite (105 measures); ANFCI orthogonalizes vs CFNAI + PCE inflation.
- **Construction-orthogonal to MACRO, NOT a proven cross-sectional equity alpha** — use as risk-on/off gate.
- **PIT:** REVISED weekly → requires ALFRED vintage.

---

## TIER C — flagged / excluded

- **SESTM news-sentiment** (Ke, Kelly & Xiu, JFE 2024): best raw alpha (EW Sharpe 4.29, FF6 alpha 29bps/day
  t=14.96, R²=7.8% → "almost entirely alpha") BUT **gross, ~94% daily turnover, small/micro-cap-concentrated
  (VW Sharpe only 1.33), and corpus = Dow Jones Newswires (NOT free).** Reproducible *method*, blocked on a
  free-news substitute (EDGAR 8-K text / free headline feed) — and the alpha must survive the corpus swap.
- **Earnings-call tone** (Price et al., JBF 2012): incremental to numeric surprise over the 60-day PEAD
  window. Transcripts via low-cost APIs; enforce transcript-publication lag. Viable Tier-A-adjacent if a
  cheap transcript feed is in scope.
- **ETF informed flows** (Xu, Yin & Zhao, Fin. Mgmt 2022): directional next-day predictability confirmed,
  but **tradable magnitudes (19.16%/22.42%) REFUTED**; relies on AP-observable creation/redemption data
  (real data wall). Skip under the free-data constraint.
- **Textual Factors** (Cong et al., NBER w33168): **ruled out** — interpretability tool that explains
  momentum, NOT a tradable signal (no Sharpe/long-short reported). Do not scope.

## Killed claims (kept OUT of findings)
1. ETF tradable magnitudes 19.16%/22.42% — refuted 1-2.
2. Insider "routine dilutes to zero / classification strictly required" — refuted 1-0.

---

## Cross-cutting traps (must address before any prereg)

1. **Overlap among regime signals.** EBP, VRP, NFCI all proxy risk-on/off → likely high pairwise
   correlation. Do NOT stack all three; pick one or orthogonalize — else you double-count one latent factor.
2. **Stackability mismatch.** Timing signals (EBP/VRP/NFCI) act on market-level *direction*; cross-sectional
   signals (Lazy Prices/DTC/insider/13F) *rank names*. Not directly additive — decide architecture:
   regime-gate-on-a-cross-sectional-book vs blended score.
3. **Decay/crowding.** Headline results are pre-2014 (Lazy Prices) / pre-2008 (insider) / ≤2016 (ETF).
   Live 2026 efficacy is an open empirical question — the dev gate exists precisely to test this.
4. **Revised-series look-ahead.** Every macro series (B-tier) silently injects look-ahead if you use the
   current value. ALFRED/vintage is mandatory for PIT reproducibility.

## Recommendation — next WS3C-style slice
**Lazy Prices (A1) as the primary candidate, DTC (A2) as the orthogonal pairing.** Both are the most
reproducible and leakage-clean under the free-data constraint, both are cross-sectional (directly
comparable to the failed price book), and Lazy Prices reuses the EDGAR fetch + filing-date-keying
infrastructure WS3C is already building. Sequence as a Reading-B preregistration surface (separate hash),
holdout blinded, after the current WS3C fundamentals slice lands. Defer all B-tier regime signals until a
cross-sectional candidate clears dev — they are gates, not the thing that beats the floor.

*Source report (verified, full evidence): deep-research run `wnfamg6z1`.*
