# v2 diagnostic — BUILD PLAN (executes the FROZEN prereg 7f54f55)

> Implements `2026-06-23-qc-edgar-diagnostic-v2-prereg.md` (frozen `7f54f55`). NON-ALPHA throughout —
> no returns/Sharpe/PnL in any artifact. Routing per prereg §7: EDGAR network fetches run DIRECT
> (Codex no-network); deterministic parsers/logic → Hermes + pytest vs fixtures; QC universe/map-file
> run is operator/Claude-side on QC cloud (project 33255206), NOT a Hermes dispatch.

## Task graph (dependency order)
| Task | What | Routing | Depends on | Status |
|---|---|---|---|---|
| **B1** | EDGAR delisting enumeration: master.idx 2015Q1–2023Q4, Form ∈ {25, 25-NSE} → `edgar_delistings_2015_2023.csv` (cik, company, form, date, accession) | DIRECT fetch | — | **RUNNING** (mechanics validated on 2017Q1: 349 filings) |
| **B2** | 8-K reason classification: per delisting CIK, fetch submissions + 8-K filings ±90d; extract Item 1.03 / 2.01 / 3.01; tag {bankruptcy, acquisition, performance, unknown} per §4.4c precedence | DIRECT fetch + Hermes parser (pytest vs saved 8-K fixtures) | B1 | TODO |
| **B3** | QC universe + delisting dates: LEAN/QuantBook PIT eligible universe (US common, ≥$1M ADV, cap pctile 20–90) monthly 2015–2023; record membership + 12-1 momentum + BVPS; delisting dates from map files for names eligible ≤12mo prior → `qc_eligible_delistings.csv` | QC-side (operator/Claude; I write the algo, operator runs on free tier) | — (parallel to B1/B2) | TODO — needs the LEAN algo written |
| **B4** | Identifier bridge / join: fuzzy-match QC delisted names (B3) ↔ EDGAR Form 25 (B1) by normalized company-name + date proximity (±15d); attach B2 class; emit match diagnostics (matched %, unmatched, ambiguous) | Hermes (deterministic) + pytest | B1, B2, B3 | TODO |
| **B5** | Counting diagnostic: count confidently-adverse per momentum decile; compute the 3 thresholds (mass ≥50 / mappability ≥85% / momentum-concentration ≥2×); emit NON-ALPHA artifact; apply §6 outcome | Hermes (deterministic) + pytest | B4 | TODO |

## Outcome wiring (frozen §6)
- **FAIL any threshold → STOP:** write the negative, close the residual-alpha / market-neutral lane (no universe iteration).
- **PASS all three →** draft a SEPARATE market-neutral overlay prereg (Shumway overlay keyed on the B2 EDGAR reasons + sensitivity bands + own null/DSR/blinded-holdout + no-iteration stop). NOT an alpha run.

## Notes / risks (live)
- **Top risk = B4 identifier bridge.** Surfaces as §5.2 mappability: <85% matched+classified → honest STOP. EDGAR-first (B1 enumerates the authoritative Form-25 set; match QC→EDGAR by name+date) avoids ticker-reuse.
- **Form 25-NSE dominates Form 25** (332 vs 17 in 2017Q1) — 25-NSE = exchange-filed removal; both are kept.
- **8-K item numbers are NOT in the submissions list** — must parse the 8-K body/header for "Item X.XX" (B2 cost driver).
- **QC fundamental-undercount** (Morningstar drops delisted names) is dodged for the GATING metric because momentum is price-only; matters only for the reported value axis.
- master.idx is pipe-delimited (`CIK|Company|Form|Date|Filename`) — cleaner than the fixed-width form.idx.

## Provenance
- B1 enumerator: `ai-logs/hermes/edgar_b1_enumerate_delistings.py`; probe: `ai-logs/hermes/edgar_form25_probe.py`.
- API research (QC has no native delisting reason; 3-layer confirmation) summarized in `…-v2-PROPOSAL.md §1`.
