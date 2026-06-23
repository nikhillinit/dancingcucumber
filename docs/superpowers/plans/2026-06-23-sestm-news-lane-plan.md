# SESTM news-sentiment lane — PLAN (report-only; not a prereg, not started)

> Consolidates the scattered SESTM references (`2026-06-18-storm-orthogonal-indicator-scan.md` TIER C;
> `2026-06-16-deferred-plans-roadmap.md`) into one current plan, now that the residual-alpha /
> market-neutral price lane is CLOSED (`2026-06-23-v2-STOP-closeout.md`). **This is a plan, not a
> prereg:** no forward thresholds are frozen here — they get frozen in the Phase-0/Phase-2 preregs
> against real constraints (per the roadmap's standing rule). Floor untouched; report-only.

## 1. Hypothesis & why this lane is different
SESTM (Ke, Kelly & Xiu, *Predicting Returns with Text Data*, JFE 2024): supervised news-sentiment →
returns (marginal word-screening → sentiment topic model → article score). Published raw alpha is
strong and "almost entirely alpha" (EW Sharpe 4.29, FF6 α ≈ 29 bps/day, t=14.96, R²≈7.8%).

Two reasons it is worth a look despite 7 prior negatives:
- It is the FIRST lane that is genuinely **text-orthogonal** to everything already DEV_FAILED (price
  momentum/value, book-to-price fundamentals, Lazy-Prices filing-text).
- Its alpha is **small/micro-cap-concentrated** (VW Sharpe only 1.33 vs EW 4.29). That targets a
  DIFFERENT universe from the 30-mega-cap floor — directly addressing the roadmap's standing finding
  that the program should **test the universe**, not merely add another signal to the dead one.

## 2. Known blockers (these DEFINE the plan's gates — from the TIER-C scan)
1. **Corpus = Dow Jones Newswires (NOT free).** The *method* is reproducible; the *data* is the
   blocker. Need an affordable, timestamped, ticker-tagged corpus — **and the alpha must survive the
   corpus swap** (a free feed has lower coverage/quality/latency than DJNW).
2. **~94% daily turnover.** Published alpha is GROSS. Any honest test must be **net of transaction
   costs**; costs plausibly destroy most of it.
3. **Small/micro-cap concentration.** The tradeable alpha lives in low-liquidity / hard-to-borrow
   names — the same survivorship/tradability hazards that just closed the price lane (delisting-aware
   data is hard, and we proved it).
4. **Leakage / timestamp discipline.** News timestamps must be strictly as-of (publication lag, not
   scrape time). Look-ahead in news data is the classic trap.

## 3. Discipline carried over from the closed lane (apply verbatim)
- **Cheapest discriminating test FIRST, before any build** (the Step-0-matrix / multiplicative-ceiling
  pattern). Surface a STOP cheaply rather than building a pipeline that fails.
- **Separate immutable prereg surface**, pre-committed thresholds, **freeze-before-run**, blinded
  holdout, **default STOP**, report-only, floor/holdout untouched.
- **State the degradation chain up front** so a pass cannot be manufactured: published EW/gross/DJNW
  Sharpe 4.29 → realistic is net-of-cost × VW/tradable × free-corpus-survival, i.e. a small fraction.
  If the honest expected net number is ≤ the bar, say so before running.

## 4. Phased plan (cheap → expensive, each gated; STOP is first-class at every phase)

### Phase 0 — news-corpus capability matrix (cheap, mostly keyless; mirrors Step-0)
Scope affordable, timestamped, ticker-tagged news sources over the dev period with small/mid-cap reach:
- **EDGAR 8-K text — KEYLESS, tooling ALREADY BUILT this session** (`edgar_b2_fetch.py` pulls 8-K
  filings + items + dates per CIK, PIT, ticker↔CIK). Material-event filings are a natural free corpus
  substitute (the TIER-C note flagged exactly this). Caveat: filings ≠ newswire — different text
  distribution and event cadence; the SESTM alpha may or may not transfer.
- Free/cheap news APIs (each with coverage/history/licensing/timestamp caveats to verify): GDELT
  (free, broad, noisy), Finnhub, Polygon, Tiingo, NewsAPI, Benzinga.
- **Must-haves (pre-set):** publication-time timestamps (not scrape time); ticker-tagged; point-in-time
  / no silent revision; small/mid-cap coverage; sufficient history for the dev window; license permits
  research use.
- **Pre-committed STOP:** if no affordable corpus clears the must-haves → STOP (the direct analog of
  "no affordable delisting-returns source" that closed the price lane).

### Phase 1 — SESTM-lite signal-survival probe (cheap; the multiplicative-ceiling analog)
Before the full pipeline, on the chosen free corpus, run a minimal SESTM-style score and check it has
ANY out-of-sample predictive **sign**, strictly as-of, **net of a turnover-cost haircut**, on a
small/mid-cap universe (QC project 33255206 for returns/universe).
- **Pre-committed kill:** no positive net OOS sign on the free corpus → STOP (the corpus swap killed
  it — the cheap discriminator, before building the full estimator).

### Phase 2 — full SESTM candidate (ONLY if Phase 0 + 1 pass)
Separate FROZEN prereg: full SESTM pipeline (screen → topic model → score), small/mid-cap universe,
blinded holdout + ledger, own random-null floor + DSR, **net-of-cost**, no-iteration stop. Judged on
**net** residual/absolute Sharpe (not "beat SPY"). PASS = a hypothesis worth a real build, not an edge.

## 5. EV & base-rate honesty
Seventh-negative base rate → low prior. The case FOR testing is narrow but real: SESTM is the only
remaining lane that is both text-orthogonal AND aimed at the small/micro-cap universe where its
published alpha actually lives. The case AGAINST: the corpus swap + net-of-cost + tradability will
erode most of the gross EW number, and small/micro-cap tradability is the exact hazard that just sank
the price lane. The phased gates exist to surface that STOP for the price of Phase 0+1, not a full build.

## 6. Routing & logistics
- Keyless news fetches (EDGAR 8-K, GDELT) run DIRECT/in-lane; keyed APIs need operator keys.
- Deterministic pipeline code (screening/topic-model/scoring) → Hermes + pytest vs fixtures; network
  fetches direct; returns/universe via QC (operator/Claude-side).
- This plan is NOT started and NOT a prereg. Needs an operator greenlight + an explicit EV decision vs
  other uses of time. First concrete step if greenlit = the Phase-0 corpus capability matrix.

## 7. Supersedes / relation to prior notes
- Replaces the roadmap's one-liner "SESTM/news sentiment stays research-only/conditional for the
  large-cap book" — reframed: SESTM is a **small/mid-cap** lane, gated on a free-corpus feasibility
  matrix, not a large-cap-book add-on.
- Inherits the TIER-C numbers + the "alpha must survive the corpus swap" caveat as the central risk.
