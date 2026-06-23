# Broad fixture — universe selection rule (report-only research)

**Recorded:** 2026-06-23, at fetch time. Report-only Phase-0 residual screen; NOT a pre-registered
floor surface, touches no holdout/prereg hash. Companion to `UNIVERSE_RULE.md` (the 30-name floor).
Built by `ai-logs/hermes/build_broad_fixture.py` → `broad_prices.csv`.

## Window
`2015-01-01 .. 2024-01-01` requested (daily); data spans `2015-01-02 .. 2023-12-29`, **2264 rows** —
the same window and calendar master (SPY) as the floor fixture, so the two are directly comparable.

## Selection rule (current-membership backfill — see survivorship disclosure)
The universe is the **current S&P 500 constituent list** scraped keyless from Wikipedia
(`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`) at fetch time, normalized to yfinance
form (`BRK.B → BRK-B`), plus `SPY` as benchmark — filtered to names with **full continuous daily
coverage across the whole window**. The survival/coverage filter is purely mechanical (no return- or
Sharpe-based pruning); names lacking full-window coverage are dropped, never hand-picked.

- **Raw symbols scraped:** 503 (dual-class listings — e.g. GOOG/GOOGL, FOX/FOXA — push the count
  above 500). **Unique normalized:** 503.
- **Kept (full-coverage):** **461 names + SPY = 462 columns.**
- **Dropped (42, all post-2015 IPOs / spinoffs / ticker changes — listing-date-explicable, not
  fetch errors):** ABNB, APP, CARR, CEG, COIN, CRWD, CTVA, CVNA, DASH, DDOG, DELL, DOW, EXE, FDXF,
  FOX, FOXA, FTV, GDDY, GEHC, GEV, HOOD, HPE, HWM, INVH, IR, KHC, KVUE, LITE, MRNA, OTIS, PLTR, PYPL,
  Q, SNDK, SOLV, TTD, UBER, VICI, VLTO, VRT, VST, XYZ.

## Data & corporate actions
- **Source:** yfinance 0.2.66, batched `yf.download(..., auto_adjust=True, group_by="ticker")`,
  pulled 2026-06-23. Constituent list from Wikipedia, same date.
- **Price:** adjusted close (`auto_adjust=True` — splits & dividends adjusted). AAPL 2015-01-02
  reconciles to `floor_prices.csv` (24.19), confirming consistent adjustment basis.
- **Missing-data policy:** restrict to dates where SPY is present; any ticker missing any row over
  the window is dropped (coverage filter). Resulting panel has zero NaN.

## Survivorship disclosure (read before interpreting the screen)
This fixture is **survivorship-biased by construction and twice-filtered toward large-cap survivors**:
1. **Current-membership backfill.** Today's S&P 500 membership is applied to 2015–2023 history.
   Names delisted/merged/removed since 2015 are absent, and names *added* because they performed
   well are present (index-addition look-ahead). This is NOT point-in-time membership; the optional
   PIT-membership guard is deferred (plan §crux).
2. **Full-history coverage requirement.** Drops the 42 post-2015 listings above — i.e. further
   filters toward old, stable, large names.

Net: **large-cap survivors**, broader than the 30 mega-caps but NOT the mid-/small-cap cross-section
where the hypothesis (and the SESTM small-cap finding) expects residual alpha. Keyless data cannot
reach that cross-section survivorship-safely (no keyless delisted-price source).

**How this bounds the residual screen (`backtest/residual_screen.py`):**
- The delisted worst-names that would populate a short/bottom decile are missing → long-side is
  upward-biased and short-side alpha is understated. A **positive** information ratio here is an
  **UPPER bound**.
- The inflation is **direction-specific**: it hits contrarian/reversal families (value,
  mean_reversion) hardest, because the missing names are the cheap-stocks-that-fell-and-delisted
  those signals would buy.
- A **RED** (all families' info ratio ≤ 0) is conclusive **only for large-cap survivors**. It does
  NOT prove price-only signals are dead on a genuine mid/small-cap universe — that requires a keyed,
  delisting-aware source and remains untested.
- **GREEN is NON-DISCRIMINATING, not a go-signal.** The screen's `max-over-families > 0` verdict
  also fires GREEN on the DEV_FAILED floor (via trend) — so GREEN here is inconclusive. See the
  T0.2b result note `docs/superpowers/notes/2026-06-23-broad-universe-residual-screen-result.md`.
