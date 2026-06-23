# 2026-06-23 — T0.2b residual screen result: broad universe vs floor (report-only)

Report-only Phase-0 go/no-go. Holdout never touched, no prereg hash, no gate logic run.
Screen = `backtest/residual_screen.py` (merged PR #12, unit-tested). Driver: ad-hoc, 6 price
families via `dataclasses.replace(DEFAULT_CANDIDATE, families=...)`; `holdout_frac=0.2`, `tau=0.0`,
dev window only, single-factor SPY residualization. Information ratio = `book_sharpe(stream − β·spy)`
(residual KEEPS the OLS intercept → alpha-preserving). Verdict rule: max family info ratio > 0 → GREEN.

## Results (info ratio; standalone Sharpe / SPY-beta in parens)
| family | FLOOR (30 mega-cap) | BROAD (461 survivors) |
|---|---|---|
| momentum        | −0.0242 (0.618 / 0.803) | **+0.2628** (0.768 / 0.855) |
| trend           | **+0.4143** (0.828 / 0.922) | **+0.4123** (0.821 / 0.949) |
| mean_reversion  | −0.5150 (0.455 / 0.957) | **+0.3104** (0.786 / 1.069) |
| breakout        | −0.0565 (0.254 / 0.170) | **+0.0927** (0.562 / 0.456) |
| long_momentum   | −0.2344 (0.573 / 0.885) | −0.0257 (0.677 / 0.927) |
| value           | −0.4062 (0.257 / 0.637) | **+0.4217** (0.836 / 1.051) |
| **VERDICT** | **GREEN** (via trend) | **GREEN** (5 of 6) |

## What this does and does NOT establish
1. **Both panels verdict GREEN.** The coarse `max-over-families > 0` rule fires on the FLOOR too,
   because **trend** has positive residual alpha (+0.41) even on mega-caps. So **GREEN-on-broad
   alone does NOT prove "the universe was the binding constraint"** — the pre-stated decision rule
   has a logical gap (it does not control for families already positive on the floor). The honest
   universe test is the **per-family difference**, not the max verdict.
2. **The universe effect is real for the CROSS-SECTIONAL families.** value −0.41 → **+0.42**,
   mean_reversion −0.52 → **+0.31**, momentum −0.02 → **+0.26**, breakout −0.06 → +0.09. These flip
   from ≤0 to >0 going from 30 mega-caps to 461 large-caps — consistent with "anomalies need a
   broader cross-section." trend is the exception: positive on both (universe-invariant).
3. **Correction to prior framing.** The blend-futility note's "every family carries ZERO/negative
   market-residual alpha on the floor" was scoped to value / fundamental_value / lazy_prices /
   momentum (the candidate-search readings) — it never residual-screened **trend**, the floor's best
   standalone family (0.828 > SPY 0.752), which has **+0.41** info ratio. The floor DEV_FAILED is a
   **blend/additivity failure** (ensemble 0.732 dilutes best 0.828; correlated books), NOT "no
   family has idiosyncratic alpha." Both statements coexist: trend alone has residual edge; blending
   it with correlated weaker families destroys the edge and trails SPY.

## Survivorship caveat (decisive for interpretation)
Every BROAD number is an **upper bound**. `broad_prices.csv` is current-S&P-500 membership
backfilled to 2015 (index-addition look-ahead) and filtered to full-history survivors (drops 42
post-2015 listings) — twice-filtered toward large-cap winners. Delisted worst-names that would
populate a short/bottom decile are absent → long-side inflated, short-side understated. The +0.42
on value is NOT an edge; it is a survivorship-inflated hypothesis. Small/mid-cap — where the SESTM
finding puts the alpha — is untested (keyless cannot reach it survivorship-safely).

## Decision
Per the pre-stated rule the verdict is **GREEN → proceed to Phase 1** — but read it correctly:
GREEN is a survivorship-inflated upper bound, and the binding-constraint inference rests on the
per-family lift (point 2), not the max verdict (point 1). Phase 1 = a **pre-registered
market-neutral (beta/dollar-neutral long-short)** candidate on a broad universe, with a
point-in-time S&P-membership guard, blinded holdout + ledger, its own immutable prereg surface
(NOT bolted onto PreRegConfig), judged on residual/absolute Sharpe (not "beat SPY"). It burns the
MinBTL N-budget → **freeze design before any run; needs operator greenlight.** The screen is a
hypothesis generator, not an edge.

Driver + raw provenance: `ai-logs/hermes/build_broad_fixture.py`, `ai-logs/hermes/runs/broad_provenance.json`.
Universe rule + survivorship disclosure: `apps/quant/advisor/tests/fixtures/UNIVERSE_RULE_BROAD.md`.
