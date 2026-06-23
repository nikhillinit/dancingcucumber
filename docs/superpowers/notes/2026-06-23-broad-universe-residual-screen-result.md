# 2026-06-23 — T0.2b residual screen result: broad universe vs floor (report-only)

> **TL;DR — INCONCLUSIVE, not a go.** The screen returns GREEN on the BROAD panel — but the
> cross-check shows it *also* returns GREEN on the FLOOR (via trend), and the floor is the one
> universe with ground truth: DEV_FAILED. A go/no-go that greenlights the known-dead case does not
> discriminate, so GREEN here is **not a go-signal**. The only informative quantity left — the
> per-family floor→broad delta — is survivorship-confounded in a *direction-specific* way that hits
> exactly the families that moved (contrarian value/mean_reversion). The keyless experiment cannot
> settle the question; a delisting-aware, point-in-time universe is required and keyless can't reach
> it. **Do NOT rest Phase 1 on this verdict.**

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
1. **GREEN is non-discriminating → it is NOT a go-signal.** The `max-over-families > 0` rule fires
   on the FLOOR too, because **trend** has positive residual alpha (+0.41) even on mega-caps.
   *One-line check: would this screen have greenlit the floor? **Yes (trend) → the verdict is not a
   go-signal.*** The floor met this necessary condition (≥1 family with in-sample residual alpha)
   and still failed every *sufficient* bar — deflation (DSR), holdout, additivity — ending
   DEV_FAILED. So "broad is GREEN → proceed to Phase 1" is *exactly* the inference that would have
   greenlit the floor. **Phase 1 must be decoupled from this verdict.** The only informative
   quantity is the **per-family floor→broad difference** (point 2), not the verdict.
2. **The per-family delta is the only informative quantity — and it is survivorship-confounded
   exactly where it moved.** value −0.41 → +0.42, mean_reversion −0.52 → +0.31, momentum −0.02 →
   +0.26, breakout −0.06 → +0.09; trend +0.41 → +0.41 (universe-invariant). The tempting read is
   "anomalies need a broader cross-section." But the confound is **direction-specific**: value and
   mean_reversion are **contrarian/reversal** signals, and survivorship bias (the absent names are
   the cheap-stocks-that-fell-and-delisted) inflates contrarian signals **mechanically**. value
   going from the **worst** family on the floor (−0.41) to the **best** on broad (+0.42, beating
   trend) is survivorship's fingerprint, not evidence of breadth. trend (universe-invariant) and
   momentum are far less exposed. So the very deltas the go-signal would lean on are the least
   trustworthy — which is why the keyless experiment cannot settle this.
3. **Correction to prior framing.** The blend-futility note's "every family carries ZERO/negative
   market-residual alpha on the floor" was scoped to value / fundamental_value / lazy_prices /
   momentum (the candidate-search readings) — it never residual-screened **trend**, the floor's best
   standalone family (0.828 > SPY 0.752), which has **+0.41** info ratio. The floor DEV_FAILED is a
   **blend/additivity failure** (ensemble 0.732 dilutes best 0.828; correlated books), NOT "no
   family has idiosyncratic alpha." Both statements coexist: trend has *in-sample, un-deflated*
   residual signal; blending it with correlated weaker families destroys the edge and trails SPY.
   **This is NOT "trend is viable":** the +0.41 rests on the same 0.828 standalone the floor already
   ran through deflation (DSR) and holdout and rejected. It is a measurement, not a tradable edge.

## Survivorship caveat (decisive for interpretation)
Every BROAD number is an **upper bound**. `broad_prices.csv` is current-S&P-500 membership
backfilled to 2015 (index-addition look-ahead) and filtered to full-history survivors (drops 42
post-2015 listings) — twice-filtered toward large-cap winners. Delisted worst-names that would
populate a short/bottom decile are absent → long-side inflated, short-side understated. The bias is
**not uniform across families**: it inflates contrarian/reversal signals (value, mean_reversion)
the most — the missing names are exactly the cheap-stocks-that-fell-and-delisted those signals would
buy — so the families that "improved" most are the least trustworthy (point 2). The +0.42 on value
is NOT an edge; it is a survivorship-inflated hypothesis. Small/mid-cap — where the SESTM finding
puts the alpha — is untested (keyless cannot reach it survivorship-safely).

## Decision
**INCONCLUSIVE on keyless data — NOT a screen-justified go.** Three reasons, in order of weight:
(1) the GREEN verdict is non-discriminating — it fires on the DEV_FAILED floor (point 1), so it
cannot license "proceed"; (2) the only informative quantity, the per-family delta, is
survivorship-confounded precisely on the contrarian families that moved (point 2); (3) every broad
number is an upper bound. A clean test of "does a broader cross-section carry residual alpha"
requires a **delisting-aware, point-in-time** universe — which keyless data cannot reach. That is
the honest ceiling of this experiment, not a defect in it.

**If the operator still pursues Phase 1, it is an eyes-open hypothesis, not a screen result.** Its
design would be a **pre-registered market-neutral (beta/dollar-neutral long-short)** candidate on a
PIT, delisting-aware universe (needs a keyed source), blinded holdout + ledger, its own immutable
prereg surface (NOT bolted onto PreRegConfig), judged on residual/absolute Sharpe (not "beat SPY").
It burns the MinBTL N-budget → **freeze design before any run; needs operator greenlight.** The
screen is, at most, a hypothesis generator — and on this data it does not even generate a clean one.

Driver + raw provenance: `ai-logs/hermes/build_broad_fixture.py`, `ai-logs/hermes/runs/broad_provenance.json`.
Universe rule + survivorship disclosure: `apps/quant/advisor/tests/fixtures/UNIVERSE_RULE_BROAD.md`.
