# 2026-06-21 — Why no candidate clears dev: it's zero market-residual alpha, not the blend

> **⚠️ CORRECTION (2026-06-22).** The "residual Sharpe = 0 for every family" reported below is an
> **OLS-intercept ARTIFACT** — subtracting `a·spy + b` forces the residual mean ≡ 0, so its Sharpe
> is identically 0 regardless of any real alpha (vacuous). The corrected screen
> (`backtest/residual_screen.py`, merged PR #12) gates on the **information ratio**
> `book_sharpe(stream − a·spy)`, which KEEPS the OLS intercept (alpha-preserving). Re-measured on
> `floor_prices.csv` across the candidate-search readings: **value −0.41, fundamental_value −0.32,
> lazy_prices −0.40, momentum −0.02 → all negative (RED), not ≈ 0.** The screen rule is
> **information ratio > 0** (GREEN), not "Sharpe ≈ 0".
>
> **⚠️ FURTHER (T0.2b, 2026-06-23):** these four are NOT the full family set. The full price-family
> screen surfaces **trend +0.41 on the floor** (the floor's best standalone, 0.828 > SPY 0.752) —
> so "zero residual alpha for EVERY family" is FALSE. The floor DEV_FAILED is a **blend/additivity
> failure**, not zero-alpha-per-family; trend's +0.41 is in-sample and *un-deflated* (the floor
> already ran that 0.828 through DSR + holdout and rejected it — NOT a tradable edge). The futility
> CONCLUSION below still survives (dev additivity gate B 1/4, C 0/4), but state it as
> blend-non-additivity, not "no family has alpha." See the T0.2b result note
> `docs/superpowers/notes/2026-06-23-broad-universe-residual-screen-result.md`,
> `[[blend-futility-residual-alpha]]`, and the plan
> `docs/superpowers/plans/2026-06-22-universe-change-residual-screen.md`.

Report-only diagnostic investigation (read-only; holdout never touched; floor verdict
unchanged). Triggered by the handoff's "pursue the next orthogonal reading (SESTM)" task and
the roadmap's "test the BLENDING/weighting, not another signal" reframing. The data says
**both framings are wrong** — and pins the actual obstacle.

## What was measured (reproducible, dev window only)
Scripts (read-only, single-family `run_dev_sweep_ext`, no `run_holdout`, no `prereg_hash`):
- `ai-logs/hermes/diag_standalone_sharpe.py` — standalone dev Sharpe per family
- `ai-logs/hermes/diag_return_blend.py` — score-blend vs return-blend, book-return correlation
- `ai-logs/hermes/diag_residual_beta.py` — SPY-beta residualization

Self-check: the 2-family runs reproduce the published gate numbers exactly (C ens 0.598 /
best 0.681; B ens 0.557 / best 0.665), so the harness reads are faithful.

## Findings
| | Reading C (lazy_prices) | Reading B (fundamental_value) |
|---|---|---|
| standalone orth leg | 0.553 | 0.413 |
| standalone momentum | 0.618 | 0.618 |
| equal-wt **score** blend (the gate) | 0.598 | 0.557 |
| best single family | 0.681 | 0.665 |
| **return**-blend 50/50 | 0.605 | 0.541 |
| book-return correlation | **0.864** | **0.746** |
| SPY beta (orth / mom) | 0.95 / 0.80 | 0.92 / 0.80 |
| **residual Sharpe after SPY (orth / mom)** | **0.00 / 0.00** | **0.00 / 0.00** |
| residual cross-corr | 0.247 | 0.101 |

## Conclusions (what is PROVEN vs OPEN)
1. **Legs are NOT weak.** 0.41–0.55 standalone, comparable to momentum (0.62). The earlier
   "weak orthogonal signal" guess is refuted by measurement.
2. **No blend/weight/extra-reading can clear the additivity gate.** Book returns are
   0.75–0.86 correlated, so the √2 diversification was never available; return-blend confirms
   it ((0.553+0.618)/√(2·1.864)=0.606 ≈ measured 0.605). The grid already fell back to equal
   weight. Reweighting a strictly-lower-Sharpe, highly-correlated leg only approaches
   momentum-alone from below. **Close blending/weighting and "add another orthogonal reading"
   — proven futile. Do not burn MinBTL N-budget on them.**
3. **The high book-return correlation IS market beta.** Residualizing on SPY collapses the
   cross-correlation (0.86→0.25, 0.75→0.10).
4. **PROVEN: these long-only books carry ZERO market-residual alpha.** Residual Sharpe = 0 for
   every family. Their entire 0.4–0.6 Sharpe is beta — which is why all trail SPY 0.752 (they
   are *diluted* beta). On the 30-name mega-cap universe over 2015–2023 these signals
   (momentum, trend, value, fundamental_value, lazy_prices) have no idiosyncratic edge. The
   DEV_FAILED floor is a TRUE statement about the universe, not a tooling artifact.

## The one remaining lever is a construction/universe redesign — and it is OPEN, not proven
- **Long-short / market-neutral construction.** The only way to (a) strip the beta masking
  everything and (b) access the SHORT leg. CRITICAL caveat (the proxy's blind spot): the
  long-only book CANNOT express the short side, where Reading C noted much of the lazy-prices
  alpha lives (short-the-changers) and where value alpha lives (short-the-expensive). So
  "residual Sharpe 0 on the long-only book" does **not** prove long-short is futile — the proxy
  literally cannot see the short leg. Long-short remains the untested lever; its entire payoff
  rests on the short side. Real product change: borrow cost, shorting constraints, capacity,
  and the "beat SPY" gate semantics all change for a market-neutral book.
- **Universe change.** Move off the 30 mega-caps (where anomalies are arbitraged away) to a
  broader/smaller-cap cross-section where residual alpha survives — consistent with the
  deep-research SESTM small-cap concentration finding.

## Measurement upgrade for any future signal
Gate future candidates on **SPY-residual Sharpe > 0** (cheap, sharp) BEFORE the additivity
gate. A pure-beta signal cannot pass the additivity gate anyway, but the residual screen says
*why* in one number and avoids wasting a pre-registration on a beta-only signal.

Caveats on the residual test: single-factor (SPY only), in-sample, equal-beta over the whole
window — strong directional read, not a final risk model. SESTM stays research-only/blocked
(no committed news fixture, needs API keys).
