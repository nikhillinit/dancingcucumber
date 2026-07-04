# Plan — Universe change: broader keyless cross-section + residual-alpha go/no-go

Status: DRAFT for operator approval. Report-only research; floor + holdout untouched.
Motivated by `docs/superpowers/notes/2026-06-21-blend-futility-residual-alpha.md` and
memory `[[blend-futility-residual-alpha]]`.

## Why
Measured (2026-06-21, **corrected 2026-06-22**): on the 30 mega-cap universe, no signal family's
long-only book carries positive market-residual alpha. The corrected screen gates on the
**information ratio** `book_sharpe(stream − a·spy)` — the residual KEEPS the OLS intercept, so it
measures idiosyncratic alpha rather than a forced-zero-mean residual. On the floor the
candidate-search readings are negative (value −0.41, fundamental_value −0.32, lazy_prices −0.40,
momentum −0.02). NOTE (T0.2b cross-check, 2026-06-23): the full price-family screen also surfaces
**trend +0.41 on the floor** — so "RED everywhere" is false for the full family set; trend is the
floor's best standalone (0.828 > SPY 0.752) and has in-sample, *un-deflated* residual signal that
the floor already ran through DSR/holdout and rejected (it is NOT a tradable edge — the floor
DEV_FAILED is a blend/additivity failure, not zero-alpha-per-family). (The earlier "residual
Sharpe = 0 for every family" was an OLS-intercept ARTIFACT:
subtracting `a·spy + b` forces residual mean ≡ 0 → Sharpe ≡ 0, vacuously. KILLED — see the
corrected note.) Blending/weighting/more-readings stay proven futile (dev additivity gate AND the
corrected screen). The operator chose **change the universe**: test whether the existing
cross-sectional signals carry market-residual alpha on a broader cross-section (mid-caps
included), where anomalies historically live and there is more idiosyncratic dispersion than
among mega-caps.

## The crux (decided)
- **Keyless is possible** (yfinance + SEC EDGAR), so this does NOT hit the WS2/SESTM API-key wall.
- **Keyless is NOT survivorship-safe**: yfinance drops delisted names; no keyless delisted-price
  source exists. A broader yfinance panel is still survivors-only.
- **Decision:** follow repo precedent — ACCEPT + DISCLOSE survivorship bias (mirror
  `tests/fixtures/UNIVERSE_RULE.md`). Rationale that makes the screen still meaningful: the
  metric we gate on is the **market-neutral residual** (beta-hedged), whose *cross-sectional*
  alpha is far less inflated by survivorship than the long-only *level* is. Survivorship still
  biases it (the delisted worst-names that would populate the short/bottom decile are missing →
  understates short-side alpha), so a positive residual is an UPPER bound and a near-zero
  residual is strong evidence. Disclose exactly this.
- Optional rigor add (keyless, ~2h): a point-in-time S&P-membership guard from the Wikipedia
  "S&P 500 changes" history → filter assets to as-of-date constituents. Reduces look-ahead in
  membership; does not recover delisted prices. Deferred to Phase 1 unless operator wants it in P0.

## Phase 0 — cheap, decisive go/no-go (NO pre-registration, NO holdout, report-only)
Reuses existing signal families and the residual screen already prototyped
(`ai-logs/hermes/diag_residual_beta.py`). Tests price-only signals first — no EDGAR refetch.

- **T0.1 (Hermes).** Commit a keyless, universe-parametrized price fetcher (thin yfinance
  wrapper, mirrors `data/edgar_xbrl_fetch.py:build_fixture` ergonomics) and build a broader
  fixture: current S&P 500 constituents (Wikipedia/keyless list) + SPY, daily adj close,
  2015-2023, same CSV schema as `floor_prices.csv`. Write `UNIVERSE_RULE_BROAD.md` with the
  survivorship disclosure. Drop names lacking full-window coverage (mechanical, no return pruning).
- **T0.2 (DONE — merged PR #12).** Report-only module `backtest/residual_screen.py`: for each
  family, regress its dev book-return stream on SPY and gate on the **information ratio**
  `book_sharpe(stream − a·spy)` (residual KEEPS the OLS intercept → alpha-preserving), tau=0
  strict (`> 0` → GREEN). Unit-tested (`tests/test_residual_screen.py`): pure-beta → residual ≈ 0,
  **orthogonal-alpha → `book_sharpe(res) ≈ book_sharpe(g) > 0`** (the non-vacuity proof), and the
  strict-tau boundary (0.0 → RED). Never imported by a gate. Run it on the broad panel for all 5
  price families (momentum, trend, mean_reversion, breakout, long_momentum) + the price `value`
  family via `python -m advisor.backtest.residual_screen --panel <broad_prices.csv>`.
- **Decision rule (pre-stated; sign-based, tau=0 strict):**
  - **GREEN** — ≥1 price family has **information ratio > 0** on the broad universe → the universe
    was the binding constraint. Proceed to Phase 1 (pre-registered market-neutral candidate).
  - **RED** — **all families' information ratio ≤ 0** → price-only signals are dead **on large-cap
    survivors**. SCOPE CAVEAT (do not overclaim "dead regardless of universe"): the keyless
    current-S&P-500 list, twice-filtered toward big stable names and pruned to full-history
    survivors, is broader than 30 mega-caps but is NOT the mid-/small-cap cross-section where the
    hypothesis (and the SESTM small-cap finding) expects residual alpha — keyless cannot reach that
    survivorship-safely. So a RED leaves two untested levers: (a) the long-SHORT short-leg
    (construction-level, which Phase 0's long-only-residual proxy cannot see) and (b) a genuine
    small/mid-cap universe (needs a keyed, delisting-aware source). Escalate as a separate operator
    decision; do NOT silently continue.
  - **Floor sanity (NOT a stop condition).** On `floor_prices.csv` the candidate-search families
    read NEGATIVE (value −0.41, fundamental_value −0.32, lazy_prices −0.40, momentum −0.02) — that
    is the EXPECTED, correctly-wired floor read, not a miswiring. Do **NOT** treat a negative floor
    residual as "broken → STOP & debug" (that intuition was calibrated to the killed `a·spy + b`
    artifact, which forced ≈ 0). Non-vacuity is proven by the passing
    `test_resid_keeps_orthogonal_alpha` unit test (`book_sharpe(res) ≈ book_sharpe(g) > 0`), NOT by
    the floor residuals being ≈ 0. (T0.2b NOTE: across the FULL price-family set the floor VERDICT
    is GREEN via trend +0.41 — see the result annotation below.)

**T0.2b RESULT (2026-06-23) — supersedes the naive "GREEN → proceed" reading above.** Run recorded
in `docs/superpowers/notes/2026-06-23-broad-universe-residual-screen-result.md`. Broad verdict GREEN
(5/6 families positive). BUT the pre-stated `max-over-families > 0` rule was found
**NON-DISCRIMINATING**: the cross-check shows it ALSO fires GREEN on `floor_prices.csv` (via trend
+0.41), and the floor is the one universe with ground truth — DEV_FAILED. A go/no-go that greenlights
the known-dead case is not a go-signal. The only informative quantity, the per-family floor→broad
delta, is survivorship-confounded *direction-specifically* on exactly the contrarian families that
moved (value −0.41→+0.42 is survivorship's fingerprint, not breadth). **Net verdict: INCONCLUSIVE on
keyless data.** A clean test needs a delisting-aware, point-in-time universe (keyless can't reach it).
Phase 1, if pursued, is an eyes-open hypothesis — NOT screen-justified. Decouple it from this verdict.

## Phase 1 — only if GREEN (sketch; detailed plan written against P0 results)
Pre-registered market-neutral (dollar/beta-neutral long-short) candidate on the broad universe,
its own immutable prereg surface (NOT bolted onto PreRegConfig — see [[validation-gate-floor-internals]]),
holdout discipline + ledger, optional EDGAR-fundamentals scale-up via the existing parametrized
builder, and the PIT-membership guard. New gate semantics: a market-neutral book is judged on
residual/absolute Sharpe, not "beat SPY." Burns toward the MinBTL N-budget — freeze design first.

## Rails / discipline
- Phase 0 touches NO prereg hash, NO holdout, does not modify `backtest/` gate logic
  (`residual_screen.py` is additive + report-only). Floor verdict stays DEV_FAILED;
  `run-floor.mjs --enforce` stays exit 1; HOLDOUT_LEDGER stays empty.
- Builds dispatched via Hermes (short file-pointer task strings per [[hermes-dispatch-windows]]);
  verify Codex's real git diff after each. Claude does intake/context/planning/verification only.
- The three `ai-logs/hermes/diag_*.py` scripts are the read-only evidence behind this plan.
