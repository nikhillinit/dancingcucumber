# Reading C — Result Record (SEC EDGAR filing-text "Lazy Prices")

Status: DEV RUN COMPLETE. The pre-registered `lazy_prices + momentum` candidate was run through the
dev gate with the reserved holdout BLINDED (`prereg_hash=None`). Outcome: the candidate did **not**
clear the dev gate, and — unlike the failure modes the plan warned about — it is a **genuine,
faithfully-tested negative (mode 1)**, not a harness artifact. Report-only research; the floor
verdict stays `DEV_FAILED`, the reserved holdout is UNTOUCHED, and nothing here authorizes sizing or
capital allocation. Authoritative floor truth: `apps/quant/advisor/backtest/FLOOR_RESULT.md`.

## What is complete
- Frozen fixture `apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv`
  (SHA-256 `372a8518bd4ce40dad41c9fb03ff64502a515bf0d6e9107c63e8815af13e09f1`, 1053 rows).
- Deterministic adapter: `select_asof` + `build_lazy_prices_panel` + `make_lazy_prices_raw`
  (`research/lazy_prices.py`); bench wiring `lazy_prices_candidate_metrics`
  (`research/candidate_floor.py`), holdout-blinded.
- Runner: `ai-logs/hermes/run_ws3d_dev.py` (`PYTHONPATH=apps/quant python ai-logs/hermes/run_ws3d_dev.py`).

## Dev-gate verdict
RUN with `lazy_prices_candidate_metrics(panel, panel_lp, DEFAULT_LAZY_PRICES_CANDIDATE,
prereg_hash=None)` (holdout blinded). Decision is driven off `dev.passed`, NOT `verdict` (which reads
`DEV_FAILED` by construction whenever the holdout is blinded).

- `dev.passed = False` — the candidate does NOT clear the §7.2 dev gate.
- `dev.reasons`: median fold delta not > 0; fewer than 70% positive folds; dev 90% bootstrap LCB of
  delta not > 0; total dev book-Sharpe lift < 0.05.
- Per-fold deltas (ensemble − best-family, purged folds): `[-0.1153, -0.0446, -0.0231, -0.2059]` —
  **0 of 4 folds positive** (0% < the 70% bar); median ≈ -0.080.
- Dev book-Sharpe: ensemble `0.598` < best_family (momentum) `0.681` < SPY (dev window) `0.752`,
  at chosen weights `lazy_prices 0.50 / momentum 0.50`. The orthogonal `lazy_prices + momentum`
  blend fails to beat its best single component — the same structural failure the price-only floor
  and Reading B showed.
- Report-only validation (deflation guard, never folds into the gate): DSR `0.708` < `0.95` bar;
  effective_N `1.02`; per-obs Sharpe `0.0377`; T `1321`; `var_sr 1e-4`. Consistent with the dev failure.

## Is the DEV_FAILED faithful, or an artifact? (the three modes)
The plan flagged three distinguishable failure modes; the diagnostics rule out both artifact modes,
leaving a genuine negative:

1. **Genuine dev failure (THIS result).** Coverage high, signal discriminates, gate still fails.
2. **Silent zeroing (ruled out).** `coverage_in_window = 0.975` — the signal is broadly available,
   not pushed past the window by an availability/fetch-date bug.
3. **Transform collapse (ruled out).** `dev_cross_sectional_dispersion = {min 0.289, median 0.373}`.
   This is the load-bearing detector: the per-asset percentile transform fits each name on its OWN
   history, which could in principle erase the cross-sectional similarity LEVEL (a name constant at
   0.96 and one constant at 0.20 both → conviction 1.0). It did NOT: median per-date cross-sectional
   std of the post-transform conviction is **0.373** (≫ 0), because names are not constant — the
   tight 0.945–0.998 raw band still redistributes into a differentiated cross-section via each name's
   own temporal percentile. The book takes genuinely differentiated positions and still adds no return.

**Non-degeneracy evidence (the real triplet):** coverage `0.975` + cross-sectional dispersion `0.373`
+ orthogonality to momentum `dev_lazy_momentum_corr = -0.018` (well below `tau = 0.40`). These three —
not the power block — are what establish this as a real test.

**On `power.power_limited = False`:** true, but near-vacuous here. It rests on per-fold median
positive-train counts of `[325, 330, 330, 330]` — and similarity is ALWAYS positive (~0.95–0.99), so
the positive-raw count is mechanically ~all rows regardless of signal quality. Unlike Reading B's
book-to-price (where positive-train was genuine evidence), here it proves only that the fixture is
populated, not that the signal carries power. Rely on the triplet above.

## What was NOT tested (scope honesty — do NOT compress this to "Lazy Prices doesn't work")
What got a fair test: **per-asset temporal-percentile of full-document TF cosine, blended 0.5/0.5
with momentum**, on the 30-name floor universe. That construction is a genuine, orthogonal,
non-degenerate signal that fails to add return here. What did NOT get tested is the **canonical
cross-sectional-level Lazy Prices anomaly** — rank names by ABSOLUTE similarity (a cross-sectional
rank transform, not the per-asset temporal percentile every floor family uses), on section-level /
numeric-stripped text rather than full-document TF cosine. Both simplifications are deliberate and
in-scope for WS3D; the healthy dispersion removes the harness-artifact escape and so makes the
in-scope negative solid, but it does NOT refute the canonical anomaly. A faithful cross-sectional-rank
+ section-text reading is an OPERATOR decision with its own prereg and golden-equality argument,
explicitly out of scope here (see plan "Operator decision").

Also note the **leg asymmetry**: the long-flat book takes only the non-changer LONG leg; the cited
18–45bps is a long–SHORT figure, and the short-the-changers leg (where much of the documented alpha
sits) is structurally unavailable, so long-leg-alone efficacy is untested in the evidence.

## Holdout
UNTOUCHED. `HOLDOUT_LEDGER.md` is empty and stays empty — the dev gate did not pass, and with
`prereg_hash=None` the reserved tail was never read on any path. The tail would unlock ONLY if the dev
gate passed AND the caller supplied `prereg_hash == lazy_prices_candidate_run_hash` =
`27c2850bae6e53580548b71b495fe87383cb3e8c39982d50812443cba4819388` (config + fixture bytes); neither
condition was met. A dev pass is necessary but not sufficient; unlocking the shared tail is a separate
operator decision that burns the tail for promotion.

## Cross-reading synthesis (the load-bearing insight)
Two INDEPENDENT orthogonal inputs now fail the dev gate the SAME way on this 30-name universe:
Reading B (book-to-price fundamentals, 1/4 positive folds) and Reading C (filing-text similarity,
0/4) — in both, the orthogonal signal DILUTES momentum and the ensemble underperforms the best single
family. Reading C is additionally confirmed orthogonal (corr -0.018) and cross-sectionally
discriminating (dispersion 0.373), so its failure is not "weak/correlated signal." Two distinct
orthogonal signals failing identically points at **blend non-additivity on this universe** (the 0.5/0.5
equal-weight, beat-the-best-part construction), not a defect specific to either signal. This reframes
the next step: test the BLENDING / weighting and the UNIVERSE, not merely "find another orthogonal
signal."

## Rails
`DEV_FAILED` stays; `backtest/` untouched; `node tools/run-floor.mjs --enforce` exits 1. This is
report-only research and does not authorize sizing or capital allocation. A candidate dev failure is
news about the candidate, not a floor change.
