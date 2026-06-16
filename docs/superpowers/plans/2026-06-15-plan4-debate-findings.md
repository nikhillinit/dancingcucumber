# Plan 4 — Hermes debate findings (second-opinion review) + disposition

**Source:** `npm run hermes -- --workflow debate --phase research --live` (comparators codex+claude+kimi, claude synthesis). Run ID `hermes-2026-06-16T05-39-38-985Z`. kimi leg failed (cp1252 / Windows); codex + claude synthesis carried it. Read-only; no files modified.

**Verdict of the review:** the provided implementations will each pass their own pytest (Codex won't flail), but several have methodological flaws that matter for a *production-release* gate. "Do not use as a production-release gate until hash-locked holdout access, regime proof, full-dev frozen weights, turnover/cost convention, and max-family bootstrap are fixed." For a *necessary-not-sufficient price-only proxy* whose most likely honest outcome is INCONCLUSIVE/UNSUPPORTED (v1 ensemble 0.32 ≪ SPY 0.85), the anti-conservative findings only bite if the gate is near PASS — but a release gate must be trustworthy in that case, so they are addressed.

## Findings → severity → disposition

| # | Finding (plan line) | Severity (release gate) | Disposition | Task |
|---|---|---|---|---|
| 1 | **Holdout unlock is procedural, not enforced.** `floor_metrics(..., prereg_hash="deadbeef")` unlocks the holdout; no check `prereg_hash == config_hash(cfg, fixture)`, no persisted "holdout-used" guard (plan 1456/1484/1555). "Largest release-gate integrity hole." | HIGH | **Fix at T13:** verify the hash in `floor_data_check.py` (it has the fixture path) before passing a real `prereg_hash`; keep the frozen-floor guard on PREREG.md as the artifact-level "touched once". Disclose residual (code trusts caller). | T13 |
| 2 | **Spec "≥2 distinct regimes" not implemented.** dev_gate checks folds/blocks, no regime labels/pass-condition (spec 112; plan 1267). Self-review's "no uncovered requirement" is false. | MED-HIGH | **Disclose at T15 + PREREG:** temporal folds over 2015–2023 (2018/2020/2022 stress) are a regime *proxy*; the floor does NOT assert formal ≥2-regime robustness. Optional lightweight regime split deferred (proportionality brake). | T10/T15/T14 |
| 3 | **`run_dev_sweep` returns LAST-fold weights as "frozen weights."** `chosen` overwritten each fold (plan 1093/1103/1129); holdout uses them. Arbitrary, not a pre-registered full-dev fit. | MED (often nil — equal-weight default) | **Fix at T9:** select weights ONCE on the full dev portion (symmetric with the transform, which run_holdout already fits on full dev); return that as `chosen_weights`. | T9 |
| 4 | **Beat-best-family LCB understated.** Holdout picks best family by point Sharpe (1166) then bootstraps ensemble − that one series (1557). Should account for family-selection uncertainty (max per resample). Anti-conservative → easier PASS. | MED | **Fix at T13 if cheap** (recompute max-family within each bootstrap resample), else **disclose** the selection bias. Small with 2 families; grows with 4–5. | T13 |
| 5 | **Turnover/cost convention.** First-row establishment cost not charged (diff→0.0 not NaN, plan 820-829); `abs(delta).sum()` is full L1 not half-L1 one-way (736). | LOW | **Accept.** Full L1 over-charges cost = conservative (safe for a floor). First-row omission negligible over years. Note only. | T6/T7 |
| 6 | **All-zero scores do not go to cash under `turnover_cap`.** Prose says flat→cash (640); impl scales toward target under the cap (739-740), so book stays invested after every signal goes flat. T6 test passes only because it uses turnover_cap=1.0. | LOW-MED | **Accept + note.** This is realistic turnover-limited de-risking (cannot liquidate instantly). Document the behavior; not a bug for "backtest what ships." | T6/T15 |
| 7 | **Self-review stale.** "No uncovered requirement" (1685) false; "six conditions" (1194) but dev_gate implements five. | LOW (doc) | **Fix doc at T10** (comment) + this file supersedes the self-review's completeness claim. | T10 |

## Clean tasks (no finding): T1 prereg, T2 stats, T3 splits, T4 signals, T5 adequacy, T8 blend, T11 universe, T12 rails. Dispatch as-written.

## Reconcile note
Advisor (pre-debate) endorsed the plan and flagged only test-vs-impl seed risk. The debate found *design* issues the advisor did not. They are not in direct conflict (different layer). Per advisor guidance, re-consult the advisor at **T13** with findings 1 & 4 in hand before finalizing the entrypoint/holdout gate.
