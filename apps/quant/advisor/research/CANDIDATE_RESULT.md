# CANDIDATE_RESULT — Lane B (Reading A: price-only intermediate-term reversal)

Methodology + gate rule were FROZEN in `CANDIDATE_PREREG.md` (commit `ac8b5c2`) **before**
this measurement. `candidate_hash = 578cce4b…69d1d`; fixture SHA-256 `d40b9959…30e2c1`
(`apps/quant/advisor/tests/fixtures/floor_prices.csv`, (2264, 31) = SPY + 30 assets).

> **Faithfulness disclosure:** the bench mirrors the floor exactly except the weight-selector
> family allowlist admits the pre-registered `value` family (`research/candidate_blend.py`,
> `RAW_METRICS | {value}`); frozen `blend.py::select_weights` rejects unknown families. Selection
> math, gates, and DSR are unchanged; equivalence on shared families is proven by the Task-4
> golden. See `CANDIDATE_PREREG.md` → "Implementation-faithfulness disclosure".

## Task 6 — orthogonality kill-gate (real fixture, dev folds only; holdout untouched)

`value(skip=126, lookback=270)` vs neighbors, pooled over dev test folds + assets:

| Neighbor | Raw Pearson | Raw Spearman | Post-transform (GATE surface) |
|---|---:|---:|---:|
| `momentum` (blend partner — diagnostic) | +0.048 | +0.017 | **+0.100** |
| `long_momentum` (gated) | −0.626 | −0.609 | **−0.139** |
| `mean_reversion` (gated) | −0.007 | +0.002 | **−0.004** |

**Gate evaluation (pre-registered rule, Amendment F4):**
- Kill axis = `max(|corr_pt(value, long_momentum)|, |corr_pt(value, mean_reversion)|)`
  = `max(0.139, 0.004)` = **0.139**.
- `0.139 < τ = 0.40` → **PASS.**
- `momentum` clause: post-transform `|corr_pt(value, momentum)|` = 0.100 is LOW → the
  demote-to-"coarse pre-filter" clause is NOT triggered → this is a clean orthogonality PASS.

## Verdict: PASS — `value` is NOT a relabel of a rejected factor on the blend surface

The raw diagnostic shows `value` is meaningfully (negatively) related to `long_momentum`
(raw −0.626 / Spearman −0.609) — as expected from the overlapping formation window and the
negation. But on the **post-transform** surface that actually enters the blend, that drops to
−0.139: the long-flat clamp (raw ≤0 → flat) makes `value` (long past *losers*) and
`long_momentum` (long recent *winners*) fire on largely **disjoint** names → distinct long-flat
books → real diversification, not redundancy (the exact F4 mechanic; verified in unit tests as
raw −1.0 → post −0.3 for a perfect negated relabel). `mean_reversion` is ~orthogonal on both
surfaces. `momentum` (the blend partner) is near-orthogonal post-transform (+0.100), consistent
with the value⊥momentum diversification hypothesis being live.

**Therefore the cheap kill did NOT happen.** This is the less-expected branch of the §5 fork:
the diversification hypothesis survives the kill-gate, and **§7.2 on the real fixture is now the
adjudicator** (a correlated long-only blend can't beat its best member — plan4 — but a genuinely
orthogonal one might; that is the whole hypothesis, to be tested honestly in Phase B2).

## Next (per §5 FORK, PASS branch) — pending operator go-ahead at the §9 checkpoint

Proceed to build Phase B2 in order, fail-fast, golden BEFORE any holdout eval:
`T3 candidate_pipeline → T4 golden + element-wise equality (MANDATORY) → T7 candidate_floor
(F1/F2/F6) → T8 pre-register + RUN eval → T9 decision`.

**Consequential step flagged:** Task 8 touches the shared reserved holdout **once, iff the dev
gate passes** (rail #5), unlocked only by `candidate_run_hash(cfg, fixture)` and logged to
`HOLDOUT_LEDGER.md`. That touch **burns** the shared reserved tail — a passing candidate could
NOT then be promoted on the peeked tail; promotion (Plans 1b/3) would require a FRESH holdout
(Amendment F2). The frozen floor stays `DEV_FAILED` / `--enforce` exit 1 regardless.
