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

Phase B2 was built (T3 mirror → T4 golden, proven faithful → T7 candidate_floor) and the
dev gate was run **report-only with the holdout BLINDED** (`prereg_hash=None`). The operator
chose Option 1 (build to the dev gate, pause before any holdout burn).

## Phase B2 — dev-gate result (real fixture, holdout NEVER touched)

`candidate_metrics(panel, DEFAULT_CANDIDATE, prereg_hash=None)` on `floor_prices.csv`:

| Field | Value |
|---|---|
| **verdict** | **DEV_FAILED** |
| holdout_touched | **false** (reserved tail pristine — rail #5: touched only iff dev passes) |
| universe | formal |
| weights (Rule A) | value 0.50 / momentum 0.50 (no Rule B grid deviation cleared) |
| fold_deltas | [+0.0063, −0.0185, +0.124, −0.1637] (50% positive < 70%; median < 0) |
| ensemble Sharpe | 0.662 vs best-family 0.668 → ensemble does NOT beat its best part (§7.2) |
| ensemble vs SPY | 0.662 vs 0.756 → does not beat SPY |
| DSR (report-only) | 0.755 < 0.95 at N=45 (per-obs Sharpe 0.042) |

dev gate reasons: median fold delta not > 0; fewer than 70% positive folds; dev 90% bootstrap
LCB of delta not > 0; total dev book-Sharpe lift < 0.05.

### Amendment-F6 power/sufficiency report — the verdict is POWER-LIMITED, not a clean refutation

| fold | min positive-train | **median positive-train** | nonzero-transformed coverage |
|---|---:|---:|---:|
| 1 | 0 | **7** | 7.2% |
| 2 | 0 | **7** | 12.3% |
| 3 | 0 | **7** | 17.4% |
| 4 | 0 | **7** | 10.0% |

`power_limited = True` (pre-registered floor = 25). The MEDIAN asset has only ~7 positive value
raw points feeding `fit_percentile_transform` per dev fold (min 0; ~7–17% of test scores fire).
The Task-4 live-in-every-fold guard checked only `assets[0]` (51 positive — atypical); across all
30 assets the typical value-leg fit is **thin**. So this `DEV_FAILED` is a **power artifact, not a
signal refutation** — it does NOT establish "Reading A exhausted," and it does NOT refute classic
LT-reversal or fundamental value (both fixture-infeasible here).

## Decision (Task 9)

- **Clean, documented NEGATIVE — power-limited.** The `value(270)+momentum` blend does not clear
  the dev gate on this 9-yr price-only fixture, but the F6 power report shows the value percentile
  fit is too thin (median 7 positive train points/fold vs a 25 floor) to call it a genuine signal
  verdict. Reading A (price-only intermediate reversal) is **inconclusive for power reasons**.
- **Holdout untouched / NOT burned.** Dev failed, so rail #5 never unlocked the reserved tail;
  `HOLDOUT_LEDGER.md` stays empty. The shared reserved tail remains pristine for a future
  fresh-holdout promotion (Plans 1b/3). No burn decision was required.
- **Frozen floor unchanged.** `npm run advisor-gate` exit 0 (floor `DEV_FAILED`);
  `node tools/run-floor.mjs --enforce` exit 1. No promotion; `allocator.py`/`ensemble_vote` never
  called. The bench is report-only throughout.
- **Recommended next investment: Reading B** (`Task 11` stub) — fundamental value with a timely
  price leg, genuinely orthogonal to price-momentum and not power-starved on a price-only fixture.
  This is the memory's actual lead (`deep-research-orthogonal-signals`) and the right follow-on to
  a power-limited Reading-A negative (Amendment F6 / Task 8 caveat).
