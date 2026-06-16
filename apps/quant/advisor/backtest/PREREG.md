# Floor v2 — pre-registration (IMMUTABLE)

**Recorded:** 2026-06-16, BEFORE the dev sweep and BEFORE the holdout was touched. This file is frozen (guard-protected). The verdict is recorded SEPARATELY in `FLOOR_RESULT.md` — never here.

## Content hash (config + fixture)
```
config_hash(DEFAULT_CONFIG, floor_prices.csv) = 1ad2ed4a27da828055675f49910122580d3898c5a5056a8fd5b4fcbc4410df8c
```
The holdout is unlocked ONLY by this hash (`tools/floor_data_check.py --holdout` recomputes it; any config/fixture change → different hash → pre-registration void).

Fixture: 30 large-caps + SPY, 2015-2023, SHA-256 `d40b9959ba34241a2ea3d60f45516c9f0781718de83f2d77e93de1e23830e2c1` (see `UNIVERSE_RULE.md`).

## Locked hyperparameters (DEFAULT_CONFIG)
folds=5, embargo=5, warmup=200, holdout_frac=0.20, primary_metric=book_sharpe.
max_asset_weight=0.20, gross_cap=1.0, turnover_cap=0.20, cost_per_turn=0.0005.
weight_grid=(0.25,0.50,0.75), train_lift_threshold=0.05.
bootstrap: block=21, draws=2000, seed=12345; dev_lcb=0.90, final_lcb=0.95.
min_universe_formal=20, min_universe_floor=12.

## Margin (LOCKED, never raised from the holdout)
**margin = 0.0** over SPY (rail: margin ≥ 0; default 0.0; no dev-fold sensitivity argued for raising it).

## Candidate order (smallest-first; stop at the first that clears the dev gate)
- **A** trend-alone — standalone reference (one of "the parts").
- **B** momentum-alone — standalone reference (one of "the parts").
- **C** 2-family continuous: (momentum, trend).
- **D** +1 family: (momentum, trend, mean_reversion).
- **E** +2 families: (momentum, trend, mean_reversion, breakout).

## Decision rule (pre-registered)
1. Dev sweep on dev folds only (`prereg_hash=None`) for C → D → E. Stop at the **smallest** candidate whose `dev.passed` is True. If none pass → verdict **UNSUPPORTED**.
2. Universe must be `formal` (≥20) for a formal claim; `micro` → "diagnostic only"; `do_not_run` → stop.
3. If a candidate clears dev: evaluate the holdout **ONCE** with this hash + the winning families. Gate on BOTH §7.2 (`delta_lcb > 0`) AND §7.1 (`spy_lcb > margin`). PASSED iff both clear, else INCONCLUSIVE.
4. Do not iterate. Record the verdict in `FLOOR_RESULT.md`.
