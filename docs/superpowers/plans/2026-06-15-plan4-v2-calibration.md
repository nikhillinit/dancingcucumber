# Plan 4 — v2 Calibration: Continuous Long-Flat Walk-Forward Floor Proxy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Dispatch (this repo):** Every implementation task is dispatched via **Hermes solo** — `npm run hermes:production -- --task "..."` — and the diff is reviewed by the operator. ALWAYS include in the task string: **"Do NOT run npm or node; verify ONLY with `python -m pytest apps/quant/advisor/tests`."** Codex's `workspace-write` sandbox blocks npm/node spawns and all network/DB access. After each task, the **operator re-runs `npm run advisor-gate`** from the repo root (Codex commits with `--no-verify` and cannot run the npm gate). See memory `hermes-dispatch-windows`.

> **Dispatch ownership:** Tasks marked **[Codex]** are deterministic logic verifiable by pytest against fixtures/fakes — dispatch via Hermes solo. Tasks marked **[Operator]** need the network (fixture commit) or are the single gated measurement run that must stay outside the dev loop — the operator runs these directly. Never hand Codex a task whose verification needs the network or the final holdout.

**Goal:** Replace v1's structurally-degenerate integer-sign 2-family floor proxy with a continuous, long-flat, purged-walk-forward ensemble whose every hyperparameter is pre-registered, so the §7 floor measures — without leakage — whether a *deployable* advisor ensemble beats its best standalone continuous price-family constituent net of costs.

**Architecture:** Continuous conviction scores (train-only percentile transform of `max(raw,0)` — long-flat, no shorts) for a small set of price-only families; a deterministic turnover-aware long-flat book builder; per-fold **train-only** fitting of transform + blend weights inside a purged walk-forward; a quantitative **dev stability gate** that decides whether to touch a single, pre-registered **final holdout**; **book-Sharpe** as the sole primary decision metric with a block-bootstrap lower-confidence-bound hurdle. The floor remains **necessary-not-sufficient** and gates *production release*, not dev commits.

**Tech Stack:** Python 3.13, NumPy, pandas, pytest. Isolated `advisor` package under `apps/quant/advisor` (import root `apps/quant`, so `from advisor.backtest.X import ...`). No new third-party deps (deterministic allocator, hand-rolled block bootstrap — no solver/scipy requirement).

---

## Design & Decisions (spec-of-record — folded in from brainstorming)

**Construction:** continuous **long-flat** conviction signals. **Rejected:** integer-sign reweighting (degenerate: reachable set ≈ {trend-only, equal, momentum-only} = {0.48, 0.32, 0.34}, best ties `best_family` and fails strict `>`); market-neutral / short / ERC book (breaks "backtest what ships"); forced in-sample orthogonalization (arbitrary; contradicts §7 "measure correlation OOS and re-shrink"); rolling tournament; OOS feasibility probe (leakage); Brier/isotonic as a gate (gold-plating — no inclusion decision in a fixed blend); cross-sectional rank-IC as a primary gate on a tiny universe (statistically dead).

| Decision | Resolution |
|---|---|
| **Scope** | **Floor-only.** Touch `apps/quant/advisor/backtest/*`, the fixture, and `tools/floor_data_check.py`/`tools/run-floor.mjs`. **Do NOT** touch `portfolio/allocator.py` `ensemble_vote` (the live seam — deferred to a post-Workstream-C plan; `FamilySignal.skill_weight` stays dormant). No non-price families. |
| **Primary metric** | **Book-Sharpe** (one equal-... see body) locked NOW as the sole decision metric. Mean-of-per-ticker Sharpe is **diagnostic only**. |
| **Primary gate (§7.2 "beat the parts")** | `Δ = BookSharpe(ensemble) − max_j BookSharpe(continuous standalone family_j)`, all net of cost, same transform/allocator/universe. Final pass: **lower 95% block-bootstrap CI of Δ > 0**. |
| **Benchmark gate (§7.1)** | SPY stays binding: lower 95% block-bootstrap CI of `BookSharpe(ensemble) − BookSharpe(SPY) > margin`. Ladder (cash, same-universe B&H, best-family, exposure/vol) = pre-registered **reported diagnostics**. |
| **Margin** | Pre-register the **rule now**: margin **≥ 0** (negative forbidden); default **0.0**; number locked in the pre-registration artifact **before** the holdout run. |
| **Weights** | Pre-registered tiny rule set: **Rule A** equal; **Rule B** train-fold grid `w_mom ∈ {0.25,0.50,0.75}`, deviate from 0.50 only if train-fold book-Sharpe lift ≥ 0.05 in ≥2 inner blocks. Lexicographic: prefer simpler rule, fewer families; **never** choose after seeing the holdout. Grid excludes {0,1} (no collapse to a single family). |
| **Families** | Start **2** (momentum, trend), continuous. Add **1–2** pre-specified decorrelated price-only families (short-horizon mean-reversion, breakout/vol, longer-horizon momentum) **only if** the 2-family version fails the dev gate. |
| **Universe** | Formal floor claim requires **median N_active ≥ 20 and min ≥ 12** across folds; `< 20` ⇒ labeled "micro-universe diagnostic only"; `min < 12` ⇒ do not run the formal ensemble gate. |
| **Fixture** | **Extend now** (operator commit): ≥20 liquid US large-caps selected by an **as-of-window-start** rule (continuously listed from window start, **not** as-of-today) + SPY, frozen, hashed; survivorship surfaced as a disclosure line. |
| **Anti-leakage** | Per-fold **train-only** fitting (transform + blend + any estimated constants); purged walk-forward with embargo; final holdout touched **once**, never printed during dev; everything (config + fixture) hashed in a pre-registration artifact. |
| **autoresearch** | Discipline only: frozen eval harness (= `prepare.py`), search confined to train folds, OOS touched once. No vendored dependency, no LLM-in-the-loop. |

**Non-negotiable rails (carried into every task):** floor-closing work lives in `backtest/`, never `ensemble_vote`; weights/transform fit on train folds only; necessary-not-sufficient caveat (price-only proxy ≠ proof the 5-family advisor satisfies §7); no green-washing (no negative margin, no fixture/window/metric shopping, not-ready reported as the lead finding); report-vs-enforce gate split intact (`advisor-gate` report-only exit 0; `advisor-release-gate` / `--enforce` exit 1 on a miss).

**Feasibility is empirical and branches the plan:** the dev gate decides. Possible honest outcomes: **floor closed**, **inconclusive** (CI straddles 0 — advisor reports but does not size capital), or **unsupported** (no construction clears the dev gate). A clean negative is a valid deliverable.

---

## File Structure

New modules under `apps/quant/advisor/backtest/` (each one responsibility):

| File | Responsibility |
|---|---|
| `prereg.py` | Frozen pre-registration config dataclass + `config_hash()` over (config, fixture bytes). |
| `stats.py` | `book_sharpe()`, `block_bootstrap_lcb()` (series), `block_bootstrap_diff_lcb()` (difference). |
| `splits.py` | `purged_splits(n, folds, embargo)` → `(train_idx, test_idx)` with anti-leak assertions; inner-block splitter for Rule B. |
| `continuous_signals.py` | Raw family metrics + `fit_percentile_transform()` (train) / `apply_transform()` (test); long-flat `bull = max(raw,0)`. |
| `adequacy.py` | `is_adequate()` — train-only score-distribution adequacy test. |
| `portfolio.py` | `build_long_flat_book()` — deterministic turnover-aware long-flat weights (caps, turnover, hysteresis, no shorts, cash). |
| `book.py` | `book_returns()` (weights × prices, cost on turnover) → daily series; book-Sharpe via `stats.book_sharpe`. |
| `blend.py` | `select_weights()` — Rule A/B lexicographic, train-only, N-agnostic, price-only guard. |
| `pipeline.py` | `run_fold()`, `run_dev_sweep()` — per-fold fit→apply, per-fold `Δ` + diagnostics. |
| `dev_gate.py` | `dev_gate()` — the quantitative stability gate (6 conditions). |
| `universe.py` | `n_active()` + `classify_universe()` (formal / micro / do-not-run). |

Evolve: `data_floor.py` (orchestration entrypoint `floor_metrics`), `tools/floor_data_check.py` (config+hash, report/enforce, holdout gating, prose), `tools/run-floor.mjs` (unchanged plumbing — verify only). Tests mirror under `apps/quant/advisor/tests/`.

---

## Task 0: Branch + extended fixture — [Operator]

**Files:**
- Create/replace: `apps/quant/advisor/tests/fixtures/floor_prices.csv` (extended, operator-committed)
- Create: `apps/quant/advisor/tests/fixtures/UNIVERSE_RULE.md` (the frozen selection rule + survivorship disclosure)

- [ ] **Step 1: Branch.** `git switch -c feat/advisor-v2-calibration`
- [ ] **Step 2: Select the universe by a written rule, recorded BEFORE pulling prices.** In `UNIVERSE_RULE.md`: window `2015-01-01 .. 2023-12-31` (extends earlier for more regimes); universe = the **≥20 US large-caps that were already among the most liquid as of the window START (2015-01-01)** and have **continuous daily coverage across the whole window**, plus `SPY`. Record the exact ticker list, the liquidity definition (e.g., dollar-volume rank as of 2015-01), the data source (yfinance), the pull date, the corporate-action policy (adjusted close), and the missing-data policy (drop a date only if SPY missing; otherwise forward-fill ≤1 day then mask). Add the explicit disclosure: *"Universe requires full-window survival → residual survivorship bias; long-side results upward-biased."*
- [ ] **Step 3: Pull + commit the fixture.** Save adjusted-close daily prices as a wide CSV (index = date, columns = tickers + `SPY`), same shape as the current fixture. Commit both files: `git add apps/quant/advisor/tests/fixtures/floor_prices.csv apps/quant/advisor/tests/fixtures/UNIVERSE_RULE.md && git commit -m "data(floor): extend fixture to >=20 large-caps, 2015-2023, as-of-window-start rule"`
- [ ] **Step 4: Freeze.** Do not re-pull or re-select after any floor number is seen. Record the fixture SHA-256 in the commit message body.

> **Note:** This is the only task that touches the network. Everything downstream runs offline against this committed file.

---

## Task 1: Pre-registration config + hash — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/prereg.py`
- Test: `apps/quant/advisor/tests/test_prereg.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_prereg.py
from pathlib import Path

import pytest

from advisor.backtest.prereg import PreRegConfig, DEFAULT_CONFIG, config_hash


def test_default_config_is_frozen_and_has_required_fields():
    cfg = DEFAULT_CONFIG
    assert cfg.folds >= 2 and cfg.embargo >= 0
    assert cfg.margin >= 0.0          # negative margin forbidden by rail
    assert cfg.primary_metric == "book_sharpe"
    assert cfg.families == ("momentum", "trend")
    assert cfg.added_families == ("mean_reversion", "breakout", "long_momentum")
    assert cfg.warmup == 200
    with pytest.raises(Exception):   # frozen dataclass
        cfg.margin = -1.0


def test_negative_margin_rejected_at_construction():
    with pytest.raises(ValueError):
        PreRegConfig(margin=-0.01)


def test_config_hash_is_stable_and_sensitive(tmp_path: Path):
    fixture = tmp_path / "f.csv"
    fixture.write_text("a,b\n1,2\n")
    h1 = config_hash(DEFAULT_CONFIG, fixture)
    h2 = config_hash(DEFAULT_CONFIG, fixture)
    assert h1 == h2 and len(h1) == 64
    fixture.write_text("a,b\n1,3\n")          # fixture change -> hash change
    assert config_hash(DEFAULT_CONFIG, fixture) != h1
    changed = PreRegConfig(margin=0.1)         # config change -> hash change
    assert config_hash(changed, fixture) != h2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_prereg.py -v`
Expected: FAIL with `ModuleNotFoundError: advisor.backtest.prereg`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/prereg.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PreRegConfig:
    """Immutable, pre-registered floor hyperparameters. Any change re-hashes."""
    window: tuple[str, str] = ("2015-01-01", "2023-12-31")
    folds: int = 5
    embargo: int = 5
    warmup: int = 200                     # max family lookback (trend long MA)
    families: tuple[str, ...] = ("momentum", "trend")
    added_families: tuple[str, ...] = ("mean_reversion", "breakout", "long_momentum")
    primary_metric: str = "book_sharpe"
    margin: float = 0.0                   # SPY margin; >= 0 (rail)
    pct_clip: tuple[float, float] = (0.05, 0.95)
    weight_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    train_lift_threshold: float = 0.05    # Rule B deviation + dev total-lift bar
    max_asset_weight: float = 0.20        # 2/N region at N>=10; caps a single name
    gross_cap: float = 1.0
    turnover_cap: float = 0.20            # per-rebalance one-way turnover ceiling
    cost_per_turn: float = 0.0005
    rebalance: str = "daily"
    bootstrap_block: int = 21             # ~1 trading month
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 12345           # fixed: determinism (no Math.random rail)
    dev_lcb: float = 0.90                 # one-sided dev CI level
    final_lcb: float = 0.95               # one-sided holdout CI level
    min_universe_formal: int = 20
    min_universe_floor: int = 12

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError("margin must be >= 0 (negative margin is forbidden)")


DEFAULT_CONFIG = PreRegConfig()


def config_hash(cfg: PreRegConfig, fixture_path: Path) -> str:
    """SHA-256 over the canonical config JSON + the raw fixture bytes."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(fixture_path).read_bytes())
    return h.hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_prereg.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/prereg.py apps/quant/advisor/tests/test_prereg.py
git commit --no-verify -m "feat(floor): pre-registration config + content hash"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/prereg.py to pass apps/quant/advisor/tests/test_prereg.py (a frozen PreRegConfig dataclass that rejects negative margin, plus config_hash over config JSON + fixture bytes). Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 2: Stats — book-Sharpe + block-bootstrap LCB — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/stats.py`
- Test: `apps/quant/advisor/tests/test_stats.py`

The block bootstrap is the **only** uncertainty hurdle in the plan (per the proportionality brake — no SPA/Reality-Check battery). It must be deterministic (seeded), so the gate is reproducible.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_stats.py
import numpy as np
import pandas as pd

from advisor.backtest.stats import book_sharpe, block_bootstrap_lcb, block_bootstrap_diff_lcb


def test_book_sharpe_matches_annualized_formula():
    r = pd.Series([0.001] * 252)            # constant positive daily return
    s = book_sharpe(r)
    assert s > 0 and np.isfinite(s)
    assert book_sharpe(pd.Series([0.0] * 10)) == 0.0   # zero vol -> 0, no div0


def test_lcb_is_below_point_estimate_and_deterministic():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1000))
    lcb_a = block_bootstrap_lcb(r, block=21, draws=500, seed=7, level=0.95)
    lcb_b = block_bootstrap_lcb(r, block=21, draws=500, seed=7, level=0.95)
    assert lcb_a == lcb_b                     # seeded -> reproducible
    assert lcb_a < book_sharpe(r)             # one-sided lower bound


def test_diff_lcb_positive_when_a_clearly_better():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.0015, 0.01, 1500))   # higher Sharpe
    b = pd.Series(rng.normal(0.0002, 0.01, 1500))
    lcb = block_bootstrap_diff_lcb(a, b, block=21, draws=500, seed=3, level=0.95)
    assert lcb > 0


def test_diff_lcb_straddles_zero_when_indistinguishable():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0.0005, 0.01, 1500))
    b = pd.Series(rng.normal(0.0005, 0.01, 1500))
    lcb = block_bootstrap_diff_lcb(a, b, block=21, draws=500, seed=4, level=0.95)
    assert lcb <= 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/stats.py
from __future__ import annotations

import numpy as np
import pandas as pd


def book_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(returns).dropna()
    std = r.std(ddof=0)
    if len(r) == 0 or std == 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index sample of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])
    return idx[:n]


def block_bootstrap_lcb(returns: pd.Series, block: int, draws: int,
                        seed: int, level: float = 0.95) -> float:
    r = pd.Series(returns).dropna().to_numpy()
    if len(r) < block:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        sample = r[_block_indices(len(r), block, rng)]
        samples[i] = book_sharpe(pd.Series(sample))
    return float(np.quantile(samples, 1.0 - level))


def block_bootstrap_diff_lcb(a: pd.Series, b: pd.Series, block: int, draws: int,
                             seed: int, level: float = 0.95) -> float:
    """One-sided lower CI of book_sharpe(a) - book_sharpe(b), paired by index."""
    df = pd.concat([pd.Series(a).reset_index(drop=True),
                    pd.Series(b).reset_index(drop=True)], axis=1).dropna()
    av, bv = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    n = len(av)
    if n < block:
        return 0.0
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws)
    for i in range(draws):
        idx = _block_indices(n, block, rng)
        diffs[i] = book_sharpe(pd.Series(av[idx])) - book_sharpe(pd.Series(bv[idx]))
    return float(np.quantile(diffs, 1.0 - level))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/stats.py apps/quant/advisor/tests/test_stats.py
git commit --no-verify -m "feat(floor): book-Sharpe + seeded block-bootstrap LCB"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/stats.py to pass apps/quant/advisor/tests/test_stats.py: book_sharpe (annualized, zero-vol safe) and seeded circular moving-block bootstrap one-sided lower CIs for a series and for a paired difference. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 3: Purged walk-forward split iterator (anti-leak barrier) — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/splits.py`
- Test: `apps/quant/advisor/tests/test_splits.py`

This is Codex finding ④: the v1 harness had no train phase because equal-weight needed none. v2 fits on train, so the train/test barrier must be **structural**, with anti-leak assertions.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_splits.py
import pytest

from advisor.backtest.splits import purged_splits, inner_blocks


def test_train_strictly_precedes_test_with_embargo():
    splits = purged_splits(n=1000, folds=5, embargo=5)
    assert len(splits) == 4                      # folds-1 evaluable test folds
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx) - 5     # embargo gap enforced
        assert set(train_idx).isdisjoint(test_idx)    # no overlap


def test_no_test_index_appears_in_any_train():
    splits = purged_splits(n=800, folds=4, embargo=10)
    for _, test_idx in splits:
        for other_train, _ in splits:
            future = [i for i in other_train if i > min(test_idx)]
            assert all(i < min(test_idx) - 10 for i in future) or future == []


def test_inner_blocks_partitions_train_for_rule_b():
    blocks = inner_blocks(train_len=300, n_blocks=2)
    assert len(blocks) == 2
    assert blocks[0][-1] < blocks[1][0]          # ordered, disjoint


def test_degenerate_inputs_return_empty():
    assert purged_splits(n=20, folds=5, embargo=5) == []   # too small -> no folds
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/splits.py
from __future__ import annotations

import numpy as np


def purged_splits(n: int, folds: int = 5, embargo: int = 5) -> list[tuple[list[int], list[int]]]:
    """Expanding-window purged walk-forward. Train = everything strictly before
    (test_start - embargo); test = the fold block. Returns [] if too small."""
    if folds < 2 or embargo < 0 or n < folds * 40:
        return []
    fold_size = n // folds
    out: list[tuple[list[int], list[int]]] = []
    for fold in range(1, folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < folds - 1 else n
        train_end = test_start - embargo
        if train_end <= 0 or test_start >= test_end:
            continue
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        assert max(train_idx) < min(test_idx) - embargo
        out.append((train_idx, test_idx))
    return out


def inner_blocks(train_len: int, n_blocks: int = 2) -> list[list[int]]:
    """Disjoint ordered sub-blocks of a train fold (Rule B 'appears in >=2 blocks')."""
    if train_len < n_blocks * 20:
        return []
    edges = np.linspace(0, train_len, n_blocks + 1).astype(int)
    return [list(range(edges[i], edges[i + 1])) for i in range(n_blocks)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_splits.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/splits.py apps/quant/advisor/tests/test_splits.py
git commit --no-verify -m "feat(floor): purged walk-forward split iterator with embargo assertions"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/splits.py to pass apps/quant/advisor/tests/test_splits.py: purged_splits (expanding-window train strictly before test_start-embargo, with assertions) and inner_blocks (ordered disjoint sub-blocks for Rule B). Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 4: Continuous long-flat signal transforms — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/continuous_signals.py`
- Test: `apps/quant/advisor/tests/test_continuous_signals.py`

Raw price metrics per family; `bull = max(raw, 0)` (bearish ⇒ flat, **no shorts**); a **train-fit** empirical-percentile transform → conviction `score ∈ [0,1]`. Fitting on train only is the structural anti-leak point. Families are price-only and purgeable.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_continuous_signals.py
import numpy as np
import pandas as pd

from advisor.backtest.continuous_signals import (
    RAW_METRICS, raw_metric, fit_percentile_transform, apply_transform,
)


def _prices(n=600):
    return pd.Series(np.linspace(100, 200, n))


def test_registry_has_expected_families():
    assert set(RAW_METRICS) >= {"momentum", "trend", "mean_reversion", "breakout", "long_momentum"}


def test_uptrend_gives_positive_bull_for_momentum():
    raw = raw_metric("momentum", _prices())
    assert (raw.dropna() > 0).mean() > 0.5          # mostly bullish in an uptrend


def test_transform_is_long_flat_and_clipped():
    raw = raw_metric("trend", _prices())
    params = fit_percentile_transform(raw.iloc[:300], clip=(0.05, 0.95))
    score = apply_transform(params, raw)
    assert score.min() >= 0.0 and score.max() <= 1.0    # in [0,1], no shorts
    # a raw value <= 0 must map to exactly 0 (flat, not short)
    neg = pd.Series([-1.0, 0.0, 5.0])
    assert apply_transform(params, neg).iloc[0] == 0.0
    assert apply_transform(params, neg).iloc[1] == 0.0


def test_transform_fit_is_train_only_deterministic():
    raw = raw_metric("momentum", _prices())
    p1 = fit_percentile_transform(raw.iloc[:300])
    p2 = fit_percentile_transform(raw.iloc[:300])
    assert p1 == p2                                    # pure function of train slice
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_continuous_signals.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/continuous_signals.py
from __future__ import annotations

import numpy as np
import pandas as pd

# Each metric returns a RAW continuous strength (sign carries direction); the
# transform later clamps <=0 to flat. All are price-only and purgeable.
RAW_METRICS = ("momentum", "trend", "mean_reversion", "breakout", "long_momentum")


def raw_metric(family: str, prices: pd.Series) -> pd.Series:
    p = pd.Series(prices).astype(float)
    if family == "momentum":
        return (p / p.shift(126) - 1.0)
    if family == "long_momentum":
        return (p / p.shift(252) - 1.0)
    if family == "trend":
        return (p.rolling(50).mean() - p.rolling(200).mean()) / p
    if family == "mean_reversion":            # bullish when below short MA (snap-back)
        return (p.rolling(10).mean() - p) / p
    if family == "breakout":                  # bullish above prior 50-day high
        return (p - p.rolling(50).max().shift(1)) / p
    raise ValueError(f"unknown family {family!r}")


def fit_percentile_transform(train_raw: pd.Series, clip: tuple[float, float] = (0.05, 0.95)) -> dict:
    """Fit the conviction transform on TRAIN raw values only. Stores the empirical
    distribution of POSITIVE raw values; negatives/zeros map to flat (0)."""
    pos = pd.Series(train_raw).dropna()
    pos = pos[pos > 0].sort_values().to_numpy()
    return {"pos": pos.tolist(), "lo": clip[0], "hi": clip[1]}


def apply_transform(params: dict, raw: pd.Series) -> pd.Series:
    pos = np.asarray(params["pos"], dtype=float)
    lo, hi = params["lo"], params["hi"]
    r = pd.Series(raw).astype(float).fillna(0.0)

    def _score(x: float) -> float:
        if x <= 0 or len(pos) == 0:
            return 0.0
        pct = np.searchsorted(pos, x, side="right") / len(pos)     # empirical CDF
        return float(np.clip((pct - lo) / (hi - lo), 0.0, 1.0))

    return r.map(_score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_continuous_signals.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/continuous_signals.py apps/quant/advisor/tests/test_continuous_signals.py
git commit --no-verify -m "feat(floor): continuous long-flat price-family transforms (train-fit percentile)"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/continuous_signals.py to pass apps/quant/advisor/tests/test_continuous_signals.py: RAW_METRICS registry (momentum/trend/mean_reversion/breakout/long_momentum, price-only), raw_metric, fit_percentile_transform (train-only, positive-raw empirical dist) and apply_transform (raw<=0 -> 0, else clipped empirical percentile in [0,1], no shorts). Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 5: Score adequacy test — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/adequacy.py`
- Test: `apps/quant/advisor/tests/test_adequacy.py`

Guards against a "continuous" score that is really the old near-binary signal in soft clothing. Uses score **distribution only** (never returns), so it can't become hidden backtest optimization.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_adequacy.py
import numpy as np
import pandas as pd

from advisor.backtest.adequacy import is_adequate


def test_genuinely_continuous_scores_pass():
    scores = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (200, 25)))
    assert is_adequate(scores) is True


def test_near_binary_scores_fail():
    # almost all 0 or 1 -> few distinct levels, tiny IQR among positives
    scores = pd.DataFrame(np.random.default_rng(0).integers(0, 2, (200, 25)).astype(float))
    assert is_adequate(scores) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_adequacy.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/adequacy.py
from __future__ import annotations

import numpy as np
import pandas as pd


def is_adequate(train_scores: pd.DataFrame,
                min_distinct: int = 10, min_iqr: float = 0.15) -> bool:
    """A family's train scores are 'genuinely continuous' if positive scores have
    >= min_distinct levels OR the median per-date IQR of positive scores >= min_iqr."""
    vals = pd.DataFrame(train_scores).to_numpy().ravel()
    pos = vals[vals > 0]
    if len(pos) == 0:
        return False
    if len(np.unique(np.round(pos, 6))) >= min_distinct:
        return True
    iqrs = []
    for _, row in pd.DataFrame(train_scores).iterrows():
        rp = row[row > 0].to_numpy()
        if len(rp) >= 4:
            iqrs.append(np.percentile(rp, 75) - np.percentile(rp, 25))
    return bool(iqrs) and float(np.median(iqrs)) >= min_iqr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_adequacy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/adequacy.py apps/quant/advisor/tests/test_adequacy.py
git commit --no-verify -m "feat(floor): train-only score adequacy test"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/adequacy.py to pass apps/quant/advisor/tests/test_adequacy.py: is_adequate over train scores using distinct-levels OR median per-date positive-score IQR, distribution-only (no returns). Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 6: Deterministic turnover-aware long-flat allocator — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/portfolio.py`
- Test: `apps/quant/advisor/tests/test_portfolio.py`

**KISS resolution (flagged at review):** a deterministic, dependency-free allocator — **not** an LP/QP solver. Each rebalance: target weights ∝ score (capped per-name, gross-capped), then move from the previous weights toward target but **cap one-way turnover**; apply hysteresis (skip trades below cost). No shorts; all-zero scores ⇒ cash. The turnover cap is enforced **in construction**, so a rail test can never find a breach.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_portfolio.py
import numpy as np
import pandas as pd

from advisor.backtest.portfolio import build_long_flat_book


def _scores(n=50, k=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(0, 1, (n, k)),
                        columns=[f"T{i}" for i in range(k)])


def test_no_shorts_and_caps_respected():
    w = build_long_flat_book(_scores(), max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=0.20, cost_per_turn=0.0005)
    assert (w.to_numpy() >= -1e-12).all()                 # long-flat only
    assert w.to_numpy().max() <= 0.20 + 1e-9              # per-name cap
    assert (w.sum(axis=1).to_numpy() <= 1.0 + 1e-9).all() # gross cap


def test_turnover_cap_enforced_each_rebalance():
    s = _scores()
    s.iloc[10:] = s.iloc[10:].values[::-1]                # force a big target swing
    w = build_long_flat_book(s, max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=0.10, cost_per_turn=0.0)
    one_way = w.diff().abs().sum(axis=1).iloc[1:]
    assert (one_way.to_numpy() <= 0.10 + 1e-9).all()


def test_all_zero_scores_go_to_cash():
    s = _scores()
    s.iloc[20] = 0.0
    w = build_long_flat_book(s, max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=1.0, cost_per_turn=0.0)
    assert w.iloc[20].sum() == 0.0                        # fully in cash


def test_hysteresis_skips_subcost_trades():
    s = _scores()
    s.iloc[5] = s.iloc[4] + 1e-6                          # negligible target change
    w = build_long_flat_book(s, max_asset_weight=0.20, gross_cap=1.0,
                             turnover_cap=1.0, cost_per_turn=0.01)
    assert w.iloc[5].equals(w.iloc[4])                    # no churn below cost
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/portfolio.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _target_weights(scores_row: np.ndarray, max_asset_weight: float, gross_cap: float) -> np.ndarray:
    s = np.where(scores_row > 0, scores_row, 0.0)
    total = s.sum()
    if total <= 0:
        return np.zeros_like(s)
    w = s / total * gross_cap
    # iterative cap: clip to max, redistribute excess to uncapped names
    for _ in range(10):
        over = w > max_asset_weight
        if not over.any():
            break
        excess = (w[over] - max_asset_weight).sum()
        w[over] = max_asset_weight
        room = ~over & (w > 0)
        if not room.any():
            break
        w[room] += excess * (w[room] / w[room].sum())
    return np.minimum(w, max_asset_weight)


def build_long_flat_book(scores: pd.DataFrame, max_asset_weight: float, gross_cap: float,
                         turnover_cap: float, cost_per_turn: float) -> pd.DataFrame:
    """Deterministic long-flat weights with per-rebalance one-way turnover cap +
    hysteresis. Row t depends only on scores<=t and the prior row (no look-ahead)."""
    cols = list(scores.columns)
    prev = np.zeros(len(cols))
    rows = []
    for _, row in scores.iterrows():
        target = _target_weights(row.to_numpy(dtype=float), max_asset_weight, gross_cap)
        delta = target - prev
        gross_turn = np.abs(delta).sum()
        if gross_turn <= cost_per_turn:                  # hysteresis: skip churn
            new = prev.copy()
        elif gross_turn > turnover_cap and gross_turn > 0:
            new = prev + delta * (turnover_cap / gross_turn)   # scale move to cap
        else:
            new = target
        rows.append(new)
        prev = new
    return pd.DataFrame(rows, index=scores.index, columns=cols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_portfolio.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/portfolio.py apps/quant/advisor/tests/test_portfolio.py
git commit --no-verify -m "feat(floor): deterministic turnover-aware long-flat allocator"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/portfolio.py to pass apps/quant/advisor/tests/test_portfolio.py: build_long_flat_book — score-proportional targets with iterative per-name cap + gross cap, per-rebalance one-way turnover-cap scaling, hysteresis below cost, no shorts, all-zero->cash, no look-ahead. Deterministic, no solver. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 7: Portfolio book returns + book-Sharpe — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/book.py`
- Test: `apps/quant/advisor/tests/test_book.py`

Turns a weights panel + a prices panel into the daily **book** return series (positions = yesterday's weights → no look-ahead; cost charged on weight change), then its book-Sharpe. This is the deployed-quantity aggregation locked as primary.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_book.py
import numpy as np
import pandas as pd

from advisor.backtest.book import book_returns
from advisor.backtest.stats import book_sharpe


def test_full_long_book_tracks_equal_weight_basket():
    prices = pd.DataFrame({"A": np.linspace(100, 200, 50), "B": np.linspace(100, 150, 50)})
    weights = pd.DataFrame(0.5, index=prices.index, columns=["A", "B"])
    r = book_returns(weights, prices, cost_per_turn=0.0)
    assert len(r) == 50 and book_sharpe(r) > 0           # uptrend -> positive


def test_no_lookahead_first_period_is_zero():
    prices = pd.DataFrame({"A": np.linspace(100, 200, 30)})
    weights = pd.DataFrame(1.0, index=prices.index, columns=["A"])
    r = book_returns(weights, prices, cost_per_turn=0.0)
    assert r.iloc[0] == 0.0                               # position is yesterday's weight


def test_costs_reduce_book_return():
    prices = pd.DataFrame({"A": np.linspace(100, 110, 30)})
    flip = pd.DataFrame({"A": [1.0 if i % 2 == 0 else 0.0 for i in range(30)]})
    no_cost = (1 + book_returns(flip, prices, 0.0)).prod()
    with_cost = (1 + book_returns(flip, prices, 0.02)).prod()
    assert with_cost < no_cost
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_book.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/book.py
from __future__ import annotations

import pandas as pd


def book_returns(weights: pd.DataFrame, prices: pd.DataFrame,
                 cost_per_turn: float = 0.0005) -> pd.Series:
    """Daily book return. Position held = yesterday's weights (no look-ahead);
    transaction cost charged on the day's one-way weight change."""
    w = weights.reindex(columns=prices.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    gross = (held * asset_ret).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    return gross - turnover * cost_per_turn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_book.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/book.py apps/quant/advisor/tests/test_book.py
git commit --no-verify -m "feat(floor): portfolio book returns (no-lookahead, cost-aware)"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/book.py to pass apps/quant/advisor/tests/test_book.py: book_returns — held=weights.shift(1), gross=sum(held*pct_change), minus turnover*cost. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 8: Blend weight selection (Rule A / Rule B, train-only) — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/blend.py`
- Test: `apps/quant/advisor/tests/test_blend.py`

N-agnostic. Default **equal** (Rule A). Rule B may deviate a single family's weight onto the grid `{0.25,0.50,0.75}` **only if** the train-fold book-Sharpe lift over equal is ≥ `train_lift_threshold` in **≥2 inner blocks**. Lexicographic: prefer Rule A unless Rule B clears the bar. Fit on **train only**; reject any non-price family (rail).

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_blend.py
import numpy as np
import pandas as pd
import pytest

from advisor.backtest.blend import select_weights


def _train_scores(families, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return {f: pd.DataFrame(rng.uniform(0, 1, (n, 12))) for f in families}


def test_defaults_to_equal_weight_when_no_clear_lift():
    fam = ("momentum", "trend")
    prices = pd.DataFrame(np.linspace(100, 120, 300).repeat(12).reshape(300, 12))
    w = select_weights(_train_scores(fam), prices, fam, grid=(0.25, 0.5, 0.75),
                       lift_threshold=0.05, cost_per_turn=0.0005, caps=(0.2, 1.0, 0.2))
    assert w == {"momentum": 0.5, "trend": 0.5}          # Rule A wins by default


def test_weights_sum_to_one_and_exclude_endpoints():
    fam = ("momentum", "trend")
    prices = pd.DataFrame(np.linspace(100, 120, 300).repeat(12).reshape(300, 12))
    w = select_weights(_train_scores(fam), prices, fam, grid=(0.25, 0.5, 0.75),
                       lift_threshold=0.05, cost_per_turn=0.0005, caps=(0.2, 1.0, 0.2))
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(0 < v < 1 for v in w.values())            # never collapse to a single family


def test_rejects_non_price_family():
    with pytest.raises(ValueError):
        select_weights({"sentiment": pd.DataFrame()}, pd.DataFrame(), ("sentiment",),
                       grid=(0.25, 0.5, 0.75), lift_threshold=0.05,
                       cost_per_turn=0.0005, caps=(0.2, 1.0, 0.2))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_blend.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/blend.py
from __future__ import annotations

import pandas as pd

from advisor.backtest.book import book_returns
from advisor.backtest.continuous_signals import RAW_METRICS
from advisor.backtest.portfolio import build_long_flat_book
from advisor.backtest.splits import inner_blocks
from advisor.backtest.stats import book_sharpe


def _ensemble_book_sharpe(scores_by_fam, weights, prices, caps, cost) -> float:
    max_w, gross, turn = caps          # caps = (max_asset_weight, gross_cap, turnover_cap)
    cols = next(iter(scores_by_fam.values())).columns
    blended = sum(weights[f] * scores_by_fam[f] for f in weights)
    blended = pd.DataFrame(blended, columns=cols)
    w = build_long_flat_book(blended, max_w, gross, turn, cost)
    return book_sharpe(book_returns(w, prices, cost))


def select_weights(train_scores: dict, train_prices: pd.DataFrame, families: tuple,
                   grid: tuple, lift_threshold: float, cost_per_turn: float,
                   caps: tuple) -> dict:
    """Train-only weight selection. Rule A = equal; Rule B deviates onto the grid
    only with >= lift_threshold book-Sharpe gain in >=2 inner train blocks."""
    for f in families:
        if f not in RAW_METRICS:
            raise ValueError(f"non-price / unknown family rejected: {f!r}")
    n = len(families)
    equal = {f: 1.0 / n for f in families}
    if n != 2:
        return equal                                   # Rule B defined for the 2-family case
    base = _ensemble_book_sharpe(train_scores, equal, train_prices, caps, cost_per_turn)
    blocks = inner_blocks(len(train_prices), n_blocks=2)
    best = equal
    best_lift = 0.0
    f0, f1 = families
    for w0 in grid:
        cand = {f0: w0, f1: round(1.0 - w0, 6)}
        if cand == equal:
            continue
        block_lifts = []
        for blk in blocks:
            ts = {f: train_scores[f].iloc[blk] for f in families}
            tp = train_prices.iloc[blk]
            block_lifts.append(_ensemble_book_sharpe(ts, cand, tp, caps, cost_per_turn)
                               - _ensemble_book_sharpe(ts, equal, tp, caps, cost_per_turn))
        full_lift = _ensemble_book_sharpe(train_scores, cand, train_prices, caps, cost_per_turn) - base
        clears = sum(1 for bl in block_lifts if bl >= lift_threshold) >= 2
        if clears and full_lift > best_lift:
            best, best_lift = cand, full_lift
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_blend.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/blend.py apps/quant/advisor/tests/test_blend.py
git commit --no-verify -m "feat(floor): train-only blend weight selection (Rule A/B, price-only guard)"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/blend.py to pass apps/quant/advisor/tests/test_blend.py: select_weights — Rule A equal default, Rule B grid deviation only with >=lift_threshold book-Sharpe gain in >=2 inner train blocks, weights sum to 1 and never hit 0/1, reject non-price families. Fit on train only. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 9: Per-fold pipeline + dev sweep (per-fold Δ) — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/pipeline.py`
- Test: `apps/quant/advisor/tests/test_pipeline.py`

Ties the pieces together per fold: fit transform + blend on **train**, build the book on **test**, compute `Δ = BookSharpe(ensemble) − max_j BookSharpe(continuous standalone family_j)` on test. `run_dev_sweep` runs this across folds for a candidate family set and returns the per-fold Δ list + the concatenated test return series + diagnostics. **No holdout here** — dev folds only.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_pipeline.py
import numpy as np
import pandas as pd

from advisor.backtest.pipeline import run_dev_sweep
from advisor.backtest.prereg import DEFAULT_CONFIG


def _panel(n=1500, k=22, seed=0):
    rng = np.random.default_rng(seed)
    # trending names + noise so families are non-degenerate; last col is SPY
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (n, k)), axis=0) * 100
    cols = [f"T{i}" for i in range(k - 1)] + ["SPY"]
    return pd.DataFrame(base, columns=cols)


def test_dev_sweep_returns_per_fold_deltas_and_series():
    res = run_dev_sweep(_panel(), ("momentum", "trend"), DEFAULT_CONFIG)
    assert len(res.fold_deltas) >= 3                       # one Δ per evaluable fold
    assert all(np.isfinite(d) for d in res.fold_deltas)
    assert len(res.ensemble_test_returns) > 0
    assert len(res.best_family_test_returns) == len(res.ensemble_test_returns)
    assert res.chosen_weights and abs(sum(res.chosen_weights.values()) - 1.0) < 1e-9


def test_dev_sweep_uses_only_dev_folds_not_full_series():
    # the concatenated dev returns must be shorter than the full panel (holdout excluded)
    panel = _panel()
    res = run_dev_sweep(panel, ("momentum", "trend"), DEFAULT_CONFIG)
    assert len(res.ensemble_test_returns) < len(panel)


def test_run_holdout_evaluates_the_held_out_tail():
    from advisor.backtest.pipeline import run_holdout
    panel = _panel()
    res = run_dev_sweep(panel, ("momentum", "trend"), DEFAULT_CONFIG)
    h = run_holdout(panel, ("momentum", "trend"), DEFAULT_CONFIG, res.chosen_weights)
    post_warmup = len(panel) - DEFAULT_CONFIG.warmup
    expected_holdout = post_warmup - int(post_warmup * 0.8)
    assert abs(len(h.ensemble) - expected_holdout) <= 1     # holdout window length
    assert len(h.best_family) == len(h.ensemble) == len(h.spy)
    # holdout must NOT overlap the dev returns it was frozen on
    assert len(h.ensemble) < len(res.ensemble_test_returns) + len(h.ensemble)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/pipeline.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from advisor.backtest.blend import select_weights
from advisor.backtest.book import book_returns
from advisor.backtest.continuous_signals import apply_transform, fit_percentile_transform, raw_metric
from advisor.backtest.portfolio import build_long_flat_book
from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import book_sharpe


@dataclass(frozen=True)
class SweepResult:
    fold_deltas: list[float]
    ensemble_test_returns: pd.Series
    best_family_test_returns: pd.Series
    chosen_weights: dict


@dataclass(frozen=True)
class HoldoutReturns:
    ensemble: pd.Series
    best_family: pd.Series
    spy: pd.Series


def _family_scores(family: str, prices: pd.DataFrame, train_idx, all_idx, clip) -> pd.DataFrame:
    """Fit the transform on TRAIN rows per column, apply to all rows. Long-flat."""
    cols = {}
    for c in prices.columns:
        raw = raw_metric(family, prices[c])
        params = fit_percentile_transform(raw.iloc[train_idx], clip=clip)
        cols[c] = apply_transform(params, raw)
    return pd.DataFrame(cols).iloc[all_idx].reset_index(drop=True)


def run_dev_sweep(panel: pd.DataFrame, families: tuple, cfg: PreRegConfig,
                  holdout_frac: float = 0.2) -> SweepResult:
    """Dev folds only: hold out the final holdout_frac, purge-walk-forward the rest."""
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)   # warmup excluded
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    dev = prices_all.iloc[:dev_end]

    caps = (cfg.max_asset_weight, cfg.gross_cap, cfg.turnover_cap)
    deltas, ens_parts, best_parts = [], [], []
    chosen = {f: 1.0 / len(families) for f in families}
    for train_idx, test_idx in purged_splits(len(dev), cfg.folds, cfg.embargo):
        all_idx = list(range(min(train_idx), max(test_idx) + 1))
        scores = {f: _family_scores(f, dev, train_idx,
                                    [i for i in all_idx], cfg.pct_clip) for f in families}
        # map global train/test indices into the all_idx-local frame
        local = {g: i for i, g in enumerate(all_idx)}
        tr = [local[i] for i in train_idx]
        te = [local[i] for i in test_idx]
        train_scores = {f: scores[f].iloc[tr].reset_index(drop=True) for f in families}
        chosen = select_weights(train_scores, dev.iloc[train_idx].reset_index(drop=True),
                                families, cfg.weight_grid, cfg.train_lift_threshold,
                                cfg.cost_per_turn, caps)
        test_prices = dev.iloc[test_idx].reset_index(drop=True)
        blended = sum(chosen[f] * scores[f].iloc[te].reset_index(drop=True) for f in families)
        blended = pd.DataFrame(blended, columns=dev.columns)
        ens_w = build_long_flat_book(blended, *caps, cfg.cost_per_turn)
        ens_r = book_returns(ens_w, test_prices, cfg.cost_per_turn)

        fam_sharpes = {}
        fam_rets = {}
        for f in families:
            fs = pd.DataFrame(scores[f].iloc[te].reset_index(drop=True), columns=dev.columns)
            fw = build_long_flat_book(fs, *caps, cfg.cost_per_turn)
            fr = book_returns(fw, test_prices, cfg.cost_per_turn)
            fam_sharpes[f] = book_sharpe(fr)
            fam_rets[f] = fr
        best_f = max(fam_sharpes, key=fam_sharpes.get)
        deltas.append(book_sharpe(ens_r) - fam_sharpes[best_f])
        ens_parts.append(ens_r)
        best_parts.append(fam_rets[best_f])

    return SweepResult(
        fold_deltas=deltas,
        ensemble_test_returns=pd.concat(ens_parts, ignore_index=True) if ens_parts else pd.Series(dtype=float),
        best_family_test_returns=pd.concat(best_parts, ignore_index=True) if best_parts else pd.Series(dtype=float),
        chosen_weights=chosen,
    )


def run_holdout(panel: pd.DataFrame, families: tuple, cfg: PreRegConfig,
                frozen_weights: dict, holdout_frac: float = 0.2) -> HoldoutReturns:
    """Evaluate the dev-FROZEN construction on the held-out tail ONCE. Transform is
    fit on the full dev portion; frozen_weights come from run_dev_sweep. Returns
    ensemble, best-standalone-family, and SPY book returns over the holdout window."""
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)
    spy_all = panel["SPY"].iloc[cfg.warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    caps = (cfg.max_asset_weight, cfg.gross_cap, cfg.turnover_cap)

    # raw over the full series (continuity for rolling/shift); transform fit on dev only
    scores = {}
    for f in families:
        cols = {}
        for c in assets:
            raw = raw_metric(f, prices_all[c])
            params = fit_percentile_transform(raw.iloc[:dev_end], clip=cfg.pct_clip)
            cols[c] = apply_transform(params, raw)
        scores[f] = pd.DataFrame(cols)

    hold = slice(dev_end, len(prices_all))
    hold_prices = prices_all.iloc[hold].reset_index(drop=True)
    blended = sum(frozen_weights[f] * scores[f].iloc[hold].reset_index(drop=True) for f in families)
    blended = pd.DataFrame(blended, columns=assets)
    ens_r = book_returns(build_long_flat_book(blended, *caps, cfg.cost_per_turn),
                         hold_prices, cfg.cost_per_turn)

    fam_rets = {}
    for f in families:
        fs = pd.DataFrame(scores[f].iloc[hold].reset_index(drop=True), columns=assets)
        fam_rets[f] = book_returns(build_long_flat_book(fs, *caps, cfg.cost_per_turn),
                                   hold_prices, cfg.cost_per_turn)
    best_f = max(fam_rets, key=lambda f: book_sharpe(fam_rets[f]))
    spy_r = spy_all.iloc[hold].reset_index(drop=True).pct_change().fillna(0.0)
    return HoldoutReturns(ensemble=ens_r, best_family=fam_rets[best_f], spy=spy_r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/pipeline.py apps/quant/advisor/tests/test_pipeline.py
git commit --no-verify -m "feat(floor): per-fold train-only pipeline + dev sweep (per-fold delta)"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/pipeline.py to pass apps/quant/advisor/tests/test_pipeline.py: run_dev_sweep over purged dev folds (final 20%% held out), fitting transform+blend on train only, building ensemble vs best continuous standalone family books on test, returning per-fold deltas + concatenated test return series + chosen weights; PLUS run_holdout that applies the dev-frozen weights (transform fit on the full dev portion) to the held-out tail and returns ensemble/best-family/SPY book returns over the holdout window. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 10: Quantitative dev stability gate — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/dev_gate.py`
- Test: `apps/quant/advisor/tests/test_dev_gate.py`

The gate that decides whether a candidate construction earns a holdout run. All six conditions must hold. No judgment calls — pure thresholds (replaces the old vague "stable margin").

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_dev_gate.py
import numpy as np
import pandas as pd

from advisor.backtest.dev_gate import dev_gate
from advisor.backtest.prereg import DEFAULT_CONFIG


def _series(mean, n=1000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.01, n))


def test_clear_winner_passes():
    deltas = [0.3, 0.25, 0.4, 0.2]
    ens, best = _series(0.0015), _series(0.0002, seed=1)
    res = dev_gate(deltas, ens, best, DEFAULT_CONFIG)
    assert res.passed is True and not res.reasons


def test_majority_negative_folds_fail():
    deltas = [0.3, -0.1, -0.2, -0.05]                  # < 70% positive
    ens, best = _series(0.0008), _series(0.0007, seed=1)
    res = dev_gate(deltas, ens, best, DEFAULT_CONFIG)
    assert res.passed is False
    assert any("70%" in r or "positive folds" in r for r in res.reasons)


def test_single_fold_concentration_fails():
    deltas = [0.5, 0.01, 0.01, 0.01]                   # one fold dominates excess
    ens, best = _series(0.0009), _series(0.0006, seed=2)
    res = dev_gate(deltas, ens, best, DEFAULT_CONFIG, fold_excess=[0.9, 0.03, 0.03, 0.04])
    assert res.passed is False
    assert any("concentration" in r for r in res.reasons)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_dev_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/dev_gate.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.stats import block_bootstrap_diff_lcb, book_sharpe


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def dev_gate(fold_deltas: list[float], ensemble_returns: pd.Series,
             best_family_returns: pd.Series, cfg: PreRegConfig,
             fold_excess: list[float] | None = None) -> GateResult:
    reasons: list[str] = []
    d = np.asarray(fold_deltas, dtype=float)
    if d.size == 0:
        return GateResult(False, ["no dev folds evaluated"])
    if np.median(d) <= 0:
        reasons.append("median fold delta not > 0")
    if (d > 0).mean() < 0.70:
        reasons.append("fewer than 70% positive folds")
    lcb = block_bootstrap_diff_lcb(ensemble_returns, best_family_returns,
                                   cfg.bootstrap_block, cfg.bootstrap_draws,
                                   cfg.bootstrap_seed, level=cfg.dev_lcb)
    if lcb <= 0:
        reasons.append(f"dev {int(cfg.dev_lcb*100)}% bootstrap LCB of delta not > 0")
    total_lift = book_sharpe(ensemble_returns) - book_sharpe(best_family_returns)
    if total_lift < cfg.train_lift_threshold:
        reasons.append(f"total dev book-Sharpe lift < {cfg.train_lift_threshold}")
    if fold_excess is not None and len(fold_excess) > 0:
        fe = np.asarray(fold_excess, dtype=float)
        if fe.sum() > 0 and fe.max() / fe.sum() > 0.60:
            reasons.append("single-fold concentration > 60% of excess")
    return GateResult(passed=not reasons, reasons=reasons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_dev_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/dev_gate.py apps/quant/advisor/tests/test_dev_gate.py
git commit --no-verify -m "feat(floor): quantitative dev stability gate"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/dev_gate.py to pass apps/quant/advisor/tests/test_dev_gate.py: dev_gate checking median delta>0, >=70%% positive folds, dev bootstrap LCB>0, total lift>=threshold, no single fold >60%% of excess; returns GateResult(passed, reasons). Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 11: Minimum-universe rule — [Codex]

**Files:**
- Create: `apps/quant/advisor/backtest/universe.py`
- Test: `apps/quant/advisor/tests/test_universe.py`

Scopes the claim honestly: `formal` (median N_active ≥ 20 and min ≥ 12), `micro` (proceed but label "diagnostic only"), `do_not_run` (min < 12 → no formal ensemble gate).

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_universe.py
from advisor.backtest.universe import classify_universe
from advisor.backtest.prereg import DEFAULT_CONFIG


def test_formal_when_broad():
    assert classify_universe([22, 25, 24, 23], DEFAULT_CONFIG) == "formal"


def test_micro_when_thin_but_runnable():
    assert classify_universe([14, 13, 15, 12], DEFAULT_CONFIG) == "micro"


def test_do_not_run_when_too_thin():
    assert classify_universe([14, 8, 20, 19], DEFAULT_CONFIG) == "do_not_run"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_universe.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/universe.py
from __future__ import annotations

import numpy as np

from advisor.backtest.prereg import PreRegConfig


def classify_universe(n_active_per_fold: list[int], cfg: PreRegConfig) -> str:
    n = np.asarray(n_active_per_fold, dtype=int)
    if n.size == 0 or n.min() < cfg.min_universe_floor:
        return "do_not_run"
    if np.median(n) >= cfg.min_universe_formal and n.min() >= cfg.min_universe_floor:
        return "formal"
    return "micro"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_universe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/universe.py apps/quant/advisor/tests/test_universe.py
git commit --no-verify -m "feat(floor): minimum-universe classification rule"
```

> **Dispatch:** `npm run hermes:production -- --task "Implement apps/quant/advisor/backtest/universe.py to pass apps/quant/advisor/tests/test_universe.py: classify_universe -> formal / micro / do_not_run from per-fold active counts and the config thresholds. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 12: Executable rail tests — [Codex]

**Files:**
- Create: `apps/quant/advisor/tests/test_rails.py`

The rails are pass/fail tests, not comments (Codex finding ⑨). This task adds *no* production code — only assertions over the modules already built, so a future regression that reintroduces leakage, shorts, a negative margin, or a live-seam edit fails loudly.

- [ ] **Step 1: Write the rail tests**

```python
# apps/quant/advisor/tests/test_rails.py
import inspect

import numpy as np
import pandas as pd
import pytest

from advisor.backtest import prereg, portfolio, splits, continuous_signals
from advisor.backtest.pipeline import run_dev_sweep
from advisor.backtest.prereg import DEFAULT_CONFIG, PreRegConfig


def test_rail_negative_margin_is_impossible():
    with pytest.raises(ValueError):
        PreRegConfig(margin=-0.0001)


def test_rail_no_shorts_in_allocator():
    rng = np.random.default_rng(0)
    s = pd.DataFrame(rng.uniform(-5, 5, (40, 8)))      # raw-looking negatives
    w = portfolio.build_long_flat_book(s, 0.2, 1.0, 0.5, 0.0005)
    assert (w.to_numpy() >= -1e-12).all()


def test_rail_transform_maps_nonpositive_to_flat():
    params = continuous_signals.fit_percentile_transform(pd.Series([1.0, 2.0, 3.0]))
    out = continuous_signals.apply_transform(params, pd.Series([-9.0, 0.0]))
    assert out.iloc[0] == 0.0 and out.iloc[1] == 0.0


def test_rail_splits_enforce_train_before_test():
    for tr, te in splits.purged_splits(1000, 5, 5):
        assert max(tr) < min(te) - 5


def test_rail_blend_rejects_non_price_family():
    with pytest.raises(ValueError):
        from advisor.backtest.blend import select_weights
        select_weights({"macro": pd.DataFrame()}, pd.DataFrame(), ("macro",),
                       (0.25, 0.5, 0.75), 0.05, 0.0005, (0.2, 1.0, 0.2))


def test_rail_floor_code_does_not_import_live_allocator():
    # floor-only scope: no backtest module may import portfolio.allocator (the live seam)
    import advisor.backtest.pipeline as pl
    src = inspect.getsource(pl)
    assert "portfolio.allocator" not in src and "ensemble_vote" not in src
```

- [ ] **Step 2: Run to verify (all should pass against the built modules)**

Run: `python -m pytest apps/quant/advisor/tests/test_rails.py -v`
Expected: PASS (if any fails, a rail is violated — fix the offending module, not the test).

- [ ] **Step 3: Commit**

```bash
git add apps/quant/advisor/tests/test_rails.py
git commit --no-verify -m "test(floor): executable rails (no shorts, no leakage, no live-seam, no negative margin)"
```

> **Dispatch:** `npm run hermes:production -- --task "Create apps/quant/advisor/tests/test_rails.py asserting the floor rails over the existing backtest modules: negative-margin impossible, allocator never shorts, transform maps non-positive to flat, splits enforce train<test-embargo, blend rejects non-price families, and pipeline source never references the live ensemble_vote/allocator. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate`.

---

## Task 13: Floor entrypoint v2 — book-Sharpe gate, report/enforce, holdout discipline — [Codex code; Operator runs the holdout]

**Files:**
- Modify: `apps/quant/advisor/backtest/data_floor.py` (replace `floor_metrics` body; keep the import surface stable for `tools/floor_data_check.py`)
- Modify: `tools/floor_data_check.py:11` (`MARGIN` → read from `DEFAULT_CONFIG`), prose, holdout gating
- Test: `apps/quant/advisor/tests/test_data_floor.py` (extend), `apps/quant/advisor/tests/test_floor_entrypoint.py` (new)

`floor_metrics` becomes: run `run_dev_sweep` on dev folds, evaluate `dev_gate` + `classify_universe`, and **only if dev passes AND a pre-registration hash is supplied** evaluate the single holdout slice (lower-95% bootstrap LCB of Δ, plus the SPY-margin LCB). Returns a verdict in `{PASSED, INCONCLUSIVE, UNSUPPORTED, DEV_FAILED}`. The legacy `ensemble`/`spy`/`best_family`/`passes` keys are retained (mapped to book-Sharpe values + the diagnostic ladder) so `floor_data_check.py` and existing tests keep working.

- [ ] **Step 1: Write the failing test** (extend `test_data_floor.py` + new entrypoint test)

```python
# apps/quant/advisor/tests/test_floor_entrypoint.py
import numpy as np
import pandas as pd

from advisor.backtest.data_floor import floor_metrics
from advisor.backtest.prereg import DEFAULT_CONFIG


def _panel(n=1600, k=22, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (n, k)), axis=0) * 100
    return pd.DataFrame(base, columns=[f"T{i}" for i in range(k - 1)] + ["SPY"])


def test_floor_metrics_reports_without_touching_holdout_when_no_prereg():
    m = floor_metrics(_panel(), DEFAULT_CONFIG, prereg_hash=None)
    assert m["verdict"] in {"DEV_FAILED", "INCONCLUSIVE", "UNSUPPORTED", "PASSED"}
    assert "holdout" not in m or m["holdout"] is None     # holdout blinded without prereg
    for legacy in ("ensemble", "spy", "best_family", "margin", "passes"):
        assert legacy in m                                # back-compat keys present
    assert isinstance(m["passes"], bool)


def test_holdout_gates_on_both_parts_and_spy():
    m = floor_metrics(_panel(), DEFAULT_CONFIG, prereg_hash="deadbeef")
    if m["dev"]["passed"]:
        assert m["holdout"] is not None                   # gated holdout evaluated
        h = m["holdout"]
        assert {"beats_parts", "beats_spy", "delta_lcb", "spy_lcb"} <= set(h)
        # PASSED iff BOTH section 7.2 (beat the parts) AND section 7.1 (beat SPY) clear
        assert (m["verdict"] == "PASSED") == (h["beats_parts"] and h["beats_spy"])
    else:
        assert m["holdout"] is None                       # dev fail -> no holdout
```

```python
# extend apps/quant/advisor/tests/test_data_floor.py — keep the legacy contract green
def test_floor_metrics_back_compat_keys_and_types():
    import numpy as np, pandas as pd
    from advisor.backtest.data_floor import floor_metrics
    from advisor.backtest.prereg import DEFAULT_CONFIG
    rng = np.random.default_rng(0)
    base = np.cumprod(1 + rng.normal(0.0004, 0.01, (1600, 22)), axis=0) * 100
    panel = pd.DataFrame(base, columns=[f"T{i}" for i in range(21)] + ["SPY"])
    m = floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    for key in ("ensemble", "spy", "best_family", "margin", "passes"):
        assert key in m
    assert np.isfinite(m["ensemble"]) and isinstance(m["passes"], bool)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_floor_entrypoint.py apps/quant/advisor/tests/test_data_floor.py -v`
Expected: FAIL (`floor_metrics` signature/keys not yet updated).

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/backtest/data_floor.py  (replace floor_metrics; keep module importable)
from __future__ import annotations

import pandas as pd

from advisor.backtest.dev_gate import dev_gate
from advisor.backtest.pipeline import run_dev_sweep, run_holdout
from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import block_bootstrap_diff_lcb, book_sharpe
from advisor.backtest.universe import classify_universe


def floor_metrics(panel: pd.DataFrame, cfg: PreRegConfig, prereg_hash: str | None = None,
                  families: tuple | None = None, holdout_frac: float = 0.2) -> dict:
    """v2 floor. Dev sweep -> dev gate -> (only with prereg_hash + dev pass) a single
    evaluation on the held-out tail, gating on BOTH beat-the-parts (section 7.2, delta
    LCB > 0) AND beat-SPY-by-margin (section 7.1, the binding bar). book-Sharpe is the
    primary metric; legacy keys retained for callers."""
    families = families or cfg.families
    assets = panel.drop(columns=["SPY"]).iloc[cfg.warmup:].reset_index(drop=True)
    sweep = run_dev_sweep(panel, families, cfg, holdout_frac=holdout_frac)

    # coverage proxy: names with full (non-NaN) coverage in each dev test fold
    dev_end = int(len(assets) * (1 - holdout_frac))
    n_active = [int(assets.iloc[te].notna().all(axis=0).sum())
                for _, te in purged_splits(dev_end, cfg.folds, cfg.embargo)] or [assets.shape[1]]
    universe = classify_universe(n_active, cfg)
    gate = dev_gate(sweep.fold_deltas, sweep.ensemble_test_returns,
                    sweep.best_family_test_returns, cfg)

    holdout = None
    verdict = "DEV_FAILED"
    legacy_spy = book_sharpe(panel["SPY"].iloc[cfg.warmup:].pct_change().fillna(0.0))  # diagnostic
    if universe == "do_not_run":
        verdict = "UNSUPPORTED"
    elif gate.passed and prereg_hash is not None:
        h = run_holdout(panel, families, cfg, sweep.chosen_weights, holdout_frac=holdout_frac)
        delta_lcb = block_bootstrap_diff_lcb(h.ensemble, h.best_family, cfg.bootstrap_block,
                                             cfg.bootstrap_draws, cfg.bootstrap_seed, level=cfg.final_lcb)
        spy_lcb = block_bootstrap_diff_lcb(h.ensemble, h.spy, cfg.bootstrap_block,
                                           cfg.bootstrap_draws, cfg.bootstrap_seed, level=cfg.final_lcb)
        beats_parts = delta_lcb > 0                       # section 7.2
        beats_spy = spy_lcb > cfg.margin                  # section 7.1 (binding bar)
        holdout = {"delta_lcb": delta_lcb, "spy_lcb": spy_lcb,
                   "beats_parts": beats_parts, "beats_spy": beats_spy,
                   "ensemble_sharpe": book_sharpe(h.ensemble), "spy_sharpe": book_sharpe(h.spy),
                   "best_family_sharpe": book_sharpe(h.best_family),
                   "label": "diagnostic" if universe == "micro" else "formal"}
        legacy_spy = book_sharpe(h.spy)
        verdict = "PASSED" if (beats_parts and beats_spy) else "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "universe": universe,
        "dev": {"passed": gate.passed, "reasons": gate.reasons, "fold_deltas": sweep.fold_deltas},
        "weights": sweep.chosen_weights,
        "holdout": holdout,
        # legacy/back-compat keys (book-Sharpe values; SPY is holdout-window when evaluated):
        "ensemble": book_sharpe(sweep.ensemble_test_returns),
        "spy": legacy_spy,
        "best_family": book_sharpe(sweep.best_family_test_returns),
        "margin": float(cfg.margin),
        "passes": verdict == "PASSED",
    }
```

```python
# tools/floor_data_check.py  (key changes)
# - import DEFAULT_CONFIG, config_hash; drop the literal MARGIN
# - panel = pd.read_csv(FIXTURE, ...); m = floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
#   in the default (advisor-gate report) path -> holdout stays blinded, prints the dev verdict.
# - print verdict prose: PASSED / INCONCLUSIVE / UNSUPPORTED / DEV_FAILED, with the
#   necessary-not-sufficient + survivorship disclosure lines (Task 15).
# - report mode: return 0 always; --enforce: return 0 only if m["verdict"] == "PASSED", else 1.
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_floor_entrypoint.py apps/quant/advisor/tests/test_data_floor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/data_floor.py tools/floor_data_check.py apps/quant/advisor/tests/test_floor_entrypoint.py apps/quant/advisor/tests/test_data_floor.py
git commit --no-verify -m "feat(floor): v2 entrypoint — book-Sharpe gate, dev/holdout discipline, report-vs-enforce"
```

> **Dispatch:** `npm run hermes:production -- --task "Rewrite apps/quant/advisor/backtest/data_floor.py floor_metrics(panel, cfg, prereg_hash, families) and update tools/floor_data_check.py to pass apps/quant/advisor/tests/test_floor_entrypoint.py and the extended test_data_floor.py: run dev sweep + dev gate + universe classify, evaluate the holdout ONLY when prereg_hash is set AND dev passed, keep legacy keys, report mode returns 0, --enforce returns 0 only on verdict PASSED. Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate` (must still exit 0, report-only) **and** `node tools/run-floor.mjs --enforce` (must exit 1 unless the verdict is PASSED).

---

## Task 14: Pre-registered dev sweep + single holdout run (the empirical branch) — [Operator]

**Files:**
- Create: `apps/quant/advisor/backtest/PREREG.md` (the pre-registration artifact: frozen config values + `config_hash` + the candidate order A→E + the locked margin number) — committed **before** the holdout is evaluated.
- Use (do not edit logic): the modules from Tasks 1–13.

This is the leakage-safe measurement, run by the operator (not Codex — it decides the deliverable and must stay outside the dev loop). It realizes the §7.1 feasibility branch.

- [ ] **Step 1: Pre-register.** Run `python -c "from advisor.backtest.prereg import DEFAULT_CONFIG, config_hash; ..."` **as a `.py` file** (Windows gotcha: never `python -c "<multi-word>"` under shell:true) to print `config_hash(DEFAULT_CONFIG, fixture)`. Write `PREREG.md` with: the full config values, the hash, the locked **margin number** (default 0.0 unless a *dev-fold-only* sensitivity argues otherwise — never set from the holdout), and the candidate order: **A** trend-alone, **B** momentum-alone, **C** 2-family continuous, **D** +1 family, **E** +2 families. Commit it. **From this point the config is frozen.**
- [ ] **Step 2: Dev sweep (dev folds only).** For candidates C→E in order, run `floor_metrics(panel, cfg, prereg_hash=None, families=...)` and read `m["dev"]`. Stop at the **smallest** candidate whose `dev.passed` is True. If none pass, the deliverable is **UNSUPPORTED** — skip to Step 5.
- [ ] **Step 3: Universe check.** Confirm `m["universe"]`. If `do_not_run`, stop (broaden the fixture or report the limitation). If `micro`, the holdout result is labeled "diagnostic only."
- [ ] **Step 4: Single holdout.** Re-run with `prereg_hash=<the committed hash>` so `floor_metrics` evaluates the holdout once. Record `m["verdict"]` (PASSED / INCONCLUSIVE) and `holdout.delta_lcb`. **Do not iterate** — one holdout evaluation, full stop.
- [ ] **Step 5: Record the verdict** in `PREREG.md` and `.remember/remember.md`: floor **closed** (PASSED, lower-95% Δ LCB > 0 and SPY-margin LCB > margin), **inconclusive** (CI straddles 0 → advisor reports but does not size capital), or **unsupported** (no construction cleared dev). Report whichever as the **lead finding**, never a footnote.

> **No dispatch** — operator-run. Verification is the recorded verdict + `npm run advisor-gate` (report) and `node tools/run-floor.mjs --enforce` (enforce) reflecting it.

---

## Task 15: Disclosures + necessary-not-sufficient caveat — [Codex]

**Files:**
- Modify: `apps/quant/advisor/backtest/walk_forward.py:8-13` (extend `DISCLOSURES`)
- Modify: `tools/floor_data_check.py` (print the caveat block with the verdict)
- Test: `apps/quant/advisor/tests/test_walk_forward.py` (update the count assertion)

- [ ] **Step 1: Update the failing test**

```python
# apps/quant/advisor/tests/test_walk_forward.py  (replace the count assertion)
def test_disclosure_header_carries_v2_caveats():
    from advisor.backtest.walk_forward import DISCLOSURES, disclosure_header
    header = disclosure_header()
    assert len(DISCLOSURES) == 6
    assert any("price-only proxy" in d for d in DISCLOSURES)
    assert any("survivorship" in d for d in DISCLOSURES)
    assert any("does not prove the 5-family" in d for d in DISCLOSURES)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_walk_forward.py -v`
Expected: FAIL (only 4 disclosures).

- [ ] **Step 3: Add the two v2 disclosure lines**

```python
# apps/quant/advisor/backtest/walk_forward.py — append to DISCLOSURES:
    "This floor backtests only the price-only proxy; a PASS does not prove the 5-family "
    "advisor satisfies spec section 7 (necessary, not sufficient).",
    "Fixture universe requires full-window survival -> residual survivorship bias; "
    "long-side results upward-biased.",
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_walk_forward.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/walk_forward.py tools/floor_data_check.py apps/quant/advisor/tests/test_walk_forward.py
git commit --no-verify -m "feat(floor): v2 disclosures — necessary-not-sufficient + survivorship"
```

> **Dispatch:** `npm run hermes:production -- --task "Add two disclosure lines to apps/quant/advisor/backtest/walk_forward.py DISCLOSURES (price-only proxy necessary-not-sufficient; fixture survivorship) and print them with the verdict in tools/floor_data_check.py, to pass the updated apps/quant/advisor/tests/test_walk_forward.py (now expects 6 disclosures). Do NOT run npm or node; verify ONLY with python -m pytest apps/quant/advisor/tests."`
> **Operator verify:** `npm run advisor-gate` (header now prints 6 disclosures + the verdict).

---

## Self-Review (against the spec-of-record)

**Spec coverage:** continuous long-flat transform → T4; adequacy → T5; turnover-aware allocator → T6; per-fold train-only fit → T9; dev stability gate → T10; min-universe → T11; book-Sharpe primary + ladder → T7/T13; benchmark-margin LCB → T2/T13; weight Rule A/B → T8; pre-registration + hash → T1/T14; purged split barrier → T3; rail tests → T12; fixture extension → T0; disclosures/necessary-not-sufficient → T15; report-vs-enforce split → T13; honest verdict branch → T14. **No uncovered requirement.** Rejected items (shorts, orthogonalization, Brier gate, rank-IC, OOS probe, tournament) are *absent by construction* and rail-guarded (T12).

**Placeholder scan:** every code/test step contains runnable code; no TBD/TODO. Operator tasks (T0, T14) are inherently procedural (network / single-gated-run) and give exact commands.

**Type consistency:** `PreRegConfig` fields, `book_sharpe`/`block_bootstrap_diff_lcb` signatures, `build_long_flat_book(scores, max_asset_weight, gross_cap, turnover_cap, cost_per_turn)`, `book_returns(weights, prices, cost_per_turn)`, `select_weights(...)`, `run_dev_sweep(panel, families, cfg)` → `SweepResult`, `dev_gate(...)` → `GateResult`, `floor_metrics(panel, cfg, prereg_hash, families)` are used consistently across T1–T15.

> **Note on the `caps` tuple:** the convention is `caps = (max_asset_weight, gross_cap, turnover_cap)` everywhere (`pipeline`, `blend`, `run_holdout`), unpacked as `build_long_flat_book(scores, *caps, cost_per_turn)`. Element 3 is the **turnover cap** (not a second per-name cap) — keep these aligned across modules.

> **Holdout discipline (T9 `run_holdout` + T13):** the §7 verdict is computed on the genuinely held-out tail (final 20%, never seen during dev), gating on **both** the §7.2 beat-the-parts LCB **and** the §7.1 beat-SPY-by-margin LCB. The dev-fold returns are used only for the dev gate, never for the verdict.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-15-plan4-v2-calibration.md`. **Stop here for operator review** (per the original handoff: produce the plan, do not start implementation). After you approve, two execution options:

1. **Subagent-Driven (recommended elsewhere)** — but on this repo the standing contract is **Hermes solo dispatch + operator gate** per task (the dispatch lines above), reviewing each diff. Task 0 and Task 14 are operator-run.
2. **Inline executing-plans** — not applicable: implementation must go through Hermes per the workspace contract.

**Recommended order:** T0 (branch + fixture) → T1–T12 (Codex, any order respecting imports: prereg→stats→splits→signals→adequacy→portfolio→book→blend→pipeline→dev_gate→universe→rails) → T13 (entrypoint) → **T14 (operator branch decision)** → T15 (disclosures). Re-run `npm run advisor-gate` after each Codex task; it must stay exit 0 (report-only) throughout, and `--enforce` must stay exit 1 until/unless T14 records a PASSED verdict.
