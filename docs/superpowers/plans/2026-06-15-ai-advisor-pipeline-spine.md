# AI Advisor — Pipeline Spine + Floor (Plan 2 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Dispatch (this repo):** Each task is dispatched via Hermes `--workflow auto --live` (financial tasks run the Codex→Claude pair loop). ALWAYS include in the task string: "Do NOT run npm or node; verify ONLY with `python -m pytest apps/quant/advisor/tests`." The `workspace-write` sandbox blocks npm/node spawns. See `docs/superpowers/plans/2026-06-14-hermes-port-bootstrap.md`.

**Goal:** Assemble the deterministic decision spine — a second decorrelated signal family, a deterministic risk manager, a portfolio allocator, an async fan-out pipeline, and the real `advisor-gate` floor — so the advisor produces a typed, risk-bounded decision and refuses to ship unless it beats SPY *and* its own best single family out-of-sample.

**Architecture:** Builds on Plan 1's `SignalBundle` seam. Families run concurrently; an equal-weight ensemble reduces them to one direction+confidence per ticker; the deterministic risk manager sets a vol-/correlation-adjusted dollar limit; the allocator picks among legal actions only. The floor gate runs a purged walk-forward backtest of the ensemble vs SPY.

**Tech Stack:** Python 3.13, Pydantic v2, pandas, numpy, pytest. Same isolated `advisor` package (`apps/quant/advisor`, import root `apps/quant`).

**Scope note:** Plan 2 of 3. Plan 3 adds the trend/macro/sentiment families, the persona critique/explain overlay, and TimescaleDB persistence. v1 ensemble is **equal-weight** (calibration/skill-weighting is deferred per spec §8).

---

### Task 1: Momentum signal family (2nd, decorrelated, price-only)

A second family that depends only on prices (no fundamentals) so the ensemble has two decorrelated inputs.

**Files:**
- Create: `apps/quant/advisor/analysis/momentum.py`
- Test: `apps/quant/advisor/tests/test_momentum.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_momentum.py`:
```python
from datetime import date

import numpy as np
import pandas as pd

from advisor.analysis.momentum import FAMILY, evaluate
from advisor.schemas import Direction


def test_uptrend_is_bullish():
    prices = pd.Series(np.linspace(100, 200, 200))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.family == FAMILY
    assert sig.direction is Direction.BULLISH
    assert sig.confidence > 50


def test_downtrend_is_bearish():
    prices = pd.Series(np.linspace(200, 100, 200))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.direction is Direction.BEARISH


def test_insufficient_history_is_neutral():
    prices = pd.Series(np.linspace(100, 110, 20))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_momentum.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/analysis/momentum.py`:
```python
from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.schemas import Direction, FamilySignal

FAMILY = "momentum"
HORIZONS = (63, 126)  # ~3m and ~6m trading days
MIN_HISTORY = max(HORIZONS) + 1


def evaluate(prices: pd.Series, as_of: date) -> FamilySignal:
    prices = prices.dropna()
    if len(prices) < MIN_HISTORY:
        return FamilySignal.neutral(FAMILY, as_of, "insufficient price history")

    returns = []
    for h in HORIZONS:
        past = prices.iloc[-(h + 1)]
        if past > 0:
            returns.append(prices.iloc[-1] / past - 1.0)
    if not returns:
        return FamilySignal.neutral(FAMILY, as_of, "no valid horizon returns")

    score = sum(returns) / len(returns)  # average multi-horizon momentum
    if score > 0.02:
        direction = Direction.BULLISH
    elif score < -0.02:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + min(abs(score), 1.0) * 100.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"avg multi-horizon momentum={score:.1%}")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_momentum.py`
Expected: 3 passed.

- [ ] **Step 5: Commit** `git add` the two files and commit `feat(advisor): momentum signal family`.

---

### Task 2: Deterministic risk manager (vol- + correlation-adjusted limits)

**Files:**
- Create: `apps/quant/advisor/risk/__init__.py` (one-line docstring, same as other subpackages)
- Create: `apps/quant/advisor/risk/limits.py`
- Test: `apps/quant/advisor/tests/test_risk.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_risk.py`:
```python
from advisor.risk.limits import (
    correlation_multiplier,
    position_limit,
    vol_adjusted_fraction,
)


def test_low_vol_gets_full_fraction():
    assert vol_adjusted_fraction(0.10) == 0.25


def test_high_vol_is_capped_low():
    assert vol_adjusted_fraction(0.60) == 0.10


def test_correlation_penalizes_crowded_and_rewards_diversifiers():
    assert correlation_multiplier(0.90) == 0.70
    assert correlation_multiplier(0.0) == 1.10
    assert correlation_multiplier(0.50) == 1.0


def test_position_limit_dollars():
    # net_liq 100k, fraction 0.25, multiplier 1.0 -> 25k
    assert position_limit(100_000, vol=0.10, correlation=0.5) == 25_000.0


def test_position_limit_never_negative():
    assert position_limit(0, vol=0.10, correlation=0.5) == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_risk.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/risk/limits.py`:
```python
from __future__ import annotations


def vol_adjusted_fraction(annualized_vol: float) -> float:
    """Fraction of net liquidation value allowed in one name, scaled by volatility."""
    if annualized_vol < 0.15:
        return 0.25
    if annualized_vol < 0.30:
        return 0.20
    if annualized_vol < 0.50:
        return 0.15
    return 0.10


def correlation_multiplier(correlation: float) -> float:
    """Penalize crowded (highly correlated) positions; reward diversifiers."""
    if correlation >= 0.80:
        return 0.70
    if correlation >= 0.50:
        return 1.0
    return 1.10


def position_limit(net_liq: float, vol: float, correlation: float) -> float:
    """Dollar cap for a single position."""
    if net_liq <= 0:
        return 0.0
    return max(0.0, net_liq * vol_adjusted_fraction(vol) * correlation_multiplier(correlation))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_risk.py`
Expected: 5 passed.

- [ ] **Step 5: Commit** the `risk/` files: `feat(advisor): deterministic vol/correlation position limits`.

---

### Task 3: Ensemble + portfolio allocator (legal actions only)

Reduces a `SignalBundle` to one direction+confidence (equal-weight vote), then picks a target action bounded by the risk limit — never exceeding it.

**Files:**
- Create: `apps/quant/advisor/portfolio/__init__.py` (one-line docstring)
- Create: `apps/quant/advisor/portfolio/allocator.py`
- Test: `apps/quant/advisor/tests/test_allocator.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_allocator.py`:
```python
from datetime import date

from advisor.portfolio.allocator import ensemble_vote, allocate
from advisor.schemas import Direction, FamilySignal, SignalBundle


def _bundle(dirs):
    sigs = [FamilySignal(family=f"f{i}", direction=d, confidence=80.0, as_of=date(2024, 5, 1))
            for i, d in enumerate(dirs)]
    return SignalBundle(ticker="AAPL", as_of=date(2024, 5, 1), signals=sigs)


def test_ensemble_vote_majority_bullish():
    d, conf = ensemble_vote(_bundle([Direction.BULLISH, Direction.BULLISH, Direction.BEARISH]))
    assert d is Direction.BULLISH
    assert 0 < conf <= 100


def test_ensemble_tie_is_neutral():
    d, _ = ensemble_vote(_bundle([Direction.BULLISH, Direction.BEARISH]))
    assert d is Direction.NEUTRAL


def test_allocate_buy_bounded_by_limit():
    # bullish, limit 25k, price 100 -> max 250 shares; target weight scaled by confidence
    a = allocate(_bundle([Direction.BULLISH, Direction.BULLISH]), price=100.0,
                 position_limit_dollars=25_000.0)
    assert a.action == "buy"
    assert 0 < a.quantity <= 250


def test_allocate_neutral_holds():
    a = allocate(_bundle([Direction.BULLISH, Direction.BEARISH]), price=100.0,
                 position_limit_dollars=25_000.0)
    assert a.action == "hold"
    assert a.quantity == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_allocator.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/portfolio/allocator.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from advisor.schemas import Direction, SignalBundle


@dataclass(frozen=True)
class Allocation:
    ticker: str
    action: str  # "buy" | "sell" | "hold"
    quantity: int
    reasoning: str


def ensemble_vote(bundle: SignalBundle) -> tuple[Direction, float]:
    """Equal-weight vote across families. Net sign decides direction."""
    score = 0.0
    weight = 0.0
    for s in bundle.signals:
        if s.direction is Direction.BULLISH:
            score += s.confidence
        elif s.direction is Direction.BEARISH:
            score -= s.confidence
        weight += s.confidence
    if weight == 0 or score == 0:
        return Direction.NEUTRAL, 50.0
    direction = Direction.BULLISH if score > 0 else Direction.BEARISH
    confidence = min(100.0, 50.0 + abs(score) / weight * 50.0)
    return direction, confidence


def allocate(bundle: SignalBundle, price: float, position_limit_dollars: float) -> Allocation:
    direction, confidence = ensemble_vote(bundle)
    if direction is Direction.NEUTRAL or price <= 0 or position_limit_dollars <= 0:
        return Allocation(bundle.ticker, "hold", 0, "neutral or no capacity")

    max_shares = int(position_limit_dollars // price)
    # scale by conviction above the 50 baseline (0..1)
    conviction = max(0.0, (confidence - 50.0) / 50.0)
    quantity = int(max_shares * conviction)
    if quantity <= 0:
        return Allocation(bundle.ticker, "hold", 0, "conviction below threshold")

    action = "buy" if direction is Direction.BULLISH else "sell"
    return Allocation(bundle.ticker, action, quantity,
                      f"{direction.value} conviction={conviction:.0%}, max_shares={max_shares}")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_allocator.py`
Expected: 4 passed.

- [ ] **Step 5: Commit** the `portfolio/` files: `feat(advisor): equal-weight ensemble + legal-action allocator`.

---

### Task 4: Async fan-out pipeline

Runs the families concurrently, builds the `SignalBundle`, then risk → allocator → a typed decision. (TimescaleDB checkpoint deferred to Plan 3; here the pipeline returns the decision.)

**Files:**
- Create: `apps/quant/advisor/pipeline/__init__.py` (one-line docstring)
- Create: `apps/quant/advisor/pipeline/run.py`
- Test: `apps/quant/advisor/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_pipeline.py`:
```python
import asyncio
from datetime import date

import numpy as np
import pandas as pd

from advisor.pipeline.run import run_pipeline
from advisor.schemas import Direction


def test_pipeline_produces_bounded_decision():
    prices = pd.Series(np.linspace(100, 200, 200))  # uptrend -> momentum bullish

    async def momentum_family(as_of):
        from advisor.analysis.momentum import evaluate
        return evaluate(prices, as_of)

    decision = asyncio.run(run_pipeline(
        ticker="AAPL", as_of=date(2024, 5, 1), price=100.0,
        net_liq=100_000.0, vol=0.10, correlation=0.5,
        family_coros=[momentum_family],
    ))
    assert decision.ticker == "AAPL"
    assert decision.action in {"buy", "sell", "hold"}
    # bounded by risk: vol 0.10 -> 25% of 100k = 25k -> <=250 shares at price 100
    assert decision.quantity <= 250
    assert decision.bundle_direction in {d.value for d in Direction}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_pipeline.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/pipeline/run.py`:
```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Awaitable, Callable

from advisor.portfolio.allocator import allocate, ensemble_vote
from advisor.risk.limits import position_limit
from advisor.schemas import SignalBundle

FamilyCoro = Callable[[date], Awaitable]


@dataclass(frozen=True)
class Decision:
    ticker: str
    action: str
    quantity: int
    bundle_direction: str
    reasoning: str


async def run_pipeline(ticker: str, as_of: date, price: float, net_liq: float,
                       vol: float, correlation: float,
                       family_coros: list[FamilyCoro]) -> Decision:
    signals = await asyncio.gather(*(coro(as_of) for coro in family_coros))
    bundle = SignalBundle(ticker=ticker, as_of=as_of, signals=list(signals))
    direction, _ = ensemble_vote(bundle)
    limit = position_limit(net_liq, vol=vol, correlation=correlation)
    alloc = allocate(bundle, price=price, position_limit_dollars=limit)
    return Decision(ticker=ticker, action=alloc.action, quantity=alloc.quantity,
                    bundle_direction=direction.value, reasoning=alloc.reasoning)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_pipeline.py`
Expected: 1 passed.

- [ ] **Step 5: Commit** the `pipeline/` files: `feat(advisor): async fan-out decision pipeline`.

---

### Task 5: Real `advisor-gate` floor — purged walk-forward, beat SPY + best family

Extends the backtest with a purged walk-forward split and a floor check that requires the ensemble to beat both SPY and its own best single family. Wires it into `advisor-gate`.

**Files:**
- Create: `apps/quant/advisor/backtest/floor.py`
- Test: `apps/quant/advisor/tests/test_floor.py`
- Modify: `package.json` — `advisor-gate` runs the floor check after the suite
- Create: `tools/run-floor.mjs`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_floor.py`:
```python
import numpy as np
import pandas as pd

from advisor.backtest.floor import purged_walk_forward_sharpe, beats_floor


def _series(slope):
    return pd.Series(np.linspace(100, 100 + slope, 300))


def test_purged_walk_forward_returns_finite_sharpe():
    prices = _series(50)
    signal = pd.Series(1.0, index=prices.index)
    s = purged_walk_forward_sharpe(prices, signal, folds=3, embargo=5)
    assert np.isfinite(s)


def test_beats_floor_requires_both_conditions():
    # ensemble 1.2 beats spy 0.8 (margin 0.3) AND best family 1.0
    assert beats_floor(ensemble=1.2, spy=0.8, best_family=1.0, margin=0.3) is True
    # fails benchmark margin
    assert beats_floor(ensemble=1.0, spy=0.8, best_family=0.5, margin=0.3) is False
    # beats benchmark but not its own best family
    assert beats_floor(ensemble=1.2, spy=0.8, best_family=1.3, margin=0.3) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_floor.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/backtest/floor.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.backtest.walk_forward import walk_forward


def purged_walk_forward_sharpe(prices: pd.Series, signal: pd.Series,
                               folds: int = 5, embargo: int = 5) -> float:
    """Average out-of-sample Sharpe across purged walk-forward folds.

    Each fold tests a forward slice; an embargo gap before the test slice is
    dropped to reduce leakage from overlapping windows.
    """
    n = len(prices)
    if n < folds * 10:
        return 0.0
    fold_size = n // folds
    sharpes = []
    for k in range(1, folds):
        test_start = k * fold_size + embargo
        test_end = (k + 1) * fold_size if k < folds - 1 else n
        if test_start >= test_end:
            continue
        p = prices.iloc[test_start:test_end].reset_index(drop=True)
        s = signal.iloc[test_start:test_end].reset_index(drop=True)
        sharpes.append(walk_forward(p, s).sharpe)
    return float(np.mean(sharpes)) if sharpes else 0.0


def beats_floor(ensemble: float, spy: float, best_family: float, margin: float) -> bool:
    """Spec §7: beat SPY by a pre-registered margin AND beat the best single family."""
    return (ensemble - spy) >= margin and ensemble >= best_family
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_floor.py`
Expected: 3 passed.

- [ ] **Step 5: Add the floor runner**

`tools/run-floor.mjs`:
```javascript
#!/usr/bin/env node
// advisor-gate stage 2: import-smoke the floor module so the gate fails if the
// floor logic is broken. (Full data-driven floor enforcement lands when live
// price history is wired in Plan 3.)
import { spawnSync } from 'node:child_process';
const r = spawnSync('python', ['-c', 'import sys; sys.path.insert(0, "apps/quant"); from advisor.backtest.floor import beats_floor, purged_walk_forward_sharpe; print("floor: importable")'], {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});
process.exit(r.status ?? 1);
```

- [ ] **Step 6: Wire `advisor-gate` to run the floor after the suite**

In `package.json`, change the `advisor-gate` script to:
```json
    "advisor-gate": "node tools/run-pytest.mjs apps/quant/advisor/tests && node tools/run-floor.mjs"
```

- [ ] **Step 7: Verify the gate**

Run: `node tools/run-pytest.mjs apps/quant/advisor/tests && node tools/run-floor.mjs`
Expected: tests pass, then `floor: importable`.

- [ ] **Step 8: Commit** `feat(advisor): purged walk-forward floor + advisor-gate floor stage`.

---

## Self-Review

**1. Spec coverage (Plan 2 scope):** momentum family → Task 1 ✅; deterministic vol/correlation risk limits (spec §3) → Task 2 ✅; equal-weight ensemble + legal-action allocator (spec §3, §8 equal-weight) → Task 3 ✅; async fan-out (spec §3, no LangGraph) → Task 4 ✅; purged walk-forward + beat-SPY-and-best-family floor (spec §7) wired into `advisor-gate` (spec §9) → Task 5 ✅. Deferred to Plan 3 (stated): trend/macro/sentiment families, persona overlay, TimescaleDB checkpoint, full data-driven floor enforcement.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step is complete. The floor runner is an explicit import-smoke with a stated Plan-3 extension — not a hidden gap.

**3. Type consistency:** `FamilySignal`/`SignalBundle`/`Direction` reused exactly from Plan 1. `ensemble_vote(bundle) -> (Direction, float)` and `allocate(bundle, price, position_limit_dollars) -> Allocation` match across Tasks 3–4. `position_limit(net_liq, vol, correlation)` signature matches Tasks 2 and 4. `walk_forward(prices, signal)` reused from Plan 1 by Task 5.
