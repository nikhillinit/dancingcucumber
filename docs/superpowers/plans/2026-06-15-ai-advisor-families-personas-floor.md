# AI Advisor — Remaining Families + Persona Overlay + Data-Driven Floor (Plan 3 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Dispatch (this repo):** Tasks are dispatched via Hermes **solo** — `npm run hermes:production -- --task "..."` — and the diff is reviewed by the operator. Solo (Codex owner + pytest gate) is the proven-reliable mode on this Windows host and needs no Claude credits. ALWAYS include in the task string: **"Do NOT run npm or node; verify ONLY with `python -m pytest apps/quant/advisor/tests`."** Codex's `workspace-write` sandbox blocks npm/node spawns and all network/DB access. See `docs/superpowers/plans/2026-06-14-hermes-port-bootstrap.md` and the `hermes-dispatch-windows` memory.

> **Dispatch ownership (read before executing):** Plan 3 splits in two. **Codex-dispatchable** (Tasks 1–6): deterministic logic verifiable by pytest against fakes/mocks. **Agent/operator-verified** (Tasks 7–8): need real network/DB the sandbox forbids — use the **fixture-first** pattern (operator commits a real-data fixture, Codex writes logic against the *committed* fixture, pytest verifies offline). Never hand Codex a task whose verification needs the network — it exits non-zero on a clean commit.

**Goal:** Complete the v1 advisor — add the three remaining deterministic signal families (trend, macro, sentiment), the read-only persona critique/explain overlay (which may veto/downgrade but **never** upsize), a real data-driven floor that backtests the ensemble vs SPY on committed price history, and TimescaleDB persistence of the `SignalBundle` checkpoint.

**Architecture:** Builds on Plan 2's `SignalBundle` seam, `run_pipeline` fan-out, deterministic risk/allocator, and `purged_walk_forward_sharpe`/`beats_floor` floor math. New families are pure-Python `evaluate()` functions returning `FamilySignal` (identical shape to `momentum`/`value_quality`). Sentiment and personas take an **injectable scorer/critic** so tests pass canned values — no live LLM (spec §12). The persona overlay's only powers are a clamped size-multiplier in `[0, 1]` and an explanation string — the trust boundary (spec §5: never upsize, never touch risk limits) is enforced in code by the clamp.

**Tech Stack:** Python 3.13, Pydantic v2, pandas, numpy, pytest. Same isolated `advisor` package (`apps/quant/advisor`, import root `apps/quant`). TimescaleDB via `psycopg`/`pg` against the existing `infra/docker-compose.yml`.

**Scope note:** Plan 3 of 3 — completes spec §13 migration steps 7–8 and the five-family runtime of §3. v1 ensemble stays **equal-weight** (calibration/skill-weighting deferred to v2 per spec §8). Information-diverse persona *deciders* remain v2 (spec §5).

---

### Task 1: Trend signal family (3rd family, price-only, MA-crossover) — **Codex**

A third price-only family (decorrelated from the point momentum family) using a moving-average crossover. Exact structural analog of `momentum.py` — the cleanest possible first dispatch to prove the loop.

**Files:**
- Create: `apps/quant/advisor/analysis/trend.py`
- Test: `apps/quant/advisor/tests/test_trend.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_trend.py`:
```python
from datetime import date

import numpy as np
import pandas as pd

from advisor.analysis.trend import FAMILY, evaluate
from advisor.schemas import Direction


def test_uptrend_is_bullish():
    prices = pd.Series(np.linspace(100, 200, 250))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.family == FAMILY
    assert sig.direction is Direction.BULLISH
    assert sig.confidence > 50


def test_downtrend_is_bearish():
    prices = pd.Series(np.linspace(200, 100, 250))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.direction is Direction.BEARISH


def test_insufficient_history_is_neutral():
    prices = pd.Series(np.linspace(100, 110, 100))
    sig = evaluate(prices, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_trend.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/analysis/trend.py`:
```python
from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.schemas import Direction, FamilySignal

FAMILY = "trend"
SHORT, LONG = 50, 200
MIN_HISTORY = LONG + 1


def evaluate(prices: pd.Series, as_of: date) -> FamilySignal:
    prices = prices.dropna()
    if len(prices) < MIN_HISTORY:
        return FamilySignal.neutral(FAMILY, as_of, "insufficient price history")

    short_ma = float(prices.rolling(SHORT).mean().iloc[-1])
    long_ma = float(prices.rolling(LONG).mean().iloc[-1])
    if long_ma <= 0:
        return FamilySignal.neutral(FAMILY, as_of, "non-positive long moving average")

    gap = short_ma / long_ma - 1.0
    if gap > 0.01:
        direction = Direction.BULLISH
    elif gap < -0.01:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + min(abs(gap), 0.5) * 100.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"SMA{SHORT}/SMA{LONG} gap={gap:.1%}")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_trend.py`
Expected: 3 passed.

- [ ] **Step 5: Commit** `git add` the two files and commit `feat(advisor): trend (MA-crossover) signal family`.

---

### Task 2: Macro/rates regime family (FRED yield-curve, faked series) — **Codex**

A regime family driven by the 10y–2y Treasury spread (FRED `T10Y2Y`). Inverted curve → risk-off/bearish; steep/positive → risk-on/bullish. Pure pandas on a series the test fakes — no FRED call.

**Files:**
- Create: `apps/quant/advisor/analysis/macro.py`
- Test: `apps/quant/advisor/tests/test_macro.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_macro.py`:
```python
from datetime import date

import pandas as pd

from advisor.analysis.macro import FAMILY, evaluate
from advisor.schemas import Direction


def test_inverted_curve_is_bearish():
    spread = pd.Series([0.5, 0.1, -0.3])  # 10y-2y inverted at the latest point
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.family == FAMILY
    assert sig.direction is Direction.BEARISH


def test_steep_curve_is_bullish():
    spread = pd.Series([0.2, 0.6, 1.0])
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.direction is Direction.BULLISH


def test_flat_curve_is_neutral():
    spread = pd.Series([0.1, 0.2, 0.3])
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL


def test_insufficient_history_is_neutral():
    spread = pd.Series([0.3])
    sig = evaluate(spread, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_macro.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/analysis/macro.py`:
```python
from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.schemas import Direction, FamilySignal

FAMILY = "macro"
MIN_HISTORY = 2
STEEP_THRESHOLD = 0.5  # 10y-2y spread (pct points) above which the curve is risk-on


def evaluate(yield_curve_spread: pd.Series, as_of: date) -> FamilySignal:
    """Regime from the 10y-2y Treasury spread (FRED T10Y2Y). Inverted -> bearish."""
    s = yield_curve_spread.dropna()
    if len(s) < MIN_HISTORY:
        return FamilySignal.neutral(FAMILY, as_of, "insufficient macro history")

    latest = float(s.iloc[-1])
    if latest < 0:
        direction = Direction.BEARISH
    elif latest > STEEP_THRESHOLD:
        direction = Direction.BULLISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + min(abs(latest), 2.0) / 2.0 * 50.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"10y-2y spread={latest:.2f}")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_macro.py`
Expected: 4 passed.

- [ ] **Step 5: Commit** the two files: `feat(advisor): macro/rates regime signal family`.

---

### Task 3: News/sentiment surprise family (injectable scorer, no live LLM) — **Codex**

Sentiment from a list of headlines scored by an **injectable** scorer callable (`headline -> [-1, 1]`). Tests pass canned scores (spec §12). Per spec §10: **no news → `neutral` ("no news available"), never fabricated.**

**Files:**
- Create: `apps/quant/advisor/analysis/sentiment.py`
- Test: `apps/quant/advisor/tests/test_sentiment.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_sentiment.py`:
```python
from datetime import date

from advisor.analysis.sentiment import FAMILY, evaluate
from advisor.schemas import Direction


def test_positive_news_is_bullish():
    sig = evaluate(["earnings beat estimates"], date(2024, 5, 1), scorer=lambda h: 1.0)
    assert sig.family == FAMILY
    assert sig.direction is Direction.BULLISH


def test_negative_news_is_bearish():
    sig = evaluate(["guidance cut, probe opened"], date(2024, 5, 1), scorer=lambda h: -1.0)
    assert sig.direction is Direction.BEARISH


def test_no_news_is_neutral_not_fabricated():
    sig = evaluate([], date(2024, 5, 1), scorer=lambda h: 1.0)
    assert sig.direction is Direction.NEUTRAL
    assert "no news" in sig.reasoning.lower()


def test_mixed_news_near_zero_is_neutral():
    sig = evaluate(["a", "b"], date(2024, 5, 1), scorer=lambda h: 1.0 if h == "a" else -1.0)
    assert sig.direction is Direction.NEUTRAL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_sentiment.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/analysis/sentiment.py`:
```python
from __future__ import annotations

from datetime import date
from typing import Callable

from advisor.schemas import Direction, FamilySignal

FAMILY = "sentiment"
THRESHOLD = 0.2

NewsScorer = Callable[[str], float]  # headline -> score in [-1, 1]


def evaluate(headlines: list[str], as_of: date, scorer: NewsScorer) -> FamilySignal:
    """Average scored news surprise. Missing source -> neutral, never fabricated (spec section 10)."""
    if not headlines:
        return FamilySignal.neutral(FAMILY, as_of, "no news available")

    scores = [max(-1.0, min(1.0, float(scorer(h)))) for h in headlines]
    avg = sum(scores) / len(scores)
    if avg > THRESHOLD:
        direction = Direction.BULLISH
    elif avg < -THRESHOLD:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + abs(avg) * 50.0)
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=f"avg news score={avg:+.2f} over {len(scores)} headlines")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_sentiment.py`
Expected: 4 passed.

- [ ] **Step 5: Commit** the two files: `feat(advisor): news/sentiment surprise family (injectable scorer)`.

---

### Task 4: Persona critique/explain overlay (veto/downgrade only, never upsize) — **Codex**

The read-only persona layer (spec §5). Runs *after* the typed `Decision`. A persona returns a `size_multiplier` and an `explanation`; the overlay **clamps the multiplier to `[0, 1]`** — so a persona can veto (`0` → hold) or downgrade (`<1`) but can **never upsize or alter risk limits**. The clamp is the trust boundary in code. The overlay is decoupled from `Decision`'s module via `dataclasses.replace` (duck-typed) so there is no circular import.

**Files:**
- Create: `apps/quant/advisor/personas/__init__.py` (one-line docstring, same as other subpackages)
- Create: `apps/quant/advisor/personas/overlay.py`
- Test: `apps/quant/advisor/tests/test_overlay.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_overlay.py`:
```python
from advisor.personas.overlay import PersonaVerdict, apply_overlay
from advisor.pipeline.run import Decision


def _decision(action="buy", quantity=100):
    return Decision(ticker="AAPL", action=action, quantity=quantity,
                    bundle_direction="bullish", reasoning="ensemble bullish")


def test_persona_veto_forces_hold():
    out = apply_overlay(_decision(), lambda d: PersonaVerdict(0.0, "distress red flag"))
    assert out.action == "hold"
    assert out.quantity == 0
    assert "distress red flag" in out.reasoning


def test_persona_downgrade_reduces_quantity():
    out = apply_overlay(_decision(quantity=100), lambda d: PersonaVerdict(0.5, "rich valuation"))
    assert out.quantity == 50
    assert out.action == "buy"


def test_persona_cannot_upsize_multiplier_is_clamped():
    # a persona that "wants" 5x is clamped to 1.0 -- the trust boundary
    out = apply_overlay(_decision(quantity=100), lambda d: PersonaVerdict(5.0, "love it"))
    assert out.quantity == 100
    assert out.action == "buy"


def test_persona_explanation_is_appended():
    out = apply_overlay(_decision(), lambda d: PersonaVerdict(1.0, "value+quality concur"))
    assert "value+quality concur" in out.reasoning
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_overlay.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/personas/__init__.py`:
```python
"""Persona overlay (v1: read-only critique + explanation; may veto/downgrade, never upsize)."""
```

`apps/quant/advisor/personas/overlay.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable


@dataclass(frozen=True)
class PersonaVerdict:
    size_multiplier: float  # intent in [0, 1]; clamped on apply -- can only downgrade, never upsize
    explanation: str


# A critic inspects the typed decision and returns a verdict. In v1 this is backed
# by a mocked/canned scorer in tests and (later) an LLM explainer; it never sets size directly.
PersonaCritic = Callable[[Any], PersonaVerdict]


def apply_overlay(decision: Any, critic: PersonaCritic) -> Any:
    """Apply a persona verdict to a typed Decision.

    Trust boundary (spec section 5): the multiplier is clamped to [0, 1], so a persona
    may veto (-> hold) or downgrade size, but can NEVER upsize or touch risk limits.
    Decoupled from Decision's module via dataclasses.replace to avoid a circular import.
    """
    verdict = critic(decision)
    multiplier = max(0.0, min(1.0, float(verdict.size_multiplier)))
    new_quantity = int(decision.quantity * multiplier)
    action = decision.action if new_quantity > 0 else "hold"
    reasoning = f"{decision.reasoning} | persona({multiplier:.0%}): {verdict.explanation}"
    return replace(decision, quantity=new_quantity, action=action, reasoning=reasoning)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_overlay.py`
Expected: 4 passed.

- [ ] **Step 5: Commit** the `personas/` files: `feat(advisor): read-only persona overlay (clamped veto/downgrade)`.

---

### Task 5: Wire the persona overlay into the pipeline (optional, post-decision) — **Codex**

Extend `run_pipeline` with an optional `persona_critic`. When provided, the typed `Decision` passes through the overlay before return. Lazy import of `apply_overlay` keeps `pipeline.run` free of a load-time dependency on `personas`.

**Files:**
- Modify: `apps/quant/advisor/pipeline/run.py`
- Test: `apps/quant/advisor/tests/test_pipeline.py` (add a case; keep the existing one)

- [ ] **Step 1: Write the failing test** — append to `apps/quant/advisor/tests/test_pipeline.py`:
```python
def test_pipeline_applies_persona_veto():
    import asyncio
    from datetime import date

    import numpy as np
    import pandas as pd

    from advisor.personas.overlay import PersonaVerdict
    from advisor.pipeline.run import run_pipeline

    prices = pd.Series(np.linspace(100, 200, 200))  # uptrend -> momentum bullish -> buy

    async def momentum_family(as_of):
        from advisor.analysis.momentum import evaluate
        return evaluate(prices, as_of)

    decision = asyncio.run(run_pipeline(
        ticker="AAPL", as_of=date(2024, 5, 1), price=100.0,
        net_liq=100_000.0, vol=0.10, correlation=0.5,
        family_coros=[momentum_family],
        persona_critic=lambda d: PersonaVerdict(0.0, "forensic red flag"),
    ))
    assert decision.action == "hold"
    assert decision.quantity == 0
    assert "forensic red flag" in decision.reasoning
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_pipeline.py`
Expected: FAIL — `run_pipeline() got an unexpected keyword argument 'persona_critic'`.

- [ ] **Step 3: Write the implementation** — modify `apps/quant/advisor/pipeline/run.py`. Change the `run_pipeline` signature and add the overlay step. The full updated function:
```python
async def run_pipeline(ticker: str, as_of: date, price: float, net_liq: float,
                       vol: float, correlation: float,
                       family_coros: list[FamilyCoro],
                       persona_critic=None) -> Decision:
    signals = await asyncio.gather(*(coro(as_of) for coro in family_coros))
    bundle = SignalBundle(ticker=ticker, as_of=as_of, signals=list(signals))
    direction, _ = ensemble_vote(bundle)
    limit = position_limit(net_liq, vol=vol, correlation=correlation)
    alloc = allocate(bundle, price=price, position_limit_dollars=limit)
    decision = Decision(ticker=ticker, action=alloc.action, quantity=alloc.quantity,
                        bundle_direction=direction.value, reasoning=alloc.reasoning)
    if persona_critic is not None:
        from advisor.personas.overlay import apply_overlay  # lazy: avoids circular import
        decision = apply_overlay(decision, persona_critic)
    return decision
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_pipeline.py`
Expected: 2 passed.

- [ ] **Step 5: Commit** `feat(advisor): optional persona overlay in decision pipeline`.

---

### Task 6: CLI `--explain` flag (persona explanation over the decision) — **Codex test; operator live-verify**

Add `--explain` to the CLI. The unit test drives `cli.run()` with a `FakeProvider` and an injected critic (Codex-verifiable). Live multi-source assembly (FRED/news providers) stays behind the data interface and is exercised manually by the operator — the unit path injects fakes.

**Files:**
- Modify: `apps/quant/advisor/cli.py`
- Test: `apps/quant/advisor/tests/test_cli.py` (add a case; keep existing)

- [ ] **Step 1: Write the failing test** — append to `apps/quant/advisor/tests/test_cli.py`:
```python
def test_run_with_explain_appends_persona_line():
    from datetime import date

    from advisor.cli import run
    from advisor.data.fakes import FakeProvider
    from advisor.data.provider import Fundamentals
    from advisor.personas.overlay import PersonaVerdict

    f = Fundamentals(period_end=date(2023, 12, 31), net_income=10.0, total_equity=100.0,
                     revenue=200.0, operating_income=30.0, total_debt=20.0, depreciation=5.0,
                     capex=4.0, shares_outstanding=10.0, market_cap=500.0)
    provider = FakeProvider(fundamentals={"AAPL": f})

    out = run(provider, "AAPL", date(2024, 6, 1),
              critic=lambda sig: PersonaVerdict(1.0, "value+quality concur"))
    assert "value+quality concur" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_cli.py`
Expected: FAIL — `run() got an unexpected keyword argument 'critic'`.

- [ ] **Step 3: Write the implementation** — replace `apps/quant/advisor/cli.py` with:
```python
from __future__ import annotations

import argparse
from datetime import date

from advisor.analysis.value_quality import evaluate
from advisor.backtest.walk_forward import disclosure_header
from advisor.data.provider import MarketDataProvider, YFinanceProvider


def run(provider: MarketDataProvider, ticker: str, as_of: date, critic=None) -> str:
    f = provider.get_fundamentals_asof(ticker, as_of)
    if f is None:
        return f"{ticker}: no point-in-time fundamentals available as of {as_of}\n{disclosure_header()}"
    sig = evaluate(f, as_of)
    line = f"{ticker} [{sig.direction.value}] confidence={sig.confidence:.0f} :: {sig.reasoning}"
    if critic is not None:
        verdict = critic(sig)
        line += f"\n  persona: {verdict.explanation}"
    return f"{line}\n{disclosure_header()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advisor", description="Single-family AI advisor (value/quality)")
    parser.add_argument("ticker")
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="YYYY-MM-DD point-in-time date")
    parser.add_argument("--explain", action="store_true",
                        help="append a persona explanation line (v1: read-only narration)")
    args = parser.parse_args(argv)
    critic = None
    if args.explain:
        from advisor.personas.overlay import PersonaVerdict
        critic = lambda sig: PersonaVerdict(1.0, f"{sig.direction.value} per value/quality family")
    print(run(YFinanceProvider(), args.ticker, date.fromisoformat(args.as_of), critic=critic))
    return 0
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_cli.py`
Expected: existing CLI tests + the new one pass.

- [ ] **Step 5: Commit** `feat(advisor): CLI --explain persona narration`.

---

### Task 7: Data-driven floor — price-only ensemble vs SPY on a committed fixture — **fixture-first (operator + Codex)**

Make the `advisor-gate` floor *data-driven* instead of an import-smoke. Per spec §6 the only honest backtest is **price/volume-only** (restated fundamentals + LLM look-ahead make the other families un-backtestable on free data), so the floor backtests the **price-only ensemble** (momentum + trend) vs SPY buy-and-hold on a committed real-price fixture.

**Operator pre-step (NOT Codex — needs network):** Fetch real daily closes for a small liquid universe + SPY spanning **≥2 regimes** (spec §7; e.g., 2018–2023 to include the 2020 crash and the 2022 drawdown) and commit them as `apps/quant/advisor/tests/fixtures/floor_prices.csv` (columns: `date, <TICKERS...>, SPY`). This fixture is the operator artifact; Codex writes logic against the *committed* file. **Pre-registered margin (spec §15): `0.0` for v1** — the ensemble must (a) not underperform SPY and (b) beat its own best single price family across the fixture window. Tighten after a sensitivity pass; record the chosen value here when changed.

**Files:**
- Create: `apps/quant/advisor/backtest/data_floor.py`
- Test: `apps/quant/advisor/tests/test_data_floor.py`
- Create: `tools/floor_data_check.py`
- Modify: `tools/run-floor.mjs`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_data_floor.py`:
```python
import numpy as np
import pandas as pd

from advisor.backtest.data_floor import floor_metrics


def _panel():
    # synthetic 2-regime panel: uptrend then drawdown, plus a benchmark column
    n = 600
    up = np.linspace(100, 180, n // 2)
    down = np.linspace(180, 130, n - n // 2)
    series = np.concatenate([up, down])
    return pd.DataFrame({"AAA": series, "BBB": series * 1.01, "SPY": series * 0.99})


def test_floor_metrics_has_expected_keys_and_types():
    m = floor_metrics(_panel(), benchmark="SPY", margin=0.0)
    for key in ("ensemble", "spy", "best_family", "margin", "passes"):
        assert key in m
    assert np.isfinite(m["ensemble"])
    assert np.isfinite(m["spy"])
    assert np.isfinite(m["best_family"])
    assert isinstance(m["passes"], bool)


def test_floor_metrics_passes_is_consistent_with_beats_floor():
    from advisor.backtest.floor import beats_floor
    m = floor_metrics(_panel(), benchmark="SPY", margin=0.0)
    assert m["passes"] == beats_floor(m["ensemble"], m["spy"], m["best_family"], m["margin"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_data_floor.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/backtest/data_floor.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.backtest.floor import beats_floor, purged_walk_forward_sharpe


def _momentum_signal(prices: pd.Series, lookback: int = 126) -> pd.Series:
    return np.sign(prices / prices.shift(lookback) - 1.0).fillna(0.0)


def _trend_signal(prices: pd.Series, short: int = 50, long: int = 200) -> pd.Series:
    return np.sign(prices.rolling(short).mean() - prices.rolling(long).mean()).fillna(0.0)


def _long_flat(signal: pd.Series) -> pd.Series:
    """Mirror the deployed allocator: it sizes a positive (long) position or holds.

    There are no shorting rails (allocator caps with a single positive dollar limit),
    so a bearish signal means FLAT (exit), not short. The floor must backtest what
    actually ships, not a more-penalized long/short variant.
    """
    return (signal > 0).astype(float)


def floor_metrics(panel: pd.DataFrame, benchmark: str = "SPY", margin: float = 0.0,
                  folds: int = 5, embargo: int = 5) -> dict:
    """Backtest the deployed long-flat price-only ensemble vs SPY (spec sections 6-7).

    Returns OOS purged-walk-forward Sharpes for the ensemble, SPY buy-and-hold, and the
    best single price family, plus whether the floor is cleared. All three strategy legs
    use identical long-flat position construction so "beat the parts" is a fair comparison.
    """
    tickers = [c for c in panel.columns if c != benchmark]
    ens, mom, tr = [], [], []
    for t in tickers:
        p = panel[t].dropna()
        ensemble_sig = np.sign(_momentum_signal(p) + _trend_signal(p))
        ens.append(purged_walk_forward_sharpe(p, _long_flat(ensemble_sig), folds, embargo))
        mom.append(purged_walk_forward_sharpe(p, _long_flat(_momentum_signal(p)), folds, embargo))
        tr.append(purged_walk_forward_sharpe(p, _long_flat(_trend_signal(p)), folds, embargo))

    ensemble = float(np.mean(ens)) if ens else 0.0
    best_family = max(float(np.mean(mom)) if mom else 0.0, float(np.mean(tr)) if tr else 0.0)

    spy = panel[benchmark].dropna()
    spy_sharpe = purged_walk_forward_sharpe(spy, pd.Series(1.0, index=spy.index), folds, embargo)

    return {"ensemble": ensemble, "spy": spy_sharpe, "best_family": best_family,
            "margin": float(margin),
            "passes": bool(beats_floor(ensemble, spy_sharpe, best_family, margin))}
```

> **Empirical result on the committed 2018–2023 fixture (long-flat, deployed semantics):** ensemble Sharpe **0.32**, SPY buy-and-hold **0.85**, best single family (trend) **0.48**. The v1 equal-weight ensemble clears **neither** §7 condition — it underperforms SPY *and* its own best family. This is expected: skill-weighting is v2 (spec §8), and equal-weighting a weak family (momentum 0.34) with a stronger one (trend 0.48) drags the blend below trend-alone. **The floor mechanism works; the v1 advisor is not production-ready.** See the gate split below — this does not block dev commits.

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_data_floor.py`
Expected: 2 passed.

- [ ] **Step 5: Add the gate runner script** (space-free args only — Windows `shell:true` gotcha, see memory).

The floor gates **production release**, not dev commits — the repo's own history ("Require advisor floor proof before *production release*") and the prior non-blocking import-smoke both confirm the per-commit gate never enforced floor *performance*. So the runner always prints the **real, honest** verdict (never green-washed), but only `exit 1` when `--enforce` is passed (the release gate). v1 will print "NOT CLEARED" — that is correct and must stay loud.

`tools/floor_data_check.py`:
```python
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "apps/quant")
from advisor.backtest.data_floor import floor_metrics  # noqa: E402

FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")
MARGIN = 0.0  # pre-registered v1 floor margin over SPY (spec section 15)


def main(argv: list[str]) -> int:
    enforce = "--enforce" in argv
    if not FIXTURE.exists():
        print(f"floor: fixture missing at {FIXTURE} -- commit real price history first", flush=True)
        return 1  # a broken mechanism always fails, in any mode
    panel = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    m = floor_metrics(panel, benchmark="SPY", margin=MARGIN)
    print("floor metrics: " + json.dumps(m), flush=True)
    if m["passes"]:
        print("floor: PASSED", flush=True)
        return 0
    print(f"floor: NOT CLEARED -- ensemble Sharpe {m['ensemble']:.2f} vs SPY {m['spy']:.2f} "
          f"and best single family {m['best_family']:.2f}. v1 equal-weight is not "
          f"production-ready; needs v2 calibration (spec section 8).", flush=True)
    return 1 if enforce else 0  # block release; do NOT block dev commits


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 6: Wire `run-floor.mjs` to run the data-driven check** (report mode by default; forwards `--enforce`). Replace `tools/run-floor.mjs` with:
```javascript
#!/usr/bin/env node
// advisor-gate stage 2: data-driven floor. Backtests the deployed long-flat price-only
// ensemble vs SPY on the committed fixture and prints the real verdict every run.
// Report mode (default) exits 0 so dev commits are not blocked; --enforce exits non-zero
// on a floor miss (used by advisor-release-gate, gating PRODUCTION RELEASE only).
import { spawnSync } from 'node:child_process';
const args = ['tools/floor_data_check.py'];
if (process.argv.includes('--enforce')) args.push('--enforce');
const r = spawnSync('python', args, {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});
process.exit(r.status ?? 1);
```

- [ ] **Step 7: Add the release gate to `package.json`.** Add a sibling script (leave `advisor-gate` as-is — it stays report-only via `run-floor.mjs`):
```json
    "advisor-release-gate": "node tools/run-pytest.mjs apps/quant/advisor/tests && node tools/run-floor.mjs --enforce"
```

- [ ] **Step 8: Verify both gates** (after the operator has committed the fixture)

Run (dev gate — must stay green): `node tools/run-pytest.mjs apps/quant/advisor/tests && node tools/run-floor.mjs`
Expected: tests pass, `floor metrics: {...}`, `floor: NOT CLEARED ...`, exit **0**.

Run (release gate — must block on v1): `node tools/run-floor.mjs --enforce`
Expected: `floor: NOT CLEARED ...`, exit **1**. This is correct: v1 is honestly not production-ready. Do **not** weaken the margin or cherry-pick a fixture to force a pass.

- [ ] **Step 9: Commit** the code (the operator commits the fixture separately, first): `feat(advisor): data-driven floor + advisor-release-gate (price-only ensemble vs SPY)`.

---

### Task 8: TimescaleDB persistence of the SignalBundle checkpoint — **Codex unit; operator integration**

Persist each `SignalBundle` to TimescaleDB as the pipeline checkpoint (spec §3). Codex writes the schema + a repository with an **injectable connection** and a fake-connection unit test (SQL-safety: parameterized, never f-string-interpolated). The operator runs the real integration against docker (`npm run db:up`) — the sandbox blocks DB access.

**Files:**
- Create: `apps/quant/advisor/persistence/__init__.py` (one-line docstring)
- Create: `apps/quant/advisor/persistence/checkpoint.py`
- Test: `apps/quant/advisor/tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_checkpoint.py`:
```python
from datetime import date

from advisor.persistence.checkpoint import SCHEMA_SQL, save_bundle
from advisor.schemas import Direction, FamilySignal, SignalBundle


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def executemany(self, sql, rows):
        self.calls.append((sql, list(rows)))


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


def _bundle():
    sigs = [
        FamilySignal(family="trend", direction=Direction.BULLISH, confidence=70.0, as_of=date(2024, 5, 1)),
        FamilySignal(family="macro", direction=Direction.BEARISH, confidence=60.0, as_of=date(2024, 5, 1)),
    ]
    return SignalBundle(ticker="AAPL", as_of=date(2024, 5, 1), signals=sigs)


def test_schema_declares_hypertable():
    assert "create_hypertable" in SCHEMA_SQL
    assert "signal_bundle" in SCHEMA_SQL


def test_save_bundle_is_parameterized_and_commits():
    conn = _FakeConn()
    n = save_bundle(conn, _bundle())
    assert n == 2
    assert conn.committed is True
    sql, rows = conn.cur.calls[0]
    assert "%s" in sql                 # parameterized -- no string interpolation
    assert "AAPL" not in sql           # ticker must be a bound param, not in the SQL text
    assert len(rows) == 2
    assert rows[0][1] == "AAPL"        # ticker bound as a value
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_checkpoint.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/persistence/__init__.py`:
```python
"""TimescaleDB persistence of the SignalBundle checkpoint (spec section 3)."""
```

`apps/quant/advisor/persistence/checkpoint.py`:
```python
from __future__ import annotations

from advisor.schemas import SignalBundle

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_bundle (
    as_of        date              NOT NULL,
    ticker       text              NOT NULL,
    family       text              NOT NULL,
    direction    text              NOT NULL,
    confidence   double precision  NOT NULL,
    skill_weight double precision  NOT NULL,
    reasoning    text              NOT NULL DEFAULT '',
    PRIMARY KEY (as_of, ticker, family)
);
SELECT create_hypertable('signal_bundle', 'as_of', if_not_exists => TRUE);
"""

_INSERT = (
    "INSERT INTO signal_bundle "
    "(as_of, ticker, family, direction, confidence, skill_weight, reasoning) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (as_of, ticker, family) DO UPDATE SET "
    "direction = EXCLUDED.direction, confidence = EXCLUDED.confidence, "
    "skill_weight = EXCLUDED.skill_weight, reasoning = EXCLUDED.reasoning"
)


def save_bundle(conn, bundle: SignalBundle) -> int:
    """Persist one row per FamilySignal. Parameterized -- never interpolate values into SQL."""
    rows = [
        (bundle.as_of, bundle.ticker, s.family, s.direction.value,
         s.confidence, s.skill_weight, s.reasoning)
        for s in bundle.signals
    ]
    with conn.cursor() as cur:
        cur.executemany(_INSERT, rows)
    conn.commit()
    return len(rows)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_checkpoint.py`
Expected: 3 passed.

- [ ] **Step 5: Commit** the `persistence/` files: `feat(advisor): TimescaleDB SignalBundle checkpoint (parameterized)`.

- [ ] **Step 6: Operator integration verify (NOT Codex — needs docker/DB).** With `npm run db:up` running, connect via `DATABASE_URL`, execute `SCHEMA_SQL`, `save_bundle(conn, bundle)`, and `SELECT count(*) FROM signal_bundle` to confirm the round-trip. Record the result; this step is not part of the pytest gate.

---

## Self-Review

**1. Spec coverage (Plan 3 scope = §13 steps 7–8 + remaining §3 families):**
- Trend family (§3 family 3) → Task 1 ✅
- Macro/rates regime (§3 family 4, FRED) → Task 2 ✅
- News/sentiment surprise (§3 family 5; §10 no-fabrication) → Task 3 ✅
- Persona overlay v1 critique/explain, veto/downgrade-never-upsize (§5, §13 step 7) → Tasks 4–6 ✅
- Data-driven floor mechanism: deployed long-flat price-only ensemble vs SPY + best family, purged walk-forward, pre-registered margin, ≥2 regimes (§6, §7); report-only in per-commit `advisor-gate`, enforced in `advisor-release-gate` (§9, §13 step 8) → Task 7 ✅ (mechanism). **⚠️ Effectiveness: on real 2018–2023 data the v1 equal-weight ensemble does NOT clear the §7 floor (ensemble 0.32 < SPY 0.85 and < trend-alone 0.48). The advisor is honestly not production-ready — that is the floor doing its job, and the v1 "definition of done" (§7 gate passing) is therefore NOT met. Closing it is v2 calibration work (spec §8), out of Plan 3 scope.**
- TimescaleDB SignalBundle checkpoint (§3) → Task 8 ✅
- Deferred to v2 (stated, spec §5/§8): skill-weighting/calibration; information-diverse persona deciders + forward-tracking harness.

**2. Dispatch-fit (the Plan-3-specific risk):** Tasks 1–6 verify with `python -m pytest` against fakes/mocks → Codex-dispatchable as written. Tasks 7–8 need network/DB the sandbox forbids → fixture-first: operator commits the real artifact, Codex writes logic against the committed fixture / fake connection, pytest verifies offline; the live steps (7.7, 8.6) are operator-run and explicitly outside the pytest gate.

**3. Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step is complete. The v1 floor margin (`0.0`) and the persona clamp (`[0,1]`) are explicit, pre-registered values with stated rationale — not hidden gaps.

**4. Type consistency:** `FamilySignal`/`SignalBundle`/`Direction`/`FamilySignal.neutral` reused exactly from the existing `schemas.py`. New families all expose `FAMILY: str` + `evaluate(...) -> FamilySignal` (matching `momentum`/`value_quality`). `Decision` is reused from `pipeline/run.py`; the overlay stays decoupled via `dataclasses.replace`. `PersonaVerdict(size_multiplier, explanation)` and `apply_overlay(decision, critic) -> Decision` are consistent across Tasks 4–6. `purged_walk_forward_sharpe`/`beats_floor` reused from `backtest/floor.py` by Task 7. `save_bundle(conn, bundle) -> int` consistent across Task 8 test + impl.

**5. Trust boundary (the load-bearing invariant):** spec §5 — a persona may veto/downgrade, never upsize or touch risk limits — is enforced *in code* by the `max(0.0, min(1.0, ...))` clamp in `apply_overlay`, and asserted directly by `test_persona_cannot_upsize_multiplier_is_clamped`. The LLM/persona never sets `direction`, `skill_weight`, or position size at the seam.
