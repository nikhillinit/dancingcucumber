# AI Advisor — Foundation Slice (Plan 1 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Workspace-contract note (this repo) — UNRESOLVED SEQUENCING:** The project contract says code edits/tests are **dispatched via Hermes** (`npm run hermes:production -- --task "<task>"`). But Hermes today hard-codes its ROOT to `C:\dev\Updog_restore` and injects "operating inside Updog_restore" — it **cannot edit files in `C:\dev\AIHedgeFund` until it is ported**, and that port is scoped into **Plan 3**. So Plan 1 cannot literally run "via Hermes" yet. This must be resolved before execution (see **Execution decision** at the bottom): either pull the Hermes port ahead as a Plan 0 prerequisite, or execute Plan 1 directly with the contract taking effect once Hermes is ported. Do not assume "via Hermes" is operational for this plan.

## Before you start (preflight)

- [ ] **Create a working branch** (do NOT commit onto the unrelated `update-codacy-instructions` branch):
```
git switch -c design/ai-advisor-foundation
```
- [ ] **CLI import path:** the documented `python -m advisor` resolves only with `apps/quant` on the path. Run it as `$env:PYTHONPATH="apps/quant"; python -m advisor AAPL` (PowerShell) or from inside `apps/quant`. (Tests are unaffected — `pytest.ini` sets `pythonpath = apps/quant`.)

**Goal:** Stand up a clean, isolated `advisor` package and ship one backtestable signal family (value/quality) end-to-end — raw fundamentals → frozen `SignalBundle` → a price/volume walk-forward backtest with mandatory bias disclosure.

**Architecture:** Deterministic-first. Pure-Python signal families compute every number; the frozen Pydantic `SignalBundle` is the trust seam; an honest point-in-time backtest with a 4-line disclosure header is the load-bearing deliverable. No LLM and no LangGraph in this slice. (Spec: `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md`, sequence steps 1–3 + minimal step 6.)

**Tech Stack:** Python 3.13, Pydantic v2, pandas, numpy, yfinance, pytest. New package isolated at `apps/quant/advisor/` (import root `apps/quant`, so modules import as `advisor.*`).

**Scope note:** This is Plan 1 of 3. Plan 2 = fan-out to all 5 families + deterministic risk manager + portfolio allocator + equal-weight ensemble gate. Plan 3 = persona critique/explain overlay + Hermes `advisor-gate` port. Each plan ships working, testable software on its own.

---

### Task 1: Repo cleanup — delete duplicate "systems" and dead ML stubs

Removes the sprawl the spec calls out (§10) so it cannot rot or be imported by mistake. Verify nothing we keep imports them before deleting.

**Files:**
- Delete: `production_ready_system.py`, `robust_trading_system.py`, `automated_trading_system.py`, `enhanced_training_system.py`, `personalized_portfolio_system.py`, `single_user_ai_system.py` (repo root)
- Delete: `apps/quant/finrl_trading_agent.py`, `apps/quant/qlib_factor_generator.py`, `apps/quant/autogluon_ensemble.py`

- [ ] **Step 1: Confirm none are imported by code we keep**

Run (PowerShell):
```
Get-ChildItem -Recurse -Filter *.py -Path apps,. -ErrorAction SilentlyContinue |
  Select-String -Pattern "import (finrl_trading_agent|qlib_factor_generator|autogluon_ensemble|production_ready_system|robust_trading_system|automated_trading_system|enhanced_training_system|personalized_portfolio_system|single_user_ai_system)" |
  Select-Object Path, LineNumber, Line
```
Expected: no matches outside the files themselves. If a match appears in a file we keep, STOP and note it — do not delete that target.

- [ ] **Step 2: Delete the files (robust to tracked AND untracked)**

The root `*_system.py` files are **untracked**; the `apps/quant/*` stubs are **tracked**. A bulk `git rm` fails atomically when any pathspec is untracked, so delete from disk first, then stage with `git add -A` (which records deletions of tracked files):
```powershell
$targets = @(
  "production_ready_system.py","robust_trading_system.py","automated_trading_system.py",
  "enhanced_training_system.py","personalized_portfolio_system.py","single_user_ai_system.py",
  "apps/quant/finrl_trading_agent.py","apps/quant/qlib_factor_generator.py","apps/quant/autogluon_ensemble.py"
)
foreach ($t in $targets) { if (Test-Path $t) { Remove-Item $t -Force } }
git add -A
```
Expected: `git status` shows the tracked `apps/quant/*` files as deleted and the root files gone from the working tree.

- [ ] **Step 3: Verify the repo still has no broken imports for kept modules**

Run: `python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('apps/quant/*.py')]"`
Expected: no SyntaxError (this only parses; it does not execute imports).

- [ ] **Step 4: Commit**

```
git add -A
git commit -m "chore: remove duplicate trading-system scripts and dead ML stubs"
```

---

### Task 2: Isolated `advisor` package + pytest harness

Creates a clean package that does not entangle the existing loose scripts, plus a pytest config and a dependency file. A trivial smoke test proves the harness runs.

**Files:**
- Create: `apps/quant/advisor/__init__.py`
- Create: `apps/quant/advisor/data/__init__.py`
- Create: `apps/quant/advisor/analysis/__init__.py`
- Create: `apps/quant/advisor/backtest/__init__.py`
- Create: `apps/quant/advisor/tests/__init__.py`
- Create: `pytest.ini` (repo root)
- Create: `requirements-advisor.txt` (repo root)
- Test: `apps/quant/advisor/tests/test_smoke.py`

- [ ] **Step 1: Create the package `__init__.py` files**

Each of these files has the same one-line content:
```python
"""AI advisor package (deterministic-first). See docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md"""
```
Create all five: `apps/quant/advisor/__init__.py`, `apps/quant/advisor/data/__init__.py`, `apps/quant/advisor/analysis/__init__.py`, `apps/quant/advisor/backtest/__init__.py`, `apps/quant/advisor/tests/__init__.py`.

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
pythonpath = apps/quant
testpaths = apps/quant/advisor/tests
addopts = -q
```

- [ ] **Step 3: Create `requirements-advisor.txt`**

```
pydantic>=2.6
pandas>=2.0
numpy>=1.26
yfinance>=0.2.40
pytest>=8.0
```

- [ ] **Step 4: Install deps**

Run: `python -m pip install -r requirements-advisor.txt`
Expected: installs without error.

- [ ] **Step 5: Write the smoke test**

`apps/quant/advisor/tests/test_smoke.py`:
```python
import advisor


def test_package_imports():
    assert advisor.__doc__ is not None
```

- [ ] **Step 6: Run it**

Run: `python -m pytest apps/quant/advisor/tests/test_smoke.py`
Expected: 1 passed.

- [ ] **Step 7: Commit**

```
git add apps/quant/advisor pytest.ini requirements-advisor.txt
git commit -m "feat(advisor): isolated package skeleton + pytest harness"
```

---

### Task 3: The frozen Pydantic seam (`schemas.py`)

Defines the trust boundary types. `FamilySignal` carries a safe `neutral()` fallback (used everywhere parsing or data can fail).

**Files:**
- Create: `apps/quant/advisor/schemas.py`
- Test: `apps/quant/advisor/tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_schemas.py`:
```python
from datetime import date

import pytest
from pydantic import ValidationError

from advisor.schemas import Direction, FamilySignal, SignalBundle


def test_family_signal_valid():
    s = FamilySignal(family="value_quality", direction=Direction.BULLISH,
                     confidence=80.0, as_of=date(2024, 1, 2))
    assert s.direction is Direction.BULLISH
    assert s.skill_weight == 1.0


def test_confidence_must_be_0_100():
    with pytest.raises(ValidationError):
        FamilySignal(family="x", direction=Direction.NEUTRAL,
                     confidence=150.0, as_of=date(2024, 1, 2))


def test_neutral_fallback():
    s = FamilySignal.neutral("value_quality", date(2024, 1, 2))
    assert s.direction is Direction.NEUTRAL
    assert s.confidence == 50.0


def test_bundle_collects_signals():
    s = FamilySignal.neutral("value_quality", date(2024, 1, 2))
    b = SignalBundle(ticker="AAPL", as_of=date(2024, 1, 2), signals=[s])
    assert b.signals[0].family == "value_quality"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_schemas.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'advisor.schemas'`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/schemas.py`:
```python
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class FamilySignal(BaseModel):
    model_config = {"frozen": True}

    family: str
    direction: Direction
    confidence: float = Field(ge=0, le=100)
    skill_weight: float = Field(default=1.0, ge=0)
    as_of: date
    reasoning: str = ""

    @classmethod
    def neutral(cls, family: str, as_of: date, reasoning: str = "insufficient data") -> "FamilySignal":
        return cls(family=family, direction=Direction.NEUTRAL, confidence=50.0,
                   as_of=as_of, reasoning=reasoning)


class SignalBundle(BaseModel):
    model_config = {"frozen": True}

    ticker: str
    as_of: date
    signals: list[FamilySignal] = Field(default_factory=list)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_schemas.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add apps/quant/advisor/schemas.py apps/quant/advisor/tests/test_schemas.py
git commit -m "feat(advisor): frozen Pydantic SignalBundle seam"
```

---

### Task 4: Data provider interface + point-in-time fundamentals guard

Defines `MarketDataProvider`, a `Fundamentals` record, the 90-day availability guard (the look-ahead defense), a `FakeProvider` test double, and a thin `YFinanceProvider`. Only the deterministic guard + fake are unit-tested; the network adapter is verified manually.

**Files:**
- Create: `apps/quant/advisor/data/provider.py`
- Create: `apps/quant/advisor/data/fakes.py`
- Test: `apps/quant/advisor/tests/test_provider.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_provider.py`:
```python
from datetime import date

from advisor.data.provider import (
    Fundamentals,
    is_available_asof,
    select_latest_available,
)
from advisor.data.fakes import FakeProvider


def _fund(period_end: date) -> Fundamentals:
    return Fundamentals(period_end=period_end, net_income=100, total_equity=500,
                        revenue=1000, operating_income=200, total_debt=100,
                        depreciation=50, capex=40, shares_outstanding=10, market_cap=900)


def test_availability_guard_blocks_too_recent():
    # period ends 2024-01-01; with 90-day lag it is only available from ~2024-03-31
    assert is_available_asof(date(2024, 1, 1), date(2024, 2, 1)) is False
    assert is_available_asof(date(2024, 1, 1), date(2024, 5, 1)) is True


def test_select_latest_available_picks_newest_lagged_record():
    records = [_fund(date(2023, 9, 30)), _fund(date(2023, 12, 31)), _fund(date(2024, 3, 31))]
    # as of 2024-04-15 only the first two are >90 days old
    chosen = select_latest_available(records, date(2024, 4, 15))
    assert chosen is not None
    assert chosen.period_end == date(2023, 12, 31)


def test_fake_provider_returns_configured_fundamentals():
    p = FakeProvider(fundamentals={"AAPL": _fund(date(2023, 12, 31))})
    f = p.get_fundamentals_asof("AAPL", date(2024, 5, 1))
    assert f is not None and f.revenue == 1000
    assert p.get_fundamentals_asof("MSFT", date(2024, 5, 1)) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_provider.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'advisor.data.provider'`.

- [ ] **Step 3: Write `provider.py`**

`apps/quant/advisor/data/provider.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

REPORTING_LAG_DAYS = 90


@dataclass(frozen=True)
class Fundamentals:
    period_end: date
    net_income: float
    total_equity: float
    revenue: float
    operating_income: float
    total_debt: float
    depreciation: float
    capex: float
    shares_outstanding: float
    market_cap: float


def is_available_asof(period_end: date, as_of: date, lag_days: int = REPORTING_LAG_DAYS) -> bool:
    """A report is only knowable after period_end + a conservative reporting lag."""
    return period_end + timedelta(days=lag_days) <= as_of


def select_latest_available(records: list[Fundamentals], as_of: date,
                            lag_days: int = REPORTING_LAG_DAYS) -> Fundamentals | None:
    eligible = [r for r in records if is_available_asof(r.period_end, as_of, lag_days)]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r.period_end)


class MarketDataProvider(Protocol):
    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...
    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None: ...


class YFinanceProvider:
    """Thin yfinance adapter. Network-bound; verified manually, not in unit tests.

    yfinance fundamentals are RESTATED, not as-reported — the point-in-time guard
    above only approximates availability. See spec §6 disclosures.
    """

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=True)
        return df

    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        fin = t.financials  # columns are period-end dates
        records: list[Fundamentals] = []
        for col in getattr(fin, "columns", []):
            period_end = col.date() if hasattr(col, "date") else col
            try:
                records.append(Fundamentals(
                    period_end=period_end,
                    net_income=float(fin.loc["Net Income", col]),
                    total_equity=float(info.get("totalStockholderEquity") or 0) or 1.0,
                    revenue=float(fin.loc["Total Revenue", col]),
                    operating_income=float(fin.loc["Operating Income", col]),
                    total_debt=float(info.get("totalDebt") or 0),
                    depreciation=float(info.get("ebitda", 0)) - float(info.get("operatingCashflow", 0) or 0),
                    capex=float(info.get("capitalExpenditures") or 0),
                    shares_outstanding=float(info.get("sharesOutstanding") or 0),
                    market_cap=float(info.get("marketCap") or 0),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return select_latest_available(records, as_of)
```

- [ ] **Step 4: Write `fakes.py`**

`apps/quant/advisor/data/fakes.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from advisor.data.provider import Fundamentals


@dataclass
class FakeProvider:
    fundamentals: dict[str, Fundamentals] = field(default_factory=dict)
    prices: dict[str, pd.Series] = field(default_factory=dict)

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        s = self.prices.get(ticker, pd.Series(dtype=float))
        return pd.DataFrame({"Close": s})

    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None:
        return self.fundamentals.get(ticker)
```

- [ ] **Step 5: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_provider.py`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```
git add apps/quant/advisor/data apps/quant/advisor/tests/test_provider.py
git commit -m "feat(advisor): data provider interface + point-in-time fundamentals guard"
```

---

### Task 5: Vendored valuation primitives (MIT, attributed)

Copies the deterministic valuation math from virattt/ai-hedge-fund (MIT). Plan 1 needs only owner-earnings + multi-stage DCF; EV/EBITDA and residual income arrive in Plan 2.

**Files:**
- Create: `apps/quant/advisor/analysis/valuation_primitives.py`
- Test: `apps/quant/advisor/tests/test_valuation_primitives.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_valuation_primitives.py`:
```python
import pytest

from advisor.analysis.valuation_primitives import intrinsic_value_dcf, owner_earnings


def test_owner_earnings_basic():
    # NI 100 + D&A 50 - capex 40 - ΔWC 10 = 100
    assert owner_earnings(100, 50, 40, 10) == 100


def test_dcf_zero_growth_zero_terminal_growth_closed_form():
    # base=100, g=0, tg=0, r=0.10, horizon=10, no margin of safety.
    # annuity(10y @10%)*100 + (100/0.10)/1.1^10 ≈ 614.46 + 385.54 = 1000.0
    v = intrinsic_value_dcf(100, growth_rate=0.0, terminal_growth=0.0,
                            discount_rate=0.10, horizon=10, margin_of_safety=0.0)
    assert v == pytest.approx(1000.0, rel=1e-3)


def test_dcf_nonpositive_cashflow_is_zero():
    assert intrinsic_value_dcf(0) == 0.0
    assert intrinsic_value_dcf(-50) == 0.0


def test_dcf_higher_growth_yields_higher_value():
    low = intrinsic_value_dcf(100, growth_rate=0.02)
    high = intrinsic_value_dcf(100, growth_rate=0.10)
    assert high > low
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_valuation_primitives.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/analysis/valuation_primitives.py`:
```python
"""Deterministic valuation primitives.

Vendored and adapted from virattt/ai-hedge-fund (MIT License):
https://github.com/virattt/ai-hedge-fund (src/agents/warren_buffett.py, src/agents/valuation.py)
Adapted to our data types; math unchanged. MIT notice retained per LICENSE.
"""
from __future__ import annotations


def owner_earnings(net_income: float, depreciation: float, capex: float,
                   working_capital_change: float = 0.0) -> float:
    """Buffett owner earnings: NI + D&A - maintenance capex - ΔWorking Capital."""
    return net_income + depreciation - abs(capex) - working_capital_change


def intrinsic_value_dcf(base_cash_flow: float, growth_rate: float = 0.06,
                        terminal_growth: float = 0.025, discount_rate: float = 0.10,
                        horizon: int = 10, margin_of_safety: float = 0.15) -> float:
    """Multi-stage DCF with Gordon terminal value and a margin-of-safety haircut."""
    if base_cash_flow <= 0:
        return 0.0
    pv = 0.0
    cf = base_cash_flow
    for year in range(1, horizon + 1):
        cf = cf * (1 + growth_rate)
        pv += cf / ((1 + discount_rate) ** year)
    terminal_cf = cf * (1 + terminal_growth)
    terminal_value = terminal_cf / (discount_rate - terminal_growth)
    pv += terminal_value / ((1 + discount_rate) ** horizon)
    return pv * (1 - margin_of_safety)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_valuation_primitives.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add apps/quant/advisor/analysis/valuation_primitives.py apps/quant/advisor/tests/test_valuation_primitives.py
git commit -m "feat(advisor): vendored DCF + owner-earnings primitives (MIT)"
```

---

### Task 6: The value/quality signal family

Turns a `Fundamentals` record into a `FamilySignal` using a deterministic quality score + a DCF-based margin of safety. No LLM.

**Files:**
- Create: `apps/quant/advisor/analysis/value_quality.py`
- Test: `apps/quant/advisor/tests/test_value_quality.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_value_quality.py`:
```python
from datetime import date

from advisor.analysis.value_quality import FAMILY, evaluate
from advisor.data.provider import Fundamentals
from advisor.schemas import Direction


def _fund(net_income, total_equity, revenue, operating_income, total_debt,
          depreciation, capex, market_cap) -> Fundamentals:
    return Fundamentals(period_end=date(2023, 12, 31), net_income=net_income,
                        total_equity=total_equity, revenue=revenue,
                        operating_income=operating_income, total_debt=total_debt,
                        depreciation=depreciation, capex=capex,
                        shares_outstanding=10, market_cap=market_cap)


def test_high_quality_cheap_is_bullish():
    # ROE 20%, op margin 25%, D/E 0.2 (quality 6/6); cheap vs DCF -> bullish
    f = _fund(net_income=200, total_equity=1000, revenue=800, operating_income=200,
              total_debt=200, depreciation=60, capex=40, market_cap=500)
    sig = evaluate(f, date(2024, 5, 1))
    assert sig.family == FAMILY
    assert sig.direction is Direction.BULLISH
    assert sig.confidence > 50


def test_expensive_is_bearish():
    f = _fund(net_income=50, total_equity=1000, revenue=800, operating_income=80,
              total_debt=900, depreciation=20, capex=60, market_cap=50_000)
    sig = evaluate(f, date(2024, 5, 1))
    assert sig.direction is Direction.BEARISH


def test_nonpositive_equity_is_neutral():
    f = _fund(net_income=10, total_equity=0, revenue=100, operating_income=5,
              total_debt=10, depreciation=5, capex=5, market_cap=100)
    sig = evaluate(f, date(2024, 5, 1))
    assert sig.direction is Direction.NEUTRAL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_value_quality.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/analysis/value_quality.py`:
```python
from __future__ import annotations

from datetime import date

from advisor.analysis.valuation_primitives import intrinsic_value_dcf, owner_earnings
from advisor.data.provider import Fundamentals
from advisor.schemas import Direction, FamilySignal

FAMILY = "value_quality"


def _quality_score(f: Fundamentals) -> int:
    """0..6 from ROE, operating margin, and leverage."""
    roe = f.net_income / f.total_equity if f.total_equity else 0.0
    op_margin = f.operating_income / f.revenue if f.revenue else 0.0
    d_e = f.total_debt / f.total_equity if f.total_equity else float("inf")
    score = 0
    if roe > 0.15:
        score += 2
    if op_margin > 0.15:
        score += 2
    if d_e < 0.5:
        score += 2
    return score


def evaluate(f: Fundamentals, as_of: date) -> FamilySignal:
    if f.total_equity <= 0 or f.revenue <= 0 or f.market_cap <= 0:
        return FamilySignal.neutral(FAMILY, as_of, "non-positive equity/revenue/market cap")

    quality = _quality_score(f)
    iv = intrinsic_value_dcf(owner_earnings(f.net_income, f.depreciation, f.capex))
    margin_of_safety = (iv - f.market_cap) / f.market_cap

    if margin_of_safety > 0.15 and quality >= 4:
        direction = Direction.BULLISH
    elif margin_of_safety < -0.15:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    confidence = min(100.0, 50.0 + quality * 5.0 + min(abs(margin_of_safety), 1.0) * 30.0)
    reasoning = (f"quality={quality}/6, margin_of_safety={margin_of_safety:.0%}, "
                 f"intrinsic={iv:.0f} vs market_cap={f.market_cap:.0f}")
    return FamilySignal(family=FAMILY, direction=direction, confidence=confidence,
                        as_of=as_of, reasoning=reasoning)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_value_quality.py`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add apps/quant/advisor/analysis/value_quality.py apps/quant/advisor/tests/test_value_quality.py
git commit -m "feat(advisor): value/quality signal family"
```

---

### Task 7: Walk-forward backtest + mandatory disclosure + floor gate

A minimal, look-ahead-safe (position = lagged signal) cost-aware backtest that ALWAYS emits the 4-line disclosure header, plus the `passes_floor` acceptance check (spec §7).

**Files:**
- Create: `apps/quant/advisor/backtest/walk_forward.py`
- Test: `apps/quant/advisor/tests/test_walk_forward.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_walk_forward.py`:
```python
import numpy as np
import pandas as pd

from advisor.backtest.walk_forward import (
    DISCLOSURES,
    disclosure_header,
    passes_floor,
    walk_forward,
)


def test_disclosure_header_has_all_four_lines():
    header = disclosure_header()
    assert len(DISCLOSURES) == 4
    for line in DISCLOSURES:
        assert line in header


def test_long_only_on_uptrend_is_profitable_and_carries_disclosures():
    prices = pd.Series(np.linspace(100, 200, 50))  # steady uptrend
    signal = pd.Series(1.0, index=prices.index)     # always long
    result = walk_forward(prices, signal, cost_per_turn=0.0)
    assert result.total_return > 0
    assert result.n_periods == 50
    assert len(result.disclosures) == 4


def test_costs_reduce_return():
    prices = pd.Series(np.linspace(100, 110, 30))
    flip = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(30)])
    no_cost = walk_forward(prices, flip, cost_per_turn=0.0).total_return
    with_cost = walk_forward(prices, flip, cost_per_turn=0.01).total_return
    assert with_cost < no_cost


def test_passes_floor():
    assert passes_floor(1.2, 0.8, margin=0.3) is True
    assert passes_floor(1.0, 0.8, margin=0.3) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_walk_forward.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`apps/quant/advisor/backtest/walk_forward.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISCLOSURES = [
    "Fundamentals are restated, not as-reported.",
    "Point-in-time lag approximated (~90-day proxy); results indicative, not as-reported.",
    "yfinance is survivorship-biased (delisted names absent); long-side results upward-biased.",
    "Any LLM/news-derived feature may carry look-ahead from pretraining that cannot be purged.",
]


def disclosure_header() -> str:
    return "DISCLOSURES:\n" + "\n".join(f"  - {d}" for d in DISCLOSURES)


@dataclass(frozen=True)
class BacktestResult:
    sharpe: float
    total_return: float
    n_periods: int
    disclosures: list[str]


def _sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    std = returns.std(ddof=0)
    if std == 0 or len(returns) == 0:
        return 0.0
    excess = returns - rf / periods_per_year
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def walk_forward(prices: pd.Series, signal: pd.Series, cost_per_turn: float = 0.0005) -> BacktestResult:
    """Position is YESTERDAY's signal (no look-ahead). Cost charged on position change."""
    position = signal.shift(1).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * asset_ret - turnover * cost_per_turn
    total = float((1 + strat_ret).prod() - 1)
    return BacktestResult(sharpe=_sharpe(strat_ret), total_return=total,
                          n_periods=int(len(strat_ret)), disclosures=list(DISCLOSURES))


def passes_floor(strategy_sharpe: float, benchmark_sharpe: float, margin: float) -> bool:
    """Spec §7 floor: beat the benchmark by a pre-registered margin."""
    return (strategy_sharpe - benchmark_sharpe) >= margin
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_walk_forward.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add apps/quant/advisor/backtest apps/quant/advisor/tests/test_walk_forward.py
git commit -m "feat(advisor): look-ahead-safe walk-forward backtest + disclosure + floor gate"
```

---

### Task 8: Single-family CLI pipeline

Wires provider → value/quality → output, with the disclosure header always printed. Runnable as `python -m advisor`.

**Files:**
- Create: `apps/quant/advisor/cli.py`
- Create: `apps/quant/advisor/__main__.py`
- Test: `apps/quant/advisor/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`apps/quant/advisor/tests/test_cli.py`:
```python
from datetime import date

from advisor.cli import run
from advisor.data.fakes import FakeProvider
from advisor.data.provider import Fundamentals


def test_run_emits_direction_and_disclosures():
    f = Fundamentals(period_end=date(2023, 12, 31), net_income=200, total_equity=1000,
                     revenue=800, operating_income=200, total_debt=200, depreciation=60,
                     capex=40, shares_outstanding=10, market_cap=500)
    provider = FakeProvider(fundamentals={"AAPL": f})
    out = run(provider, "AAPL", date(2024, 5, 1))
    assert "AAPL" in out
    assert "bullish" in out
    assert "DISCLOSURES:" in out


def test_run_handles_missing_fundamentals():
    out = run(FakeProvider(), "ZZZZ", date(2024, 5, 1))
    assert "no point-in-time fundamentals" in out
    assert "DISCLOSURES:" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_cli.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'advisor.cli'`.

- [ ] **Step 3: Write `cli.py`**

`apps/quant/advisor/cli.py`:
```python
from __future__ import annotations

import argparse
from datetime import date

from advisor.analysis.value_quality import evaluate
from advisor.backtest.walk_forward import disclosure_header
from advisor.data.provider import MarketDataProvider, YFinanceProvider


def run(provider: MarketDataProvider, ticker: str, as_of: date) -> str:
    f = provider.get_fundamentals_asof(ticker, as_of)
    if f is None:
        return f"{ticker}: no point-in-time fundamentals available as of {as_of}\n{disclosure_header()}"
    sig = evaluate(f, as_of)
    return (f"{ticker} [{sig.direction.value}] confidence={sig.confidence:.0f} :: {sig.reasoning}\n"
            f"{disclosure_header()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advisor", description="Single-family AI advisor (value/quality)")
    parser.add_argument("ticker")
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="YYYY-MM-DD point-in-time date")
    args = parser.parse_args(argv)
    print(run(YFinanceProvider(), args.ticker, date.fromisoformat(args.as_of)))
    return 0
```

- [ ] **Step 4: Write `__main__.py`**

`apps/quant/advisor/__main__.py`:
```python
from advisor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_cli.py`
Expected: 2 passed.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest`
Expected: all tests pass (schemas, provider, valuation, value_quality, walk_forward, cli, smoke).

- [ ] **Step 7: Commit**

```
git add apps/quant/advisor/cli.py apps/quant/advisor/__main__.py apps/quant/advisor/tests/test_cli.py
git commit -m "feat(advisor): single-family CLI pipeline with mandatory disclosures"
```

---

## Self-Review

**1. Spec coverage (this slice):**
- §10 cleanup → Task 1 ✅
- §3 seam / §4 `schemas/` → Task 3 ✅
- §4 `data/` + §6 point-in-time guard → Task 4 ✅
- §1 reuse virattt math / §4 `valuation_primitives.py` → Task 5 ✅
- §3 one signal family end-to-end → Task 6 ✅
- §6 honest backtest + 4-line disclosure / §7 floor gate → Task 7 ✅
- §13 step "one CLI command runs the pipeline" → Task 8 ✅
- Deferred to Plan 2/3 (intentionally out of slice): remaining 4 families, risk manager, portfolio allocator, ensemble + calibration, LLM extractor/explainer, persona overlay, Hermes `advisor-gate`. Noted in the scope note.

**2. Placeholder scan:** No "TBD/TODO/handle edge cases/similar to Task N". Every code step shows complete code. ✅

**3. Type consistency:** `Fundamentals`, `FamilySignal`, `Direction`, `SignalBundle` field names and signatures are identical across Tasks 3–8. `evaluate(f, as_of)`, `walk_forward(prices, signal, cost_per_turn)`, `passes_floor(strategy_sharpe, benchmark_sharpe, margin)`, `run(provider, ticker, as_of)` match their tests. `select_latest_available` / `is_available_asof` signatures match Task 4 tests. ✅

**Known non-blocking note:** `YFinanceProvider.get_fundamentals_asof` field mapping (esp. depreciation) is approximate and network-bound; it is explicitly out of unit-test scope and must be manually validated against one real ticker before Plan 2 relies on it.
