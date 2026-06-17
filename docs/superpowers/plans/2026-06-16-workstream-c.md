# Workstream C — Live 5-Family CLI Assembly Implementation Plan

> **For agentic workers:** This plan is executed via **Hermes solo dispatch** (Codex owns edits), NOT subagent/inline. See "Execution via Hermes" below. Steps use checkbox (`- [ ]`) syntax for tracking. Each task = tests-first, one Hermes dispatch, one commit.

**Goal:** Wire all five signal families (value/quality, trend, momentum, macro, sentiment) into the live CLI through the existing `run_pipeline`, with FRED + news provider adapters, a deterministic lexicon news scorer, and fake-provider unit tests — strictly report-only.

**Architecture:** Add two network adapters behind tiny Protocols (FRED `T10Y2Y` series; a composite news provider fronting an Alpha Vantage adapter with graceful degradation). Add a deterministic, auditable lexicon scorer (no LLM → no pretraining look-ahead). Wrap each existing pure `evaluate()` in an async family-coro factory that fetches inputs (capped at `≤ as_of`) and falls back to `FamilySignal.neutral` on any missing input/exception. A new `--families all` CLI path assembles the five coros and calls `run_pipeline`. The floor, allocator, and ensemble are NEVER touched.

**Tech Stack:** Python 3.11, pydantic v2, pandas, pytest, `requests` (network adapters, injected for tests), argparse.

---

## Hard rails (violating any fails the task)

1. **Never modify** `portfolio/allocator.py`, `ensemble_vote`, `allocate`, `risk/limits.py`, or anything under `backtest/` (floor). `run_pipeline` already calls them; you only assemble inputs. (`allocator.py` is guarded by `.claude/hooks/guard-frozen-floor.mjs`.)
2. **Never weaken the release gate.** `node tools/run-floor.mjs --enforce` must stay **exit 1**; `npm run advisor-gate` stays **exit 0**. C touches only the live decision/reporting path.
3. **No fabricated data (spec §10).** A missing/failed source ⇒ `FamilySignal.neutral(...)`, NEVER a synthesized/optimistic signal. Adapters return empty (`pd.Series(dtype=float)` / `[]`) when keyless or on error; coros convert empty/None → neutral.
4. **As-of bounding is a correctness requirement, not a detail.** Every adapter query is capped at `≤ as_of` (FRED `observation_end=as_of`; AV `time_to=as_of`). Tests assert the requested upper bound equals `as_of`. Unbounded queries silently leak look-ahead and break the report-only honesty the floor exists to protect.
5. **Report-only caveat is load-bearing.** A working 5-family CLI produces **report-only** signals; the floor still blocks and the advisor must **not size real capital**. Keep `disclosure_header()` on every CLI output.
6. **No secrets in code or commits.** Keys come from env only (`FRED_API_KEY`, `ALPHAVANTAGE_API_KEY`). The hardcoded keys in the untracked root scripts (`fred_economic_analysis.py`, `alpha_vantage_enhanced_analysis.py`) are NOT a pattern to copy; never `git add` them.
7. **Scorer is deterministic lexicon, never an LLM.** An LLM scorer reintroduces exactly the pretraining look-ahead the floor excludes.

## Execution via Hermes (per memory `hermes-dispatch-windows`)

For each task: write the full task instruction — including the **COMPLETE final content** of every file the task creates/modifies — to `ai-logs/hermes/c-task-N.md`, then dispatch the short file-pointer:

```powershell
$env:PYTHONUTF8=1
npm run hermes:production -- --task "In this repository, open the file ai-logs/hermes/c-task-N.md and follow its instructions EXACTLY. Do NOT run npm or node; verify ONLY with: python -m pytest <task test path> -q. Do NOT commit; leave changes in the working tree for review."
```

Then: `git status` + `git diff --stat` to verify Codex's REAL git state (watch for unexpected DELETIONS / clobber), run the task's pytest yourself, and commit one-per-task with the task's message. Set `$env:PYTHONUTF8=1`. Plan-bug rule: if a seeded assertion fails against the impl, fix the fixture/threshold in place yourself — do NOT re-dispatch in a loop.

---

### Task 1: Shared test fakes module

**Files:**
- Create: `apps/quant/advisor/tests/fakes.py`
- Test: `apps/quant/advisor/tests/test_fakes.py`

Recording fakes for all three provider Protocols, reused by Tasks 2/4/5/6. Created complete here and never re-touched (clobber-safe).

- [ ] **Step 1: Write the failing test** — `apps/quant/advisor/tests/test_fakes.py`

```python
from datetime import date

import pandas as pd

from advisor.tests.fakes import (
    FakeFredProvider, FakeMarketDataProvider, FakeNewsProvider,
    inverted_curve, rising_prices, steep_curve,
)


def test_market_fake_records_calls_and_returns_prices():
    p = FakeMarketDataProvider()
    df = p.get_prices("AAPL", date(2024, 1, 1), date(2024, 5, 1))
    assert "Close" in df.columns and len(df) == 260
    assert p.calls[-1] == ("prices", "AAPL", date(2024, 1, 1), date(2024, 5, 1))


def test_market_fake_fundamentals_default_none():
    assert FakeMarketDataProvider().get_fundamentals_asof("AAPL", date(2024, 5, 1)) is None


def test_fred_fake_records_and_returns_series():
    f = FakeFredProvider(steep_curve())
    s = f.get_series("T10Y2Y", date(2024, 4, 1), date(2024, 5, 1))
    assert isinstance(s, pd.Series) and float(s.iloc[-1]) > 0.5
    assert f.calls[-1] == ("T10Y2Y", date(2024, 4, 1), date(2024, 5, 1))


def test_news_fake_returns_and_can_raise():
    assert FakeNewsProvider(["a"]).get_headlines("AAPL", date(2024, 5, 1)) == ["a"]
    raiser = FakeNewsProvider(raises=True)
    try:
        raiser.get_headlines("AAPL", date(2024, 5, 1))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_curve_fixtures():
    assert float(inverted_curve().iloc[-1]) < 0
    assert float(steep_curve().iloc[-1]) > 0.5
```

- [ ] **Step 2: Run test to verify it fails** — `python -m pytest apps/quant/advisor/tests/test_fakes.py -q` → FAIL (module `advisor.tests.fakes` not found).

- [ ] **Step 3: Write `apps/quant/advisor/tests/fakes.py`**

```python
from __future__ import annotations

from datetime import date

import pandas as pd

from advisor.data.provider import Fundamentals


def rising_prices(n: int = 260, start: float = 50.0, step: float = 0.5) -> pd.DataFrame:
    closes = [start + step * i for i in range(n)]
    idx = pd.date_range(end="2024-05-01", periods=n, freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


def steep_curve() -> pd.Series:
    idx = pd.date_range(end="2024-05-01", periods=5, freq="D")
    return pd.Series([0.8, 0.9, 1.0, 1.1, 1.2], index=idx)  # latest > 0.5 -> bullish


def inverted_curve() -> pd.Series:
    idx = pd.date_range(end="2024-05-01", periods=5, freq="D")
    return pd.Series([-0.1, -0.2, -0.3, -0.4, -0.5], index=idx)  # latest < 0 -> bearish


class FakeMarketDataProvider:
    def __init__(self, prices: pd.DataFrame | None = None, fundamentals: Fundamentals | None = None) -> None:
        self._prices = prices if prices is not None else rising_prices()
        self._fundamentals = fundamentals
        self.calls: list[tuple] = []

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append(("prices", ticker, start, end))
        return self._prices

    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None:
        self.calls.append(("fundamentals", ticker, as_of))
        return self._fundamentals


class FakeFredProvider:
    def __init__(self, series: pd.Series | None = None) -> None:
        self._series = series if series is not None else pd.Series(dtype="float64")
        self.calls: list[tuple] = []

    def get_series(self, series_id: str, start: date, end: date) -> pd.Series:
        self.calls.append((series_id, start, end))
        return self._series


class FakeNewsProvider:
    def __init__(self, headlines: list[str] | None = None, raises: bool = False) -> None:
        self._headlines = list(headlines or [])
        self._raises = raises
        self.calls: list[tuple] = []

    def get_headlines(self, ticker: str, as_of: date) -> list[str]:
        self.calls.append((ticker, as_of))
        if self._raises:
            raise RuntimeError("boom")
        return list(self._headlines)
```

- [ ] **Step 4: Run test to verify it passes** — `python -m pytest apps/quant/advisor/tests/test_fakes.py -q` → PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/tests/fakes.py apps/quant/advisor/tests/test_fakes.py
git commit -m "test(advisor): shared recording fakes for Workstream C providers"
```

---

### Task 2: FRED macro adapter

**Files:**
- Create: `apps/quant/advisor/data/fred_provider.py`
- Test: `apps/quant/advisor/tests/test_fred_provider.py`

Real adapter reads `FRED_API_KEY` from env; injectable `http_get` makes parsing + the `observation_end=as_of` bound unit-testable without network. Keyless ⇒ empty Series ⇒ (in the coro) neutral.

- [ ] **Step 1: Write the failing test** — `apps/quant/advisor/tests/test_fred_provider.py`

```python
from datetime import date

import pandas as pd

from advisor.data.fred_provider import FRED_SERIES_T10Y2Y, FredApiProvider


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_keyless_provider_returns_empty_series():
    s = FredApiProvider(api_key="").get_series(FRED_SERIES_T10Y2Y, date(2024, 1, 1), date(2024, 5, 1))
    assert isinstance(s, pd.Series) and s.empty


def test_parses_observations_and_bounds_end_at_as_of():
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = params
        return _Resp({"observations": [
            {"date": "2024-04-30", "value": "1.20"},
            {"date": "2024-05-01", "value": "."},  # FRED missing marker -> NaN
        ]})

    s = FredApiProvider(api_key="k", http_get=fake_get).get_series(
        FRED_SERIES_T10Y2Y, date(2024, 4, 1), date(2024, 5, 1))
    assert len(s) == 2
    assert float(s.iloc[0]) == 1.20
    assert pd.isna(s.iloc[1])
    assert calls["params"]["observation_end"] == "2024-05-01"  # as-of upper bound


def test_http_error_degrades_to_empty():
    def boom(url, params=None, timeout=None):
        raise RuntimeError("network down")

    s = FredApiProvider(api_key="k", http_get=boom).get_series(
        FRED_SERIES_T10Y2Y, date(2024, 4, 1), date(2024, 5, 1))
    assert s.empty
```

- [ ] **Step 2: Run test to verify it fails** — `python -m pytest apps/quant/advisor/tests/test_fred_provider.py -q` → FAIL (module not found).

- [ ] **Step 3: Write `apps/quant/advisor/data/fred_provider.py`**

```python
from __future__ import annotations

import os
from datetime import date
from typing import Callable, Protocol

import pandas as pd

FRED_SERIES_T10Y2Y = "T10Y2Y"
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredProvider(Protocol):
    def get_series(self, series_id: str, start: date, end: date) -> pd.Series: ...


class FredApiProvider:
    """Thin FRED adapter. Network-bound; unit-tested via an injected ``http_get``.

    Reads FRED_API_KEY from the environment. Returns an empty Series when the key is
    missing or the request fails, so the macro coro degrades to neutral (spec section 10).
    """

    def __init__(self, api_key: str | None = None, http_get: Callable | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")
        self._http_get = http_get

    def get_series(self, series_id: str, start: date, end: date) -> pd.Series:
        if not self._api_key:
            return pd.Series(dtype="float64")
        getter = self._http_get
        if getter is None:  # pragma: no cover - network path
            import requests
            getter = requests.get
        try:
            resp = getter(_FRED_URL, params={
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),  # as-of upper bound
            }, timeout=15)
            observations = resp.json().get("observations", [])
        except Exception:
            return pd.Series(dtype="float64")
        idx: list = []
        vals: list[float] = []
        for o in observations:
            try:
                vals.append(float(o.get("value")))
            except (TypeError, ValueError):
                vals.append(float("nan"))  # FRED uses "." for missing
            idx.append(pd.Timestamp(o.get("date")))
        return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype="float64")
```

- [ ] **Step 4: Run test to verify it passes** — `python -m pytest apps/quant/advisor/tests/test_fred_provider.py -q` → PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/data/fred_provider.py apps/quant/advisor/tests/test_fred_provider.py
git commit -m "feat(advisor): FRED T10Y2Y adapter (keyless+error -> empty, as-of bounded)"
```

---

### Task 3: Deterministic lexicon news scorer

**Files:**
- Create: `apps/quant/advisor/analysis/news_scorer.py`
- Test: `apps/quant/advisor/tests/test_news_scorer.py`

A `Callable[[str], float]` → `[-1, 1]` from a small audited word list. No model, no network, no look-ahead.

- [ ] **Step 1: Write the failing test** — `apps/quant/advisor/tests/test_news_scorer.py`

```python
from advisor.analysis.news_scorer import lexicon_score


def test_positive_headline_is_positive():
    assert lexicon_score("Earnings beat estimates, record profit") > 0


def test_negative_headline_is_negative():
    assert lexicon_score("Guidance cut, probe opened, lawsuit filed") < 0


def test_neutral_headline_is_zero():
    assert lexicon_score("Company schedules annual shareholder meeting") == 0.0


def test_score_is_bounded():
    assert -1.0 <= lexicon_score("beat beat beat surge upgrade") <= 1.0
    assert -1.0 <= lexicon_score("miss cut probe fraud recall") <= 1.0
```

- [ ] **Step 2: Run test to verify it fails** — `python -m pytest apps/quant/advisor/tests/test_news_scorer.py -q` → FAIL (module not found).

- [ ] **Step 3: Write `apps/quant/advisor/analysis/news_scorer.py`**

```python
from __future__ import annotations

# Deterministic, auditable sentiment lexicon. Intentionally NOT a model: an LLM scorer
# would reintroduce pretraining look-ahead, which the floor exists to exclude.
POSITIVE = {
    "beat", "beats", "surge", "surges", "upgrade", "upgraded", "growth", "record",
    "strong", "raises", "raised", "outperform", "profit", "profits", "gains", "wins",
    "approval", "approved", "soars", "rally", "tops",
}
NEGATIVE = {
    "miss", "misses", "cut", "cuts", "downgrade", "downgraded", "probe", "lawsuit",
    "weak", "loss", "losses", "falls", "plunge", "plunges", "recall", "fraud",
    "warning", "warns", "halts", "slumps", "bankruptcy",
}


def lexicon_score(headline: str) -> float:
    """Bag-of-words sentiment in [-1, 1]. 0.0 when no lexicon word is present."""
    tokens = [t.strip(".,!?:;\"'()[]").lower() for t in headline.split()]
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)
```

- [ ] **Step 4: Run test to verify it passes** — `python -m pytest apps/quant/advisor/tests/test_news_scorer.py -q` → PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/analysis/news_scorer.py apps/quant/advisor/tests/test_news_scorer.py
git commit -m "feat(advisor): deterministic lexicon news scorer (auditable, no look-ahead)"
```

---

### Task 4: News provider — Protocol + Composite + Alpha Vantage adapter

**Files:**
- Create: `apps/quant/advisor/data/news_provider.py`
- Test: `apps/quant/advisor/tests/test_news_provider.py`

`CompositeNewsProvider` unions + dedupes across sources (redundancy; all-empty → `[]`; one source raising is swallowed). `AlphaVantageNewsProvider` reads `ALPHAVANTAGE_API_KEY`, caps `time_to=as_of`, and treats throttle/`Note`/error as unavailable → `[]`.

- [ ] **Step 1: Write the failing test** — `apps/quant/advisor/tests/test_news_provider.py`

```python
from datetime import date

from advisor.data.news_provider import AlphaVantageNewsProvider, CompositeNewsProvider
from advisor.tests.fakes import FakeNewsProvider


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_composite_unions_and_dedupes():
    a = FakeNewsProvider(["Earnings beat", "Record profit"])
    b = FakeNewsProvider(["earnings  beat", "New product launch"])  # 1st dup (case/space)
    out = CompositeNewsProvider([a, b]).get_headlines("AAPL", date(2024, 5, 1))
    assert out == ["Earnings beat", "Record profit", "New product launch"]


def test_composite_all_empty_is_empty():
    assert CompositeNewsProvider([FakeNewsProvider([]), FakeNewsProvider([])]).get_headlines(
        "AAPL", date(2024, 5, 1)) == []


def test_composite_swallows_a_raising_source():
    good = FakeNewsProvider(["Strong guidance"])
    out = CompositeNewsProvider([FakeNewsProvider(raises=True), good]).get_headlines(
        "AAPL", date(2024, 5, 1))
    assert out == ["Strong guidance"]


def test_av_keyless_returns_empty():
    assert AlphaVantageNewsProvider(api_key="").get_headlines("AAPL", date(2024, 5, 1)) == []


def test_av_bounds_time_to_at_as_of_and_parses_titles():
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = params
        return _Resp({"feed": [{"title": "Earnings beat"}, {"title": "Record profit"}, {"x": 1}]})

    out = AlphaVantageNewsProvider(api_key="k", http_get=fake_get).get_headlines(
        "AAPL", date(2024, 5, 1))
    assert out == ["Earnings beat", "Record profit"]
    assert calls["params"]["time_to"].startswith("20240501T")  # as-of upper bound
    assert calls["params"]["tickers"] == "AAPL"


def test_av_throttle_note_returns_empty():
    def throttled(url, params=None, timeout=None):
        return _Resp({"Note": "rate limit"})

    assert AlphaVantageNewsProvider(api_key="k", http_get=throttled).get_headlines(
        "AAPL", date(2024, 5, 1)) == []
```

- [ ] **Step 2: Run test to verify it fails** — `python -m pytest apps/quant/advisor/tests/test_news_provider.py -q` → FAIL (module not found).

- [ ] **Step 3: Write `apps/quant/advisor/data/news_provider.py`**

```python
from __future__ import annotations

import os
from datetime import date
from typing import Callable, Protocol

_AV_URL = "https://www.alphavantage.co/query"


class NewsProvider(Protocol):
    def get_headlines(self, ticker: str, as_of: date) -> list[str]: ...


class CompositeNewsProvider:
    """Union + dedupe headlines across providers for redundancy.

    A keyless/down/raising source contributes nothing; only an all-empty result yields []
    (which the sentiment coro maps to neutral). Redundancy buys availability, NOT signal
    validity - the floor still blocks.
    """

    def __init__(self, providers: list[NewsProvider]) -> None:
        self._providers = list(providers)

    def get_headlines(self, ticker: str, as_of: date) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self._providers:
            try:
                headlines = p.get_headlines(ticker, as_of)
            except Exception:
                headlines = []
            for h in headlines or []:
                key = " ".join(str(h).lower().split())
                if key and key not in seen:
                    seen.add(key)
                    out.append(h)
        return out


class AlphaVantageNewsProvider:
    """Alpha Vantage NEWS_SENTIMENT adapter. Network-bound; unit-tested via injected http_get.

    Reads ALPHAVANTAGE_API_KEY from env. Caps ``time_to`` at as_of. Treats a missing key,
    a throttle response ({"Note"/"Information": ...}), or any error as unavailable -> [].
    """

    def __init__(self, api_key: str | None = None, http_get: Callable | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self._http_get = http_get

    def get_headlines(self, ticker: str, as_of: date) -> list[str]:
        if not self._api_key:
            return []
        getter = self._http_get
        if getter is None:  # pragma: no cover - network path
            import requests
            getter = requests.get
        try:
            resp = getter(_AV_URL, params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_to": as_of.strftime("%Y%m%dT2359"),  # as-of upper bound
                "limit": 50,
                "apikey": self._api_key,
            }, timeout=15)
            data = resp.json()
        except Exception:
            return []
        if not isinstance(data, dict) or "feed" not in data:  # throttle / error
            return []
        return [a["title"] for a in data.get("feed", []) if isinstance(a, dict) and a.get("title")]
```

- [ ] **Step 4: Run test to verify it passes** — `python -m pytest apps/quant/advisor/tests/test_news_provider.py -q` → PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/data/news_provider.py apps/quant/advisor/tests/test_news_provider.py
git commit -m "feat(advisor): composite+AlphaVantage news adapter (dedupe, as-of bound, throttle->[])"
```

---

### Task 5: Family-coro factories

**Files:**
- Create: `apps/quant/advisor/pipeline/families.py`
- Test: `apps/quant/advisor/tests/test_families.py`

Each factory returns `async (as_of) -> FamilySignal`, fetches inputs capped at `≤ as_of`, calls the matching `evaluate`, and falls back to `FamilySignal.neutral(...)` on any missing input/exception.

- [ ] **Step 1: Write the failing test** — `apps/quant/advisor/tests/test_families.py`

```python
import asyncio
from datetime import date

from advisor.analysis.news_scorer import lexicon_score
from advisor.pipeline.families import (
    make_macro_coro, make_momentum_coro, make_sentiment_coro,
    make_trend_coro, make_value_quality_coro,
)
from advisor.schemas import Direction
from advisor.tests.fakes import (
    FakeFredProvider, FakeMarketDataProvider, FakeNewsProvider,
    inverted_curve, rising_prices, steep_curve,
)

AS_OF = date(2024, 5, 1)


def _run(coro):
    return asyncio.run(coro(AS_OF))


def test_trend_coro_bullish_and_bounds_end_at_as_of():
    p = FakeMarketDataProvider(prices=rising_prices())
    sig = _run(make_trend_coro(p, "AAPL"))
    assert sig.direction is Direction.BULLISH
    assert p.calls[-1][0] == "prices" and p.calls[-1][3] == AS_OF  # end == as_of


def test_momentum_coro_bullish():
    sig = _run(make_momentum_coro(FakeMarketDataProvider(prices=rising_prices()), "AAPL"))
    assert sig.direction is Direction.BULLISH


def test_value_quality_coro_none_fundamentals_is_neutral():
    sig = _run(make_value_quality_coro(FakeMarketDataProvider(fundamentals=None), "AAPL"))
    assert sig.direction is Direction.NEUTRAL


def test_macro_coro_bounds_end_at_as_of_and_reads_curve():
    fred = FakeFredProvider(inverted_curve())
    sig = _run(make_macro_coro(fred))
    assert sig.direction is Direction.BEARISH
    assert fred.calls[-1][0] == "T10Y2Y" and fred.calls[-1][2] == AS_OF  # end == as_of


def test_macro_coro_empty_series_is_neutral():
    assert _run(make_macro_coro(FakeFredProvider())).direction is Direction.NEUTRAL


def test_steep_curve_macro_is_bullish():
    assert _run(make_macro_coro(FakeFredProvider(steep_curve()))).direction is Direction.BULLISH


def test_sentiment_coro_empty_is_neutral_not_fabricated():
    news = FakeNewsProvider([])
    sig = _run(make_sentiment_coro(news, lexicon_score, "AAPL"))
    assert sig.direction is Direction.NEUTRAL
    assert news.calls[-1] == ("AAPL", AS_OF)  # as_of passed through for bounding


def test_sentiment_coro_positive_news_is_bullish():
    news = FakeNewsProvider(["Earnings beat estimates", "Record profit"])
    assert _run(make_sentiment_coro(news, lexicon_score, "AAPL")).direction is Direction.BULLISH


def test_coro_swallows_exception_into_neutral():
    sig = _run(make_sentiment_coro(FakeNewsProvider(raises=True), lexicon_score, "AAPL"))
    assert sig.direction is Direction.NEUTRAL
```

- [ ] **Step 2: Run test to verify it fails** — `python -m pytest apps/quant/advisor/tests/test_families.py -q` → FAIL (module not found).

- [ ] **Step 3: Write `apps/quant/advisor/pipeline/families.py`**

```python
from __future__ import annotations

from datetime import date, timedelta
from typing import Awaitable, Callable

import pandas as pd

from advisor.analysis import macro, momentum, sentiment, trend, value_quality
from advisor.data.fred_provider import FRED_SERIES_T10Y2Y, FredProvider
from advisor.data.news_provider import NewsProvider
from advisor.data.provider import MarketDataProvider
from advisor.schemas import FamilySignal

FamilyCoro = Callable[[date], Awaitable[FamilySignal]]

PRICE_LOOKBACK_DAYS = 420   # > trend MIN_HISTORY (201 trading days)
MACRO_LOOKBACK_DAYS = 30    # daily T10Y2Y; macro needs only 2 points


def close_series(df) -> pd.Series:
    """Extract a 1-D close price Series from a provider price frame."""
    if isinstance(df, pd.Series):
        return df
    col = df["Close"] if "Close" in getattr(df, "columns", []) else df.iloc[:, 0]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col


def make_value_quality_coro(provider: MarketDataProvider, ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            f = provider.get_fundamentals_asof(ticker, as_of)
            if f is None:
                return FamilySignal.neutral(value_quality.FAMILY, as_of, "no fundamentals available")
            return value_quality.evaluate(f, as_of)
        except Exception as e:
            return FamilySignal.neutral(value_quality.FAMILY, as_of, f"value_quality unavailable: {e!s}")
    return coro


def make_trend_coro(provider: MarketDataProvider, ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            df = provider.get_prices(ticker, as_of - timedelta(days=PRICE_LOOKBACK_DAYS), as_of)
            return trend.evaluate(close_series(df), as_of)
        except Exception as e:
            return FamilySignal.neutral(trend.FAMILY, as_of, f"trend unavailable: {e!s}")
    return coro


def make_momentum_coro(provider: MarketDataProvider, ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            df = provider.get_prices(ticker, as_of - timedelta(days=PRICE_LOOKBACK_DAYS), as_of)
            return momentum.evaluate(close_series(df), as_of)
        except Exception as e:
            return FamilySignal.neutral(momentum.FAMILY, as_of, f"momentum unavailable: {e!s}")
    return coro


def make_macro_coro(fred: FredProvider, series_id: str = FRED_SERIES_T10Y2Y) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            s = fred.get_series(series_id, as_of - timedelta(days=MACRO_LOOKBACK_DAYS), as_of)
            return macro.evaluate(s, as_of)
        except Exception as e:
            return FamilySignal.neutral(macro.FAMILY, as_of, f"macro unavailable: {e!s}")
    return coro


def make_sentiment_coro(news: NewsProvider, scorer: Callable[[str], float], ticker: str) -> FamilyCoro:
    async def coro(as_of: date) -> FamilySignal:
        try:
            headlines = news.get_headlines(ticker, as_of)
            return sentiment.evaluate(headlines, as_of, scorer)
        except Exception as e:
            return FamilySignal.neutral(sentiment.FAMILY, as_of, f"sentiment unavailable: {e!s}")
    return coro
```

- [ ] **Step 4: Run test to verify it passes** — `python -m pytest apps/quant/advisor/tests/test_families.py -q` → PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/pipeline/families.py apps/quant/advisor/tests/test_families.py
git commit -m "feat(advisor): async family-coro factories (as-of bounded, missing->neutral)"
```

---

### Task 6: CLI `--families all` command

**Files:**
- Modify: `apps/quant/advisor/cli.py` (final content below — replaces the file)
- Test: `apps/quant/advisor/tests/test_cli_all.py`

Adds a five-family path while preserving the existing single-family default. Risk inputs are CLI args with conservative, documented, illustrative defaults; latest price comes from the provider. `run_all` accepts injected providers/scorer so it is testable without network.

- [ ] **Step 1: Write the failing test** — `apps/quant/advisor/tests/test_cli_all.py`

```python
from datetime import date

from advisor.analysis.news_scorer import lexicon_score
from advisor.cli import run_all
from advisor.personas.overlay import PersonaVerdict
from advisor.tests.fakes import FakeFredProvider, FakeMarketDataProvider, FakeNewsProvider, steep_curve

AS_OF = date(2024, 5, 1)


def _providers():
    return (
        FakeMarketDataProvider(fundamentals=None),  # value/quality -> neutral; prices rise -> trend/mom bullish
        FakeFredProvider(steep_curve()),            # macro bullish
        FakeNewsProvider(["Earnings beat estimates", "Record profit"]),  # sentiment bullish
    )


def test_run_all_produces_nonzero_buy_with_disclosure():
    market, fred, news = _providers()
    out = run_all(market, fred, news, lexicon_score, "AAPL", AS_OF,
                  net_liq=100_000.0, vol=0.20, correlation=0.50)
    assert "AAPL" in out
    assert "buy qty=" in out
    qty = int(out.split("qty=")[1].split(" ")[0])
    assert qty > 0  # ensemble bullish -> allocate exercised, not vacuous
    assert "DISCLOSURES" in out


def test_run_all_explain_appends_persona():
    market, fred, news = _providers()
    critic = lambda d: PersonaVerdict(1.0, f"{d.bundle_direction} per 5-family ensemble")
    out = run_all(market, fred, news, lexicon_score, "AAPL", AS_OF,
                  net_liq=100_000.0, vol=0.20, correlation=0.50, persona_critic=critic)
    assert "persona" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails** — `python -m pytest apps/quant/advisor/tests/test_cli_all.py -q` → FAIL (`run_all` not defined).

- [ ] **Step 3: Replace `apps/quant/advisor/cli.py` with exactly:**

```python
from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

from advisor.analysis.news_scorer import lexicon_score
from advisor.analysis.value_quality import evaluate
from advisor.backtest.walk_forward import disclosure_header
from advisor.data.fred_provider import FredApiProvider, FredProvider
from advisor.data.news_provider import AlphaVantageNewsProvider, CompositeNewsProvider, NewsProvider
from advisor.data.provider import MarketDataProvider, YFinanceProvider
from advisor.pipeline.families import (
    close_series, make_macro_coro, make_momentum_coro, make_sentiment_coro,
    make_trend_coro, make_value_quality_coro,
)
from advisor.pipeline.run import run_pipeline

# Conservative, illustrative defaults for report-only sizing. These do NOT authorize
# real capital; the floor still blocks (spec section 6). Override on the CLI.
DEFAULT_NET_LIQ = 100_000.0
DEFAULT_VOL = 0.30
DEFAULT_CORRELATION = 0.50


def run(provider: MarketDataProvider, ticker: str, as_of: date, critic=None) -> str:
    """Single-family (value/quality) path - unchanged back-compat behavior."""
    f = provider.get_fundamentals_asof(ticker, as_of)
    if f is None:
        return f"{ticker}: no point-in-time fundamentals available as of {as_of}\n{disclosure_header()}"
    sig = evaluate(f, as_of)
    line = f"{ticker} [{sig.direction.value}] confidence={sig.confidence:.0f} :: {sig.reasoning}"
    if critic is not None:
        verdict = critic(sig)
        line += f"\n  persona: {verdict.explanation}"
    return f"{line}\n{disclosure_header()}"


def _latest_price(provider: MarketDataProvider, ticker: str, as_of: date) -> float:
    df = provider.get_prices(ticker, as_of - timedelta(days=10), as_of)
    s = close_series(df).dropna()
    return float(s.iloc[-1]) if len(s) else 0.0


def run_all(provider: MarketDataProvider, fred: FredProvider, news: NewsProvider,
            scorer, ticker: str, as_of: date, net_liq: float, vol: float,
            correlation: float, persona_critic=None) -> str:
    """Five-family report-only path: assembles all coros and calls run_pipeline."""
    coros = [
        make_value_quality_coro(provider, ticker),
        make_trend_coro(provider, ticker),
        make_momentum_coro(provider, ticker),
        make_macro_coro(fred),
        make_sentiment_coro(news, scorer, ticker),
    ]
    price = _latest_price(provider, ticker, as_of)
    decision = asyncio.run(run_pipeline(
        ticker, as_of, price=price, net_liq=net_liq, vol=vol, correlation=correlation,
        family_coros=coros, persona_critic=persona_critic))
    line = (f"{decision.ticker} [{decision.bundle_direction}] {decision.action} "
            f"qty={decision.quantity} :: {decision.reasoning}")
    return f"{line}\n{disclosure_header()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advisor", description="AI advisor (report-only)")
    parser.add_argument("ticker")
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="YYYY-MM-DD point-in-time date")
    parser.add_argument("--families", choices=["value", "all"], default="value",
                        help="'value' (single-family, default) or 'all' (five-family pipeline)")
    parser.add_argument("--net-liq", type=float, default=DEFAULT_NET_LIQ,
                        help="illustrative net liquidation value for sizing (report-only)")
    parser.add_argument("--vol", type=float, default=DEFAULT_VOL,
                        help="illustrative annualized volatility for the position limit")
    parser.add_argument("--correlation", type=float, default=DEFAULT_CORRELATION,
                        help="illustrative correlation for the position limit")
    parser.add_argument("--explain", action="store_true",
                        help="append a persona explanation line (read-only narration)")
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)

    if args.families == "all":
        provider = YFinanceProvider()
        fred = FredApiProvider()
        news = CompositeNewsProvider([AlphaVantageNewsProvider()])
        critic = None
        if args.explain:
            from advisor.personas.overlay import PersonaVerdict
            critic = lambda d: PersonaVerdict(1.0, f"{d.bundle_direction} per 5-family ensemble")
        print(run_all(provider, fred, news, lexicon_score, args.ticker, as_of,
                      net_liq=args.net_liq, vol=args.vol, correlation=args.correlation,
                      persona_critic=critic))
        return 0

    critic = None
    if args.explain:
        from advisor.personas.overlay import PersonaVerdict
        critic = lambda sig: PersonaVerdict(1.0, f"{sig.direction.value} per value/quality family")
    print(run(YFinanceProvider(), args.ticker, as_of, critic=critic))
    return 0
```

- [ ] **Step 4: Run tests to verify they pass** — `python -m pytest apps/quant/advisor/tests/test_cli_all.py -q` → PASS (2 passed). Then run the full suite: `python -m pytest apps/quant/advisor/tests -q` → all green, count risen.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/cli.py apps/quant/advisor/tests/test_cli_all.py
git commit -m "feat(advisor): --families all CLI assembles 5-family run_pipeline (report-only)"
```

---

### Task 7: Operator news-coverage probe (non-gate)

**Files:**
- Create: `scripts/news_coverage_probe.py`

Operator-run, **not** in the pytest gate (network). For a ticker list + as_of, prints per-source headline counts so the operator can compare sources empirically by what actually returns data (coverage), and confirms each adapter honors the `≤ as_of` bound. Reads keys from env only.

- [ ] **Step 1: Write `scripts/news_coverage_probe.py`**

```python
"""Operator-run coverage probe (NOT a unit test - hits the network).

Usage:
    $env:ALPHAVANTAGE_API_KEY="..."   # never commit keys
    python scripts/news_coverage_probe.py AAPL MSFT NVDA --as-of 2024-05-01

Prints, per ticker, how many headlines each configured source returns. Empty counts
(missing key / throttle / no coverage) degrade to a neutral sentiment signal - they are
NOT failures. This is an availability comparison, not a signal-quality claim: news
sentiment remains report-only and cannot close the floor (spec section 6/10).
"""
from __future__ import annotations

import argparse
import os
from datetime import date

from advisor.data.news_provider import AlphaVantageNewsProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="news_coverage_probe")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--as-of", default=date.today().isoformat())
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)

    sources = {"alpha_vantage": AlphaVantageNewsProvider()}
    have_keys = {"alpha_vantage": bool(os.environ.get("ALPHAVANTAGE_API_KEY"))}
    print(f"as_of={as_of} (time_to capped at as_of) | keys: {have_keys}")
    for ticker in args.tickers:
        for name, src in sources.items():
            heads = src.get_headlines(ticker, as_of)
            sample = heads[0] if heads else "-"
            print(f"  {ticker:6s} {name:14s} {len(heads):3d} headlines | e.g. {sample!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it imports (no network call needed; keyless → 0 headlines)** — `python scripts/news_coverage_probe.py AAPL --as-of 2024-05-01` → prints `0 headlines` lines without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/news_coverage_probe.py
git commit -m "chore(advisor): operator news-coverage probe (non-gate, env keys only)"
```

---

### Task 8: Docs + roadmap update

**Files:**
- Modify: `docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md`

Mark Workstream C (Plan 2) done for the unit path; record the deferred live smoke; note 1b/3 are unblocked-by-prerequisite but still gated on a real candidate.

- [ ] **Step 1: Replace the `Plan 2 ...` paragraph (lines 13-16) with:**

```markdown
Plan 2 — Workstream C completion. DONE (unit/fake path, report-only). The five-family
run_pipeline is wired into the CLI via `--families all` with FRED + composite/Alpha
Vantage adapters, a deterministic lexicon news scorer, and async family-coro factories;
every adapter is as-of bounded and degrades to FamilySignal.neutral on missing
input/error (no fabrication, spec section 10). Floor UNTOUCHED: advisor-gate exit 0,
`run-floor --enforce` exit 1. DEFERRED: the live multi-source smoke (real FRED_API_KEY +
ALPHAVANTAGE_API_KEY) and adding Finnhub/NewsAPI to the composite - both are operator
steps, not in the pytest gate. See docs/superpowers/plans/2026-06-16-workstream-c.md.
```

- [ ] **Step 2: Append after the Plan 3 paragraph:**

```markdown
Status note (post-C): Plan 1b (wire validation["passes"] into --enforce) and Plan 3
(post-C signal program) are now unblocked-by-prerequisite (C provides the live
five-family path) but remain gated on a real candidate that clears dev. Do not start
either until such a candidate exists; the accepted DEV_FAILED floor stays undisturbed.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md
git commit -m "docs(roadmap): mark Workstream C done (unit path); note 1b/3 prerequisite status"
```

---

## Definition of done

- All 8 tasks committed (one commit each) and pushed to `origin/main`.
- New FRED + composite/Alpha Vantage news adapters, lexicon scorer, and five family-coro factories exist, each unit-tested with **fakes** (no network in the gate); missing-source ⇒ neutral and as-of bounding are asserted.
- `advisor --families all <TICKER>` assembles all five families through `run_pipeline` and prints a `Decision` + `disclosure_header()`; `--explain` works; single-family default unchanged.
- `python -m pytest apps/quant/advisor/tests` green, **test count risen from 131** (≈ +29: 5+3+4+6+9+2). `npm run advisor-gate` **exit 0**; `node tools/run-floor.mjs --enforce` **exit 1** (UNCHANGED).
- `allocator.py` / `ensemble_vote` / `risk` / `backtest` untouched; no secrets committed; report-only caveat carried in CLI output (disclosure) and the roadmap.

## Deferred (operator, not in this build)

- Live multi-source smoke with real `FRED_API_KEY` + `ALPHAVANTAGE_API_KEY` (Task-5 of the handoff).
- Adding Finnhub/NewsAPI concrete adapters to the composite (1-class each).

## Self-review notes

- **Spec coverage:** handoff §6 tasks 1-6 all mapped (FRED→T2, news→T4, scorer→T3 [new, per advisor: no production scorer existed], coros→T5, CLI→T6, probe→T7 [user-requested], docs→T8; live smoke deferred per operator). Fakes module (T1) added for DRY/clobber-safety.
- **Type consistency:** `close_series` defined in T5, imported by T6. `FamilyCoro`, `FredProvider`, `NewsProvider`, `MarketDataProvider` Protocols consistent across tasks. `run_pipeline`/`allocate`/`position_limit`/`apply_overlay` signatures match the verified source. `FamilySignal.neutral(family, as_of, reasoning)` and `.FAMILY` module constants confirmed against source.
- **No placeholders:** every step has full code/commands.
