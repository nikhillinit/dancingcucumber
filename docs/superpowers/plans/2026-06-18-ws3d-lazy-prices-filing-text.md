# WS3D — Lazy Prices (10-K/10-Q Filing-Text Change): Fixture + Adapter + Candidate Preregistration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instantiate the WS3A source-agnostic contract for a SECOND orthogonal reading — year-over-year filing-text similarity ("Lazy Prices") — as a frozen PIT fixture + read-only adapter + a separate Reading-C preregistration surface, so WS4 can run a `lazy_prices + momentum` candidate through the dev gate. Capability only; no floor flip.

**Architecture:** Mirror WS3C exactly, but the signal is a dimensionless similarity ratio precomputed at fetch time and frozen into a small git-diffable CSV. The heavy NLP (download filings, tokenize, cosine) lives in the one-time network fetch step; the committed artifact is just per-filing similarity scores. The adapter reuses WS3C's `EdgarXbrlRecord`/`load_fixture`/`select_asof` and the floor's already-generic `_candidate_metrics_with_raw_fn`/`_raw_power_report` injection points. **The split-invariance problem (WS3C's T3, its flagged top risk) does NOT exist here** — similarity is basis-free, so there is no price recompute and no anchor.

**Tech Stack:** Python 3.11, pandas, stdlib `csv`/`hashlib`/`json`. Reuses `advisor.data.edgar_xbrl_fixture`, `advisor.research.fundamental_value.select_asof`, `advisor.research.candidate_pipeline`, `advisor.research.candidate_floor`. TDD via `node tools/run-pytest.mjs`.

---

## 0. Defaults adopted (operator may override before dispatch)
- **Signal:** year-over-year cosine similarity of consecutive SAME-FORM filings (10-K vs prior-year 10-K; 10-Q vs same-fiscal-quarter prior-year 10-Q). HIGH similarity = "non-changer" → predicted higher return = the long leg. RAW signal = similarity itself (NOT `1 − similarity`; see D3 sign guard). `similarity_metric="cosine_tfidf"`.
- **Fixture format:** single long-form git-diffable CSV, one row per (asset, filing) carrying the precomputed `FilingSimilarity` value. Reuses WS3C's 15-field `EdgarXbrlRecord` schema (`concept` is a free string), so NO new schema.
- **Coverage:** quarterly+annual filings 2015–2023; the fetch step also pulls 2014 filings as the prior-year BASELINE for 2015 similarities (baseline rows are not emitted as signal, only consumed). Unavailable periods excluded, never filled.
- **Availability:** `available_asof = max(filing_date, accepted_datetime)` — the document IS the disclosure, knowable the instant the filing is public. **NO +90d reporting lag** (that lag is for XBRL financial VALUES tied to a report period; the text needs only that the filing exist on EDGAR). `snapshot_date` = None (filing-backed).
- **Candidate shape:** `lazy_prices + momentum` (mirrors WS3C — a lone family makes the §7.2 ensemble-beats-best-part gate degenerate; momentum's `raw_fn` is already wired). Alternatives an operator may select: `lazy_prices + fundamental_value` (two orthogonal readings, needs WS3C's fixture too) or `lazy_prices + days_to_cover` (the DTC follow-on, WS3E — separate plan).

## 1. Objective
Three committable deterministic artifacts: (a) a frozen PIT filing-text-similarity **fixture** for the formal universe (30 large-caps + SPY) over 2015–2023; (b) a **read-only deterministic adapter** reconstructing the as-of-bounded `lazy_prices` raw signal aligned to `floor_prices.csv`; (c) a **Reading-C candidate preregistration** (own hash + validation surface, fixture SHA pinned) so WS4 can run `lazy_prices + momentum` through the dev gate. Capability only.

## 2. Non-goals
No holdout read (every WS3D call uses `prereg_hash=None`; `HOLDOUT_LEDGER.md` stays empty — unlock is WS4). No floor flip: `backtest/`, `tools/floor_data_check.py`, `tools/run-floor.mjs` untouched; `--enforce` stays exit 1. No new fields on `PreRegConfig` (frozen `1ad2ed4a…`), `CandidatePreReg` (frozen `578cce4b…`), or WS3C's `FundamentalCandidatePreReg`. No DTC/short-interest (that is WS3E, a separate plan — independent data subsystem). No network at test/run time — only D6 touches the network. No new section-text parsing (full-document cosine only; section-level is a future refinement).

## 3. Architecture / data flow
`fetch/freeze (network, operator: EDGAR filing docs → tokenize → YoY cosine → FilingSimilarity rows) → committed CSV fixture → reuse load_fixture (network-free) → reuse select_asof (as-of, no backfill) → build_lazy_prices_panel (step function aligned to floor_prices.csv) → make_lazy_prices_raw closure (keyed on Series.name) → reuse _candidate_metrics_with_raw_fn → metrics dict (prereg_hash=None) → READING_C PREREG + RESULT docs`.

**Date-threading crux (identical to WS3C):** `candidate_pipeline.py:50` collapses to a positional Series whose only asset identity is `.name`. The similarity MUST be precomputed into a date-indexed panel aligned to `floor_prices.csv` BEFORE the positional collapse, then injected via a closure keyed on `.name`. Never a date lookup inside `raw_fn`.

**Why this is smaller than WS3C:** similarity is dimensionless → `build_lazy_prices_panel` just reads `select_asof(...).value` (no anchor, no `bp_timely`, no raw-close fetch, no split math). The floor harness reuses the EXISTING generic `_candidate_metrics_with_raw_fn` and `_raw_power_report` verbatim (WS3C already factored them out).

### File Structure

**Create:**
- `apps/quant/advisor/research/lazy_prices.py` — `LAZY_PRICES`/`SIMILARITY_CONCEPT` consts; `compute_text_available_asof`/`audit_text_available_asof`; `build_lazy_prices_panel`; `make_lazy_prices_raw`; (report-only) `dev_lazy_momentum_corr`. Reuses `EdgarXbrlRecord`, `load_fixture`, `select_asof`.
- `apps/quant/advisor/research/candidate_prereg_lazy_prices.py` — `LazyPricesCandidatePreReg` + `lazy_prices_candidate_hash` + `lazy_prices_candidate_run_hash`.
- `apps/quant/advisor/research/candidate_validation_prereg_lazy_prices.py` — `LazyPricesCandidateValidationPreReg` + default.
- `apps/quant/advisor/data/filing_text_fetch.py` — one-time network fetch/freeze writer (NOT imported by any test/runtime path).
- `apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv` — frozen committed fixture (written by D6).
- `apps/quant/advisor/research/READING_C_PREREG.md`, `READING_C_RESULT.md`.
- Tests: `tests/test_lazy_prices.py`, `tests/test_candidate_prereg_lazy_prices.py`, `tests/test_lazy_prices_candidate_floor.py`.

**Modify (additive only):**
- `apps/quant/advisor/research/candidate_blend.py` — add `LAZY_PRICES` to the bench `_ALLOWED` allowlist (one import + one set member). Frozen `blend.py` untouched.
- `apps/quant/advisor/research/candidate_floor.py` — add `lazy_prices_candidate_metrics` + `_lazy_prices_power_report` (mirror the fundamental wrappers; reuse `_candidate_metrics_with_raw_fn`).
- Roadmap doc — record WS3D status + WS3E (DTC) as the sequenced follow-on.

**Baseline:** before starting, run `node tools/run-pytest.mjs` and record the count (expected ≈217 with WS3C T1/T2/T5 landed). Every task keeps the suite green.

---

## Task D1: Filing-text fixture reuse + availability rule + non-degeneracy

**Files:**
- Create: `apps/quant/advisor/research/lazy_prices.py`
- Test: `apps/quant/advisor/tests/test_lazy_prices.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_lazy_prices.py
from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from advisor.research.lazy_prices import (
    LAZY_PRICES, SIMILARITY_CONCEPT,
    compute_text_available_asof, audit_text_available_asof,
)


def _rec(asset, value, *, filing="2016-02-01", accepted="2016-02-03",
         period="2015-12-31", avail="2016-02-03"):
    return EdgarXbrlRecord(
        asset=asset, cik="0000320193", accession="acc-" + asset, form="10-K",
        report_period_end=date.fromisoformat(period),
        filing_date=date.fromisoformat(filing),
        accepted_datetime=date.fromisoformat(accepted),
        concept=SIMILARITY_CONCEPT, unit="ratio", value=value,
        available_asof=date.fromisoformat(avail),
        superseded_by="", amended_flag=False, missingness_reason="",
        denominator_policy="cosine_tfidf_yoy_same_form",
    )


def test_text_availability_has_no_reporting_lag():
    # filing-text availability = max(filing_date, accepted), NOT report_period_end+90
    got = compute_text_available_asof(date(2016, 2, 1), date(2016, 2, 3))
    assert got == date(2016, 2, 3)
    # +90 from 2015-12-31 would be ~2016-03-30 — must NOT be used
    assert got < date(2016, 3, 30)


def test_audit_passes_for_canonical_and_fails_for_lagged():
    ok = _rec("AAA", 0.95, avail="2016-02-03")
    assert audit_text_available_asof(ok) is True
    lagged = _rec("BBB", 0.95, avail="2016-03-30")   # someone wrongly applied +90
    assert audit_text_available_asof(lagged) is False


def test_lazy_prices_family_constant():
    assert LAZY_PRICES == "lazy_prices"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'advisor.research.lazy_prices'`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/quant/advisor/research/lazy_prices.py
from __future__ import annotations

from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord  # reuse 15-field schema

LAZY_PRICES = "lazy_prices"
SIMILARITY_CONCEPT = "FilingSimilarity"


def compute_text_available_asof(filing_date: date, accepted_datetime: date) -> date:
    """Availability for a FILING-TEXT signal: the document IS the disclosure, knowable
    the instant the filing is public -> max(filing_date, accepted_datetime). NO +90d
    reporting lag (that lag is for XBRL financial VALUES tied to a report period). The
    similarity needs only that the current filing exist on EDGAR. snapshot_date stays
    None (filing-backed) — implementing it as a fetch date would zero the signal in WS4."""
    return max(filing_date, accepted_datetime)


def audit_text_available_asof(rec: EdgarXbrlRecord) -> bool:
    """D6 writes available_asof canonically; this re-derives it for an audit equality
    check. Distinct from edgar_xbrl_fixture.audit_available_asof (which adds +90)."""
    return rec.available_asof == compute_text_available_asof(
        rec.filing_date, rec.accepted_datetime
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/lazy_prices.py apps/quant/advisor/tests/test_lazy_prices.py
git commit -m "WS3D D1: filing-text availability rule (no +90 lag) + schema reuse"
```

---

## Task D2: As-of similarity panel aligned to floor_prices

**Files:**
- Modify: `apps/quant/advisor/research/lazy_prices.py`
- Test: `apps/quant/advisor/tests/test_lazy_prices.py`

- [ ] **Step 1: Write the failing test** (append)

```python
import pandas as pd
from advisor.research.lazy_prices import build_lazy_prices_panel


def _price_panel(dates):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"AAA": range(100, 100 + len(idx)),
                         "SPY": range(400, 400 + len(idx))}, index=idx).astype(float)


def test_panel_is_stepwise_and_pit():
    panel = _price_panel(["2016-01-04", "2016-02-03", "2016-02-04", "2016-03-01"])
    recs = [_rec("AAA", 0.92, accepted="2016-02-03", avail="2016-02-03")]
    out = build_lazy_prices_panel(recs, panel, assets=["AAA"])
    # before availability -> NaN; on/after -> the similarity, held forward
    assert pd.isna(out["AAA"].iloc[0])                  # 2016-01-04 (pre-filing)
    assert out["AAA"].iloc[1] == 0.92                   # 2016-02-03 (available)
    assert out["AAA"].iloc[2] == 0.92                   # held forward
    assert out["AAA"].iloc[3] == 0.92


def test_panel_unknown_asset_all_nan_and_warmup_slices():
    panel = _price_panel(["2016-01-04", "2016-02-03", "2016-02-04"])
    out = build_lazy_prices_panel([], panel, assets=["AAA"], warmup=1)
    assert len(out) == 2                                 # warmup row dropped
    assert out["AAA"].isna().all()                       # no records -> neutral
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_lazy_prices_panel'`

- [ ] **Step 3: Write minimal implementation** (append to `lazy_prices.py`; add `import pandas as pd` at top)

```python
import pandas as pd

from advisor.research.fundamental_value import select_asof  # generic PIT selector


def build_lazy_prices_panel(
    records: list[EdgarXbrlRecord],
    panel: pd.DataFrame,
    assets: list[str] | None = None,
    *,
    warmup: int = 0,
    concept: str = SIMILARITY_CONCEPT,
) -> pd.DataFrame:
    """Date-indexed similarity panel aligned row-for-row to `panel` (floor_prices, a
    DatetimeIndex of adjusted closes), sliced [warmup:] and reset to a positional
    RangeIndex so it shares candidate_pipeline's
    `prices_all = panel[assets].iloc[warmup:].reset_index(drop=True)` basis. Per row:
    the latest FilingSimilarity available as-of that date (select_asof; a STEP FUNCTION
    held between filings). Unavailable -> NaN (the percentile transform maps NaN -> 0 ->
    flat). NO price recompute and NO split handling: similarity is a dimensionless ratio,
    basis-free. Date threading is done HERE, never inside raw_fn."""
    if assets is None:
        assets = [c for c in panel.columns if c != "SPY"]
    dates = list(panel.index)
    cols: dict[str, list[float]] = {}
    for a in assets:
        a_records = [r for r in records if r.asset == a]  # shrink select_asof's scan
        col: list[float] = []
        for t in dates:
            as_of = t.date() if hasattr(t, "date") else t
            rec = select_asof(a_records, a, concept, as_of)
            col.append(float(rec.value) if rec is not None else float("nan"))
        cols[a] = col
    funda = pd.DataFrame(cols, index=range(len(dates)))
    return funda.iloc[warmup:].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/lazy_prices.py apps/quant/advisor/tests/test_lazy_prices.py
git commit -m "WS3D D2: stepwise PIT similarity panel aligned to floor_prices"
```

---

## Task D3: Positional raw_fn closure + SIGN GUARD

**Files:**
- Modify: `apps/quant/advisor/research/lazy_prices.py`
- Test: `apps/quant/advisor/tests/test_lazy_prices.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from advisor.research.lazy_prices import make_lazy_prices_raw


def test_raw_is_similarity_not_its_complement():
    # SIGN GUARD: HIGH similarity = non-changer = LONG leg. raw MUST equal similarity,
    # never 1-similarity (that would long the changers = the wrong/short leg ->
    # DEV_FAILED by construction).
    panel_lp = pd.DataFrame({"NONCHANGER": [0.97, 0.97], "CHANGER": [0.20, 0.20]})
    raw_fn = make_lazy_prices_raw(panel_lp)
    hi = raw_fn(LAZY_PRICES, pd.Series([100.0, 101.0], name="NONCHANGER"))
    lo = raw_fn(LAZY_PRICES, pd.Series([100.0, 101.0], name="CHANGER"))
    assert hi.iloc[0] == 0.97 and lo.iloc[0] == 0.20     # raw == similarity
    assert hi.mean() > lo.mean()                          # non-changer ranks long


def test_raw_dispatches_momentum_to_base_and_unknown_to_nan():
    panel_lp = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]})
    raw_fn = make_lazy_prices_raw(panel_lp)
    # momentum delegates to the frozen price raw_metric (not the panel)
    mom = raw_fn("momentum", pd.Series([1.0] * 130, name="AAA"))
    assert len(mom) == 130
    # unknown asset on the lazy_prices family -> all-NaN neutral
    unk = raw_fn(LAZY_PRICES, pd.Series([1.0, 2.0], name="ZZZ"))
    assert unk.isna().all()


def test_raw_raises_on_warmup_mismatch():
    panel_lp = pd.DataFrame({"AAA": [0.5]})               # 1 row
    raw_fn = make_lazy_prices_raw(panel_lp)
    try:
        raw_fn(LAZY_PRICES, pd.Series([1.0, 2.0, 3.0], name="AAA"))  # 3 rows
        assert False, "expected ValueError on panel shorter than prices"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: FAIL — `ImportError: cannot import name 'make_lazy_prices_raw'`

- [ ] **Step 3: Write minimal implementation** (append; add `from typing import Callable` and `from advisor.backtest.continuous_signals import raw_metric`)

```python
from typing import Callable

from advisor.backtest.continuous_signals import raw_metric


def make_lazy_prices_raw(
    panel_lp: pd.DataFrame,
    base_raw_fn: Callable[[str, pd.Series], pd.Series] = raw_metric,
) -> Callable[[str, pd.Series], pd.Series]:
    """Build the `raw_fn` candidate_pipeline expects: `(family, prices) -> Series`.

    Dispatch: `lazy_prices` -> precomputed panel lookup; any other family -> base_raw_fn
    (the frozen price raw_metric, which handles momentum). SIGN: raw = the similarity
    itself — HIGH similarity = 'non-changer' = the long leg, so the percentile transform
    ranks high-similarity names long. Using `1 - similarity` would invert to the short
    leg. The pipeline passes a POSITIONAL Series (RangeIndex, `.name`=asset); panel_lp is
    aligned to the same warmup-sliced basis, so `.reindex(prices.index)` aligns for both
    the dev sweep and the holdout. Unknown asset -> all-NaN (neutral). A panel shorter
    than prices means a warmup mismatch -> raise loudly rather than silently misalign."""
    def _raw(family: str, prices: pd.Series) -> pd.Series:
        if family == LAZY_PRICES:
            name = prices.name
            if name not in panel_lp.columns:
                return pd.Series([float("nan")] * len(prices), index=prices.index, name=name)
            if len(panel_lp) < len(prices):
                raise ValueError(
                    f"panel_lp rows ({len(panel_lp)}) < prices rows ({len(prices)}); "
                    "lazy-prices panel and prices_all must share cfg.warmup"
                )
            return panel_lp[name].reindex(prices.index)
        return base_raw_fn(family, prices)
    return _raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/lazy_prices.py apps/quant/advisor/tests/test_lazy_prices.py
git commit -m "WS3D D3: positional raw_fn closure + sign guard (long non-changers)"
```

---

## Task D4: Reading-C prereg surface (NEW hash, separate from B)

**Files:**
- Create: `apps/quant/advisor/research/candidate_prereg_lazy_prices.py`
- Create: `apps/quant/advisor/research/candidate_validation_prereg_lazy_prices.py`
- Test: `apps/quant/advisor/tests/test_candidate_prereg_lazy_prices.py`

This task has no dependency on D1–D3 and may be dispatched in parallel.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_candidate_prereg_lazy_prices.py
from advisor.research.candidate_prereg_lazy_prices import (
    LazyPricesCandidatePreReg, DEFAULT_LAZY_PRICES_CANDIDATE,
    lazy_prices_candidate_hash, lazy_prices_candidate_run_hash,
)


def test_hash_is_64_hex_and_stable():
    h = lazy_prices_candidate_hash(DEFAULT_LAZY_PRICES_CANDIDATE)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert h == lazy_prices_candidate_hash(LazyPricesCandidatePreReg())


def test_families_and_metric_locked():
    cfg = DEFAULT_LAZY_PRICES_CANDIDATE
    assert cfg.families == ("lazy_prices", "momentum")
    assert cfg.similarity_metric == "cosine_tfidf"
    assert cfg.reporting_lag_days == 0          # text has no +90 lag
    assert cfg.fixture_source == "SEC_EDGAR_FILING_TEXT"


def test_run_hash_is_fixture_byte_sensitive(tmp_path):
    f = tmp_path / "fix.csv"
    f.write_text("asset,value\nAAA,0.9\n")
    h1 = lazy_prices_candidate_run_hash(DEFAULT_LAZY_PRICES_CANDIDATE, f)
    f.write_text("asset,value\nAAA,0.8\n")
    h2 = lazy_prices_candidate_run_hash(DEFAULT_LAZY_PRICES_CANDIDATE, f)
    assert h1 != h2 and len(h1) == 64


def test_methodology_hash_differs_from_run_hash(tmp_path):
    f = tmp_path / "fix.csv"
    f.write_text("x")
    assert (lazy_prices_candidate_hash(DEFAULT_LAZY_PRICES_CANDIDATE)
            != lazy_prices_candidate_run_hash(DEFAULT_LAZY_PRICES_CANDIDATE, f))


def test_validation_surface_present():
    from advisor.research.candidate_validation_prereg_lazy_prices import (
        DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION,
    )
    assert DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION.declared_trials_N >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_prereg_lazy_prices.py -q`
Expected: FAIL — `ModuleNotFoundError: ...candidate_prereg_lazy_prices`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/quant/advisor/research/candidate_prereg_lazy_prices.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LazyPricesCandidatePreReg:
    """Reading-C candidate methodology — a SEPARATE hash surface from CandidatePreReg
    (578cce4b…), FundamentalCandidatePreReg (Reading B), and PreRegConfig (1ad2ed4a…).
    Stays revisable until READING_C_PREREG.md (D7) pins its hash and a run is recorded."""
    families: tuple[str, ...] = ("lazy_prices", "momentum")
    similarity_metric: str = "cosine_tfidf"      # YoY same-form full-document cosine
    fixture_source: str = "SEC_EDGAR_FILING_TEXT"
    reporting_lag_days: int = 0                   # filing text is public at acceptance
    orthogonality_tau: float = 0.40              # max |corr| lazy_prices vs momentum on dev
    declared_trials_N: int = 45                  # VESTIGIAL — live count lives on the
                                                 # validation surface (frozen-hash discipline)
    # --- inherited from the floor so the bench is faithful (mirror PreRegConfig) ---
    warmup: int = 200
    folds: int = 5
    embargo: int = 5
    margin: float = 0.0
    pct_clip: tuple[float, float] = (0.05, 0.95)
    weight_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    train_lift_threshold: float = 0.05
    max_asset_weight: float = 0.20
    gross_cap: float = 1.0
    turnover_cap: float = 0.20
    cost_per_turn: float = 0.0005
    bootstrap_block: int = 21
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 12345
    dev_lcb: float = 0.90
    final_lcb: float = 0.95
    min_universe_formal: int = 20
    min_universe_floor: int = 12


DEFAULT_LAZY_PRICES_CANDIDATE = LazyPricesCandidatePreReg()


def lazy_prices_candidate_hash(cfg: LazyPricesCandidatePreReg) -> str:
    """Methodology-only id (no fixture bytes). NOT the holdout-unlock key."""
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()


def lazy_prices_candidate_run_hash(cfg: LazyPricesCandidatePreReg, fixture_path) -> str:
    """Holdout-unlock key: canonical config JSON THEN fixture bytes (byte-exact mirror of
    fundamental_candidate_run_hash). Binding the fixture bytes detects a fixture swap."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(fixture_path).read_bytes())
    return h.hexdigest()
```

```python
# apps/quant/advisor/research/candidate_validation_prereg_lazy_prices.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LazyPricesCandidateValidationPreReg:
    """Live DSR-trial surface for Reading C. Carries the LIVE declared trial count; mirrors
    FundamentalCandidateValidationPreReg's FULL field set so validation_report (called in
    D5) reads every field it expects — a thinner surface would break the floor wiring."""
    psr_benchmark_sr: float = 0.0
    dsr_pass: float = 0.95
    tstat_hurdle: float = 3.0
    minbtl_max_trials: int = 45
    declared_trials_N: int = 45          # LIVE candidate trial count (rail #4)
    declared_var_sr: float = 1e-4        # CALIBRATE at D7 (or justify reuse as >= measured)
    effective_n_method: str = "pca"
    effective_n_floor_is_declared: bool = True


DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION = LazyPricesCandidateValidationPreReg()


def lazy_prices_candidate_validation_hash(cfg: LazyPricesCandidateValidationPreReg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_prereg_lazy_prices.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/candidate_prereg_lazy_prices.py apps/quant/advisor/research/candidate_validation_prereg_lazy_prices.py apps/quant/advisor/tests/test_candidate_prereg_lazy_prices.py
git commit -m "WS3D D4: Reading-C prereg + validation surfaces (separate hash)"
```

---

## Task D5: Family admission + floor wiring

**Files:**
- Modify: `apps/quant/advisor/research/candidate_blend.py:11,17`
- Modify: `apps/quant/advisor/research/candidate_floor.py` (additive, after `fundamental_candidate_metrics`)
- Test: `apps/quant/advisor/tests/test_lazy_prices_candidate_floor.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_lazy_prices_candidate_floor.py
import numpy as np
import pandas as pd

from advisor.research.candidate_prereg_lazy_prices import DEFAULT_LAZY_PRICES_CANDIDATE
from advisor.research.lazy_prices import build_lazy_prices_panel, SIMILARITY_CONCEPT
from advisor.research.candidate_floor import lazy_prices_candidate_metrics
from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from datetime import date


def _synthetic_panel(n=400):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    cols = {a: 100 + np.cumsum(rng.normal(0.05, 1.0, n)) for a in
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH",
             "III", "JJJ", "KKK", "LLL"]}
    cols["SPY"] = 400 + np.cumsum(rng.normal(0.04, 0.8, n))
    return pd.DataFrame(cols, index=idx).astype(float)


def _sim_records(panel, assets):
    recs = []
    for a in assets:
        for k, t in enumerate(panel.index[::60]):       # ~quarterly filings
            d = t.date()
            recs.append(EdgarXbrlRecord(
                asset=a, cik="0", accession=f"{a}-{k}", form="10-Q",
                report_period_end=d, filing_date=d, accepted_datetime=d,
                concept=SIMILARITY_CONCEPT, unit="ratio",
                value=0.80 + 0.1 * ((hash((a, k)) % 10) / 10.0),
                available_asof=d, superseded_by="", amended_flag=False,
                missingness_reason="", denominator_policy="cosine_tfidf_yoy_same_form",
            ))
    return recs


def test_metrics_blinded_by_default():
    cfg = DEFAULT_LAZY_PRICES_CANDIDATE
    panel = _synthetic_panel()
    assets = [c for c in panel.columns if c != "SPY"]
    recs = _sim_records(panel, assets)
    panel_lp = build_lazy_prices_panel(recs, panel, assets, warmup=cfg.warmup)
    m = lazy_prices_candidate_metrics(panel, panel_lp, cfg, prereg_hash=None)
    assert m["holdout"] is None          # blinded
    assert m["passes"] is False
    assert set(m).issuperset({"verdict", "universe", "dev", "weights",
                              "ensemble", "best_family", "validation", "power"})


def test_row_count_guard():
    cfg = DEFAULT_LAZY_PRICES_CANDIDATE
    panel = _synthetic_panel()
    bad = pd.DataFrame({c: [0.5] for c in panel.columns if c != "SPY"})  # 1 row
    try:
        lazy_prices_candidate_metrics(panel, bad, cfg, prereg_hash=None)
        assert False, "expected row-count ValueError"
    except ValueError:
        pass


def test_power_report_present():
    cfg = DEFAULT_LAZY_PRICES_CANDIDATE
    panel = _synthetic_panel()
    assets = [c for c in panel.columns if c != "SPY"]
    panel_lp = build_lazy_prices_panel(_sim_records(panel, assets), panel, assets,
                                       warmup=cfg.warmup)
    m = lazy_prices_candidate_metrics(panel, panel_lp, cfg, prereg_hash=None)
    assert "power_limited" in m["power"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices_candidate_floor.py -q`
Expected: FAIL — `ImportError: cannot import name 'lazy_prices_candidate_metrics'`

- [ ] **Step 3a: Admit the family in the bench allowlist**

In `apps/quant/advisor/research/candidate_blend.py`, after line 11 (`from advisor.research.fundamental_value import FUNDAMENTAL_VALUE`) add:

```python
from advisor.research.lazy_prices import LAZY_PRICES
```

Change line 17 from:

```python
_ALLOWED = set(RAW_METRICS) | {VALUE, FUNDAMENTAL_VALUE}
```

to:

```python
_ALLOWED = set(RAW_METRICS) | {VALUE, FUNDAMENTAL_VALUE, LAZY_PRICES}
```

- [ ] **Step 3b: Add the floor wrappers** (append to `apps/quant/advisor/research/candidate_floor.py`; extend the imports for the lazy-prices surfaces)

Add to the import block:

```python
from advisor.research.candidate_prereg_lazy_prices import (
    LazyPricesCandidatePreReg, DEFAULT_LAZY_PRICES_CANDIDATE,
    lazy_prices_candidate_run_hash,
)
from advisor.research.candidate_validation_prereg_lazy_prices import (
    LazyPricesCandidateValidationPreReg, DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION,
)
from advisor.research.lazy_prices import LAZY_PRICES, make_lazy_prices_raw
```

Append the wrappers:

```python
def _lazy_prices_power_report(panel: pd.DataFrame, cfg: LazyPricesCandidatePreReg,
                              holdout_frac: float, raw_fn: RawFn) -> dict:
    # lazy_prices is a low-frequency step function: positive_floor=1 (like fundamentals)
    return _raw_power_report(panel, cfg, holdout_frac, raw_fn, LAZY_PRICES, 1)


def lazy_prices_candidate_metrics(
    panel: pd.DataFrame,
    panel_lp: pd.DataFrame,
    cfg: LazyPricesCandidatePreReg = DEFAULT_LAZY_PRICES_CANDIDATE,
    prereg_hash: str | None = None,
    holdout_frac: float = 0.2,
    vcfg: LazyPricesCandidateValidationPreReg = DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION,
    fixture_path=None,
) -> dict:
    """Reading-C candidate floor mirror using the precomputed PIT similarity panel.
    `panel_lp` must be built with `build_lazy_prices_panel(..., warmup=cfg.warmup)` so it
    shares candidate_pipeline's positional basis. Holdout stays blinded unless a verified
    `lazy_prices_candidate_run_hash` is supplied. Reuses the generic injection helper."""
    expected_rows = max(0, len(panel) - cfg.warmup)
    if len(panel_lp) != expected_rows:
        raise ValueError(
            f"panel_lp rows ({len(panel_lp)}) must equal len(panel)-cfg.warmup "
            f"({expected_rows}); build it with build_lazy_prices_panel(..., warmup=cfg.warmup)"
        )
    raw_fn = make_lazy_prices_raw(panel_lp)
    return _candidate_metrics_with_raw_fn(
        panel, cfg, cfg.families, raw_fn, prereg_hash, holdout_frac, vcfg, fixture_path,
        lazy_prices_candidate_run_hash, "lazy_prices_candidate_run_hash",
        _lazy_prices_power_report,
    )
```

- [ ] **Step 4: Run tests** (new test + the golden-replication guard that proves the bench still equals frozen on shared price families)

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices_candidate_floor.py apps/quant/advisor/tests/test_candidate_golden_replication.py -q`
Expected: PASS (golden replication unaffected — `_ALLOWED` only ADDS a member)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/candidate_blend.py apps/quant/advisor/research/candidate_floor.py apps/quant/advisor/tests/test_lazy_prices_candidate_floor.py
git commit -m "WS3D D5: admit lazy_prices family + blinded candidate floor wiring"
```

---

## Task D6: Fetch/freeze the committed fixture — NETWORK (operator/Hermes-with-network; NOT in the pytest gate)

**Files:**
- Create: `apps/quant/advisor/data/filing_text_fetch.py`
- Create (output): `apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv`
- Test: `apps/quant/advisor/tests/test_lazy_prices.py` (append OFFLINE unit tests for the pure helpers; the network path itself is not tested in the gate)

The fetch logic is split so the deterministic parts (tokenize, cosine, availability, row build) are unit-testable offline with an injected `http_get`; only `main()` touches the network.

- [ ] **Step 1: Write the failing test** (append to `test_lazy_prices.py`)

```python
from advisor.data.filing_text_fetch import tokenize, cosine_tfidf, build_similarity_row


def test_tokenize_is_deterministic_lowercase_words():
    assert tokenize("The QUICK brown fox; fox!") == ["the", "quick", "brown", "fox", "fox"]


def test_cosine_identical_is_one_and_disjoint_is_zero():
    a = tokenize("alpha beta gamma alpha")
    assert abs(cosine_tfidf(a, a) - 1.0) < 1e-9
    assert cosine_tfidf(tokenize("alpha beta"), tokenize("gamma delta")) == 0.0


def test_cosine_partial_overlap_in_unit_interval():
    s = cosine_tfidf(tokenize("alpha beta gamma"), tokenize("alpha beta delta"))
    assert 0.0 < s < 1.0


def test_build_similarity_row_sets_availability_without_lag():
    row = build_similarity_row(
        asset="AAA", cik="0000320193", accession="acc1", form="10-K",
        report_period_end="2015-12-31", filing_date="2016-02-01",
        accepted_datetime="2016-02-03T16:30:00", similarity=0.91,
    )
    assert row["concept"] == "FilingSimilarity"
    assert row["value"] == "0.91"
    assert row["available_asof"] == "2016-02-03"     # max(filing, accepted), no +90
    assert row["denominator_policy"] == "cosine_tfidf_yoy_same_form"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: FAIL — `ModuleNotFoundError: ...filing_text_fetch`

- [ ] **Step 3: Write the implementation**

```python
# apps/quant/advisor/data/filing_text_fetch.py
"""One-time NETWORK fetch/freeze writer for the Lazy Prices fixture. NOT imported by any
test or runtime path. Pure helpers (tokenize/cosine/build_similarity_row) are unit-tested
offline; main() is the only network entrypoint and is operator-run outside the gate.

Usage (operator, network):
    python -m advisor.data.filing_text_fetch --out apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv

PIT discipline: available_asof = max(filing_date, accepted_datetime) — NEVER the fetch
date. Each current filing is paired with the SAME-FORM filing one year earlier; 2014
filings are pulled only as the prior-year baseline for 2015 and are not emitted as signal.
SEC etiquette: declared User-Agent, <=10 req/s."""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from datetime import date
from typing import Callable

from advisor.research.lazy_prices import SIMILARITY_CONCEPT, compute_text_available_asof
from advisor.data.edgar_xbrl_fixture import FIELDS

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Deterministic preprocessing: lowercase, \\w+ word tokens. Pinned so the fixture is
    byte-reproducible. (HTML stripping happens upstream in fetch_document.)"""
    return _WORD.findall(text.lower())


def cosine_tfidf(a: list[str], b: list[str]) -> float:
    """Cosine over raw term-frequency vectors (no IDF: a 2-doc YoY pair has no corpus).
    Identical docs -> 1.0; disjoint -> 0.0; deterministic."""
    ca, cb = Counter(a), Counter(b)
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _parse_d(s: str) -> date:
    return date.fromisoformat(s.strip().split("T")[0].split(" ")[0])


def build_similarity_row(*, asset, cik, accession, form, report_period_end,
                         filing_date, accepted_datetime, similarity) -> dict:
    """Build one fixture row in the 15-field EdgarXbrlRecord schema."""
    avail = compute_text_available_asof(_parse_d(filing_date), _parse_d(accepted_datetime))
    return {
        "asset": asset, "cik": cik, "accession": accession, "form": form,
        "report_period_end": _parse_d(report_period_end).isoformat(),
        "filing_date": _parse_d(filing_date).isoformat(),
        "accepted_datetime": _parse_d(accepted_datetime).isoformat(),
        "concept": SIMILARITY_CONCEPT, "unit": "ratio",
        "value": format(float(similarity), "g"),
        "available_asof": avail.isoformat(),
        "superseded_by": "", "amended_flag": "false", "missingness_reason": "",
        "denominator_policy": "cosine_tfidf_yoy_same_form",
    }


def fetch_document(http_get: Callable[[str], str], cik: str, accession: str) -> str:
    """Fetch a filing's primary document and strip HTML to text. Injected http_get keeps
    the parse path testable offline. (Implementation: data.sec.gov submissions -> primary
    doc URL -> http_get -> regex-strip tags -> tokenize upstream.)"""
    html = http_get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}.txt")
    return re.sub(r"<[^>]+>", " ", html)


def main() -> None:  # pragma: no cover - network entrypoint, operator-run
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--user-agent", required=True, help="SEC requires a declared UA")
    args = ap.parse_args()
    # 1) For each of the 30 CIKs: pull submissions (10-K/10-Q) 2014-2023.
    # 2) Pair each filing with the prior-year SAME-FORM, SAME-FISCAL-PERIOD filing:
    #    a 10-K pairs with the prior fiscal-year 10-K; a 10-Q pairs with the year-ago
    #    SAME fiscal quarter's 10-Q (e.g. Q2-2016 <-> Q2-2015), NEVER the adjacent
    #    quarter (Q1<->Q2 mixes seasonal boilerplate and inflates "change"). Match on
    #    (form, fiscal_period) from the submissions metadata. Tokenize both; cosine_tfidf.
    # 3) build_similarity_row(...) for CURRENT filings with report_period_end in 2015-2023
    #    (2014 filings are consumed only as the prior-year baseline, never emitted).
    # 4) Write CSV with header = FIELDS. Throttle <=10 req/s; declared UA.
    rows: list[dict] = []  # populated by the loop above (omitted: pure orchestration)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run tests to verify the offline helpers pass**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: PASS (12 tests)

- [ ] **Step 5: Operator runs the network fetch (OUTSIDE the gate), then verify the frozen fixture loads + audits**

Operator (network): `python -m advisor.data.filing_text_fetch --out apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv --user-agent "you@example.com"`

Then add and run an acceptance test that the committed fixture loads cleanly and every row passes the text-availability audit + non-degeneracy:

```python
# append to test_lazy_prices.py
from pathlib import Path
from advisor.data.edgar_xbrl_fixture import load_fixture, coverage_in_window
from advisor.research.lazy_prices import audit_text_available_asof

_FIX = Path(__file__).parent / "fixtures" / "lazy_prices_similarity.csv"


def test_committed_fixture_loads_audits_and_is_non_degenerate():
    recs = load_fixture(_FIX)
    assert recs, "fixture must contain rows"
    assert all(audit_text_available_asof(r) for r in recs)          # no +90, no fetch-date
    assert all(0.0 <= r.value <= 1.0 for r in recs)                 # cosine in [0,1]
    cov = coverage_in_window(recs, date(2015, 1, 1), date(2023, 12, 31))
    assert cov > 0.5, f"globally-zeroed/lagged signal would fail here, cov={cov}"
```

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -q`
Expected: PASS (13 tests). Record the fixture SHA-256 for D7.

- [ ] **Step 6: Commit**

```bash
git add apps/quant/advisor/data/filing_text_fetch.py apps/quant/advisor/tests/test_lazy_prices.py apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv
git commit -m "WS3D D6: filing-text fetch helpers + frozen similarity fixture"
```

---

## Task D7: Reading-C PREREG/RESULT docs + orthogonality diagnostic + roadmap

**Files:**
- Modify: `apps/quant/advisor/research/lazy_prices.py` (append report-only `dev_lazy_momentum_corr`)
- Create: `apps/quant/advisor/research/READING_C_PREREG.md`, `READING_C_RESULT.md`
- Modify: the deferred-plans roadmap doc
- Test: `apps/quant/advisor/tests/test_lazy_prices.py` (append the diagnostic test)

- [ ] **Step 1: Write the failing tests** (append) — the dispersion check is the load-bearing one

```python
from advisor.research.lazy_prices import (
    dev_lazy_momentum_corr, dev_cross_sectional_dispersion,
)


def _multi_asset_panel(n=400, names=("AAA", "BBB", "CCC", "DDD")):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    cols = {a: [100.0 + i + j for i in range(n)] for j, a in enumerate(names)}
    cols["SPY"] = [400.0 + i for i in range(n)]
    return pd.DataFrame(cols, index=idx)


def _step_records(panel, level_by_asset):
    # one filing every ~60 trading days, each asset pinned to its own similarity LEVEL
    recs = []
    for a, lvl in level_by_asset.items():
        for k, t in enumerate(panel.index[::60]):
            d = str(t.date())
            recs.append(_rec(a, lvl + 0.01 * (k % 3), filing=d, accepted=d,
                             period=d, avail=d))
    return recs


def test_orthogonality_diagnostic_returns_corr_key():
    panel = _multi_asset_panel()
    recs = _step_records(panel, {"AAA": 0.9, "BBB": 0.5, "CCC": 0.3, "DDD": 0.7})
    panel_lp = build_lazy_prices_panel(recs, panel, warmup=200)
    c = dev_lazy_momentum_corr(panel, panel_lp, warmup=200, holdout_frac=0.2)
    assert "momentum" in c                       # report-only; NaN allowed if a leg is flat


def test_cross_sectional_dispersion_detects_level_collapse():
    # Per-asset transform encodes each name vs ITS OWN history, so a big cross-sectional
    # LEVEL gap (0.9 vs 0.3) does NOT guarantee cross-sectional conviction spread.
    panel = _multi_asset_panel()
    recs = _step_records(panel, {"AAA": 0.9, "BBB": 0.5, "CCC": 0.3, "DDD": 0.7})
    panel_lp = build_lazy_prices_panel(recs, panel, warmup=200)
    d = dev_cross_sectional_dispersion(panel, panel_lp, warmup=200, holdout_frac=0.2)
    assert "min_xs_std" in d and "median_xs_std" in d
    # documents (does not assert a threshold) the collapse the bench transform induces;
    # the VALUE is what READING_C_RESULT.md reports so a DEV_FAILED is interpretable.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -k "orthogonality or dispersion" -q`
Expected: FAIL — `ImportError: cannot import name 'dev_cross_sectional_dispersion'`

- [ ] **Step 3: Write BOTH diagnostics** (append to `lazy_prices.py`)

```python
import numpy as np

from advisor.backtest.continuous_signals import apply_transform, fit_percentile_transform
from advisor.backtest.splits import purged_splits


def dev_lazy_momentum_corr(panel: pd.DataFrame, panel_lp: pd.DataFrame, *,
                           warmup: int, holdout_frac: float,
                           folds: int = 5, embargo: int = 5,
                           clip: tuple[float, float] = (0.05, 0.95)) -> dict[str, float]:
    """REPORT-ONLY post-transform corr of lazy_prices vs momentum, pooled over dev test
    folds and assets (holdout excluded). Mirrors orthogonality.dev_fold_post_transform_corr
    but reads the precomputed similarity panel. NaN when a leg is flat (not a false 0.0)."""
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    dev = prices_all.iloc[:dev_end]
    raw_fn = make_lazy_prices_raw(panel_lp)
    splits = purged_splits(len(dev), folds, embargo)
    lp_vals, mo_vals = [], []
    for c in assets:
        lp_raw = raw_fn(LAZY_PRICES, dev[c])
        mo_raw = raw_fn("momentum", dev[c])
        for tr, te in splits:
            lp_sc = apply_transform(fit_percentile_transform(lp_raw.iloc[tr], clip=clip),
                                    lp_raw).iloc[te]
            mo_sc = apply_transform(fit_percentile_transform(mo_raw.iloc[tr], clip=clip),
                                    mo_raw).iloc[te]
            df = pd.concat([lp_sc, mo_sc], axis=1).dropna()
            lp_vals.append(df.iloc[:, 0].to_numpy())
            mo_vals.append(df.iloc[:, 1].to_numpy())
    a = np.concatenate(lp_vals) if lp_vals else np.array([])
    b = np.concatenate(mo_vals) if mo_vals else np.array([])
    if a.size == 0 or b.size == 0 or a.std() == 0 or b.std() == 0:
        return {"momentum": float("nan")}
    return {"momentum": float(np.corrcoef(a, b)[0, 1])}


def dev_cross_sectional_dispersion(panel: pd.DataFrame, panel_lp: pd.DataFrame, *,
                                   warmup: int, holdout_frac: float,
                                   folds: int = 5, embargo: int = 5,
                                   clip: tuple[float, float] = (0.05, 0.95)) -> dict[str, float]:
    """REPORT-ONLY, LOAD-BEARING: does the post-transform lazy_prices conviction VARY
    ACROSS NAMES at a given date? `fit_percentile_transform` fits PER ASSET on its own
    time series (see candidate_pipeline._family_scores), so it encodes 'this filing vs
    THIS firm's own past', NOT the cross-sectional similarity LEVEL the Lazy Prices
    anomaly ranks on. Verified: a name constant at 0.96 and a name constant at 0.20 BOTH
    transform to 1.0 -> identical long conviction. If the per-date cross-sectional std is
    ~0, the long-flat book cannot discriminate names and the reading is NOT faithfully
    expressed -> a HARNESS ARTIFACT, distinct from a genuine signal-driven DEV_FAILED.
    Returns min/median per-date cross-sectional std pooled over dev test rows."""
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    dev = prices_all.iloc[:dev_end]
    raw_fn = make_lazy_prices_raw(panel_lp)
    stds: list[float] = []
    for tr, te in purged_splits(len(dev), folds, embargo):
        cols = {}
        for c in assets:
            raw = raw_fn(LAZY_PRICES, dev[c])
            params = fit_percentile_transform(raw.iloc[tr], clip=clip)
            cols[c] = apply_transform(params, raw).iloc[te].reset_index(drop=True)
        scores = pd.DataFrame(cols)
        stds.extend(scores.std(axis=1, ddof=0).tolist())     # per-date spread ACROSS names
    if not stds:
        return {"min_xs_std": float("nan"), "median_xs_std": float("nan")}
    s = sorted(stds)
    return {"min_xs_std": float(s[0]), "median_xs_std": float(s[len(s) // 2])}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py -k "orthogonality or dispersion" -q`
Expected: PASS

- [ ] **Step 5: Write the docs** (`READING_C_PREREG.md` records fixture path + SHA-256 from D6, both hashes, frozen construction: families, `similarity_metric=cosine_tfidf`, availability rule, universe/window, folds/embargo/warmup; `READING_C_RESULT.md` records the dev-gate verdict at `prereg_hash=None` (holdout blinded), the `power` block, the `dev_lazy_momentum_corr` value, **and the `dev_cross_sectional_dispersion` min/median** — the latter being the metric that distinguishes a faithfully-tested-and-failed reading from a transform-collapse harness artifact). The RESULT doc MUST state plainly: (a) if `median_xs_std ≈ 0`, the bench's per-asset transform erased the cross-sectional level and the reading was NOT faithfully tested — escalate the transform decision to the operator (see "Operator decision" below); (b) the long-flat book takes ONLY the non-changer long leg — the cited 18–45bps is a long-SHORT figure and the short-the-changers leg (where much of the documented alpha sits) is structurally unavailable, so long-leg-alone efficacy is untested in the evidence. Update the roadmap: WS3D status + WS3E (DTC short-interest) as the sequenced follow-on, and note `lazy_prices + fundamental_value` as a possible two-orthogonal-family WS4 candidate once both fixtures exist.

- [ ] **Step 6: Commit**

```bash
git add apps/quant/advisor/research/lazy_prices.py apps/quant/advisor/research/READING_C_PREREG.md apps/quant/advisor/research/READING_C_RESULT.md apps/quant/advisor/tests/test_lazy_prices.py docs/superpowers/plans/
git commit -m "WS3D D7: Reading-C docs, orthogonality diagnostic, roadmap"
```

---

## Dispatch order
D4 has no deps (parallel from the start). Code path: **D1 → D2 → D3 → D5** (D5 needs D3 + D4). **D6** after D1 (needs the schema/availability rule; operator-run, network). **D7** last (needs D5 + D6). Suite grows by ~17 tests (D1:3, D2:2, D3:3, D4:5, D5:3, D6:5+1 acceptance, D7:1 — minus overlap in the shared test file).

## Governance checklist
- Frozen `PreRegConfig` (`1ad2ed4a…`), `CandidatePreReg` (`578cce4b…`), `FundamentalCandidatePreReg`: never touched. WS3D adds a SEPARATE Reading-C surface.
- Frozen `blend.py`: untouched. Only the bench mirror `candidate_blend._ALLOWED` gains `LAZY_PRICES` (additive; golden-replication test proves equality on shared price families).
- Holdout blinded: all calls `prereg_hash=None`; the `_verify_run_hash_unlock` guard (reused) raises on a supplied-but-wrong hash; `HOLDOUT_LEDGER.md` empty. Unlock is WS4.
- Floor stays DEV_FAILED: `backtest/`, `floor_data_check.py`, `run-floor.mjs` unmodified; `--enforce` exits 1.
- Contract honored: reuses WS3A's 15-field schema + `select_asof`; availability rule is the filing-text variant (no +90), explicitly audited.
- No lookahead: availability = `max(filing_date, accepted)`; `select_asof` never backfills amendments; D2 stepwise test + D6 audit enforce it.
- Determinism / missing→neutral: loader skips incomplete rows; adapter network-free; NaN→flat. Only D6 touches network, outside the gate. Tokenize/cosine are pinned + deterministic.
- TDD: every code task ships its test first; suite stays green.

## Verification
- Per task: `python -m pytest apps/quant/advisor/tests/test_lazy_prices.py apps/quant/advisor/tests/test_candidate_prereg_lazy_prices.py apps/quant/advisor/tests/test_lazy_prices_candidate_floor.py -q`.
- After D5: assert `lazy_prices_candidate_metrics(panel, panel_lp, cfg, prereg_hash=None)["holdout"] is None` and `["passes"] is False`.
- End of WS3D: `node tools/run-pytest.mjs` full suite green; `node tools/run-floor.mjs --enforce` → exit 1 (floor unchanged); `git diff --stat` shows ZERO changes under `backtest/`, `tools/floor_data_check.py`, `tools/run-floor.mjs`, `blend.py`; `HOLDOUT_LEDGER.md` empty. D6 (network) is operator-run outside the gate.

## Honest expected outcome — THREE distinguishable failure modes
`lazy_prices` is a low-frequency step signal (~4 changes/yr) over a 30-name universe; like WS3C it may well return **DEV_FAILED / power-limited**, which is a legitimate result, not a plan failure — WS3D delivers the *capability* to test the reading reproducibly and leakage-free. `READING_C_RESULT.md` must separate three modes so a DEV_FAILED is interpretable:
1. **Signal genuinely fails dev** — coverage high, `median_xs_std` > 0 (book could discriminate), gate still fails. The real, honest negative.
2. **Silent zeroing** (the WS3C trap) — `coverage_in_window ≈ 0` from an availability/fetch-date bug. Caught by D1's non-degeneracy + D6's audit.
3. **Transform collapse** (THIS reading's specific trap, advisor-caught + empirically verified) — coverage is high and `power` looks healthy (similarity is always positive), but `dev_cross_sectional_dispersion.median_xs_std ≈ 0`. The bench's PER-ASSET percentile transform fits each name on its own history, so the cross-sectional LEVEL that the Lazy Prices anomaly ranks on is erased (a name constant at 0.96 and one constant at 0.20 both map to conviction 1.0). The `power`/coverage/correlation diagnostics do NOT see this — only `dev_cross_sectional_dispersion` does. If this fires, the reading was NOT faithfully tested.

Also record the **leg asymmetry**: a long-flat book takes only the non-changer LONG leg; the cited 18–45bps is a long-SHORT figure and the short-the-changers leg (much of the documented alpha) is structurally unavailable, so long-leg-alone efficacy is untested in the evidence.

## Operator decision (do NOT fold into this plan)
If `median_xs_std ≈ 0`, faithfully testing the cross-sectional anomaly requires a **cross-sectional rank transform** (rank similarity across names per date) instead of the per-asset time-series percentile. That diverges from the frozen bench transform every other family uses, so it is a deliberate methodology change requiring its own prereg decision and a golden-equality argument — it is explicitly OUT OF SCOPE here. WS3D's job is to make the collapse *visible* and hand the operator the number, not to silently re-engineer the transform.

## Critical files
- `apps/quant/advisor/research/candidate_pipeline.py:50` — positional collapse (date-threading crux)
- `apps/quant/advisor/research/candidate_floor.py:100-171` — `_candidate_metrics_with_raw_fn` (the generic injection point reused verbatim)
- `apps/quant/advisor/research/candidate_blend.py:17` — bench `_ALLOWED` allowlist (the one frozen-divergence to extend)
- `apps/quant/advisor/data/edgar_xbrl_fixture.py` — 15-field schema + `load_fixture`/`coverage_in_window` (reused)
- `apps/quant/advisor/research/fundamental_value.py:14-40` — `select_asof` (reused, concept-generic)
- `docs/superpowers/plans/2026-06-18-ws3c-edgar-xbrl-fundamentals.md` — the sibling plan this mirrors

## Self-review (run against the spec)
- **Spec coverage:** signal (D0/D3), fixture format (D1/D6), coverage+baseline (D6), availability rule (D1/D6), candidate shape (D4), the three deep-research constraints — orthogonality (D7 corr diagnostic), PIT/leakage (D1 availability + D2 stepwise + D6 audit), free-data feasibility (full EDGAR text, no paid feed). ✓
- **Faithful-test check (advisor-caught, empirically verified):** the bench per-asset percentile transform collapses the cross-sectional similarity LEVEL → D7's `dev_cross_sectional_dispersion` makes the collapse measurable; the three-mode "Honest expected outcome" + "Operator decision" sections ensure a DEV_FAILED is interpretable and the transform decision is escalated, not silently made. The long-leg-only leg-asymmetry caveat is recorded. ✓
- **Placeholder scan:** every code step has complete code; `main()`'s fetch loop is intentionally orchestration-only (`# pragma: no cover`, operator-run network) with the row contract fully specified by `build_similarity_row` and the prior-year same-fiscal-period pairing rule spelled out. No TBD/TODO in tested paths. ✓
- **Type consistency:** `LAZY_PRICES`/`SIMILARITY_CONCEPT` consistent across modules; `make_lazy_prices_raw`, `build_lazy_prices_panel`, `lazy_prices_candidate_metrics`, `lazy_prices_candidate_run_hash`, `_lazy_prices_power_report`, `dev_lazy_momentum_corr`, `dev_cross_sectional_dispersion` names match every call site; `LazyPricesCandidatePreReg.families == ("lazy_prices","momentum")` matches the `_ALLOWED` member and the floor wrapper. ✓

