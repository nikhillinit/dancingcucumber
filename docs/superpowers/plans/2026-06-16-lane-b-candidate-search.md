# Lane B — Candidate Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Execution channel:** all code edits go through Hermes solo file-pointer dispatch (memory `hermes-dispatch-windows`): write the full file content to `ai-logs/hermes/<task>.md`, dispatch `npm run hermes:production -- --task "open ai-logs/hermes/<task>.md and follow EXACTLY; verify ONLY with python -m pytest <path> -q; do NOT commit"`, then verify Codex's real git state, run the test yourself, commit one-per-task with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer. Measurement tasks (runs, not tests) are operator/Claude-direct, not Hermes.

**Goal:** Build a rail-safe, self-pre-registered candidate-evaluation bench that tests whether an orthogonal `value` signal blended with `momentum` can beat its own parts under the floor's exact methodology — producing a *proven-or-disproven candidate + evidence*, without touching the frozen floor or `ensemble_vote`.

**Architecture:** A new `apps/quant/advisor/research/` package imports the frozen `backtest/` primitives **read-only** and re-orchestrates them (~40 lines) over a new price-derived `value` (long-term-reversal) signal. It owns an immutable `CandidatePreReg` (new hash surface — never touches `PreRegConfig`), reuses `floor_prices.csv` read-only, and emits its own verdict. A **golden-replication test** proves the bench reproduces the floor's own numbers before any new signal is trusted. An **orthogonality kill-gate** stops cheaply if `value` is a relabel of an already-rejected factor. The frozen floor (`run-floor --enforce` → exit 1) and `allocator.py` are untouched throughout; **promotion of a passing candidate is explicitly out of scope** (it is Plans 1b/3, deliberate future steps).

**Tech Stack:** Python 3, pandas, numpy, pytest. Reuses `advisor.backtest.{splits,stats,blend,book,portfolio,dev_gate,validation,validation_prereg,continuous_signals}` as read-only libraries.

---

## Hard rails (any violation fails the task)

1. **Never modify anything under `backtest/`**, nor `portfolio/allocator.py` / `ensemble_vote` / `allocate` / `risk/limits.py`. The bench *imports* them; it never edits them. `floor_prices.csv` / `PREREG.md` / `UNIVERSE_RULE.md` / `allocator.py` are guard-blocked by `.claude/hooks/guard-frozen-floor.mjs`.
2. **Never weaken the release gate.** `npm run advisor-gate` stays exit 0; `node tools/run-floor.mjs --enforce` stays **exit 1**. This plan adds report-only research code; it does not size capital and does not promote a candidate.
3. **No fabricated data.** The bench uses only the existing `floor_prices.csv` (Reading A). Missing/NaN signal → flat (0), never synthesized. Reading B (fundamentals fixture) is gated behind an explicit operator decision (Task 11) and is NOT built here.
4. **Pre-commit methodology, not outcomes.** `value_skip`, `value_lookback`, the family set + order, `orthogonality_tau`, and `declared_trials_N` are frozen in `CandidatePreReg` + `CANDIDATE_PREREG.md` BEFORE any result is seen. The floor's existing gates (§7.2 `ensemble>best_family` LCB>0, §7.1 beat-SPY by margin, DSR>0.95) **are** the acceptance bar — no new outcome thresholds invented. Each additional lookback/family-set tried increments `declared_trials_N` **on `CandidateValidationPreReg`** (the surface `validation_report` actually reads — Amendment F1) and re-runs DSR at the higher N.
5. **Holdout-leakage guard.** `floor_prices.csv`'s tail IS the reserved holdout. The value signal is constructed/tuned on the dev split only; `holdout_frac` is applied identically to the floor; the holdout is touched **once, iff** the dev gate passes. **(Amendment F2)** Unlock the tail only via a `candidate_run_hash(cfg, fixture)` over config **+ fixture bytes** (not the fixture-blind `candidate_hash`); append every reserved-tail touch to a shared `HOLDOUT_LEDGER.md`; a side-bench touch BURNS the shared tail, so promotion (Plans 1b/3) requires a FRESH holdout, never the peeked one.
6. **Do NOT add fields to `PreRegConfig`** (frozen, SHA-hashed `1ad2ed4a…`). All candidate config lives in the new `CandidatePreReg`.

---

## Debate-hardening amendments (2026-06-17 — authoritative deltas)

> These override the task bodies below where they conflict. Each came from the Hermes
> adversarial debate and was verified against the frozen code; rationale + file:line in
> `docs/superpowers/plans/2026-06-16-lane-b-debate-findings.md`. Apply them as part of the
> task they name.

### Amendment F1 (CRITICAL) — candidate DSR must use a candidate validation surface

`validation_report(returns, family_returns, vcfg, ...)` reads `vcfg.declared_trials_N` /
`vcfg.declared_var_sr` ONLY (`backtest/validation.py:123,126`). Task 7 passing the floor's
`DEFAULT_VALIDATION` makes `CandidatePreReg.declared_trials_N` a **dead field**, and rail #4's
"increment `declared_trials_N`, re-run DSR at higher N" **unimplementable**. Fix:

- Add `apps/quant/advisor/research/candidate_validation_prereg.py` — a frozen
  `CandidateValidationPreReg` (its OWN hash surface; never touches `ValidationPreReg`) with
  `declared_trials_N` (the candidate's pre-registered trial count — primary=45; the secondary
  `value+momentum+trend` run bumps it) and `declared_var_sr` **recalibrated** from the
  candidate's own per-obs trial Sharpes (or, if reused as `1e-4`, a one-line written
  justification that 1e-4 ≥ the candidate book's measured cross-trial dispersion, i.e.
  stricter — `SR0 ∝ √var_sr`). `DEFAULT_CANDIDATE_VALIDATION = CandidateValidationPreReg()`.
- Task 7 `candidate_metrics` imports and passes `DEFAULT_CANDIDATE_VALIDATION` to
  `validation_report` (NOT `DEFAULT_VALIDATION`).
- New test (Task 7): `dataclasses.replace(DEFAULT_CANDIDATE_VALIDATION, declared_trials_N=90)`
  fed through `candidate_metrics` changes `validation["n_used"]` (proves the field is live).
- **Keep validation report-only (floor-faithful):** do NOT fold DSR into the machine
  `passes` — `data_floor.py:74` sets `passes = verdict=="PASSED"` and the `"validation"`
  key never mutates the verdict. DSR-confirmation stays the operator promotion-readiness gate
  (Task 8 Step 3 / Task 9), exactly as the floor treats it. Rail #4's "increment N" now points
  at `CandidateValidationPreReg`.

### Amendment F2 (CRITICAL) — control reuse of the shared reserved holdout

`floor_prices.csv`'s tail IS the floor's reserved holdout (`FLOOR_RESULT.md:32` "do NOT run
it … reserved"). The floor already hardened this (`tools/floor_data_check.py:63-71`, its own
"debate finding #1"): the tail unlocks ONLY via `config_hash(DEFAULT_CONFIG, FIXTURE)`, which
**includes fixture bytes**. The plan's `candidate_hash()` deliberately EXCLUDES fixture bytes,
and `candidate_metrics` unlocks the holdout on any `prereg_hash is not None` — strictly weaker
than the floor on the exact axis the floor already fixed. Fix:

- **Candidate run-hash over config + fixture bytes.** Add `candidate_run_hash(cfg, fixture_path)`
  (mirror `config_hash(cfg, FIXTURE)`) and use ITS output as the `prereg_hash` that unlocks the
  holdout in Task 8 — never the fixture-blind `candidate_hash()` and never an arbitrary string.
  (`candidate_hash()` stays the methodology-only id in `CANDIDATE_PREREG.md`.)
- **Shared holdout-touch ledger.** Add `apps/quant/advisor/research/HOLDOUT_LEDGER.md`, an
  append-only record; EVERY evaluation of the reserved tail (the floor's own future `--holdout`
  run AND every candidate that earns a holdout) appends a row: date, run-hash, families,
  verdict. The ledger makes the multiple-testing count on the shared tail honest and visible.
- **Promotion needs a FRESH holdout.** State (rail #5 + Task 9) that ANY side-bench holdout
  touch *burns* the shared tail; a PASSED candidate cannot be promoted on the peeked tail —
  re-pre-registration (Plans 1b/3) must reserve a fresh holdout (e.g. extended/rolled fixture).

### Amendment F3 (HIGH) — golden replication must prove mirror == frozen element-wise

`abs=0.01` on a 0.732 aggregate Sharpe (~1.4% relative) can hide a one-line mirror divergence,
conflates a legit floor change with a mirror bug, and never exercises the value 270-row NaN
prefix / empty-`pos` percentile-fit path (`continuous_signals.py:26-31`). Fix Task 4:

- **Add an element-wise equality test (the real drift guard):** with `raw_fn=raw_metric`,
  `run_dev_sweep_ext(panel, fams)` must equal frozen `run_dev_sweep(panel, fams)` — assert
  `fold_deltas` equal (exact), `pd.testing.assert_series_equal` on `ensemble_test_returns` and
  `best_family_test_returns`, and equal `chosen_weights`. This goes red the moment EITHER
  pipeline changes without the other — independent of any hardcoded number.
- Keep the 0.732/0.828 assertion as a *documentation* check of the published numbers, but it is
  no longer the trust anchor.
- **Holdout-mirror equality on SYNTHETIC data only** (Amendment F2): a `run_holdout_ext` vs
  `run_holdout` equality check touches the real reserved tail — run it on a synthetic panel, not
  `floor_prices.csv`.
- **Add an empty-`pos` value-transform test:** a price window where the formation return is ≥0
  everywhere → `fit_percentile_transform` gets empty `pos` → value scores flat (0), no crash —
  proving the degenerate path the golden families never hit is handled.

### Amendment F4 (HIGH) — measure orthogonality on the surface that enters the blend

The blend uses TRANSFORMED long-flat conviction scores (`pipeline.py:67` blends
`apply_transform` outputs; ≤0 raw → 0, `continuous_signals.py:34-45`), so the diversification
§7.2 rewards lives in the transformed scores, not the raw. Raw pooled Pearson across
heterogeneous assets (plan Task 5) can disagree with that. Fix Tasks 5–6:

- Keep raw Pearson as a pre-registered **diagnostic**, but ALSO compute/gate the
  **post-transform fold-level** correlation: fit `fit_percentile_transform` on each dev fold's
  TRAIN rows, `apply_transform`, then correlate `value`'s scores vs each neighbor's scores on
  the fold TEST rows (holdout excluded) — this is what actually blends.
- Add **Spearman / rank** correlation as a scale-robust cross-check on the raw (heterogeneous
  assets make Pearson scale-fragile).
- Make the **`momentum` decision rule explicit**: momentum is the blend partner, so its
  correlation is reported AND, if `|corr(value, momentum)|` (post-transform) is high, the
  orthogonality gate is demoted from "the pivot / clean PASS" to a **coarse pre-filter** — §7.2
  on the real fixture becomes the adjudicator, not a green light. Document `τ=0.40` as a
  pre-filter threshold, not a proof of diversification.

<!-- amendments-end -->

---

## Design decision: the fixture-feasible horizon (settled before pre-registration)

The frozen pipeline is **slice-then-compute**: `_family_scores` runs the raw metric on the already-sliced `dev` frame, so `value`'s `shift(value_lookback)` makes the **first `value_lookback` dev rows NaN→flat**, independent of `warmup`. On the real fixture (~2268 days, warmup 200 → dev ≈ 1654, 5-fold `purged_splits` → 4 test folds of ~330; fold-1 train ends at ~325):

> **`dead_prefix_folds = value_lookback / (dev_rows / folds)`.** If a fold's *train* slice is entirely below `value_lookback`, `fit_percentile_transform` gets an empty `pos` → value is flat for that whole fold → the blend collapses toward momentum-standalone → negative fold deltas → `dev_gate` ("median Δ>0 AND ≥70% folds positive") fails **for non-signal reasons**. A classic 36–60-mo LT-reversal lookback (756–1260) gives `dead_prefix ≈ 1.8–3.8` → a **guaranteed false negative** that would burn the pre-registered trial.

**Therefore classic long-term reversal is NOT testable on this 9-yr price-only fixture.** The settled decision: pre-register an **intermediate-term reversal at a feasible horizon** — `value_lookback = 270`, `value_skip = 126` (formation window ≈ 13mo→6mo ago, still excluding the recent 6-mo momentum window), keeping `value_lookback < ~325` so **every dev fold has a live value leg** (a fair dev gate). This horizon sits *near* `long_momentum`, so the **orthogonality kill-gate (Task 6) becomes the genuine pivot** — if intermediate reversal is just negated momentum, it dies there cheaply, and classic LT-reversal (and fundamental value) move to **Reading B** (Task 11) on a longer or fundamentals-bearing fixture. Picking 756-with-a-caveat is rejected; the feasibility ceiling is enforced by a unit test (Task 1) and a real-fixture guard (Task 4).

---

## File structure (created; all NEW, none frozen)

| File | Responsibility |
|---|---|
| `apps/quant/advisor/research/__init__.py` | Package marker. |
| `apps/quant/advisor/research/candidate_prereg.py` | `CandidatePreReg` frozen dataclass + `candidate_hash()`; `DEFAULT_CANDIDATE`. The immutable methodology surface. |
| `apps/quant/advisor/research/candidate_validation_prereg.py` | (Amendment F1) `CandidateValidationPreReg` frozen dataclass + `DEFAULT_CANDIDATE_VALIDATION`. Own hash surface for the candidate's DSR `declared_trials_N` / recalibrated `declared_var_sr`; the live input to `validation_report`. |
| `apps/quant/advisor/research/candidate_signals.py` | `candidate_raw(family, prices, *, value_skip, value_lookback)` — delegates known families to frozen `raw_metric`; adds `value` (long-term reversal). |
| `apps/quant/advisor/research/candidate_pipeline.py` | `run_dev_sweep_ext` / `run_holdout_ext` — frozen pipeline mirrored with an injected `raw_fn`. Reuses all frozen helpers. |
| `apps/quant/advisor/research/orthogonality.py` | `dev_fold_raw_corr(...)` — Pearson corr of `value` raw vs `long_momentum` / `mean_reversion` on dev folds (holdout excluded). |
| `apps/quant/advisor/research/candidate_floor.py` | `candidate_metrics(...)` — mirrors `floor_metrics` (dev gate → holdout-once → §7.1/§7.2 + validation_report) over the candidate pipeline + `CandidatePreReg`. |
| `apps/quant/advisor/research/CANDIDATE_PREREG.md` | Immutable pre-registration record (hash, frozen constants, candidate order). |
| `apps/quant/advisor/research/CANDIDATE_RESULT.md` | Measured verdict + evidence (written by the measurement tasks). |
| `apps/quant/advisor/research/HOLDOUT_LEDGER.md` | (Amendment F2) Append-only ledger of every reserved-tail touch (floor + each candidate): date, run-hash, families, verdict. Makes shared-holdout multiple-testing honest. |
| `apps/quant/advisor/tests/test_candidate_*.py` | Unit tests (collected by the gate; keep the suite green, count rising). |

---

## Phase B0 — The research bench (rail-safe, floor-faithful infrastructure)

### Task 1: `CandidatePreReg` — the immutable methodology surface

**Files:**
- Create: `apps/quant/advisor/research/__init__.py` (empty)
- Create: `apps/quant/advisor/research/candidate_prereg.py`
- Test: `apps/quant/advisor/tests/test_candidate_prereg.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_candidate_prereg.py
from advisor.research.candidate_prereg import (
    CandidatePreReg, DEFAULT_CANDIDATE, candidate_hash,
)

def test_default_candidate_freezes_methodology():
    c = DEFAULT_CANDIDATE
    # Pre-committed methodology (rail #4) — these are the falsifiable choices.
    assert c.families == ("value", "momentum")
    assert c.value_skip == 126            # exclude the 6mo momentum window
    assert c.value_lookback == 270        # fixture-feasible intermediate reversal (see note)
    # Feasibility ceiling (rail #5): value must be live in EVERY dev fold. On the
    # 2015-2023 fixture, dev≈1654, fold_size≈330, fold-1 train ends ~325, so a
    # value_lookback >= ~325 leaves fold 1 with an all-NaN train -> dead leg ->
    # dev gate unwinnable for non-signal reasons. Keep margin for the percentile fit.
    assert c.value_lookback <= 300
    assert 0.0 < c.orthogonality_tau <= 0.5
    assert c.declared_trials_N >= 45      # inherits the conservative MinBTL budget
    # Inherited floor gate params must match the floor so the bench is faithful.
    assert c.folds == 5 and c.embargo == 5 and c.margin == 0.0
    assert c.final_lcb == 0.95 and c.dev_lcb == 0.90

def test_candidate_hash_is_stable_and_sensitive():
    import dataclasses
    h0 = candidate_hash(DEFAULT_CANDIDATE)
    assert isinstance(h0, str) and len(h0) == 64
    # Any methodology change must re-hash (no silent p-hacking).
    mutated = dataclasses.replace(DEFAULT_CANDIDATE, value_lookback=1260)
    assert candidate_hash(mutated) != h0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_prereg.py -q`
Expected: FAIL with `ModuleNotFoundError: advisor.research.candidate_prereg`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/quant/advisor/research/candidate_prereg.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CandidatePreReg:
    """Immutable, pre-registered methodology for the candidate bench. A NEW hash
    surface — deliberately NOT PreRegConfig (whose hash 1ad2ed4a… is floor-frozen).
    Outcome thresholds live in the floor gates; this freezes only the search."""
    # --- candidate-specific (the falsifiable choices) ---
    families: tuple[str, ...] = ("value", "momentum")
    value_skip: int = 126                 # skip recent 6mo -> excludes momentum window
    value_lookback: int = 270             # ~13mo->6mo formation; FIXTURE-FEASIBLE (< ~325)
    orthogonality_tau: float = 0.40       # max |Pearson corr| of value vs LM / MR on dev
    declared_trials_N: int = 45           # DSR multiple-testing N (>= MinBTL budget 45)
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


DEFAULT_CANDIDATE = CandidatePreReg()


def candidate_hash(cfg: CandidatePreReg) -> str:
    """SHA-256 over the canonical config JSON (no fixture bytes — the bench reuses
    the floor's fixture, whose hash is recorded separately in CANDIDATE_PREREG.md)."""
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_prereg.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/__init__.py apps/quant/advisor/research/candidate_prereg.py apps/quant/advisor/tests/test_candidate_prereg.py
git commit -m "feat(research): CandidatePreReg immutable methodology surface for Lane B bench"
```

---

### Task 2: `candidate_raw` — the `value` (long-term-reversal) signal

**Construction & the relabel risk** (rail #4, advisor; see the horizon note above):
- `momentum` = `p/p.shift(126)-1` (6mo); `long_momentum` = `p/p.shift(252)-1` (12mo) — **recent** trend.
- `mean_reversion` = `(SMA10 - p)/p` — **short-horizon** (10-day) snap-back.
- `value` (this task) = `-(p.shift(126)/p.shift(270) - 1)` — the **negated** return over the ~13mo→6mo *formation* window. Skipping the most recent 126 days (`shift(126)`) excludes the momentum window, so `value` is bullish on intermediate-term *losers*. Because the fixture forces a feasible (sub-325) lookback, this horizon sits **near `long_momentum`** — so this is genuinely at risk of being negated-momentum in disguise. That is exactly why the **orthogonality gate (Task 6) is the pivot**: if `value` is empirically correlated with `long_momentum`/`mean_reversion`, it is the rejected factor relabeled and the plan stops there. Classic 36–60mo LT-reversal is infeasible here and moves to Reading B (Task 11).

**Files:**
- Create: `apps/quant/advisor/research/candidate_signals.py`
- Test: `apps/quant/advisor/tests/test_candidate_signals.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_candidate_signals.py
import numpy as np
import pandas as pd

from advisor.research.candidate_signals import VALUE, candidate_raw

def _prices(n=900, drift=0.0):
    return pd.Series(100.0 * np.exp(np.cumsum(np.full(n, drift))))

def _declining_then_flat(n_decline=374, n_flat=126, drift=-0.003):
    # ONE continuous series (no concat discontinuity): falls across the formation
    # window, then flat inside the recent skip window.
    dec = 100.0 * np.exp(np.cumsum(np.full(n_decline, drift)))
    flat = np.full(n_flat, dec[-1])
    return pd.Series(np.concatenate([dec, flat]))

def test_value_is_bullish_for_intermediate_term_losers():
    # Falls across [t-270, t-126], flat in the last 126 -> formation return < 0 -> value > 0.
    p = _declining_then_flat()
    v = candidate_raw(VALUE, p, value_skip=126, value_lookback=270)
    assert v.dropna().iloc[-1] > 0   # past loser -> bullish reversal

def test_value_excludes_recent_window_via_skip():
    # value at t depends on p[t-skip] and p[t-lookback], never on p[t-1..t-skip+1].
    p = _prices(900)
    base = candidate_raw(VALUE, p, value_skip=126, value_lookback=270)
    spiked = p.copy()
    spiked.iloc[-50:] *= 2.0          # perturb only the last 50 days (inside the skip)
    after = candidate_raw(VALUE, spiked, value_skip=126, value_lookback=270)
    assert np.isclose(base.iloc[-1], after.iloc[-1])   # last 50 days don't enter value(t)

def test_known_families_delegate_to_frozen_raw():
    from advisor.backtest.continuous_signals import raw_metric
    p = _prices(900)
    for fam in ("momentum", "trend", "long_momentum", "mean_reversion", "breakout"):
        pd.testing.assert_series_equal(candidate_raw(fam, p), raw_metric(fam, p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_signals.py -q`
Expected: FAIL with `ModuleNotFoundError: advisor.research.candidate_signals`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/quant/advisor/research/candidate_signals.py
from __future__ import annotations

import pandas as pd

from advisor.backtest.continuous_signals import raw_metric as _frozen_raw

VALUE = "value"


def candidate_raw(family: str, prices: pd.Series, *, value_skip: int = 126,
                  value_lookback: int = 270) -> pd.Series:
    """Long-flat RAW strength (sign carries direction; transform clamps <=0 to flat).
    Known price families delegate to the frozen floor metric (read-only). The new
    'value' family is intermediate-term reversal: the NEGATIVE of the formation-window
    return from `value_lookback` ago to `value_skip` ago. Skipping the recent
    `value_skip` days excludes the momentum window. Whether this is decorrelated from
    momentum is the Task-6 kill-gate, NOT an assumption. NaN for the first
    `value_lookback` rows (handled downstream as flat); keep `value_lookback` below
    fold-1's train end so no dev fold is dead (see plan horizon note)."""
    if family == VALUE:
        p = pd.Series(prices).astype(float)
        formation = p.shift(value_skip) / p.shift(value_lookback) - 1.0
        return -formation
    return _frozen_raw(family, prices)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_signals.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/candidate_signals.py apps/quant/advisor/tests/test_candidate_signals.py
git commit -m "feat(research): value (long-term reversal) raw signal; known families delegate to frozen raw_metric"
```

---

### Task 3: `candidate_pipeline` — frozen sweep/holdout mirrored with an injected `raw_fn`

**Files:**
- Create: `apps/quant/advisor/research/candidate_pipeline.py`
- Test: `apps/quant/advisor/tests/test_candidate_pipeline.py`

**Note:** this is a faithful mirror of `backtest/pipeline.py` (`run_dev_sweep`/`run_holdout`), changing exactly one thing — `raw_metric(f, …)` becomes `raw_fn(f, …)`. Every other primitive is imported from the frozen modules. The faithfulness is *proven* in Task 4, not asserted here; this task's test only checks the injection plumbing on a synthetic panel.

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_candidate_pipeline.py
import numpy as np
import pandas as pd

from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.continuous_signals import raw_metric
from advisor.research.candidate_pipeline import run_dev_sweep_ext, SweepResultExt

def _panel(n=900, k=12, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)

def test_run_dev_sweep_ext_with_frozen_raw_matches_shape_and_runs():
    panel = _panel()
    cfg = PreRegConfig()
    res = run_dev_sweep_ext(panel, ("momentum", "trend"), cfg,
                            raw_fn=raw_metric, holdout_frac=0.2)
    assert isinstance(res, SweepResultExt)
    assert len(res.fold_deltas) >= 1
    assert isinstance(res.ensemble_test_returns, pd.Series)
    assert set(res.chosen_weights) == {"momentum", "trend"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: advisor.research.candidate_pipeline`

- [ ] **Step 3: Write minimal implementation** (mirror of frozen `pipeline.py`; only `raw_fn` differs)

```python
# apps/quant/advisor/research/candidate_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from advisor.backtest.blend import select_weights
from advisor.backtest.book import book_returns
from advisor.backtest.continuous_signals import apply_transform, fit_percentile_transform
from advisor.backtest.portfolio import build_long_flat_book
from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import book_sharpe

RawFn = Callable[[str, pd.Series], pd.Series]


@dataclass(frozen=True)
class SweepResultExt:
    fold_deltas: list[float]
    ensemble_test_returns: pd.Series
    best_family_test_returns: pd.Series
    chosen_weights: dict


@dataclass(frozen=True)
class HoldoutReturnsExt:
    ensemble: pd.Series
    best_family: pd.Series
    spy: pd.Series


def _family_scores(raw_fn: RawFn, family: str, prices: pd.DataFrame,
                   train_idx, all_idx, clip) -> pd.DataFrame:
    cols = {}
    for c in prices.columns:
        raw = raw_fn(family, prices[c])
        params = fit_percentile_transform(raw.iloc[train_idx], clip=clip)
        cols[c] = apply_transform(params, raw)
    return pd.DataFrame(cols).iloc[all_idx].reset_index(drop=True)


def run_dev_sweep_ext(panel: pd.DataFrame, families: tuple, cfg: PreRegConfig,
                      raw_fn: RawFn, holdout_frac: float = 0.2) -> SweepResultExt:
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    dev = prices_all.iloc[:dev_end]

    caps = (cfg.max_asset_weight, cfg.gross_cap, cfg.turnover_cap)
    deltas, ens_parts, best_parts = [], [], []
    chosen = {f: 1.0 / len(families) for f in families}
    for train_idx, test_idx in purged_splits(len(dev), cfg.folds, cfg.embargo):
        all_idx = list(range(min(train_idx), max(test_idx) + 1))
        scores = {f: _family_scores(raw_fn, f, dev, train_idx, list(all_idx), cfg.pct_clip)
                  for f in families}
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

        fam_sharpes, fam_rets = {}, {}
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

    full_idx = list(range(len(dev)))
    full_scores = {f: _family_scores(raw_fn, f, dev, full_idx, full_idx, cfg.pct_clip)
                   for f in families}
    frozen = select_weights(full_scores, dev.reset_index(drop=True), families,
                            cfg.weight_grid, cfg.train_lift_threshold,
                            cfg.cost_per_turn, caps) if full_idx else chosen

    return SweepResultExt(
        fold_deltas=deltas,
        ensemble_test_returns=pd.concat(ens_parts, ignore_index=True) if ens_parts else pd.Series(dtype=float),
        best_family_test_returns=pd.concat(best_parts, ignore_index=True) if best_parts else pd.Series(dtype=float),
        chosen_weights=frozen,
    )


def run_holdout_ext(panel: pd.DataFrame, families: tuple, cfg: PreRegConfig,
                    frozen_weights: dict, raw_fn: RawFn,
                    holdout_frac: float = 0.2) -> HoldoutReturnsExt:
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[cfg.warmup:].reset_index(drop=True)
    spy_all = panel["SPY"].iloc[cfg.warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    caps = (cfg.max_asset_weight, cfg.gross_cap, cfg.turnover_cap)

    scores = {}
    for f in families:
        cols = {}
        for c in assets:
            raw = raw_fn(f, prices_all[c])
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
    return HoldoutReturnsExt(ensemble=ens_r, best_family=fam_rets[best_f], spy=spy_r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_pipeline.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/candidate_pipeline.py apps/quant/advisor/tests/test_candidate_pipeline.py
git commit -m "feat(research): candidate_pipeline mirrors frozen sweep/holdout with injected raw_fn"
```

---

### Task 4: Golden-replication gate — the bench MUST reproduce the floor's own numbers

**This is the trust anchor (advisor).** Before the bench is allowed to judge any new signal, prove it reproduces the frozen floor's published Construction-C result on the real fixture. If it does not replicate, the mirror has drifted — STOP and fix Task 3 before proceeding. Floor's published C numbers (`FLOOR_RESULT.md`): ensemble book-Sharpe **0.732**, best standalone **0.828**, all four fold deltas negative, DEV_FAILED.

**Files:**
- Test: `apps/quant/advisor/tests/test_candidate_golden_replication.py`
- Read-only: `apps/quant/advisor/backtest/fixtures/floor_prices.csv` (locate via the floor's own loader — see Step 1)

- [ ] **Step 1: Find how the floor loads its fixture into a `panel`**

Run: `python -m pytest apps/quant/advisor/backtest -q -k floor` then inspect the floor test that builds `panel` from `floor_prices.csv` (grep `floor_prices` under `apps/quant/advisor/`). Reuse that exact loader in the test (do not re-invent column handling). Expected: a `pd.DataFrame` with a `SPY` column + 30 asset columns.

- [ ] **Step 2: Write the failing test**

```python
# apps/quant/advisor/tests/test_candidate_golden_replication.py
import pandas as pd
import pytest

from advisor.backtest.continuous_signals import raw_metric
from advisor.backtest.prereg import PreRegConfig
from advisor.backtest.stats import book_sharpe
from advisor.backtest.pipeline import run_dev_sweep            # frozen (Amendment F3)
from advisor.research.candidate_pipeline import run_dev_sweep_ext

# Concrete loader (Amendment F7 — no placeholder; the floor's fixture is script-local):
from pathlib import Path
def load_floor_panel():
    return pd.read_csv(Path("apps/quant/advisor/tests/fixtures/floor_prices.csv"),
                       index_col=0, parse_dates=True)

def test_ext_pipeline_equals_frozen_pipeline_elementwise():
    # Amendment F3: the REAL trust anchor. With raw_fn=raw_metric the mirror must be
    # bit-for-bit the frozen pipeline on shared families -> any drift in EITHER goes red.
    # Dev-only (run_dev_sweep never touches the reserved tail) -> holdout stays blinded.
    panel = load_floor_panel()
    cfg = PreRegConfig()
    ext = run_dev_sweep_ext(panel, ("momentum", "trend"), cfg, raw_fn=raw_metric, holdout_frac=0.2)
    ref = run_dev_sweep(panel, ("momentum", "trend"), cfg, holdout_frac=0.2)
    assert ext.fold_deltas == ref.fold_deltas
    pd.testing.assert_series_equal(ext.ensemble_test_returns, ref.ensemble_test_returns)
    pd.testing.assert_series_equal(ext.best_family_test_returns, ref.best_family_test_returns)
    assert ext.chosen_weights == ref.chosen_weights

def test_bench_reproduces_floor_construction_C():
    # Documentation check of the published numbers (no longer the trust anchor).
    panel = load_floor_panel()                  # the real 2015-2023 fixture
    cfg = PreRegConfig()                         # floor's own config (warmup=200)
    res = run_dev_sweep_ext(panel, ("momentum", "trend"), cfg,
                            raw_fn=raw_metric, holdout_frac=0.2)
    ens = book_sharpe(res.ensemble_test_returns)
    best = book_sharpe(res.best_family_test_returns)
    assert ens == pytest.approx(0.732, abs=0.01)     # FLOOR_RESULT.md C ensemble
    assert best == pytest.approx(0.828, abs=0.01)    # FLOOR_RESULT.md C best family
    assert all(d < 0 for d in res.fold_deltas)       # every fold delta negative

def test_value_leg_is_live_in_every_dev_fold():
    # Feasibility guard (rail #5): the pre-registered value_lookback must leave a
    # non-degenerate positive-value training set in EVERY dev fold on the real fixture,
    # else the dev gate is rigged to fail for non-signal reasons.
    from advisor.backtest.splits import purged_splits
    from advisor.research.candidate_prereg import DEFAULT_CANDIDATE as C
    from advisor.research.candidate_signals import VALUE, candidate_raw
    panel = <load_panel>()
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[C.warmup:].reset_index(drop=True)
    dev = prices_all.iloc[:int(len(prices_all) * 0.8)]
    val = candidate_raw(VALUE, dev[assets[0]], value_skip=C.value_skip,
                        value_lookback=C.value_lookback)
    for train_idx, _ in purged_splits(len(dev), C.folds, C.embargo):
        live_pos = val.iloc[train_idx].dropna()
        assert (live_pos > 0).sum() >= 10   # non-degenerate percentile fit per fold
```

- [ ] **Step 3: Run the test**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_golden_replication.py -q`
Expected: PASS. **If it fails on the numbers** (not import), the mirror in Task 3 has drifted from `backtest/pipeline.py` — diff the two line-by-line and reconcile before any later task. Do NOT loosen the tolerance to force a pass.

- [ ] **Step 4: Commit**

```bash
git add apps/quant/advisor/tests/test_candidate_golden_replication.py
git commit -m "test(research): golden-replication — bench reproduces floor Construction-C (0.732/0.828, DEV_FAILED)"
```

---

## Phase B1 — The candidate signal + the orthogonality kill-gate (the pivot)

### Task 5: `orthogonality` — dev-fold raw correlation of `value` vs rejected neighbors

**Files:**
- Create: `apps/quant/advisor/research/orthogonality.py`
- Test: `apps/quant/advisor/tests/test_orthogonality.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/quant/advisor/tests/test_orthogonality.py
import numpy as np
import pandas as pd

from advisor.research.orthogonality import dev_fold_raw_corr

def _panel(n=900, k=12, seed=1):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)

def test_returns_corr_against_each_neighbor_on_dev_only():
    panel = _panel()
    corr = dev_fold_raw_corr(panel, warmup=200, holdout_frac=0.2,
                             value_skip=126, value_lookback=400,
                             neighbors=("long_momentum", "mean_reversion"))
    assert set(corr) == {"long_momentum", "mean_reversion"}
    assert all(-1.0 <= v <= 1.0 for v in corr.values())

def test_perfect_relabel_is_detected():
    # If "value" were literally -long_momentum, |corr| ~ 1 -> must be caught.
    panel = _panel()
    corr = dev_fold_raw_corr(panel, warmup=200, holdout_frac=0.2,
                             value_skip=0, value_lookback=252,           # == -long_momentum
                             neighbors=("long_momentum",))
    assert abs(corr["long_momentum"]) > 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_orthogonality.py -q`
Expected: FAIL with `ModuleNotFoundError: advisor.research.orthogonality`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/quant/advisor/research/orthogonality.py
from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.backtest.splits import purged_splits
from advisor.research.candidate_signals import VALUE, candidate_raw


def dev_fold_raw_corr(panel: pd.DataFrame, *, warmup: int, holdout_frac: float,
                      value_skip: int, value_lookback: int,
                      neighbors: tuple[str, ...],
                      folds: int = 5, embargo: int = 5) -> dict[str, float]:
    """Pearson corr of `value` raw vs each neighbor raw, pooled over the dev test
    folds and all assets (holdout strictly excluded). NaN rows dropped pairwise."""
    assets = [c for c in panel.columns if c != "SPY"]
    prices_all = panel[assets].iloc[warmup:].reset_index(drop=True)
    dev_end = int(len(prices_all) * (1 - holdout_frac))
    dev = prices_all.iloc[:dev_end]

    test_rows = sorted({i for _, te in purged_splits(len(dev), folds, embargo) for i in te})
    out: dict[str, float] = {}
    for nb in neighbors:
        vv, nn = [], []
        for c in assets:
            v = candidate_raw(VALUE, dev[c], value_skip=value_skip,
                              value_lookback=value_lookback).iloc[test_rows]
            n = candidate_raw(nb, dev[c]).iloc[test_rows]
            df = pd.concat([v, n], axis=1).dropna()
            vv.append(df.iloc[:, 0].to_numpy())
            nn.append(df.iloc[:, 1].to_numpy())
        a, b = np.concatenate(vv), np.concatenate(nn)
        out[nb] = 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_orthogonality.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/orthogonality.py apps/quant/advisor/tests/test_orthogonality.py
git commit -m "feat(research): dev-fold raw-correlation diagnostic for the value orthogonality gate"
```

---

### Task 6: Run the orthogonality kill-gate on the real fixture (MEASUREMENT — the pivot)

**Not a pass/fail unit test — a recorded measurement that decides whether Phase B2 runs at all** (rail #4/#5; advisor: "elevate it to the gate the whole plan pivots on"). Claude-direct (operator-class), not Hermes.

- [ ] **Step 1: Run the gate on the real fixture, dev folds only**

```powershell
$env:PYTHONUTF8=1; $env:PYTHONPATH="apps/quant"
python -c "from advisor.research.orthogonality import dev_fold_raw_corr; from advisor.backtest.<floor_fixture_module> import <load_panel> as L; from advisor.research.candidate_prereg import DEFAULT_CANDIDATE as C; print(dev_fold_raw_corr(L(), warmup=C.warmup, holdout_frac=0.2, value_skip=C.value_skip, value_lookback=C.value_lookback, neighbors=('momentum','long_momentum','mean_reversion'), folds=C.folds, embargo=C.embargo))"
```

- [ ] **Step 2: Apply the pre-registered decision rule**

- **GATE NEIGHBORS** = `long_momentum`, `mean_reversion` (the rejected-factor relabel check). `momentum` is reported as a **diagnostic only** — it is `value`'s blend partner, so its correlation is the most decision-relevant number to *see* pre-holdout (the AQR thesis is value⊥momentum), but §7.2 ultimately adjudicates the blend; do not gate on it. **(Amendment F4)** Run the gate on the **post-transform fold-level** correlation (what blends), not only raw Pearson; report Spearman as a scale-robust cross-check; a high post-transform `|corr(value, momentum)|` demotes the gate to a coarse pre-filter (§7.2 adjudicates), it is not a clean PASS.
- **PASS (proceed to B2):** `max(|corr(value, long_momentum)|, |corr(value, mean_reversion)|) < orthogonality_tau (0.40)`. The value signal is empirically decorrelated from the rejected neighbors → the diversification hypothesis is live. Record the `momentum` diagnostic alongside.
- **FAIL (STOP — cheap negative):** if either gated |corr| ≥ 0.40, `value` is a relabel of an already-rejected factor (plan4). Record the negative in `CANDIDATE_RESULT.md`, skip B2 entirely, and jump to Task 11 (Reading B). This is the *intended* cheap-falsification outcome and a valid deliverable.

- [ ] **Step 3: Record the measurement (no secrets) and commit the doc**

Write to `apps/quant/advisor/research/CANDIDATE_RESULT.md`: the two correlations, the tau, the PASS/FAIL verdict, the `candidate_hash`, and the fixture SHA. Commit docs only:

```bash
git add apps/quant/advisor/research/CANDIDATE_RESULT.md
git commit -m "docs(research): orthogonality kill-gate measurement (value vs long_momentum/mean_reversion)"
```

---

## Phase B2 — Pre-registered evaluation (only if Task 6 PASSED)

### Task 7: `candidate_metrics` — mirror `floor_metrics` over the candidate pipeline

**Files:**
- Create: `apps/quant/advisor/research/candidate_floor.py`
- Test: `apps/quant/advisor/tests/test_candidate_floor.py`

**Note:** faithful mirror of `backtest/data_floor.py::floor_metrics`, swapping `run_dev_sweep`/`run_holdout` → the `_ext` variants (with `raw_fn` bound to the pre-registered value horizons) and `cfg` → `CandidatePreReg`. Verdict/holdout/§7.1/§7.2/validation logic is identical, so a passing candidate clears the *same* bar the floor sets.

- [ ] **Step 1: Write the failing test** (synthetic panel; checks plumbing + report-only invariant)

```python
# apps/quant/advisor/tests/test_candidate_floor.py
import numpy as np
import pandas as pd

from advisor.research.candidate_prereg import DEFAULT_CANDIDATE
from advisor.research.candidate_floor import candidate_metrics

def _panel(n=1400, k=24, seed=2):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)

def test_candidate_metrics_reports_full_floor_schema():
    m = candidate_metrics(_panel(), DEFAULT_CANDIDATE, prereg_hash="deadbeef")
    for key in ("verdict", "universe", "dev", "weights", "holdout",
                "ensemble", "spy", "best_family", "margin", "passes", "validation"):
        assert key in m
    assert m["verdict"] in {"DEV_FAILED", "UNSUPPORTED", "INCONCLUSIVE", "PASSED"}

def test_holdout_blinded_until_dev_passes():
    # With prereg_hash=None the holdout is never touched (leakage guard, rail #5).
    m = candidate_metrics(_panel(), DEFAULT_CANDIDATE, prereg_hash=None)
    assert m["holdout"] is None
    assert m["passes"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_floor.py -q`
Expected: FAIL with `ModuleNotFoundError: advisor.research.candidate_floor`

- [ ] **Step 3: Write minimal implementation** (mirror of `floor_metrics`)

```python
# apps/quant/advisor/research/candidate_floor.py
from __future__ import annotations

import pandas as pd

from advisor.backtest.dev_gate import dev_gate
from advisor.backtest.splits import purged_splits
from advisor.backtest.stats import block_bootstrap_diff_lcb, book_sharpe
from advisor.backtest.universe import classify_universe
from advisor.backtest.validation import validation_report
from advisor.research.candidate_validation_prereg import DEFAULT_CANDIDATE_VALIDATION  # Amendment F1
from advisor.research.candidate_prereg import CandidatePreReg
from advisor.research.candidate_pipeline import run_dev_sweep_ext, run_holdout_ext
from advisor.research.candidate_signals import candidate_raw


def candidate_metrics(panel: pd.DataFrame, cfg: CandidatePreReg,
                      prereg_hash: str | None = None, holdout_frac: float = 0.2) -> dict:
    """Candidate bench mirror of floor_metrics: dev gate first, one blinded holdout.
    Report-only; never authorizes sizing; frozen floor untouched."""
    families = cfg.families

    def raw_fn(family: str, prices: pd.Series) -> pd.Series:
        return candidate_raw(family, prices, value_skip=cfg.value_skip,
                             value_lookback=cfg.value_lookback)

    assets = panel.drop(columns=["SPY"]).iloc[cfg.warmup:].reset_index(drop=True)
    sweep = run_dev_sweep_ext(panel, families, cfg, raw_fn=raw_fn, holdout_frac=holdout_frac)

    dev_end = int(len(assets) * (1 - holdout_frac))
    n_active = [int(assets.iloc[te].notna().all(axis=0).sum())
                for _, te in purged_splits(dev_end, cfg.folds, cfg.embargo)] or [assets.shape[1]]
    universe = classify_universe(n_active, cfg)
    gate = dev_gate(sweep.fold_deltas, sweep.ensemble_test_returns,
                    sweep.best_family_test_returns, cfg)

    holdout = None
    verdict = "DEV_FAILED"
    legacy_spy = book_sharpe(panel["SPY"].iloc[cfg.warmup:].pct_change().fillna(0.0))
    if universe == "do_not_run":
        verdict = "UNSUPPORTED"
    elif gate.passed and prereg_hash is not None:
        h = run_holdout_ext(panel, families, cfg, sweep.chosen_weights,
                            raw_fn=raw_fn, holdout_frac=holdout_frac)
        delta_lcb = block_bootstrap_diff_lcb(h.ensemble, h.best_family, cfg.bootstrap_block,
                                             cfg.bootstrap_draws, cfg.bootstrap_seed,
                                             level=cfg.final_lcb)
        spy_lcb = block_bootstrap_diff_lcb(h.ensemble, h.spy, cfg.bootstrap_block,
                                           cfg.bootstrap_draws, cfg.bootstrap_seed,
                                           level=cfg.final_lcb)
        beats_parts = delta_lcb > 0
        beats_spy = spy_lcb > cfg.margin
        holdout = {
            "delta_lcb": delta_lcb, "spy_lcb": spy_lcb,
            "beats_parts": beats_parts, "beats_spy": beats_spy,
            "ensemble_sharpe": book_sharpe(h.ensemble), "spy_sharpe": book_sharpe(h.spy),
            "best_family_sharpe": book_sharpe(h.best_family),
            "label": "diagnostic" if universe == "micro" else "formal",
        }
        legacy_spy = book_sharpe(h.spy)
        verdict = "PASSED" if (beats_parts and beats_spy) else "INCONCLUSIVE"

    validation = validation_report(
        sweep.ensemble_test_returns,
        {"ensemble": sweep.ensemble_test_returns,
         "best_family": sweep.best_family_test_returns},
        DEFAULT_CANDIDATE_VALIDATION,   # Amendment F1: candidate's own N/var_sr, not the floor's
    )
    return {
        "verdict": verdict, "universe": universe,
        "dev": {"passed": gate.passed, "reasons": gate.reasons, "fold_deltas": sweep.fold_deltas},
        "weights": sweep.chosen_weights, "holdout": holdout,
        "ensemble": book_sharpe(sweep.ensemble_test_returns), "spy": legacy_spy,
        "best_family": book_sharpe(sweep.best_family_test_returns),
        "margin": float(cfg.margin), "passes": verdict == "PASSED",
        "validation": validation,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_candidate_floor.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/research/candidate_floor.py apps/quant/advisor/tests/test_candidate_floor.py
git commit -m "feat(research): candidate_metrics mirrors floor_metrics over the value+momentum bench"
```

---

### Task 8: Pre-register the run, then evaluate on the real fixture (MEASUREMENT)

**Claude-direct.** First freeze the pre-registration doc (so the run cannot be re-specified after seeing results), THEN run.

- [ ] **Step 1: Write & commit `CANDIDATE_PREREG.md` BEFORE running**

Contents: `candidate_hash(DEFAULT_CANDIDATE)` (methodology id), `candidate_run_hash(DEFAULT_CANDIDATE, FIXTURE)` (config+fixture — the holdout-unlock key, Amendment F2), the `CandidateValidationPreReg` hash (Amendment F1), the frozen constants (families/order `("value","momentum")`, `value_skip=126`, `value_lookback=270`, `orthogonality_tau=0.40`; `declared_trials_N=45` **on `CandidateValidationPreReg`**), the fixture path + SHA, the candidate order (primary: `value+momentum`; pre-registered secondary ONLY if primary dev-passes-but-holdout-fails: `value+momentum+trend`, which increments effective N), and the acceptance bar (the floor's gates verbatim — no new thresholds). Commit before Step 2.

```bash
git add apps/quant/advisor/research/CANDIDATE_PREREG.md
git commit -m "docs(research): pre-register the value+momentum candidate (hash, frozen horizons, acceptance=floor gates)"
```

- [ ] **Step 2: Run the pre-registered evaluation**

```powershell
$env:PYTHONUTF8=1; $env:PYTHONPATH="apps/quant"
python -c "from advisor.research.candidate_floor import candidate_metrics; from advisor.research.candidate_prereg import DEFAULT_CANDIDATE as C, candidate_hash; from advisor.backtest.<floor_fixture_module> import <load_panel> as L; import json; print(json.dumps(candidate_metrics(L(), C, prereg_hash=candidate_hash(C)), default=str, indent=2))"
```

- [ ] **Step 3: Interpret against the pre-registered bar (no post-hoc threshold changes)**

- `verdict == "PASSED"` (dev gate passed AND holdout `beats_parts` AND `beats_spy`) → **a real candidate exists.** Additionally confirm `validation["passes"]` (DSR ≥ 0.95 at N=45) for promotion-readiness. Proceed to Task 9 (promotion path).
- `verdict in {"DEV_FAILED","INCONCLUSIVE","UNSUPPORTED"}` → **clean negative.** Record it. The holdout was either never touched (DEV_FAILED) or touched once and failed (INCONCLUSIVE). Do NOT retry horizons ad hoc — the only permitted second run is the pre-registered `value+momentum+trend`, and it increments `declared_trials_N` (re-run DSR at the higher N). After the pre-registered set is exhausted, Reading A is concluded.
- **Scope caveat (record it):** the feasibility fix (Task 1, `value_lookback=270`, verified live in every dev fold by Task 4's guard) means a `DEV_FAILED`/`INCONCLUSIVE` here is a **genuine signal verdict, not a rigged one**. But it tests *intermediate-term* reversal only — a negative does **not** refute classic 36–60mo LT-reversal or fundamental value, both of which are fixture-infeasible here and move to Reading B (Task 11). In practice the cheap kill happens earlier at Task 6 (orthogonality); reaching a clean dev/holdout verdict at all means `value` was decorrelated yet still didn't beat the parts.

- [ ] **Step 4: Append results to `CANDIDATE_RESULT.md` and commit**

```bash
git add apps/quant/advisor/research/CANDIDATE_RESULT.md
git commit -m "docs(research): value+momentum candidate evaluation result (verdict + holdout status + DSR)"
```

---

## Phase B3 — Decision, promotion path, and the reading-fork

### Task 9: Decision record (NO promotion in this plan)

**Claude-direct doc.** Write the decision into `CANDIDATE_RESULT.md` `## Decision`:

- [ ] **If a candidate PASSED dev+holdout+DSR:** record that Plans 1b and 3 are now **unblocked-by-candidate**. State explicitly that promotion is a *separate, deliberate* effort requiring its own plan + operator sign-off: (1) Plan 1b wires `validation["passes"]` into `--enforce` (report-only → blocking); (2) a new floor pre-registration adopts the value family (re-hash, new `PREREG.md` — a conscious replacement of the accepted DEV_FAILED negative); (3) only then is `ensemble_vote`/`skill_weight` promotion considered (deep-research: skill_weight is viable *only after* an orthogonal signal exists). None of this happens in Lane B.
- [ ] **If the candidate FAILED:** record the clean negative; Reading A is exhausted; the frozen floor stays DEV_FAILED, `--enforce` stays exit 1, gates unchanged. Recommend Reading B (Task 11) as the higher-conviction follow-on.
- [ ] **Either way — verify rails held:** `python -m pytest apps/quant/advisor/tests -q` (count risen, all green); `npm run advisor-gate` (exit 0); `node tools/run-floor.mjs --enforce` (**exit 1**); `git diff --stat <base>..HEAD -- apps/quant/advisor/backtest apps/quant/advisor/portfolio apps/quant/advisor/risk` (zero). Commit the decision doc.

### Task 10 (OPTIONAL, YAGNI-gated): PBO-via-CSCV selection audit

Deep-research verdict: PBO/CSCV is **audit-only, medium-confidence, synthetic-validated**; explicitly deferred in the validation-gate slice. Build it **only if** a candidate PASSES and a reviewer wants an overfitting audit of the selection process before promotion. If built: `research/pbo.py::pbo_cscv(returns_matrix, n_splits)` returning the probability of backtest overfitting; AUDIT a passing candidate, never gate on it. Skip otherwise — do not build speculatively.

### Task 11 (THE READING-FORK — explicit decision, gated on operator): Reading B = fundamental value with a timely-price leg

**This is the memory's *actual* lead** (`deep-research-orthogonal-signals`: "make the existing value family use TIMELY prices"). The live `value_quality` family uses book value with 90-day-lagged fundamentals; refreshing only the *price* leg (market cap from a timely price) yields a **fundamental** value signal genuinely orthogonal to price-momentum — a stronger candidate than Reading A's pure-price reversal.

**Why it is NOT in this plan's build scope:** `floor_prices.csv` is price-only. Reading B requires a **fundamentals-bearing fixture** (point-in-time book value / shares, as-of bounded with the existing `REPORTING_LAG_DAYS=90` guard) + its own `CandidatePreReg` extension + its own pre-registration. That is new data, materially heavier, and gated on an operator decision.

- [ ] **Deliverable of this task (planning only):** a short spec stub `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` capturing: the fundamentals fixture requirement (source, point-in-time discipline, as-of assertion), the value-with-timely-price construction, reuse of this plan's bench (the bench is signal-agnostic — only `candidate_raw` + the fixture change), and acceptance = the same floor gates. Do NOT build the fixture or signal until the operator greenlights the data work. If Reading A failed for *power* reasons (Task 8 caveat), Reading B is the recommended next investment.

---

## Self-review

**Spec coverage** (against the design + advisor strengtheners):
- Rail-safe separate bench reusing frozen primitives → Tasks 1–3, 7. ✓
- Golden-replication trust anchor → Task 4. ✓
- Orthogonality as the *pivot* kill-gate → Tasks 5–6 (gates `long_momentum`/`mean_reversion`; `momentum` reported as diagnostic). ✓
- Fixture-feasible horizon (no rigged DEV_FAILED) → Design-decision note + `value_lookback=270` + feasibility ceiling test (Task 1) + real-fixture live-in-every-fold guard (Task 4). ✓
- Holdout-leakage guard (dev-only construction, holdout once iff dev passes) → built into Tasks 3/7, asserted in `test_holdout_blinded_until_dev_passes`. ✓
- Pre-committed methodology, outcomes by floor gates, N counts trials → Tasks 1, 8. ✓
- Reading-fork explicit decision → Task 11. ✓
- Promotion deferred (1b/3), frozen floor + `--enforce` exit 1 untouched → Tasks 9, rails. ✓

**Placeholder scan:** the only intentional placeholders are `<floor_fixture_module>` / `<load_panel>` (resolved in Task 4 Step 1 by locating the floor's real loader) — every executing task points to that resolution. No `TBD`/`add error handling`/`similar to Task N`.

**Type consistency:** `SweepResultExt`/`HoldoutReturnsExt` field names match `candidate_metrics`'s reads (`fold_deltas`, `ensemble_test_returns`, `best_family_test_returns`, `chosen_weights`; `ensemble`/`best_family`/`spy`). `candidate_raw(family, prices, *, value_skip, value_lookback)` signature matches the `raw_fn` closure in Tasks 6/7. `CandidatePreReg` exposes every attribute the mirrored pipeline reads from `cfg`.

**Scope:** one coherent deliverable (a proven-or-disproven Reading-A candidate + bench). Reading B is correctly split into its own future spec (Task 11), not built here.

---

## Execution handoff

Plan complete. Recommended path: **subagent-driven-development** for Tasks 1–5 and 7 (TDD code, one Hermes dispatch + commit each), with Tasks 4, 6, 8, 9 run **Claude-direct** (golden-replication + measurements + decision — operator-class, not Hermes). Task 6 is a hard gate: if orthogonality fails, skip Phase B2 and go to Task 11. Tasks 10–11 are gated/optional.
