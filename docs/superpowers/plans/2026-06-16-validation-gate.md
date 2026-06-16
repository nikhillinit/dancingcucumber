# Validation Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a signal-agnostic deflation / multiple-testing validation gate to the backtest harness as report-only diagnostics that can only confirm `DEV_FAILED` harder — never green-wash, never unlock the holdout, never authorize production sizing.

**Architecture:** A new immutable `ValidationPreReg` (its own hash surface — **not** new fields on the frozen `PreRegConfig`, which would invalidate the recorded floor hash `1ad2ed4a…`). A dependency-light `validation.py` of pure functions (per-observation Sharpe + moments, PSR, DSR, MinBTL, HLZ t-stat, PCA effective-N over family return series). Diagnostics are appended to `floor_metrics()`'s report dict under a new `"validation"` key; the `verdict` branch is untouched. The dominant multiple-testing count `N` is a pre-registered integer (the MinBTL budget, 45 — conservative); PCA effective-N over families is built and tested per the chosen scope but is **bounded below by the declared integer**, so it can only ever raise the bar, never lower it.

**Tech Stack:** Python 3, numpy, pandas, stdlib `statistics.NormalDist` (no scipy dependency added). pytest. Existing `apps/quant/advisor/backtest/` harness.

---

## Scope (this plan only)

This is **Plan 1 of 3** (split decision, 2026-06-16). It builds ONLY the signal-agnostic validation gate. Explicitly **out of scope** and deferred to separate plans written later:

- **Workstream C completion** (full five-family `run_pipeline` CLI) — its own plan; prerequisite for any candidate-family work.
- **Post-C signal program** (registry, append-only evidence ledger, candidate families, eligibility report, `skill_weight` activation) — a roadmap stub, not a committed plan, until C lands AND a candidate signal exists. Do NOT pre-commit forward thresholds (90-day windows, observation counts, Brier lifts) now.

**Hard rails (inherited, do not violate):**
- Do NOT add fields to `PreRegConfig` (`prereg.py`) — it is SHA-hashed into immutable `PREREG.md`. Validation params live in a separate config.
- Do NOT modify `portfolio/allocator.py` / `ensemble_vote`.
- The gate is report-only: `floor_metrics()['verdict']` logic is unchanged; the gate cannot flip `DEV_FAILED`→`PASSED` or unlock the holdout.
- `node tools/run-floor.mjs --enforce` must stay exit 1.

---

## File Structure

- **Create** `apps/quant/advisor/backtest/validation_prereg.py` — frozen `ValidationPreReg` + `validation_hash()`. Separate immutable surface.
- **Create** `apps/quant/advisor/backtest/validation.py` — pure functions: `per_obs_sharpe`, `sharpe_moments`, `psr`, `deflated_sharpe`, `effective_n_pca`, `minbtl_exceeded`, `tstat_meets_hurdle`, `validation_report`.
- **Modify** `apps/quant/advisor/backtest/data_floor.py` — append `"validation"` to the `floor_metrics()` dict (one block; verdict untouched).
- **Modify** `tools/floor_data_check.py` — print validation caveats in report mode.
- **Create** `apps/quant/advisor/backtest/VALIDATION_PREREG.md` — the pre-registration artifact (hash, declared-N rationale, DSR bar, units convention, effective-N method + guardrail).
- **Create** tests: `tests/test_validation_prereg.py`, `tests/test_validation.py`, and extend `tests/test_data_floor.py`.

---

## Key conventions (read before coding)

**Units gotcha (load-bearing).** `stats.py:book_sharpe` *annualizes* (`× √252`). The PSR/DSR formulas require a **per-observation** Sharpe (`mean/std`, no annualization). `validation.py` computes its own per-obs Sharpe and must never call `book_sharpe`. Task 2 asserts `per_obs_sharpe(r) == book_sharpe(r) / √252`.

**Formulas (Bailey & López de Prado 2014).** Φ, Φ⁻¹ via `statistics.NormalDist().cdf / .inv_cdf`. γ3 = skewness, γ4 = kurtosis (Pearson, normal = 3.0). γ = Euler–Mascheroni ≈ 0.5772156649.

```
PSR(SR*) = Φ[ (SR̂ − SR*)·√(T−1) / √(1 − γ3·SR̂ + ((γ4−1)/4)·SR̂²) ]

SR0 = √(V[SR̂]) · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
DSR = PSR(SR0)
```

`SR̂` is per-observation. **`V[SR̂]` in `SR0` is the cross-trial dispersion of the N trial Sharpe estimates**, not a single strategy's sampling variance — using the latter shrinks `SR0` and makes the gate lenient (Task 4). This harness has no stored 45-trial book and only 2 report-level series, so `V[SR̂]` is a **pre-registered declared constant** (`declared_var_sr`), calibrated once from the realized trial Sharpes (Task 9) — NOT estimated live. `SR0 ∝ √var_sr`, so this constant is the gate's dominant leniency knob; higher = stricter.

---

## Task 1: `ValidationPreReg` — separate immutable config + floor-hash guard

**Files:**
- Create: `apps/quant/advisor/backtest/validation_prereg.py`
- Test: `apps/quant/advisor/tests/test_validation_prereg.py`

- [ ] **Step 1: Write the failing test** — including a regression guard proving `PreRegConfig` was NOT touched.

```python
# apps/quant/advisor/tests/test_validation_prereg.py
from dataclasses import asdict
from pathlib import Path

import pytest

from advisor.backtest.prereg import DEFAULT_CONFIG
from advisor.backtest.validation_prereg import (
    DEFAULT_VALIDATION, ValidationPreReg, validation_hash,
)


def test_floor_config_fields_unchanged():
    """Guard: adding validation params must NOT add fields to the frozen floor
    config, which is SHA-hashed into immutable PREREG.md (hash 1ad2ed4a...)."""
    assert set(asdict(DEFAULT_CONFIG).keys()) == {
        "window", "folds", "embargo", "warmup", "families", "added_families",
        "primary_metric", "margin", "pct_clip", "weight_grid",
        "train_lift_threshold", "max_asset_weight", "gross_cap", "turnover_cap",
        "cost_per_turn", "rebalance", "bootstrap_block", "bootstrap_draws",
        "bootstrap_seed", "dev_lcb", "final_lcb",
        "min_universe_formal", "min_universe_floor",
    }


def test_validation_prereg_frozen_and_has_fields():
    v = DEFAULT_VALIDATION
    assert v.dsr_pass == 0.95
    assert v.tstat_hurdle == 3.0
    assert v.minbtl_max_trials == 45
    assert v.declared_trials_N == 45          # conservative: N = the budget ceiling
    assert v.effective_n_method == "pca"
    assert v.psr_benchmark_sr == 0.0
    assert v.declared_var_sr > 0              # pre-registered cross-trial Sharpe dispersion (calibrated, Task 9)
    with pytest.raises(Exception):            # frozen dataclass
        v.dsr_pass = 0.5


def test_validation_hash_stable_and_sensitive():
    h1 = validation_hash(DEFAULT_VALIDATION)
    assert h1 == validation_hash(DEFAULT_VALIDATION) and len(h1) == 64
    assert validation_hash(ValidationPreReg(dsr_pass=0.99)) != h1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_validation_prereg.py -v`
Expected: FAIL — `ModuleNotFoundError: advisor.backtest.validation_prereg`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/quant/advisor/backtest/validation_prereg.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ValidationPreReg:
    """Immutable, pre-registered VALIDATION-GATE params. Separate hash surface from
    PreRegConfig so the recorded floor hash (PREREG.md) is never disturbed."""
    psr_benchmark_sr: float = 0.0         # SR* threshold the deflated bar must clear
    dsr_pass: float = 0.95               # Deflated Sharpe pass bar
    tstat_hurdle: float = 3.0            # Harvey-Liu-Zhu factor/signal selection hurdle
    minbtl_max_trials: int = 45          # MinBTL throughput budget on ~5yr daily sample
    declared_trials_N: int = 45          # dominant multiple-testing N (pre-registered,
                                         # = budget ceiling; conservative, can only deflate harder)
    effective_n_method: str = "pca"      # PCA participation-ratio over family return series
    effective_n_floor_is_declared: bool = True  # effective-N may never lower N below declared
    declared_var_sr: float = 4e-4        # PRE-REGISTERED cross-trial Sharpe dispersion V[{SR_n}].
                                         # A DECLARED CONSTANT, not estimated live (the harness has
                                         # no stored 45-trial book; 2 report-level series can't
                                         # estimate it). SR0 ∝ sqrt(var_sr) -> this is the gate's
                                         # dominant leniency knob; HIGHER = STRICTER. Interim 4e-4
                                         # errs high; Task 9 calibrates it from realized trial Sharpes.


DEFAULT_VALIDATION = ValidationPreReg()


def validation_hash(cfg: ValidationPreReg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_validation_prereg.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/validation_prereg.py apps/quant/advisor/tests/test_validation_prereg.py
git commit -m "feat(validation): add ValidationPreReg on separate immutable hash surface"
```

---

## Task 2: Per-observation Sharpe + moments

**Files:**
- Create: `apps/quant/advisor/backtest/validation.py`
- Test: `apps/quant/advisor/tests/test_validation.py`

- [ ] **Step 1: Write the failing test** — pins the units convention against `book_sharpe`.

```python
# apps/quant/advisor/tests/test_validation.py
import numpy as np
import pandas as pd

from advisor.backtest.stats import book_sharpe
from advisor.backtest.validation import per_obs_sharpe, sharpe_moments


def _r(mean, sd=0.01, n=1250, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n))


def test_per_obs_sharpe_is_annualized_over_sqrt_252():
    r = _r(0.0005)
    assert abs(per_obs_sharpe(r) - book_sharpe(r) / np.sqrt(252)) < 1e-12


def test_per_obs_sharpe_zero_on_zero_vol():
    assert per_obs_sharpe(pd.Series([0.001, 0.001, 0.001])) == 0.0


def test_sharpe_moments_returns_T_skew_kurt():
    r = _r(0.0005, seed=3)
    T, skew, kurt = sharpe_moments(r)
    assert T == 1250
    assert abs(kurt - 3.0) < 0.5        # ~normal -> Pearson kurtosis near 3
    assert abs(skew) < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -v`
Expected: FAIL — `ImportError: cannot import name 'per_obs_sharpe'`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/quant/advisor/backtest/validation.py
from __future__ import annotations

import numpy as np
import pandas as pd


def per_obs_sharpe(returns: pd.Series) -> float:
    """Per-observation Sharpe (mean/std). NOT annualized -- DSR/PSR require this.
    Distinct from stats.book_sharpe, which multiplies by sqrt(252)."""
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=0)
    if len(r) == 0 or sd == 0:
        return 0.0
    return float(r.mean() / sd)


def sharpe_moments(returns: pd.Series) -> tuple[int, float, float]:
    """Return (T, skewness, Pearson-kurtosis[normal=3]) of the returns series."""
    r = pd.Series(returns).dropna().to_numpy()
    T = len(r)
    if T < 2:
        return T, 0.0, 3.0
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return T, 0.0, 3.0
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))      # Pearson (normal -> 3.0)
    return T, skew, kurt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/validation.py apps/quant/advisor/tests/test_validation.py
git commit -m "feat(validation): per-observation Sharpe + moments (units-correct vs book_sharpe)"
```

---

## Task 3: PSR (Probabilistic Sharpe Ratio)

**Files:**
- Modify: `apps/quant/advisor/backtest/validation.py`
- Test: `apps/quant/advisor/tests/test_validation.py`

- [ ] **Step 1: Write the failing test** — golden value computed directly from the formula.

```python
# append to apps/quant/advisor/tests/test_validation.py
from statistics import NormalDist

from advisor.backtest.validation import psr


def test_psr_matches_closed_form():
    # SR_hat=0.10 (per-obs), SR*=0.0, T=1250, skew=0, kurt=3 (normal)
    sr, T = 0.10, 1250
    denom = (1 - 0.0 * sr + ((3.0 - 1) / 4) * sr ** 2) ** 0.5
    expected = NormalDist().cdf((sr - 0.0) * (T - 1) ** 0.5 / denom)
    assert abs(psr(sr_hat=sr, sr_benchmark=0.0, T=T, skew=0.0, kurt=3.0) - expected) < 1e-12


def test_psr_rises_with_more_observations():
    base = dict(sr_hat=0.08, sr_benchmark=0.0, skew=0.0, kurt=3.0)
    assert psr(T=300, **base) < psr(T=3000, **base)


def test_psr_negative_skew_lowers_confidence():
    base = dict(sr_hat=0.08, sr_benchmark=0.0, T=1250, kurt=6.0)
    assert psr(skew=-2.0, **base) < psr(skew=0.0, **base)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k psr -v`
Expected: FAIL — `ImportError: cannot import name 'psr'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to apps/quant/advisor/backtest/validation.py
from statistics import NormalDist

_Z = NormalDist()


def psr(sr_hat: float, sr_benchmark: float, T: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark) given the estimate's
    moments. sr_hat is PER-OBSERVATION (see per_obs_sharpe). Bailey & LdP 2014."""
    if T < 2:
        return 0.0
    var = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    if var <= 0:
        return 0.0
    return float(_Z.cdf((sr_hat - sr_benchmark) * (T - 1) ** 0.5 / var ** 0.5))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k psr -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/validation.py apps/quant/advisor/tests/test_validation.py
git commit -m "feat(validation): probabilistic Sharpe ratio (PSR)"
```

---

## Task 4: Deflated Sharpe Ratio + variance estimate

**Files:**
- Modify: `apps/quant/advisor/backtest/validation.py`
- Test: `apps/quant/advisor/tests/test_validation.py`

> **`var_sr` semantics (load-bearing — do not get this wrong).** `_sr0`'s `V[SR̂]` is the **cross-trial dispersion of the N trial Sharpe estimates**, NOT a single strategy's sampling variance. Using the (smaller) single-strategy sampling variance shrinks `SR0` and makes DSR **too high → gate too lenient**, violating the "can only fail harder" invariant. This harness exposes only 2 series at report level — too few to estimate dispersion — so `var_sr` is a **pre-registered declared constant** `vcfg.declared_var_sr` (calibrated once in Task 9 from the realized trial Sharpes via `var_sr_trials`), used directly in the production path. It is the gate's dominant leniency knob (`SR0 ∝ √var_sr`); the protective test below binds to the **shipping** value, not a hand-picked strict one.
>
> **De-circular reference (fixes the verify-before-pinning guard).** Tests re-derive the expected DSR **inline from the published formula** using `statistics.NormalDist` directly — independent of the module's `_sr0`/`deflated_sharpe` internals (only `NormalDist` is shared, legitimately). A miscoded γ, `e`, or term would diverge. The handoff's "SR 2.5 → DSR≈0.90" uses an *annualized* 2.5; convert to per-obs `2.5/√252` before any comparison.

- [ ] **Step 1: Write the failing test**

```python
# append to apps/quant/advisor/tests/test_validation.py
from math import e
from statistics import NormalDist

from advisor.backtest.validation import deflated_sharpe, var_sr_trials

_GAMMA = 0.5772156649015329


def _ref_sr0(N, var_sr):
    """Independent re-derivation of the deflated benchmark from the published formula."""
    z = NormalDist()
    return var_sr ** 0.5 * ((1 - _GAMMA) * z.inv_cdf(1 - 1.0 / N)
                            + _GAMMA * z.inv_cdf(1 - 1.0 / (N * e)))


def _ref_dsr(returns, N, var_sr, sr_benchmark=0.0):
    sr = per_obs_sharpe(returns)
    T, skew, kurt = sharpe_moments(returns)
    sr0 = max(sr_benchmark, _ref_sr0(N, var_sr))
    denom = (1 - skew * sr + ((kurt - 1) / 4) * sr ** 2) ** 0.5
    return NormalDist().cdf((sr - sr0) * (T - 1) ** 0.5 / denom)


def test_dsr_matches_independent_reference():
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    got = deflated_sharpe(r, n_trials=45, var_sr=4e-4, sr_benchmark=0.0)
    assert abs(got - _ref_dsr(r, 45, 4e-4)) < 1e-12


def test_dsr_units_handoff_example_reconciled():
    # Annualized 2.5 -> per-obs; verifies the module agrees with the hand formula
    # on the handoff's scenario. NOT a magic 0.90 literal; an independent re-derivation.
    import numpy as np
    r = _r(2.5 / np.sqrt(252) * 0.01, sd=0.01, n=1250, seed=11)  # ~per-obs SR 0.1575
    got = deflated_sharpe(r, n_trials=100, var_sr=3e-4)
    assert abs(got - _ref_dsr(r, 100, 3e-4)) < 1e-12


def test_dsr_falls_as_trial_count_rises():
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    assert deflated_sharpe(r, n_trials=200, var_sr=4e-4) < deflated_sharpe(r, n_trials=5, var_sr=4e-4)


def test_dsr_n1_is_undeflated_and_does_not_raise():
    # N=1 -> no multiple testing -> SR0=0 -> DSR == PSR at the benchmark. inv_cdf(0) must
    # never be hit (StatisticsError). Guards n_for_dsr's floor-of-1 path.
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    T, skew, kurt = sharpe_moments(r)
    assert abs(deflated_sharpe(r, n_trials=1, var_sr=4e-4)
               - psr(per_obs_sharpe(r), 0.0, T, skew, kurt)) < 1e-12


def test_larger_var_sr_is_stricter():
    # Direction of the dominant knob: larger cross-trial dispersion -> larger SR0 -> lower DSR.
    r = _r(0.0012, sd=0.01, n=1250, seed=7)
    assert deflated_sharpe(r, n_trials=45, var_sr=1e-3) < deflated_sharpe(r, n_trials=45, var_sr=1e-9)


def test_var_sr_trials_is_sample_variance_of_trial_sharpes():
    import numpy as np
    sharpes = [0.05, 0.10, 0.15, 0.20]
    assert abs(var_sr_trials(sharpes) - float(np.var(sharpes, ddof=1))) < 1e-12


def test_deflation_bites_at_production_var_sr():
    # Bind the "fails harder" property to the value that SHIPS, not a hand-picked strict one.
    from advisor.backtest.validation_prereg import DEFAULT_VALIDATION
    v = DEFAULT_VALIDATION.declared_var_sr
    r = _r(0.0005, sd=0.01, n=1250, seed=5)            # moderate Sharpe near the floor's level
    T, skew, kurt = sharpe_moments(r)
    undeflated = psr(per_obs_sharpe(r), 0.0, T, skew, kurt)   # N=1 equivalent
    deflated = deflated_sharpe(r, n_trials=45, var_sr=v)
    assert deflated < undeflated                       # multiple-testing penalty bites at ship config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k "dsr or var_sr" -v`
Expected: FAIL — `ImportError: cannot import name 'deflated_sharpe'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to apps/quant/advisor/backtest/validation.py
from math import e

_EULER_GAMMA = 0.5772156649015329


def _sr0(N: int, var_sr: float) -> float:
    """Deflated benchmark = expected max of N trial Sharpes (LdP). var_sr is the
    CROSS-TRIAL dispersion V[{SR_n}], NOT a single strategy's sampling variance.
    N<=1 means no multiple testing -> no deflation (and inv_cdf(0) would raise)."""
    if N <= 1 or var_sr <= 0:
        return 0.0
    g = _EULER_GAMMA
    term = (1 - g) * _Z.inv_cdf(1 - 1.0 / N) + g * _Z.inv_cdf(1 - 1.0 / (N * e))
    return float(var_sr ** 0.5 * term)


def var_sr_trials(trial_sharpes) -> float:
    """Cross-trial dispersion V[{SR_n}] = sample variance of the per-trial Sharpe
    estimates. This is the correct variance for _sr0 (see Task 4 var_sr note)."""
    s = [float(x) for x in trial_sharpes]
    if len(s) < 2:
        return 0.0
    return float(np.var(s, ddof=1))


def deflated_sharpe(returns: pd.Series, n_trials: int, var_sr: float,
                    sr_benchmark: float = 0.0) -> float:
    """DSR = PSR evaluated at the multiple-testing-deflated benchmark SR0.
    var_sr is the CROSS-TRIAL Sharpe dispersion (caller supplies it, floored)."""
    sr = per_obs_sharpe(returns)
    T, skew, kurt = sharpe_moments(returns)
    sr0 = max(sr_benchmark, _sr0(n_trials, var_sr))
    return psr(sr_hat=sr, sr_benchmark=sr0, T=T, skew=skew, kurt=kurt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k "dsr or var_sr" -v`
Expected: PASS (7 tests). If `test_dsr_units_handoff_example_reconciled` fails, STOP — it means the module diverges from the independent formula; fix the impl, never the reference.

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/validation.py apps/quant/advisor/tests/test_validation.py
git commit -m "feat(validation): deflated Sharpe ratio with multiple-testing benchmark"
```

---

## Task 5: PCA effective-N over family return series (bounded below by declared N)

**Files:**
- Modify: `apps/quant/advisor/backtest/validation.py`
- Test: `apps/quant/advisor/tests/test_validation.py`

> **Design note (honest scope).** Effective-N is computed via the **PCA participation ratio** `Neff = (Σλ)² / Σλ²` on the correlation matrix of the candidate family OOS return series — correlated families collapse to fewer independent trials. But effective-N would *lower* the deflation penalty, in tension with a gate meant to only fail harder. Guardrail: `n_for_dsr = max(declared_trials_N, ceil(Neff))`. Since the floor's declared N is the budget ceiling (45) and the family set is ≤5, the declared integer dominates and effective-N is **dormant for the current floor** — by design. The machinery is built and tested for forward use (a future candidate set larger than the declared meta-count).

- [ ] **Step 1: Write the failing test**

```python
# append to apps/quant/advisor/tests/test_validation.py
from advisor.backtest.validation import effective_n_pca, n_for_dsr


def test_effective_n_collapses_correlated_series():
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 500)
    fams = {                                   # 3 near-identical -> Neff ~ 1
        "a": pd.Series(base + rng.normal(0, 1e-5, 500)),
        "b": pd.Series(base + rng.normal(0, 1e-5, 500)),
        "c": pd.Series(base + rng.normal(0, 1e-5, 500)),
    }
    assert effective_n_pca(fams) < 1.5


def test_effective_n_independent_series_near_count():
    import numpy as np
    rng = np.random.default_rng(0)
    fams = {k: pd.Series(rng.normal(0, 0.01, 500)) for k in ("a", "b", "c", "d")}
    assert effective_n_pca(fams) > 3.0         # ~4 independent


def test_n_for_dsr_never_below_declared():
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 500)
    fams = {k: pd.Series(base) for k in ("a", "b")}   # Neff ~ 1
    assert n_for_dsr(fams, declared_trials_N=45) == 45  # declared dominates
    assert n_for_dsr(fams, declared_trials_N=0) == 1    # floors at >=1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k effective_n -v`
Expected: FAIL — `ImportError: cannot import name 'effective_n_pca'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to apps/quant/advisor/backtest/validation.py
from math import ceil


def effective_n_pca(family_returns: dict[str, pd.Series]) -> float:
    """PCA participation ratio on the family return-series correlation matrix.
    Neff = (sum eigenvalues)^2 / sum(eigenvalues^2). Correlated families -> small Neff."""
    series = [pd.Series(s).reset_index(drop=True) for s in family_returns.values()]
    if len(series) <= 1:
        return float(len(series))
    mat = pd.concat(series, axis=1).dropna()
    if mat.shape[0] < 2:
        return float(mat.shape[1])
    corr = np.corrcoef(mat.to_numpy(), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    lam = np.linalg.eigvalsh(corr)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 1.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def n_for_dsr(family_returns: dict[str, pd.Series], declared_trials_N: int) -> int:
    """Trials used for DSR: max(declared integer, ceil(effective-N)), floored at 1.
    Effective-N may never drop the penalty below the pre-registered declared count."""
    eff = ceil(effective_n_pca(family_returns)) if family_returns else 0
    return max(1, declared_trials_N, eff)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k effective_n -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/validation.py apps/quant/advisor/tests/test_validation.py
git commit -m "feat(validation): PCA effective-N over families, bounded below by declared N"
```

---

## Task 6: MinBTL budget + Harvey-Liu-Zhu t-stat hurdle

**Files:**
- Modify: `apps/quant/advisor/backtest/validation.py`
- Test: `apps/quant/advisor/tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to apps/quant/advisor/tests/test_validation.py
from advisor.backtest.validation import minbtl_exceeded, tstat_meets_hurdle


def test_minbtl_exceeded_above_budget():
    assert minbtl_exceeded(n_trials=60, max_trials=45) is True
    assert minbtl_exceeded(n_trials=45, max_trials=45) is False
    assert minbtl_exceeded(n_trials=10, max_trials=45) is False


def test_tstat_hurdle():
    assert tstat_meets_hurdle(3.5, hurdle=3.0) is True
    assert tstat_meets_hurdle(3.0, hurdle=3.0) is True
    assert tstat_meets_hurdle(2.9, hurdle=3.0) is False
    assert tstat_meets_hurdle(None, hurdle=3.0) is False   # no claim -> not met
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k "minbtl or tstat" -v`
Expected: FAIL — `ImportError: cannot import name 'minbtl_exceeded'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to apps/quant/advisor/backtest/validation.py
def minbtl_exceeded(n_trials: int, max_trials: int) -> bool:
    """True when the search budget is over-spent (gate is meaningless past this)."""
    return n_trials > max_trials


def tstat_meets_hurdle(tstat: float | None, hurdle: float) -> bool:
    """Harvey-Liu-Zhu selection hurdle. None (no t-stat claim) does not meet it."""
    return tstat is not None and tstat >= hurdle
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k "minbtl or tstat" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/validation.py apps/quant/advisor/tests/test_validation.py
git commit -m "feat(validation): MinBTL budget check + Harvey-Liu-Zhu t-stat hurdle"
```

---

## Task 7: Assemble `validation_report` and wire into `floor_metrics`

**Files:**
- Modify: `apps/quant/advisor/backtest/validation.py`
- Modify: `apps/quant/advisor/backtest/data_floor.py:55-66` (the return dict)
- Test: `apps/quant/advisor/tests/test_validation.py`, `apps/quant/advisor/tests/test_data_floor.py`

> **Wiring note.** `floor_metrics` exposes `sweep.ensemble_test_returns` and `sweep.best_family_test_returns` at report level (not per-family series). With only 2 series, both the effective-N count AND any live dispersion estimate are inert, so `n_used = declared_N` (45 dominates) and `var_sr = vcfg.declared_var_sr` (the pre-registered, Task-9-calibrated constant) are used directly. The verdict branch in `floor_metrics` is **not touched**; `"validation"` is additive.

- [ ] **Step 1: Write the failing test**

```python
# append to apps/quant/advisor/tests/test_validation.py
from advisor.backtest.validation import validation_report
from advisor.backtest.validation_prereg import DEFAULT_VALIDATION


def test_validation_report_shape_and_deflation():
    cand = _r(0.0012, sd=0.01, n=1250, seed=7)
    fams = {"ensemble": cand, "best_family": _r(0.0002, seed=8)}
    rep = validation_report(cand, fams, DEFAULT_VALIDATION)
    for k in ("per_obs_sharpe", "T", "skew", "kurt", "declared_N", "effective_N",
              "n_used", "var_sr", "dsr", "dsr_pass_bar", "dsr_passes", "tstat_met",
              "minbtl_exceeded", "passes"):
        assert k in rep
    assert rep["n_used"] == 45                 # declared dominates effective-N
    assert rep["dsr_pass_bar"] == 0.95
    # tstat is None here -> no-op; passes = dsr_passes and not over-budget
    assert rep["passes"] == (rep["dsr_passes"] and not rep["minbtl_exceeded"])


def test_minbtl_checks_actual_n_used_not_declared():
    # n_used (here = declared) above the budget must flag minbtl_exceeded.
    from advisor.backtest.validation_prereg import ValidationPreReg
    over = ValidationPreReg(declared_trials_N=50, minbtl_max_trials=45)
    cand = _r(0.0012, sd=0.01, n=1250, seed=7)
    fams = {"ensemble": cand, "best_family": _r(0.0002, seed=8)}
    rep = validation_report(cand, fams, over)
    assert rep["n_used"] == 50 and rep["minbtl_exceeded"] is True and rep["passes"] is False


def test_tstat_claim_below_hurdle_fails_otherwise_passing_candidate():
    # A strong (DSR-passing) candidate with a supplied weak t-stat must NOT pass;
    # the same candidate with no t-stat claim is unaffected by the hurdle.
    from advisor.backtest.validation_prereg import ValidationPreReg
    lax = ValidationPreReg(declared_trials_N=2, dsr_pass=0.0)   # force dsr_passes True
    cand = _r(0.0015, sd=0.01, n=1250, seed=9)
    fams = {"ensemble": cand, "best_family": _r(0.0002, seed=8)}
    assert validation_report(cand, fams, lax, tstat=None)["passes"] is True
    assert validation_report(cand, fams, lax, tstat=2.0)["passes"] is False
    assert validation_report(cand, fams, lax, tstat=3.5)["passes"] is True
```

```python
# append to apps/quant/advisor/tests/test_data_floor.py
def test_floor_metrics_validation_is_additive_only(monkeypatch):
    """Report-only invariant: validation_report's result must NOT change verdict,
    passes, holdout, or any legacy metric. Forcing passes=False changes ONLY the
    additive 'validation' key."""
    from advisor.backtest import data_floor
    from advisor.backtest.prereg import DEFAULT_CONFIG
    panel = _panel()
    baseline = data_floor.floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    assert "validation" in baseline and baseline["validation"]["dsr_pass_bar"] == 0.95

    # Force the gate to "fail" and prove nothing else moves.
    monkeypatch.setattr(data_floor, "validation_report",
                        lambda *a, **k: {"passes": False, "sentinel": True})
    forced = data_floor.floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    for k in set(baseline) | set(forced):
        if k == "validation":
            continue
        assert forced[k] == baseline[k], f"validation leaked into {k!r}"
    assert forced["validation"] == {"passes": False, "sentinel": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py -k validation_report apps/quant/advisor/tests/test_data_floor.py -v`
Expected: FAIL — `ImportError: cannot import name 'validation_report'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to apps/quant/advisor/backtest/validation.py
from advisor.backtest.validation_prereg import ValidationPreReg


def validation_report(candidate_returns: pd.Series,
                      family_returns: dict[str, pd.Series],
                      vcfg: ValidationPreReg,
                      tstat: float | None = None) -> dict:
    """Signal-agnostic deflation diagnostics. Report-only: never mutates verdict."""
    T, skew, kurt = sharpe_moments(candidate_returns)
    var_sr = vcfg.declared_var_sr          # pre-registered constant (Task 9 calibrated); NOT
                                           # estimated from 2 report-level series (would be noise)
    eff = effective_n_pca(family_returns) if family_returns else 0.0
    n_used = n_for_dsr(family_returns, vcfg.declared_trials_N)
    dsr = deflated_sharpe(candidate_returns, n_trials=n_used, var_sr=var_sr,
                          sr_benchmark=vcfg.psr_benchmark_sr)
    over_budget = minbtl_exceeded(n_used, vcfg.minbtl_max_trials)   # ACTUAL trials, not declared
    tstat_met = tstat_meets_hurdle(tstat, vcfg.tstat_hurdle)
    dsr_passes = dsr >= vcfg.dsr_pass
    return {
        "per_obs_sharpe": per_obs_sharpe(candidate_returns),
        "T": T, "skew": skew, "kurt": kurt,
        "declared_N": vcfg.declared_trials_N,
        "effective_N": eff,
        "n_used": n_used,
        "var_sr": var_sr,
        "dsr": dsr,
        "dsr_pass_bar": vcfg.dsr_pass,
        "dsr_passes": dsr_passes,
        "tstat": tstat,
        "tstat_hurdle": vcfg.tstat_hurdle,
        "tstat_met": tstat_met,
        "minbtl_exceeded": over_budget,
        # t-stat is a no-op when no selection claim is supplied (tstat is None),
        # enforced only when a claim IS made -> can fail an otherwise-passing candidate.
        "passes": dsr_passes and not over_budget and (tstat is None or tstat_met),
        "note": "report-only deflation guard; does not unlock holdout or authorize sizing",
    }
```

```python
# modify apps/quant/advisor/backtest/data_floor.py
# add imports near the top:
from advisor.backtest.validation import validation_report
from advisor.backtest.validation_prereg import DEFAULT_VALIDATION

# inside floor_metrics(), build the report before `return {...}`:
    validation = validation_report(
        sweep.ensemble_test_returns,
        {"ensemble": sweep.ensemble_test_returns,
         "best_family": sweep.best_family_test_returns},
        DEFAULT_VALIDATION,
    )

# add ONE key to the returned dict (verdict/passes logic unchanged):
        "validation": validation,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_validation.py apps/quant/advisor/tests/test_data_floor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/quant/advisor/backtest/validation.py apps/quant/advisor/backtest/data_floor.py apps/quant/advisor/tests/test_validation.py apps/quant/advisor/tests/test_data_floor.py
git commit -m "feat(validation): wire report-only deflation diagnostics into floor_metrics"
```

---

## Task 8: Print the validation caveat in report mode

**Files:**
- Modify: `tools/floor_data_check.py:15-44` (`_print_verdict`)
- Test: `apps/quant/advisor/tests/test_floor_entrypoint.py`

- [ ] **Step 1: Write the failing test**

```python
# append to apps/quant/advisor/tests/test_floor_entrypoint.py
def test_verdict_print_includes_validation_caveat(capsys):
    from tools.floor_data_check import _print_verdict  # if import path differs, see note
    m = {
        "verdict": "DEV_FAILED", "universe": "formal",
        "ensemble": 0.73, "spy": 0.85, "best_family": 0.83,
        "dev": {"reasons": ["median fold delta not > 0"]},
        "validation": {"dsr": 0.41, "dsr_pass_bar": 0.95, "passes": False,
                       "n_used": 45, "minbtl_exceeded": False},
    }
    _print_verdict(m)
    out = capsys.readouterr().out
    assert "DSR" in out and "0.41" in out and "report-only" in out.lower()
```

> Note: if `tools/` is not importable in the test env, move `_print_verdict` is unnecessary — instead assert on `main()` stdout via `capsys` as `test_floor_entrypoint.py` already does for the existing path, passing a temp fixture.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest apps/quant/advisor/tests/test_floor_entrypoint.py -k validation_caveat -v`
Expected: FAIL — assertion error (no "DSR" in output)

- [ ] **Step 3: Write minimal implementation** — add a caveat line at the end of `_print_verdict`.

```python
# in tools/floor_data_check.py, append inside _print_verdict (after the if/elif/else):
    v = m.get("validation")
    if v:
        flag = "PASS" if v["passes"] else "FAIL"
        print(
            f"floor: validation (report-only) -- DSR {v['dsr']:.2f} vs bar "
            f"{v['dsr_pass_bar']:.2f} [{flag}]; N_used {v['n_used']}; "
            f"MinBTL_exceeded {v['minbtl_exceeded']}. This guard can only confirm "
            f"the floor, never unlock the holdout or authorize sizing.",
            flush=True,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest apps/quant/advisor/tests/test_floor_entrypoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/floor_data_check.py apps/quant/advisor/tests/test_floor_entrypoint.py
git commit -m "feat(validation): print report-only deflation caveat in floor report"
```

---

## Task 9: Pre-registration artifact + roadmap stub + full-suite verification

**Files:**
- Create: `apps/quant/advisor/backtest/VALIDATION_PREREG.md`
- Create: `docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md`

- [ ] **Step 1: Write the pre-registration artifact** (the honest "what bar, declared before seeing results" record).

```markdown
# Validation Gate Pre-Registration

Separate immutable surface from PREREG.md (which records the floor config hash
`1ad2ed4a...`). This file pre-registers the deflation gate parameters.

- validation_hash(DEFAULT_VALIDATION) = <paste output of `python -c "import sys; sys.path.insert(0,'apps/quant'); from advisor.backtest.validation_prereg import DEFAULT_VALIDATION, validation_hash; print(validation_hash(DEFAULT_VALIDATION))"`>
- DSR pass bar: 0.95 (PSR at the multiple-testing-deflated benchmark).
- Sharpe units: PER-OBSERVATION (mean/std). `book_sharpe` (annualized x sqrt(252)) is NOT used in the gate.
- Declared multiple-testing N: 45 = the MinBTL budget ceiling. Conservative by
  construction (larger N -> stricter), chosen over fragile reconstruction of the
  exact families x weights x constructions (C/D/E) x versions (v1/v2) trial count.
- Effective-N method: PCA participation ratio over candidate family OOS return
  series. GUARDRAIL: n_used = max(declared_N, ceil(effective_N)); effective-N may
  never lower the penalty below the declared integer without a new pre-registration.
  Dormant-as-a-count for the current floor (family set <= 5 << 45) BY DESIGN — the
  PCA machinery is built/tested for forward use when a candidate set exceeds the
  declared meta-count. (The family Sharpes still feed the var_sr dispersion channel.)
- V[SR_hat] for SR0 = declared_var_sr, a PRE-REGISTERED CONSTANT (NOT the single-strategy
  sampling variance, which biases the gate lenient; NOT a live estimate off 2 series, which
  is noise). It is the gate's dominant leniency knob: SR0 ∝ sqrt(var_sr), higher = stricter.
  Calibrated value: <paste from the calibration step below>. Method: sample variance (ddof=1)
  of the per-obs Sharpes of every trial actually run (each family standalone + each C/D/E
  construction) on the dev portion of the fixture. Recalibration requires a new pre-registration.
- HLZ t-stat hurdle: 3.0 (applies only where a factor/signal selection claim is made).
- PBO/CSCV: deferred (audit-only, medium-confidence, synthetic-only; not built in this slice).
- Gross-vs-net: the floor already nets costs (cost_per_turn=0.0005, turnover_cap=0.20);
  gross-vs-net rejection deferred to a future high-turnover candidate.
- Posture: report-only. Cannot flip DEV_FAILED->PASSED, unlock the holdout, or authorize sizing.
```

- [ ] **Step 2: Write the calibration helper** — concrete, enumerates the pre-registered A–E trial book from `PREREG.md:23-28`.

Create `tools/calibrate_var_sr.py`:

```python
# tools/calibrate_var_sr.py -- compute declared_var_sr from the pre-registered trial book.
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "apps/quant")
from advisor.backtest.pipeline import run_dev_sweep            # noqa: E402
from advisor.backtest.prereg import DEFAULT_CONFIG             # noqa: E402
from advisor.backtest.validation import per_obs_sharpe, var_sr_trials  # noqa: E402

FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")

# The pre-registered candidate order (PREREG.md "Candidate order", smallest-first).
TRIAL_BOOK = {
    "A": ("trend",),
    "B": ("momentum",),
    "C": ("momentum", "trend"),
    "D": ("momentum", "trend", "mean_reversion"),
    "E": ("momentum", "trend", "mean_reversion", "breakout"),
}


def main() -> int:
    panel = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    sharpes = {}
    for name, fams in TRIAL_BOOK.items():
        sweep = run_dev_sweep(panel, fams, DEFAULT_CONFIG)
        sharpes[name] = per_obs_sharpe(sweep.ensemble_test_returns)
    vals = list(sharpes.values())
    measured = var_sr_trials(vals)
    declared = math.ceil(measured * 1e4) / 1e4          # round UP to next 1e-4 -> errs strict
    print(f"trial_count = {len(vals)}")
    print(f"trial_sharpes (per-obs) = {sharpes}")
    print(f"var_sr_trials (measured) = {measured:.6e}")
    print(f"declared_var_sr (rounded up) = {declared:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run calibration; set `declared_var_sr`; note the small-sample caveat**

Run: `python tools/calibrate_var_sr.py`
Set `ValidationPreReg.declared_var_sr` to the printed `declared_var_sr` and paste the four printed lines into `VALIDATION_PREREG.md`'s "Calibrated value" section. Note there: the estimate is from a 5-trial book (small sample for a variance) and is frozen by pre-registration; recalibration requires a new pre-registration. Rounding up errs strict (`SR0 ∝ √var_sr`).

- [ ] **Step 4: Fill the hash** — run the command and paste the real value (AFTER Step 3 sets the final `declared_var_sr`).

Run: `python -c "import sys; sys.path.insert(0,'apps/quant'); from advisor.backtest.validation_prereg import DEFAULT_VALIDATION, validation_hash; print(validation_hash(DEFAULT_VALIDATION))"`
Paste the 64-char hash into `VALIDATION_PREREG.md`.

- [ ] **Step 5: Write the deferred-plans roadmap stub** (NOT a committed plan; thresholds intentionally absent).

```markdown
# Deferred Plans Roadmap (post validation gate)

Plan 1 (this slice): validation gate, REPORT-ONLY. DONE when Task 9 commits. The gate
computes DSR/MinBTL/t-stat diagnostics but does NOT influence `verdict` or the
`--enforce` exit code (proven by test_floor_metrics_validation_is_additive_only).

Plan 1b — Wire validation into the release gate (write after Plan 1 + first real
candidate). Make `node tools/run-floor.mjs --enforce` require BOTH verdict==PASSED AND
validation["passes"], so a deflation-failing candidate cannot be released. This is the
deliberate step that turns the report-only guard into a blocking one; deferred so the
current accepted DEV_FAILED floor is not disturbed.

Plan 2 — Workstream C completion. Wire the full five-family run_pipeline into the
CLI (value/quality, momentum, trend, macro, sentiment) with provider adapters and
fake-provider tests. Prerequisite for ALL candidate-family work. See
docs/superpowers/plans/2026-06-15-followups-handoff.md.

Plan 3 — Post-C signal program (write ONLY after C lands AND a candidate exists):
registry, append-only evidence ledger (separate from checkpoint upsert), report-only
candidate families (filing/accounting events; macro/credit expansion), orthogonality
diagnostics, dormant eligibility report, and a later separate skill_weight activation
plan. Do NOT pre-commit forward thresholds (windows, observation counts, Brier lifts)
until that plan is written against real constraints. SESTM/news sentiment stays
research-only/conditional for the large-cap book.
```

- [ ] **Step 6: Verify no paste-placeholders remain in the artifact**

The pre-registration artifact must not ship with unfilled `<paste …>` markers (would look pre-registered but be mechanically incomplete).

Run: `grep -n "<paste\|<\.\.\.\|TODO\|TBD" apps/quant/advisor/backtest/VALIDATION_PREREG.md; test $? -eq 1`
Expected: exit 0 (grep finds nothing → `test $? -eq 1` succeeds). If any marker is found, fill it before committing.

- [ ] **Step 7: Run the full advisor suite + the gates**

Run: `python -m pytest apps/quant/advisor/tests -q`
Expected: PASS (all existing + new validation tests)

Run: `npm run advisor-gate`
Expected: exit 0, report mode green; floor still prints `DEV_FAILED` with the new validation caveat line.

Run: `node tools/run-floor.mjs --enforce`
Expected: **exit 1** (verdict ≠ PASSED). The gate must NOT have changed this.

- [ ] **Step 8: Commit**

```bash
git add apps/quant/advisor/backtest/validation_prereg.py apps/quant/advisor/backtest/VALIDATION_PREREG.md tools/calibrate_var_sr.py docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md
git commit -m "docs(validation): pre-registration artifact (calibrated var_sr) + calibration helper + deferred-plans roadmap"
```

---

## Hermes dispatch note (per memory `hermes-dispatch-windows`)

Each task is dispatchable solo: `npm run hermes:production -- --task "<task text>"`. Always include *"Do NOT run npm or node; verify ONLY with pytest."* Set `PYTHONUTF8=1`. After each dispatch: `git show --stat` (only the task's files) → `python -m pytest <task tests>` → commit if Codex didn't. Verify Codex's REAL git state (it cherry-picks/skips). If a seeded assertion fails against the provided impl, that's a plan-bug — fix the fixture/threshold here, don't re-dispatch. The DSR ballpark reconciliation (Task 4) is the most likely plan-bug surface.

---

## Self-Review

**Spec coverage** (against the corrected slice-1 scope):
- Separate immutable config (no `PreRegConfig` field additions) → Task 1, with explicit field-set guard test. ✓
- Per-obs Sharpe units distinct from `book_sharpe` → Task 2. ✓
- PSR / DSR with verify-before-pinning → Tasks 3–4. ✓
- Multiple-testing N: declared integer (dominant) + PCA effective-N bounded below → Tasks 1, 5. ✓
- `V[SR̂]` for `SR0` = pre-registered declared constant `declared_var_sr` (calibrated in Task 9 from realized trial Sharpes), NOT single-strategy sampling variance and NOT a noisy 2-series live estimate → Tasks 1, 4, 7, 9; protective test binds to the shipping value (`test_deflation_bites_at_production_var_sr`). ✓ Known residual: `declared_var_sr` is the dominant leniency knob — Task 9 calibration is mandatory before the bar is trusted. ⚠
- MinBTL (checks actual `n_used`, not declared) + HLZ t-stat (enforced in `passes` when a claim is supplied) → Tasks 6, 7. ✓
- Report-only wiring, verdict + all legacy metrics untouched → Task 7, proven by the monkeypatch independence test (`test_floor_metrics_validation_is_additive_only`). Green-wash *blocking* (require validation.passes for release) is deliberately deferred to Plan 1b so the accepted DEV_FAILED floor is not disturbed. ✓
- Purging/embargo: `purged_splits()` ALREADY EXISTS — audited via the existing sweep, no new code (intentionally not a task). ✓
- Pre-registration artifact + release gate stays exit 1 → Task 9. ✓
- Deferred: CSCV/PBO, gross-vs-net, registry, ledger, candidate families, `skill_weight` → roadmap stub, out of scope. ✓

**Placeholder scan:** `declared_var_sr` is now calibrated by an executable helper (`tools/calibrate_var_sr.py`, Task 9 Step 2) over the pre-registered A–E trial book — no prose placeholder. The remaining human paste actions (validation_hash, calibrated value) are guarded by a grep-for-`<paste` check (Task 9 Step 6) that fails before commit. No silent placeholders.

**Type consistency:** `ValidationPreReg` fields (`declared_trials_N`, `dsr_pass`, `minbtl_max_trials`, `effective_n_method`) are referenced identically in Tasks 1, 5, 7, 9. `validation_report` keys match the assertions in Tasks 7–8 (`dsr`, `dsr_pass_bar`, `n_used`, `passes`, `minbtl_exceeded`). `per_obs_sharpe`/`deflated_sharpe`/`effective_n_pca`/`n_for_dsr` signatures consistent across tasks.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-validation-gate.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. (Or per the project's Hermes convention: dispatch each task solo via `hermes:production`, verify Codex's real git state, commit between tasks.)
2. **Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints.

Which approach?
