# Decision-5 L/S Reversal Lane — Gate-1 Kill Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans. Every code/test edit is dispatched via Hermes
> (`node orchestrate.mjs --phase production --task "..."` from Git Bash); Claude gathers
> context, judges the tree after each dispatch, and runs postflight.

**Goal:** Freeze and build the Gate-1 cost-overlay kill screen for the pre-registered
long/short reversal hypothesis (operator ruling 2026-07-06: Decision 5 picked). The
screen decides — under a kill rule frozen at this doc's commit, BEFORE any real-data
execution — whether the reversed-book IRs survive realistic short costs.

**Architecture:** New separate prereg surface (`LongShortReversalPreReg`) + a report-only
screen module in `backtest/` that reuses `residual_screen.py` conventions and
`run_dev_sweep_ext` streams; two additive fields on `SweepResultExt` expose per-day
turnover and gross exposure. Zero floor coupling; holdout stays blinded; dev folds only.

**Tech stack:** Python/pandas/numpy, pytest; existing purged-CV candidate pipeline.

---

## 1. Ruling and provenance

- **Ruling (operator, 2026-07-06):** open Decision 5 (L/S market-neutral), ranked ahead
  of insider Form-4 > paid-data lever > disclosure tone. Insider lane stays queued 2nd.
- **Hypothesis (pre-registered here):** the corrected residual screen (PR #12,
  `apps/quant/advisor/backtest/residual_screen.py`) measured **negative** SPY-hedged
  info ratios on the 30-mega-cap floor dev window: **value −0.41, fundamental_value
  −0.32, lazy_prices −0.40** (momentum −0.02 ≈ 0, excluded). A book that SHORTS each
  family's long-flat book and holds a static long **a·SPY** hedge realizes the sign-flipped
  IR **+0.32..+0.41 in-sample, PRE-COST**. Sources:
  `docs/superpowers/notes/2026-06-21-blend-futility-residual-alpha.md` (correction header),
  `ai-logs/hermes/diag_residual_screen_bc.py` (T0.2c readings B/C).
- **Known weakness, stated up front:** this is an in-sample sign flip — textbook snooping
  risk. Gate 1 is therefore a **futility screen only**: PASS never asserts alpha; it only
  authorizes designing Gate 2 (one-shot deflated L/S dev run, holdout blinded).

## 2. Gate structure

| Gate | What | Built when |
|---|---|---|
| **Gate 1 (this plan)** | Report-only cost overlay on the three families' reversed streams; frozen kill rule | Now (Hermes T1–T4) |
| **Gate 2 (sketch, §10)** | Separate L/S candidate prereg + state-machine port (borrow/txn/maintenance margin) + one-shot deflated dev run | ONLY on Gate-1 PASS |

## 3. FROZEN kill criterion and cost model

Frozen at this document's commit on `exec/decision5-ls-gate1`. No parameter may change
afterward; a changed value = a NEW prereg filename (v2), and the original outcome stands
(QC-lane lesson: post-hoc goalpost moves are demoted to new eyes-open programs).

- **Families:** `("value", "fundamental_value", "lazy_prices")` — momentum excluded
  (−0.02, no reversal signal); **no family may be added or removed post-hoc**.
- **Published pre-cost IRs (reproduction tripwire):** value −0.41, fundamental_value
  −0.32, lazy_prices −0.40; each family's zero-cost screen value must reproduce within
  **±0.02** or the run ABORTs (wiring drift) with post-cost outputs suppressed.
- **Borrow:** 50 bps/yr (`borrow_rate_annual = 0.005`) charged daily on the held gross
  short notional (mega-caps = general collateral; 50 bps is the conservative end).
- **Short rebate:** 0 (conservative: no rebate credited).
- **Transaction cost:** `cost_per_turn = 0.0005` (5 bps one-way, mirrors floor
  `PreRegConfig.cost_per_turn`), charged on the reversed book's own turnover — which is
  the mirrored long-book turnover (see §4 for why this lands as **2×** on the net stream).
- **Hedge:** static OLS beta per `residual_screen.resid` convention; zero rebalance cost
  (hedge is fit once per stream; entry cost < 1 bp one-time, immaterial vs a 0.20 IR
  threshold — stated, not hidden).
- **Kill rule:** compute `postcost_reversed_ir` per family. Lane **PASSES Gate 1 iff at
  least 2 of the 3 families have postcost_reversed_ir ≥ τ_LS = 0.20**; otherwise the lane
  is **CLOSED** and the negative result is recorded. τ_LS = 0.20 ≈ half the best pre-cost
  point estimate — below that, costs have eaten the effect and Gate 2 is not worth its
  MinBTL budget. A single surviving family is treated as idiosyncratic noise (the
  hypothesis is a multi-family reversal), hence 2-of-3.
- **One-shot:** exactly one real-data execution. Re-runs permitted ONLY after an ABORT
  (the tripwire compares against already-published numbers, so it leaks nothing), never
  after PASS/CLOSED.
- **Decision rights:** operator records the verdict; agent may determine ABORT/CLOSED
  mechanically but never relaxes a threshold.
- **Report-only:** no floor run, no holdout access, no `PreRegConfig` change (immutable
  PREREG.md hash), no change to frozen `backtest/pipeline.py` / `blend.py` / floor path.
- **Frozen inputs (eng review A5):** the panel AND both fixture paths are prereg fields
  (`panel`, `fundamental_fixture`, `lazy_prices_fixture`) so a path swap changes the
  methodology hash; the run hash additionally binds all three files' bytes.
- **No-execution guard (eng review A1):** NO test, Hermes verification step, or "smoke
  check" may execute `run_screen`/`main()` against the real panel/fixtures. Tests are
  synthetic-only, plus a wiring test that builds `_readings` WITHOUT computing any
  returns. The §7.5 one-shot is the only real-data execution, ever.

## 4. The reversed-stream math (load-bearing correctness detail)

`book.py::book_returns` returns **net** returns: `net_t = gross_t − turnover_t·c`.
Negating a net stream flips costs into fake gains. The reversed book must:

1. recover gross: `gross_t = net_t + turnover_t·c`
2. negate: short-the-book P&L = `−gross_t`
3. charge the mirrored trades' own costs (same turnover): `−turnover_t·c`
4. charge borrow on held short notional: `−gross_exposure_t·(borrow−rebate)/252`
5. add the static hedge: `+a·spy_t`, with `a` fit by `residual_screen.resid` on the
   **net** stream (keeps the zero-cost identity below exact).

Combined: **`rev_t = −net_t − 2·turnover_t·c − gross_exposure_t·(b−r)/252 + a·spy_t`**

Zero-cost identity (unit-tested): with `c = b = r = 0`,
`book_sharpe(rev) == −book_sharpe(net − a·spy)` exactly — the sign-flipped published IR.
`turnover_t` and `gross_exposure_t` must mirror `book.py` expressions byte-for-byte:
`turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))`;
`gross_exposure = w.shift(1).fillna(0.0).abs().sum(axis=1)` (held weights earn returns
and are what the short must borrow).

**Pinned quirk (eng review A4):** in `book.py:14` the `.fillna(...)` is dead code —
`sum(axis=1)` over the all-NaN first `diff()` row returns 0.0, so day-1 (and each fold's
entry-day) turnover is 0 and entry costs go uncharged. This is the frozen floor's
semantics; the mirror reproduces it deliberately (≈2.5 bps total over the window,
immaterial vs τ_LS = 0.20). NEVER "fix" `book.py` — it would move the floor verdict pins.

## 5. File map

| File | Action | Responsibility |
|---|---|---|
| `apps/quant/advisor/research/ls_reversal_prereg.py` | Create | Frozen dataclass + methodology hash + run hash (binds panel + both fixture bytes) |
| `apps/quant/advisor/research/LS_REVERSAL_PREREG.md` | Create | Pins the methodology hash (mirror READING_B_PREREG.md ceremony) |
| `apps/quant/advisor/research/candidate_pipeline.py` | Modify | Two ADDITIVE fields on `SweepResultExt`: `ensemble_test_turnover`, `ensemble_test_gross` |
| `apps/quant/advisor/backtest/ls_reversal_screen.py` | Create | Screen module + CLI (reversed stream, tripwire, verdict) |
| `apps/quant/advisor/tests/test_ls_reversal_prereg.py` | Create | Hash stable/sensitive + pinned-hash test |
| `apps/quant/advisor/tests/test_ls_reversal_screen.py` | Create | Identity/monotonicity/verdict/ABORT-suppression tests |
| `apps/quant/advisor/tests/test_candidate_pipeline.py` | Modify | Turnover/gross exposure reconstruction test |
| `TODOS.md` | Modify | L/S diagnostics entry: gate becomes "Gate-1 PASS" (Decision 5 picked 2026-07-06) |

Expected `backtest/**` diffs: `ls_reversal_screen.py` (new) ONLY. `floor_prices.csv`,
`edgar_xbrl_fundamentals.csv`, `lazy_prices_similarity.csv` are read-only inputs.

## 6. Hermes task breakdown

Dispatch order: **T1 → T2 → T3 → T4** (T3 needs T1's cfg and T2's fields; T4 needs T3).
Each task = one Hermes task file in repo root with a short slash-free pointer string.
Judge the TREE after each dispatch; one retry on Codex network reconnect loops.

**Every task file MUST carry this line (A1):** "Do NOT execute
`ls_reversal_screen.py` as a script or call `run_screen`/`main` in any test or
verification step — real-data execution is a one-shot ceremony owned by the operator.
Verify via pytest on synthetic data only."

### T1 — Prereg surface (TDD)

**Files:** create `apps/quant/advisor/research/ls_reversal_prereg.py`,
`apps/quant/advisor/tests/test_ls_reversal_prereg.py`

- [ ] **Step 1: failing test** (mirror `test_candidate_prereg_fundamental.py`):

```python
from dataclasses import replace

import pytest

from advisor.research.ls_reversal_prereg import (
    DEFAULT_LS_REVERSAL, LongShortReversalPreReg, ls_reversal_hash, ls_reversal_run_hash,
)


def test_frozen_values():
    cfg = DEFAULT_LS_REVERSAL
    assert cfg.families == ("value", "fundamental_value", "lazy_prices")
    assert cfg.published_precost_ir == (-0.41, -0.32, -0.40)
    assert cfg.reproduction_tolerance == 0.02
    assert cfg.borrow_rate_annual == 0.005
    assert cfg.short_rebate_annual == 0.0
    assert cfg.cost_per_turn == 0.0005
    assert cfg.tau_ls == 0.20
    assert cfg.pass_rule == "at_least_2_of_3_ge_tau"
    assert cfg.fundamental_fixture.endswith("edgar_xbrl_fundamentals.csv")
    assert cfg.lazy_prices_fixture.endswith("lazy_prices_similarity.csv")


def test_hash_stable_and_sensitive():
    h0 = ls_reversal_hash(DEFAULT_LS_REVERSAL)
    assert ls_reversal_hash(DEFAULT_LS_REVERSAL) == h0
    assert ls_reversal_hash(replace(DEFAULT_LS_REVERSAL, tau_ls=0.19)) != h0


def test_run_hash_binds_input_bytes(tmp_path):
    p = tmp_path / "panel.csv"; p.write_text("a")
    f = tmp_path / "fix.csv"; f.write_text("b")
    h0 = ls_reversal_run_hash(DEFAULT_LS_REVERSAL, p, f)
    f.write_text("c")
    assert ls_reversal_run_hash(DEFAULT_LS_REVERSAL, p, f) != h0


def test_dataclass_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_LS_REVERSAL.tau_ls = 0.0
```

- [ ] **Step 2:** run `pytest apps/quant/advisor/tests/test_ls_reversal_prereg.py -v`
      (PYTHONPATH=apps/quant) → FAIL: module not found.
- [ ] **Step 3: implementation:**

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LongShortReversalPreReg:
    """Gate-1 kill screen for the Decision-5 L/S reversal lane — a SEPARATE hash
    surface from PreRegConfig (floor-frozen), CandidatePreReg (578cce4b…) and the
    reading surfaces. Freezes the reversed-book cost overlay and kill rule only.
    PASS authorizes designing Gate 2; it NEVER asserts alpha (in-sample screen).

    Revisable until LS_REVERSAL_PREREG.md pins its hash; frozen thereafter — any
    change is a NEW prereg filename and the original outcome stands."""
    families: tuple[str, ...] = ("value", "fundamental_value", "lazy_prices")
    published_precost_ir: tuple[float, ...] = (-0.41, -0.32, -0.40)
    reproduction_tolerance: float = 0.02
    borrow_rate_annual: float = 0.005
    short_rebate_annual: float = 0.0
    cost_per_turn: float = 0.0005          # mirrors floor PreRegConfig.cost_per_turn
    hedge_cost_model: str = "static_ols_zero_rebalance"
    tau_ls: float = 0.20
    pass_rule: str = "at_least_2_of_3_ge_tau"
    trading_days: int = 252
    holdout_frac: float = 0.2              # dev folds only; holdout stays blinded
    panel: str = "apps/quant/advisor/tests/fixtures/floor_prices.csv"
    fundamental_fixture: str = "apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv"
    lazy_prices_fixture: str = "apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv"
    value_cfg: str = "CandidatePreReg/DEFAULT_CANDIDATE"
    fundamental_cfg: str = "FundamentalCandidatePreReg/DEFAULT_FUNDAMENTAL_CANDIDATE"
    lazy_prices_cfg: str = "LazyPricesCandidatePreReg/DEFAULT_LAZY_PRICES_CANDIDATE"


DEFAULT_LS_REVERSAL = LongShortReversalPreReg()


def ls_reversal_hash(cfg: LongShortReversalPreReg) -> str:
    """Methodology-only id (no input bytes)."""
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()


def ls_reversal_run_hash(cfg: LongShortReversalPreReg, panel_path, *fixture_paths) -> str:
    """Run id: canonical config JSON THEN panel bytes THEN each fixture's bytes, in
    order. Binding input bytes detects a panel/fixture swap between freeze and run."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(panel_path).read_bytes())
    for fp in fixture_paths:
        h.update(Path(fp).read_bytes())
    return h.hexdigest()
```

- [ ] **Step 4:** re-run the test file → PASS.
- [ ] **Step 5:** commit `feat(ls-gate1): LongShortReversalPreReg separate prereg surface`.

### T2 — Expose turnover + gross exposure from the sweep (TDD, ADDITIVE only)

**Files:** modify `apps/quant/advisor/research/candidate_pipeline.py`,
`apps/quant/advisor/tests/test_candidate_pipeline.py`

- [ ] **Step 1: failing test** (append to `test_candidate_pipeline.py`; reuse its
      existing `_panel()`/fixture helpers for the panel):

```python
def test_sweep_exposes_turnover_and_gross_exposure():
    # single-family sweep, same _panel()/raw_metric setup as the test above
    res = run_dev_sweep_ext(_panel(), ("momentum",), PreRegConfig(),
                            raw_fn=raw_metric, holdout_frac=0.2)
    assert len(res.ensemble_test_turnover) == len(res.ensemble_test_returns)
    assert len(res.ensemble_test_gross) == len(res.ensemble_test_returns)
    assert (res.ensemble_test_turnover >= 0).all()
    assert (res.ensemble_test_gross >= 0).all()


def test_turnover_reconstructs_gross_from_net():
    # net + turnover*c must equal the zero-cost book on identical weights (book.py mirror)
    import pandas as pd
    from advisor.backtest.book import book_returns
    idx = pd.RangeIndex(6)
    prices = pd.DataFrame({"A": [10, 11, 12, 11, 12, 13], "B": [20, 19, 21, 22, 21, 23]}, index=idx)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.0, 0.5, 0.5, 0.5], "B": [0.5, 0.0, 0.5, 0.5, 0.5, 0.0]}, index=idx)
    c = 0.0005
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    pd.testing.assert_series_equal(
        book_returns(w, prices, c) + turnover * c,
        book_returns(w, prices, 0.0),
        atol=1e-12, rtol=0.0, check_names=False,
    )
```

- [ ] **Step 2:** run → FAIL (`SweepResultExt` has no `ensemble_test_turnover`).
- [ ] **Step 3: implementation.** In `SweepResultExt` add (defaults keep existing
      constructors valid — ADDITIVE, frozen dataclass):

```python
    ensemble_test_turnover: pd.Series = None
    ensemble_test_gross: pd.Series = None
```

In the fold loop of `run_dev_sweep_ext` (after `ens_r = book_returns(...)`,
`candidate_pipeline.py:72`), accumulate byte-for-byte mirrors of `book.py`:

```python
        turn_parts.append(ens_w.diff().abs().sum(axis=1).fillna(ens_w.abs().sum(axis=1)))
        gross_parts.append(ens_w.shift(1).fillna(0.0).abs().sum(axis=1))
```

(initialize `turn_parts, gross_parts = [], []` beside `ens_parts`), and in the return:

```python
        ensemble_test_turnover=pd.concat(turn_parts, ignore_index=True) if turn_parts else pd.Series(dtype=float),
        ensemble_test_gross=pd.concat(gross_parts, ignore_index=True) if gross_parts else pd.Series(dtype=float),
```

- [ ] **Step 4:** run the new tests AND `pytest apps/quant/advisor/tests/test_candidate_golden_replication.py -v`
      → all PASS (returns streams must be byte-identical; additive fields only).
- [ ] **Step 5:** commit `feat(ls-gate1): expose per-day turnover and gross exposure from run_dev_sweep_ext`.

### T3 — Screen module (TDD)

**Files:** create `apps/quant/advisor/backtest/ls_reversal_screen.py`,
`apps/quant/advisor/tests/test_ls_reversal_screen.py`

- [ ] **Step 1: failing tests:**

```python
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from advisor.backtest.ls_reversal_screen import decide, reversed_net_stream
from advisor.backtest.residual_screen import resid
from advisor.backtest.stats import book_sharpe
from advisor.research.ls_reversal_prereg import DEFAULT_LS_REVERSAL

RNG = np.random.default_rng(7)


def _streams(n=500):
    spy = pd.Series(RNG.normal(0.0004, 0.01, n))
    net = pd.Series(0.9 * spy.values + RNG.normal(-0.0002, 0.004, n))  # negative-alpha book
    turnover = pd.Series(RNG.uniform(0.0, 0.2, n))
    gross = pd.Series(RNG.uniform(0.8, 1.0, n))
    return net, turnover, gross, spy


def test_zero_cost_identity():
    net, turnover, gross, spy = _streams()
    cfg = replace(DEFAULT_LS_REVERSAL, cost_per_turn=0.0, borrow_rate_annual=0.0)
    rev, a = reversed_net_stream(net, turnover, gross, spy, cfg)
    res, a2 = resid(net, spy)
    assert a == a2
    assert book_sharpe(rev) == pytest.approx(-book_sharpe(res), abs=1e-12)


def test_hand_computed_costs():
    net = pd.Series([0.01, -0.01]); spy = pd.Series([0.0, 0.0])
    turnover = pd.Series([0.1, 0.2]); gross = pd.Series([1.0, 1.0])
    cfg = replace(DEFAULT_LS_REVERSAL, trading_days=252)
    rev, a = reversed_net_stream(net, turnover, gross, spy, cfg)
    b = 0.005 / 252
    # rev = -net - 2*turn*c - gross*b + a*spy (spy=0 kills the hedge term)
    assert rev.iloc[0] == pytest.approx(-0.01 - 2 * 0.1 * 0.0005 - b)
    assert rev.iloc[1] == pytest.approx(+0.01 - 2 * 0.2 * 0.0005 - b)


def test_costs_are_monotone_drag():
    net, turnover, gross, spy = _streams()
    lo, _ = reversed_net_stream(net, turnover, gross, spy, DEFAULT_LS_REVERSAL)
    hi, _ = reversed_net_stream(net, turnover, gross, spy,
                                replace(DEFAULT_LS_REVERSAL, borrow_rate_annual=0.05))
    assert book_sharpe(hi) < book_sharpe(lo)


def _family(precost, postcost):
    return {"precost_ir": precost, "postcost_reversed_ir": postcost, "beta": 0.9}


def test_tripwire_abort_suppresses_postcost():
    fams = {"value": _family(-0.10, 0.5),                       # drifted vs published -0.41
            "fundamental_value": _family(-0.32, 0.5),
            "lazy_prices": _family(-0.40, 0.5)}
    out = decide(fams, DEFAULT_LS_REVERSAL)
    assert out["verdict"] == "ABORT"
    assert all("postcost_reversed_ir" not in f for f in out["families"].values())


def test_pass_needs_two_of_three_at_tau():
    ok = {"value": _family(-0.41, 0.20), "fundamental_value": _family(-0.32, 0.20),
          "lazy_prices": _family(-0.40, 0.05)}
    assert decide(ok, DEFAULT_LS_REVERSAL)["verdict"] == "PASS"
    one = {"value": _family(-0.41, 0.35), "fundamental_value": _family(-0.32, 0.10),
           "lazy_prices": _family(-0.40, 0.10)}
    assert decide(one, DEFAULT_LS_REVERSAL)["verdict"] == "CLOSED"
```

- [ ] **Step 2:** run → FAIL (module not found).
- [ ] **Step 3: implementation:**

```python
from __future__ import annotations

import pandas as pd

from advisor.backtest.residual_screen import resid
from advisor.backtest.stats import book_sharpe
from advisor.research.ls_reversal_prereg import LongShortReversalPreReg


def reversed_net_stream(net: pd.Series, turnover: pd.Series, gross_exposure: pd.Series,
                        spy: pd.Series, cfg: LongShortReversalPreReg):
    """Post-cost reversed hedged stream. book.py returns NET of one-way costs, so a
    bare sign flip turns costs into gains: recover gross (+turn*c), negate, then
    charge the mirrored trades' own costs — hence the 2x. Borrow accrues on the held
    short notional; hedge is static OLS beta (resid convention), zero rebalance cost."""
    assert len(net) == len(turnover) == len(gross_exposure) == len(spy)
    _, a = resid(net, spy)
    carry = (cfg.borrow_rate_annual - cfg.short_rebate_annual) / cfg.trading_days
    rev = (-net.reset_index(drop=True)
           - 2.0 * turnover.reset_index(drop=True) * cfg.cost_per_turn
           - gross_exposure.reset_index(drop=True) * carry
           + a * spy.reset_index(drop=True))
    return rev, a


def decide(families: dict, cfg: LongShortReversalPreReg) -> dict:
    """Frozen rule. Tripwire first: every family's precost_ir must reproduce the
    published value within tolerance, else ABORT with post-cost outputs SUPPRESSED
    (the tripwire compares only to already-published numbers — no outcome peek)."""
    published = dict(zip(cfg.families, cfg.published_precost_ir))
    drifted = [f for f in cfg.families
               if abs(families[f]["precost_ir"] - published[f]) > cfg.reproduction_tolerance]
    if drifted:
        redacted = {f: {k: v for k, v in s.items() if k != "postcost_reversed_ir"}
                    for f, s in families.items()}
        return {"verdict": "ABORT", "drifted": drifted, "families": redacted,
                "tau_ls": cfg.tau_ls}
    survivors = [f for f in cfg.families
                 if families[f]["postcost_reversed_ir"] >= cfg.tau_ls]
    verdict = "PASS" if len(survivors) >= 2 else "CLOSED"
    return {"verdict": verdict, "survivors": survivors, "families": families,
            "tau_ls": cfg.tau_ls}
```

- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(ls-gate1): reversed-stream cost overlay + frozen verdict rule`.

### T4 — CLI wiring, hash pin, TODOS (TDD where testable)

**Files:** modify `apps/quant/advisor/backtest/ls_reversal_screen.py` (add `run_screen` +
`main`), create `apps/quant/advisor/research/LS_REVERSAL_PREREG.md`, modify `TODOS.md`,
append a pinned-hash test to `test_ls_reversal_prereg.py`.

- [ ] **Step 1:** append `run_screen(cfg)` + `main()` to the screen module — wiring
      copied from `ai-logs/hermes/diag_residual_screen_bc.py` (the T0.2c ceremony):

```python
def _readings(cfg):
    """family -> (panel, family_cfg, raw_fn). Each family runs on its OWN frozen
    reading surface, single-family sweep, dev folds only (how the published numbers
    were measured: A via residual_screen.main, B/C via diag_residual_screen_bc.py)."""
    import pandas as pd
    from advisor.backtest.residual_screen import _default_raw_fn
    from advisor.data.edgar_xbrl_fixture import load_fixture
    from advisor.research.candidate_prereg import DEFAULT_CANDIDATE
    from advisor.research.candidate_prereg_fundamental import DEFAULT_FUNDAMENTAL_CANDIDATE
    from advisor.research.candidate_prereg_lazy_prices import DEFAULT_LAZY_PRICES_CANDIDATE
    from advisor.research.fundamental_value import build_fundamental_panel, make_fundamental_raw
    from advisor.research.lazy_prices import build_lazy_prices_panel, make_lazy_prices_raw

    panel = pd.read_csv(cfg.panel, index_col=0, parse_dates=True)
    cfg_a = DEFAULT_CANDIDATE
    rb = load_fixture(cfg.fundamental_fixture)
    rc = load_fixture(cfg.lazy_prices_fixture)
    cfg_b, cfg_c = DEFAULT_FUNDAMENTAL_CANDIDATE, DEFAULT_LAZY_PRICES_CANDIDATE
    return {
        "value": (panel, cfg_a, _default_raw_fn(cfg_a)),   # reuse, don't duplicate (A3)
        "fundamental_value": (panel, cfg_b,
            make_fundamental_raw(build_fundamental_panel(rb, panel, warmup=cfg_b.warmup))),
        "lazy_prices": (panel, cfg_c,
            make_lazy_prices_raw(build_lazy_prices_panel(rc, panel, warmup=cfg_c.warmup))),
    }


def run_screen(cfg) -> dict:
    from advisor.backtest.residual_screen import spy_dev_stream
    from advisor.research.candidate_pipeline import run_dev_sweep_ext
    families = {}
    for family, (panel, fam_cfg, raw_fn) in _readings(cfg).items():
        sweep = run_dev_sweep_ext(panel, (family,), fam_cfg, raw_fn=raw_fn,
                                  holdout_frac=cfg.holdout_frac)
        spy = spy_dev_stream(panel, fam_cfg, cfg.holdout_frac)
        net = sweep.ensemble_test_returns
        res, _ = resid(net, spy)
        rev, a = reversed_net_stream(net, sweep.ensemble_test_turnover,
                                     sweep.ensemble_test_gross, spy, cfg)
        families[family] = {"precost_ir": book_sharpe(res), "beta": a,
                            "postcost_reversed_ir": book_sharpe(rev)}
    return decide(families, cfg)


def main() -> None:
    from advisor.research.ls_reversal_prereg import (
        DEFAULT_LS_REVERSAL, ls_reversal_hash, ls_reversal_run_hash)
    cfg = DEFAULT_LS_REVERSAL
    print(f"methodology_hash {ls_reversal_hash(cfg)}")
    print(f"run_hash {ls_reversal_run_hash(cfg, cfg.panel, cfg.fundamental_fixture, cfg.lazy_prices_fixture)}")
    out = run_screen(cfg)
    for family, s in out["families"].items():
        post = s.get("postcost_reversed_ir")
        post_txt = f"{post:.4f}" if post is not None else "SUPPRESSED"
        print(f"{family} | precost {s['precost_ir']:.4f} | beta {s['beta']:.4f} | postcost_rev {post_txt}")
    print(f"VERDICT | {out['verdict']} (tau_ls={out['tau_ls']}, rule 2-of-3)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** write `apps/quant/advisor/research/LS_REVERSAL_PREREG.md` mirroring
      `READING_B_PREREG.md`'s ceremony: the frozen field table from §3, the computed
      `ls_reversal_hash(DEFAULT_LS_REVERSAL)` pinned as a literal, the kill rule, the
      one-shot/ABORT semantics, and "any change = new filename, original outcome stands."
- [ ] **Step 3:** append the pinned-hash test to `test_ls_reversal_prereg.py`:

```python
def test_pinned_hash_matches_md():
    from pathlib import Path
    md = Path("apps/quant/advisor/research/LS_REVERSAL_PREREG.md").read_text(encoding="utf-8")
    assert ls_reversal_hash(DEFAULT_LS_REVERSAL) in md
```

- [ ] **Step 3b (A2): wiring test** in `test_ls_reversal_screen.py` — builds the
      readings dict (fixtures load, panels build) but computes NO returns, so it leaks
      no outcome:

```python
def test_readings_wiring_builds_without_running_any_sweep():
    from advisor.backtest.ls_reversal_screen import _readings
    r = _readings(DEFAULT_LS_REVERSAL)
    assert set(r) == set(DEFAULT_LS_REVERSAL.families)
    for family, (panel, fam_cfg, raw_fn) in r.items():
        assert "SPY" in panel.columns
        assert callable(raw_fn)
```

- [ ] **Step 4:** update `TODOS.md` "P3 — Long/short book diagnostics" gate line:
      Decision 5 PICKED 2026-07-06 (this plan); now **gated on Gate-1 PASS** — build in
      the Gate-2 slice, still not speculatively.
- [ ] **Step 5:** run the full suite (`PYTHONPATH=apps/quant pytest apps/quant -q`);
      expected: prior count (392) + new tests, zero regressions. Commit
      `feat(ls-gate1): screen CLI, LS_REVERSAL_PREREG.md hash pin, TODOS gate update`.

## 7. Freeze + run ceremony (order is the point)

1. **Freeze #1 (this doc):** commit this plan on `exec/decision5-ls-gate1` BEFORE
   dispatching T1 — the kill rule is frozen at that commit hash.
2. Dispatch T1–T4 via Hermes; judge the tree after each.
3. **Freeze #2:** LS_REVERSAL_PREREG.md pins the methodology hash (T4); its values must
   match §3 byte-for-byte — any mismatch is a plan bug, fix toward §3, never away.
4. Postflight (§8) → PR → **operator merges** (self-merge classifier-blocked).
5. **The one-shot run (post-merge, separate step):**
   `PYTHONPATH=apps/quant python apps/quant/advisor/backtest/ls_reversal_screen.py`
   — exactly once; record stdout (hashes + per-family numbers + verdict) in
   `docs/superpowers/notes/2026-07-XX-ls-reversal-gate1-result.md`; operator records the
   ruling. ABORT → fix wiring, rerun allowed. PASS → design Gate 2 (§10). CLOSED → lane
   closed, negative recorded, insider Form-4 lane (queued 2nd) comes up for ruling.
   **Tests on synthetic data are not "runs"; only this real-data execution is.**

## 8. Postflight checklist (before the PR)

- [ ] Full suite passes; count rose from 392 by exactly the new tests (record the number).
- [ ] `node tools/run-floor.mjs --enforce` exits **1** (healthy DEV_FAILED state).
- [ ] Verdict pins byte-identical: 0.7323 / 0.7562 / 0.8277.
- [ ] `git diff main --stat`: `backtest/**` shows ONLY `ls_reversal_screen.py` (new);
      `candidate_pipeline.py` diff is the two additive fields + two accumulators only.
- [ ] `test_candidate_golden_replication.py` green (streams byte-identical).
- [ ] Secret scan clean; no fixture/holdout/ledger changes; `.claude/settings.local.json`
      dirt never committed.
- [ ] PR from `exec/decision5-ls-gate1`; operator merges.

## 9. Rails (inherited, non-negotiable)

- NEVER add fields to `PreRegConfig` (immutable PREREG.md hash); frozen `backtest/pipeline.py`,
  `blend.py`, PREREG.md, UNIVERSE_RULE.md, floor_prices.csv untouched.
- Holdout operator-locked and blinded; ledger stays empty; report-only everywhere; the
  floor gates release, not commits.
- No threshold may be relaxed after freeze; decide off recorded rule outputs, not vibes.
- Do NOT touch Q3/P2 observability (tripwire-gated) or the diagnostics kill-criterion
  evaluation (due ~2026-09-04, operator-reported runs).

## 10. Gate-2 sketch (design ONLY on Gate-1 PASS — commitments frozen now)

- **Separate surface:** `LongShortCandidatePreReg` (new file + MD pin; never PreRegConfig).
- **State machine:** port pattern from virattt/ai-hedge-fund `src/backtesting/portfolio.py`
  (weighted-avg cost basis, proportional margin release, realized-gain separation) into
  `apps/quant/advisor/research/ls_portfolio.py` — keeping the frozen floor path untouched —
  and it MUST add what the external repo lacks: borrow cost accrual, transaction costs,
  maintenance margin (Reg-T 50% initial / 25% maintenance), short-proceeds cash convention.
- **Test triad:** long-only / long-short / short-only integration tests (their structure,
  our costs).
- **Gate semantics:** market-neutral book ⇒ "beat SPY" is the wrong bar; prereg a deflated
  LCB(IR) > 0 rule with declared trials N, one-shot dev run, holdout blinded.
- **Unlocks:** TODOS.md L/S diagnostics (gross/net, sleeves, abs-weight concentration) and
  external-review items 3/4 land inside this slice.

## 11. Eng review record (2026-07-06, pre-freeze)

Amendments applied: **A1** no-execution guard (one-shot cannot be burned by a test or
Hermes verification), **A2** `_readings` wiring test (no returns computed), **A3** reuse
`residual_screen._default_raw_fn`, **A4** pinned `book.py` day-1 turnover quirk (mirror,
never fix), **A5** fixture paths promoted to frozen prereg fields.

**Coverage map (every new branch → its test):**

```
reversed_net_stream()      zero-cost identity ✓  hand-computed terms ✓  monotone drag ✓
decide()                   ABORT+suppression ✓   PASS at exactly tau ✓  1-survivor CLOSED ✓
SweepResultExt new fields  length/sign ✓         net+turn*c == gross reconstruction ✓
prereg surface             frozen values ✓  hash stable/sensitive ✓  run-hash binds bytes ✓
                           frozen dataclass ✓    MD pin matches ✓
_readings wiring           builds, no sweep ✓
run_screen on real data    [NO TEST — BY DESIGN] one-shot ceremony §7.5 only (A1)
```

**NOT in scope:** Gate-2 state machine / L/S candidate prereg / test triad (§10, PASS-gated);
L/S diagnostics TODO (Gate-1 PASS-gated); any `book.py`/floor change (A4); insider Form-4
lane (queued 2nd); constant-mix & broker ingestion (Decision-2 kill-criterion-gated).

**What already exists and is reused:** `residual_screen.resid`/`spy_dev_stream`/
`_default_raw_fn`, `run_dev_sweep_ext`, the B/C reading builders + fixtures, the
candidate-prereg hash/MD-pin ceremony, `book_sharpe`.

**Failure modes:** silent cost-sign error → hand-computed test (T3); mirror drift →
reconstruction test (T2); fixture swap → run-hash + tripwire ABORT; accidental real-data
run → A1 guard; length misalignment → asserts in `reversed_net_stream`. No silent-failure
path without a test or a loud abort.

## 12. Risks

1. **Snooping (highest):** in-sample sign flip. Contained: Gate 1 is futility-only; Gate 2
   is one-shot + deflated; PASS language is frozen as "authorizes Gate-2 design" only.
2. **Cost-mirror drift:** turnover/gross expressions in T2 must mirror `book.py:14`
   byte-for-byte; reconstruction identity test is the tripwire.
3. **Sign errors in the 2× re-charge:** hand-computed-values test (T3) pins each term.
4. **Fixture drift vs published numbers:** run hash binds panel+fixture bytes; the ±0.02
   reproduction tripwire ABORTs on drift with post-cost outputs suppressed.
5. **Scope creep into the floor:** file map + postflight diff check bound the blast radius.
