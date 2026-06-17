# Lane B Candidate-Search — Implementation Handoff

> **Audience:** a fresh Claude Code session tasked with BUILDING the amended Lane B plan via
> Hermes orchestration. This handoff is the entry point — read it top-to-bottom, then execute.
> **Authoritative sources it points to (do not re-derive):** the amended plan and the debate
> findings, named in §2. Where this handoff and the plan agree, either is fine; where the plan's
> task body conflicts with an **Amendment F1–F7**, the **amendment wins** (so does this handoff).

---

## 0. Mission & the most likely outcome (read this first)

Build a **rail-safe, report-only research bench** (`apps/quant/advisor/research/`) that tests
whether an orthogonal `value` (intermediate-term reversal) signal blended with `momentum` can
beat its own parts under the floor's exact methodology — producing a **proven-or-disproven
candidate + evidence**, without touching the frozen floor.

**The expected outcome is a cheap kill at Task 6 (the orthogonality gate).** `value_lookback=270`
sits near `long_momentum`; the plan's own thesis is that `value` is probably negated-momentum in
disguise and dies cheaply at the kill-gate → you record the negative and write the Reading-B spec
stub (Task 11). **A clean negative is a valid, complete deliverable.** Do not try to make the
candidate pass; build honestly and let the gate decide.

You are NOT promoting anything. You are NOT sizing capital. The release gate stays exit 1.

---

## 1. Current state (where things stand right now)

- **Branch:** `main`. **Implementation base commit:** `94408df` (the latest debate-fix commit).
  Capture it now: `git rev-parse HEAD` → expect `94408df…`. All your work builds on top of this.
- **Not pushed, not built.** The plan + amendments + findings are committed (8 commits
  `7e1ed37`→`94408df`); **no `apps/quant/advisor/research/` package exists yet** (verify:
  `ls apps/quant/advisor/research` → not found).
- **Tests green:** `node tools/run-pytest.mjs apps/quant/advisor/tests` → **160 passed**. Your job
  raises this count; it must never fall.
- **Floor verdict:** `DEV_FAILED`; `node tools/run-floor.mjs --enforce` → **exit 1**;
  `npm run advisor-gate` → exit 0. These must be UNCHANGED at the end.
- **Frozen floor reserved holdout is pristine** (`FLOOR_RESULT.md:32` — tail never touched).

---

## 2. Read first — exact files, zero re-derivation

**Canonical plan + findings (READ FULLY):**
1. `docs/superpowers/plans/2026-06-16-lane-b-candidate-search.md` — the plan. Its
   **"## Debate-hardening amendments (F1–F7)"** section near the top OVERRIDES the task bodies.
2. `docs/superpowers/plans/2026-06-16-lane-b-debate-findings.md` — why each amendment exists,
   with the frozen-code evidence (file:line) behind it.

**Memories (in your session context; if not, read them):**
- `lane-b-candidate-plan` — the SLICE-THEN-COMPUTE landmine + bench architecture + debate summary.
- `hermes-dispatch-windows` — **how to dispatch** (the file-pointer pattern + codex quirks). §4 below condenses it.
- `validation-gate-floor-internals` — the PreRegConfig-hash landmine + how `validation.py`/`var_sr` work.
- `plan4-v2-calibration` — why a correlated long-only blend structurally can't beat its best member (the §7.2 bar).
- `deep-research-orthogonal-signals` — why `value` (timely-price) is the cheapest orthogonal lead; Reading B.

**Frozen code the bench MIRRORS / imports read-only (read before writing the mirror):**
`apps/quant/advisor/backtest/{pipeline,data_floor,validation,validation_prereg,splits,continuous_signals,dev_gate,blend,stats,universe,prereg,book,portfolio}.py`.
The mirror changes exactly ONE thing vs `pipeline.py`: `raw_metric(f,…)` → `raw_fn(f,…)`.

---

## 3. Hard rails — any violation FAILS the task

1. **Never modify anything under `backtest/`**, nor `portfolio/allocator.py` /
   `ensemble_vote` / `allocate` / `risk/limits.py`. The bench *imports* them read-only.
   `floor_prices.csv` / `PREREG.md` / `UNIVERSE_RULE.md` / `allocator.py` are guard-blocked by
   `.claude/hooks/guard-frozen-floor.mjs`.
2. **Never weaken the release gate.** `npm run advisor-gate` stays exit 0;
   `node tools/run-floor.mjs --enforce` stays **exit 1**.
3. **No fabricated data.** Reading A uses only the existing `floor_prices.csv`. Missing/NaN signal
   → flat (0), never synthesized. Reading B (fundamentals) is NOT built here (Task 11 = spec stub only).
4. **Pre-commit methodology, not outcomes.** Freeze `CANDIDATE_PREREG.md` BEFORE the eval run.
   Acceptance = the floor's gates verbatim (§7.2 LCB>0, §7.1 beat-SPY, DSR≥0.95). No new thresholds.
5. **Holdout-leakage guard.** The `floor_prices.csv` tail IS the reserved holdout. Construct/tune
   `value` on the dev split ONLY; touch the holdout **once, iff** the dev gate passes — and only via
   the **config+fixture run-hash** (Amendment F2). Every touch is logged to `HOLDOUT_LEDGER.md`.
6. **Do NOT add fields to `PreRegConfig` or `ValidationPreReg`** (both SHA-frozen). Candidate config
   lives in `CandidatePreReg`; candidate validation config in the NEW `CandidateValidationPreReg`
   (Amendment F1).

---

## 4. Execution channel — Hermes solo file-pointer dispatch

Per `hermes-dispatch-windows`. **Solo (codex owner + pytest gate) is the proven-reliable mode**
— no `--workflow`, no Claude credits needed. Codex runs `--sandbox workspace-write`; its sandbox
**blocks spawning npm/node**; large `--task` literals hit the Windows ~8191-char command-line
limit → use the **short file-pointer** pattern.

**Per code task (Tasks 1, 1b, 2, 3, 5, 7):**

1. Write the **COMPLETE** target file content(s) — the plan's code block **with the relevant
   amendment deltas already folded in** — into `ai-logs/hermes/lane-b-task-<N>.md`. Use
   *"the file must end up EXACTLY as: <full content>"* phrasing for each file (codex sometimes
   regenerates from a snippet and drops earlier content — give it the whole file).
2. Dispatch (PowerShell):
   ```powershell
   $env:PYTHONUTF8=1
   npm run hermes:production -- --task "In this repository, open the file ai-logs/hermes/lane-b-task-<N>.md and follow its instructions EXACTLY. Do NOT run npm or node; verify ONLY with: python -m pytest <test path> -q. Do NOT commit."
   ```
3. **Verify codex's REAL git state yourself** (don't trust the wrapper): `git status --short`,
   `git diff --stat` the touched files (watch for unexpected DELETIONS), then **run the test
   yourself** and confirm the suite count ROSE.
4. **Commit one-per-task** with the trailer:
   `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Triage note:** a Hermes wrapper **exit 1 after a clean commit + passing pytest is cosmetic**
(ceremony/gate tooling on Windows), not a real failure. Verify the git state + test count, not the
wrapper exit code. Kimi is broken on Windows — never reintroduce it.

**Claude-direct (operator-class, NOT Hermes):** Task 6 (run the kill-gate), Task 8 (calibrate +
freeze pre-reg + run the eval), Task 9 (decision + final rail verification), and the **RUN +
drift-reconciliation of Task 4's golden test** (writing the test file may go via Hermes; loosening
the tolerance to force a pass is forbidden).

---

## 5. Build order (Amendment F5) — the authoritative sequence with the FORK

Track each as a TodoWrite item. Build **fail-fast**: the cheap kill-gate runs BEFORE the
expensive pipeline mirror.

| Step | Task | Channel | Amendments folded in | Done when |
|------|------|---------|----------------------|-----------|
| 1 | **T1** `CandidatePreReg` | Hermes | — | `test_candidate_prereg.py` 2 passed |
| 2 | **T1b** `CandidateValidationPreReg` (NEW) | Hermes | **F1** | new test: candidate N flows to `validation["n_used"]` |
| 3 | **T2** `candidate_signals` (`value`) | Hermes | — | `test_candidate_signals.py` 3 passed |
| 4 | **T5** `orthogonality` | Hermes | **F4** (post-transform corr + Spearman) | `test_orthogonality.py` passed |
| 5 | **T6** RUN kill-gate on real fixture | **Claude-direct** | **F4, F7** | corrs recorded in `CANDIDATE_RESULT.md` |
| — | **◆ FORK ◆** | | | see below |
| 6 | **T3** `candidate_pipeline` | Hermes | — | `test_candidate_pipeline.py` passed |
| 7 | **T4** golden + element-wise equality | Hermes write / **Claude run** | **F3, F7** | `test_candidate_golden_replication.py` passed; mirror == frozen exactly |
| 8 | **T7** `candidate_floor` | Hermes | **F1, F2, F6** | `test_candidate_floor.py` passed |
| 9 | **T8** pre-register + RUN eval | **Claude-direct** | **F1, F2, F6, F7** | `CANDIDATE_PREREG.md` frozen+committed BEFORE run; verdict in `CANDIDATE_RESULT.md`; `HOLDOUT_LEDGER.md` row if tail touched |
| 10 | **T9** decision + rail verification | **Claude-direct** | — | `CANDIDATE_RESULT.md ## Decision`; rails re-verified |
| 11 | **T11** Reading-B spec stub | Claude-direct | — | `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` |

**◆ FORK after Task 6 ◆**
- **Kill-gate FAILS** (`max(|corr(value, long_momentum)|, |corr(value, mean_reversion)|) ≥ 0.40`):
  record the negative in `CANDIDATE_RESULT.md`, **SKIP steps 6–10**, jump to **T11** (Reading-B
  stub). This is the *expected* path. Then run §7 final verification.
- **Kill-gate PASSES** (`< 0.40` on both gated neighbors): proceed to steps 6–10 in order. Task 4
  (golden) is **mandatory before** Task 8 (no holdout eval until the mirror is proven faithful).

---

## 6. Per-task implementation spec — the amendment artifacts (zero-ambiguity skeletons)

The plan's existing code blocks (Tasks 1,2,3,5,7) are correct EXCEPT for the amendment deltas
below. The plan's Self-review confirms type/field consistency. Build the plan code blocks verbatim,
then apply these deltas.

### T1b — `CandidateValidationPreReg` (Amendment F1, NEW file)

Create `apps/quant/advisor/research/candidate_validation_prereg.py`. It is a **duck-typed twin** of
`ValidationPreReg` — `validation_report(returns, fam, vcfg, tstat=None)` reads exactly these six
attrs: `declared_var_sr`, `declared_trials_N`, `minbtl_max_trials`, `tstat_hurdle`, `dsr_pass`,
`psr_benchmark_sr` (`backtest/validation.py:123-131`). Its OWN hash surface — never touches
`ValidationPreReg`.

```python
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class CandidateValidationPreReg:
    """Candidate's pre-registered DSR params. Own hash surface (Amendment F1).
    Field names mirror ValidationPreReg so validation_report reads them unchanged."""
    psr_benchmark_sr: float = 0.0
    dsr_pass: float = 0.95
    tstat_hurdle: float = 3.0
    minbtl_max_trials: int = 45
    declared_trials_N: int = 45          # candidate trial count; secondary run bumps this (rail #4)
    declared_var_sr: float = 1e-4        # CALIBRATE (see T8) or justify reuse as >= measured
    effective_n_method: str = "pca"
    effective_n_floor_is_declared: bool = True

DEFAULT_CANDIDATE_VALIDATION = CandidateValidationPreReg()

def candidate_validation_hash(cfg: CandidateValidationPreReg) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=list).encode()).hexdigest()
```

Test (prove the field is LIVE, not dead): feed `dataclasses.replace(DEFAULT_CANDIDATE_VALIDATION,
declared_trials_N=90)` through `candidate_metrics` (T7) and assert `validation["n_used"]` changes
(45 → 90). This is the regression guard for the F1 dead-field bug.

### T5 — orthogonality (Amendment F4)

In addition to the plan's raw-Pearson `dev_fold_raw_corr`: add a **post-transform** fold-level
correlation (fit `fit_percentile_transform` on each fold's TRAIN rows, `apply_transform`, correlate
`value` vs neighbor scores on fold TEST rows, holdout excluded) and a **Spearman** cross-check on
the raw. The Task-6 gate decision reads the post-transform number; raw Pearson + Spearman are
reported diagnostics. High `|corr(value, momentum)|` (post-transform) demotes the gate to a
"coarse pre-filter" note, not a clean PASS.

### T7 — `candidate_floor` (Amendments F1, F2, F6)

Start from the plan's `candidate_floor.py`, then:
- **F1:** `from advisor.research.candidate_validation_prereg import DEFAULT_CANDIDATE_VALIDATION`
  and pass it (NOT `DEFAULT_VALIDATION`) to `validation_report(...)`. Keep validation
  **report-only** — do NOT fold DSR into `passes` (`passes = verdict == "PASSED"`, mirroring
  `data_floor.py:74`).
- **F2:** add `candidate_run_hash(cfg, fixture_path)` to `candidate_prereg.py`, a byte-exact mirror
  of `prereg.config_hash` (`prereg.py:44-49`):
  ```python
  from pathlib import Path
  def candidate_run_hash(cfg: CandidatePreReg, fixture_path) -> str:
      h = hashlib.sha256()
      h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
      h.update(Path(fixture_path).read_bytes())
      return h.hexdigest()
  ```
  In T8 you pass `candidate_run_hash(C, FIXTURE)` as `prereg_hash` — never the fixture-blind
  `candidate_hash()`, never an arbitrary string. (`candidate_metrics` still gates the holdout on
  `prereg_hash is not None`, mirroring `data_floor.py:34`; the run-hash is what makes that honest.)
- **F6:** emit a `power` block alongside the verdict: per-fold positive-raw count of `value`,
  nonzero-transformed coverage fraction, effective obs. If any fold's positive-train count is below
  a pre-registered floor (recommend **< 25**), set a `power_limited: True` flag so T8 labels the
  verdict power-limited rather than "Reading A exhausted."

### T8 — pre-register + run (Amendments F1, F2, F6, F7); Claude-direct

**Order is load-bearing — freeze BEFORE you run:**
1. **Calibrate `declared_var_sr` (F1).** Run the candidate trials once (value-standalone,
   momentum-standalone, value+momentum ensemble; +trend if secondary), take per-obs Sharpes
   (`validation.per_obs_sharpe`), `var_sr_trials([...])` (ddof=1). **Recommended:** if measured
   `var_sr ≤ 1e-4`, keep `declared_var_sr = 1e-4` and write the one-line justification "1e-4 ≥
   measured candidate cross-trial dispersion → stricter" (mirrors the floor's calibrate-then-round-up,
   `validation-gate-floor-internals`). Else set it to the measured value rounded UP. Freeze it in
   `CandidateValidationPreReg`. (Validation is report-only, so this never unlocks the holdout — it
   only sets the DSR bar you report.)
2. **Write & commit `CANDIDATE_PREREG.md`** (BEFORE the eval): `candidate_hash(DEFAULT_CANDIDATE)`,
   `candidate_run_hash(DEFAULT_CANDIDATE, FIXTURE)`, `candidate_validation_hash(DEFAULT_CANDIDATE_VALIDATION)`,
   the frozen constants, fixture path + SHA, candidate order (primary `value+momentum`; secondary
   `value+momentum+trend` ONLY if primary dev-passes-but-holdout-fails, which bumps
   `declared_trials_N`), acceptance = floor gates verbatim. Commit, THEN run.
3. **Run** the eval with the F7 concrete loader and the F2 run-hash (the command is already
   corrected in the plan's Task 8 Step 2).
4. If the holdout is touched (dev passed), append a row to `HOLDOUT_LEDGER.md`:
   `| date | run_hash | families | verdict | who |`.

### Fixture loader (Amendment F7 — use everywhere, no placeholders)

```python
import pandas as pd
from pathlib import Path
FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")   # NOT backtest/fixtures
def load_floor_panel():
    return pd.read_csv(FIXTURE, index_col=0, parse_dates=True)         # SPY col + 30 assets
```

### New files this implementation creates (all under `apps/quant/advisor/research/`)
`__init__.py`, `candidate_prereg.py` (+`candidate_run_hash`), `candidate_validation_prereg.py`,
`candidate_signals.py`, `candidate_pipeline.py`, `orthogonality.py`, `candidate_floor.py`,
`CANDIDATE_PREREG.md`, `CANDIDATE_RESULT.md`, `HOLDOUT_LEDGER.md`; tests under
`apps/quant/advisor/tests/test_candidate_*.py`; plus `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` (T11).

---

## 7. Definition of done — final verification ritual (run verbatim)

Whether you killed at Task 6 or ran the full eval, finish with ALL of these and record results in
`CANDIDATE_RESULT.md`:

```powershell
# 1. Tests rose and are green (was 160; expect higher)
node tools/run-pytest.mjs apps/quant/advisor/tests
# 2. Report gate unchanged
npm run advisor-gate                       # expect exit 0, prints floor: DEV_FAILED
# 3. Release gate STILL blocks
node tools/run-floor.mjs --enforce         # expect exit 1
# 4. Frozen surfaces untouched (MUST be empty)
git diff --stat 94408df..HEAD -- apps/quant/advisor/backtest apps/quant/advisor/portfolio apps/quant/advisor/risk
```

- §4 invariants must hold: `advisor-gate` exit 0, `--enforce` exit 1, the `git diff` is EMPTY.
- The new package is **report-only**: no call into `allocator.py` / `ensemble_vote` / sizing.
- Commits are one-per-task with the Co-Authored-By trailer; leave local (do not push) unless told.

---

## 8. Gotchas & do-NOTs (the traps that already bit this codebase)

- **SLICE-THEN-COMPUTE landmine:** `_family_scores` runs the raw metric on the already-sliced dev
  frame, so `value`'s `shift(270)` makes the first 270 dev rows NaN→flat. `value_lookback` MUST stay
  `<≈325` (fold-1 train end) or fold 1 dies for non-signal reasons. 270 is the pre-registered ceiling
  — do not "fix" it upward. Verified by T1's `value_lookback <= 300` assert + T4's live-in-every-fold guard.
- **Don't make `passes` depend on DSR.** Validation is report-only by design (floor parity). DSR
  ≥0.95 is an operator promotion-readiness check (T8/T9), not the machine verdict.
- **Holdout discipline:** never run `run_holdout_ext` on `floor_prices.csv` during dev/golden work.
  The F3 holdout-mirror equality check runs on a SYNTHETIC panel only. The real tail is touched once,
  iff dev passes, via the run-hash, logged to the ledger.
- **Codex quirks:** it can't spawn npm/node (tell it "verify ONLY with python -m pytest"); it may
  regenerate a file from a snippet (give it COMPLETE file content); a wrapper exit 1 after a clean
  commit + green pytest is cosmetic — verify git state + test count, not the exit code.
- **Don't loosen the golden tolerance** to force Task 4 green. A numbers mismatch means the mirror
  drifted — diff `candidate_pipeline.py` against `backtest/pipeline.py` line-by-line and reconcile.
- **Guard hook:** `.claude/hooks/guard-frozen-floor.mjs` blocks Edit/Write that OVERWRITE the frozen
  files; creating new `research/` files is fine. If you hit the guard, you're editing the wrong file.

---

## 9. One-line kickoff for the implementing session

> Read `docs/superpowers/plans/2026-06-17-lane-b-implementation-handoff.md`, then implement the
> amended Lane B plan (`2026-06-16-lane-b-candidate-search.md`) via Hermes solo file-pointer dispatch
> in the fail-fast order of §5. Stop and report at the Task-6 fork.
