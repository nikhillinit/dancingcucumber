# Handoff — draft "Plan 4: v2 calibration" (fresh session)

**Date:** 2026-06-15 · **Author:** prior session (Workstreams A & B complete, pushed `origin/main` @ `12b569c`).
**Your mission:** write **"Plan 4 — v2 calibration"** with `superpowers:brainstorming` → `superpowers:writing-plans`. This handoff is *context assembly only* — do the actual v2 design in-session; do not pre-bake it here.

**Read first (in order):**
1. This file.
2. Spec §6, §7, §8, §15: `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md`.
3. Memories: `advisor-v1-fails-floor` (load-bearing), `advisor-architecture-decision`, `hermes-dispatch-windows`, `docker-wsl-roundtrip`.
4. Prior handoff (A & B closed): `docs/superpowers/plans/2026-06-15-followups-handoff.md` (Workstream D = the work you're planning).
5. Plan 3: `docs/superpowers/plans/2026-06-15-ai-advisor-families-personas-floor.md`.

---

## 1. Current state (precise)
- **v1 mechanics are built and green:** 5 families, persona overlay, asyncio pipeline, deterministic risk+allocator, the floor harness, TimescaleDB checkpoint. `advisor-gate` = **62 passed** (report mode, exit 0).
- **v1 is NOT production-ready and that is by design.** v1 ships **equal-weight** (spec §8: "you cannot weight on skill you have not measured OOS"). It **fails its own §7 floor** on the committed real fixture.
- **Floor numbers (purged walk-forward OOS Sharpe, fixture `apps/quant/advisor/tests/fixtures/floor_prices.csv`, 2018–2023, 6 liquid large-caps AAPL/MSFT/JNJ/JPM/XOM/PG + SPY — spans the 2020 crash + 2022 drawdown = §7's ≥2 regimes):**
  - ensemble = **0.32**, SPY buy-and-hold = **0.85**, momentum = **0.34**, trend = **0.48**.
  - Fails **both** §7 conditions: not > SPY (0.32 < 0.85) and not > best single family (0.32 < trend 0.48).
- **Workstreams A (repo cleanup) and B (TimescaleDB round-trip / Plan 3 Task 8) are DONE & pushed.** Plan 3 is fully closed except its optional env-guarded checkpoint test (Workstream B step 5).

## 2. The TWO equal-weight seams — do not conflate them (the #1 ambiguity)
v2 skill-weighting touches both, but they are different code over different signals:

1. **`apps/quant/advisor/backtest/data_floor.py` → `floor_metrics()`** — the **price-only proxy**: `ensemble_sig = np.sign(_momentum_signal(p) + _trend_signal(p))` (line ~44), per-ticker long-flat, averaged across the 6 names. **This is the ONLY thing that moves the §7 floor.** Floor-closing work lives here.
2. **`apps/quant/advisor/portfolio/allocator.py` → `ensemble_vote()`** — the **live path**: confidence-weighted vote across the 5 families' `FamilySignal`s; **currently ignores `FamilySignal.skill_weight`** (already in the schema, defaults 1.0). This is the production decision seam — but per `advisor-v1-fails-floor`, **the CLI today ships value/quality ONLY** (the full 5-family wiring is Workstream C), so changing `ensemble_vote` barely touches any live decision yet. Largely academic until C lands.

State this split explicitly in Plan 4 so tasks don't aim skill-weighting at the wrong seam.

## 3. The goal — spec §8 (quote verbatim; let the session design the rest)
> v2 adds **shrinkage-to-equal-weight** skill-weighting: rolling **rank-IC / information-ratio** per family, **cap any single family at 1/N–2/N**, require a **minimum OOS window** before any weight deviates from equal, and **gate inclusion on Brier improvement over base rate** — not raw IC. In-sample weighting is overfitting with extra steps.

Hard constraint from §7.2: **weights must be chosen *before* the test window** (train folds only → apply to test folds). Any in-sample fit is "overfitting with extra steps" and disqualifies the result.

## 4. What "closing the floor" requires — and the empirical-first framing
The §7 floor (in `tools/floor_data_check.py` → `data_floor.floor_metrics` → `backtest/floor.beats_floor`) needs **both**:
- **§7.1 beat SPY** by a pre-registered margin (currently `MARGIN = 0.0` in `floor_data_check.py`; spec §15 says the margin is *"to be set before the first gated run"* — still open).
- **§7.2 beat the parts** (ensemble > best single family) across ≥2 regimes.

**Do NOT pre-conclude feasibility — make it Plan 4's first task.** Two distinct outlooks, both to be measured, not assumed:
- **§7.2 (beat the parts) = the plausibly-achievable v2 win.** Down-weighting weak momentum (0.34) toward trend (0.48) should lift the blend toward/above trend-alone. *Hypothesis to verify, not a guarantee.*
- **§7.1 (beat SPY 0.85) = open empirical question.** Note the objects differ: `ensemble`/`best_family` are **means of per-ticker long-flat Sharpes** (`data_floor.py:49`), while SPY 0.85 is **index buy-and-hold** (`signal=1.0`). A diversified 6-name book beats the average of its names' Sharpes (corr < 1), and mega-caps (AAPL/MSFT) may individually out-Sharpe SPY — the mean is dragged by laggards (XOM/PG/JNJ/JPM), not purely by long-flat timing. So whether any momentum/trend weighting clears §7.1 turns on cross-sectional interaction you **cannot reason out** — Plan 4 must measure it.

**If §7.1 is structurally unreachable on price-only long-flat, the honest conclusion stays "the advisor reports but does not size real capital" — never weaken the margin to force green.**

## 5. Realism rails (from `advisor-v1-fails-floor` — load-bearing, non-negotiable)
- **Necessary-not-sufficient:** the floor backtests only the **2-family price-only proxy**. macro/sentiment/value-quality **cannot** be purged of look-ahead on free data (spec §6). So even a future floor PASS would **not** prove the 5-family advisor satisfies §7. Carry this caveat in Plan 4.
- **No green-washing:** do NOT cherry-pick fixture/window or weaken the pre-registered margin to force a pass. Report any not-ready result as the **lead finding**, never a footnote.
- **Gate split (do not weaken):** `advisor-gate` = report-only (exit 0, "floor: NOT CLEARED…"); `advisor-release-gate` / `node tools/run-floor.mjs --enforce` exits 1 on a miss — it currently blocks correctly. The floor gates **production release, not dev commits**.

## 6. Operating discipline (every implementation task in Plan 4)
- **Workflow contract:** Claude does intake/context/planning/verification; every edit/test is dispatched via **Hermes solo** — `npm run hermes:production -- --task "..."`. Always include in the task string: *"Do NOT run npm or node; verify ONLY with `python -m pytest apps/quant/advisor/tests`."* (Codex's `workspace-write` sandbox blocks npm/node/network/DB.)
- **Codex commits with `--no-verify`** (no pre-commit hook exists now, but it cannot run the npm gate). So **re-run `npm run advisor-gate` yourself from the repo root after each task** — that is what actually verifies coverage and prints the live floor verdict.
- **Sequential dispatch:** one Codex owner at a time (concurrent owners race the shared checkout). See `hermes-bulk-delete-deviation` — verify Codex's real git state; it sometimes cherry-picks bulk ops.
- **Fixture-first:** reuse `apps/quant/advisor/tests/fixtures/floor_prices.csv`; extend with per-family OOS IC over the **same purged folds** (folds=5, embargo=5 — see `backtest/floor.purged_walk_forward_sharpe`). Operator commits any new real-data artifact; Codex writes logic against the committed file.
- **TDD with fakes** for the unit path (Codex-verifiable via `data/fakes.py`); verify the live floor verdict manually (operator).
- **DB (if any task needs it):** Docker Desktop is gone; engine is in WSL2 `Ubuntu-22.04`. `npm run db:*` fail on Windows — drive compose via `wsl.exe`, one continuous session (see `docker-wsl-roundtrip`). v2 calibration is pure compute on the CSV fixture, so it likely needs **no DB**.

## 7. Exact files
- **Move the floor (primary):** `apps/quant/advisor/backtest/data_floor.py` (`floor_metrics`, `_momentum_signal`, `_trend_signal`, `_long_flat`), `apps/quant/advisor/backtest/floor.py` (`purged_walk_forward_sharpe`, `beats_floor`), `apps/quant/advisor/backtest/walk_forward.py` (`walk_forward`, `_sharpe`).
- **Floor entrypoint / margin:** `tools/floor_data_check.py` (`MARGIN`, PASS/FAIL prose, exit codes), `tools/run-floor.mjs` (`--enforce` plumbing).
- **Live seam (secondary, academic until C):** `apps/quant/advisor/portfolio/allocator.py` (`ensemble_vote`, `allocate`), `apps/quant/advisor/schemas.py` (`FamilySignal.skill_weight`), `apps/quant/advisor/pipeline/run.py` (`run_pipeline`).
- **Tests to mirror:** `apps/quant/advisor/tests/test_floor.py`, `test_walk_forward.py`, `test_allocator.py`, `test_data_floor.py`.
- **Do NOT touch:** anything under `apps/` unrelated to calibration; the persona overlay; the cleanup invariant guards.

## 8. Open questions Plan 4 must resolve
1. **The pre-registered margin (§15):** run a short sensitivity pass on the fixture and *register a margin before the first gated run*, or justify keeping 0.0. Decide and document — do not leave implicit.
2. **§7.1 feasibility:** first task — does *any* train-fold-chosen momentum/trend weighting clear §7.1 on this fixture? If no, define the honest deliverable (report-only) explicitly.
3. **How `skill_weight` flows into `ensemble_vote`** (the live seam) and whether to sequence it after Workstream C (full 5-family CLI), since it's academic until then.
4. **Where calibration lives:** a new module (e.g. `apps/quant/advisor/backtest/calibration.py`) computing per-family rank-IC/IR + Brier on purged folds, returning weights consumed by both seams.
5. **Fixture sufficiency:** 6 names / one 2018–2023 window — enough for ≥2 regimes per §7.2, but thin for cross-sectional IC. Decide whether to extend the universe/window (operator commits real data) or scope v2 to the existing fixture.

## 9. Deliverable & acceptance (for the PLAN, not the implementation)
- A written **Plan 4** under `docs/superpowers/plans/` with bite-sized, Hermes-dispatchable tasks, each with explicit verification (`python -m pytest apps/quant/advisor/tests` + operator `npm run advisor-gate`).
- Plan 4 must encode: the §8 mechanism, the train-before-test rule, the §7.1/§7.2 split with empirical-first framing, the necessary-not-sufficient caveat, the no-green-washing rails, and the open questions above with a decision or a first-task-to-decide for each.
- **Branch first** (v2 is real code change, not operator-run): `git switch -c feat/advisor-v2-calibration` before any implementation.
