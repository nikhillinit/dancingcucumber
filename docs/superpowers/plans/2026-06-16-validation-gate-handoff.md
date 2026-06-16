# Handoff — Execute the Validation Gate plan via Hermes

> **You are picking up the AIHedgeFund advisor in a fresh session. Your job is to IMPLEMENT Plan 1 (the validation/deflation gate) by dispatching each task to Codex via Hermes, verifying, and committing — task by task.** The plan is fully specified with tests + impl. Do not redesign it. Read §1 (task), then §2–§6, then follow §7 (first moves).

---

## 1. Your task

Execute **`docs/superpowers/plans/2026-06-16-validation-gate.md`** task-by-task (Tasks 1→9). It builds a signal-agnostic deflation/multiple-testing **validation gate** into the backtest harness as **report-only** diagnostics: Deflated Sharpe (DSR), attempt-count N + MinBTL budget, PCA effective-N, Harvey-Liu-Zhu t-stat hurdle, purge/embargo audit — wired additively into `floor_metrics()`. It can only confirm the existing `DEV_FAILED` floor; it never flips the verdict, unlocks the holdout, or authorizes sizing.

Every task ships **tests + impl together** (TDD; the plan contains the exact test and implementation code for each step). Dispatch via Hermes (§4). This is **Plan 1 of 3** — Workstream C and the post-C signal program are explicitly out of scope (deferred to separate plans; a roadmap stub is produced in Task 9).

---

## 2. State (verified 2026-06-16 — SETTLED, do not re-litigate)

- The floor verdict is **`DEV_FAILED`** (formal 30-name large-cap universe). Accepted (Option 1). Family-reweighting lane is CLOSED. The advisor is NOT authorized for production sizing.
- The enforced floor path is: `node tools/run-floor.mjs --enforce` → `spawnSync` → `python tools/floor_data_check.py --enforce` → `floor_metrics(panel, DEFAULT_CONFIG)`. **That Python path is what you wire into.** It runs **n=2 families (`momentum`, `trend`)** — not 5. (`added_families` and constructions A–E are the meta-search.)
- `apps/quant/advisor/backtest/stats.py` has only `book_sharpe` (annualized ×√252) + block bootstrap. No skew/kurt/PSR/DSR exist — all net-new.
- `purged_splits()` ALREADY EXISTS in `splits.py` (audit it, don't rebuild).

## 3. Hard rails (violating any of these fails the task)

1. **Never add fields to `PreRegConfig`** (`backtest/prereg.py`). It is frozen and SHA-hashed into immutable `PREREG.md` (`config_hash = 1ad2ed4a…`); a new field changes the hash and breaks the recorded floor provenance. Validation params live in the **separate** `ValidationPreReg` (Task 1).
2. **Never modify `portfolio/allocator.py` / `ensemble_vote`.** Out of scope.
3. **Report-only.** The `verdict` branch in `data_floor.py:floor_metrics` is untouched; `"validation"` is an additive key. Proven by `test_floor_metrics_validation_is_additive_only`.
4. **`node tools/run-floor.mjs --enforce` must still exit 1** after every task.
5. **DSR uses per-observation Sharpe** (mean/std), never `book_sharpe` (annualized). `V[SR̂]` in `SR0` is the **cross-trial** dispersion (`declared_var_sr`), never single-strategy sampling variance.
6. The frozen-floor guard (`.claude/hooks/guard-frozen-floor.mjs`) blocks Edit/Write *overwrite* of `floor_prices.csv` / `PREREG.md` / `UNIVERSE_RULE.md` / `allocator.py` (creation is allowed).

## 4. Hermes execution mechanics (READ memory `hermes-dispatch-windows` FIRST)

- **Solo dispatch only:** `npm run hermes:production -- --task "<task text>"` (Codex owns the edit).
- **Always include in every task string:** *"Do NOT run npm or node; verify ONLY with pytest."* (Codex sandbox blocks npm/node.)
- Set `PYTHONUTF8=1` in the environment (dodges the cp1252 crash).
- **Slashy task strings** (file paths contain `/`) are blocked by the PowerShell guard → write the task instruction to a file under `ai-logs/hermes/task-N.md` and dispatch via the task-file mechanism documented in the memory. Confirm the exact flag from that memory before the first dispatch.
- **Per-task loop (sequential, one at a time):**
  1. Dispatch task N (point Codex at the plan: "implement Task N from `docs/superpowers/plans/2026-06-16-validation-gate.md` exactly — the test and impl code are given; do not deviate. Do NOT run npm or node; verify ONLY with pytest").
  2. **Verify Codex's REAL git state** (`git status`, `git show --stat`) — Codex cherry-picks/skips; trust the diff, not its summary.
  3. Run the task's own tests: `$env:PYTHONUTF8=1; python -m pytest <task's test path> -v`.
  4. If green and Codex didn't commit, commit (the plan gives the message). One task = one commit.
  5. Next task.
- **Plan-bug rule:** if a seeded assertion fails against the provided impl, that's a plan-bug — fix the fixture/threshold yourself in the plan/code, do NOT re-dispatch in a loop. The two most likely surfaces: Task 4 DSR units reconciliation, and Task 7 wiring.

## 5. Two human-in-the-loop steps Codex CANNOT do (you do these yourself)

- **Task 9 Step 3 — calibrate `declared_var_sr`.** Run `python tools/calibrate_var_sr.py` (created in Task 9 Step 2), read the printed `declared_var_sr`, set `ValidationPreReg.declared_var_sr` to it, and paste the printed lines into `VALIDATION_PREREG.md`. This constant is the gate's **dominant leniency knob** (`SR0 ∝ √var_sr`); do not ship the interim `4e-4` uncalibrated.
- **Task 9 Step 4/Step 1 — paste `validation_hash`.** Run the one-liner, paste the 64-char hash into `VALIDATION_PREREG.md`. Step 6's grep guard fails the commit if any `<paste …>` marker remains.

## 6. Task map (dependencies)

Tasks 1→8 are sequential (each appends to `validation.py` / its tests). Task 9 finalizes (calibration, pre-reg artifact, roadmap, full-suite + gate verification).

- **T1** `validation_prereg.py` — `ValidationPreReg` (separate hash) + the `PreRegConfig`-field-set guard test.
- **T2** `validation.py` — `per_obs_sharpe`, `sharpe_moments` (units pinned vs `book_sharpe`).
- **T3** PSR (independent closed-form test).
- **T4** DSR + `var_sr_trials` + `_sr0` (N≤1 guard) — independent-reference + units-reconciliation + production-var_sr tests.
- **T5** `effective_n_pca` + `n_for_dsr` (bounded below by declared N).
- **T6** `minbtl_exceeded` + `tstat_meets_hurdle`.
- **T7** `validation_report` + wire into `floor_metrics` (additive). MinBTL on `n_used`; t-stat enforced when supplied; monkeypatch independence test.
- **T8** print report-only caveat in `tools/floor_data_check.py`.
- **T9** `tools/calibrate_var_sr.py`, calibrate, `VALIDATION_PREREG.md`, `2026-06-16-deferred-plans-roadmap.md`, paste-guard, full pytest + `advisor-gate` (exit 0) + `run-floor --enforce` (exit 1).

## 7. First moves

1. Read this handoff, the plan (`docs/superpowers/plans/2026-06-16-validation-gate.md`), and memory: `validation-gate-floor-internals`, `hermes-dispatch-windows`, `plan4-v2-calibration`.
2. Confirm green baseline: `$env:PYTHONUTF8=1; python -m pytest apps/quant/advisor/tests -q` (pass); `npm run advisor-gate` (exit 0); `node tools/run-floor.mjs --enforce` (**exit 1**).
3. Dispatch **Task 1** via Hermes per §4. Verify Codex's real git state, run `python -m pytest apps/quant/advisor/tests/test_validation_prereg.py -v`, commit.
4. Proceed Task 2→8 the same way, one task per dispatch+commit.
5. Do **Task 9** yourself (calibration + hash paste are human steps); run the full-suite + both gates; confirm `--enforce` still exits 1.
6. Do NOT touch `PreRegConfig` fields or `allocator.py`; keep the release gate blocked.

## 8. Definition of done

All 9 tasks committed; `python -m pytest apps/quant/advisor/tests` green; `npm run advisor-gate` exit 0 (floor still prints `DEV_FAILED` + the new report-only validation caveat); `node tools/run-floor.mjs --enforce` exit 1; `VALIDATION_PREREG.md` committed with no `<paste>` markers and a calibrated `declared_var_sr`; `2026-06-16-deferred-plans-roadmap.md` committed.
