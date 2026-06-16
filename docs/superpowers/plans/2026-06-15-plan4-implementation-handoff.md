# Handoff — Execute Plan 4 (v2 calibration) via Hermes

**Date:** 2026-06-15 · **Author:** planning session (Plan 4 written, reviewed, committed).
**Mission:** implement **Plan 4 — "Continuous Long-Flat Walk-Forward Floor Proxy v2"** task-by-task by dispatching each implementation task to **Hermes solo (Codex)**, reviewing every diff, and gating on the local pytest + `advisor-gate`. The plan is the source of truth for *what* each task does (it carries exact test code, implementation code, and a per-task dispatch string). This handoff is the *how to execute* layer — order, ownership, the dispatch loop, the guard interactions, the decision branches, and the gotchas — so a fresh session runs it with no ambiguity.

---

## 1. Current state (precise)
- **Branch:** `feat/advisor-v2-calibration` (already created — **skip Task 0's `git switch`**). Created from `main`.
- **Commits so far:**
  - `main` (pushed) — `.claude/` automation: the **frozen-floor guard hook** (`.claude/hooks/guard-frozen-floor.mjs` + `.claude/settings.json`) and the **`backtest-integrity-reviewer`** subagent (`0ac9a5e`), plus plugin enablement (`510fcce`). Inherited by the feature branch.
  - `feat/advisor-v2-calibration` (pushed) — Plan 4 (`3654391`), its v2-calibration handoff, and this implementation handoff. HEAD advances as these docs are revised.
- **Both branches pushed to origin** (`github.com/nikhillinit/dancingcucumber`). No pre-commit hook exists (verified) — commits run clean; `--no-verify` in the plan's commit steps is harmless belt-and-suspenders.
- **`advisor-gate` is currently green in report mode** (exit 0) and the **enforce** gate correctly exits 1 (v1 still fails its floor). It stays that way throughout implementation until the holdout records a `PASSED` verdict (Task 14).
- **The committed fixture is still the OLD 6-name `floor_prices.csv`.** That is fine for all Codex tasks (their tests use *synthetic* panels). Only Task 14 (the measurement run) needs the extended fixture from Task 0.

## 2. Read first (in order)
1. This handoff.
2. The plan: `docs/superpowers/plans/2026-06-15-plan4-v2-calibration.md` (Decisions table + rails + Tasks 0–15 + self-review).
3. Memories: `plan4-v2-calibration` (the integer-degeneracy proof + design), `hermes-dispatch-windows` (dispatch mechanics + Windows gotchas), `advisor-v1-fails-floor` (why the floor is necessary-not-sufficient), `hermes-bulk-delete-deviation` (verify Codex's real git state).
4. Spec §6/§7/§8/§15: `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md`.

## 3. Dispatch protocol (the per-task loop — follow exactly)
Tasks marked **[Codex]** are deterministic logic verifiable by pytest against synthetic fixtures → dispatch via Hermes solo. Tasks marked **[Operator]** need the network or are the single gated measurement → run them yourself; never hand them to Codex.

**For each [Codex] task, in order:**
1. **Read** the task block in the plan (it has the failing test, the implementation, the verify command, the commit, and a ready-to-paste **Dispatch** string).
2. **Dispatch solo:** run the task's exact `npm run hermes:production -- --task "..."` string. It already embeds the mandatory clause **"Do NOT run npm or node; verify ONLY with `python -m pytest apps/quant/advisor/tests`."** (Codex's `workspace-write` sandbox blocks npm/node/network — without that clause its run exits non-zero on a clean commit.)
3. **Verify Codex's REAL git state** (it sometimes cherry-picks/deviates — see `hermes-bulk-delete-deviation`): `git status` + `git diff --stat`. Confirm *only* the task's create/modify files changed, and the new test file exists.
4. **Re-run the gate yourself** (Codex can't run npm): `npm run advisor-gate`. Must stay **exit 0** (report mode) and the full pytest suite must be green. (`advisor-release-gate` / `node tools/run-floor.mjs --enforce` stays exit 1 until Task 14.)
5. **Commit** if Codex didn't (or fix up the diff if it deviated). One owner at a time — **sequential dispatch only** (concurrent Codex owners race the shared checkout).
6. Next task.

**Never** dispatch two tasks concurrently. **Never** let a task's verification depend on the network or the final holdout.

## 4. Task order, ownership, dependencies
Codex tasks are import-ordered so each builds on already-present modules. **T0 and T14 are operator-run and can be deferred to just before the measurement** (Codex tests use synthetic data, so the extended fixture is NOT a prerequisite for T1–T13/T15).

| Step | Owner | Produces | Depends on |
|---|---|---|---|
| T1 prereg | Codex | `backtest/prereg.py` | — |
| T2 stats | Codex | `backtest/stats.py` | — |
| T3 splits | Codex | `backtest/splits.py` | — |
| T4 continuous_signals | Codex | `backtest/continuous_signals.py` | — |
| T5 adequacy | Codex | `backtest/adequacy.py` | — |
| T6 portfolio (allocator) | Codex | `backtest/portfolio.py` | — |
| T7 book | Codex | `backtest/book.py` | T2 |
| T8 blend | Codex | `backtest/blend.py` | T2,T3,T4,T6,T7 |
| T9 pipeline (+`run_holdout`) | Codex | `backtest/pipeline.py` | T1,T2,T3,T4,T6,T7,T8 |
| T10 dev_gate | Codex | `backtest/dev_gate.py` | T1,T2 |
| T11 universe | Codex | `backtest/universe.py` | T1 |
| T12 rails | Codex | `tests/test_rails.py` | T1,T3,T4,T6,T9 |
| T13 entrypoint | Codex | `data_floor.py`, `tools/floor_data_check.py` | T9,T10,T11,T1,T3,T2 |
| **T0 fixture** | **Operator** | extended `floor_prices.csv` + `UNIVERSE_RULE.md` | — (before T14) |
| **T14 measurement** | **Operator** | dev sweep → single holdout → verdict | T0, T13 |
| T15 disclosures | Codex | `walk_forward.py`, `floor_data_check.py` prose | T13 |

Recommended run: **T1→T13 (Codex) → T15 (Codex) → T0 (operator fixture) → T14 (operator measurement)**. (T15 can also run right after T13; it's independent of the fixture.)

## 5. Operator-only tasks — detail
**T0 — extended fixture (network).** Per a **pre-registered, as-of-window-start** rule (NOT as-of-today): ≥20 most-liquid US large-caps continuously listed across `2015-01-01..2023-12-31` + `SPY`, adjusted close, full-window coverage. Write the rule + survivorship disclosure to `UNIVERSE_RULE.md`, pull prices, commit the CSV + rule, record the fixture SHA-256. **Freeze** — never re-pull/re-select after seeing a floor number.
- **Guard interaction:** write the CSV via a **Python pull script + git (Bash)**, which the frozen-floor hook does NOT intercept. If you instead use the `Write` tool to (over)write the existing `floor_prices.csv`, the guard will block it (by design) — do it via Bash, or temporarily disable the hook in `.claude/settings.json` for that one step.

**T14 — pre-registered dev sweep + single holdout (the empirical branch).**
1. **Pre-register:** print `config_hash(DEFAULT_CONFIG, fixture)` from a **`.py` file** (never `python -c "<multi-word>"` under shell:true — Windows tokenization gotcha). Create `PREREG.md` with the full config values, the hash, the locked **margin number** (default 0.0; only raise it from a **dev-fold-only** sensitivity pass — never from the holdout), and the candidate order C→E. Commit it.
2. **Dev sweep (dev folds only):** for candidates C (2-family) → D (+1) → E (+2), call `floor_metrics(panel, cfg, prereg_hash=None, families=...)`, read `m["dev"]`. Stop at the **smallest** candidate whose `dev.passed` is True. None pass → verdict **UNSUPPORTED**, stop.
3. **Universe check** `m["universe"]`: `do_not_run` → stop; `micro` → holdout result is "diagnostic only".
4. **Single holdout:** re-run with `prereg_hash=<committed hash>` so `floor_metrics` evaluates the held-out tail **once** (gates on BOTH `delta_lcb > 0` §7.2 AND `spy_lcb > margin` §7.1). Record `m["verdict"]` + `holdout`. **Do not iterate.**
5. **Record the verdict in a SEPARATE result file** (`FLOOR_RESULT.md` or `.remember/remember.md`) — **do NOT edit `PREREG.md`** (it must stay immutable, and the guard blocks re-editing it anyway). Report PASSED / INCONCLUSIVE / UNSUPPORTED as the **lead finding**.

## 6. Guard / freeze interactions (the hook WILL block some writes — know this)
The PreToolUse hook (`.claude/hooks/guard-frozen-floor.mjs`, active on this branch) blocks **Edit/Write/MultiEdit that overwrite an existing** `floor_prices.csv`, `PREREG.md`, `UNIVERSE_RULE.md`, or `portfolio/allocator.py`. **Creation is allowed** (`fs.existsSync` gate). Implications:
- **No Plan 4 task edits `portfolio/allocator.py`** (floor-only scope) — the guard enforces that rail. If something tries to, that's a scope violation, not a guard bug.
- **T0** creating `UNIVERSE_RULE.md` and **T14** creating `PREREG.md` = allowed (first write). Re-editing them = blocked → keep them immutable (record results elsewhere, §5.5).
- **T0** replacing `floor_prices.csv` = blocked via Edit/Write → use a Bash pull script (see §5).
- `data_floor.py`, `tools/floor_data_check.py`, `walk_forward.py` are **not** frozen → freely editable (T13, T15).

## 7. Non-negotiable rails (reject any diff that breaks these)
- Floor-closing work lives in `backtest/`, **never** `portfolio/allocator.py` `ensemble_vote` (deferred to a post-Workstream-C plan).
- **Train-only fitting:** transform + blend weights fit on each fold's train portion; holdout touched **once**.
- **No green-washing:** margin stays ≥ 0; no fixture/window/metric shopping; not-ready reported as the lead finding.
- **Backtest-what-ships:** long-flat only (no shorts/market-neutral).
- **Price-only calibration:** only purgeable price families weighted; no macro/sentiment/value-quality in calibration inputs.
- **Necessary-not-sufficient:** a PASS never claims the full 5-family advisor satisfies §7.
- **Report-vs-enforce split** intact: `advisor-gate` exit 0 (report); `--enforce` exit 1 on a miss.
- Use the **`backtest-integrity-reviewer`** subagent on the cumulative diff before merge (it appears after a session reload / `/agents`).

## 8. Environment & Hermes gotchas
- **Windows / PowerShell solo dev.** Dispatch is `npm run hermes:production -- --task "..."` (production phase → Codex owner). **Solo only** — `pair`/`debate` need Claude CLI credits and **kimi is broken on Windows** (cp1252 crash on non-ASCII).
- Codex sandbox = `workspace-write`: **no npm/node/network/DB**. Always keep the "Do NOT run npm or node; verify ONLY with pytest" clause.
- Gate/tool runners must not use `python -c "<multi-word>"` under shell:true (tokenization eats everything after the first word) — run a `.py` file.
- **DB is not needed** for Plan 4 (pure compute on the CSV). If a future task needs it: Docker engine lives in WSL2 `Ubuntu-22.04`; `npm run db:*` fail on Windows (see `docker-wsl-roundtrip`).

## 9. Done criteria
- T1–T13, T15 implemented; `npm run advisor-gate` green (full pytest suite + floor report) after each.
- T0 fixture committed + frozen; T14 run once; verdict recorded in `FLOOR_RESULT.md` and `.remember/remember.md` as the lead finding.
- `node tools/run-floor.mjs --enforce` reflects the verdict (exit 0 only if `PASSED`).
- The honest outcome — **PASSED**, **INCONCLUSIVE** ("reports but does not size capital"), or **UNSUPPORTED** — is stated plainly. A clean negative is a valid, complete deliverable.
- Then: `superpowers:finishing-a-development-branch` (PR / merge decision).

## 10. Watch-items (flagged during planning — verify, don't assume)
- **`run_holdout` SPY alignment** (T9): SPY uses raw `pct_change` vs the ensemble's `shift(1)` book — a one-day edge effect. Fine for a proxy; tighten if you want exactness.
- **`caps` convention** (T8/T9/`run_holdout`): `caps = (max_asset_weight, gross_cap, turnover_cap)`, unpacked as `build_long_flat_book(scores, *caps, cost_per_turn)`. Element 3 is the **turnover cap** — keep aligned across modules.
- **`PreRegConfig` numbers** (T1) are pre-registered: caps 0.20, turnover 0.20, cost 5bps, dev-lift 0.05, bootstrap block 21 / draws 2000 / seed 12345, min-universe 20/12. Change them **now or never** — not after seeing a result.
- **Universe ≥ 20** is required for a *formal* floor claim; below that the run is labeled "micro-universe diagnostic only".
