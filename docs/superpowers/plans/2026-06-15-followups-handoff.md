# Handoff Memo — Post-Plan-3 Follow-ups (fresh session)

**Date:** 2026-06-15
**Author:** prior session (Plan 3 complete, pushed to `origin/main` @ `6be5fde`)
**Read first:** `.remember/remember.md`; memories `advisor-v1-fails-floor`, `hermes-dispatch-windows`, `hermes-bulk-delete-deviation`, `advisor-architecture-decision`. Spec: `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md`. Plan 3: `docs/superpowers/plans/2026-06-15-ai-advisor-families-personas-floor.md`.

> **Status (updated):** ✅ **Workstream A is DONE and pushed** — `origin/main` @ `ff4fc84` (was `6be5fde`). 367 files retired (99 root scripts + 266-file nested `AIHedgeFund/` duplicate + 2 consensus files); cleanup harness extended with two invariant guards; `advisor-gate` = **62 passed**; release gate still blocks (exit 1). ✅ **Workstream B is also DONE** (2026-06-15) — Plan 3 Task 8 fully closed; round-trip verified live (`saved 2 rows; table count = 2`, idempotent). **▶ Next: Workstream D (v2 calibration) — the only path that closes the floor.**

## Operating discipline (applies to every workstream below)
- **Workflow contract:** Claude does intake/context/planning/verification; every edit/test is dispatched via **Hermes solo** — `npm run hermes:production -- --task "..."`. Always include in the task string: *"Do NOT run npm or node; verify ONLY with `python -m pytest apps/quant/advisor/tests`."* Codex's `workspace-write` sandbox blocks npm/node and all network/DB.
- **Sequential dispatch:** one Codex owner at a time — concurrent owners race on the shared checkout.
- **Codex commits with `--no-verify`** (the pre-commit hook is the npm gate it's forbidden to run). So **re-run `npm run advisor-gate` yourself from the repo root** after each task — that is what actually verifies coverage.
- **Fixture-first** for anything needing network/DB: operator (Claude) commits the real artifact; Codex writes logic against the committed file.
- **Gate split:** `advisor-gate` is report-only (exit 0); `advisor-release-gate` / `node tools/run-floor.mjs --enforce` exits 1 — it currently **blocks correctly** because v1 fails its floor. Do NOT weaken it.

---

## Workstream A — §10 repo-sprawl cleanup (extend the existing harness) — ✅ DONE

**Completed 2026-06-15, merged to `main`, pushed @ `ff4fc84`.** Commits: `008aaed` (99 root `.py`), `3bd34f0` (2 consensus files), `58208af` (266-file nested `AIHedgeFund/` duplicate + 2 invariant guard tests), `ff4fc84` (root `.gitignore` + debris/`nul` hygiene). Root `.py` = 0, `AIHedgeFund/` = 0, advisor code untouched. New guards: `test_no_tracked_python_scripts_at_repo_root`, `test_nested_aihedgefund_directory_absent`. Note (see memory `hermes-bulk-delete-deviation`): Codex twice skipped the bulk `git rm -r` and cherry-picked files; the nested-dir delete + invariant tests were done **direct** (retry-once-then-fallback) and verified by `npm run advisor-gate`.

<details><summary>Original Workstream A plan (for reference)</summary>

**State of play (verified):** The §10 *enumerated* list (`production_ready_system.py`, `robust_trading_system.py`, the finrl/qlib/autogluon stubs, etc.) is **already deleted and gated** by `apps/quant/advisor/tests/test_repo_cleanup.py` (passes in the 60-test suite). But there are **~100 other tracked `.py` files at the repo root** (e.g. 24 `*system*.py`, plus `*_integration.py`, `quick_*.py`, `test_*` at root, `fidelity_*`, `congressional_*`, etc.) that are legacy sprawl the §10 *spirit* covers but the test does not yet enumerate. **The real advisor code is isolated under `apps/quant/advisor` (50 tracked files under `apps/`) and must NOT be touched.**

**The safety harness already exists — reuse it.** `test_repo_cleanup.py` has two gates:
1. `test_retired_cleanup_targets_are_absent` — the `RETIRED_PATHS` set must not exist on disk.
2. `test_tracked_python_modules_do_not_import_retired_modules` — no *kept* tracked module may import a retired one.

So the cleanup method is mechanical and safe: **add a batch of root files to `RETIRED_PATHS`, `git rm` them, run the gate.** If any kept module (anything under `apps/`) still imports one, gate (2) fails loudly — investigate before deleting.

**Execution plan:**
- [ ] **Branch first** (this is large/destructive — do NOT do it directly on main): `git switch -c chore/repo-sprawl-cleanup`.
- [ ] **Get explicit operator sign-off on the keep/delete boundary** before mass deletion ("never break userspace"). Proposed KEEP: everything under `apps/`, `packages/`, `infra/`, `tools/`, `scripts/`, `docs/`, plus root config (`package.json`, `tsconfig*`, `drizzle*`, `*.md`). Proposed DELETE: the loose root-level `.py` scripts (legacy research/one-off systems). Confirm none is a live entry point — the advisor CLI is `python -m advisor` under `apps/quant`, so the root scripts are legacy, but **confirm**.
- [ ] **Dispatch in reviewable batches via Hermes solo.** Per batch: Codex adds the batch paths to `RETIRED_PATHS` in `test_repo_cleanup.py`, runs `git rm` on them, verifies with `python -m pytest apps/quant/advisor/tests/test_repo_cleanup.py`, commits `chore(repo): retire legacy <group> sprawl`. Keep batches grouped by theme (`*system*.py`, `quick_*`, root `test_*`, `fidelity_*`, …) so review is tractable.
- [ ] **Grep for the remaining §10 named items** as their own batch: `consensus_engine` (fake Nash/Bayesian claims), the always-positive **mock NewsAPI fallback** (the new `analysis/sentiment.py` already enforces no-fabrication, so the old root mock is pure dead code). Salvaging "debate scaffolding into the persona overlay" is **optional** — the overlay already exists; only salvage if a specific piece is genuinely useful, otherwise delete.
- [ ] **Local hygiene (not a Codex task — do directly):** remove untracked `__pycache__/`, `*.pyc`, `*.log`, and the stray `nul` file; add a root `.gitignore` for `__pycache__/`, `*.pyc`, `*.log`.
- [ ] **Verify** `npm run advisor-gate` green (60 passed + floor report) and the `apps/` suite untouched, then PR/merge the branch.

</details>

---

## Workstream B — Live TimescaleDB round-trip (Task 8 operator integration) — ✅ DONE

**Completed 2026-06-15.** Plan 3 Task 8 is fully closed. Round-trip verified against a live TimescaleDB: `saved 2 rows; table count = 2`, idempotent on rerun (the `ON CONFLICT` upsert in `checkpoint.py` holds count steady). **Deviation from the runbook below:** Docker Desktop is uninstalled on this device — docker engine lives in **WSL2 `Ubuntu-22.04`** (so `npm run db:up`/`db:down` fail on Windows; drive compose via `wsl.exe`). Because WSL reaps the distro when the launching process exits, the bring-up + health-gate + round-trip were run in **one continuous WSL session**, with the round-trip executed from **WSL python** (psycopg2-binary + pydantic) against `localhost:5432` — see memory `docker-wsl-roundtrip`. Script: `tools/db_roundtrip_check.py` (uncommitted scratch). Step 5 (env-guarded test) remains **optional/open**.

Closes the one operator-owned step of Plan 3 Task 8 — the last open item from Plan 3.

**Why now / why this first:** smallest scope, no new code on the critical path (the round-trip is operator-run, **outside** the pytest gate), and it fully closes Plan 3. Only blocker last session was docker being down.

**Concrete facts (re-verified 2026-06-15):** `infra/docker-compose.yml` → `timescale/timescaledb:latest-pg14`, creds **aihf/aihf/aihf** on **localhost:5432** → `DATABASE_URL = postgresql://aihf:aihf@localhost:5432/aihf` (compose also starts `redis:7` on 6379 and `adminer` on 8080). Python driver: **psycopg2 2.9.10 is installed** (psycopg3 is NOT). `apps/quant/advisor/persistence/checkpoint.py` exposes `SCHEMA_SQL` (CREATE TABLE `signal_bundle` + `create_hypertable(...)`) and `save_bundle(conn, bundle) -> int`, DBAPI-compatible (`conn.cursor()` / `executemany` / `commit`). Schemas confirmed: `SignalBundle(ticker, as_of, signals=[FamilySignal(family, direction, confidence, as_of)])`, `Direction.BULLISH/BEARISH`.

### ▶ Start here — operator runbook

> **Discipline:** the round-trip below is **operator-run by Claude/you directly** (it touches the live DB and is outside the pytest gate — Codex's `workspace-write` sandbox blocks network/DB anyway). Only the *optional* env-guarded integration test in step 5 is a code edit → dispatch that one via **Hermes solo** per the contract.

- [ ] **0. Docker prerequisite.** `docker` is **not on the Git Bash PATH** this session (`docker: command not found`). Start **Docker Desktop**, then run docker/`npm run db:*` commands from **PowerShell** (or whichever shell has `docker` on PATH). Confirm with `docker version` (server line must print) before continuing.
- [ ] **1. Bring the DB up.** `npm run db:up` (= `docker-compose -f infra/docker-compose.yml up -d`). Wait for health: `docker compose -f infra/docker-compose.yml ps` until the timescale service is `healthy`, or poll `docker exec <timescale-container> pg_isready -U aihf -d aihf`.
- [ ] **2. Run the round-trip.** Save the reference script (below) to a scratch file (e.g. `tools/db_roundtrip_check.py` — do **not** commit it, or keep it under `tools/` if you want it reusable) and run `python tools/db_roundtrip_check.py`. **Windows gotcha (memory `hermes-dispatch-windows`):** do NOT use `python -c "<multi-word string>"` under `shell:true` — the shell tokenizes it and `-c` gets only the first word. **Run a `.py` file.**
- [ ] **3. Confirm persistence.** Expect `saved 2 rows; table count = 2`. Optionally eyeball via Adminer at `http://localhost:8080` (System: PostgreSQL, Server: `localhost`, user/pass/db: `aihf`) → `SELECT * FROM signal_bundle;`.
- [ ] **4. Tear down when done.** `npm run db:down` (add `-v` to also drop the data volume if you want a clean slate next time).
- [ ] **5. (Optional, higher value — via Hermes solo) Formalize as a guarded test.** Add `test_checkpoint_roundtrip` skipped unless `AIHF_DB_URL` is set (`@pytest.mark.skipif(not os.getenv("AIHF_DB_URL"), ...)`), so it runs on demand but stays **excluded** from the default `advisor-gate`. Add `psycopg2-binary` to the Python deps if you formalize it. Verify with `python -m pytest apps/quant/advisor/tests`. This is the only step that edits tracked code → Hermes task string must include *"Do NOT run npm or node; verify ONLY with `python -m pytest apps/quant/advisor/tests`."*
- [ ] **6. Done criteria.** Rows persisted + count verified ⇒ Plan 3 Task 8 is fully closed. Note it in `.remember/remember.md`.

**Reference script for step 2** (`tools/db_roundtrip_check.py`):
  ```python
  import sys; sys.path.insert(0, "apps/quant")
  from datetime import date
  import psycopg2
  from advisor.persistence.checkpoint import SCHEMA_SQL, save_bundle
  from advisor.schemas import Direction, FamilySignal, SignalBundle

  conn = psycopg2.connect("postgresql://aihf:aihf@localhost:5432/aihf")
  with conn.cursor() as cur:
      cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")  # belt-and-suspenders; image preloads it
      cur.execute(SCHEMA_SQL)
  conn.commit()
  b = SignalBundle(ticker="AAPL", as_of=date(2024, 5, 1), signals=[
      FamilySignal(family="trend", direction=Direction.BULLISH, confidence=70.0, as_of=date(2024, 5, 1)),
      FamilySignal(family="macro", direction=Direction.BEARISH, confidence=60.0, as_of=date(2024, 5, 1)),
  ])
  n = save_bundle(conn, b)
  with conn.cursor() as cur:
      cur.execute("SELECT count(*) FROM signal_bundle")
      print("saved", n, "rows; table count =", cur.fetchone()[0])
  ```

---

## Workstream C — Live multi-source CLI assembly

Today the CLI (`apps/quant/advisor/cli.py`) ships **value/quality only**; the floor's 2-family ensemble and the other families are not yet on the live decision path. Wire the full 5-family `run_pipeline` into a live CLI command.

- [ ] Add provider adapters behind the existing data interface: FRED (10y-2y `T10Y2Y` for `analysis/macro.py`) and a news source (for `analysis/sentiment.py`'s scorer; respect spec §10 — missing source ⇒ `unavailable`, never fabricated). Prices already have `YFinanceProvider`.
- [ ] Assemble `family_coros` (value/quality, momentum, trend, macro, sentiment) → `run_pipeline(...)` with optional `--explain` persona pass.
- [ ] TDD with **fakes** for the unit path (Codex-verifiable); verify live assembly manually (operator). This is Plan-4-sized; consider folding it into Workstream D's plan.

---

## Workstream D — v2 calibration (the real next plan; the ONLY thing that closes the floor)

v1 ships equal-weight by design and **fails its own §7 floor** (ensemble Sharpe 0.32 < SPY 0.85 and < trend-alone 0.48 on the committed 2018–2023 fixture). Per spec §8, the fix is **shrinkage-to-equal-weight skill-weighting**:
- Rolling **rank-IC / information-ratio** per family on a purged walk-forward.
- **Cap any single family at 1/N–2/N**; require a **minimum OOS window** before any weight deviates from equal.
- **Gate family inclusion on Brier improvement over base rate**, not raw IC. In-sample weighting is overfitting with extra steps.

- [ ] Write **"Plan 4 — v2 calibration"** with `superpowers:writing-plans`, dispatch via Hermes solo + fixture-first (reuse `apps/quant/advisor/tests/fixtures/floor_prices.csv`; extend with per-family IC over the same purged folds).
- [ ] **Carry the necessary-not-sufficient caveat (load-bearing):** the floor backtests only the **price-only proxy** (momentum+trend) because spec §6 says only price/volume is honestly backtestable. macro/sentiment/value-quality **cannot** be purged of look-ahead on free data. So even a future price-only floor PASS would **not** prove the 5-family advisor satisfies §7. Until calibration clears the floor, the advisor **reports but must not size real capital**.

---

## Recommended order
1. ~~**A (§10 cleanup)**~~ — ✅ **DONE & pushed** @ `ff4fc84` (see Workstream A above).
2. ~~**B (TimescaleDB round-trip)**~~ — ✅ **DONE** 2026-06-15 (see Workstream B above); Plan 3 fully closed.
3. **▶ D (v2 calibration)** — **DO THIS NEXT.** The highest-value work; the only path to a production-ready advisor. C can fold into D's plan.

No hard dependencies between workstreams. Each is independently shippable.
