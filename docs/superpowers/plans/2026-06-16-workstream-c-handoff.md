# Handoff — Build Workstream C: live 5-family CLI assembly (Plan 2 of the deferred roadmap)

> **You are picking up the AIHedgeFund advisor in a fresh session. Plan 1 (the report-only validation/deflation gate) is DONE and pushed. Your job is to build Workstream C — wire the full five-family `run_pipeline` into a live CLI command — via Hermes solo, TDD with fakes, report-only.** Read §1 (task), then §2 (state), §3 (rails), §4 (Hermes mechanics — has hard-won lessons), §5 (verified code contracts), §6 (task skeleton), §7 (first moves), §8 (done). The state/contracts in §2 & §5 are verified against the code on 2026-06-16 — do not re-derive.

---

## 1. Your task

The live CLI (`apps/quant/advisor/cli.py`) today ships **value/quality only** (single-family). The orchestration engine `run_pipeline` (`apps/quant/advisor/pipeline/run.py`) already exists and assembles `family_coros → SignalBundle → ensemble_vote → allocate → Decision`, but nothing wires the **five** families into it on the live path. Build that:

1. **Data adapters** for the two missing inputs: **FRED** (10y–2y `T10Y2Y` series for `analysis/macro.py`) and a **news source** (headlines for `analysis/sentiment.py`'s scorer). Prices + fundamentals already have `YFinanceProvider`.
2. **Async family-coro factories** that wrap each existing `evaluate()` into a `Callable[[date], Awaitable[FamilySignal]]`, fetching inputs from the provider(s) and returning the family signal (or a **neutral** signal when the source is unavailable — never fabricated).
3. **A live CLI command** that assembles all five coros (value/quality, momentum, trend, macro, sentiment) and calls `run_pipeline(...)`, with an optional `--explain` persona pass; output carries the existing `disclosure_header()`.

**Scope rails:** this is the **report-only live decision path**, NOT a floor change. It does NOT close the floor and does NOT authorize sizing real capital (see §3). TDD the unit path with **fakes** (Codex-verifiable); verify the live multi-source assembly **manually** (operator). First **write the plan** (`superpowers:writing-plans`) from the §6 skeleton, then execute task-by-task via Hermes.

---

## 2. State (verified 2026-06-16 — SETTLED, do not re-litigate)

- **Done & pushed to `origin/main` (`nikhillinit/dancingcucumber`):** Workstream A (repo cleanup), Workstream B (TimescaleDB round-trip), Workstream D / Plan 4 (v2 calibration → machine verdict **DEV_FAILED**, floor accepted as a negative, family-reweighting lane CLOSED), and **Plan 1 (validation/deflation gate, report-only)** @ `c0ff501`.
- **The floor verdict is `DEV_FAILED` and the release gate blocks (`node tools/run-floor.mjs --enforce` exit 1).** Workstream C must NOT change that. The advisor is **not authorized for production capital sizing**; it reports.
- **The five family `evaluate()` functions ALL already exist** as pure synchronous functions under `apps/quant/advisor/analysis/` (value_quality, trend, momentum, sentiment, macro — exact signatures in §5). The work is adapters + async wrappers + CLI wiring, NOT writing the family logic.
- **`run_pipeline` already exists** and is the orchestration seam (§5). It already calls `ensemble_vote`, `allocate`, `position_limit`, and the persona overlay. You CALL it; you do not rebuild it.
- **`YFinanceProvider` already supplies prices + fundamentals** (`get_prices`, `get_fundamentals_asof`). FRED and news are the only missing providers.
- The advisor package is self-contained under `apps/quant/advisor`; `python -m pytest apps/quant/advisor/tests` is the gate. Current suite: **131 passed** (after Plan 1).

## 3. Hard rails (violating any fails the task)

1. **Never modify `portfolio/allocator.py` / `ensemble_vote` / `allocate`.** `run_pipeline` already calls them; you only assemble inputs. (Same frozen-seam rail as Plan 1; enforced by `.claude/hooks/guard-frozen-floor.mjs` for `allocator.py`.)
2. **Never weaken the release gate.** `node tools/run-floor.mjs --enforce` must stay **exit 1**; `npm run advisor-gate` stays exit 0. Workstream C touches the live decision/reporting path, never the floor or its verdict.
3. **No fabricated data (spec §10).** A missing/failed source ⇒ `FamilySignal.neutral(family, as_of, "<reason>")` or `"unavailable"`, NEVER a synthesized/optimistic signal. `analysis/sentiment.py` already enforces no-fabrication; the news adapter must return empty/unavailable (→ neutral) when there is no API key.
4. **Load-bearing report-only caveat (carry into CLI output + the plan):** the floor backtests only the **price-only proxy** (momentum+trend) because spec §6 says only price/volume is honestly backtestable. macro/sentiment/value-quality **cannot** be purged of look-ahead on free data. So a working 5-family CLI produces **report-only** signals; the advisor must **not size real capital** while the floor blocks. Keep `disclosure_header()` on every CLI output (`run()` already appends it).
5. **Fixture-first for anything network/secret-bound.** FRED + news keys are operator secrets and Codex's `workspace-write` sandbox blocks network. Codex writes logic against **fakes/fixtures**; the operator (you/Claude) verifies the live path manually. Do NOT commit secrets.
6. **Don't add fields to `PreRegConfig`** (immutable, SHA-hashed) — irrelevant to C but the standing repo rail.

## 4. Hermes execution mechanics (READ memory `hermes-dispatch-windows` FIRST — updated this session)

- **Solo dispatch only:** plain `npm run hermes:production -- --task "<task text>"` (Codex owns the edit). No `--workflow` (pair needs Claude credits, fragile on Windows).
- **THE BIG LESSON FROM PLAN 1 (do this):** a large `--task` literal hits the **Windows ~8191-char command-line limit** (`The command line is too long`) — slashes are NOT the blocker, length is. **Fix = the SHORT FILE-POINTER DISPATCH:** write the full instruction (including the COMPLETE target file contents) to `ai-logs/hermes/task-N.md`, then dispatch a tiny task:
  ```powershell
  $env:PYTHONUTF8=1
  npm run hermes:production -- --task "In this repository, open the file ai-logs/hermes/task-N.md and follow its instructions EXACTLY. Do NOT run npm or node; verify ONLY with: python -m pytest <task's test path>. Do NOT commit; leave changes in the working tree for review."
  ```
  Codex reads the file from its sandbox. This worked for all 9 Plan-1 tasks.
- **Give Codex COMPLETE cumulative file content, not "append this block."** When several tasks touch the same file, Codex sometimes regenerates it from the latest snippet and silently drops earlier code. Put "the file must end up EXACTLY as: <full content>" in each task file. Backstop: after each dispatch run `git diff --stat` on the touched files (watch for unexpected DELETIONS) and run the FULL test file confirming the test count ROSE.
- **Always include** *"Do NOT run npm or node; verify ONLY with pytest."* (Codex's sandbox blocks npm/node; it sometimes runs them anyway — verify the real git state regardless.) Set `PYTHONUTF8=1`.
- **Per-task loop:** dispatch → `git status`/`git diff --stat` to verify Codex's REAL git state (it cherry-picks/skips) → run the task's pytest yourself → commit yourself if Codex didn't (one task = one commit, with the message from the plan) → next task.
- **Plan-bug rule:** if a seeded assertion fails against the given impl, fix the fixture/threshold/code in place yourself (the handoff/runbook sanctions it) — do NOT re-dispatch the same task in a loop.
- **Human-step carve-out:** anything Codex cannot do (operator-run live verification with real API keys, committing a real fixture) you do directly — that is not a workflow-contract violation when the runbook carves it out.

## 5. Verified code contracts (exact, as of 2026-06-16 — build against these)

**Schemas** (`advisor/schemas.py`):
- `Direction(str, Enum)`: `BULLISH="bullish"`, `BEARISH="bearish"`, `NEUTRAL="neutral"`.
- `FamilySignal(BaseModel, frozen)`: `family:str, direction:Direction, confidence:float[0..100], skill_weight:float=1.0, as_of:date, reasoning:str=""`. Classmethod `FamilySignal.neutral(family, as_of, reasoning="insufficient data")`.
- `SignalBundle(BaseModel, frozen)`: `ticker:str, as_of:date, signals:list[FamilySignal]`.

**Orchestration** (`advisor/pipeline/run.py`):
- `FamilyCoro = Callable[[date], Awaitable]` (returns a `FamilySignal`).
- `async run_pipeline(ticker, as_of, price, net_liq, vol, correlation, family_coros: list[FamilyCoro], persona_critic=None) -> Decision` — `gather`s the coros, builds `SignalBundle`, calls `ensemble_vote`, `position_limit(net_liq, vol, correlation)`, `allocate(bundle, price, position_limit_dollars)`, returns `Decision(ticker, action, quantity, bundle_direction, reasoning)`; applies persona overlay if `persona_critic` given.

**Family evaluators** (`advisor/analysis/`, all return `FamilySignal`):
- `value_quality.evaluate(f: Fundamentals, as_of) ` — `Fundamentals` from `data/provider.py`.
- `trend.evaluate(prices: pd.Series, as_of)` and `momentum.evaluate(prices: pd.Series, as_of)` — price series.
- `macro.evaluate(yield_curve_spread: pd.Series, as_of)` — **needs FRED `T10Y2Y` series**.
- `sentiment.evaluate(headlines: list[str], as_of, scorer: NewsScorer)` — **needs a news adapter + `NewsScorer`** (see `analysis/sentiment.py` + `tests/test_sentiment.py` for the `NewsScorer` protocol).

**Providers** (`advisor/data/provider.py`):
- `MarketDataProvider(Protocol)`: `get_prices(ticker, start, end) -> pd.DataFrame`, `get_fundamentals_asof(ticker, as_of) -> Fundamentals | None`. `YFinanceProvider` implements both.
- `Fundamentals` dataclass + `is_available_asof` / `select_latest_available` enforce the ~90-day point-in-time lag.
- **New adapters to add** (behind small Protocols, fixture/fake-testable): a FRED series provider (`get_series("T10Y2Y", start, end) -> pd.Series`) and a news provider (`get_headlines(ticker, as_of) -> list[str]`, returns `[]` when unavailable).

## 6. Task skeleton (formalize with `superpowers:writing-plans` into `2026-06-16-workstream-c.md`, then dispatch)

Each task = tests-first, one Hermes dispatch, one commit. Suggested order (sequential where they share files):

1. **FRED macro adapter** — `data/fred_provider.py` with a `FredProvider` Protocol + a real adapter (env key `FRED_API_KEY`; returns `T10Y2Y` as `pd.Series`). TDD with a **FakeFredProvider** returning a fixture series; assert `macro.evaluate(fake.get_series(...), as_of)` yields the expected direction. Unavailable key ⇒ adapter returns empty ⇒ coro yields `neutral`.
2. **News adapter + scorer wiring** — `data/news_provider.py` with `NewsProvider` Protocol (`get_headlines -> list[str]`, `[]` when no key) + the `NewsScorer` already used by `sentiment.evaluate`. TDD with a **FakeNewsProvider** + fake scorer; assert no-fabrication on empty headlines (→ neutral). Respect spec §10.
3. **Family-coro factories** — `pipeline/families.py` (new): `make_value_quality_coro(provider, ticker)`, `make_trend_coro`, `make_momentum_coro`, `make_macro_coro(fred)`, `make_sentiment_coro(news, scorer, ticker)`, each returning an `async (as_of) -> FamilySignal` that fetches inputs and calls the matching `evaluate`, falling back to `FamilySignal.neutral(...)` on any missing input/exception. TDD each with fakes (no network).
4. **CLI command** — extend `cli.py` with a `recommend`/`--families all` path (keep the existing single-family default for back-compat) that builds the five coros and `asyncio.run(run_pipeline(...))`. Decide how `price`/`net_liq`/`vol`/`correlation` are sourced (latest price from provider; net_liq/vol/correlation as CLI args with documented defaults). Keep `--explain`. TDD the assembly with a **FakeProvider bundle** (no network); assert a `Decision` is produced and `disclosure_header()` is in the output.
5. **(Operator, manual) Live multi-source smoke** — with real `FRED_API_KEY` + news key, run the CLI against a couple of tickers; confirm sane output + disclosures. NOT in the pytest gate (network). Document results; do not commit secrets.
6. **Docs + roadmap update** — mark Workstream C done in the deferred-plans roadmap; note that Plan 1b (wire `validation["passes"]` into `--enforce`) and Plan 3 (post-C signal program) are now unblocked-by-prerequisite but still gated on a real candidate.

## 7. First moves

1. Read this handoff; memories `hermes-dispatch-windows` (updated), `validation-gate-floor-internals`, `plan4-v2-calibration`, `deep-research-orthogonal-signals`, `advisor-v1-fails-floor`; the spec `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md` (esp. §6 backtestability, §10 no-fabrication); and the prior Workstream-C sketch in `docs/superpowers/plans/2026-06-15-followups-handoff.md` (§ "Workstream C").
2. **Confirm green baseline:** `$env:PYTHONUTF8=1; python -m pytest apps/quant/advisor/tests -q` (pass); `npm run advisor-gate` (exit 0); `node tools/run-floor.mjs --enforce` (**exit 1**).
3. **Resolve the one open operator decision (ASK the user):** which **news source** to use and whether `FRED_API_KEY` / the news key are available now. If keys are not available, build the full fake/unit path anyway (adapters fall back to neutral) and defer the live smoke (Task 5) until keys arrive — the unit path is the gated deliverable.
4. Read `analysis/{macro,sentiment}.py` + their tests to lock the `NewsScorer`/series contracts, then write `2026-06-16-workstream-c.md` with `superpowers:writing-plans` from the §6 skeleton (consider `/plan-eng-review`).
5. Execute task-by-task via Hermes solo using the §4 file-pointer dispatch; verify real git state, run pytest, commit one-per-task; push when done.

## 8. Definition of done

- `2026-06-16-workstream-c.md` written; all its tasks committed (one commit each) and pushed.
- New FRED + news adapters and the five family-coro factories exist, each unit-tested with **fakes** (no network in the gate); missing-source ⇒ neutral (no fabrication) is asserted.
- A live CLI command assembles all five families through `run_pipeline` and prints a `Decision` + `disclosure_header()`; `--explain` works.
- `python -m pytest apps/quant/advisor/tests` green (test count risen); `npm run advisor-gate` **exit 0**; `node tools/run-floor.mjs --enforce` **exit 1** (UNCHANGED — C never touches the floor).
- `allocator.py`/`ensemble_vote` untouched; no secrets committed; report-only caveat carried in output and docs.
- Roadmap updated (C done; 1b/3 status noted).

## 9. Open decisions for the operator (resolve before the live step)

- **News source + key** (Task 2/5): which provider, and is a key available? (Without it, ship the fake/unit path; defer live smoke.)
- **`FRED_API_KEY`** availability (Task 1/5).
- **CLI risk inputs** (Task 4): default values / sourcing for `net_liq`, `vol`, `correlation` feeding `position_limit`/`allocate` (propose: CLI args with conservative documented defaults; latest price from the provider).
