# Handoff — Post Workstream C: live smoke, optional news redundancy, and the candidate question

> **You are picking up the AIHedgeFund advisor in a fresh session. Workstream C (Plan 2) is DONE and PUSHED to `origin/main` — the five-family `run_pipeline` is wired into the CLI (`--families all`), report-only, floor untouched.** Read §1 (state, verified — do not re-derive), §2 (the three lanes + which to pick), §3 (hard rails), §4 (Hermes mechanics), §5 (exact code/contracts), §6 (open operator decisions — ASK), §7 (first moves), §8 (definition of done per lane). State in §1/§5 is verified against the code on 2026-06-16; do not re-derive.

---

## 1. State (verified 2026-06-16 — SETTLED, do not re-litigate)

- **Workstream C / Plan 2 = DONE & PUSHED.** 8 commits `8662bba`→`cb0bd3e` on `main` (range `a215f9c..cb0bd3e`), pushed to `origin/main` (`nikhillinit/dancingcucumber`). Tests **131 → 160**.
- The five families (value_quality, trend, momentum, macro, sentiment) assemble through the existing `run_pipeline` on the live path via the new `advisor --families all <TICKER>` command. **Report-only**: the floor verdict is still `DEV_FAILED`, `node tools/run-floor.mjs --enforce` is **exit 1**, and the advisor is **NOT authorized to size real capital**.
- Verified at completion: `npm run advisor-gate` **exit 0**; `node tools/run-floor.mjs --enforce` **exit 1** (UNCHANGED — C never touched the floor); frozen paths (`portfolio/`, `risk/`, `backtest/`) zero-diff since `a215f9c`; no secrets committed. A **keyless entrypoint smoke** passed: real `yf.download` returns MultiIndex columns `('Close','AAPL')`, `close_series` squeezes to the correct 1-D Series, and `main([...,"--families","all"])` printed a real Decision (`AAPL [bearish] sell qty=86`) with macro/sentiment correctly degrading to neutral (no keys).
- **Two corrections to the original C handoff (so you don't trust the wrong lines):**
  1. The original §5 claimed a production `NewsScorer` existed. **It did not** — only the `Callable[[str],float]` alias + test lambdas. This session built `analysis/news_scorer.py::lexicon_score` (deterministic, NOT an LLM).
  2. As-of bounding was not in the original skeleton; it is now enforced at BOTH adapter (`observation_end`/`time_to`=as_of) and coro level, asserted via recording fakes.
- **What is NOT yet done (this handoff's scope):** the **keyed** live smoke (real FRED + Alpha Vantage keys), optional Finnhub/NewsAPI redundancy in the composite, and rotating the hardcoded keys. Plans 1b/3 remain **blocked on a real candidate** (none exists; floor is DEV_FAILED).
- Plan: `docs/superpowers/plans/2026-06-16-workstream-c.md`. Roadmap: `docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md`. Memories to read first: `workstream-c-done`, `hermes-dispatch-windows`, `validation-gate-floor-internals`, `plan4-v2-calibration`, `deep-research-orthogonal-signals`, `advisor-v1-fails-floor`.

## 2. The three lanes — pick one with the operator (§6)

**Lane A — Finish C's deferred tail (small, mostly operator-run). RECOMMENDED next.**
- A1. **Keyed live smoke** (operator step; needs real keys). Confirms macro/sentiment produce non-neutral signals on the live path end-to-end. Not in the pytest gate (network/secrets).
- A2. **(Optional, YAGNI-gated) news redundancy.** Only if the coverage probe shows Alpha Vantage thin: add `FinnhubNewsProvider` and/or `NewsApiProvider` behind the existing `NewsProvider` Protocol and drop them into `CompositeNewsProvider([...])` (≈1 class + 1 test each).
- A3. **Hygiene.** Rotate the FRED + Alpha Vantage keys that are hardcoded in the untracked root scripts (`fred_economic_analysis.py:16`, `alpha_vantage_enhanced_analysis.py:16`); the production adapters read env only.

**Lane B — Candidate search (the real unblock for 1b/3; open-ended research → implement loop).**
- The floor is DEV_FAILED because no candidate beats its own price-only proxy across purged folds. Per memory `deep-research-orthogonal-signals`: the cheapest no-new-data step is **timely-price value+momentum**; SESTM news NLP is genuine orthogonal alpha but **research-only/conditional** for the large-cap book; **skill_weight reweighting is WARNED against** on a correlated pool; and the **Seams 3/4 validation tooling is a coverage gap** needing follow-up. This is a research→implement→measure cycle, not a quick build — scope it with `superpowers:brainstorming` + `deep-research`, then `writing-plans`.

**Lane C — Plan 1b (BLOCKED — do NOT start now).** Wire `validation["passes"]` into `--enforce` so a deflation-failing candidate cannot release. Starting this with no candidate would disturb the accepted DEV_FAILED floor for zero benefit. Defer until a candidate clears dev.

## 3. Hard rails (unchanged from C — violating any fails the task)

1. **Never modify** `portfolio/allocator.py`, `ensemble_vote`, `allocate`, `risk/limits.py`, or anything under `backtest/`. `allocator.py` is guarded by `.claude/hooks/guard-frozen-floor.mjs`.
2. **Never weaken the release gate.** `advisor-gate` stays exit 0; `run-floor --enforce` stays exit 1. Until a candidate clears dev AND Plan 1b lands, the advisor reports; it does not size capital.
3. **No fabricated data (spec §10).** Missing/failed source ⇒ `FamilySignal.neutral(...)`, never a synthesized signal. Any new news adapter returns `[]` when keyless/throttled/errored.
4. **As-of bounding is mandatory.** Every adapter query is capped at `≤ as_of`; assert it with recording fakes.
5. **No secrets in code or commits.** Keys via env only (`FRED_API_KEY`, `ALPHAVANTAGE_API_KEY`, and e.g. `FINNHUB_API_KEY`/`NEWSAPI_API_KEY` if added). Never `git add` the hardcoded-key root scripts.
6. **Don't add fields to `PreRegConfig`** (immutable, SHA-hashed) — standing repo rail.

## 4. Hermes execution mechanics (memory `hermes-dispatch-windows` — proven this session for all 8 C tasks)

- **Solo dispatch only:** `npm run hermes:production -- --task "<short text>"`. No `--workflow` (needs Claude credits, fragile on Windows).
- **Short file-pointer dispatch** (sidesteps the Windows ~8191-char cmdline limit): write the full instruction — **including the COMPLETE final content of every file** the task creates/replaces — to `ai-logs/hermes/<task>.md`, then dispatch:
  ```powershell
  $env:PYTHONUTF8=1
  npm run hermes:production -- --task "In this repository, open the file ai-logs/hermes/<task>.md and follow its instructions EXACTLY. Do NOT run npm or node; verify ONLY with: python -m pytest <test path> -q. Do NOT commit; leave changes in the working tree for review."
  ```
- **Per-task loop:** dispatch → `git status`/`git diff --stat` to verify Codex's REAL git state (it sometimes runs `npm run check` anyway and cherry-picks; verify regardless) → run the task's pytest yourself → commit one-per-task (end the message with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer) → next.
- **Plan-bug rule:** if a seeded assertion/command fails against the impl, fix it in place yourself — do NOT re-dispatch in a loop. (This session: the probe needed a `sys.path` bootstrap to run standalone; fixed in place.)
- **Human-step carve-out:** operator-run live verification with real keys, and planning-doc edits, are done directly by Claude — not a workflow-contract violation.

## 5. Exact code & contracts (as of 2026-06-16 — build against these)

New files (all under `apps/quant/advisor/` unless noted):
- `data/fred_provider.py` — `FredProvider(Protocol).get_series(series_id, start, end) -> pd.Series`; `FredApiProvider(api_key=None, http_get=None)` (env `FRED_API_KEY`; keyless/error → empty Series; `observation_end=end`); const `FRED_SERIES_T10Y2Y="T10Y2Y"`.
- `data/news_provider.py` — `NewsProvider(Protocol).get_headlines(ticker, as_of) -> list[str]`; `CompositeNewsProvider(providers)` (union+dedupe by normalized lower/space, all-empty→[], swallows a raising source); `AlphaVantageNewsProvider(api_key=None, http_get=None)` (env `ALPHAVANTAGE_API_KEY`; `time_to=as_of.strftime("%Y%m%dT2359")`; missing key / `{"Note"/"Information":...}` / error → []). **A new adapter copies this shape.**
- `analysis/news_scorer.py` — `lexicon_score(headline: str) -> float` in [-1,1], `POSITIVE`/`NEGATIVE` word sets. Used as the `scorer` arg to `sentiment.evaluate`.
- `pipeline/families.py` — `FamilyCoro = Callable[[date], Awaitable[FamilySignal]]`; factories `make_value_quality_coro(provider, ticker)`, `make_trend_coro(provider, ticker)`, `make_momentum_coro(provider, ticker)`, `make_macro_coro(fred, series_id=FRED_SERIES_T10Y2Y)`, `make_sentiment_coro(news, scorer, ticker)`; helper `close_series(df) -> pd.Series`; consts `PRICE_LOOKBACK_DAYS=420`, `MACRO_LOOKBACK_DAYS=30`. Each coro: fetch (≤ as_of) → matching `evaluate` → `FamilySignal.neutral(...)` on missing/exception.
- `cli.py` — `run(provider, ticker, as_of, critic=None)` (single-family, unchanged); `run_all(provider, fred, news, scorer, ticker, as_of, net_liq, vol, correlation, persona_critic=None) -> str`; `_latest_price(provider, ticker, as_of)`; `main(argv)` with flags `--families {value,all}`, `--net-liq`, `--vol`, `--correlation`, `--explain`; defaults `DEFAULT_NET_LIQ=100_000`, `DEFAULT_VOL=0.30`, `DEFAULT_CORRELATION=0.50`.
- `tests/fakes.py` — `FakeMarketDataProvider`, `FakeFredProvider`, `FakeNewsProvider` (all record `.calls`), `rising_prices`, `steep_curve`, `inverted_curve`.
- `scripts/news_coverage_probe.py` (repo root `scripts/`) — operator probe, path-bootstrapped, non-gate.

Pre-existing seams you CALL (do not modify): `pipeline/run.py::run_pipeline(ticker, as_of, price, net_liq, vol, correlation, family_coros, persona_critic=None) -> Decision`; `portfolio/allocator.py::{ensemble_vote, allocate}`; `risk/limits.py::position_limit`; `personas/overlay.py::{PersonaVerdict, apply_overlay}`; `backtest/walk_forward.py::disclosure_header`. Schemas: `schemas.py::{Direction, FamilySignal (frozen; .neutral(family, as_of, reasoning) → conf 50.0), SignalBundle}`.

**Gate setup:** `pytest.ini` sets `pythonpath = apps/quant`, `testpaths = apps/quant/advisor/tests`. Gate command: `python -m pytest apps/quant/advisor/tests -q` (currently 160 passed). `cli.py` has **no `__main__` guard / console entrypoint**; the verified live-run method is below.

## 6. Open operator decisions (ASK before the live/build step)

1. **Keys for the live smoke (Lane A1):** are real `FRED_API_KEY` + `ALPHAVANTAGE_API_KEY` available (rotated, not the hardcoded ones)? If not, Lane A1/A3 wait; consider Lane B.
2. **News redundancy (Lane A2):** add Finnhub and/or NewsAPI to the composite, or stay AV-only? (Decide AFTER the coverage probe shows whether AV coverage is thin.)
3. **Which lane this session:** A (finish the tail), B (start candidate search), or hold.
4. **Entrypoint ergonomics:** is a clean `python -m advisor` entrypoint wanted (3-line `apps/quant/advisor/__main__.py`), or is the import-and-call method (below) fine for operator smokes?

## 7. First moves

1. Read this handoff + the §1 memories + the C plan. Confirm green baseline: `$env:PYTHONUTF8=1; python -m pytest apps/quant/advisor/tests -q` (160 passed); `npm run advisor-gate` (exit 0); `node tools/run-floor.mjs --enforce` (**exit 1**). Confirm `git log --oneline -1` is `cb0bd3e` and `git status` is clean.
2. Ask the operator the §6 questions.
3. **If Lane A1 (keyed live smoke):** with rotated keys in env, run the coverage probe and the CLI, e.g.:
   ```powershell
   $env:PYTHONUTF8=1; $env:FRED_API_KEY="..."; $env:ALPHAVANTAGE_API_KEY="..."
   python scripts/news_coverage_probe.py AAPL MSFT NVDA --as-of 2024-05-01
   $env:PYTHONPATH="apps/quant"; python -c "from advisor.cli import main; main(['AAPL','--families','all','--as-of','2024-05-01'])"
   ```
   Confirm: probe shows non-zero headline counts; CLI prints a Decision + `disclosure_header()`; macro/sentiment now non-neutral. Document results (no secrets). Commit only docs.
4. **If Lane A2 (new news adapter):** copy the `AlphaVantageNewsProvider` shape (env key, injected `http_get`, `time_to`/date bound, throttle/error→[]), TDD with `FakeNewsProvider`-style tests, add to the composite in `cli.py`'s `--families all` branch. Dispatch via §4 Hermes file-pointer.
5. **If Lane B (candidate search):** `superpowers:brainstorming` to frame the hypothesis, optionally `deep-research`, then `superpowers:writing-plans`. Start from the timely-price value+momentum step and the Seams 3/4 validation-tooling gap (memory `deep-research-orthogonal-signals`). Do NOT pre-commit forward thresholds.

## 8. Definition of done (per lane)

- **A1:** keyed live smoke run on ≥2 tickers; documented (sane Decision + disclosures + non-neutral macro/sentiment); no secrets committed; gates unchanged.
- **A2:** new adapter(s) unit-tested with fakes (missing/throttle→[] asserted, as-of bound asserted), wired into the composite, full suite green (count risen), gates unchanged.
- **A3:** hardcoded keys rotated; root scripts read env or are removed from disk; nothing secret committed.
- **B:** a written plan (`docs/superpowers/plans/<date>-<name>.md`) for the candidate program, scoped against real constraints, with the floor/validation gate as the acceptance test. (Implementation is a later session.)
- All lanes: `allocator.py`/`ensemble_vote`/`risk`/`backtest` untouched; report-only caveat preserved; floor `--enforce` stays exit 1 until a candidate + Plan 1b deliberately change it.
