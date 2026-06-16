# Handoff — Plan the next phase of the AIHedgeFund advisor

> **You are picking up the AIHedgeFund quant advisor in a fresh session. Your job is to PRODUCE AN IMPLEMENTATION PLAN for the next phase of development — not to write code yet.** Everything you need is below. The decisions marked "SETTLED" are not to be re-litigated. Start by reading §1 (your task), then §2–§8 for context, then follow §9 (first moves).

---

## 1. Your task (the deliverable)

Produce an implementation plan for **Phase A: build the validation/deflation gate into the existing backtest harness.** This is recommended as the next phase because it is the **highest-confidence, lowest-risk, signal-agnostic** work AND it is the prerequisite gate that every future signal must clear. Optionally scope **Phase B: the timely-price value+momentum signal**, validated *through* the Phase A gate.

**Before finalizing scope, confirm ONE decision with the user** (see §7-Q1): start with the gate (Phase A, recommended) or jump straight to a new signal (Phase B). Default to Phase A unless the user redirects.

Use the `superpowers:brainstorming` skill to confirm scope, then `superpowers:writing-plans` to produce the plan. The plan must ship **tests + impl together** (this project's convention) and be dispatchable via Hermes (§6).

---

## 2. Where we are (TL;DR — SETTLED)

- The advisor is a **price-only ensemble**: signal families (momentum, trend, mean-reversion, breakout, value/quality, macro, sentiment) combined by an **equal-weight `ensemble_vote`** into allocations.
- Its pre-registered walk-forward **FLOOR returned `DEV_FAILED`** (formal 30-name large-cap universe, 2015–2023): the ensemble cannot beat its best standalone price family or SPY. "Beat the parts" is **structural** for a fixed blend of correlated long-only price families.
- **DECISION (2026-06-16, Option 1 accepted, do NOT reopen):** accept the negative. The **family-reweighting lane is CLOSED/exhausted.** The release floor stays blocked (`node tools/run-floor.mjs --enforce` → exit 1); the advisor is **NOT authorized for production capital sizing** until a defensible floor clears. This project's ethos is the **honest negative** — it blocks its own release rather than ship a weak result. Preserve that.
- A three-run **deep-research effort (2026-06-16)** priced out the path forward. Net conclusion below.

## 3. What the deep research concluded (SETTLED — evidence in `ai-logs/deep-research-2026-06-16/`)

**The sober read:** the proven lane is closed; the best-evidenced *new* lane (SESTM news sentiment) is **gross-only** for a large-cap price-only book; and the validation rigor to be added is a **gate that raises the bar, not an advance that clears one**. For a 30-name large-cap price-only book the achievable *net* edge looks thin. Reordered menu:

1. **Validation gate = do-now #1 (signal-agnostic).** Confirmed bolt-ons (all 3-0 adversarially verified):
   - **N-tracking is the precondition** — dev/holdout alone is provably inadequate (ignores #trials); walk-forward OOS is NOT a multiple-testing defense (~20 WF iterations find a false 5%-significant strategy).
   - **Deflated Sharpe Ratio**: `DSR = PSR(SR₀)`, `SR₀ = √V[SR̂ₙ]·((1−γ)Φ⁻¹[1−1/N] + γΦ⁻¹[1−1/(Ne)])`, γ≈0.5772. Inputs (N, V[SR̂ₙ], T, skew, kurt) are **all already produced by the dev sweep**. Worked example: 2.5 Sharpe, N=100, skew −3, kurt 10, T=1250 → **DSR≈0.90 → FAILS** the 0.95 bar.
   - **MinBTL**: ≤~45 independent configs on 5yr data before the gate is meaningless (a pre-registered throughput budget).
   - **Harvey-Liu-Zhu t > 3.0** hurdle for signal/factor selection.
   - **Purging + embargo** (h≈0.01T) — highest-confidence, IID-independent leakage guard; adopt even without full CPCV.
   - **PBO via CSCV**: use to AUDIT the selection process, **never as the selection objective**; its IID-block assumption is fragile for overlapping labels. CPCV>walk-forward is medium-confidence (synthetic-only).
   - **IMPORTANT FRAMING:** applied to the *current* candidates this can only confirm `DEV_FAILED` harder. Its value is **forward** — the gate the next signal must survive.
2. **Cheapest signal lane = timely-price value+momentum** (no new data — reuses existing families). Value and momentum are negatively correlated (diversify, not recombine): 60/40 lifts long-short Sharpe 0.46→0.79. **But not a free lunch:** value (HML) is spanned/redundant in FF5 with a post-2017 drawdown; it must clear the same deflated bar. Cheapest *live* lane, not evidenced-to-work.
3. **SESTM news NLP = research-only/conditional (DEMOTED from #1).** Gross EW Sharpe 4.29 is microcap-concentrated, ~95% daily turnover; value-weighted (large-cap) only 1.33; net survives only by engineering turnover down (peak 2.30); post-pub decay worst in large-caps. **Gross-only for this book's universe.** If ever pursued: pre-registered min-viable test (weekly, value-weighted, large-cap, free text [EDGAR 8-K / Alpha Vantage] + Loughran-McDonald dictionary first), measured NET, must clear a net floor before any paid feed / supervised pipeline.
4. **Do NOT adopt LLM trading-agent architectures** (TradingAgents/FinMem/FinCON): returns collapse ~50%, Sharpe decays 51-62% past the model's training cutoff = leakage/memorization ("Profit Mirage," arXiv:2510.07920). Adopt their post-cutoff *evaluation methodology*, not the agents.

## 4. Codebase state & the seams you'll touch

- **Phase A target (floor/validation):** `apps/quant/advisor/backtest/` — esp. `stats.py` (add DSR/PSR, skew/kurtosis), `splits.py` (purging/embargo, optional CPCV), `dev_gate.py` (wire the deflated threshold + N counter + MinBTL budget), `prereg.py`, `pipeline.py`, `walk_forward.py`.
- **The N counter** is the load-bearing new primitive — instrument the dev sweep to honestly count every candidate config evaluated; it feeds DSR + MinBTL.
- **Phase B target (signal), if scoped:** `apps/quant/advisor/analysis/` (new/adjusted family — make `value_quality.py`/value use *timely* prices), feeding `portfolio/allocator.py`.
- **SETTLED RAIL:** floor/validation work lives in `backtest/`. **NEVER** modify `portfolio/allocator.py` `ensemble_vote` for floor work — that is the deferred LIVE seam. `ensemble_vote` is equal-weight and **ignores the existing `FamilySignal.skill_weight` field** (`apps/quant/advisor/schemas.py:21`, default 1.0). Only wire `skill_weight` sizing AFTER a genuinely orthogonal signal exists (the forecast-combination puzzle says reweighting a correlated price-only pool adds estimation noise).
- **Gate commands:** `npm run advisor-gate` (report mode, exit 0, ~35–45s — runs full dev sweep + bootstrap each call; candidate caching is a worthwhile optimization if it bites). `node tools/run-floor.mjs --enforce` (release gate; currently exit 1 — must stay blocked until a defensible floor clears).
- **Frozen-floor guard** (`.claude/hooks/guard-frozen-floor.mjs`) blocks Edit/Write *overwrite* of `floor_prices.csv` / `PREREG.md` / `UNIVERSE_RULE.md` / `allocator.py` (creation allowed). Bash data-pulls are NOT intercepted.
- **Immutable artifacts:** `apps/quant/advisor/backtest/PREREG.md` (immutable), `FLOOR_RESULT.md` (the recorded decision); fixture `apps/quant/advisor/tests/fixtures/{floor_prices.csv, UNIVERSE_RULE.md}` (SHA-256 `d40b9959…`).

## 5. Open methodology questions to resolve during planning

- **Effective-N (Q-critical):** correlated candidate configs mean raw count over-deflates, ignoring correlation under-deflates. Choose: raw count / PCA dimension / ONC hierarchical clustering. The DSR & MinBTL thresholds are sensitive to this; it is the unsolved practical step.
- **Order of operations:** apply DSR deflation at the dev gate (to decide what advances) or only at the final holdout? How does pre-registered candidate order interact with the multiple-testing penalty owed?
- **CSCV/CPCV vs the existing bootstrap:** does the IID-block assumption degrade gracefully for this fixture's label-overlap structure, or would the stationary/block bootstrap already present give materially different PBO?

## 6. Load-bearing execution mechanics (SETTLED — see memory `hermes-dispatch-windows`)

- **Hermes dispatch, solo only:** `npm run hermes:production -- --task "..."` (Codex owner). Codex sandbox blocks npm/node → always include *"Do NOT run npm or node; verify ONLY with pytest."* `PYTHONUTF8=1` dodges the kimi cp1252 crash. A `--workflow` (debate/pair) needs `--live` to execute. **Verify Codex's REAL git state every time** (it cherry-picks/skips). For slashy task strings, use a task file (PowerShell guard blocks them).
- **Per-task loop that works:** dispatch → `git show --stat` (only the task's files) → `npm run advisor-gate` exit 0 → commit if Codex didn't → next. Sequential only.
- **Plan ships test + impl:** if a seeded assertion fails against the provided impl, that's a **plan-bug** — fix the fixture/threshold yourself, don't re-dispatch.
- Windows / PowerShell 7 environment; solo developer; direct push to `main` acceptable for small fixes, else branch.

## 7. Decisions the planner must put to the user

- **Q1 (scope):** Phase A (validation gate, recommended) first, or jump to Phase B (value+momentum signal)?
- **Q2 (universe fork):** SESTM's residual edge lives in microcaps; capacity is NOT binding at $1M–$500M (the book is small enough to reach them) — but that's a **universe change away from the current large-cap mandate, which has not been signaled**. In scope or out?
- **Q3 (gate strictness):** what DSR/PBO thresholds and effective-N method become the new pre-registered release criteria?

## 8. Artifacts & memory (read these first)

- **`ai-logs/deep-research-2026-06-16/COMBINED-SYNTHESIS.md`** — the merged, advisor-reviewed conclusion. Start here.
- `ai-logs/deep-research-2026-06-16/{sestm-net-of-cost,validation-rigor-seam3-4,orthogonal-signals-survey}.json` — full verified findings + caveats + refuted claims + sources. (README explains the genuine-refute vs API-outage-abstention distinction.)
- **Project memory:** `deep-research-orthogonal-signals.md`, `plan4-v2-calibration.md`, `hermes-dispatch-windows.md`, `hermes-bulk-delete-deviation.md`, `advisor-v1-fails-floor.md`.
- `apps/quant/advisor/backtest/{PREREG.md, FLOOR_RESULT.md}` and `docs/superpowers/plans/2026-06-15-plan4-debate-findings.md`.

## 9. First moves for the fresh session

1. Read `COMBINED-SYNTHESIS.md`, this handoff, and the memory files above.
2. `npm run advisor-gate` to confirm green baseline (exit 0); `node tools/run-floor.mjs --enforce` to confirm the release gate is still blocked (exit 1).
3. Invoke `superpowers:brainstorming` → resolve Q1–Q3 (§7) with the user.
4. Invoke `superpowers:writing-plans` → produce the Phase A implementation plan (DSR + N-counter + MinBTL + purging/embargo into `backtest/`), tests + impl, dispatchable via Hermes, with the new deflated release criteria pre-registered.
5. Do NOT touch `portfolio/allocator.py`; do NOT reopen family reweighting; keep the release gate blocked until the new floor is defined and clears.
