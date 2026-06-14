# AI Investment Advisor — Architecture & Adoption Design

**Date:** 2026-06-14
**Status:** Draft for review
**Topic:** To what extent to adopt `virattt/ai-hedge-fund`, and the target architecture for the most effective AI investment advisor.

---

## 1. Decision summary

We adopt the **engineering pattern** of `virattt/ai-hedge-fund` and **reject its headline premise**.

- **Adopt:** the "deterministic quant → structured facts → constrained LLM → typed signal" pattern; the deterministic risk/position-sizing trust boundary; the multi-model valuation ensemble (DCF, owner-earnings, EV/EBITDA, residual income); typed outputs with safe fallbacks.
- **Reject:** AI investor *personas as the deciders*. Same-model personas reading the same data are one model in costumes — stylistically diverse, informationally correlated. They do not size capital.
- **Keep (ours):** the production execution rails virattt lacks — TimescaleDB, dual kill-switch, vectorbt cost-aware backtesting, broker-ready execution, and free data adapters (yfinance, FRED, NewsAPI).
- **Reuse surgically (MIT):** copy virattt's pure valuation math verbatim, attributed; do **not** inherit its data-access layer or backtest harness (that is where its naive as-reported assumptions live).

This decision was pressure-tested by a 3-round design council and a first-principles challenge. The supporting reasoning is in §11.

## 2. Goals / non-goals

**Goals**
1. A defensible, calibrated, deterministic decision core whose edge can be *bounded* by an honest backtest.
2. A trust boundary where LLMs never touch position size or hard risk limits.
3. An honest point-in-time backtest harness with mandatory bias disclosure — the single highest-leverage deliverable.
4. A persona/deliberation layer that adds real value where it is strong (idea generation, risk critique, explanation), phased toward genuine information-diverse deciders.
5. Repo cleanup: delete duplicate "system" sprawl and dead ML stubs.

**Non-goals (v1)**
- Live broker execution of advisor signals (rails exist; wiring is post-v1).
- Calibration / skill-weighting (v2 — you cannot weight on skill you have not measured out-of-sample).
- Information-diverse persona deciders (v2 — requires per-persona decorrelated data + a forward-tracking harness).
- Reviving FinRL/Qlib/FinBERT/AutoGluon stubs.

## 3. Runtime architecture

The advisor is one typed pipeline. **The graph proposes; the rails dispose.**

```
Data adapters (yfinance, FRED, NewsAPI)  ── our free sources, behind one interface
   │   (+ LLM fact-extractor upstream: filings/news → typed ExtractedFact)
   ▼
Deterministic quant — 5 signal families (pure Python, no LLM):
   1. value/quality fundamentals   2. price/time-series momentum
   3. trend overlays               4. macro/rates regime (FRED)
   5. news/sentiment surprise
   ▼
SignalBundle  { family, direction, confidence, skill_weight, as_of }   ← frozen Pydantic seam
   ▼
Risk manager (deterministic): vol- + correlation-adjusted position limits
   ▼
Portfolio allocator (deterministic v1): picks among pre-filtered LEGAL actions only
   ▼
Persona overlay (v1: read-only critique + explanation; may flag/veto, never upsize)
   ▼
Typed decision  →  our rails (TimescaleDB, kill-switch, vectorbt backtest w/ costs)
```

**Orchestration:** plain `asyncio.gather` over the independent signal families, with the `SignalBundle` serialized to TimescaleDB as the checkpoint. **No LangGraph** — the topology is a fixed fan-out/fan-in that never branches; a framework would add a churning dependency and a state machine we do not need. Adopt a framework only if we ever need mid-graph crash recovery.

**The load-bearing rule:** every number is computed in deterministic Python; the LLM only converts unstructured inputs to typed facts and (later) explains. The LLM never invents figures and never sets `direction` or `skill_weight` at the seam.

## 4. Components (target layout under `apps/quant/`)

| Module | Responsibility |
|---|---|
| `schemas/` | Pydantic models — `SignalBundle`, `ExtractedFact`, `Decision`, risk/portfolio types. Single source of truth. |
| `data/` | Existing adapters behind one `MarketDataProvider` interface + a point-in-time guard. Free data only. |
| `analysis/` | Deterministic signal families: `valuation.py` (DCF + owner-earnings + EV/EBITDA + residual income), `fundamentals.py`, `technicals.py`, `macro.py`, `sentiment.py`. |
| `valuation_primitives.py` | virattt's pure math, copied verbatim + MIT attribution, wrapped behind our adapter. |
| `llm/` | Provider abstraction — add Anthropic (Claude) + Ollama alongside OpenAI. Used only by the extractor/explainer. |
| `extract/` | LLM fact-extractor / red-flag tagger: filings/news → typed `ExtractedFact` (upstream of the seam; recall-oriented). |
| `risk/` | Deterministic vol- + correlation-adjusted position limits. |
| `portfolio/` | Deterministic allocator with `compute_allowed_actions` (LLM-free in v1). |
| `personas/` | Persona overlay — v1 critique + explanation; v2 information-diverse deciders. |
| `pipeline/` | `asyncio` fan-out/fan-in + TimescaleDB checkpoint. |

## 5. The persona/deliberation layer — phased

Deliberation adds value only when agents **know different things**, not when they have different personalities. Therefore:

**v1 — critique + explanation overlay (backtestable core stays pure).**
- The deterministic ensemble sizes positions.
- Personas run *after* the typed decision: they generate theses, flag qualitative red flags (accounting forensics, distress, structural risk), and explain the decision in-character ("value+quality flagged this").
- A persona may **veto/downgrade** on a red flag; it may **never upsize** or alter risk limits.
- This layer needs no backtest proof — its value (catching what numbers miss, explaining) is real live.

**v2 — information-diverse deciders.**
- Each persona is given its *own decorrelated data feed* (e.g. Burry → short-interest/credit-distress; Wood → innovation/TAM; Munger → accounting-quality). Only then is their "deliberation" informational.
- They may shade **direction/conviction** (still never hard risk limits).
- Validated by **forward paper-tracking**, not backtest — historically an LLM's pretraining already knows the outcomes, so backtest cannot prove their edge. Any LLM-derived *feature* that enters the deterministic core must pass the same purged CV as every other feature.

## 6. Data & the honest-backtest discipline (highest-leverage deliverable)

**Defensible:** a **price/volume-only walk-forward backtest** with vectorbt cost-aware fills. OHLCV is effectively point-in-time.

**Disclosed, not fixed:** yfinance fundamentals are *restated, not as-reported*, with no point-in-time vintage. On free data a true as-reported backtest is impossible. Mitigations:
1. Lag every fundamental by one reporting period + ~45–90 days to approximate availability.
2. **Snapshot fundamentals to TimescaleDB going forward from day one** to build our own point-in-time vintage history.
3. Reliability/calibration curves computed on price-derived signals only, where data is clean.

**Mandatory disclosure header — printed on every backtest report:**
1. Fundamentals are restated, not as-reported.
2. Point-in-time lag is approximated (~90-day proxy); results are "indicative, not as-reported."
3. yfinance is survivorship-biased (delisted names absent) → long-side results upward-biased.
4. Any LLM/news-derived feature may carry look-ahead from pretraining that cannot be purged.

## 7. Acceptance gates (the floor — prove it or ship nothing)

v1 is "effective" only if **both** hold on a frozen **purged walk-forward CV**, net of vectorbt costs:
1. **Beat the benchmark:** OOS Sharpe beats SPY buy-and-hold by a *pre-registered* margin.
2. **Beat the parts:** the ensemble beats its own best single signal family across **≥2 distinct market regimes**, with weights chosen *before* the test window.

Realized cross-family correlation is an empirical claim, not a design assumption — measure it OOS and re-shrink when families converge (macro and trend quietly correlate in regime shifts).

## 8. Calibration roadmap (v2)

- v1 ships **equal-weight** across families (the honest prior).
- v2 adds **shrinkage-to-equal-weight** skill-weighting: rolling rank-IC / information-ratio per family, cap any single family at 1/N–2/N, require a minimum OOS window before any weight deviates from equal, and gate inclusion on **Brier improvement over base rate** — not raw IC. In-sample weighting is overfitting with extra steps.

## 9. Build & dispatch — Hermes

Per the workspace contract, implementation edits/tests are dispatched via **Hermes** (`C:\dev\Updog_restore\orchestrate.js`). Hermes is a **build-time, multi-model dev-task router** (research→Claude, production→Codex, distribution→Claude; CLI binaries, not SDK). It is **not** the advisor's runtime and does not replace the `asyncio` pipeline.

**To target this Python repo, Hermes needs a small port:**
- New `.claude/hermes/model-routing.json` with portfolio/risk specialists (not waterfall/xirr).
- Python-flavored `DEV_BRAIN.md` / `.claude/hermes/SOUL.md`; de-hardcode the "Updog_restore" prompt string.
- Python gates (`pytest`, `ruff`) instead of `npm run check`.

**The high-value idea to steal from Hermes:** its non-negotiable financial `calc-gate`. Define an **`advisor-gate`** (the `calc-gate` analog) that runs the purged walk-forward CV + the §6 disclosure check + the §7 acceptance gates, and wire it as Hermes's **postflight gate** for this project. The §7 floor is then enforced by the build system on every change, not left to discipline.

**Caveats:** Codex fails ~5–10% on complex multi-file edits (Hermes's repair loop mitigates); Hermes is local/synchronous/single-machine; argument handling is brittle for prompts that begin with `---` frontmatter.

## 10. Cleanup (part of the win)

Delete:
- 7 duplicate root "systems": `production_ready_system.py`, `robust_trading_system.py`, `automated_trading_system.py`, `enhanced_training_system.py`, `personalized_portfolio_system.py`, `single_user_ai_system.py`, and siblings.
- Dead ML stubs: `finrl_trading_agent.py`, `qlib_factor_generator.py`, `autogluon_ensemble.py` (imports with no logic).
- The fake "Nash/Bayesian" `consensus_engine` claims → replaced by the deterministic risk + portfolio managers. (Salvage any genuinely useful debate scaffolding into the read-only persona overlay.)
- The always-positive mock NewsAPI fallback → missing source means the signal is marked `unavailable` and excluded, never fabricated.

## 11. Why this shape (rationale of record)

- **Personas not deciders:** deliberation among agents fed identical inputs sharing one base model is not deliberation — it is correlated opinion-averaging that manufactures confidence. Debate helps on tasks with a *verifiable answer* to converge toward; near-term markets have none.
- **Valuable-live ≠ validatable:** forward advice a human reads can carry real persona value, but a backtest cannot *prove* persona edge → unprovable signals must not size capital.
- **Edge lives in the deterministic layer + honest backtest + (later) calibration** — the capability virattt entirely lacks.
- **The most likely self-deception:** look-ahead laundered through the LLM — restated filing text smuggling future knowledge into "point-in-time" features. Honest prices do not save us if the words are post-hoc.

## 12. Testing strategy

- **Unit:** deterministic math (DCF, owner-earnings, vol/correlation position limits) against fixtures — fully deterministic, genuinely testable.
- **Contract:** persona/extractor tests with a **mocked LLM** returning canned JSON → assert fact assembly + Pydantic schema + neutral fallback on parse failure.
- **Integration:** full pipeline on a fixture ticker (stubbed data + LLM) → a valid typed decision within risk limits.
- **Backtest:** the §7 acceptance gates on the vectorbt harness with the §6 disclosure header.

## 13. Migration sequence (phased, each independently shippable)

1. Delete sprawl + dead stubs; repo builds clean.
2. `valuation_primitives.py` ported verbatim (MIT-attributed) behind our adapter on lagged free data.
3. **One signal family end-to-end** (raw → `SignalBundle`) as the reference slice; freeze the Pydantic contract; persist to TimescaleDB.
4. `asyncio` fan-out across all 5 families.
5. Deterministic risk + portfolio allocator → a position.
6. Equal-weight ensemble + frozen purged walk-forward CV harness + disclosure header (the §7 gate).
7. Persona overlay (v1 critique + explanation; `--explain`). *Cut first if time runs short — pure narration, zero effect on positions.*
8. Wire `advisor-gate` into the Hermes postflight.
9. (v2) Calibration; (v2) information-diverse persona deciders + forward-tracking harness.

**Definition of done (v1):** steps 1–7 green, the §7 gate passing on a pre-registered margin, one CLI command runs the pipeline with optional `--explain`.

## 14. Attribution

`virattt/ai-hedge-fund` is MIT-licensed. Copied valuation math retains attribution and the MIT notice.

## 15. Open assumptions

- Default LLM provider for extraction/explanation: **Anthropic (Claude)**, with OpenAI/Ollama via the `llm/` abstraction.
- Pre-registered Sharpe margin over SPY: **to be set before the first gated run** (placeholder pending a quick sensitivity pass on the free-data history length).
- Universe for v1 backtest: liquid US large-caps where yfinance coverage is cleanest.
