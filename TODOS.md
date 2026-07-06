# TODOS

Deferred work with context. Added by /plan-ceo-review 2026-07-06 (skill_weight seam plan).

## P2 — Surface skill weights in decision output (S)

- **What:** When a bundle carries non-uniform `skill_weight`s, surface them in human-facing output — append weights to `Allocation.reasoning` (`apps/quant/advisor/portfolio/allocator.py:46`) and/or print `vote_parity()` in the CLI `--families all` report.
- **Why:** Post seam-activation, a non-default weight shifts live votes with no human-visible trace (weights persist to the checkpoint DB only). This is the designated silent-failure surface once weights ever go non-uniform.
- **Pros:** Closes the last silent path around the weighted vote.
- **Cons:** Dead code until a validated calibration exists; the run_pipeline tripwire (same PR as seam activation) blocks non-uniform weights meanwhile.
- **Context:** The seam-activation PR adds a fail-fast in `pipeline/run.py` rejecting non-uniform weights until a validated calibration artifact exists (spec §8: shrinkage, rank-IC/IR, caps, min OOS window, Brier gating — `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md:119`). **When that tripwire is relaxed, this observability MUST land in the same change.**
- **Depends on / blocked by:** Nothing to build; activation blocked by a validated calibration source (none exists as of 2026-07).

## P3 — Constant-mix mode for the diagnostics CLI (M)

- **What:** Add a `--weights` input mode to `advisor.diagnostics` that accepts a static-weights CSV and reports risk for the *constant-mix rebalanced* strategy those weights define (daily rebalance to target), clearly labeled as such — distinct from the held-book qty mode.
- **Why:** v1 rejects weights-only input by design (eng review T3): a static-weights book is constant-mix, a different strategy whose numbers do not describe the held portfolio. But operators sometimes only *have* target weights (model portfolios, IPS allocations); a correctly-labeled constant-mix report is the honest way to serve that input.
- **Pros:** Unlocks the most common non-qty input without lying about what it measures; reuses the whole report/banner/sentinel stack unchanged.
- **Cons:** Two modes = doubled input-validation surface and a new mislabeling risk (users reading constant-mix numbers as held-book numbers); needs its own banner line and tests.
- **Context:** Rejection error message in `apps/quant/advisor/diagnostics/portfolio.py` already names constant-mix as the reason; plan doc `docs/superpowers/plans/2026-07-06-decision2-diagnostics-cli.md` pins it as a TODO candidate, not v1.
- **Depends on / blocked by:** Decision-2 lane surviving its +60-day kill criterion (no point widening a tool nobody runs).

## P3 — Long/short book diagnostics (M) — GATED ON DECISION 5

- **What:** Accept negative qty in `advisor.diagnostics` and report L/S-aware risk: gross/net exposure, long and short sleeves separately, drawdown on the combined book; concentration computed on absolute weights.
- **Why:** v1 rejects shorts outright (long-flat only). If Decision 5 (program-review memo) ever opens a long-short lane — the reversed +0.32..+0.41 in-sample IRs from the residual screen are a pre-registered L/S *hypothesis* — the operator's real book could carry shorts and v1 would refuse it.
- **Pros:** Removes the hard rejection for a real book shape; sleeves-separated reporting is strictly more informative.
- **Cons:** Sharpe/Sortino/LCB semantics get subtler with negative weights (weights_book normalization, cash treatment for short proceeds); easy to produce plausible-but-wrong numbers; meaningful only if a short book ever exists.
- **Context:** Shorts-rejection lives in `apps/quant/advisor/diagnostics/portfolio.py` (non-positive-qty guard). L/S research direction documented in the blend-futility memory and program-review memo §Decision 5.
- **Depends on / blocked by:** **Decision 5 being picked by the operator** (not picked as of 2026-07-06); do not build speculatively.

## P3 — Broker-export ingestion for positions.csv (S)

- **What:** A small documented converter (separate script or `--from-broker <format>` flag) that turns common broker position exports (Fidelity/Schwab/IBKR CSV) into the canonical `ticker,qty[,CASH]` positions.csv, mapping cash sweep rows to the CASH convention.
- **Why:** v1's positions.csv is hand-authored; every manual transcription of a broker export is an error opportunity (wrong qty, missed cash, option symbols pasted as tickers). The kill criterion (≥8 real runs in 60 days) lives or dies on friction.
- **Pros:** Directly attacks the adoption friction the kill criterion measures; converter is pure text-to-text and trivially testable with fixture exports.
- **Cons:** Broker formats churn without notice (silent breakage risk); N formats = N maintenance surfaces; option/bond/MMF rows need explicit rejection rules to avoid garbage-in.
- **Context:** CASH-row and qty-only conventions pinned in `docs/superpowers/plans/2026-07-06-decision2-diagnostics-cli.md` (interface contract). Converter must stay OUTSIDE the CLI's no-network, no-write core (same pattern as the separate price_fetch.py step).
- **Depends on / blocked by:** Decision-2 lane surviving its +60-day kill criterion; sample export fixtures from the operator's actual broker(s).

