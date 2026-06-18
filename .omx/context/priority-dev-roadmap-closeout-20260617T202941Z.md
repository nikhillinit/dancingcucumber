# Priority Dev Roadmap Closeout Context

Task statement: review the priority dev roadmap and produce a corresponding dev spec that closes outstanding tasks with robust, durable output.

Desired outcome:
- A planning/spec artifact, not implementation.
- Preserve the accepted `DEV_FAILED` floor and production-capital block.
- Sequence outstanding work so truth/hygiene, keyed live smoke, Reading B, candidate-bench execution, and conditional release blocking happen in the right order.

Known facts/evidence:
- `apps/quant/advisor/backtest/FLOOR_RESULT.md` says the floor verdict is `DEV_FAILED`, holdout was not evaluated, family reweighting is closed, and `node tools/run-floor.mjs --enforce` must remain release-blocking.
- `apps/quant/advisor/backtest/VALIDATION_PREREG.md` records validation as report-only; it cannot flip the verdict, unlock holdout, or authorize sizing.
- `docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md` defers validation-as-release-blocking until a real candidate exists.
- `apps/quant/advisor/cli.py` has `--families all`, assembling value/quality, trend, momentum, macro, and sentiment.
- `apps/quant/advisor/data/fred_provider.py` and `apps/quant/advisor/data/news_provider.py` read `FRED_API_KEY` and `ALPHAVANTAGE_API_KEY` from env and degrade safely when unavailable.
- `apps/quant/advisor/research/CANDIDATE_RESULT.md` records Reading A as `DEV_FAILED`, power-limited, and not a clean refutation.
- `apps/quant/advisor/research/HOLDOUT_LEDGER.md` says the shared tail is untouched and may only unlock via `candidate_run_hash(cfg, fixture)`.
- `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` is a planning-only stub for the next candidate.
- `START_HERE.md` and `SOPHISTICATION_ROADMAP.md` contain stale paper-trading/high-return claims that conflict with the floor.
- Latest feedback validated against repo reality:
  - `docs/superpowers/specs/2026-06-17-priority-dev-roadmap-closeout.md` already absorbs most earlier corrections but still overstates that the Reading B schema "lives" in the Reading B spec.
  - Reading B should split into source-agnostic contract hardening, PIT source feasibility, and source-specific fixture/prereg design.
  - `run_pipeline(...) -> Decision` hides family/provider internals, so keyed smoke needs a helper.
  - FRED and Alpha Vantage providers collapse missing keys, throttles, empty data, and request failures into empty outputs; smoke output must separate provider status from signal status.
  - `tools/run-floor.mjs` forwards only `--enforce`; unknown flags such as `--holdout` must fail fast through the Node wrapper.
  - Truth quarantine needs archive/allow-marker policy so obsolete historical claims do not create false positives while current operator-facing claims still fail.

Constraints:
- Planning/spec-only under `$ralplan`; do not start implementation.
- Do not normalize unrelated dirty worktree state.
- No secrets in code, docs, commits, or logs.
- Do not touch holdout unless a dev-passing candidate earns it and the unlock hash is verified.
- Do not wire validation into release blocking before a real dev-passing candidate exists.
- Broker/product automation and real sizing stay behind research gates.

Unknowns/open questions:
- Exact live-key smoke output is unknown until an operator runs with rotated keys.
- Fundamentals data source/licensing for Reading B is not selected.
- Whether Reading B should reuse the same reserved tail or create a fresh tail remains a gated decision after dev pass, not a spec-time assumption.
- Whether SEC EDGAR accession reconstruction is sufficient for the specific Reading B fields remains a feasibility-gate output, not an assumption.

Likely codebase touchpoints:
- `START_HERE.md`
- `SOPHISTICATION_ROADMAP.md`
- `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md`
- `docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md`
- `apps/quant/advisor/cli.py`
- `apps/quant/advisor/data/fred_provider.py`
- `apps/quant/advisor/data/news_provider.py`
- `apps/quant/advisor/research/`
- `apps/quant/advisor/backtest/`
