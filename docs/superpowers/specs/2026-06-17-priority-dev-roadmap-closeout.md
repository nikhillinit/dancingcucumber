# Priority Dev Roadmap Closeout Spec

> Status: planning-only dev spec. Do not implement from this document until an operator explicitly starts an execution lane.

## Purpose

Close the current roadmap debt without weakening the advisor rails. The repo's authoritative state is a research-grade investment advisor with a healthy reporting gate and a correctly blocked release gate. The floor remains `DEV_FAILED`, the shared holdout is untouched, and production capital sizing is not authorized.

This spec turns the outstanding roadmap items into an executable sequence that preserves the accepted negative, verifies the live five-family path, and prepares the next candidate lane, Reading B, without burning holdout or inventing new thresholds.

## No Promotion Semantics

None of these closeout tasks promotes the advisor. A cleaned-up roadmap, a keyed provider smoke, a validation report, or a completed Reading B spec does not change `DEV_FAILED`, does not touch holdout, does not authorize production sizing, and does not implement Plan 1b. The only success state for this closeout is a more truthful repo with a better-bounded next research lane.

## Source of Truth

- `apps/quant/advisor/backtest/FLOOR_RESULT.md`: `DEV_FAILED`; holdout not evaluated; family reweighting is closed; `node tools/run-floor.mjs --enforce` must continue to exit `1`.
- `apps/quant/advisor/backtest/VALIDATION_PREREG.md`: DSR/MinBTL validation is report-only and cannot flip the verdict, unlock holdout, or authorize sizing.
- `docs/superpowers/plans/2026-06-16-deferred-plans-roadmap.md`: validation-as-release-blocking is deferred until a real dev-passing candidate exists.
- `apps/quant/advisor/cli.py`: `--families all` assembles value/quality, trend, momentum, macro, and sentiment through the live CLI.
- `apps/quant/advisor/data/fred_provider.py` and `apps/quant/advisor/data/news_provider.py`: live providers read env keys and degrade to empty inputs when unavailable.
- `apps/quant/advisor/research/CANDIDATE_RESULT.md`: Reading A failed the dev gate but is power-limited, not a clean refutation of fundamental value.
- `apps/quant/advisor/research/HOLDOUT_LEDGER.md`: the shared reserved tail is untouched and may only unlock via `candidate_run_hash(cfg, fixture)`.
- `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md`: Reading B is the next candidate concept but is still a stub.

## RALPLAN-DR

Principles:
- Current truth leads: every public roadmap/status document must say `DEV_FAILED` until a gate proves otherwise.
- Holdout discipline is stronger than schedule pressure: no `--holdout` run without a dev-passing, pre-registered candidate and verified run hash.
- Live smokes are evidence, not promotion: keyed provider output can validate integration availability but cannot authorize sizing.
- Reading B must be point-in-time by construction; missing data is excluded, never filled with optimistic defaults.
- Keep unrelated worktree dirt out of scope.

Decision drivers:
- Auditability: status claims need a file, command, hash, or ledger anchor.
- Statistical discipline: no new thresholds, no new release authority, no trial-count ambiguity hidden in prose.
- Operator safety: no secrets in commits, logs, docs, shell history snippets, or generated artifacts.

Viable approaches:

1. Truth-first closure plus keyed smoke plus Reading B hardening.
   - Pros: closes stale claims, validates Workstream C's live seams, and prepares the next candidate without touching holdout.
   - Cons: does not unblock production capital immediately.

2. Wire validation into release blocking first.
   - Pros: improves long-run false-positive control.
   - Cons: explicitly premature without a dev-passing candidate; risks changing release behavior in a lane that has no promotable strategy.

3. Build a new alpha candidate first.
   - Pros: fastest path to another measurement.
   - Cons: repeats the old failure mode if stale docs, secret hygiene, preregistration, and holdout boundaries remain ambiguous.

Decision: choose approach 1. Defer approach 2 until a real candidate clears dev. Reject approach 3 for now because the repo needs truth and rail cleanup before more alpha work is trustworthy.

## Workstream 1: Truth and Hygiene Reset

Goal: make the repo's operator-facing docs match the accepted floor and classify workspace hygiene risk without normalizing unrelated dirt.

Scope:
- Update `START_HERE.md` so the first viewport states: research-grade advisor, `DEV_FAILED`, production sizing blocked, holdout untouched, report/dev-run only.
- Update `SOPHISTICATION_ROADMAP.md` so historical high-return, high-accuracy, immediate paper-trading, and production automation claims are removed or clearly marked obsolete.
- Point both docs to `apps/quant/advisor/backtest/FLOOR_RESULT.md`, `apps/quant/advisor/backtest/VALIDATION_PREREG.md`, and the Reading B spec.
- Inventory untracked scratch outputs and root scripts into a hygiene note or checklist. Classify each category as keep, delete, ignore, or secret-risk; do not delete broad sets without a separate cleanup approval.
- Search untracked local scripts for likely embedded keys without printing secret values. If real keys are found, document only file path, variable/provider, and rotation requirement.

Non-goals:
- No root-script resurrection.
- No mass deletion of untracked files in this spec lane.
- No edit to `.claude/settings.local.json` unless the operator explicitly asks.

Acceptance criteria:
- Text scan no longer finds unqualified current claims such as `92% ML accuracy`, `28-35% annually`, `50-60% annual returns`, `Paper trading operational`, `Start Paper Trading TODAY`, or `Go Live Small`.
- Both status docs state `node tools/run-floor.mjs --enforce` exits `1` until the floor clears.
- Hygiene note records the dirty tracked file and untracked categories without committing secrets or unrelated generated output.

## Workstream 2: Keyed Live Smoke for Workstream C

Goal: verify that the existing five-family CLI can consume rotated live `FRED_API_KEY` and `ALPHAVANTAGE_API_KEY` with env-only secrets, as-of-bounded provider calls, disclosure-preserving output, and honest neutral degradation.

Scope:
- Use at least two liquid tickers, for example `AAPL` and `MSFT`.
- Run with explicit `PYTHONPATH=apps/quant`, rotated env keys, and an as-of date supplied by the operator.
- Capture per-family direction, confidence, and reasoning for macro and sentiment in a redacted smoke note.
- Also record the aggregate `--families all` CLI output for each ticker.
- Treat neutral macro/sentiment as a valid result if provider coverage, thresholding, missing data, or unavailable response explains it.
- Where observable through a helper or injected provider, confirm FRED `observation_end` and Alpha Vantage `time_to` are capped at `as_of`.

Recommended command shape:

```powershell
$env:PYTHONPATH = "apps/quant"
# Operator sets rotated keys in the shell. Do not paste them into docs or scripts.
python -m advisor AAPL --families all --as-of 2026-06-17
python -m advisor MSFT --families all --as-of 2026-06-17
```

If the aggregate CLI output does not expose enough per-family detail, add a small operator-only smoke helper under `scripts/` that imports the existing family coros and prints redacted family summaries. That helper must read env keys only, must not echo key presence beyond boolean `present`/`missing`, and must not write persistent logs by default.

Acceptance criteria:
- Missing-key behavior remains neutral/unavailable, not fabricated.
- Present-key run records macro and sentiment as bullish, bearish, neutral, or unavailable for each ticker, with reasoning text but no secrets. Non-neutral output is useful evidence, not a requirement.
- Any provider throttle, quota, or coverage limitation is documented as an availability result, not a strategy failure.
- CLI output still includes the disclosure header.
- `npm run advisor-gate` still reports `DEV_FAILED`; `node tools/run-floor.mjs --enforce` still exits `1`.

## Workstream 3: Reading B Spec Completion

Goal: expand Reading B from a stub into an executable data-contract and preregistration spec for fundamental value with a timely price leg. No Reading B score should be interpreted until the fixture/provenance contract is satisfied.

Required design decisions:
- Fundamentals source and license: name the source, redistribution constraints, and whether data can be committed as a fixture.
- Point-in-time model: use `report_date + REPORTING_LAG_DAYS` or a stricter as-of rule before any fundamental is visible.
- Restatement policy: state whether the fixture uses as-reported values or restated historical values; if restated values are used, disclose that limitation and keep the lag rule strict.
- Fixture schema: include asset, report date, available-as-of date, snapshot date, book value or book value per share, shares outstanding if needed, source id, source vintage if available, and missingness reason.
- Snapshot-forward rule: after a datum becomes available, carry it forward only until superseded by the next available datum.
- Missing data rule: mark unavailable and exclude from candidate scoring; never fill with zero, sector median, current value, or future restatement.
- Timely price leg: denominator comes from the price fixture as of the evaluation date; numerator comes only from lagged fundamentals.
- Pre-registration: record candidate hash, candidate validation hash, fixture SHA, family set, lag, transforms, orthogonality kill-gate, trial count, and secondary-run rule before any measurement.

Bench reuse:
- Reuse `candidate_pipeline`, `candidate_blend`, `candidate_floor`, validation, and holdout logic.
- Add only the minimal signal/fixture adapters needed for Reading B.
- Preserve existing acceptance bars: ensemble beats best family, beats SPY by the floor margin, and DSR meets the candidate validation bar.

Acceptance criteria:
- A leakage test fails if a fundamental value appears before its allowed as-of date.
- A fixture hash is recorded before the first dev run.
- Candidate hash and candidate validation hash are recorded before the first dev run.
- Source/license/provenance, snapshot dates, reporting lag, restatement policy, and missing-data exclusion are explicit before any score is interpreted.
- The candidate can run dev-only with `prereg_hash=None` and no reserved-tail reads.
- The spec explicitly says Reading B can fail or be power-limited without authorizing production or changing the floor.

## Workstream 4: Candidate Bench Run for Reading B

Goal: after Workstream 3 is approved and implemented, run Reading B through the existing candidate bench without touching holdout unless the dev gate earns it.

Dev-run sequence:
- Confirm prereg artifact and fixture hash are committed before measurement.
- Run candidate metrics with `prereg_hash=None`.
- Record dev verdict, fold deltas, weights, validation report, power/sufficiency report, and orthogonality diagnostics.
- If dev fails, stop. Do not run holdout. Update result docs and leave `HOLDOUT_LEDGER.md` unchanged.
- If dev passes, stop for an operator holdout decision. Do not auto-run holdout from an execution agent.

Acceptance criteria:
- `HOLDOUT_LEDGER.md` remains empty unless a verified holdout run is explicitly approved and appended.
- Any holdout unlock uses only `candidate_run_hash(cfg, fixture)`, never an arbitrary string or fixture-blind candidate hash.
- Reading B result wording distinguishes clean refutation, power-limited negative, and dev pass.

## Workstream 5: Conditional Release Blocking

Goal: keep Plan 1b deferred until it has a real purpose, then wire validation into release blocking only after a dev-passing candidate exists.

Trigger:
- A candidate has passed dev using pre-registered methodology and fixture.
- The operator approves writing the release-blocking validation plan.

Required behavior when triggered:
- `node tools/run-floor.mjs --enforce` must require both strategy pass and validation pass before release approval.
- A deflation-failing candidate cannot be released.
- The change must not retroactively reinterpret today's accepted `DEV_FAILED` floor.

Acceptance criteria:
- Before trigger: validation stays report-only.
- After trigger: tests prove validation failure blocks release even if the strategy verdict would otherwise pass.
- Documentation says release blocking applies prospectively to the candidate lane, not as a greenwash of the current floor.

## Global Stop Rules

Stop and report instead of continuing if:
- Any task would run `--holdout` or touch reserved-tail data before a dev pass and operator decision.
- Any command would print, persist, or commit API keys.
- Any doc would imply production capital, broker execution, or real sizing is authorized.
- Any implementation requires a new dependency.
- Any cleanup would delete or move broad untracked categories without an explicit cleanup approval.
- Any candidate result requires changing the acceptance bars after seeing metrics.

## Verification Plan

Local proof expected after implementation lanes:
- `npm run advisor-gate` passes and reports `DEV_FAILED`.
- `node tools/run-floor.mjs --enforce` exits `1` until a future release-blocking plan changes the condition after a real candidate exists.
- `rg -n "92% ML accuracy|28-35% annually|50-60% annual returns|Paper trading operational|Start Paper Trading TODAY|Go Live Small" START_HERE.md SOPHISTICATION_ROADMAP.md` returns no unqualified current claims.
- Keyed smoke notes redact keys, preserve disclosures, and report macro/sentiment state or provider limitation for at least two tickers.
- Reading B prereg, fixture hash, candidate hash, and candidate validation hash exist before any dev-run result doc.
- Holdout ledger remains unchanged unless a holdout run is explicitly approved and logged.

Team verification path:
- `writer`: doc truthfulness scan and patch.
- `test-engineer`: live-smoke helper/procedure and redaction checks.
- `architect`: Reading B fixture/prereg boundary review.
- `critic`: holdout and validation-release-blocking challenge review.
- `verifier`: final gate and text-scan evidence.

## Execution Handoff

Recommended sequential `ralph` path:

```text
$ralph execute docs/superpowers/specs/2026-06-17-priority-dev-roadmap-closeout.md through Workstream 1 only. Preserve unrelated dirt. Stop before any keyed live smoke or holdout-related action.
```

Recommended parallel `$team` path after Workstream 1:

```text
$team execute docs/superpowers/specs/2026-06-17-priority-dev-roadmap-closeout.md with lanes: keyed-smoke-procedure, Reading-B-spec-hardening, and verifier. Do not run holdout. Do not expose keys. Keep validation report-only.
```

Suggested reasoning by lane:
- Truth docs: medium.
- Hygiene/secret-risk inventory: high.
- Keyed smoke: medium, with strict redaction.
- Reading B prereg/fixture design: high.
- Final verification: high.

## ADR

Decision: close roadmap debt in the order truth/hygiene, keyed smoke, Reading B spec, Reading B dev-run, then conditional release blocking.

Drivers: preserve accepted negative evidence, avoid holdout leakage, prevent secret exposure, and make the next candidate auditable before measurement.

Alternatives considered:
- Validation release blocking immediately: rejected as premature until a dev-passing candidate exists.
- New alpha first: rejected because stale docs and rail ambiguity would make the result harder to trust.
- Broker/product automation: rejected because the research gate is the bottleneck.

Consequences:
- The repo remains blocked for production capital.
- The operator gets a safer roadmap and a concrete next research lane.
- Reading B can still fail; success is measured by disciplined evidence, not by forcing a pass.

Follow-ups:
- After Workstream 1, decide whether to execute keyed smoke in the current workspace or a clean worktree.
- After Reading B spec completion, run an adversarial review specifically on leakage and holdout discipline.
- After a dev-passing candidate, write a separate Plan 1b spec for validation release blocking.

## Self-Review

Placeholder scan: no unfinished marker text or paste markers are intentionally left.

Scope check: this is one closeout spec with five ordered workstreams. Implementation is explicitly gated and holdout is explicitly excluded until a dev pass plus operator decision.

Consistency check: every production-readiness statement agrees with `FLOOR_RESULT.md`; validation remains report-only until the Plan 1b trigger.

Ambiguity resolved: the live-key smoke can produce non-neutral, neutral, or unavailable macro/sentiment results. The required output is a redacted observation, not a pass/fail investment claim.
