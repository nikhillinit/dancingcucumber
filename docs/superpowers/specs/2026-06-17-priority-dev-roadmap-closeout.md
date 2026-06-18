# Priority Dev Roadmap Closeout Spec

> Status: planning complete — **0/6 workstreams executed**. Planning-only dev spec; do not implement until an operator explicitly starts an execution lane. Roadmap debt is **sequenced, not closed**: **8 tracked** operator-facing docs (incl. `README.md`, `START_HERE.md`, `COMPLETE_SYSTEM_STATUS.md`, `IMPLEMENTATION_GUIDE.md`) still carry stale production/live-trading claims (WS1), and Reading B feasibility (provable point-in-time source) is unresolved (WS3 gate). Floor remains `DEV_FAILED`; holdout untouched; production sizing not authorized.

## Purpose

Sequence the closure of the current roadmap debt without weakening the advisor rails. Writing this spec does not by itself close any debt; closure happens only as the workstreams below are executed. The repo's authoritative state is a research-grade investment advisor with a healthy reporting gate and a correctly blocked release gate. The floor remains `DEV_FAILED`, the shared holdout is untouched, and production capital sizing is not authorized.

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
- `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md`: Reading B is the next candidate concept and is **still a stub** — it does NOT yet contain the fundamentals fixture schema or availability rule. WS3 must ADD them there (it is their home, not this closeout spec); this spec must not claim they already exist.
- `apps/quant/advisor/pipeline/run.py`: `run_pipeline(...) -> Decision` returns only the final decision; per-family diagnostics require a dedicated smoke helper.
- `apps/quant/advisor/data/provider.py`: yfinance fundamentals are RESTATED, not as-reported — live `value_quality` is not point-in-time evidence.
- `tools/run-floor.mjs` forwards only `--enforce`; `tools/floor_data_check.py` has a live `--holdout` unlock path, so the wrapper silently ignores `--holdout` (WS0 guard).

## RALPLAN-DR

Principles:
- Current truth leads: every public roadmap/status document must say `DEV_FAILED` until a gate proves otherwise.
- Holdout discipline is stronger than schedule pressure: no `--holdout` run without a dev-passing, pre-registered candidate and verified run hash.
- Live smokes are evidence, not promotion: keyed provider output can validate integration availability but cannot authorize sizing.
- Reading B must be point-in-time by construction — provable as-of visibility, NOT a strict lag applied to restated data; missing data is excluded, never filled with optimistic defaults.
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

## Workstream 0: Baseline and Gate Guard

Goal: record exact pre-cleanup repo state and close a release-semantics footgun before any other lane runs.

Scope:
- Record current commit SHA, `git status --short`, and the expected outcomes of `python -m pytest apps/quant/advisor/tests -q`, `npm run advisor-gate` (reports `DEV_FAILED`), and `node tools/run-floor.mjs --enforce` (exits `1`).
- Patch `tools/run-floor.mjs` to accept only no-args or `--enforce` and reject unknown flags. Today the wrapper forwards only `--enforce` and silently ignores everything else, while `tools/floor_data_check.py:69` has a live `--holdout` unlock path — so `node tools/run-floor.mjs --holdout` runs report mode while an operator may believe a holdout path executed. The wrapper must exit non-zero on `--holdout` (or any unknown flag) with a message that holdout is not reachable through this wrapper and needs a separate operator-approved lane. The legitimate direct path `python tools/floor_data_check.py --holdout` is unchanged.

Acceptance criteria:
- Floor verdict unchanged; no holdout run.
- `node tools/run-floor.mjs --holdout` fails fast instead of silently ignoring the flag.
- `node tools/run-floor.mjs --enforce` still exits `1` while the floor is `DEV_FAILED`.

## Workstream 1: Truth and Hygiene Reset

Goal: make the repo's operator-facing docs match the accepted floor and classify workspace hygiene risk without normalizing unrelated dirt.

Scope:
- Truth quarantine covers ALL tracked operator-facing docs, not just two. Verified tracked surfaces carrying stale production / live-trading / high-alpha claims: `README.md`, `START_HERE.md`, `SOPHISTICATION_ROADMAP.md`, `COMPLETE_SYSTEM_STATUS.md`, `IMPLEMENTATION_GUIDE.md`, `OPTIMIZED_SYSTEM_SUMMARY.md`, `OPTIONS_FLOW_SYSTEM_SUMMARY.md`, `bt_integration_summary.md`, plus any further tracked `.md`/`.txt` the truth scan finds.
- Each top-level operator doc's first viewport must state: research-grade advisor, `DEV_FAILED`, production sizing / live trading / broker execution / paper-trading promotion not authorized, holdout untouched, report/dev-run only. Old claims may survive ONLY inside an explicitly labelled obsolete/historical archive section that cites `FLOOR_RESULT.md` as superseding.
- Point the docs to `apps/quant/advisor/backtest/FLOOR_RESULT.md`, `apps/quant/advisor/backtest/VALIDATION_PREREG.md`, and the Reading B spec.
- Replace one-shot phrase greps with a test-backed denylist: add `apps/quant/advisor/tests/test_docs_truth.py`, modelled on the existing `apps/quant/advisor/tests/test_repo_cleanup.py` retired-path pattern, failing on unqualified current claims (`production ready`, `live trading`, `go live`, `real money`, `paper trading operational`, `automated order execution`, `Fidelity` automation, `expected annual alpha`, `50-70%`, `50-60% annual`, `28-35% annually`, `92% accuracy`, `95%+`, `Sharpe >2.5`, `immediate deployment`) outside an allow-listed obsolete archive.
- Inventory UNTRACKED scratch (e.g. `FIDELITY_TRADE_ORDERS.md`, `OVERALL_RECOMMENDATION.md`, `INSIDER_13F_INTELLIGENCE.md`, `EVALUATION_RESPONSE.md`, root `*.py`) into a hygiene note classified keep/delete/ignore/secret-risk; do not mass-delete without a separate cleanup approval. Untracked files get inventory, not quarantine.
- Secret scan stays redacted: report only file path, variable/provider class, and rotation-required yes/no — never the value. The earlier root-script key concern is already remediated (`fred_economic_analysis.py`/`alpha_vantage_enhanced_analysis.py` read env-only; a repo-wide scan found no hardcoded key literals) — still re-confirm fresh before closing WS1.

Non-goals:
- No root-script resurrection.
- No mass deletion of untracked files in this spec lane.
- No edit to `.claude/settings.local.json` unless the operator explicitly asks.

Acceptance criteria:
- `python -m pytest apps/quant/advisor/tests/test_docs_truth.py -q` passes: no tracked doc carries an unqualified current production / live-trading / high-alpha claim outside the obsolete archive; `test_repo_cleanup.py` still passes.
- Every top-level operator doc (the 8 named above) leads with `DEV_FAILED`, production sizing blocked, holdout untouched, and states `node tools/run-floor.mjs --enforce` exits `1` until the floor clears.
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

A smoke helper is REQUIRED, not optional: `run_pipeline(...) -> Decision` (`apps/quant/advisor/pipeline/run.py`) collapses the per-family signals internally and returns only the final `Decision`, so the aggregate `--families all` CLI cannot satisfy the per-family acceptance criteria. Add an operator-only helper under `scripts/` (e.g. `scripts/live_family_smoke.py`) that imports the same family coros the live path uses and prints, per family, `direction`/`confidence`/`reasoning` plus provider status metadata (FRED series id + observation window with `observation_end <= as_of`; Alpha Vantage `time_to <= as_of` + headline count; throttle/error category when observable; yfinance last-price date `<= as_of`). It must read env keys only, echo key presence as boolean `present`/`missing` only, never print keys or key-bearing URLs, and not write persistent logs by default.

Point-in-time caveat: the keyed smoke validates integration availability ONLY. It is NOT evidence that the `value_quality` leg is point-in-time safe — `apps/quant/advisor/data/provider.py` states yfinance fundamentals are RESTATED, not as-reported, and only approximate availability. The smoke note must say "integration availability only; no promotion; value_quality not PIT-proven."

Acceptance criteria:
- Missing-key behavior remains neutral/unavailable, not fabricated.
- The smoke helper (not just the aggregate CLI) records per-family `direction`/`confidence`/`reasoning` with provider status metadata, so "neutral" is attributable to coverage, thresholding, missing data, throttle, or outage.
- Present-key run records macro and sentiment as bullish, bearish, neutral, or unavailable for each ticker, with reasoning text but no secrets. Non-neutral output is useful evidence, not a requirement.
- Any provider throttle, quota, or coverage limitation is documented as an availability result, not a strategy failure.
- CLI output still includes the disclosure header.
- `npm run advisor-gate` still reports `DEV_FAILED`; `node tools/run-floor.mjs --enforce` still exits `1`.

## Workstream 3: Reading B Spec Completion

Goal: expand Reading B from a stub into an executable data-contract and preregistration spec for fundamental value with a timely price leg. No Reading B score should be interpreted until the fixture/provenance contract is satisfied.

Execution-order note: the tightened plan `.omx/plans/priority-dev-roadmap-closeout-tightened-ralplan-dr.md` splits this workstream into WS3A (ADD the source-agnostic contract to the Reading B stub — do not claim it already lives there), WS3B (PIT source feasibility gate; record under a `## PIT Source Feasibility Record` section in the Reading B spec), and WS3C (source-specific fixture/prereg, blocked on WS3B). That plan is the execution-order authority; this section states the gates.

Feasibility gate (blocking — resolve and record before any other WS3 work, and before the Reading B lane may be called "prepared"):
- Point-in-time means provable as-of visibility, NOT a strict lag on restated data. A current/restated value with a lag rule still leaks future amendments and does not qualify. Reading B may proceed only if the source supports one of: (1) as-reported accession-level reconstruction (the value traces to a specific filing/accession accepted before `as_of`); (2) a vendor/API point-in-time endpoint answering "what was knowable as of date X" whose license permits committing or regenerating the normalized values without redistribution; or (3) committed historical source snapshots whose snapshot hash proves what was visible at the snapshot date.
- Evaluate SEC EDGAR/XBRL first: company `submissions` and extracted XBRL company-facts are available via keyless JSON APIs and SEC filing content is free to reuse, subject to a declared User-Agent and the published request-rate ceiling. EDGAR qualifies ONLY if the implementation can select the exact accession/form/filed record knowable by the evaluation date; aggregate current company-facts alone do not.
- If the source provides only current/restated historical values and cannot prove pre-`as_of` visibility, STOP: label it `RESTATED_PROXY_ONLY`, do not build a fixture, do not preregister, do not run dev. A restated proxy may be explored as a non-candidate diagnostic only, never scored as a candidate.
- Output: a one-line provenance record — `source`, `data_class` (accession-XBRL | PIT-API | snapshot), `license` (commit normalized values yes/no), `redistribution` constraint, `as_of_mechanism` (accepted_datetime/filed_date/snapshot), `restatement_policy` (as-reported | restated-proxy), `fixture_committable` (yes/no), and fair-access limits if applicable — committed before any schema or dev-run work.

Required design decisions:
- Fundamentals source and license: name the source, redistribution constraints, and whether data can be committed as a fixture.
- Point-in-time model: use `report_date + REPORTING_LAG_DAYS` or a stricter as-of rule before any fundamental is visible.
- Restatement policy: prefer as-reported/accession-level values. If only restated values exist, the feasibility gate forces `RESTATED_PROXY_ONLY` — a strict lag does NOT convert restated data into point-in-time, so it cannot be scored as a candidate.
- Fixture schema and availability rule belong in the Reading B spec, not here — but that spec is currently a STUB and does NOT yet contain them. WS3 must ADD to `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` the full schema (asset/CIK/accession/form/report-period-end/filing-date/accepted-datetime/snapshot-hash/concept/unit/value/available-as-of/superseded-by/amended-flag/missingness-reason/denominator-policy) and the strict rule `available_asof = max(report_period_end + REPORTING_LAG_DAYS, filing_date, accepted_datetime, snapshot_date)`. This closeout spec REQUIRES they be added and gated there; it does not duplicate them and does not claim they already exist.
- Denominator / corporate-action bridge: freeze the denominator (book equity / market cap, or book value per share / price) before any score is interpreted, and prove price and share counts are split-adjusted on the same basis. Do not combine adjusted total-return prices with raw SEC share counts without a documented adjustment bridge.
- Snapshot-forward rule: after a datum becomes available, carry it forward only until superseded by the next available datum.
- Missing data rule: mark unavailable and exclude from candidate scoring; never fill with zero, sector median, current value, or future restatement.
- Timely price leg: denominator comes from the price fixture as of the evaluation date; numerator comes only from lagged fundamentals.
- Pre-registration: record candidate hash, candidate validation hash, fixture SHA, family set, lag, transforms, orthogonality kill-gate, trial count, and secondary-run rule before any measurement.

Bench reuse:
- Reuse `candidate_pipeline`, `candidate_blend`, `candidate_floor`, validation, and holdout logic.
- The current raw seam is price-series-only (`family + price series -> raw signal`); fundamental value needs fundamentals + filing dates + a price denominator. Introduce the minimal fixture/score-panel adapter (e.g. a `CandidateFixture` carrying price panel + fundamentals + provenance, or precomputed family score panels) and PROVE existing price-only family results are unchanged under the new path.
- Pre-register the trial count on `CandidateValidationPreReg.declared_trials_N` (the live DSR surface) — never the vestigial `CandidatePreReg.declared_trials_N` (`candidate_prereg.py:19`, already test-pinned).
- Do not assume the per-asset time-series percentile transform suits quarterly fundamentals; if reused for parity, add a power/coverage report before interpreting a dev failure.
- Preserve existing acceptance bars: ensemble beats best family, beats SPY by the floor margin, and DSR meets the candidate validation bar.

Acceptance criteria:
- The feasibility gate passes and its provenance record is committed before any fixture schema, preregistration, or dev-run work begins; the Reading B lane is not described as "prepared" until then.
- A leakage test fails if a fundamental value appears before its allowed as-of date.
- A fixture hash is recorded before the first dev run.
- Candidate hash and candidate validation hash are recorded before the first dev run.
- Source/license/provenance, snapshot dates, reporting lag, restatement policy, and missing-data exclusion are explicit before any score is interpreted.
- The candidate can run dev-only with `prereg_hash=None` and no reserved-tail reads.
- Source is as-reported/accession-level or PIT-API, or the lane stops as `RESTATED_PROXY_ONLY`; a restated proxy is never scored as a candidate.
- Existing price-only candidate results are invariant under the new fixture/adapter path (equivalence test), and the denominator/adjustment basis is frozen and documented before any score is interpreted.
- The spec explicitly says Reading B can fail or be power-limited without authorizing production or changing the floor.

## Workstream 4: Candidate Bench Run for Reading B

Goal: after Workstream 3 is approved and implemented, run Reading B through the existing candidate bench without touching holdout unless the dev gate earns it.

Dev-run sequence:
- Confirm prereg artifact and fixture hash are committed before measurement.
- Run candidate metrics with `prereg_hash=None`.
- Record dev verdict, fold deltas, weights, validation report, power/sufficiency report, and orthogonality diagnostics.
- If dev fails, stop. Do not run holdout. Update result docs and leave `HOLDOUT_LEDGER.md` unchanged.
- If dev passes but validation (DSR/MinBTL) fails, the DEFAULT is to stop and recommend no holdout burn; an operator may override only with an explicit holdout-burn decision memo. Validation stays report-only and cannot itself flip the floor, but a validation failure is a default holdout-stop.
- If dev passes and validation passes, stop for an operator holdout decision. Do not auto-run holdout from an execution agent.

Acceptance criteria:
- `HOLDOUT_LEDGER.md` remains empty unless a verified holdout run is explicitly approved and appended.
- Any holdout unlock uses only `candidate_run_hash(cfg, fixture)`, never an arbitrary string or fixture-blind candidate hash.
- Reading B result wording distinguishes: source-blocked, restated-proxy-only, power-limited negative, clean dev failure, dev-pass/validation-fail, and dev-pass/validation-pass. No production sizing language appears.

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
- Reading B's source cannot prove accession-level or API-level point-in-time visibility (stop as `RESTATED_PROXY_ONLY`), or a fixture mixes adjusted prices with raw share counts without a documented adjustment bridge.
- `candidate_hash` is confused with `candidate_run_hash`, or the vestigial `CandidatePreReg.declared_trials_N` is used as the live validation trial count.

## Verification Plan

Local proof expected after implementation lanes:
- `npm run advisor-gate` passes and reports `DEV_FAILED`.
- `node tools/run-floor.mjs --enforce` exits `1` until a future release-blocking plan changes the condition after a real candidate exists.
- `python -m pytest apps/quant/advisor/tests/test_docs_truth.py -q` passes (test-backed denylist over all tracked docs) and `test_repo_cleanup.py` still passes — replacing the brittle one-shot grep over just two files.
- `node tools/run-floor.mjs --holdout` fails fast (non-zero) instead of silently ignoring the flag.
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
$ralph execute docs/superpowers/specs/2026-06-17-priority-dev-roadmap-closeout.md through Workstream 0 and Workstream 1 only. Preserve unrelated dirt. Stop before any keyed live smoke or holdout-related action.
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

Scope check: this is one closeout spec with six ordered workstreams (WS0–WS5). Implementation is explicitly gated and holdout is explicitly excluded until a dev pass plus operator decision. Reading B's fundamentals fixture schema is referenced, not duplicated; WS3 must ADD it to the Reading B spec, which is currently a stub and does not yet contain it.

Consistency check: every production-readiness statement agrees with `FLOOR_RESULT.md`; validation remains report-only until the Plan 1b trigger.

Ambiguity resolved: the live-key smoke can produce non-neutral, neutral, or unavailable macro/sentiment results. The required output is a redacted observation, not a pass/fail investment claim.
