# Tightened RALPLAN-DR: Priority Dev Roadmap Closeout

Status: planning-only draft for `C:\dev\AIHedgeFund`. Do not edit source, do not run holdout, do not implement from this artifact until a separate execution handoff is approved.

Primary artifacts reviewed:
- `docs/superpowers/specs/2026-06-17-priority-dev-roadmap-closeout.md`
- `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md`
- `.omx/context/priority-dev-roadmap-closeout-20260617T202941Z.md`

## RALPLAN-DR

### Principles

1. Current truth leads: `DEV_FAILED`, holdout untouched, validation report-only, and production sizing blocked remain the dominant public claims.
2. Holdout discipline outranks roadmap closure: no `--holdout`, no reserved-tail read, and no fixture-blind unlock path before dev pass plus explicit operator decision.
3. Evidence must be fresh and scoped: secret scans, provider smokes, and gate outcomes are evidence to collect during execution, not settled claims in the plan.
4. Reading B must be point-in-time by construction: accession/as-reported or true PIT source evidence is required before source-specific fixture or prereg design.
5. Smoke telemetry is diagnostic, not promotion: provider availability/status must be separated from investment signal status.

### Decision Drivers

1. Auditability: every changed roadmap claim needs a file, command, hash, marker, or ledger anchor.
2. Statistical integrity: preserve `CandidateValidationPreReg.declared_trials_N`, validation-fail default holdout stop, existing acceptance bars, and no outcome-driven threshold changes.
3. Operator safety: no secrets in logs/docs, no stale production claims in first viewport, no broad provider refactor disguised as smoke cleanup.

### Options

#### Option A: Tighten the current closeout plan in-place, with WS3 split and WS2 smoke-helper contract

Pros:
- Preserves the existing six-workstream sequence while correcting the validated defects.
- Makes `run-floor.mjs --holdout` rejection a mandatory WS0 rail before any other work.
- Turns Reading B from an overclaimed prepared lane into a gated contract and feasibility sequence.
- Keeps provider-status work bounded to the smoke helper and fixtures.

Cons:
- Leaves the Reading B spec stub incomplete until a follow-on docs execution lane edits it.
- Does not produce immediate candidate data or production unblock.

#### Option B: Pause closeout and rewrite Reading B first

Pros:
- Fixes the most technically consequential gap before touching broad roadmap docs.
- Reduces risk that downstream execution treats Reading B as prepared.

Cons:
- Leaves stale operator-facing truth claims and `run-floor.mjs` flag ambiguity unresolved longer.
- Does not address provider smoke/reporting hygiene or secret-scan evidence.
- Creates sequencing drift from the already validated closeout context.

#### Option C: Execute keyed provider smoke and Reading B feasibility before truth quarantine

Pros:
- Produces new evidence earlier for live provider availability and source feasibility.

Cons:
- Requires credentials and external calls before first fixing public truth and flag-guard rails.
- Risks confusing nonneutral signal output with provider health.
- Does not address current stale docs before generating more operator-facing notes.

Decision: choose Option A. It is the smallest correction that incorporates the validated feedback while preserving the core rail structure.

Execution order refinement: run the cheap PIT feasibility decision earlier than the credentialed smoke. The recommended first execution slice is WS0 -> WS1 -> WS3A -> WS3B. WS2 remains required, but it should not delay a cheap `RESTATED_PROXY_ONLY` stop if Reading B has no provable PIT source.

## Concrete Revised Workstream Deltas

### WS0: Baseline and Gate Guard

Delta:
- Make unknown flag rejection mandatory, not optional cleanup.
- `tools/run-floor.mjs` must reject every arg except no args and `--enforce`.
- `node tools/run-floor.mjs --holdout` must fail fast with an explicit message that holdout is not reachable through the wrapper and requires a separate operator-approved lane.
- The direct Python holdout path remains unchanged but out of scope.

Acceptance criteria:
- Baseline records current SHA and dirty state without normalizing unrelated files.
- `node tools/run-floor.mjs --holdout` exits non-zero without touching holdout data.
- `node tools/run-floor.mjs --enforce` still exits `1` while the floor is `DEV_FAILED`.
- `npm run advisor-gate` remains report-only and does not promote the floor.

### WS1: Truth Quarantine and Fresh Hygiene Evidence

Delta:
- Truth quarantine needs an archive/allow-marker policy, not only phrase replacement.
- Current first viewport of `README.md` and other operator entry docs must not be archived away. It must lead with `DEV_FAILED`, production sizing blocked, holdout untouched, and report/dev-run only.
- Historical claims may remain only inside explicitly marked obsolete archive blocks that cite `apps/quant/advisor/backtest/FLOOR_RESULT.md` as superseding.
- Secret scan cleanliness must be fresh execution evidence. Do not state it as already settled in the revised closeout spec.

Truth-scan scope:
- Required active-doc set: `README.md`, `START_HERE.md`, `SOPHISTICATION_ROADMAP.md`, `COMPLETE_SYSTEM_STATUS.md`, `IMPLEMENTATION_GUIDE.md`, `OPTIMIZED_SYSTEM_SUMMARY.md`, `OPTIONS_FLOW_SYSTEM_SUMMARY.md`, and `bt_integration_summary.md`.
- Additional tracked scope: every tracked `.md` or `.txt` returned by `git ls-files '*.md' '*.txt'` that matches the denylist.
- Untracked scratch is inventoried separately via `git ls-files -o --exclude-standard`; it is not quarantined or mass-deleted in this lane.

Denylist terms to encode from the closeout spec:
- `production ready`
- `live trading`
- `go live`
- `real money`
- `paper trading operational`
- `automated order execution`
- `Fidelity` automation
- `expected annual alpha`
- `50-70%`
- `50-60% annual`
- `28-35% annually`
- `92% accuracy`
- `95%+`
- `Sharpe >2.5`
- `immediate deployment`

Archive marker syntax:

```text
<!-- AIHF_TRUTH_ARCHIVE_START superseded_by="apps/quant/advisor/backtest/FLOOR_RESULT.md" reason="obsolete historical claim" -->
...obsolete historical content...
<!-- AIHF_TRUTH_ARCHIVE_END -->
```

Marker rules:
- Start and end markers must be balanced and non-nested.
- `superseded_by` must exactly reference `apps/quant/advisor/backtest/FLOOR_RESULT.md`.
- `reason` is required and must be non-empty.
- Archive markers are invalid inside the first 40 physical lines of `README.md` or any required active-doc entrypoint.
- First-viewport claims are never exempt: current truth must appear before any archive block.

Fresh redacted secret-scan command/scope:

```powershell
$patterns = [ordered]@{
  FRED = '(?i)\b(FRED_API_KEY|fred[_-]?api[_-]?key)\b\s*[:=]\s*[''"]?[A-Za-z0-9_\-]{12,}';
  ALPHAVANTAGE = '(?i)\b(ALPHAVANTAGE_API_KEY|alpha[_-]?vantage.*api[_-]?key|apikey)\b\s*[:=]\s*[''"]?[A-Za-z0-9_\-]{8,}';
}
$files = git ls-files -co --exclude-standard |
  Where-Object { $_ -match '\.(py|js|mjs|ts|tsx|md|txt|json|yaml|yml|env)$' -and $_ -notmatch '(^|/)(\.git|node_modules|\.venv|venv|dist|build|__pycache__)/' }
$findings = foreach ($f in $files) {
  $text = Get-Content -LiteralPath $f -Raw -ErrorAction SilentlyContinue
  foreach ($name in $patterns.Keys) {
    if ($text -match $patterns[$name]) {
      [pscustomobject]@{ path = $f; provider_or_variable_class = $name; rotation_required = 'yes' }
    }
  }
}
$unique = @($findings | Sort-Object path, provider_or_variable_class -Unique)
if ($unique.Count -eq 0) {
  "[]"
} else {
  $unique | ConvertTo-Json
}
```

The scan output may contain only `path`, `provider_or_variable_class`, and `rotation_required`. It must not print matching lines, values, URLs, or snippets. If no findings exist, record `[]` plus command/date/scope. `rotation_required = yes` means a key-like literal or secret-bearing assignment was found and the operator should assume rotation is required until manually disproven; `rotation_required = no` is allowed only in a separate manual review note for a false-positive class, not in the raw scan output.

Acceptance criteria:
- A docs truth test denies unqualified current production/live-trading/high-alpha claims outside allow-marked obsolete archive sections.
- Top-level entry docs, especially `README.md`, have first-viewport truth text, not only archived corrections later in the file.
- The docs truth test fails on unmatched markers, nested markers, missing `superseded_by`, archive markers in the first 40 lines of required active docs, and denylist claims in unmarked current sections.
- Secret scan report records command/date/scope and redacted findings only: path, provider/variable class, rotation-required yes/no. No key values.
- Untracked scratch is inventoried, not mass-deleted.

### WS2: Keyed Live Smoke, Provider-Status First

Delta:
- Scope the work to an operator-only smoke helper and tests, not a broad production-provider refactor.
- Add separate concepts in smoke output: `provider_status` and `signal_status`.
- `provider_status` answers whether the provider call was usable, missing-key, throttled, unavailable, empty, or error-redacted.
- `signal_status` answers whether the produced family signal is bullish, bearish, neutral, or unavailable.
- `nonneutral_signal` is not provider status and must not be used as a provider-health proxy.

Acceptance criteria:
- Helper reads keys from env only and reports key presence as boolean only.
- Helper reports provider metadata sufficient to explain neutral/unavailable outcomes: FRED series/window and `observation_end <= as_of`; Alpha Vantage `time_to <= as_of`, headline count, throttle/error category when observable; yfinance last price date `<= as_of`.
- Present-key run may produce bullish, bearish, neutral, or unavailable. Non-neutral output is useful evidence but never required.
- Aggregate CLI disclosure remains visible, and `npm run advisor-gate` plus `run-floor --enforce` remain unchanged.

### WS3A: Reading B Source-Agnostic Contract Amendment

Delta:
- First edit the Reading B stub itself. Do not claim the strengthened schema already lives there.
- Add source-agnostic contract language before selecting a provider.
- Required contract includes: accession/form/report-period-end/filing-date/accepted-datetime or snapshot date, concept/unit/value, available-as-of, supersession/amended flag, missingness reason, denominator policy, price/share adjustment bridge, fixture hash, candidate hash, candidate validation hash, and `CandidateValidationPreReg.declared_trials_N`.
- Explicitly remove or supersede the stub's `CandidatePreReg` trial-count implication for validation; `CandidatePreReg` may describe strategy config, but validation trial count lives on `CandidateValidationPreReg`.

Acceptance criteria:
- `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` is no longer a stub and contains the source-agnostic schema/availability contract.
- The spec says a strict lag on restated data is not PIT evidence.
- The spec says missing fundamentals are unavailable/excluded, never zero-filled or median-filled.
- The spec preserves `DEV_FAILED`, holdout untouched, and no production authorization.

### WS3B: PIT Source Feasibility Gate

Delta:
- Separate source feasibility from fixture/prereg design.
- Evaluate candidate sources only against PIT evidence, license, committability, redistribution, fair-access limits, and regeneration path.
- EDGAR/XBRL can qualify only if the lane reconstructs values from exact accession/form/accepted records knowable before `as_of`; aggregate current company-facts alone do not qualify.
- If no source proves pre-`as_of` visibility, stop as `RESTATED_PROXY_ONLY`.
- Record the one-line decision in `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` under a required section named `## PIT Source Feasibility Record`.

Acceptance criteria:
- A committed one-line provenance record exists before any source-specific fixture or prereg design:
  `source`, `data_class`, `license`, `redistribution`, `as_of_mechanism`, `restatement_policy`, `fixture_committable`, `fair_access_limits`.
- `RESTATED_PROXY_ONLY` blocks fixture build, preregistration, dev run, and candidate scoring.
- No candidate score is interpreted until source feasibility passes.

### WS3C: Source-Specific Fixture and Prereg Design, Blocked on WS3B

Delta:
- Mark this work explicitly blocked until WS3B passes.
- After WS3B, design the fixture/adapter for the selected source only.
- Preserve existing price-only family equivalence under any new fixture/adapter path.
- Freeze denominator/share/price adjustment basis before any score interpretation.

Acceptance criteria:
- Leakage test fails if fundamentals appear before `available_asof`.
- Fixture SHA, candidate hash, and candidate validation hash are recorded before first dev run.
- Existing price-only candidate results are invariant under the adapter path.
- Candidate can run dev-only with `prereg_hash=None` and no reserved-tail reads.

### WS4: Reading B Candidate Bench Run

Delta:
- Keep validation report-only but operationalize the default stop: dev-pass/validation-fail does not burn holdout unless an explicit holdout-burn decision memo overrides.
- Keep result taxonomy precise: source-blocked, restated-proxy-only, power-limited negative, clean dev failure, dev-pass/validation-fail, dev-pass/validation-pass.

Acceptance criteria:
- `HOLDOUT_LEDGER.md` unchanged unless a verified, approved holdout run occurs.
- Any future holdout unlock uses only `candidate_run_hash(cfg, fixture)`.
- Validation report includes `CandidateValidationPreReg.declared_trials_N`; the vestigial `CandidatePreReg.declared_trials_N` is not used as live DSR control.

### WS5: Conditional Release Blocking

Delta:
- Keep deferred. Trigger only after a dev-passing candidate and operator approval to write a separate release-blocking validation plan.

Acceptance criteria:
- Before trigger, validation stays report-only.
- After trigger, tests prove validation failure blocks release prospectively.
- The current `DEV_FAILED` floor is not reinterpreted.

## Verification Plan

Planning verification:
- Confirm the closeout spec no longer says the strengthened Reading B schema already lives in the Reading B stub.
- Confirm Reading B edits are explicitly split into WS3A, WS3B, and WS3C with WS3C blocked on WS3B.
- Confirm all acceptance criteria preserve `DEV_FAILED`, holdout untouched, validation report-only, and no production sizing.

Implementation-lane verification after future execution:
- `node tools/run-floor.mjs --holdout` fails fast without holdout access.
- `node tools/run-floor.mjs --enforce` exits `1` while current floor remains `DEV_FAILED`.
- `npm run advisor-gate` remains report-only and reports `DEV_FAILED`.
- Docs truth test passes across tracked operator docs with archive allow-marker policy.
- Secret scan command/date/scope is recorded fresh with redacted output.
- Smoke helper records `provider_status` separately from `signal_status`; no secrets or key-bearing URLs appear.
- Reading B source feasibility record exists before fixture/prereg design.
- Holdout ledger remains unchanged unless a separate approved holdout lane appends it.

## ADR

Decision: tighten the current closeout plan in-place by making WS0 flag rejection mandatory, bounding WS2 to smoke-helper-first provider telemetry, and splitting WS3 into source-agnostic contract amendment, PIT source feasibility gate, and blocked source-specific fixture/prereg design.

Drivers:
- The Reading B spec is still a stub, so the closeout plan must stop claiming the strengthened schema already lives there.
- Provider smoke needs status taxonomy, not broad production-provider refactoring.
- Holdout and validation rails must remain stricter than roadmap momentum.

Alternatives considered:
- Rewrite Reading B before any other closeout work: rejected because stale public truth claims and `run-floor.mjs --holdout` ambiguity remain live operator risks.
- Execute keyed smoke first: rejected because it requires credentials and could produce misleading nonneutral output before provider-status semantics are defined.
- Wire validation into release blocking now: rejected because there is no dev-passing candidate and validation is explicitly report-only.

Why chosen:
- Option A preserves validated roadmap sequencing while correcting the defects most likely to mislead execution agents.
- It creates testable gates before credentials, source-specific fixture design, candidate measurement, or holdout decisions.

Consequences:
- The roadmap remains blocked for production.
- Reading B may stop at `RESTATED_PROXY_ONLY` without failure of the closeout effort.
- A later execution lane must still edit both closeout and Reading B specs; this plan does not implement those edits.

Follow-ups:
- After plan approval, run a narrow guard/docs/test execution lane for WS0, WS1, WS3A, and the cheap WS3B feasibility record before any keyed smoke.
- After WS3A/WS3B, run an adversarial leakage and holdout-discipline review before WS3C.
- After any dev-passing candidate, create a separate Plan 1b release-blocking validation spec.

## Agent Roster and Staffing Guidance

Available agent types:
- `explore`: repo-local lookup and line/file mapping.
- `writer`: spec/doc amendments and truth quarantine wording.
- `executor`: narrow implementation for WS0/WS1/helper/test edits after approval.
- `test-engineer`: regression tests, docs truth tests, smoke helper tests.
- `architect`: Reading B PIT contract, fixture boundary, denominator/share adjustment review.
- `security-reviewer`: secret scan procedure and redaction review.
- `critic`: holdout, validation, and promotion-semantics challenge review.
- `verifier`: final evidence collection and claim validation.

Recommended `ralph` path:
- Sequential owner with medium-high reasoning.
- Scope: WS0, WS1, then spec-only WS3A edits and the cheap WS3B feasibility record.
- Stop before keyed live smoke, external credentials, source-specific fixture work, candidate dev run, or holdout.

Suggested launch hint:
```text
$ralph execute .omx/plans/priority-dev-roadmap-closeout-tightened-ralplan-dr.md for WS0, WS1, WS3A, and the cheap WS3B feasibility record only. Planning rails apply: preserve DEV_FAILED, do not run holdout, do not expose secrets, and stop before keyed live smoke or source-specific fixture work.
```

Recommended `$team` path after WS0/WS1:
- `writer` (medium): closeout and Reading B spec amendments.
- `test-engineer` (medium): docs truth test, unknown-flag rejection test, smoke helper redaction/status tests.
- `architect` (high): WS3A/WS3B PIT and fixture-boundary review.
- `security-reviewer` (high): fresh secret scan and redaction protocol.
- `critic` (high): promotion, validation, and holdout rail challenge.
- `verifier` (high): final evidence bundle and acceptance checklist.

Suggested launch hint:
```text
$team execute .omx/plans/priority-dev-roadmap-closeout-tightened-ralplan-dr.md with lanes writer,test-engineer,architect,security-reviewer,critic,verifier. Do not run holdout, do not use credentials unless explicitly assigned to the smoke lane, and keep validation report-only.
```

Team verification path:
1. `writer` submits exact doc/spec diffs and confirms first-viewport truth coverage.
2. `test-engineer` submits targeted test output for docs truth, run-floor unknown flags, and smoke helper redaction/status.
3. `architect` signs off that WS3C remains blocked until WS3B PIT feasibility passes.
4. `security-reviewer` confirms fresh redacted secret scan evidence.
5. `critic` challenges holdout, validation, promotion, and source/PIT leakage claims.
6. `verifier` runs final accepted command set and reports evidence without claiming production readiness.

## Overlooked Improvement Opportunities

- Add a small glossary in the specs for `provider_status`, `signal_status`, `candidate_hash`, `candidate_run_hash`, `candidate_validation_hash`, and `RESTATED_PROXY_ONLY`.
- Require every future result note to include a "What this does not authorize" block.
- Add a docs truth fixture for first-viewport checks so `README.md` cannot hide current truth below an archive section.
- Add a no-secrets snapshot rule for smoke notes: redact URL query strings by default, not only known key names.
- Add a source-feasibility decision table to Reading B so failed providers are recorded once and not re-litigated.

## Consensus Changelog

- Applied architect recommendation to define archive-marker syntax and fresh secret-scan scope before execution.
- Applied critic iteration items: exact WS1 truth-scan scope, balanced archive marker rules, redacted secret-scan command/output contract, corrected WS0/WS1 handoff wording, and pinned WS3B provenance record location.
- Added execution-order refinement: WS0 -> WS1 -> WS3A -> cheap WS3B before credentialed smoke, so PIT infeasibility can stop early.
- Applied final critic approval cleanups: explicit denylist terms, deterministic `[]` secret-scan output on zero findings, and `rotation_required` yes/no semantics.
