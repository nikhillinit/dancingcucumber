# Execution Handoff — Priority Dev Roadmap Closeout (Hermes orchestration)

> **How to use this file:** Paste the **"FRESH-SESSION KICKOFF PROMPT"** block (bottom of this
> document) as your first message in a new Claude Code session opened at `C:\dev\AIHedgeFund`.
> Everything that prompt needs is in this file. This handoff is self-contained: do not assume any
> prior conversation context.

---

## 0. Mission (what this session implements)

Implement the **first execution slice** of the tightened closeout plan via **Hermes agent
orchestration**, then verify locally. The slice is, in strict order:

```
WS0  ->  WS1  ->  WS3A  ->  cheap WS3B
```

**In scope this session:** WS0 (gate guard), WS1 (truth quarantine + fresh hygiene evidence),
WS3A (Reading B source-agnostic contract), WS3B (PIT source feasibility *decision/record only*).

**OUT of scope this session (do NOT start):** WS2 keyed live smoke (needs operator credentials),
WS3C source-specific fixture/prereg (blocked on WS3B passing), WS4 candidate bench run, WS5 release
blocking, and **any `git push`**.

Success = a more truthful repo + a gated, honest next research lane, with **every rail intact**.
Nothing here promotes the advisor, touches holdout, or authorizes sizing.

---

## 1. Authoritative artifacts (read these first, in order)

1. **Tightened plan (execution authority):**
   `.omx/plans/priority-dev-roadmap-closeout-tightened-ralplan-dr.md`
2. **Refreshed context:**
   `.omx/context/priority-dev-roadmap-closeout-20260617T202941Z.md`
3. **Closeout spec (gates; just hardened, committed at `c0d505d`):**
   `docs/superpowers/specs/2026-06-17-priority-dev-roadmap-closeout.md`
4. **Reading B spec — STILL A STUB (WS3A/WS3B edit this):**
   `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md`
5. **Floor truth anchors (do not weaken):**
   `apps/quant/advisor/backtest/FLOOR_RESULT.md`, `apps/quant/advisor/backtest/VALIDATION_PREREG.md`,
   `apps/quant/advisor/research/HOLDOUT_LEDGER.md`, `apps/quant/advisor/research/CANDIDATE_RESULT.md`

The tightened plan is the **execution-order authority**. This handoff operationalizes it; if they
ever disagree, the tightened plan wins and you stop and report the discrepancy.

---

## 2. Current repo state (verified at handoff time)

- Branch **`main`** at commit **`c0d505d`** ("review: harden priority-dev-roadmap closeout spec").
  `main` is **ahead of `origin/main` by 2** (`f5a6406` + `c0d505d`) and is **NOT pushed**. Do not push.
- Branch `review/closeout-spec-hardening` exists and is identical to `main` (merged, fast-forward). Ignore it.
- `.claude/settings.local.json` is modified in the working tree — **leave it alone** (do not stage, do not revert).
- Heavy **untracked scratch** is present at repo root and elsewhere (`*.py`, `*.pdf`, `*.json`,
  `EVALUATION_RESPONSE.md`, `FIDELITY_TRADE_ORDERS.md`, `INSIDER_13F_INTELLIGENCE.md`,
  `OVERALL_RECOMMENDATION.md`, `Council of PMs/`, etc.). **Do not normalize, stage, move, or delete it.**
  Untracked files get *inventory*, never quarantine or deletion, in this slice.
- Test baseline expected GREEN before you start: `python -m pytest apps/quant/advisor/tests -q`.
  Record the pass count first; it must not regress.

---

## 3. Hard rails / invariants (NEVER violate — stop and report if a task would)

1. **Floor stays failed.** `node tools/run-floor.mjs --enforce` must still exit **1**.
   `npm run advisor-gate` stays report-only and prints `DEV_FAILED` (exit 0).
2. **No holdout, ever, this slice.** No `--holdout`, no reserved-tail read, no `HOLDOUT_LEDGER.md`
   change. The only legitimate holdout unlock is `candidate_run_hash(cfg, fixture)` — not in scope.
3. **No secrets** printed, persisted, or committed. Secret scan output is path + provider/variable
   class + `rotation_required` only — never a value, line, URL, or snippet.
4. **No new dependency.** No new pip/npm packages.
5. **No acceptance-bar changes.** Do not alter floor/validation thresholds.
6. **No unrelated-dirt cleanup.** Don't touch the untracked scratch or `.claude/settings.local.json`.
7. **No push.** Commit locally on a feature branch only; the operator pushes.
8. **No production/live-trading/sizing language** introduced anywhere.
9. **WS3C/WS4/WS5/WS2 stay closed.** WS3C is blocked until WS3B passes; WS2 needs operator keys.

If any Hermes task drifts into these, **revert that task's changes, do not commit, report it.**

---

## 4. Verified codebase facts (use these; do not re-derive)

| Fact | Anchor |
|---|---|
| `run-floor.mjs` forwards only `--enforce`; silently ignores any other flag | `tools/run-floor.mjs` |
| A live `--holdout` unlock path exists in the Python script (the footgun) | `tools/floor_data_check.py:69` |
| `run_pipeline(...) -> Decision` — per-family signals are internal, not returned | `apps/quant/advisor/pipeline/run.py:24` |
| yfinance fundamentals are RESTATED, not as-reported (so live `value_quality` is NOT PIT) | `apps/quant/advisor/data/provider.py:47` |
| `CandidatePreReg.declared_trials_N` is VESTIGIAL | `apps/quant/advisor/research/candidate_prereg.py:19` |
| LIVE DSR trial count is `CandidateValidationPreReg.declared_trials_N` | `apps/quant/advisor/research/candidate_validation_prereg.py:16` |
| Model for the new docs-truth test (retired-path denylist pattern) | `apps/quant/advisor/tests/test_repo_cleanup.py` |
| `python -m advisor ...` works (entrypoint exists) | `apps/quant/advisor/__main__.py` |
| FRED/AlphaVantage providers read env keys, degrade to empty on missing/throttle/error | `apps/quant/advisor/data/fred_provider.py`, `apps/quant/advisor/data/news_provider.py` |
| 8 tracked operator docs carry stale claims (confirmed via `git ls-files`) | see WS1 list below |
| Root scripts already env-only (secret debt remediated) — but **re-scan fresh**, don't assume | `fred_economic_analysis.py`, `alpha_vantage_enhanced_analysis.py` |

---

## 5. Workflow contract + Hermes dispatch mechanics (Windows-specific, load-bearing)

**Contract:** Claude Code does intake, planning, dispatch, and verification. **Every edit/test is
implemented by Hermes (Codex), not hand-written by Claude.** Claude verifies after each dispatch.

**Dispatch command:** `npm run hermes:production -- --task "<SHORT pointer>"`
(use `hermes:research` only for read-only analysis).

**Windows gotchas (from prior incidents — obey exactly):**
- **Keep the `--task` string SHORT.** Long task strings hit the Windows command-line length limit and
  the PowerShell dispatch guard rejects "slashy" strings. **Put the full task in a file**, dispatch a
  short pointer. (Slashes themselves are not the blocker — *length* + the guard are.)
- **Codex has NO `node`/`npm`.** It cannot run `npm test`, `npm run advisor-gate`, or
  `node tools/run-floor.mjs`. **Claude runs every node/npm verification** after the dispatch returns.
- **Codex cherry-picks / skips bulk work.** After each dispatch, **verify Codex's real git state**
  (`git status --short`, `git diff --stat`) before trusting its summary. If it skipped or deviated,
  re-dispatch narrower or apply that one edit directly.
- **Prefer solo** (Kimi is broken). One focused task per dispatch — do not bundle workstreams.

**Per-workstream dispatch recipe:**
```
1. Branch first:  git checkout -b exec/closeout-ws0-ws3b   (one branch for the whole slice)
2. Write the task file:  ai-logs/hermes/<ws>.md   (full scope, files, acceptance, the rails from §3)
3. Dispatch (SHORT, slash-free --task; constraints live in the task file, not the command):
       npm run hermes:production -- --task "do ws0 per ai-logs\hermes\ws0.md"
   (Long/forward-slashy --task strings fail on Windows. Use backslash paths or a bare filename; if the
   guard still rejects, shorten to "run ws0 task file" after pre-placing it. This SHORT file-pointer
   shape is the known-good pattern that carried prior multi-task dispatches.)
4. Verify real state:  git status --short ; git diff --stat
5. Verify behavior (Claude runs — Codex can't):  the §9 command set relevant to that WS.
6. If clean: stage only the intended files and commit (one commit per WS). If not: revert that WS, report.
```

---

## 6. WS0 — Baseline & Gate Guard

**Files:** `tools/run-floor.mjs` (patch) + a new test.

**Do:**
- Record the starting SHA (`c0d505d`) and `git status --short` into the WS0 commit message or a note.
- Patch `tools/run-floor.mjs` to accept **only** no-args or `--enforce`. Any other arg (especially
  `--holdout`) → print an explicit message ("holdout is not reachable through this wrapper; use a
  separate operator-approved lane") and **exit non-zero**. Keep the legitimate direct path
  `python tools/floor_data_check.py --holdout` **unchanged** (do not touch `floor_data_check.py`).
- Add a test that asserts: `run-floor.mjs --holdout` exits non-zero; `--enforce` still exits 1;
  bare invocation still exits 0. (Node test or a pytest that shells out — match existing test style;
  do not add a new test framework/dependency.)

**Acceptance (Claude verifies):**
- `node tools/run-floor.mjs --holdout` → non-zero, no holdout touched, `HOLDOUT_LEDGER.md` unchanged.
- `node tools/run-floor.mjs --enforce` → exit 1 (floor still `DEV_FAILED`).
- `npm run advisor-gate` → exit 0, reports `DEV_FAILED`.
- `python -m pytest apps/quant/advisor/tests -q` → no regression vs baseline.

---

## 7. WS1 — Truth Quarantine + Fresh Hygiene Evidence

**Required tracked-doc set (verified):** `README.md`, `START_HERE.md`, `SOPHISTICATION_ROADMAP.md`,
`COMPLETE_SYSTEM_STATUS.md`, `IMPLEMENTATION_GUIDE.md`, `OPTIMIZED_SYSTEM_SUMMARY.md`,
`OPTIONS_FLOW_SYSTEM_SUMMARY.md`, `bt_integration_summary.md` — **plus** every tracked `.md`/`.txt`
from `git ls-files '*.md' '*.txt'` that matches the denylist.

> **Highest cherry-pick risk dispatch.** WS1 bundles 8 doc rewrites + a new test + a secret scan + an
> inventory; Codex is documented to skip/cherry-pick bulk work. **Split it** (e.g. docs-batch-1,
> docs-batch-2, test, scan as separate dispatches) **or**, after a single dispatch, confirm
> `git diff --stat` shows **all eight** required doc paths changed BEFORE committing. Do not trust the
> Codex summary — verify the real diff.

**Do:**
- Give each required doc a **first-viewport truth header** (within the first lines, before any
  archive block): research-grade advisor; `DEV_FAILED`; production sizing / live trading / broker
  execution / paper-trading promotion NOT authorized; holdout untouched; report/dev-run only;
  `node tools/run-floor.mjs --enforce` exits 1 until the floor clears. Point to `FLOOR_RESULT.md`.
- Move obsolete claims into **balanced, non-nested archive blocks** using this exact marker syntax:

  ```text
  <!-- AIHF_TRUTH_ARCHIVE_START superseded_by="apps/quant/advisor/backtest/FLOOR_RESULT.md" reason="obsolete historical claim" -->
  ...obsolete historical content...
  <!-- AIHF_TRUTH_ARCHIVE_END -->
  ```

  Marker rules (the docs-truth test must enforce all of these): start/end markers balanced and
  non-nested; `superseded_by` must equal exactly `apps/quant/advisor/backtest/FLOOR_RESULT.md`;
  `reason` required and non-empty; archive markers are INVALID inside the first 40 physical lines of
  `README.md` or any required active-doc entrypoint; first-viewport current truth must appear before
  any archive block.
- Add `apps/quant/advisor/tests/test_docs_truth.py`, modeled on `test_repo_cleanup.py`. It must fail
  on: unqualified current denylist claims outside archive blocks; unmatched/nested markers; missing
  or wrong `superseded_by`; missing `reason`; archive markers inside the first 40 lines of required
  active docs. **Denylist terms (from the plan):** `production ready`, `live trading`, `go live`,
  `real money`, `paper trading operational`, `automated order execution`, `Fidelity` automation,
  `expected annual alpha`, `50-70%`, `50-60% annual`, `28-35% annually`, `92% accuracy`, `95%+`,
  `Sharpe >2.5`, `immediate deployment`.
- **Fresh redacted secret scan** — run this exact scan; output **only** `path`,
  `provider_or_variable_class`, `rotation_required`; print `[]` on zero findings; record
  command/date/scope alongside. Never print a value/line/URL/snippet.

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
  if ($unique.Count -eq 0) { "[]" } else { $unique | ConvertTo-Json }
  ```

  `rotation_required = yes` means a key-like literal was found and the operator must assume rotation is
  needed until manually disproven. Expected result is likely `[]` (root scripts are env-only) — but
  run it fresh, do not assume.
- **Inventory** untracked scratch separately via `git ls-files -o --exclude-standard` into a hygiene
  note classified keep/delete/ignore/secret-risk. **No mass-delete.**

**Acceptance (Claude verifies):**
- `python -m pytest apps/quant/advisor/tests/test_docs_truth.py -q` passes; `test_repo_cleanup.py`
  still passes; full suite no regression.
- Each required doc leads with the truth header; no denylist claim survives outside an archive block.
- Secret-scan note records command/date/scope + redacted findings (or `[]`).
- Gates unchanged: `advisor-gate` → `DEV_FAILED`; `run-floor --enforce` → exit 1.

---

## 8. WS3A + WS3B — Reading B contract + PIT feasibility (spec-only, the cheap slice)

Both edit **only** `docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md` (the stub).
**No fixtures, no code, no dev run, no holdout.** WS3C (source-specific fixture/prereg) stays BLOCKED.

**WS3A — source-agnostic contract (do first):**
- Turn the stub into a source-agnostic data contract. **Do not claim a source is chosen.** Add:
  schema fields — asset/CIK/accession/form/report-period-end/filing-date/accepted-datetime (or
  snapshot date)/concept/unit/value/available-as-of/superseded-by/amended-flag/missingness-reason/
  denominator-policy; the strict availability rule
  `available_asof = max(report_period_end + REPORTING_LAG_DAYS, filing_date, accepted_datetime, snapshot_date)`;
  price/share adjustment (denominator) bridge; and the required hashes (fixture SHA, candidate hash,
  candidate **validation** hash).
- State explicitly: **a strict lag on restated data is NOT point-in-time evidence**; missing
  fundamentals are `unavailable`/excluded — never zero/median/current/future-restatement filled.
- Pin the live trial-count surface: **`CandidateValidationPreReg.declared_trials_N`** (not the
  vestigial `CandidatePreReg.declared_trials_N`).
- Preserve `DEV_FAILED`, holdout untouched, no production authorization.

**WS3B — PIT source feasibility decision/record (cheap; can stop the lane early):**
- Evaluate candidate source(s) against: provable pre-`as_of` visibility, license, committability,
  redistribution, fair-access limits, regeneration path. Evaluate **SEC EDGAR/XBRL first**; it
  qualifies **only** if values can be reconstructed from the exact accession/form/accepted record
  knowable before `as_of` — aggregate *current* company-facts do **not** qualify.
- **Feasibility evidence to consult first (decide on evidence, not by defaulting to the safe stop):**
  the in-repo worked SEC examples at `stefan-jansen-ml/02_market_and_fundamental_data/04_sec_edgar/`;
  and the SEC API facts — company `submissions` and extracted XBRL company-facts are served by
  **keyless** JSON APIs, SEC filing content is free to reuse, subject to a declared User-Agent and a
  ~10 requests/second ceiling. The open question to answer: can the exact accession/form/accepted
  record for book value + shares be selected as-of the evaluation date? If yes → EDGAR qualifies; if
  unprovable → `RESTATED_PROXY_ONLY`.
- If no source proves pre-`as_of` visibility → stop as **`RESTATED_PROXY_ONLY`** (blocks fixture,
  prereg, dev run, candidate scoring).
- Record the decision as a one-line provenance record under a new required section
  **`## PIT Source Feasibility Record`** in the Reading B spec, with fields:
  `source`, `data_class`, `license`, `redistribution`, `as_of_mechanism`, `restatement_policy`,
  `fixture_committable`, `fair_access_limits`.

**Acceptance (Claude verifies — these are spec edits, so verification is review + suite-green):**
- Reading B spec is no longer a stub: contains the source-agnostic schema/availability contract and a
  `## PIT Source Feasibility Record` section.
- Spec says strict-lag-on-restated ≠ PIT, and missing → excluded.
- `CandidateValidationPreReg.declared_trials_N` named as the live surface.
- No source-specific fixture/prereg work appears (WS3C still blocked).
- Full suite still green; gates unchanged.

---

## 9. Verification command set (Claude runs after each dispatch; Codex cannot)

```powershell
$env:PYTHONPATH = "apps/quant"                        # required for advisor imports and `python -m advisor`
python -m pytest apps/quant/advisor/tests -q          # full suite, no regression
python -m pytest apps/quant/advisor/tests/test_docs_truth.py -q   # after WS1
npm run advisor-gate                                  # exit 0, prints DEV_FAILED
node tools/run-floor.mjs --enforce                    # exit 1 (floor failed)
node tools/run-floor.mjs --holdout                    # after WS0: must FAIL FAST (non-zero), no holdout
git status --short                                    # confirm only intended files changed
git diff --stat                                       # confirm Codex actually applied the edits
git diff -- apps/quant/advisor/research/HOLDOUT_LEDGER.md   # MUST be empty
```

Expected end state: suite green; `advisor-gate` → `DEV_FAILED`; `run-floor --enforce` → exit 1;
`run-floor --holdout` → fail-fast; `HOLDOUT_LEDGER.md` untouched; only the intended files changed.

---

## 10. Definition of done (this slice) + what to hand back

- One feature branch (e.g. `exec/closeout-ws0-ws3b`), **four focused commits** (WS0, WS1, WS3A, WS3B),
  **not pushed**.
- All §9 checks green; all §3 rails intact; `HOLDOUT_LEDGER.md` unchanged; no secrets; untracked
  scratch and `.claude/settings.local.json` untouched.
- A short closeout note listing: commits, the secret-scan result (or `[]`), the WS3B feasibility
  decision (source chosen **or** `RESTATED_PROXY_ONLY`), and the explicit statement that nothing was
  promoted, no holdout was run, and WS2/WS3C/WS4/WS5 remain unstarted.
- Ask the operator before any merge to `main` or push.

---

## 11. Stop-and-report triggers

Stop immediately and report (do not "work around") if: a task would touch holdout/reserved tail;
print/persist/commit a key; imply production/sizing is authorized; require a new dependency; need to
mass-delete untracked files; change an acceptance bar; or if WS3B yields `RESTATED_PROXY_ONLY` (that
is a clean, expected stop — report it, do not force a source). Also stop if Codex repeatedly skips a
task after two narrowed re-dispatches; surface it for an operator decision.

---

## FRESH-SESSION KICKOFF PROMPT (paste this into the new session)

```
You are Linus Torvalds operating under the repo's workflow contract: you plan, dispatch, and verify;
Hermes (Codex) implements every edit/test. Working dir C:\dev\AIHedgeFund, Windows/PowerShell.

Read docs/superpowers/plans/2026-06-17-closeout-execution-handoff.md in full, then implement its
WS0 -> WS1 -> WS3A -> cheap WS3B slice via Hermes, exactly as specified, with these non-negotiables:

- Preserve all rails: node tools/run-floor.mjs --enforce stays exit 1; npm run advisor-gate stays
  report-only DEV_FAILED; HOLDOUT_LEDGER.md unchanged; no holdout, no secrets, no new deps, no push.
- Do NOT touch the untracked scratch or .claude/settings.local.json.
- Dispatch one focused Hermes task per workstream via a SHORT, slash-free --task that points to a task
  file under ai-logs\hermes (long/slashy strings fail on Windows; put all detail in the file). WS1 is
  the highest skip-risk dispatch (8 docs + test + scan) — split it or confirm all 8 docs changed before
  committing. Codex has no node/npm, so YOU run every pytest/advisor-gate/run-floor check (after
  $env:PYTHONPATH='apps/quant') and confirm Codex's real git state (git status/diff) before trusting
  its summary.
- WS3A/WS3B edit only the Reading B spec stub (docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md);
  do not claim a source is chosen, and if no point-in-time source qualifies, stop as RESTATED_PROXY_ONLY.
- Keep WS2 (credentialed smoke), WS3C, WS4, WS5 unstarted.

Branch first (exec/closeout-ws0-ws3b), one commit per workstream, do not push. When the slice is done
and all checks are green, hand back a closeout note (commits, secret-scan result, WS3B decision) and
ask before merging to main.

Start by confirming branch/SHA and a green pytest baseline, then proceed WS0 first.
```
