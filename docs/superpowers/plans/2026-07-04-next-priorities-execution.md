# Next-Priorities Execution Plan (Slice 1 closeout + SESTM Phase-0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the repo closeout slice (docs reconciled+committed, stale diagnostics
branch resolved), then run the SESTM Phase-0 corpus capability matrix under a frozen
prereg, ending with the unconditional program review memo.

**Architecture:** Two strictly sequential slices (one in flight at a time). Slice 1 is
git/doc operations plus one agent-gated branch integration. Slice 2 is freeze-first
research: commit+hash the criteria file, then evaluate keyless corpora against it,
then write the memo regardless of verdict. Everything is report-only; no production
code paths change.

**Tech Stack:** git, npm gate scripts (`advisor-gate`, `advisor-release-gate`),
pytest via `tools/run-pytest.mjs`, existing EDGAR machinery
(`apps/quant/advisor/data/filing_text_fetch.py`, `edgar_xbrl_fetch.py`).

**Spec:** `docs/superpowers/specs/2026-07-04-next-priorities-design.md`

## Global Constraints

- Holdout stays blinded; `apps/quant/advisor/research/HOLDOUT_LEDGER.md` stays empty.
- Zero edits to frozen surfaces: `apps/quant/advisor/backtest/prereg.py`,
  `validation_prereg.py`, `PREREG.md`, `VALIDATION_PREREG.md`, and all
  `research/*prereg*.py` / `research/*PREREG*.md` files.
- Report-only: no changes to verdict logic or floor numbers.
- $0 data spend; keyless sources only.
- Gate invariants after every task: `npm run advisor-gate` exit 0;
  `npm run advisor-release-gate` exit 1 (floor still fails by design).
- Exactly one slice in flight; Slice 1 fully landed before Slice 2 starts.
- Any production code edit (none planned) goes via Hermes dispatch
  (`npm run hermes:production -- --task "..."`, short file-pointer form).
  Doc/git operations are executed directly per established project convention.
- Push safety: `git fetch origin` and confirm fast-forward before ANY push; if
  origin/main moved, stop and reconcile — never force-push main (incident on record).
- Windows: PowerShell-safe commands; no `rm -rf` / `mkdir -p` in shell steps.

---

### Task 1: Reconcile and commit the 4 untracked docs

**Files:**
- Modify: `docs/superpowers/plans/2026-06-18-ws3c-edgar-xbrl-fundamentals.md` (append addendum)
- Commit (as-is): `docs/superpowers/notes/2026-06-18-storm-orthogonal-indicator-scan.md`,
  `docs/superpowers/notes/2026-06-21-blend-futility-residual-alpha.md`,
  `docs/superpowers/plans/2026-06-22-universe-change-residual-screen.md`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: all four docs tracked on main; WS3C plan carries a truthful status addendum
  that Task 6's memo cites.

- [ ] **Step 1: Append the completion addendum to the WS3C plan**

Append verbatim to the END of
`docs/superpowers/plans/2026-06-18-ws3c-edgar-xbrl-fundamentals.md`:

```markdown

---

## Completion addendum (2026-07-04)

The task checklist above is a historical snapshot and is STALE. Actual outcome:
the fixture/adapter/prereg chain was completed and consumed by two dev runs, both
clean negatives — Reading B (fundamental_value+momentum, 2026-06-19, DEV_FAILED,
not power-limited; see `apps/quant/advisor/research/READING_B_RESULT.md`) and
Reading C (lazy_prices, 2026-06-21, DEV_FAILED 0/4 folds; see
`apps/quant/advisor/research/READING_C_RESULT.md`). Holdout was never touched.
This plan is CLOSED. Do not execute tasks from it.
```

- [ ] **Step 2: Verify docs-truth suite still passes**

Run: `node tools/run-pytest.mjs apps/quant/advisor/tests/test_docs_truth.py`
Expected: all tests pass (docs/superpowers is outside the operator-doc scan set).

- [ ] **Step 3: Commit the four docs**

```powershell
git add docs/superpowers/notes/2026-06-18-storm-orthogonal-indicator-scan.md docs/superpowers/notes/2026-06-21-blend-futility-residual-alpha.md docs/superpowers/plans/2026-06-18-ws3c-edgar-xbrl-fundamentals.md docs/superpowers/plans/2026-06-22-universe-change-residual-screen.md
git commit -m "Track WS3C/universe/STORM/blend-futility research docs with truthful status"
```

- [ ] **Step 4: Verify clean status for those paths**

Run: `git status --porcelain docs/superpowers`
Expected: no output for the four files (spec/plan files from this session may still show
until Task 7 commits them).

---

### Task 2: Resolve the stale diagnostics branch (rebase-or-reimplement)

**Files:**
- Possibly merge: branch `origin/claude/dancing-cucumber-integration-review-0qox2x`
  (commits `d3d8d0a`, `0e6e1da` — report-only floor diagnostics)
- Create (conflict path only): `docs/superpowers/notes/2026-07-04-dancing-cucumber-diagnostics-extraction.md`

**Interfaces:**
- Consumes: main after Task 1.
- Produces: either merged diagnostics on main, or an extraction note + archive tag;
  in both cases zero remaining unmerged claude/* branches.

- [ ] **Step 1: Attempt the rebase**

```powershell
git fetch origin
git checkout -b integrate/floor-diagnostics origin/claude/dancing-cucumber-integration-review-0qox2x
git rebase main
```

Expected: either "Successfully rebased" or conflict markers. The branch forked 47
commits behind main — treat conflicts as likely.

- [ ] **Step 2A (rebase CLEAN): integrity review, gates, merge**

1. Dispatch the `backtest-integrity-reviewer` agent on `git diff main...HEAD`.
   Acceptance: report-only — no changes to verdict logic, PREREG files, or floor
   numbers; no look-ahead introduced by the new diagnostics.
2. Run: `node tools/run-pytest.mjs apps/quant/advisor/tests` — Expected: all pass.
3. Run: `npm run advisor-gate` — Expected: exit 0.
   Run: `npm run advisor-release-gate` — Expected: exit 1.
4. Run: `git status --porcelain apps/quant/advisor/backtest/PREREG.md apps/quant/advisor/backtest/VALIDATION_PREREG.md` — Expected: empty.
5. Merge (fast-forward check before push, per Global Constraints):

```powershell
git checkout main
git merge --no-ff integrate/floor-diagnostics -m "Merge report-only floor diagnostics (Sortino/drawdown/concentration/minimax)"
git fetch origin
git rev-list --left-right --count main...origin/main
git push origin main
git push origin --delete claude/dancing-cucumber-integration-review-0qox2x
git branch -d integrate/floor-diagnostics
```

If the integrity review rejects the diff OR any gate fails after the rebase: do NOT
merge — run `git checkout main` and fall through to Step 2B (extract, archive, delete),
recording the rejection reason in the extraction note.

- [ ] **Step 2B (rebase CONFLICTS): abort, extract, archive, delete**

```powershell
git rebase --abort
git checkout main
git branch -D integrate/floor-diagnostics
```

Write `docs/superpowers/notes/2026-07-04-dancing-cucumber-diagnostics-extraction.md`:

```markdown
# Extraction note — dancing-cucumber floor diagnostics (2026-07-04)

Branch `claude/dancing-cucumber-integration-review-0qox2x` (d3d8d0a, 0e6e1da) carried
report-only advisor-gate diagnostics: Sortino, max drawdown, concentration checks,
minimax-robust family pick. It forked 47 commits behind main and did not rebase
cleanly. Intent preserved here; tip archived as tag
`archive/dancing-cucumber-diagnostics`. If these diagnostics are wanted, re-dispatch
as a fresh small task against current main (report-only acceptance criteria apply).
Decision on re-dispatch: deferred to the program review memo.
```

Then archive and delete:

```powershell
git tag archive/dancing-cucumber-diagnostics 0e6e1da
git push origin archive/dancing-cucumber-diagnostics
git add docs/superpowers/notes/2026-07-04-dancing-cucumber-diagnostics-extraction.md
git commit -m "Extract dancing-cucumber diagnostics intent; archive stale branch"
git push origin --delete claude/dancing-cucumber-integration-review-0qox2x
```

- [ ] **Step 3: Verify branch topology**

Run: `git branch -r`
Expected: no `origin/claude/*` branches remain; `origin/main` current.

---

### Task 3: Slice-1 gate verification

**Files:** none (verification only).

**Interfaces:**
- Consumes: main after Tasks 1-2.
- Produces: verified-green baseline that Slice 2 builds on.

- [ ] **Step 1: Full suite**

Run: `node tools/run-pytest.mjs apps/quant/advisor/tests`
Expected: all tests pass (295 baseline; more if Task 2 merged diagnostics with tests).

- [ ] **Step 2: Gates**

Run: `npm run advisor-gate` — Expected: exit 0.
Run: `npm run advisor-release-gate` — Expected: exit 1 (floor still DEV_FAILED by design).

- [ ] **Step 3: Frozen surfaces and ledger untouched**

Run: `git log --oneline -5 -- apps/quant/advisor/backtest/PREREG.md apps/quant/advisor/backtest/VALIDATION_PREREG.md apps/quant/advisor/research/HOLDOUT_LEDGER.md`
Expected: no new commits from this slice touching these files.

- [ ] **Step 4: Land Slice 1 (push)**

```powershell
git fetch origin
git rev-list --left-right --count main...origin/main
```

Expected: `N 0` (local ahead only). If the right-hand count is nonzero, origin moved —
stop and reconcile before pushing. Then:

Run: `git push origin main`
Run: `git rev-list --left-right --count main...origin/main` — Expected: `0 0`.
Slice 1 is landed only when this reads `0 0`.

---

### Task 4: Author and freeze the SESTM Phase-0 prereg

**Files:**
- Create: `docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md`

**Interfaces:**
- Consumes: verified baseline from Task 3.
- Produces: frozen criteria file + recorded SHA that Task 5 scores against and
  Task 6 cites. Freeze commit hash is recorded inside the Task 5 matrix note.

- [ ] **Step 1: Write the prereg file**

Create `docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md` with exactly this
content (numbers are frozen at commit time; if the operator wants different values,
change them BEFORE Step 3 — never after):

```markdown
# SESTM Phase-0 prereg — news-corpus capability matrix (FROZEN)

Parent plan: docs/superpowers/plans/2026-06-23-sestm-news-lane-plan.md
Frozen: 2026-07-04, BEFORE any source evaluation. Report-only. Non-alpha.
PASS unlocks nothing but a separately greenlit Phase-1 prereg. Default STOP.

## Must-have criteria (ALL required for a source to PASS)

1. **Cost:** $0 (keyless/free). Any registration-gated or paid tier = FAIL.
2. **Timestamps:** publication timestamp precise enough to assign each document to a
   trading date without look-ahead (time-of-day, or an explicit pre/post-close flag).
   Date-only granularity = FAIL unless publication is provably post-close.
3. **Ticker mappability:** >= 90% of corpus documents resolve entity/CIK -> ticker via
   SEC company_tickers.json (current snapshot). DISCLOSED LIMIT: this measures corpus
   mappability and is survivorship-biased; tradability-as-of-date is deferred to
   Phase-1 under the QC-project universe named in the parent plan.
4. **Breadth (corpus-intrinsic):** >= 2,000 distinct mappable tickers with median
   >= 4 documents per name per year across 2015-2023. DISCLOSED DIVERGENCE: this is
   deliberately NOT an index-membership test (no keyless PIT membership list exists —
   the T0.2b confound); coverage against the actual Phase-1 trading universe is
   re-measured in Phase-1.
5. **History:** continuous coverage 2015-2023 (mirrors existing fixture eras).
6. **Reproducibility:** stable immutable document identifiers (e.g., EDGAR accession
   numbers) or a documented versioning policy, so a frozen fixture with SHA can be
   built, matching the WS3C/WS3D fixture pattern.
7. **Corpus suitability:** third-party-authored news text about the company. Issuer
   self-disclosures (8-K bodies, issuer press releases) = FAIL for the SESTM-as-
   published hypothesis. If a self-disclosure corpus is the only survivor, the verdict
   is STOP for this lane; "self-disclosure tone" goes to the program review memo as a
   separately named NEW hypothesis, not a Phase-1 input.

## Decision rule (pre-committed)

- Any source meeting ALL seven -> Phase-0 PASS (name the source; PASS != alpha).
- No source meeting all seven -> STOP. No criterion may be relaxed post-freeze;
  a near-miss is a STOP (the v2 mappability precedent, 68.9% < 85%, binds).
- Budget cap: <= 1 session, $0 spend. Phase-0 is a desk-check-grade gate — its value
  is the frozen public record, not discovery; the real kill lives in Phase-1's
  net-OOS-sign gate. Cap exhaustion without a verdict -> STOP.

## Candidate source list (initial; additions are recorded in the MATRIX NOTE —
## this file never changes after the freeze commit)

EDGAR 8-K / material-event filings (via existing filing_text_fetch.py machinery and
the QC-diagnostic B1 dataset, 16,663 filings / 5,196 CIKs); EDGAR full-text search
(efts.sec.gov); GDELT event/news metadata; other keyless feeds discovered in-session
(list them in the matrix note under "Sources added during evaluation").
```

- [ ] **Step 2: Verify docs-truth suite passes with the new doc**

Run: `node tools/run-pytest.mjs apps/quant/advisor/tests/test_docs_truth.py`
Expected: all pass.

- [ ] **Step 3: Freeze commit + record SHA**

```powershell
git add docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md
git commit -m "Freeze SESTM Phase-0 corpus criteria before source evaluation"
git rev-parse HEAD
git hash-object docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md
```

Expected: commit lands; record BOTH hashes — they go verbatim into the Task 5 matrix
note header. No source evaluation before this commit exists.

---

### Task 5: Build the corpus capability matrix

**Files:**
- Create: `docs/superpowers/notes/2026-07-04-sestm-phase0-corpus-matrix.md`

**Interfaces:**
- Consumes: frozen criteria + hashes from Task 4; existing EDGAR machinery
  (`apps/quant/advisor/data/filing_text_fetch.py`, `edgar_xbrl_fetch.py`).
- Produces: scored matrix + verdict (PASS with named source, or STOP) that Task 6
  consumes verbatim.

- [ ] **Step 1: Evaluate each candidate against the six frozen criteria**

For each source in the candidate list: gather evidence (API docs, sample keyless
pulls — keyless EDGAR pulls agent-side are established practice from the B1 build).
Score each criterion PASS/FAIL with one line of evidence each. No criterion may be
reinterpreted; ambiguity resolves to FAIL (default STOP).

- [ ] **Step 2: Write the matrix note**

Create `docs/superpowers/notes/2026-07-04-sestm-phase0-corpus-matrix.md` with header:

```markdown
# SESTM Phase-0 corpus capability matrix (report-only)

Frozen criteria: docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md
Freeze commit: <git rev-parse HEAD from Task 4> | blob SHA: <git hash-object from Task 4>
Sessions used: <n of 1 cap>
```

Body: one table — rows = sources, columns = the six criteria + verdict; one evidence
line per cell; final section "Verdict" applying the pre-committed decision rule
verbatim. If the cap expires without a verdict, the verdict IS "STOP (cap exhausted)".

- [ ] **Step 3: Commit**

```powershell
git add docs/superpowers/notes/2026-07-04-sestm-phase0-corpus-matrix.md
git commit -m "SESTM Phase-0: corpus capability matrix and verdict"
```

- [ ] **Step 4: Verify the frozen criteria file did not change after freeze**

Run: `git log --oneline -- docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md`
Expected: exactly one commit (the Task 4 freeze). If more than one commit appears, the
freeze is broken: the Phase-0 run is procedurally invalid and the verdict is recorded
as "STOP (freeze violated)" — no re-run without a fresh prereg under a new filename.

---

### Task 6: Program review memo (unconditional)

**Files:**
- Create: `docs/superpowers/notes/2026-07-04-program-review-memo.md`

**Interfaces:**
- Consumes: Task 5 verdict; the recorded program history below.
- Produces: the strategic go/no-go record; on PASS it is the required input to any
  Phase-1 greenlight; on STOP no further signal lanes open without it.

- [ ] **Step 1: Write the memo**

Create `docs/superpowers/notes/2026-07-04-program-review-memo.md`. Required skeleton —
the eight-result record is fixed content (below); the Phase-0 row and §3-§4 are filled
from Task 5's verdict:

```markdown
# Program review memo — alpha-lane record and go-forward (2026-07-04)

## 1. The record (eight results)

| # | Lane | Verdict | Key number |
|---|------|---------|------------|
| 1 | Advisor v1 equal-weight price ensemble (2026-06-14) | FAILS FLOOR | Sharpe 0.32 < SPY 0.85 |
| 2 | Plan 4 v2 continuous long-flat ensemble (2026-06-16) | DEV_FAILED | ens 0.73 < best family 0.83 |
| 3 | Lane B value+momentum candidate (2026-06-17) | DEV_FAILED (power-limited) | ens 0.662 < best 0.668 |
| 4 | WS4 Reading B fundamental_value+momentum (2026-06-19) | DEV_FAILED (not power-limited) | ens 0.557 < momentum 0.665 < SPY 0.752 |
| 5 | WS3D Reading C lazy_prices (2026-06-21) | DEV_FAILED | 0/4 folds; ens 0.598 < momentum 0.681 |
| 6 | Universe-change broad screen T0.2b (2026-06-23) | INCONCLUSIVE | GREEN non-discriminating; survivorship-confounded |
| 7 | QC+EDGAR source-integrity diagnostic v2 (2026-06-23) | STOP | mappability 68.9% < 85% frozen gate |
| 8 | SESTM Phase-0 corpus matrix (2026-07-04) | <PASS source-named / STOP> | <from matrix note> |

Structural finding: blend non-additivity on the 30-mega-cap floor (Readings B and C
fail identically) and no positive long-only residual alpha on the floor (corrected
info-ratio screen: value -0.41, fundamental_value -0.32, lazy_prices -0.40,
momentum -0.02).

## 2. What the record rules out
<filled at write time: long-only price/fundamental/filing-text blends on the
30-mega-cap universe; market-neutral via free delisting-aware data>

## 3. Reframes evaluated
(a) validation-harness-as-asset; (b) risk/diagnostics advisor product;
(c) paid-data universe lever as a costed budget decision.
<one paragraph each: what it would take, what it would cost, what it would prove>

## 4. Recommendation and operator decision points
<filled at write time; if Phase-0 PASSED, this section is the required input to the
Phase-1 greenlight decision — a PASS without this memo is not a greenlight>
```

- [ ] **Step 2: Commit and verify final state**

```powershell
git add docs/superpowers/notes/2026-07-04-program-review-memo.md
git commit -m "Program review memo: eight-result record and go-forward options"
```

Run: `npm run advisor-gate` — Expected: exit 0.
Run: `npm run advisor-release-gate` — Expected: exit 1.
Run: `git status --porcelain apps/quant/advisor` — Expected: empty.

---

### Task 7: Verify planning artifacts are landed

The design spec and this plan were committed and pushed at plan-approval time (planning
session, 2026-07-04). This task only verifies.

- [ ] **Step 1: Verify**

Run: `git status --porcelain docs/superpowers`
Expected: empty (all research docs from this program tracked and clean).
Fallback: if the spec or this plan is somehow still untracked, commit and push them now
(`git add docs/superpowers/specs/2026-07-04-next-priorities-design.md docs/superpowers/plans/2026-07-04-next-priorities-execution.md` then commit+push with the fast-forward check).
Run: `git rev-list --left-right --count main...origin/main` — Expected: `0 0`.

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | CLEAR | HOLD SCOPE; 6 findings applied (unconditional memo, freeze mechanics, budget cap, reuse mandate, stale-branch rule, single-slice) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 5 findings applied (2A failure exit, prereg immutability contradiction, landed=pushed, freeze-violation consequence, push safety) |
| Outside Voice | adversarial subagent | Independent 2nd opinion | 1 | CLEAR | 8 findings triaged, all applied (corpus-intrinsic breadth, suitability criterion, mapping-file disclosure, trade-date timestamps, checkable reproducibility, Task-7 premise, 2A ff-check, 1-session cap) |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | no UI scope |

**UNRESOLVED:** 0
**VERDICT:** CEO + ENG + OUTSIDE VOICE CLEARED — ready to implement.
