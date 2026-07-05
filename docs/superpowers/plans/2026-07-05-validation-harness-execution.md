# Validation-Harness Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.
> **Repo override:** code/test edits are dispatched via Hermes to Codex (workflow
> contract); docs are written directly. Task 6 is the Hermes dispatch.

**Status:** DEBATE-REFINED — Hermes debate workflow: Claude comparator in-session,
Codex comparator via Hermes (run hermes-2026-07-05T12-27-27-299Z, read-only verified),
Kimi comparator SKIPPED (CLI broken, disclosed), Claude synthesis in-session.
9 merged findings applied, incl. Codex BLOCKER (chain tail-deletion evasion → tip
footer) and one cross-model agreement (verbatim-vs-restructure task-file bug).
**Spec:** `docs/superpowers/specs/2026-07-05-validation-harness-asset-design.md` (REVIEWED, `264a7f9`)

**Goal:** Package the prereg/floor/holdout governance machinery as the program's
primary internal artifact: playbook + program record (hash-chained) + 3 templates +
1 conformance test file.

**Architecture:** Six new docs under `docs/superpowers/harness/` (Lane A, direct);
one new pytest file `apps/quant/advisor/tests/test_prereg_conformance.py` (Lane B,
Hermes→Codex). No existing file is modified. No frozen surface is touched.

**Tech Stack:** Markdown, pytest, `subprocess` + git plumbing (`git rev-parse
<commit>:<path>`), `hashlib.sha256`.

---

## Preflight (before Task 1)

- [ ] `git fetch origin && git rev-list --left-right --count main...origin/main` → expect `0 0`
- [ ] `git checkout -b exec/validation-harness-package`
- [ ] `npm run advisor-gate` → exit 0; `apps/quant/advisor/research/HOLDOUT_LEDGER.md` still empty

---

### Task 1: Harness README (Lane A)

**Files:**
- Create: `docs/superpowers/harness/README.md`

- [ ] **Step 1: Write the file** with exactly this content:

```markdown
# Research-Governance Harness

The falsification machinery of this program, packaged as the primary artifact.
Consumer: future-self / this program. It exists because the eight-negative record
(`PROGRAM_RECORD.md`) showed the durable value is the discipline, not any signal.

## Contents
- `LANE_LIFECYCLE.md` — the operating loop, stage by stage, with canonical exemplars
- `PROGRAM_RECORD.md` — living append-only verdict table (hash-chained)
- `templates/prereg-template.md` — copy to start a new lane's prereg
- `templates/source-matrix-template.md` — copy for a Phase-0-style source evaluation
- `templates/closeout-template.md` — copy to close a lane

## Start a new lane in 5 steps
1. Get the operator election (a memo-based decision — see `LANE_LIFECYCLE.md` §1).
2. Copy `templates/prereg-template.md` to `docs/superpowers/plans/YYYY-MM-DD-<lane>-prereg.md`
   and fill every section. The conformance test enforces the anchors.
3. Commit the prereg, record the freeze commit + blob SHA
   (`git rev-parse HEAD` / `git rev-parse HEAD:docs/superpowers/plans/<file>`).
4. Run the lane on a separate surface (new files only; frozen surfaces untouched).
5. Close out from `templates/closeout-template.md`; append one row to
   `PROGRAM_RECORD.md` in the same commit (paste the chain hash pytest prints).

Enforcement floor: `apps/quant/advisor/tests/test_prereg_conformance.py`.
What it does NOT enforce is listed in `LANE_LIFECYCLE.md` §"Not machine-checked".
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/harness/README.md
git commit -m "harness: add package README and quickstart"
```

---

### Task 2: Lane lifecycle playbook (Lane A)

**Files:**
- Create: `docs/superpowers/harness/LANE_LIFECYCLE.md`

- [ ] **Step 1: Write the file.** Content = the ten stages from spec §3.2, each with
its canonical exemplar cited by repo path, PLUS the consolidated rules below (this is
the full rule list to transcribe — the repo-self-containment payload):

```markdown
# Lane Lifecycle — the operating loop

Every research lane moves through these stages. Each cites its canonical exemplar.
Rules here were paid for with eight negative verdicts; do not relearn them.

## 1. Operator election
No lane opens without a memo-based operator decision.
Exemplar: `docs/superpowers/notes/2026-07-04-program-review-memo.md` §4 +
`docs/superpowers/specs/2026-07-04-next-priorities-design.md` §7.

## 2. Review pipeline
brainstorm → spec → CEO review → eng review → implementation plan (debate-refined
when stakes warrant). Exemplar: the 2026-07-04 and 2026-07-05 spec/plan pairs.

## 3. Prereg + freeze protocol
Copy `templates/prereg-template.md` → `docs/superpowers/plans/YYYY-MM-DD-<lane>-prereg.md`.
Freeze BEFORE any evaluation: commit the prereg, then record in the evaluation
artifact the canonical citation
`Frozen criteria: <path> | Freeze commit: <40-hex> | blob SHA: <40-hex>`.
The conformance test git-verifies every such citation.
Exemplar: `docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md` (frozen at
`d420af7` BEFORE source evaluation — the v3 goalpost-move lesson made this mandatory).

## 4. Fixtures
Data enters as SHA-pinned bytes; record the fixture SHA in the result doc.
Exemplar: WS3D filing-text fixture (SHA 372a8518…, coverage 0.975).

## 5. Separate-surface rule
Never modify a frozen surface. New files/classes only. NEVER add fields to a frozen
prereg (its hash is pinned). Any divergence from byte-identical-to-floor is DISCLOSED
in the result doc. Exemplar: Lane B's three disclosed divergences
(`apps/quant/advisor/research/READING_B_RESULT.md` lineage).

## 6. Execution via Hermes
Code edits are dispatched: `npm run hermes:production -- --task "<short string>"`
from the Bash tool (not PowerShell), short slash-free task string pointing at a
repo-root task file. VERIFY Codex's real git state afterwards (`git status`,
`git log --oneline -3`) — Codex has skipped bulk operations before.

## 7. Verdict semantics
- Default STOP. A near-miss is a STOP (precedent: mappability 68.9% < 85%, v2).
- Decide off `dev.passed`, not the verdict enum, when the holdout is blinded
  (the enum reads DEV_FAILED by construction under blinding).
- "Power-limited" is an annotation, never an excuse to rerun with relaxations.
- Budget-cap exhaustion is itself a STOP, never a reason to extend.
- Post-freeze goalpost-moves are NEW programs under NEW filenames, not continuations
  (precedent: v3 Form-25 reclassification, demoted).

## 8. Holdout governance
Holdout stays blinded. EVERY touch appends a row to
`apps/quant/advisor/research/HOLDOUT_LEDGER.md`. Unlock only via verified
`candidate_run_hash(cfg, fixture)`. A side-bench touch burns the shared tail:
promotion then requires a FRESH holdout.

## 9. Closeout + record append
Copy `templates/closeout-template.md`. Append exactly one row to `PROGRAM_RECORD.md`
in the SAME commit (run pytest; paste the printed chain hash).
Exemplar: `docs/superpowers/notes/2026-06-23-v2-STOP-closeout.md`.

## 10. End-of-slice verification
Full suite green; `npm run advisor-gate` exit 0; `npm run advisor-release-gate` exit
code MATCHES the recorded floor state (currently 1 = DEV_FAILED, the honest state; a
change in this exit code is a headline event requiring an operator decision, never a
checklist pass); `git diff` empty on frozen files; ledger unchanged; docs-truth green.

## Not machine-checked
The conformance test enforces artifact discipline only (anchors, citations, record
chain, template drift, prereg location). It does NOT check: operator election, review
pipeline execution, separate-surface discipline, goalpost-moves, holdout ledger
semantics (enforced by the unlock-hash code path), or closeout quality. Those remain
review-enforced. One green test is never governance.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/harness/LANE_LIFECYCLE.md
git commit -m "harness: add lane lifecycle playbook (rules transcribed from record)"
```

---

### Task 3: Program record with hash chain (Lane A)

**Files:**
- Create: `docs/superpowers/harness/PROGRAM_RECORD.md`

- [ ] **Step 1: Compute the seed chain hashes.** Run this snippet (also used by the
test — keep the algorithm identical):

```python
import hashlib
PTR = "notes/2026-07-04-program-review-memo.md"
rows = [
    f"1|Advisor v1 equal-weight price ensemble (2026-06-14)|FAILS FLOOR|Sharpe 0.32 < SPY 0.85|{PTR}",
    f"2|Plan 4 v2 continuous long-flat ensemble (2026-06-16)|DEV_FAILED|ens 0.73 < best family 0.83|{PTR}",
    f"3|Lane B value+momentum candidate (2026-06-17)|DEV_FAILED (power-limited)|ens 0.662 < best 0.668|{PTR}",
    f"4|WS4 Reading B fundamental_value+momentum (2026-06-19)|DEV_FAILED (not power-limited)|ens 0.557 < momentum 0.665 < SPY 0.752|{PTR}",
    f"5|WS3D Reading C lazy_prices (2026-06-21)|DEV_FAILED|0/4 folds; ens 0.598 < momentum 0.681|{PTR}",
    f"6|Universe-change broad screen T0.2b (2026-06-23)|INCONCLUSIVE|GREEN non-discriminating; survivorship-confounded|{PTR}",
    f"7|QC+EDGAR source-integrity diagnostic v2 (2026-06-23)|STOP|mappability 68.9% < 85% frozen gate|{PTR}",
    f"8|SESTM Phase-0 corpus matrix (2026-07-04)|STOP|no keyless corpus clears the 7 frozen criteria|{PTR}",
]
chain = "GENESIS"
for r in rows:
    chain = hashlib.sha256(f"{chain}|{r}".encode()).hexdigest()[:12]
    print(chain)
print(f"Chain tip: {chain} over {len(rows)} rows")
```

The pointer cell is the repo path `notes/2026-07-04-program-review-memo.md` in BOTH
the snippet and the table — they must byte-agree or the chain breaks at Task 7.

- [ ] **Step 2: Write the file** — header, append rule, and the memo §1 table with two
extra columns (`Record pointer` = `notes/2026-07-04-program-review-memo.md` for all
eight seed rows; `Chain` = the hash from Step 1, row by row):

```markdown
# Program Record — append-only verdict table

One row per lane closeout, appended in the closeout commit. NEVER edit, reorder, or
delete a row — every row carries a chain hash over all prior rows, and the footer
pins the chain tip + row count; both are recomputed by `test_prereg_conformance.py`.
Any edit/reorder/interior-deletion fails the chain; tail deletion fails the tip
footer. (A deliberate rewrite of rows AND footer together defeats this — the guard
targets sloppy edits, the documented failure mode; deliberate tampering is a git-
history matter.) To append: write the row with chain cell `TBD`, run pytest, paste
the printed expected hash, update the tip footer the same way. Exactly ONE table
lives in this file — the test enforces that.
Seeded 2026-07-05 from `notes/2026-07-04-program-review-memo.md` §1; historic rows
point at the memo (not all eight have a dedicated closeout — stating that beats
faking it).

| # | Lane | Verdict | Key number | Record pointer | Chain |
|---|------|---------|-----------|----------------|-------|
| 1 | Advisor v1 equal-weight price ensemble (2026-06-14) | FAILS FLOOR | Sharpe 0.32 < SPY 0.85 | notes/2026-07-04-program-review-memo.md | <hash1> |
| 2 | Plan 4 v2 continuous long-flat ensemble (2026-06-16) | DEV_FAILED | ens 0.73 < best family 0.83 | notes/2026-07-04-program-review-memo.md | <hash2> |
| 3 | Lane B value+momentum candidate (2026-06-17) | DEV_FAILED (power-limited) | ens 0.662 < best 0.668 | notes/2026-07-04-program-review-memo.md | <hash3> |
| 4 | WS4 Reading B fundamental_value+momentum (2026-06-19) | DEV_FAILED (not power-limited) | ens 0.557 < momentum 0.665 < SPY 0.752 | notes/2026-07-04-program-review-memo.md | <hash4> |
| 5 | WS3D Reading C lazy_prices (2026-06-21) | DEV_FAILED | 0/4 folds; ens 0.598 < momentum 0.681 | notes/2026-07-04-program-review-memo.md | <hash5> |
| 6 | Universe-change broad screen T0.2b (2026-06-23) | INCONCLUSIVE | GREEN non-discriminating; survivorship-confounded | notes/2026-07-04-program-review-memo.md | <hash6> |
| 7 | QC+EDGAR source-integrity diagnostic v2 (2026-06-23) | STOP | mappability 68.9% < 85% frozen gate | notes/2026-07-04-program-review-memo.md | <hash7> |
| 8 | SESTM Phase-0 corpus matrix (2026-07-04) | STOP | no keyless corpus clears the 7 frozen criteria | notes/2026-07-04-program-review-memo.md | <hash8> |

Chain tip: <tip-hash> over 8 rows
```

(`<hashN>`/`<tip-hash>` are the Step-1 outputs — paste them; they are generated
values, not placeholders left to judgment. The tip footer is what catches TAIL
deletion — a shorter chain is still internally valid, so the tip pins length + head.
Deliberately rewriting the footer alongside a deletion is outside the threat model
(sloppy edits), which the file header states honestly.)

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/harness/PROGRAM_RECORD.md
git commit -m "harness: seed hash-chained program record (8 verdicts from memo)"
```

---

### Task 4: Templates (Lane A)

**Files:**
- Create: `docs/superpowers/harness/templates/prereg-template.md`
- Create: `docs/superpowers/harness/templates/source-matrix-template.md`
- Create: `docs/superpowers/harness/templates/closeout-template.md`

- [ ] **Step 1: Write `prereg-template.md`:**

```markdown
# <Lane name> Prereg — <one-line hypothesis>

Copy to `docs/superpowers/plans/YYYY-MM-DD-<lane>-prereg.md`. Fill EVERY section
before the freeze commit. The conformance test requires the anchors below.

## Hypothesis
<what would have to be true; what this lane is NOT (name adjacent hypotheses it
does not test)>

## Universe / data
<universe, fixture source, fixture SHA once built, as-of discipline>

## Frozen criteria / thresholds
<numbered, machine-checkable where possible; near-miss = STOP>

## Pre-committed decision rule
> <exact PASS condition> → PASS (PASS ≠ alpha claim).
> Anything else → STOP. No criterion may be relaxed post-freeze; a near-miss is a
> STOP. Cap exhaustion without a verdict → STOP.

## Budget cap
<sessions and dollars; exhaustion = STOP, never extension>

## Holdout statement
Holdout blinded throughout; every touch appends to HOLDOUT_LEDGER.md; unlock only
via verified candidate_run_hash(cfg, fixture).

## Freeze protocol footer
After committing this file, record in the evaluation artifact (format below is
fenced so instructional copies never trip the citation scan; the REAL citation in
your matrix/closeout goes at column 0, unfenced, with real values):

    Frozen criteria: <repo-relative-path> | Freeze commit: <40-hex> | blob SHA: <40-hex>
```

- [ ] **Step 2: Write `source-matrix-template.md`:**

```markdown
# <Lane> source capability matrix (report-only)

Frozen criteria: <repo-relative-path> | Freeze commit: <40-hex> | blob SHA: <40-hex>
Sessions used: <n> of <cap>

| Source | <criterion 1> | <criterion 2> | ... | Verdict |
| --- | --- | --- | --- | --- |
| <source> | PASS/FAIL: <evidence with citation> | ... | ... | PASS/STOP: <binding failures> |

## Verdict
Pre-committed decision rule (quoted verbatim from the frozen prereg):
> <paste>

<PASS/STOP>. <binding failures, each structural/measurement labelled>
```

- [ ] **Step 3: Write `closeout-template.md`:**

```markdown
# <Lane> closeout — <VERDICT>

Frozen criteria: <repo-relative-path> | Freeze commit: <40-hex> | blob SHA: <40-hex>

## Verdict
<STOP/PASS/DEV_FAILED + the one number that decided it>

## Binding failures
<numbered; label each structural vs measurement>

## What the record now rules out
<carefully scoped — only what this result actually excludes>

## What is explicitly NOT ruled out
<adjacent hypotheses that remain open and would need their own election + prereg>

## Program record row
<the row appended to PROGRAM_RECORD.md, verbatim, including chain hash>

## Next decision pointer
<which memo/decision point governs what may happen next>
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/harness/templates/
git commit -m "harness: add prereg, source-matrix, and closeout templates"
```

---

### Task 5: Verify Lane A locally

- [ ] Run: `python -m pytest apps/quant/advisor/tests -q --tb=short` → expect 322 passed + 1 xfailed (unchanged — no test exists yet for the new docs)
- [ ] Run: `npm run advisor-gate` → exit 0
- [ ] `git status --porcelain` → only the four harness commits, no stray files

---

### Task 6: Conformance test via Hermes (Lane B)

**Files:**
- Create: `TASK_HARNESS_TEST.md` (repo root, temporary — the Codex task file)
- Create (by Codex): `apps/quant/advisor/tests/test_prereg_conformance.py`

- [ ] **Step 1: Write `TASK_HARNESS_TEST.md`** containing the COMPLETE test file below
plus these instructions verbatim: "You must be on branch
`exec/validation-harness-package` — verify with `git branch --show-current` and STOP
if it differs. Create apps/quant/advisor/tests/test_prereg_conformance.py with
exactly this content, then run `python -m pytest apps/quant/advisor/tests/test_prereg_conformance.py -v`
and fix only mechanical errors (imports, paths); do not weaken any assertion. Stage
ONLY that one file: `git add apps/quant/advisor/tests/test_prereg_conformance.py`
(never `git add .` — the repo root contains an untracked task file that must NOT be
committed). Commit with message 'harness: add prereg conformance test'."

```python
"""Conformance floor for research-governance artifacts (spec 2026-07-05).

Machine-checks artifact discipline ONLY (see LANE_LIFECYCLE.md "Not machine-checked"):
  1. anchors on future preregs in the canonical home
  2. prereg location enforcement (repo-wide)
  3. canonical freeze citations, git-verified (rev-parse <commit>:<path> == blob SHA)
  4. PROGRAM_RECORD row hash chain (append-only)
  5. template self-check
Helpers are module-level and unit-tested against synthetic inputs (negative tests).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
DOCS = REPO_ROOT / "docs" / "superpowers"
HARNESS = DOCS / "harness"
CANONICAL_PREREG_HOME = DOCS / "plans"

ANCHOR_PATTERNS = {
    "decision rule": re.compile(r"decision rule", re.IGNORECASE),
    "budget cap": re.compile(r"budget", re.IGNORECASE),
    "STOP condition": re.compile(r"\bSTOP\b"),
}

# Predate the anchor phrasing; verifiably lack it. Never grows.
GRANDFATHERED_PREREGS = {
    "2026-06-23-qc-edgar-diagnostic-v2-prereg.md",
    "2026-06-23-qc-source-integrity-diagnostic-prereg.md",
}

# Frozen surfaces governed by their own hash pins. Never grows without an
# operator decision recorded in the closeout that adds the entry.
LEGACY_PREREG_PATHS = {
    "apps/quant/advisor/backtest/PREREG.md",
    "apps/quant/advisor/backtest/VALIDATION_PREREG.md",
    "apps/quant/advisor/research/READING_B_PREREG.md",
    "apps/quant/advisor/research/READING_C_PREREG.md",
    "apps/quant/advisor/research/CANDIDATE_PREREG.md",
}

EXCLUDED_DIRS = {
    ".git", "node_modules", "stefan-jansen-ml", ".omx", ".claude", ".remember",
    ".venv", "venv",
}

CITATION_RE = re.compile(
    r"^Frozen criteria: (?P<path>\S+) \| Freeze commit: (?P<commit>[0-9a-f]{40})"
    r" \| blob SHA: (?P<blob>[0-9a-f]{40})\s*$"
)

# The SESTM Phase-0 matrix predates the one-line canonical format (two-line
# citation); its freeze claim is pinned here and verified like any other.
PINNED_LEGACY_CITATION = (
    "docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md",
    "d420af72518096a332ac85ee70e4fd104db7e972",
    "a42cc3dd643477724bcb8a8ab34c2eea0464198d",
)


def missing_anchors(text: str) -> list[str]:
    return [name for name, pat in ANCHOR_PATTERNS.items() if not pat.search(text)]


def find_misplaced_preregs(root: Path) -> list[Path]:
    # Allowed homes derive from ROOT (not module constants) so the tmp_path
    # negative/positive tests exercise the identical code path used live.
    canonical = (root / "docs" / "superpowers" / "plans").resolve()
    templates = (root / "docs" / "superpowers" / "harness" / "templates").resolve()
    hits = []
    for p in root.rglob("*.md"):
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        if "prereg" not in p.name.lower():
            continue
        if p.parent.resolve() in (canonical, templates):
            continue
        if p.relative_to(root).as_posix() in LEGACY_PREREG_PATHS:
            continue
        hits.append(p)
    return hits


def citation_candidates(text: str) -> list[tuple[int, str]]:
    # Skip fenced code blocks and indented literals: docs (including THIS plan and
    # the templates' instructional footers) may QUOTE the citation format; only
    # column-0 unfenced lines are real freeze claims.
    out, fenced = [], False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.startswith(("    ", "\t")):
            continue
        if line.startswith("Frozen criteria:") and "Freeze commit:" in line:
            out.append((i, line))
    return out


def git_blob_sha(commit: str, path: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"freeze citation unverifiable: git rev-parse {commit}:{path} -> "
            f"{proc.stderr.strip()} (unreachable commit/path is REAL corruption, "
            "never skipped)"
        )
    return proc.stdout.strip()


TIP_RE = re.compile(r"^Chain tip: (?P<tip>[0-9a-f]{12}) over (?P<count>\d+) rows\s*$")


def record_table(text: str) -> tuple[list[list[str]], str | None, int | None, int]:
    """Header-anchored parse: only rows of THE schema table count (a future second
    table or a stray pipe in prose must not join the chain). Returns
    (rows, tip, count, header_count)."""
    rows, tip, count, headers, in_table = [], None, None, 0, False
    for line in text.splitlines():
        if line.startswith("| #"):
            headers += 1
            in_table = True
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and set(cells[0]) <= {"-", " ", ":"}:
                continue
            rows.append(cells)
            continue
        in_table = False
        m = TIP_RE.match(line)
        if m:
            tip, count = m["tip"], int(m["count"])
    return rows, tip, count, headers


def expected_chain(rows: list[list[str]]) -> list[str]:
    chain, out = "GENESIS", []
    for cells in rows:
        row_text = "|".join(cells[:-1])
        chain = hashlib.sha256(f"{chain}|{row_text}".encode()).hexdigest()[:12]
        out.append(chain)
    return out


# ---------- live checks against the real tree ----------

def test_grandfather_files_exist():
    for name in GRANDFATHERED_PREREGS:
        assert (CANONICAL_PREREG_HOME / name).is_file(), f"grandfather pin rotted: {name}"
    assert (CANONICAL_PREREG_HOME / "2026-07-04-sestm-phase0-prereg.md").is_file()


def test_future_preregs_carry_anchors():
    for p in sorted(CANONICAL_PREREG_HOME.glob("*-prereg.md")):
        if p.name in GRANDFATHERED_PREREGS:
            continue
        missing = missing_anchors(p.read_text(encoding="utf-8"))
        assert not missing, f"{p.name} missing anchors: {missing}"


def test_no_misplaced_preregs():
    hits = find_misplaced_preregs(REPO_ROOT)
    assert not hits, f"prereg outside canonical home + legacy set: {[str(h) for h in hits]}"


def test_canonical_citations_verify():
    for sub in ("notes", "plans"):
        for p in sorted((DOCS / sub).glob("*.md")):
            text = p.read_text(encoding="utf-8")
            for lineno, line in citation_candidates(text):
                m = CITATION_RE.match(line)
                assert m, (
                    f"{p.name}:{lineno} looks like a freeze citation but does not "
                    f"parse canonically — a lane must not LOOK frozen while "
                    f"unverifiable: {line!r}"
                )
                actual = git_blob_sha(m["commit"], m["path"])
                assert actual == m["blob"], (
                    f"{p.name}:{lineno} freeze claim FALSE: blob at commit is "
                    f"{actual}, cited {m['blob']}"
                )


def test_pinned_sestm_freeze_verifies():
    path, commit, blob = PINNED_LEGACY_CITATION
    assert git_blob_sha(commit, path) == blob


def test_program_record_chain():
    text = (HARNESS / "PROGRAM_RECORD.md").read_text(encoding="utf-8")
    rows, tip, count, headers = record_table(text)
    assert headers == 1, "PROGRAM_RECORD must contain exactly ONE table"
    assert len(rows) >= 8, "seed rows missing"
    for i, cells in enumerate(rows, 1):
        assert len(cells) == 6, (
            f"row {i} has {len(cells)} cells; schema needs 6 — forgot the chain cell?"
        )
    exp = expected_chain(rows)
    for i, (cells, want) in enumerate(zip(rows, exp), 1):
        assert cells[-1] == want, (
            f"PROGRAM_RECORD row {i} chain mismatch: has {cells[-1]!r}, expected "
            f"{want!r}. If appending a new row, paste this expected hash. If NOT "
            "appending, a row was edited/reordered/deleted — restore it."
        )
    assert tip is not None and count is not None, "chain-tip footer missing"
    assert count == len(rows) and tip == exp[-1], (
        f"chain tip says {tip!r}/{count} rows but table has {exp[-1]!r}/{len(rows)}: "
        "either tail rows were deleted, or you appended without updating the footer "
        f"— expected footer: 'Chain tip: {exp[-1]} over {len(rows)} rows'"
    )


def test_templates_self_check():
    t = HARNESS / "templates"
    prereg = (t / "prereg-template.md").read_text(encoding="utf-8")
    assert not missing_anchors(prereg)
    for name in ("source-matrix-template.md", "closeout-template.md"):
        body = (t / name).read_text(encoding="utf-8")
        assert "Frozen criteria: <" in body, f"{name} lost the canonical citation line"


# ---------- negative self-tests (prove the checks detect violations) ----------

def test_neg_missing_anchor_detected():
    assert missing_anchors("decision rule and budget only") == ["STOP condition"]


def test_neg_malformed_citation_detected():
    bad = "Frozen criteria: some/path | Freeze commit: deadbeef | blob SHA: cafe"
    (lineno, line), = citation_candidates(bad)
    assert CITATION_RE.match(line) is None


def test_neg_fenced_citation_ignored():
    quoted = "```\nFrozen criteria: x | Freeze commit: y | blob SHA: z\n```\n" \
             "    Frozen criteria: a | Freeze commit: b | blob SHA: c"
    assert citation_candidates(quoted) == []


def test_neg_unreachable_commit_fails():
    with pytest.raises(AssertionError, match="unverifiable"):
        git_blob_sha("0" * 40, "README.md")


def _chained(n):
    rows = [[str(i), f"lane{i}", "STOP", "n", "ptr"] for i in range(1, n + 1)]
    hashes = expected_chain([r + [""] for r in rows])
    return [r + [h] for r, h in zip(rows, hashes)]


def test_neg_tampered_record_row_detected():
    good = _chained(2)
    good[0][1] = "edited-lane"  # tamper
    assert expected_chain(good)[0] != good[0][-1]


def test_neg_reordered_rows_detected():
    good = _chained(3)
    swapped = [good[1], good[0], good[2]]
    exp = expected_chain(swapped)
    assert any(r[-1] != w for r, w in zip(swapped, exp))


def test_neg_interior_deletion_detected():
    good = _chained(3)
    cut = [good[0], good[2]]  # delete middle row
    assert cut[1][-1] != expected_chain(cut)[1]


def test_neg_tail_deletion_detected_by_tip():
    good = _chained(3)
    tip_before = good[-1][-1]
    cut = good[:2]  # delete last row; footer still pins tip_before
    assert expected_chain(cut)[-1] != tip_before


def test_neg_misplaced_prereg_detected(tmp_path):
    nest = tmp_path / "apps" / "sub"
    nest.mkdir(parents=True)
    (nest / "sneaky-prereg.md").write_text("x", encoding="utf-8")
    (tmp_path / "docs" / "superpowers" / "plans").mkdir(parents=True)
    hits = find_misplaced_preregs(tmp_path)
    assert [h.name for h in hits] == ["sneaky-prereg.md"]


def test_neg_allowed_homes_not_flagged(tmp_path):
    plans = tmp_path / "docs" / "superpowers" / "plans"
    templates = tmp_path / "docs" / "superpowers" / "harness" / "templates"
    plans.mkdir(parents=True)
    templates.mkdir(parents=True)
    (plans / "2099-01-01-lane-prereg.md").write_text("x", encoding="utf-8")
    (templates / "prereg-template.md").write_text("x", encoding="utf-8")
    assert find_misplaced_preregs(tmp_path) == []
```

- [ ] **Step 2: Dispatch (Bash tool, NOT PowerShell; short slash-free task string):**

```bash
npm run hermes:production -- --task "Implement the test described in TASK_HARNESS_TEST.md"
```

- [ ] **Step 3: Verify Codex's REAL git state** (it has skipped work before):

```bash
git status --porcelain && git log --oneline -3 && git diff HEAD~1 --stat
```

Expected: exactly one new file `apps/quant/advisor/tests/test_prereg_conformance.py`,
one commit, no other files touched.

---

### Task 7: Integration verification

- [ ] Run: `python -m pytest apps/quant/advisor/tests/test_prereg_conformance.py -v` → all pass (≈17 tests). A `test_program_record_chain` failure HERE means the Task 3 snippet and the test's `expected_chain` drifted — recompute the seed hashes with the TEST's helpers, not the snippet; do not touch the assertions.
- [ ] Run: `python -m pytest apps/quant/advisor/tests -q --tb=short` → 322+~17 passed + 1 xfailed
- [ ] Run: `npm run advisor-gate` → exit 0; `npm run advisor-release-gate` → exit 1
- [ ] Frozen-surface diff (ALL frozen surfaces, not just two):

```bash
git diff main -- apps/quant/advisor/backtest/PREREG.md \
  apps/quant/advisor/backtest/VALIDATION_PREREG.md \
  apps/quant/advisor/backtest/FLOOR_RESULT.md \
  apps/quant/advisor/research/READING_B_PREREG.md \
  apps/quant/advisor/research/READING_C_PREREG.md \
  apps/quant/advisor/research/CANDIDATE_PREREG.md \
  apps/quant/advisor/research/HOLDOUT_LEDGER.md
```

Expected: empty output.
- [ ] Holdout ledger still empty (also covered by the diff above)
- [ ] `rm TASK_HARNESS_TEST.md` (it was never committed — the task file forbids `git add .`); `git status --porcelain` → clean

---

### Task 8: Land

- [ ] `git fetch origin && git rev-list --left-right --count exec/validation-harness-package...origin/main` → left N, right 0 (fast-forwardable)
- [ ] Push branch; open PR to main titled "Harness package: playbook, hash-chained record, templates, conformance floor"; body cites the spec and review record
- [ ] After merge: on main, re-run Task 7 checks; update memory index

## Rollback

Single revert of the merge commit restores everything; no migrations, no state.
