# Validation-Harness-as-Asset Design — Internal Governance Package

**Date:** 2026-07-05
**Status:** REVIEWED + DEBATE-AMENDED — CEO pass (HOLD SCOPE, 6 findings applied),
eng pass (3 findings applied), outside voice (Codex, 8 findings: 6 applied, 1
rejected as decided-scope re-litigation, 1 operator-decided). Amended 2026-07-05
after the Hermes debate on the execution plan (run hermes-2026-07-05T12-27-27-299Z):
chain-tip footer added — a bare hash chain cannot catch tail deletion.
**Type:** Packaging/documentation design. Not a prereg; no thresholds are frozen here.
**Provenance:** Reframe (a) of `docs/superpowers/notes/2026-07-04-program-review-memo.md`,
elected by the operator 2026-07-05 (memo decision point 1, recommended default).

## 1. Context

- **Program record:** eight independent negative/STOP verdicts (memo §1). The floor is
  DEV_FAILED and stays that way; no signal lane is open. The memo's decision point 1
  asks whether the durable value is the falsification machinery itself.
- **Operator decisions (2026-07-05):** consumer = future-self / this program (internal
  template, not OSS/product); form = playbook + copyable templates; enforcement = docs
  plus ONE conformance test; scope = full operating loop; add a living PROGRAM_RECORD;
  location = `docs/superpowers/harness/`.
- **What exists today:** the harness machinery is real but scattered — frozen preregs
  (each lane hand-rolled its own surface), gate scripts (`advisor-gate`,
  `advisor-release-gate`), `HOLDOUT_LEDGER.md`, the Phase-0 freeze-then-evaluate matrix
  pattern, closeout notes, and hard-won rules living only in session memory and merged
  PR history.

## 2. Goal

Package the prereg/floor/holdout governance machinery as the program's primary
artifact: any future lane (paid-data lever, cold hypothesis, anything) starts from a
documented, reusable scaffold instead of re-deriving discipline from old lanes and
out-of-repo memory. No signal claim. Every frozen surface untouched.

## 3. Deliverables

### 3.1 `docs/superpowers/harness/README.md`
Index of the package plus a "start a new lane in 5 steps" quickstart.

### 3.2 `docs/superpowers/harness/LANE_LIFECYCLE.md`
The playbook: the full operating loop, each stage citing its canonical in-repo exemplar.

1. **Operator election** — no lane opens without a memo-based operator decision
   (§7 of `2026-07-04-next-priorities-design.md` is the pattern).
2. **Review pipeline** — brainstorm → spec → CEO review → eng review → implementation
   plan (the 2026-07-04 spec/plan pair is the exemplar).
3. **Prereg + freeze protocol** — write the prereg from the template; commit it; record
   the freeze commit + blob SHA in the evaluation artifact (exemplar: SESTM Phase-0
   prereg, frozen at `d420af7`).
4. **Fixtures** — SHA-pinned bytes; fixture hash recorded (exemplar: WS3D filing-text
   fixture).
5. **Separate-surface rule** — never modify a frozen surface; new files/classes only;
   any divergence from byte-identical-to-floor is disclosed in the result doc
   (exemplar: Lane B's three disclosed divergences).
6. **Hermes dispatch protocol** — code edits via `npm run hermes:production`; short
   slash-free task string pointing at a repo-root task file; verify Codex's real git
   state afterwards.
7. **Verdict semantics** — default STOP; near-miss = STOP (the 68.9% < 85% precedent
   binds); decide off `dev.passed`, not the verdict enum, when the holdout is blinded;
   power-limited is an annotation, not an excuse; budget-cap exhaustion is a STOP,
   never an extension; post-freeze goalpost-moves are new programs, not continuations.
8. **Holdout governance** — holdout stays blinded; every touch appends a ledger row;
   unlock only via verified `candidate_run_hash(cfg, fixture)`; a side-bench touch
   burns the shared tail.
9. **Closeout + record append** — closeout note from the template; one row appended to
   `PROGRAM_RECORD.md` in the same commit.
10. **End-of-slice verification checklist** — full suite green; `advisor-gate` exit 0;
    `advisor-release-gate` exit code MATCHES the recorded floor state (currently 1 =
    DEV_FAILED, the honest state; a change in this exit code is itself a headline
    event requiring an operator decision, never a checklist pass); `git diff` empty on
    frozen files; ledger unchanged (empty unless a holdout was legitimately touched);
    docs-truth green.

Rules that today live only in session memory get transcribed here so the repo is
self-contained.

### 3.3 `docs/superpowers/harness/PROGRAM_RECORD.md`
Living append-only verdict table. Schema (explicit; extends the memo's table by one
column): `# | Lane | Verdict | Key number | Record pointer`. Seeded with the eight
rows from the 2026-07-04 memo §1; for those historic rows the record pointer is the
memo itself (not all eight have a dedicated closeout — stating that beats faking it).
Append rule: one row per lane closeout, added in the closeout commit. Memos remain
frozen point-in-time snapshots; this file carries the record forward.

Append-only is machine-checked by a **row hash chain plus a chain-tip footer**
(threat model: the stale-status-doc failure mode from PRs #17/#18 — sloppy edits,
not adversaries). Every data row ends with a short chain-hash column:
`sha256(prev_chain_hash + row_text)[:12]`; the file ends with
`Chain tip: <hash> over <N> rows`. The conformance test recomputes the chain from
row 1 and checks the tip: any edit, reorder, or interior deletion breaks the chain;
tail deletion breaks the tip footer (a shorter chain is internally valid — the tip
pins length, per debate finding hermes-2026-07-05). Stated honestly: a deliberate
rewrite of rows AND footer together defeats this; that is git-history territory, not
this guard's job. Append ceremony: write the row, run pytest once, paste the printed
hash, update the footer. No helper script, no new files. Exactly one table lives in
the file (enforced), so a future second table cannot pollute the chain.

### 3.4 `docs/superpowers/harness/templates/`
- `prereg-template.md` — hypothesis; universe/data; frozen criteria/thresholds;
  pre-committed decision rule (with explicit STOP default and near-miss=STOP clause);
  budget cap; holdout statement; freeze-protocol footer.
- `source-matrix-template.md` — Phase-0-style capability matrix: criteria columns,
  PASS/FAIL cells each carrying a citation, pre-committed verdict rule quoted verbatim,
  and the canonical freeze citation (below).
- `closeout-template.md` — verdict; binding failures; what the record now rules out;
  what is explicitly NOT ruled out; program-record row; next-decision pointer; the
  canonical freeze citation.

**Canonical freeze citation (defined once, used by both evaluation templates):**

```
Frozen criteria: <repo-relative-path> | Freeze commit: <40-hex> | blob SHA: <40-hex>
```

One line, parseable deterministically; this is the format the conformance test
verifies. Scan scope is `docs/superpowers/notes/` and `docs/superpowers/plans/` ONLY
— templates (which must contain `<...>` placeholders) and specs (which quote the
format, like this one) are exempt by location; templates are checked separately by
the self-check (§3.5 test 4). The existing SESTM Phase-0 matrix predates the format
(two-line citation) and is verified as a pinned special case (§3.5 test 2). Legacy
prose mentions (`Freeze commit hash: \`7f54f55\``-style in the two 2026-06-23
preregs, and narrative references in the execution plan) do not match the canonical
candidate prefix and are deliberately ignored.

The verification checklist is a playbook stage (3.2 item 10), not a fourth template.

### 3.5 `apps/quant/advisor/tests/test_prereg_conformance.py`
One test file, dispatched via Hermes. Check logic lives in small module-level helpers
so each check is unit-testable against synthetic inputs. Tests:

1. **Future-prereg anchors** — every `*-prereg.md` under `docs/superpowers/plans/`
   (the canonical home) NOT in the grandfather set must contain: a pre-committed
   decision rule, a budget cap, and an explicit STOP condition. Grandfather set
   (exact filenames, pinned — these predate the anchor phrasing and verifiably lack
   it): `2026-06-23-qc-edgar-diagnostic-v2-prereg.md`,
   `2026-06-23-qc-source-integrity-diagnostic-prereg.md`. The SESTM Phase-0 prereg
   already passes all three anchors and is deliberately NOT grandfathered — it
   exercises the positive path against a real file. The test asserts all three legacy
   files exist, so the pin cannot rot silently and the glob is provably non-vacuous.
2. **Misplaced-prereg location enforcement** — any file matching `*prereg*.md`
   (case-insensitive) ANYWHERE in the repo outside the canonical home and not in the
   pinned legacy set FAILS as misplaced. Legacy set: `apps/quant/advisor/backtest/
   PREREG.md`, `backtest/VALIDATION_PREREG.md`, `research/READING_B_PREREG.md`,
   `research/READING_C_PREREG.md`, `research/CANDIDATE_PREREG.md` (final list
   enumerated at implementation from a repo-wide glob; each is a frozen surface
   governed by its own hash pins). A future lane cannot evade governance by filing
   its prereg in an unwatched directory.
3. **Git-verified freeze citations** — within `docs/superpowers/{notes,plans}/`,
   every line starting `Frozen criteria:` and containing `Freeze commit:` must parse
   as the canonical citation (§3.4); a partial or malformed match FAILS loudly (a
   lane must not look frozen while unverifiable). Each parsed citation is verified:
   `git rev-parse <commit>:<path>` equals the declared blob SHA. Unreachable commit =
   FAIL with file+citation context, never skip (main is never force-pushed;
   unreachable means something is actually wrong). The existing SESTM Phase-0
   two-line citation is verified as a pinned special case (immediate value: the most
   recent freeze claim is machine-checked on day one). Subprocess in list form,
   `shell=False`. (Precedent for shelling to git inside a test: `test_docs_truth.py`.)
4. **PROGRAM_RECORD hash chain + tip** — recompute the row chain (§3.3) from row 1
   (header-anchored parse; exactly one table; six cells per row) and verify the tip
   footer (final hash + row count — catches tail deletion, which a chain alone
   cannot). Failure names the offending row and prints the expected hash.
5. **Template self-check** — the three templates exist and themselves carry the
   required anchors (including the canonical citation line with `<...>`
   placeholders), so template drift is caught.
6. **Negative self-tests** — the helpers are exercised against synthetic bad inputs
   (`tmp_path`): prereg missing an anchor, tampered/reordered/deleted record row,
   malformed citation, misplaced prereg — proving the checks detect violations, not
   merely pass on today's tree.

**What this test deliberately does NOT check** (stated here and in the playbook, so
one test cannot manufacture false governance confidence): operator election, review
pipeline execution, separate-surface discipline, post-freeze goalpost-moves, holdout
ledger semantics (enforced by the unlock-hash code path), and closeout quality. Those
remain review-enforced; the test is the machine-checkable floor for artifact
discipline, not the governance loop itself.

## 4. Non-goals

- No code scaffolding, generators, or base-class extraction (no lane is greenlit; YAGNI).
- No OSS packaging, no product claims, no external consumer work.
- No edits to existing preregs, gate logic, verdict semantics, or floor numbers
  (`test_floor_diagnostics.py` pins stay byte-identical).
- No new fields on any frozen prereg surface.

## 5. Testing & verification

- Suite grows 322+1 → ~332+1 (one file; six conformance checks plus negative
  self-tests).
- Standard end-of-slice checklist: suite green; `advisor-gate` exit 0;
  `advisor-release-gate` exit 1; frozen-diff empty; holdout ledger empty.
- `docs/superpowers/` remains excluded from the docs-truth operator-doc scan; the new
  conformance test is a separate, narrower guard and must not widen that scan.

## 6. Risks

- **Playbook drift:** the playbook could rot as practice evolves. Mitigation: the
  conformance test pins the machine-checkable minimum; the playbook cites exemplars by
  path so staleness is visible.
- **False conformance:** a future prereg could carry the anchors textually while
  violating their spirit. The test is a floor, not a substitute for review; the
  playbook says so explicitly.
- **Git-verification brittleness:** shallow clones or rewritten history would break
  test 3. Position: main is never force-pushed (standing constraint), so an
  unreachable freeze commit is evidence of real corruption — the test FAILS with
  context; it never skips and never passes silently.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | CLEAR (HOLD SCOPE) | 6 findings, all applied; 0 critical gaps open |
| Codex Review | outside voice | Independent 2nd opinion | 1 | issues found → resolved | 8 findings: BLOCKER + 5 applied, 1 rejected (re-litigated decided scope), 1 operator-decided (rider dropped) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 3 findings (self-breaking citation scan, grandfather trim, prereg-location evasion), all applied |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | no UI scope |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | not applicable |

- **CROSS-MODEL:** Codex's BLOCKER (self-breaking citation scan) was independently
  found by the eng pass minutes earlier — strong agreement signal; fix applied once,
  resolves both.
- **UNRESOLVED:** 0 — both operator decisions (record hash chain, rider drop) made.
- **VERDICT:** CEO + ENG + OUTSIDE VOICE CLEARED — ready for writing-plans.
