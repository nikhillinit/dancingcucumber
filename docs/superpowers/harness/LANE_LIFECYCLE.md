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
