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
