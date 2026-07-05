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
