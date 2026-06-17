# Holdout Touch Ledger — shared reserved tail (Amendment F2)

Append-only. EVERY evaluation of the reserved holdout (the floor's own future `--holdout`
run AND every candidate that earns a holdout) appends one row, so multiple-testing on the
single shared tail is counted and visible. A side-bench touch BURNS the shared tail:
promotion (Plans 1b/3) then requires a FRESH holdout, never the peeked one.

The tail unlocks ONLY via `candidate_run_hash(cfg, fixture)` (config + fixture bytes),
never a fixture-blind hash or an arbitrary string.

| Date | Run Hash | Families | Verdict | Who |
|------|----------|----------|---------|-----|
| _(none yet — holdout untouched)_ | | | | |
