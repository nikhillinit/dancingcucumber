---
name: record-append
description: Run the PROGRAM_RECORD append ceremony with full pre-flight integrity checks (user-only; commits).
disable-model-invocation: true
---

# /record-append — PROGRAM_RECORD append ceremony

Appends exactly ONE row to the append-only verdict ledger
`docs/superpowers/harness/PROGRAM_RECORD.md` after a lane closeout (decision/result),
with the pre-flight integrity checks that keep the hash chain honest. This commits, so
it is **user-invoked only** (`disable-model-invocation: true`) — never run autonomously.

The ledger is hash-chained: each row carries a 12-hex chain cell computed as a running
`sha256("GENESIS"|row1|row2|…")[:12]` over the first five cells of every prior row plus
its own, and a footer pins `Chain tip: <hash> over <N> rows`. Both are recomputed by
`test_program_record_chain` in
`apps/quant/advisor/tests/test_prereg_conformance.py`. Edit/reorder/interior-delete a row
and the chain fails; tail-delete and the tip footer fails.

**Observed baseline (2026-07-10):** current tip `8f3625f75365 over 10 rows`.
Verify live before appending — do not trust this number blindly.

---

## STEP 1 — PRE-APPEND INTEGRITY (all must hold, else STOP)

Do NOT append if any check fails. This is a governance ledger; a bad append is worse
than no append.

1. **Working tree clean except `.claude/settings.local.json`** (which is always dirty).
   The frozen diff — everything else — must be EMPTY.

   ```bash
   git status --porcelain
   ```

   Only `.claude/settings.local.json` (and your intended new PROGRAM_RECORD row, once you
   write it) may appear. Anything else → STOP and resolve first.

2. **Frozen pins byte-identical.** `PREREG.md`, `UNIVERSE_RULE.md`, and
   `fixtures/floor_prices.csv` must be unchanged. A PreToolUse guard
   (`.claude/hooks/guard-frozen-floor.mjs`, matcher `Edit|Write|MultiEdit`) blocks
   overwrites of these surfaces — confirm no drift slipped through another path:

   ```bash
   git diff --stat -- '**/PREREG.md' '**/UNIVERSE_RULE.md' '**/floor_prices.csv'
   ```

   Expect empty output. Any drift → STOP.

3. **Holdout untouched / blinded; ledger empty.** Confirm no holdout series was read and
   the run ledger carries no burn for this decision. If the closeout you are recording
   claims the holdout was touched, that is a separate escalation — STOP.

4. **Conformance suite green** (pytest + report-mode floor, exits 0):

   ```bash
   npm run advisor-gate
   ```

5. **Release gate behaves as pinned.** The `--enforce` floor **exits 1 by construction**
   for this DEV_FAILED program — that non-zero exit is EXPECTED and correct, not a
   failure. What you are verifying is that it still fails the same way, not that it passes:

   ```bash
   npm run advisor-release-gate
   ```

   Exit 1 = pinned/correct. Exit 0 would mean the floor started passing — STOP and
   investigate; do NOT append.

---

## STEP 2 — APPEND

Append **exactly one** new row to
`docs/superpowers/harness/PROGRAM_RECORD.md`, inside the single existing table, following
the row format already in the file. The columns (header `| # | Lane | Verdict | Key
number | Record pointer | Chain |`) are:

| Column | Meaning |
|--------|---------|
| `#` | Sequential row number (prior max + 1). |
| `Lane` | Short lane name + `(YYYY-MM-DD)` decision date, e.g. `L/S reversal Gate-1 kill screen (2026-07-06)`. |
| `Verdict` | The ruling: `FAILS FLOOR` / `DEV_FAILED` / `INCONCLUSIVE` / `STOP` / `CLOSED` (add a parenthetical like `(power-limited)` if used before). |
| `Key number` | The one load-bearing number/reason, e.g. `ens 0.557 < momentum 0.665 < SPY 0.752`. |
| `Record pointer` | Path to the closeout note that justifies the row, e.g. `notes/2026-07-06-ls-reversal-gate1-result.md`. |
| `Chain` | Leave as `TBD` on first write — pytest prints the real hash (Step 3). |

Write the row with the chain cell set to `TBD`. Do NOT touch, reorder, or reformat any
existing row. Do NOT update the tip footer yet.

---

## STEP 3 — RECOMPUTE / VERIFY THE CHAIN HASH

Run the conformance test that recomputes the chain and, on the `TBD`/mismatch, prints the
expected hash in its assertion message:

```bash
python -m pytest apps/quant/advisor/tests/test_prereg_conformance.py::test_program_record_chain -q
```

The failing assertion tells you exactly which hash to paste ("If appending a new row,
paste this expected hash"). Then:

1. Replace the `TBD` chain cell of your new row with the printed 12-hex hash.
2. Update the footer to `Chain tip: <new-hash> over <N> rows`, where `N` is the new row
   count (prior `N` + 1 — from the observed baseline, `10` → `11`).

Re-run the same test; it must now pass, confirming the new tip extends the prior one by
exactly one row:

```bash
python -m pytest apps/quant/advisor/tests/test_prereg_conformance.py::test_program_record_chain -q
```

Optionally re-run `npm run advisor-gate` to confirm the whole conformance floor is still
green with the appended row.

---

## STEP 4 — COMMIT (docs / JSON / log ONLY — never code)

Stage only the ledger and any accompanying closeout note / JSON / log for this decision.
**Never** stage code, PREREG.md, UNIVERSE_RULE.md, or floor_prices.csv.

```bash
git add docs/superpowers/harness/PROGRAM_RECORD.md docs/superpowers/notes/<closeout-note>.md
git commit -m "record(<lane>): <verdict> — chain tip <new-hash> over <N> rows"
```

The commit message must name the decision and the new chain tip so the ledger row and the
git history agree.

---

## Abort conditions (STOP, do not commit)

- Any Step 1 check fails (dirty tree beyond `settings.local.json`, pin drift, holdout
  touched, `advisor-gate` red, or `advisor-release-gate` unexpectedly exit 0).
- The chain test still fails after you paste the hash and update the footer.
- You would need to edit an existing row to make the chain pass — that means a prior row
  was corrupted; restore it, do not paper over it.
