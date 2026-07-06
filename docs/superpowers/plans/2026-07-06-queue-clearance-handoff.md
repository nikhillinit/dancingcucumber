# Queue-Clearance Handoff — Hermes Orchestration

**Written:** 2026-07-06, immediately after PR #21 merged (skill_weight seam activation).
**Purpose:** clear the remaining tracked work queue via Hermes dispatch. Everything here is
chore/hardening — no new signal lanes (those are operator-gated by the program review memo,
`docs/superpowers/notes/2026-07-04-program-review-memo.md`).

## State snapshot (verify before starting)

| Invariant | Expected |
|---|---|
| `git log --oneline -1` on main | `7c4f742` (merge PR #21) or descendant |
| `npm run check` (full advisor suite) | **353 passed, 0 xfail** |
| `npm run advisor-gate` | exit 0 (report-only, prints `floor: DEV_FAILED`) |
| `node tools/run-floor.mjs --enforce` | **exit 1** (release blocked — this never changes here) |
| Frozen (hook-enforced) | `tests/fixtures/floor_prices.csv`, `PREREG.md`, `UNIVERSE_RULE.md` — allocator.py is UNFROZEN as of `ccc25b0` |
| Holdout | untouched, ledger empty |

## Hermes dispatch mechanics (proven 2026-07-06, 3/3 clean)

1. **`npm run hermes:production -- --task` DOES NOT WORK on Windows** — npm swallows
   `--task`. Dispatch with: `node orchestrate.mjs --phase production --task "Read HERMES_TASK_<X>.md at the repo root and execute it exactly."`
2. Write each task as a repo-root `HERMES_TASK_<X>.md` (short file-pointer pattern; long
   inline task strings hit the Windows cmdline limit). Delete the task files before committing.
3. Every task file ends with the hard rules: **edit ONLY the listed files; NO git commands;
   NO tests/npm scripts** (Claude runs all verification and owns git).
4. orchestrate.mjs runs `npm run check` pre- AND post-flight (~3 min each). A nonzero dispatch
   exit can be just the postflight gate racing a sibling task — **judge the diff, not the exit
   code**; verify `git status --porcelain` after every task (Codex has skipped steps before).
5. Parallel dispatch is safe ONLY with disjoint file lists (background PowerShell calls).
   Sequence anything that shares a file.

## The queue

### Q1 — TODOS.md P3: finite upper bound on `FamilySignal.skill_weight` (actionable NOW)

The one deliberate schema change excluded from PR #21. `skill_weight` is the only field
admitting `float("inf")` (`ge=0` passes inf); inf yields `inf/inf → nan → min(100, nan) → 100.0`
— a silently confident garbage vote. Interim mitigation (run_pipeline tripwire) is live, so
this is hardening, not a hot fix.

- **Change:** `apps/quant/advisor/schemas.py:21` — add `le=100` to the Field
  (recommended default: keeps the existing `ge=0` NaN-rejection semantics intact; operator may
  substitute `allow_inf_nan=False` — pick ONE, don't stack both without need).
- **Tests:** in `test_schemas.py`, add rejection tests for `float("inf")` and an over-bound
  value (e.g. `101.0`). Existing negative/NaN rejection tests stay untouched.
- **Rider in same PR:** update `TODOS.md` — mark P3 done/remove it; P2 stays.
- **Pre-flight check for the dispatcher:** grep confirms no fixture uses skill_weight > 5.0,
  so `le=100` breaks nothing. NEVER touch `PreRegConfig` (immutable PREREG.md hash).
- **Acceptance:** full suite green (353 + 2 new = 355 expected); floor `--enforce` exit 1;
  `test_rails.py` green.
- Branch `exec/skill-weight-upper-bound`, PR to main. One Hermes task (single file + test file
  + TODOS.md).

### Q2 — Diagnostics reviewer follow-ups: REGENERATE, then fix (two phases)

Provenance: the 2026-07-04 diagnostics merge (`1e65100`) got a backtest-integrity-reviewer
ACCEPT with **7 MEDIUM/LOW follow-ups** that were chipped to a background task and never
persisted — the full list is LOST. Only three exemplars survive in project memory:
1. SPY window mismatch in the Sortino/maxDD display,
2. untested `json.dumps` crash surface,
3. a parity skill_weight caveat — **likely OBSOLETE post-PR #21** (parity semantics were
   rewritten); verify and close rather than fix.

- **Phase 2a (Claude, read-only, no Hermes):** re-run the `backtest-integrity-reviewer` agent
  over the diagnostics surface (the report-only `diagnostics` block added in `1e65100`;
  `git show 1e65100` for the diff scope) to regenerate the findings list. Triage: drop
  anything obsoleted by PR #21, keep genuine MEDIUM/LOW items. This is the gate for 2b —
  do not dispatch fixes from memory's three-exemplar stub.
- **Phase 2b (Hermes):** one task file per disjoint fix-cluster; parallel only if file lists
  don't overlap. All fixes are report-only surface — the floor verdict pin
  (`test_diagnostics_do_not_change_the_verdict`, numbers 0.7323/0.7562/0.8277) must stay
  byte-identical. Any fix that would move a pinned number is out of scope — STOP and surface it.
- **Acceptance:** suite green with new tests covering each fixed item; floor JSON byte-identical
  except inside the report-only diagnostics block; gates 0/1 unchanged.
- Branch `exec/diagnostics-followups`, PR to main.

### Q3 — TODOS.md P2: weight observability (CONDITIONAL — do NOT build now)

Standing tripwire, not a task: P2 (surface non-uniform skill_weights in
`Allocation.reasoning` / CLI report) **must land in the same change that ever relaxes the
run_pipeline non-uniform-weights tripwire**. Until a validated spec-§8 calibration exists,
building it is dead code. Whoever relaxes the tripwire owns P2. Leave in TODOS.md.

### Parked (blocked on inputs, not on work)

- **WS2 keyed live smoke** (Finnhub/NewsAPI/FRED live adapters) — blocked on operator-provided
  API keys.
- **WS4/WS5** — blocked on a dev-passing candidate (none exists; floor DEV_FAILED stands).
- **New signal lanes** (e.g. the pre-registerable long-short hypothesis from the corrected
  residual screen) — operator decision points in the program review memo. NOT queue clearance;
  do not open via this handoff.

## Sequencing

```
Q1 (Hermes, 1 task) ──────────────┐
                                   ├─→ separate PRs, merge in any order
Q2a (Claude review, read-only) ─→ Q2b (Hermes, N parallel-disjoint tasks) ─┘
```

Q1 and Q2a can start in parallel (Q1 edits schemas/tests; Q2a is read-only). Q2b waits on Q2a
triage. Each PR independently satisfies: suite green, `--enforce` exit 1, rails green, holdout
untouched, `.claude/settings.local.json` dirt never committed.

## Rails (inherited, non-negotiable)

- Floor/backtest/holdout untouched; `node tools/run-floor.mjs --enforce` exits 1 forever until
  a defensible floor clears. Never weaken a gate to make a change pass.
- No `PreRegConfig` field changes (immutable PREREG.md hash). PREREG.md / UNIVERSE_RULE.md /
  floor_prices.csv are hook-frozen.
- Feature branch + PR per queue item; verify Hermes' real git state after every dispatch.
- Decide off `dev.passed`, not the verdict enum, if anything ever touches gate output.
