# Closeout Execution Slice — WS0 → WS1 → WS3A → WS3B (report)

**Date:** 2026-06-18. **Branch:** `exec/closeout-ws0-ws3b` (base `cd67c13` = main `c0d505d` + handoff).
**Merge state reconciled:** PR #7 (`nikhillinit/dancingcucumber#7`) merged on 2026-06-18 at
`b5f511931c8018a88941af7ae62953365ff30fab`; current local `main` has advanced past it to `5c5be0b`,
with `b5f5119` present in history.
**Original branch publish:** yes — `origin/exec/closeout-ws0-ws3b` was the pre-merge feature branch.
**Method:** Claude planned/dispatched/verified; Hermes (Codex, gpt-5.5) implemented every edit/test.

## Commits (four, one per workstream)
| SHA | Workstream | What |
|---|---|---|
| `ba8f125` | WS0 | `run-floor.mjs` rejects any arg except bare/`--enforce` BEFORE spawning python (closes the `--holdout` footgun, exit 2); `test_run_floor_guard.py` (3 fast rejection tests). |
| `5c35a19` | WS1 | 8 operator docs given first-viewport truth headers; 41 obsolete promo claims quarantined into `AIHF_TRUTH_ARCHIVE` blocks; `test_docs_truth.py` (explicit 8-doc set, header + denylist + marker enforcement); hygiene-evidence note. |
| `68b4506` | WS3A | Reading B stub → source-agnostic data contract (14-field schema, strict availability formula, denominator bridge, hash/trial-surface pins). |
| `95800d9` | WS3B | Reading B `## PIT Source Feasibility Record`: **SEC EDGAR XBRL QUALIFIES** (not `RESTATED_PROXY_ONLY`). |

## Verification (Claude ran; Codex cannot run node/npm)
- Full suite **195 → 203 passed** (advisor-gate, independently re-run at finish).
- `npm run advisor-gate` → exit 0, verdict **DEV_FAILED** (ensemble Sharpe 0.73 < best family 0.83; DSR 0.80 < 0.95, report-only).
- `node tools/run-floor.mjs --enforce` → **exit 1**; bare report → exit 0.
- `node tools/run-floor.mjs --holdout` and `--bogus` → **exit 2**, refused before any python spawn.
- `git diff -- apps/quant/advisor/research/HOLDOUT_LEDGER.md` → **EMPTY** (untouched).

## Secret scan (Claude ran the scan; redacted)
Result **`[]`** — no FRED/AlphaVantage key-like literal in tracked or untracked files (env-only confirmed).

## WS3B feasibility decision (decided on evidence, not the safe stop)
**`SEC_EDGAR_XBRL` QUALIFIES.** Discriminator answered YES: the exact accession/form/accepted record for
book value (`StockholdersEquity`) + shares can be selected as-of the evaluation date — each NUM fact is
keyed to an accession whose SUB record carries `filed`/`accepted` (in-repo stefan-jansen `04_sec_edgar`
SUB/NUM evidence; SEC keyless APIs, declared User-Agent, ~10 req/s). Aggregate "current" company-facts
do NOT qualify. Binding constraint: XBRL coverage from 2009. This authorizes only the NEXT planning step.

## Rails held (nothing promoted)
No holdout run; no sizing/production/live/broker language introduced; no new dependency; no acceptance-bar
change; floor stays `DEV_FAILED`. Untracked scratch and `.claude/settings.local.json` left untouched
(140-file inventory recorded, no mass-delete).

## Disclosed process notes
- `--skip-preflight-gate` (with honest reasons) used on the WS1-docs and WS3A/WS3B/reorg dispatches: the
  suite was intentionally red during the WS1 TDD loop (new `test_docs_truth.py` fails until docs fixed),
  and the spec edits are code-independent; Claude re-verified the full suite green before each commit.
- WS3B initially inserted the PIT record mid-`## Data contract`, orphaning three `###` subsections; a
  focused reorder dispatch made the PIT record a clean top-level sibling (no wording changed).
- README gained an "Operator Reading Checklist" (verbose but truthful) so its archive block lands past
  line 40 while the first viewport stays current-truth.

## Still gated / unstarted (what unblocks each)
- **WS2** (keyed live FRED/AlphaVantage smoke) — needs operator-rotated credentials; cannot run headless.
- **WS3C** (source-specific fixture + pre-registration) — now **UNBLOCKED** by WS3B QUALIFIES, but is the
  next *planning/data-build* step and needs an operator greenlight (it builds a committable EDGAR slice).
- **WS4 / WS5** (candidate bench run / release blocking) — require a dev-passing candidate that does not
  yet exist; floor remains `DEV_FAILED`.

## Current operator handoff
No merge action remains for `exec/closeout-ws0-ws3b`; treat the slice as landed through PR #7. Continue
only from the post-merge `main` state: Reading B remains report-only, the advisor floor still reports
`DEV_FAILED`, the reserved holdout stays untouched, and no sizing/production/live/broker authorization
was created by the merge.
