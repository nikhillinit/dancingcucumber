# Next-Priorities Design — Post-7-Negative Program State

**Date:** 2026-07-04
**Status:** REVIEWED — CEO pass (HOLD SCOPE, 6 findings), eng pass (5 findings),
adversarial outside voice (8 findings, all triaged and applied)
**Type:** Program prioritization design. Not a prereg; no thresholds are frozen here.

## 1. Context (evidence from 3-agent codebase review, 2026-07-04)

- **Program record:** 7 independent negative/STOP verdicts. All price, fundamental, and
  filing-text families DEV_FAILED on the 30-mega-cap floor; the structural finding is
  blend non-additivity (Readings B 1/4, C 0/4 fail identically) plus best-standalone < SPY.
- **Closed lanes:** residual-alpha / market-neutral (v2 diagnostic STOP, 2026-06-23,
  mappability ceiling 68.9% < 85% frozen gate). Universe-change screen: INCONCLUSIVE;
  keyless approach exhausted; Phase-1 requires delisting-aware PIT data (paid).
- **Open lanes (roadmap):** SESTM news-sentiment lane (planned, not started; Phase-0
  corpus capability matrix; default STOP). WS3E Days-to-Cover (sequenced behind SESTM).
- **Repo state:** main == origin/main (533fef9), clean. 295 tests / 62 files. Gates intact
  (advisor-gate exit 0; advisor-release-gate exit 1 by design). Holdout blinded; ledger empty.
- **Loose ends:** 4 untracked docs (2 plans, 2 notes); WS3C plan doc task statuses stale
  (T3/T4 marked pending though Readings B/C ran to completion); 1 unmerged remote branch
  `claude/dancing-cucumber-integration-review-0qox2x` (2 commits, report-only floor
  diagnostics: Sortino, max drawdown, concentration, minimax-robust family pick).

## 2. Goal

Convert the post-7-negative state into (a) a protected, truthful record, and (b) one cheap,
pre-gated test of the only roadmap-open orthogonal lane — with the fallback pre-committed
now so a STOP cannot drift into ad-hoc iteration.

## 3. Assumptions (standing in for interactive clarification)

1. Deliverable is program-level prioritization, not a single-lane pick.
2. Free/keyless data only; any paid data purchase is a separate explicit operator decision.
3. Governance invariants hold: report-only, frozen preregs immutable (separate surfaces
   only), holdout blinded, pre-committed thresholds, default STOP.
4. The diagnostics branch is desirable if it passes backtest-integrity review.
5. Solo developer; code edits dispatched via Hermes per workflow contract.

## 4. Approaches considered

- **A. Hygiene-first, then SESTM Phase-0 (CHOSEN).** Protect the record, then run the
  roadmap-designated next lane with its pre-committed STOP.
- **B. Program-review memo first.** Honest but premature: the roadmap already pre-committed
  SESTM Phase-0 as the cheap next test; a memo before that data re-litigates a decided
  sequence. Embedded instead as the pre-committed fallback on Phase-0 STOP.
- **C. WS3E Days-to-Cover on the existing floor.** Lowest prior: same universe where blend
  additivity fails structurally and mega-cap short interest is weakest. Stays sequenced
  behind SESTM per the deferred-plans roadmap.

## 5. Slice 1 — Repo closeout & observability (~1 session)

**5.1 Reconcile and commit the 4 untracked docs.**
- Files: `docs/superpowers/notes/2026-06-18-storm-orthogonal-indicator-scan.md`,
  `docs/superpowers/notes/2026-06-21-blend-futility-residual-alpha.md`,
  `docs/superpowers/plans/2026-06-18-ws3c-edgar-xbrl-fundamentals.md`,
  `docs/superpowers/plans/2026-06-22-universe-change-residual-screen.md`.
- The WS3C plan gets a dated completion addendum correcting stale task statuses
  (Readings B and C ran; results in `apps/quant/advisor/research/READING_B_RESULT.md`
  and `READING_C_RESULT.md`). Append; never rewrite recorded history.
- Safe to commit: `docs/superpowers/` is excluded from the operator-doc truth scan
  (`apps/quant/advisor/tests/test_docs_truth.py:10-13`).

**5.2 Diagnostics-branch decision.**
- The branch `origin/claude/dancing-cucumber-integration-review-0qox2x` (commits d3d8d0a,
  0e6e1da) forked from a merge-base **47 commits behind current main** — do not assume a
  clean merge.
- Decision rule: attempt a rebase onto main first. If trivially clean → dispatch the
  `backtest-integrity-reviewer` agent; acceptance = report-only (no verdict-logic, PREREG,
  or floor-number changes), suite green, frozen-hash diff empty → PR and merge. If the
  rebase conflicts non-trivially → extract the diagnostic intent (Sortino, max drawdown,
  concentration, minimax-robust family pick) into a note, re-dispatch as a fresh small
  task against current main, and delete the stale branch.
- Sequencing rule: exactly one slice in flight at a time; Slice 1 lands before Slice 2
  starts (shared-working-tree incident on record).

**5.3 Verification (gate for Slice 1 done).**
- Full suite green (295+ tests), `npm run advisor-gate` exit 0,
  `npm run advisor-release-gate` exit 1 (unchanged by design), holdout ledger still empty.

## 6. Slice 2 — SESTM Phase-0 corpus capability matrix (~1–2 sessions)

Per `docs/superpowers/plans/2026-06-23-sestm-news-lane-plan.md`:

- **Freeze first, mechanically:** write the Phase-0 prereg (seven must-have corpus
  criteria: $0 cost; trade-date-assignable timestamps — date-only granularity fails
  unless provably post-close; named mapping file with disclosed survivorship limit;
  **corpus-intrinsic** breadth floor — deliberately not an index-membership test, since
  no keyless PIT membership list exists (the T0.2b confound); 2015-2023 history;
  immutable document IDs; **corpus suitability** — third-party news text, issuer
  self-disclosures fail the SESTM-as-published hypothesis), then **commit the criteria
  file and record its SHA before evaluating any source** — the Step-0 vendor-matrix
  pattern, hardened by the v3 goalpost-move lesson.
- **Hard budget cap:** Phase-0 ≤ 1 session and $0 data spend. Phase-0 is a desk-check-
  grade gate: its value is the frozen record, not discovery — the real kill lives in
  Phase-1's net-OOS-sign gate. Not finishing inside the cap is itself a STOP-relevant
  signal to carry into the program review memo — never a reason to extend.
- **Reuse mandate:** the matrix builds on existing machinery —
  `apps/quant/advisor/data/filing_text_fetch.py`, `edgar_xbrl_fetch.py`, and the QC-
  diagnostic EDGAR 8-K dataset (16,663 filings / 5,196 CIKs) — no parallel fetchers.
- **Build the matrix:** evaluate candidate free corpora (EDGAR material-event filings /
  8-K items as the natural keyless corpus; plus keyless news feeds) against the frozen
  criteria. Report-only artifact.
- **Pre-committed STOP:** if no affordable corpus clears the must-haves → STOP and write
  the closeout note. PASS ≠ alpha claim; PASS only unlocks a separately greenlit Phase-1
  minimal probe (its own prereg, its own kill: no positive net OOS sign → STOP).
- Floor, holdout, and all frozen surfaces untouched throughout.

## 7. Pre-committed decision tree

The **program review memo is unconditional after Phase-0** — it consolidates all eight
results (7 recorded negatives + the Phase-0 verdict) into a strategic record with explicit
reframes to evaluate: (a) validation-harness-as-asset, (b) risk/diagnostics advisor
product, (c) paid-data universe lever as a costed budget decision.

- **Phase-0 PASS** → the memo is the required input to the operator's Phase-1 greenlight
  decision (Phase-1 minimal probe is already specified in the SESTM plan; separate prereg
  surface). A PASS without the consolidated record in front of the operator is not a
  greenlight.
- **Phase-0 STOP** → the memo is the go/no-go artifact. No further signal lanes open
  without it.

## 8. Non-goals

- No holdout touch. No new fields on any frozen prereg (separate surfaces only).
- No paid data purchases. No WS3E work before the SESTM Phase-0 verdict.
- No changes to floor verdict logic or PREREG.md. No production/deployment claims.

## 9. Testing & verification strategy

Every slice ends with: full pytest suite green; `advisor-gate` exit 0;
`advisor-release-gate` exit 1 (the floor still fails by design — that is the honest state);
docs-truth tests green; `git diff` empty on frozen prereg files; holdout ledger empty.

## 10. Risks

- **SESTM corpus-swap risk (central):** published alpha (gross Sharpe 4.29 in-paper) may
  not survive a free corpus + net-of-cost + tradability; Phase-0/1 gates exist to surface
  that STOP for the price of the matrix, not a full pipeline build.
- **Universe mismatch:** SESTM is a small/mid-cap lane; a Phase-0 PASS implies later
  universe/fixture work — flagged in the matrix criteria (coverage must-have), not deferred
  silently.
- **Diagnostics branch:** written by another agent; integrity review is mandatory, not a
  formality (shared-working-tree incidents on record).
