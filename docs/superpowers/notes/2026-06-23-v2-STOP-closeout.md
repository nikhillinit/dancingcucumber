# 2026-06-23 — CLOSEOUT (negative): residual-alpha / market-neutral lane CLOSED

> Terminal artifact for the lane opened after T0.2b INCONCLUSIVE. Pre-committed §6 outcome of the
> frozen v2 diagnostic (`2026-06-23-qc-edgar-diagnostic-v2-prereg.md`, `7f54f55`). **Report-only:**
> floor DEV_FAILED, `tools/run-floor.mjs --enforce` exit 1, holdout blinded, ledger empty — UNCHANGED.

## The result, in one line
**No affordable, mechanically-valid data path repairs the survivorship defect well enough to test the
market-neutral residual-alpha hypothesis. Lane CLOSED — no universe iteration.**

## Why (the determined STOP)
1. **Step-0 matrix:** no affordable source NATIVELY supplies delisting *returns* in the total-return
   stream (only CRSP's `DLRET`, access-locked for a solo non-academic). Cost is not the constraint;
   fidelity is.
2. **Diagnostic rung (v1):** the agreed cheap discriminator was a QC-free, NON-ALPHA source-integrity
   diagnostic. v1 (QC-native classification) proved INFEASIBLE — QC exposes no delisting reason at all.
3. **v2 (QC prices + EDGAR reasons):** built B1 (16,663 delisting filings / 5,196 issuers) + B2
   (classifier, frozen precedence). The frozen **§5.2 mappability gate** (≥85%) is the QC↔EDGAR
   bridge-quality test. Mappability is multiplicative: P(match) × P(classify). The EDGAR-side
   classification ceiling — an upper bound on combined mappability — is **68.9%** on the
   operating-company proxy, even after the in-spec Form-25 12d2-2(b)→performance completion. **68.9% <
   85% ⇒ §5.2 cannot be met regardless of the QC run.** The gate fired exactly as designed: a weak
   bridge → honest STOP, reached without burning the operator's QC effort.

## What this is and is NOT
- It IS a clean, pre-committed negative: the bridge required to run the test honestly is not reachable
  under the frozen ruleset at acceptable cost/fidelity.
- It is NOT "there is provably no edge." The market-neutral long-short hypothesis (and the orthogonal
  SESTM news lane) remain *untested* — they were never reachable with valid keyless/affordable data.
- A re-frozen **v3** (Form-25-paragraph reason classification) might clear §5.2, but using a signal held
  at freeze time *because the frozen test failed* is post-hoc respecification — distinct from the v1→v2
  infeasibility supersession. v3 is an **eyes-open NEW program the operator may elect cold**, with the
  goalpost-moving critique on record (see `2026-06-23-v2-mappability-ceiling.md`). Not pursued here.

## Base rate (context for the close)
Seventh independent negative in this research program: advisor-v1, Plan 4, WS4 fundamental_value, WS3D
lazy_prices Reading-C, the 30-name floor, the T0.2b broad-universe screen (INCONCLUSIVE), and now this
determined §5.2 STOP. The disciplined and the wise reads converge: there is no cheaply-accessible,
survivorship-valid residual-alpha edge reachable in this lane. That convergence — proven, not assumed —
is the finding.

## Genuinely-open (separate decisions, NOT continuations of this lane)
- **SESTM news** (orthogonal signal) — needs a news fixture + keys; its own EV comparison and prereg.
- **v3** — only if the operator elects it cold, accepting the post-hoc critique.

## Disposition
Lane CLOSED. Branch `exec/qc-source-integrity-diagnostic-prereg` holds the full frozen record (v1
`f863c56`, v2 `7f54f55`) + the B1/B2 build + this closeout; NOT pushed; main untouched `697ee47`.
