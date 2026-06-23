# 2026-06-23 — v2 diagnostic: EDGAR classification-ceiling finding (report-only)

Frozen v2 prereg = `2026-06-23-qc-edgar-diagnostic-v2-prereg.md` (`7f54f55`). NON-ALPHA, report-only.

## The measurement
Mappability (frozen §5.2, bar ≥85%) is **multiplicative**: P(QC delisting matched to a Form 25) ×
P(classified | matched). B2 measured only the 2nd factor. The EDGAR-side **classification ceiling** is
therefore an **upper bound** on combined mappability (join match-rate ≤ 1) — cheap to compute agent-side,
and if it's clearly <85% the QC run cannot rescue §5.2.

- B2 classifier (frozen precedence, 8-K items 1.03/2.01/3.01 + DEFM14A): full-set mappability **48.8%**;
  on the operating-company (8-K-filer) proxy **65.9%**. Half the full-set unknowns are non-8-K filers
  (funds/ETF/foreign) the QC US-common filter removes.
- **In-spec completion** (frozen §4.4c's omitted clause: "Form 25 cites a listing-standard rule
  paragraph" → performance; mapped ONLY Rule **12d2-2(b)**): recovers **162** events → ceiling **68.9%**
  (3730/5411). Paragraph distribution among the 1,843 unknown-8-K-filers: **(a) 1757, (b) 162, (c) 76**.

## Verdict on §5.2 (as frozen): FAIL is determined
Combined mappability ≤ 68.9% < 85%. The QC universe run (B3) cannot lift it (QC-eligible ⊂ 8-K filers;
no basis for a +16pt jump). **Per §5.2/§6 the frozen v2 STOPs — the QC↔EDGAR reason bridge is too weak
under the frozen ruleset.** This is the "weak bridge → honest STOP" §5.2 was designed to produce, and it
is reached WITHOUT burning the operator's QC effort.

## VERDICT: STOP. The §5.2 gate fired; close the lane.
The frozen v2 ran and produced a determined FAIL. Per §6, the residual-alpha / market-neutral lane is
CLOSED; negative written (see `2026-06-23-v2-STOP-closeout.md`). This STOP was reached agent-side
WITHOUT burning the operator's QC run — the value of the multiplicative-ceiling read.

## Why v3 is NOT a clean supersession (goalpost-moving critique, on record)
A re-frozen v3 (Form-25-paragraph reason classification) is **not** analogous to v1→v2 — strike that
equivalence:
- **v1→v2 was forced by INFEASIBILITY** (QC has zero reason field; v1 could not measure anything — a
  test that can't run isn't a test). **v2 is feasible, RAN, and FAILED.** "The frozen test failed and I
  found a respecification that passes" is the textbook definition of what pre-registration forbids.
- **The signal was in hand at freeze time.** §4.4c explicitly invokes the Form-25 paragraph — for the
  performance branch only. I *chose* the 8-K-2.01/DEFM14A path for acquisitions at design time. Switching
  to a signal I held at freeze, *because* the test failed, is post-hoc respecification regardless of how
  authoritative the signal is.
- **"v3 would clear 85%" is unverified twice:** (1) the measurement is paragraph letter `(a)` = 1757,
  NOT `(a)(3)` — `(a)` bundles rights-expiry (a)(1), redemption (a)(2), retirement (a)(3); "mostly
  mergers" is an unmeasured inference. (2) P(join match) — B4, the stated top risk — is STILL unmeasured,
  so "v3 passes" again assumes the ≤1 factor that this whole episode was burned ignoring. (Do NOT go
  measure these to settle it — that is building the goalpost-move.)
- **"Unknowns are benign mergers" cuts AGAINST a pass, not for it.** Unknown means unknown: if any are
  performance delistings that filed neither 8-K 3.01 nor a (b) paragraph, adverse mass is UNDERCOUNTED and
  §5.3 concentration is unreliable — the exact failure §5.2 guards against. A frozen gate cannot be
  demoted to "mis-specified proxy" after it fires; freeze time was the time to argue spec.

**v3 is therefore an eyes-open NEW research program the operator may elect COLD** — with this critique on
record and against a 6-negative base rate (advisor-v1, Plan 4, WS4-B, WS3D-C, floor, INCONCLUSIVE screen,
now this §5.2 STOP). It is not tee'd up here; not pre-built; not re-frozen.

Provenance: `ai-logs/hermes/edgar_form25_paragraph_ceiling.py`, `…/edgar_form25_format_probe.py`.
