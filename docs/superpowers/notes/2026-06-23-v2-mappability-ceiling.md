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

## The honest complication (a v2-ruleset artifact, not pure data limitation)
The 1,757 (a)-paragraph unknowns are dominated by **mergers (a)(3)** + redemptions — which the Form 25
paragraph identifies AUTHORITATIVELY. v2's ruleset invoked the Form 25 paragraph ONLY for performance
(b), classifying acquisitions via the weaker 8-K-2.01-in-±90d / DEFM14A-365d heuristic — so it left
authoritative (a)(3) acquisition signal on the table. Using Form-25 (a)(3)→acquisition is a **freeze
deviation** under §4.4c (acquisition is defined there as 8-K-2.01/DEFM14A only), so it CANNOT be applied
to the frozen v2. But it means a re-frozen **v3** (Form-25-paragraph-based reason classification:
(a)(3)→acquisition, (b)→performance, (c)→voluntary) would likely clear 85% comfortably.

## Operator fork (pre-committed STOP vs method-improvement v3)
- **(A) STOP v2** — the disciplined pre-committed outcome; write the negative, close the lane. The gate
  fired as designed.
- **(B) Supersede with v3** — re-freeze using the authoritative Form-25 paragraph as the primary reason
  source (a genuine method improvement, like v1→v2, since v2 under-used a field we've now shown is
  cleaner than 8-K-item timing). v2 was never run to a verdict (no QC run, no results) → pre-verdict
  supersession is clean. RISK to weigh honestly: this is closer to "tune the ruleset until it passes"
  than v1→v2 was (v1 was infeasible; v2 is feasible-but-suboptimal) — guard against goalpost-moving.

Provenance: `ai-logs/hermes/edgar_form25_paragraph_ceiling.py`, `…/edgar_form25_format_probe.py`.
