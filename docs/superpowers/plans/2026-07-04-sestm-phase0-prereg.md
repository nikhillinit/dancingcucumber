# SESTM Phase-0 prereg — news-corpus capability matrix (FROZEN)

Parent plan: docs/superpowers/plans/2026-06-23-sestm-news-lane-plan.md
Frozen: 2026-07-04, BEFORE any source evaluation. Report-only. Non-alpha.
PASS unlocks nothing but a separately greenlit Phase-1 prereg. Default STOP.

## Must-have criteria (ALL required for a source to PASS)

1. **Cost:** $0 (keyless/free). Any registration-gated or paid tier = FAIL.
2. **Timestamps:** publication timestamp precise enough to assign each document to a
   trading date without look-ahead (time-of-day, or an explicit pre/post-close flag).
   Date-only granularity = FAIL unless publication is provably post-close.
3. **Ticker mappability:** >= 90% of corpus documents resolve entity/CIK -> ticker via
   SEC company_tickers.json (current snapshot). DISCLOSED LIMIT: this measures corpus
   mappability and is survivorship-biased; tradability-as-of-date is deferred to
   Phase-1 under the QC-project universe named in the parent plan.
4. **Breadth (corpus-intrinsic):** >= 2,000 distinct mappable tickers with median
   >= 4 documents per name per year across 2015-2023. DISCLOSED DIVERGENCE: this is
   deliberately NOT an index-membership test (no keyless PIT membership list exists —
   the T0.2b confound); coverage against the actual Phase-1 trading universe is
   re-measured in Phase-1.
5. **History:** continuous coverage 2015-2023 (mirrors existing fixture eras).
6. **Reproducibility:** stable immutable document identifiers (e.g., EDGAR accession
   numbers) or a documented versioning policy, so a frozen fixture with SHA can be
   built, matching the WS3C/WS3D fixture pattern.
7. **Corpus suitability:** third-party-authored news text about the company. Issuer
   self-disclosures (8-K bodies, issuer press releases) = FAIL for the SESTM-as-
   published hypothesis. If a self-disclosure corpus is the only survivor, the verdict
   is STOP for this lane; "self-disclosure tone" goes to the program review memo as a
   separately named NEW hypothesis, not a Phase-1 input.

## Decision rule (pre-committed)

- Any source meeting ALL seven -> Phase-0 PASS (name the source; PASS != alpha).
- No source meeting all seven -> STOP. No criterion may be relaxed post-freeze;
  a near-miss is a STOP (the v2 mappability precedent, 68.9% < 85%, binds).
- Budget cap: <= 1 session, $0 spend. Phase-0 is a desk-check-grade gate — its value
  is the frozen public record, not discovery; the real kill lives in Phase-1's
  net-OOS-sign gate. Cap exhaustion without a verdict -> STOP.

## Candidate source list (initial; additions are recorded in the MATRIX NOTE —
## this file never changes after the freeze commit)

EDGAR 8-K / material-event filings (via existing filing_text_fetch.py machinery and
the QC-diagnostic B1 dataset, 16,663 filings / 5,196 CIKs); EDGAR full-text search
(efts.sec.gov); GDELT event/news metadata; other keyless feeds discovered in-session
(list them in the matrix note under "Sources added during evaluation").
