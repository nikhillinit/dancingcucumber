# SESTM Phase-0 corpus capability matrix (report-only)

Frozen criteria: docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md
Freeze commit: d420af72518096a332ac85ee70e4fd104db7e972 | blob SHA: a42cc3dd643477724bcb8a8ab34c2eea0464198d
Sessions used: 1 of 1 cap

## Sources added during evaluation

- Common Crawl News (CC-NEWS): added as a keyless/free broad-news corpus whose documented archive properties can be desk-checked under the no-network cap [doc].

| Source | Cost | Timestamps | Ticker mappability | Breadth | History | Reproducibility | Corpus suitability | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EDGAR 8-K / material-event filings | PASS: SEC EDGAR is keyless/free with declared User-Agent discipline (docs/superpowers/plans/2026-06-23-qc-edgar-diagnostic-v2-prereg.md; apps/quant/advisor/source_integrity/edgar.py). | PASS: EDGAR submissions carry acceptance timestamps usable as as-of keys (apps/quant/advisor/data/filing_text_fetch.py; apps/quant/advisor/tests/test_edgar_xbrl_fetch.py). | FAIL: B1 proves 16,663 filings / 5,196 issuers, not >=90% current company_tickers resolution (docs/superpowers/notes/2026-06-23-v2-STOP-closeout.md). | FAIL: B1 event corpus does not prove median >=4 documents/name/year; ambiguity resolves FAIL (docs/superpowers/notes/2026-06-23-v2-STOP-closeout.md; docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md). | PASS: B1/frozen v2 scope covers 2015-2023 (docs/superpowers/plans/2026-06-23-qc-edgar-diagnostic-v2-prereg.md). | PASS: EDGAR accessions are stable document identifiers (apps/quant/advisor/source_integrity/edgar.py; docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md). | FAIL: 8-K bodies are issuer self-disclosures, explicitly disallowed (docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md). | STOP: fails ticker mappability, breadth, and corpus suitability. |
| EDGAR full-text search (efts.sec.gov) | PASS: EDGAR full-text/daily index is keyless/free with SEC fair-access mechanics (docs/superpowers/plans/2026-06-23-qc-edgar-diagnostic-v2-prereg.md; apps/quant/advisor/source_integrity/edgar.py). | PASS: EDGAR filing records expose accepted/filed as-of fields (apps/quant/advisor/data/filing_text_fetch.py; apps/quant/advisor/data/edgar_xbrl_fetch.py). | FAIL: full-search corpus has no in-repo proof of >=90% current company_tickers mapping; unverifiable within cap (docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md). | FAIL: no in-repo proof of >=2,000 mappable tickers with median >=4 docs/name/year; unverifiable within cap (docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md). | PASS: repo EDGAR fixture/fetch plans target 2015-2023 history (apps/quant/advisor/data/filing_text_fetch.py; docs/superpowers/plans/2026-06-23-qc-edgar-diagnostic-v2-prereg.md). | PASS: accession-based EDGAR documents match the frozen-fixture pattern (apps/quant/advisor/data/filing_text_fetch.py; docs/superpowers/specs/2026-06-16-reading-b-fundamental-value.md). | FAIL: SEC filing text is issuer/regulatory disclosure, not third-party-authored news text (docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md). | STOP: fails ticker mappability, breadth, and corpus suitability. |
| GDELT event/news metadata | PASS: GDELT is documented as a public/keyless global event/news metadata feed [doc]. | FAIL: documented metadata/ingest times do not prove publication timestamp per article; ambiguity resolves FAIL [doc]; docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md. | FAIL: no native SEC CIK -> ticker mapping or >=90% company_tickers proof; unverifiable within cap [doc]. | FAIL: no proof of >=2,000 SEC-mappable tickers with median >=4 docs/name/year; unverifiable within cap (docs/superpowers/plans/2026-06-23-sestm-news-lane-plan.md). | PASS: GDELT 2.x is documented as continuous through the 2015-2023 window [doc]. | FAIL: article URLs/metadata are not immutable frozen document-text identifiers; unverifiable within cap [doc]. | FAIL: metadata is not the third-party-authored company news text corpus required for SESTM scoring [doc]; docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md. | STOP: fails timestamps, ticker mappability, breadth, reproducibility, and corpus suitability. |
| Common Crawl News (CC-NEWS) | PASS: Common Crawl News is documented as a free/keyless public web-news crawl archive [doc]. | FAIL: crawl timestamp is not a publication timestamp precise enough for trade-date assignment [doc]; docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md. | FAIL: no native SEC CIK -> ticker mapping or >=90% company_tickers proof; unverifiable within cap [doc]. | FAIL: no proof of >=2,000 SEC-mappable tickers with median >=4 docs/name/year; unverifiable within cap [doc]. | FAIL: CC-NEWS does not document continuous 2015-2023 coverage; 2015 is unsupported within cap [doc]. | PASS: WARC records can be frozen by crawl file/offset/digest under the fixture-SHA pattern [doc]; apps/quant/advisor/data/filing_text_fetch.py. | FAIL: broad crawl text is not proven to be company-about news with ticker-level mapping; ambiguity resolves FAIL (docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md). | STOP: fails timestamps, ticker mappability, breadth, history, and corpus suitability. |

## Verdict

Pre-committed decision rule:

> - Any source meeting ALL seven -> Phase-0 PASS (name the source; PASS != alpha).
> - No source meeting all seven -> STOP. No criterion may be relaxed post-freeze;
>   a near-miss is a STOP (the v2 mappability precedent, 68.9% < 85%, binds).
> - Budget cap: <= 1 session, $0 spend. Phase-0 is a desk-check-grade gate — its value
>   is the frozen public record, not discovery; the real kill lives in Phase-1's
>   net-OOS-sign gate. Cap exhaustion without a verdict -> STOP.

STOP. No evaluated source meets all seven frozen criteria.

Binding failures:
- EDGAR 8-K / material-event filings: ticker mappability not proven, breadth not proven, and corpus suitability fails because issuer self-disclosure is explicitly disallowed.
- EDGAR full-text search: ticker mappability not proven, breadth not proven, and corpus suitability fails because SEC filing text is not third-party-authored news text.
- GDELT event/news metadata: timestamps, ticker mappability, breadth, reproducibility, and corpus suitability fail under the cap.
- Common Crawl News (CC-NEWS): timestamps, ticker mappability, breadth, history, and corpus suitability fail under the cap.

Binding failures are structural (corpus type / mappability / timestamps), not
measurement shortfalls — no evaluated source failed on cost.
