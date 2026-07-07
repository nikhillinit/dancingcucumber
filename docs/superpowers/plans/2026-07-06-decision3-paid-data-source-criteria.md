# Decision-3 Paid-Data Source Criteria

Boundary: This artifact is no-alpha due diligence. It does not open a signal lane, does not authorize a prereg, and does not touch returns, CAR, Sharpe, IR, floor, holdout, DSR, candidate pipeline, or effect-size metrics. PASS means "eligible for a later operator decision," not "eligible to test alpha."

## Frozen Question

Can any legally usable paid source mechanically settle the survivorship/data-fidelity confound from the broad-universe residual screen by providing delisting-aware, point-in-time equity data with native or explicitly documented delisting-return handling?

## Frozen Ceilings (operator ruling 2026-07-06)

- Scoping budget: $0 recurring, $250 one-time max, spent only on sample files, trial access, or documentation access. No production data subscription may be purchased inside the scoping task.
- Qualifying acquisition ceiling: a source PASSes cost/access only if <= $1,000 one-time plus <= $100/month recurring, cancellable without a multi-year lock, and licensed for this research use. Anything above returns to the operator as an explicit "enterprise-data purchase?" decision, not a PASS.
- Integration-time ceiling: <= 3 focused dev-days for sample proof plus loader integration; the scoping matrix itself is one session.
- These ceilings do not move to make a vendor fit.

## Delta Scope (Step-0 is binding)

- Consume `docs/superpowers/plans/2026-06-23-qc-source-integrity-diagnostic-prereg.md` §1 as settled: QC / Sharadar / Norgate delisting-return FAILs carry over by citation and are re-examined only on documented vendor change since 2026-06-23.
- Evaluate live: CRSP/WRDS access and cost at the frozen acquisition ceiling only.
- Disregard: S&P Capital IQ and other enterprise platforms (cannot clear the ceiling by construction).
- Honest prior, predeclared: at these ceilings the expected verdict is STOP unless a vendor can now prove native, legally usable delisting-return fidelity.

## Required Source Criteria

Every candidate source must be evaluated on these fields:

| Criterion | PASS requirement | STOP condition |
| --- | --- | --- |
| Legal use | License permits the intended solo-dev research use and a private normalized fixture or reproducible regeneration path. | License unclear, redistribution/regeneration forbidden for the needed workflow, or terms require a production/professional setup outside the operator budget. |
| Sample proof | Vendor provides sample file, trial, schema documentation, or official support reply proving the required fields. | Marketing page only, no field-level proof, or support refuses sample proof. |
| Delisted names retained | Historical universe keeps delisted securities visible after delisting. | Delisted names are absent, manually patched, or not auditable. |
| Delisting returns | Total-return stream includes delisting returns natively, or the vendor exposes a documented delisting-return field sufficient to compute them without self-imposed terminal losses. | Delisting returns are missing, vendor-caveated, self-built, or only inferable from disappearance. |
| Point-in-time membership | Tradable universe membership is knowable as of each historical date. | Only current membership, current symbol map, or restated history. |
| Small/mid coverage | Source reaches below the current 30 mega-cap floor enough to test the known size/survivorship confound. | Mega-cap-only or coverage unclear. |
| Corporate actions | Splits, dividends, ticker changes, mergers, and delistings have documented adjustment policy. | Adjustment policy absent or not PIT-auditable. |
| Cost and access | <= $1,000 one-time plus <= $100/month, cancellable without multi-year lock, licensed for this research use (Frozen Ceilings above). | Above ceiling, enterprise-sales-only pricing, or no public/sample-backed price evidence. |
| Integration estimate | One engineer can build a fixture/provenance loader without changing the frozen advisor floor or holdout. | Requires broad architecture rewrite, production credentials in tests, or touching frozen surfaces. |

## Decision Rule

PASS if at least one source clears every required criterion with official documentation, a sample, or a written vendor answer. STOP if no source clears every required criterion, if the session/budget cap is exhausted, or if evidence would require inspecting alpha performance.

## Allowed Evidence

- Official vendor documentation.
- Official sample files or schemas.
- Written vendor support answers.
- Existing repo notes cited by path.
- Cost quotes or public pricing, if available without purchase.

## Forbidden Evidence

- Any backtest output.
- Any return, CAR, Sharpe, IR, DSR, holdout, floor, or effect-size metric.
- Any self-filled terminal-loss assumption used as a substitute for native delisting-return proof.
- Any data purchase without a separate operator decision.
