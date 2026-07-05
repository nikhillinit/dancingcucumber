# Program review memo — alpha-lane record and go-forward (2026-07-04)

## 1. The record (eight results)

| # | Lane | Verdict | Key number |
|---|------|---------|------------|
| 1 | Advisor v1 equal-weight price ensemble (2026-06-14) | FAILS FLOOR | Sharpe 0.32 < SPY 0.85 |
| 2 | Plan 4 v2 continuous long-flat ensemble (2026-06-16) | DEV_FAILED | ens 0.73 < best family 0.83 |
| 3 | Lane B value+momentum candidate (2026-06-17) | DEV_FAILED (power-limited) | ens 0.662 < best 0.668 |
| 4 | WS4 Reading B fundamental_value+momentum (2026-06-19) | DEV_FAILED (not power-limited) | ens 0.557 < momentum 0.665 < SPY 0.752 |
| 5 | WS3D Reading C lazy_prices (2026-06-21) | DEV_FAILED | 0/4 folds; ens 0.598 < momentum 0.681 |
| 6 | Universe-change broad screen T0.2b (2026-06-23) | INCONCLUSIVE | GREEN non-discriminating; survivorship-confounded |
| 7 | QC+EDGAR source-integrity diagnostic v2 (2026-06-23) | STOP | mappability 68.9% < 85% frozen gate |
| 8 | SESTM Phase-0 corpus matrix (2026-07-04) | STOP | no keyless corpus clears the 7 frozen criteria (binding failures structural) |

Structural finding: blend non-additivity on the 30-mega-cap floor (Readings B and C
fail identically) and no positive long-only residual alpha on the floor (corrected
info-ratio screen: value -0.41, fundamental_value -0.32, lazy_prices -0.40,
momentum -0.02).

## 2. What the record rules out

The record rules out more long-only price/fundamental/filing-text blends on the
30-mega-cap floor as a research continuation. Reading B and Reading C failed the same
additivity pattern, and the blend-futility note shows that highly correlated book
returns made extra blending unable to clear the floor; the corrected information-ratio
screen stayed negative for value, fundamental_value, lazy_prices, and momentum
(`docs/superpowers/notes/2026-06-21-blend-futility-residual-alpha.md`). The fixed
record above keeps that outcome visible in rows 4 and 5, and the floor remains
DEV_FAILED.

The record also rules out the market-neutral residual-alpha lane through free
delisting-aware data. The v2 closeout says the required QC-to-EDGAR bridge failed the
frozen mappability gate at 68.9% against an 85% threshold, so the test could not be
run honestly under the frozen rules and affordable/keyless data constraints
(`docs/superpowers/notes/2026-06-23-v2-STOP-closeout.md`;
`docs/superpowers/plans/2026-06-23-sestm-news-lane-plan.md`). That STOP is a data-path
closure, not a new signal result.

The record finally rules out SESTM-as-published on any known keyless corpus. The
Phase-0 matrix applied the frozen seven criteria to EDGAR 8-K/material-event filings,
EDGAR full-text search, GDELT event/news metadata, and Common Crawl News; no source
met all criteria, with binding failures in corpus type, ticker mappability, timestamps,
history, or reproducibility depending on the source
(`docs/superpowers/notes/2026-07-04-sestm-phase0-corpus-matrix.md`;
`docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md`).

## 3. Reframes evaluated

(a) Validation-harness-as-asset. What it would take: package the prereg/floor/holdout
discipline, frozen-threshold pattern, DEV_FAILED reporting, and source-integrity
matrix style as a reusable research-governance template rather than another signal
lane. What it would cost: documentation, example fixtures, and maintenance of the
gate semantics, without relaxing any advisor-gate or holdout boundary. What it would
prove: whether the durable value is the falsification machinery itself — a way to
avoid narrative-led research drift — not a claim that any tested family clears the
floor.

(b) Risk/diagnostics advisor product. What it would take: treat report-only Sortino,
drawdown, concentration, and related diagnostics as the value surface, anchored to the
diagnostics work merged on 2026-07-04 and explicitly separated from any signal claim.
What it would cost: interface work, user-facing explanation, and tests that prove the
diagnostics are deterministic and report-only. What it would prove: whether users
value disciplined portfolio diagnostics and decision support even while the research
floor remains DEV_FAILED.

(c) Paid-data universe lever. What it would take: an explicit dollar-budget decision
for delisting-aware point-in-time equity data, such as CRSP-grade returns and
delisting fields, before reopening the universe hypothesis. What it would cost:
license money, integration time, and a fresh prereg that states exactly what the paid
data is allowed to test. What it would prove: whether the previous market-neutral and
broader-universe blockers were data-fidelity blockers, rather than reusable evidence
for another keyless-data attempt.

(d) Self-disclosure tone. Phase-0 criterion 7 surfaced issuer text — 8-K bodies and
other issuer-authored disclosures — as a separately named NEW eyes-open hypothesis.
It is explicitly NOT a Phase-1 input for SESTM and NOT screen-justified by the Phase-0
matrix, because the SESTM lane required third-party-authored news text and failed
issuer self-disclosures on corpus suitability. What it would take: a cold prereg under
a new filename and a new decision rule. What it would cost: accepting that this is not
SESTM-as-published and may share the same event-cadence and mappability constraints.
What it would prove: only whether issuer-disclosure tone deserves its own research
lane after an operator elects it cold.

## 4. Recommendation and operator decision points

No further signal lanes open without an operator decision on the reframes above. The floor is DEV_FAILED and stays that way. A Phase-0 STOP does not authorize a Phase-1
SESTM run, and no criterion may be relaxed to manufacture a continuation.

Decision points:

1. Choose whether to turn the validation harness into the primary artifact. Prerequisite:
   define the intended consumer and the minimum reusable package.
2. Choose whether to pursue the risk/diagnostics advisor surface. Prerequisite: define
   the report-only outputs and acceptance tests, with no signal-floor claim.
3. Choose whether to buy paid delisting-aware point-in-time data. Prerequisite: make a
   dollar budget decision before any renewed universe test.
4. Choose whether to open self-disclosure tone as a cold new hypothesis. Prerequisite:
   write a new prereg that acknowledges it is not SESTM Phase-1 and not justified by
   the Phase-0 screen.

Recommended default ordering: (1) validation harness, then (2) risk/diagnostics advisor,
then (3) paid-data universe lever only if budget appetite exists, with (4)
self-disclosure tone deferred until the operator deliberately elects a new hypothesis.
Rationale: the first two reuse proven governance and deterministic diagnostics while
preserving the DEV_FAILED research record; the latter two require a new budget or a
new hypothesis boundary.
