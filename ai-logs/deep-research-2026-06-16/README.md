# Deep-research outputs — 2026-06-16

Persisted from `/deep-research` workflow runs investigating how to advance the
advisor beyond the closed family-reweighting lane (see floor result `DEV_FAILED`).
All three JSONs share the schema: `result.{summary, findings[], caveats, openQuestions, refuted[], sources[], stats}`.
Each finding carries a 3-vote adversarial `vote` and a `confidence`. Read with `jq` or `python -m json.tool`.

| File | Run ID | What it answers |
|---|---|---|
| `COMBINED-SYNTHESIS.md` | — | The merged, advisor-reviewed conclusion across all three runs. **Start here.** |
| `orthogonal-signals-survey.json` | `wf_1b968504` | 4-seam survey: which validated methods add ORTHOGONAL info vs merely recombine price families. Headline: SESTM news NLP; cheapest no-data step = timely-price value+momentum; LLM trading agents fail post-cutoff (leakage). |
| `sestm-net-of-cost.json` | `wf_ec516ac1` | SESTM net-of-cost feasibility for a small/medium book. Verdict: **gross-only** for a large-cap price-only universe; residual edge is microcap-concentrated. |
| `validation-rigor-seam3-4.json` | `wf_4abf6aa9` | Deflation/overfitting guards to bolt onto `backtest/`: Deflated Sharpe, MinBTL, Harvey-Liu-Zhu t>3.0, purging/embargo, PBO-via-CSCV. The gate any future signal must clear. |

## Caveats on provenance
- Both follow-up runs (`sestm`, `validation`) hit a severe API-500 storm during their **verify** phase.
  In each `result.refuted[]`, distinguish genuine adversarial refutes (`vote: "0-3"`/`"1-2"`) from
  abstention casualties of the outage (`"0-0"`/`"1-0"` — verifiers died, claim not actually disproven).
  The abstention casualties (notably Chen-Velikov "Anomaly Zoo") *reinforce* the verdicts.
- Resolved conclusions also recorded in project memory: `deep-research-orthogonal-signals.md`.
