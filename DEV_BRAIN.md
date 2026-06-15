# DEV_BRAIN — AIHedgeFund advisor

This repo builds a deterministic-first AI investment advisor. Design spec:
`docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md`. Active plans:
`docs/superpowers/plans/`.

## Non-negotiable constraints (every task inherits these)
1. **The LLM never sizes positions or sets risk limits.** Every number is computed in
   deterministic Python; the LLM only converts unstructured inputs to typed facts and explains.
2. **Trust boundary = the Pydantic `SignalBundle` seam.** Deterministic families produce signals;
   the deterministic risk manager + allocator own position size. "Graph proposes, rails dispose."
3. **Honest backtest discipline.** Price/volume walk-forward only is defensible. Fundamentals from
   yfinance are restated, not as-reported — disclose (never fabricate), lag ~90 days, snapshot forward.
   The 4-line disclosure header is mandatory on every backtest report.
4. **The `advisor-gate` is sacred.** Never weaken it to make a change pass. v1 floor: purged
   walk-forward OOS Sharpe net of costs beats SPY by the pre-registered margin AND beats the best
   single family across >=2 regimes.
5. **No fabricated data.** Missing source => signal `unavailable` and excluded, never a default value.

## Working rules
- Produce the smallest safe diff. TDD: failing test first.
- Search for existing implementations before adding new abstractions.
- Touching backtest/risk/valuation = financial risk => routes to production-financial + advisor-gate.
