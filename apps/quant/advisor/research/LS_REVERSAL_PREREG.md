# L/S reversal Gate-1 kill screen prereg (Decision 5, operator ruling 2026-07-06).

Status: PREREGISTRATION pin, frozen before the one-shot Gate-1 kill-screen ceremony.
This is a research gate only. PASS never asserts alpha; it only authorizes Gate-2 design.
The reserved holdout stays blinded, and nothing here authorizes sizing or capital allocation.

## Frozen field table

| Field | Frozen value |
| --- | --- |
| families | `("value", "fundamental_value", "lazy_prices")` |
| published_precost_ir | `(-0.41, -0.32, -0.40)` |
| reproduction_tolerance | `0.02` |
| borrow_rate_annual | `0.005` |
| short_rebate_annual | `0.0` |
| cost_per_turn | `0.0005` |
| hedge_cost_model | `static_ols_zero_rebalance` |
| tau_ls | `0.20` |
| pass_rule | `at_least_2_of_3_ge_tau` |
| trading_days | `252` |
| holdout_frac | `0.2` |
| panel | `apps/quant/advisor/tests/fixtures/floor_prices.csv` |
| fundamental_fixture | `apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv` |
| lazy_prices_fixture | `apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv` |
| result_path | `apps/quant/advisor/research/LS_REVERSAL_RESULT.json` |
| value_cfg | `CandidatePreReg/DEFAULT_CANDIDATE` |
| fundamental_cfg | `FundamentalCandidatePreReg/DEFAULT_FUNDAMENTAL_CANDIDATE` |
| lazy_prices_cfg | `LazyPricesCandidatePreReg/DEFAULT_LAZY_PRICES_CANDIDATE` |

## Pinned methodology hash

`3784b0998303861f8392f308324d789c032bbd64c0f56f20a3ddaaa3c2394c6b`

Computed with:

```bash
PYTHONPATH=apps/quant python -c "from advisor.research.ls_reversal_prereg import DEFAULT_LS_REVERSAL, ls_reversal_hash; print(ls_reversal_hash(DEFAULT_LS_REVERSAL))"
```

This is config hashing only and touches no returns data.

## Kill rule

PASS iff >=2 of 3 families have `postcost_reversed_ir >= tau_LS 0.20`; otherwise CLOSED.
Before that decision, the reproduction tripwire compares each pre-cost IR against the
published values within +-0.02:

| Family | Published pre-cost IR |
| --- | ---: |
| value | `-0.41` |
| fundamental_value | `-0.32` |
| lazy_prices | `-0.40` |

If any family misses the reproduction tripwire, the verdict is ABORT and post-cost outputs
are suppressed. PASS never asserts alpha; it only authorizes Gate-2 design.

## One-shot semantics

The CLI writes `LS_REVERSAL_RESULT.json` at the frozen `result_path` and REFUSES to rerun
on a recorded PASS/CLOSED forever. After ABORT, a rerun requires
`--rerun-after-abort "<operator reason>"`, and that reason is recorded in the new result.
The result JSON is committed by the operator as part of the record ceremony.

## Freeze clause

any change = new prereg filename, and the original outcome stands.
