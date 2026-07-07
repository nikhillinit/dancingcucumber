from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LongShortReversalPreReg:
    """Gate-1 kill screen for the Decision-5 L/S reversal lane — a SEPARATE hash
    surface from PreRegConfig (floor-frozen), CandidatePreReg (578cce4b…) and the
    reading surfaces. Freezes the reversed-book cost overlay and kill rule only.
    PASS authorizes designing Gate 2; it NEVER asserts alpha (in-sample screen).

    Revisable until LS_REVERSAL_PREREG.md pins its hash; frozen thereafter — any
    change is a NEW prereg filename and the original outcome stands."""
    families: tuple[str, ...] = ("value", "fundamental_value", "lazy_prices")
    published_precost_ir: tuple[float, ...] = (-0.41, -0.32, -0.40)
    reproduction_tolerance: float = 0.02
    borrow_rate_annual: float = 0.005
    short_rebate_annual: float = 0.0
    cost_per_turn: float = 0.0005          # mirrors floor PreRegConfig.cost_per_turn
    hedge_cost_model: str = "static_ols_zero_rebalance"
    tau_ls: float = 0.20
    pass_rule: str = "at_least_2_of_3_ge_tau"
    trading_days: int = 252
    holdout_frac: float = 0.2              # dev folds only; holdout stays blinded
    panel: str = "apps/quant/advisor/tests/fixtures/floor_prices.csv"
    fundamental_fixture: str = "apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv"
    lazy_prices_fixture: str = "apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv"
    result_path: str = "apps/quant/advisor/research/LS_REVERSAL_RESULT.json"  # write-once lock
    value_cfg: str = "CandidatePreReg/DEFAULT_CANDIDATE"
    fundamental_cfg: str = "FundamentalCandidatePreReg/DEFAULT_FUNDAMENTAL_CANDIDATE"
    lazy_prices_cfg: str = "LazyPricesCandidatePreReg/DEFAULT_LAZY_PRICES_CANDIDATE"


DEFAULT_LS_REVERSAL = LongShortReversalPreReg()


def ls_reversal_hash(cfg: LongShortReversalPreReg) -> str:
    """Methodology-only id (no input bytes)."""
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()


def ls_reversal_run_hash(cfg: LongShortReversalPreReg, panel_path, *fixture_paths) -> str:
    """Run id: canonical config JSON THEN panel bytes THEN each fixture's bytes, in
    order. Binding input bytes detects a panel/fixture swap between freeze and run."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(panel_path).read_bytes())
    for fp in fixture_paths:
        h.update(Path(fp).read_bytes())
    return h.hexdigest()
