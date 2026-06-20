from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LazyPricesCandidateValidationPreReg:
    """Reading-C candidate's pre-registered DSR params — own hash surface. Carries the
    LIVE multiple-testing trial count (declared_trials_N), mirroring
    FundamentalCandidateValidationPreReg so validation_report reads the fields unchanged."""
    psr_benchmark_sr: float = 0.0
    dsr_pass: float = 0.95
    tstat_hurdle: float = 3.0
    minbtl_max_trials: int = 45
    declared_trials_N: int = 45          # LIVE candidate trial count (rail #4)
    declared_var_sr: float = 1e-4        # CALIBRATE at D7 (or justify reuse as >= measured)
    effective_n_method: str = "pca"
    effective_n_floor_is_declared: bool = True


DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION = LazyPricesCandidateValidationPreReg()


def lazy_prices_candidate_validation_hash(cfg: LazyPricesCandidateValidationPreReg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()
