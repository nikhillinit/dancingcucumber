from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CandidateValidationPreReg:
    """Candidate's pre-registered DSR params. Own hash surface (Amendment F1).
    Field names mirror ValidationPreReg so validation_report reads them unchanged."""
    psr_benchmark_sr: float = 0.0
    dsr_pass: float = 0.95
    tstat_hurdle: float = 3.0
    minbtl_max_trials: int = 45
    declared_trials_N: int = 45          # candidate trial count; secondary run bumps this (rail #4)
    declared_var_sr: float = 1e-4        # CALIBRATE (see T8) or justify reuse as >= measured
    effective_n_method: str = "pca"
    effective_n_floor_is_declared: bool = True


DEFAULT_CANDIDATE_VALIDATION = CandidateValidationPreReg()


def candidate_validation_hash(cfg: CandidateValidationPreReg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()
