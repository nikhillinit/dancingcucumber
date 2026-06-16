from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ValidationPreReg:
    """Immutable, pre-registered VALIDATION-GATE params. Separate hash surface from
    PreRegConfig so the recorded floor hash (PREREG.md) is never disturbed."""
    psr_benchmark_sr: float = 0.0         # SR* threshold the deflated bar must clear
    dsr_pass: float = 0.95               # Deflated Sharpe pass bar
    tstat_hurdle: float = 3.0            # Harvey-Liu-Zhu factor/signal selection hurdle
    minbtl_max_trials: int = 45          # MinBTL throughput budget on ~5yr daily sample
    declared_trials_N: int = 45          # dominant multiple-testing N (pre-registered,
                                         # = budget ceiling; conservative, can only deflate harder)
    effective_n_method: str = "pca"      # PCA participation-ratio over family return series
    effective_n_floor_is_declared: bool = True  # effective-N may never lower N below declared
    declared_var_sr: float = 4e-4        # PRE-REGISTERED cross-trial Sharpe dispersion V[{SR_n}].
                                         # A DECLARED CONSTANT, not estimated live (the harness has
                                         # no stored 45-trial book; 2 report-level series can't
                                         # estimate it). SR0 ∝ sqrt(var_sr) -> this is the gate's
                                         # dominant leniency knob; HIGHER = STRICTER. Interim 4e-4
                                         # errs high; Task 9 calibrates it from realized trial Sharpes.


DEFAULT_VALIDATION = ValidationPreReg()


def validation_hash(cfg: ValidationPreReg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()
