from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FundamentalCandidatePreReg:
    """Reading-B candidate methodology — a SEPARATE hash surface from CandidatePreReg
    (578cce4b…) and from PreRegConfig (floor-frozen 1ad2ed4a…). Freezes the fundamental
    value+momentum search only; outcome thresholds live in the floor gates.

    Not yet recorded: this surface stays revisable until READING_B_PREREG.md (T8) pins
    its hash and a run is recorded against it."""
    # --- candidate-specific (the falsifiable choices) ---
    families: tuple[str, ...] = ("fundamental_value", "momentum")
    value_metric: str = "book_to_price"    # BVPS / timely price; orthogonal to price families
    fixture_source: str = "SEC_EDGAR_XBRL"
    reporting_lag_days: int = 90            # availability lag (mirrors data.provider.REPORTING_LAG_DAYS)
    orthogonality_tau: float = 0.40         # max |Pearson corr| fundamental_value vs momentum on dev
    declared_trials_N: int = 45             # VESTIGIAL — the LIVE DSR trial count is
                                            # FundamentalCandidateValidationPreReg.declared_trials_N.
                                            # Kept so a later removal can't mutate this frozen hash
                                            # (same discipline as CandidatePreReg).
    # --- inherited from the floor so the bench is faithful (mirror PreRegConfig) ---
    warmup: int = 200
    folds: int = 5
    embargo: int = 5
    margin: float = 0.0
    pct_clip: tuple[float, float] = (0.05, 0.95)
    weight_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    train_lift_threshold: float = 0.05
    max_asset_weight: float = 0.20
    gross_cap: float = 1.0
    turnover_cap: float = 0.20
    cost_per_turn: float = 0.0005
    bootstrap_block: int = 21
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 12345
    dev_lcb: float = 0.90
    final_lcb: float = 0.95
    min_universe_formal: int = 20
    min_universe_floor: int = 12


DEFAULT_FUNDAMENTAL_CANDIDATE = FundamentalCandidatePreReg()


def fundamental_candidate_hash(cfg: FundamentalCandidatePreReg) -> str:
    """Methodology-only id (no fixture bytes). NOT the holdout-unlock key."""
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()


def fundamental_candidate_run_hash(cfg: FundamentalCandidatePreReg, fixture_path) -> str:
    """Holdout-unlock key: canonical config JSON THEN fixture bytes (byte-exact mirror
    of candidate_run_hash). Binding the EDGAR fixture bytes detects a fixture swap."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(fixture_path).read_bytes())
    return h.hexdigest()
