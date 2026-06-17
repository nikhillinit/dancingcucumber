from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidatePreReg:
    """Immutable, pre-registered methodology for the candidate bench. A NEW hash
    surface — deliberately NOT PreRegConfig (whose hash 1ad2ed4a… is floor-frozen).
    Outcome thresholds live in the floor gates; this freezes only the search."""
    # --- candidate-specific (the falsifiable choices) ---
    families: tuple[str, ...] = ("value", "momentum")
    value_skip: int = 126                 # skip recent 6mo -> excludes momentum window
    value_lookback: int = 270             # ~13mo->6mo formation; FIXTURE-FEASIBLE (< ~325)
    orthogonality_tau: float = 0.40       # max |Pearson corr| of value vs LM / MR on dev
    declared_trials_N: int = 45           # DSR multiple-testing N (>= MinBTL budget 45)
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


DEFAULT_CANDIDATE = CandidatePreReg()


def candidate_hash(cfg: CandidatePreReg) -> str:
    """SHA-256 over the canonical config JSON (no fixture bytes — the bench reuses
    the floor's fixture, whose hash is recorded separately in CANDIDATE_PREREG.md).
    This is the methodology-only id; it is NOT the holdout-unlock key."""
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()


def candidate_run_hash(cfg: CandidatePreReg, fixture_path) -> str:
    """Holdout-unlock key (Amendment F2): byte-exact mirror of prereg.config_hash —
    canonical config JSON THEN fixture bytes. Including fixture bytes is what makes
    unlocking the shared reserved tail honest (it detects a fixture swap), matching
    the floor's own hardening (tools/floor_data_check.py). Task 8 passes THIS as the
    prereg_hash; never the fixture-blind candidate_hash()."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(fixture_path).read_bytes())
    return h.hexdigest()
