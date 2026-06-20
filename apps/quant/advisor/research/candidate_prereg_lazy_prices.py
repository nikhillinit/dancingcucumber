from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LazyPricesCandidatePreReg:
    """Reading-C candidate methodology — a SEPARATE hash surface from CandidatePreReg
    (578cce4b…), FundamentalCandidatePreReg (Reading B), and PreRegConfig (1ad2ed4a…).
    Stays revisable until READING_C_PREREG.md (D7) pins its hash and a run is recorded."""
    # --- candidate-specific (the falsifiable choices) ---
    families: tuple[str, ...] = ("lazy_prices", "momentum")
    similarity_metric: str = "cosine_tfidf"      # YoY same-form full-document cosine
    fixture_source: str = "SEC_EDGAR_FILING_TEXT"
    reporting_lag_days: int = 0                   # filing text is public at acceptance
    orthogonality_tau: float = 0.40              # max |corr| lazy_prices vs momentum on dev
    declared_trials_N: int = 45                  # VESTIGIAL — live count lives on the
                                                 # validation surface (frozen-hash discipline)
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


DEFAULT_LAZY_PRICES_CANDIDATE = LazyPricesCandidatePreReg()


def lazy_prices_candidate_hash(cfg: LazyPricesCandidatePreReg) -> str:
    """Methodology-only id (no fixture bytes). NOT the holdout-unlock key."""
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    ).hexdigest()


def lazy_prices_candidate_run_hash(cfg: LazyPricesCandidatePreReg, fixture_path) -> str:
    """Holdout-unlock key: canonical config JSON THEN fixture bytes (byte-exact mirror of
    fundamental_candidate_run_hash). Binding the fixture bytes detects a fixture swap."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(fixture_path).read_bytes())
    return h.hexdigest()
