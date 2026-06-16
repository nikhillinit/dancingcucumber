from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PreRegConfig:
    """Immutable, pre-registered floor hyperparameters. Any change re-hashes."""
    window: tuple[str, str] = ("2015-01-01", "2023-12-31")
    folds: int = 5
    embargo: int = 5
    warmup: int = 200                     # max family lookback (trend long MA)
    families: tuple[str, ...] = ("momentum", "trend")
    added_families: tuple[str, ...] = ("mean_reversion", "breakout", "long_momentum")
    primary_metric: str = "book_sharpe"
    margin: float = 0.0                   # SPY margin; >= 0 (rail)
    pct_clip: tuple[float, float] = (0.05, 0.95)
    weight_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    train_lift_threshold: float = 0.05    # Rule B deviation + dev total-lift bar
    max_asset_weight: float = 0.20        # 2/N region at N>=10; caps a single name
    gross_cap: float = 1.0
    turnover_cap: float = 0.20            # per-rebalance one-way turnover ceiling
    cost_per_turn: float = 0.0005
    rebalance: str = "daily"
    bootstrap_block: int = 21             # ~1 trading month
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 12345           # fixed: determinism (no Math.random rail)
    dev_lcb: float = 0.90                 # one-sided dev CI level
    final_lcb: float = 0.95               # one-sided holdout CI level
    min_universe_formal: int = 20
    min_universe_floor: int = 12

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError("margin must be >= 0 (negative margin is forbidden)")


DEFAULT_CONFIG = PreRegConfig()


def config_hash(cfg: PreRegConfig, fixture_path: Path) -> str:
    """SHA-256 over the canonical config JSON + the raw fixture bytes."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(Path(fixture_path).read_bytes())
    return h.hexdigest()
