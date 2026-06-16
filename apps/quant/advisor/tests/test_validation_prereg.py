from dataclasses import asdict
from pathlib import Path

import pytest

from advisor.backtest.prereg import DEFAULT_CONFIG
from advisor.backtest.validation_prereg import (
    DEFAULT_VALIDATION, ValidationPreReg, validation_hash,
)


def test_floor_config_fields_unchanged():
    """Guard: adding validation params must NOT add fields to the frozen floor
    config, which is SHA-hashed into immutable PREREG.md (hash 1ad2ed4a...)."""
    assert set(asdict(DEFAULT_CONFIG).keys()) == {
        "window", "folds", "embargo", "warmup", "families", "added_families",
        "primary_metric", "margin", "pct_clip", "weight_grid",
        "train_lift_threshold", "max_asset_weight", "gross_cap", "turnover_cap",
        "cost_per_turn", "rebalance", "bootstrap_block", "bootstrap_draws",
        "bootstrap_seed", "dev_lcb", "final_lcb",
        "min_universe_formal", "min_universe_floor",
    }


def test_validation_prereg_frozen_and_has_fields():
    v = DEFAULT_VALIDATION
    assert v.dsr_pass == 0.95
    assert v.tstat_hurdle == 3.0
    assert v.minbtl_max_trials == 45
    assert v.declared_trials_N == 45          # conservative: N = the budget ceiling
    assert v.effective_n_method == "pca"
    assert v.psr_benchmark_sr == 0.0
    assert v.declared_var_sr > 0              # pre-registered cross-trial Sharpe dispersion (calibrated, Task 9)
    with pytest.raises(Exception):            # frozen dataclass
        v.dsr_pass = 0.5


def test_validation_hash_stable_and_sensitive():
    h1 = validation_hash(DEFAULT_VALIDATION)
    assert h1 == validation_hash(DEFAULT_VALIDATION) and len(h1) == 64
    assert validation_hash(ValidationPreReg(dsr_pass=0.99)) != h1
