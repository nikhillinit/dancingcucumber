from pathlib import Path

import pytest

from advisor.backtest.prereg import PreRegConfig, DEFAULT_CONFIG, config_hash


def test_default_config_is_frozen_and_has_required_fields():
    cfg = DEFAULT_CONFIG
    assert cfg.folds >= 2 and cfg.embargo >= 0
    assert cfg.margin >= 0.0          # negative margin forbidden by rail
    assert cfg.primary_metric == "book_sharpe"
    assert cfg.families == ("momentum", "trend")
    assert cfg.added_families == ("mean_reversion", "breakout", "long_momentum")
    assert cfg.warmup == 200
    with pytest.raises(Exception):   # frozen dataclass
        cfg.margin = -1.0


def test_negative_margin_rejected_at_construction():
    with pytest.raises(ValueError):
        PreRegConfig(margin=-0.01)


def test_config_hash_is_stable_and_sensitive(tmp_path: Path):
    fixture = tmp_path / "f.csv"
    fixture.write_text("a,b\n1,2\n")
    h1 = config_hash(DEFAULT_CONFIG, fixture)
    h2 = config_hash(DEFAULT_CONFIG, fixture)
    assert h1 == h2 and len(h1) == 64
    fixture.write_text("a,b\n1,3\n")          # fixture change -> hash change
    assert config_hash(DEFAULT_CONFIG, fixture) != h1
    changed = PreRegConfig(margin=0.1)         # config change -> hash change
    assert config_hash(changed, fixture) != h2
