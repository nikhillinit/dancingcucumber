from dataclasses import replace

import pytest

from advisor.research.ls_reversal_prereg import (
    DEFAULT_LS_REVERSAL, LongShortReversalPreReg, ls_reversal_hash, ls_reversal_run_hash,
)


def test_frozen_values():
    cfg = DEFAULT_LS_REVERSAL
    assert cfg.families == ("value", "fundamental_value", "lazy_prices")
    assert cfg.published_precost_ir == (-0.41, -0.32, -0.40)
    assert cfg.reproduction_tolerance == 0.02
    assert cfg.borrow_rate_annual == 0.005
    assert cfg.short_rebate_annual == 0.0
    assert cfg.cost_per_turn == 0.0005
    assert cfg.tau_ls == 0.20
    assert cfg.pass_rule == "at_least_2_of_3_ge_tau"
    assert cfg.fundamental_fixture.endswith("edgar_xbrl_fundamentals.csv")
    assert cfg.lazy_prices_fixture.endswith("lazy_prices_similarity.csv")
    assert cfg.result_path.endswith("LS_REVERSAL_RESULT.json")


def test_hash_stable_and_sensitive():
    h0 = ls_reversal_hash(DEFAULT_LS_REVERSAL)
    assert ls_reversal_hash(DEFAULT_LS_REVERSAL) == h0
    assert ls_reversal_hash(replace(DEFAULT_LS_REVERSAL, tau_ls=0.19)) != h0


def test_run_hash_binds_panel_and_each_fixture_independently(tmp_path):
    p = tmp_path / "panel.csv"; p.write_text("p0")
    fb = tmp_path / "fundamental.csv"; fb.write_text("b0")
    fc = tmp_path / "lazy.csv"; fc.write_text("c0")
    h0 = ls_reversal_run_hash(DEFAULT_LS_REVERSAL, p, fb, fc)
    p.write_text("p1")
    h1 = ls_reversal_run_hash(DEFAULT_LS_REVERSAL, p, fb, fc)
    fb.write_text("b1")
    h2 = ls_reversal_run_hash(DEFAULT_LS_REVERSAL, p, fb, fc)
    fc.write_text("c1")
    h3 = ls_reversal_run_hash(DEFAULT_LS_REVERSAL, p, fb, fc)
    assert len({h0, h1, h2, h3}) == 4


def test_dataclass_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_LS_REVERSAL.tau_ls = 0.0


def test_pinned_hash_matches_md():
    from pathlib import Path
    md = Path("apps/quant/advisor/research/LS_REVERSAL_PREREG.md").read_text(encoding="utf-8")
    assert ls_reversal_hash(DEFAULT_LS_REVERSAL) in md
