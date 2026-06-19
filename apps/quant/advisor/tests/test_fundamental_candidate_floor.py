import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest

import advisor.research.candidate_floor as candidate_floor
from advisor.backtest.dev_gate import GateResult
from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from advisor.research.candidate_floor import fundamental_candidate_metrics
from advisor.research.candidate_prereg_fundamental import (
    DEFAULT_FUNDAMENTAL_CANDIDATE,
    fundamental_candidate_run_hash,
)
from advisor.research.candidate_validation_prereg_fundamental import (
    DEFAULT_FUNDAMENTAL_CANDIDATE_VALIDATION,
)
from advisor.research.fundamental_value import build_fundamental_panel


CFG = dataclasses.replace(
    DEFAULT_FUNDAMENTAL_CANDIDATE,
    warmup=20,
    folds=3,
    embargo=2,
    bootstrap_block=5,
    bootstrap_draws=25,
    min_universe_floor=4,
    min_universe_formal=8,
)
VCFG = dataclasses.replace(
    DEFAULT_FUNDAMENTAL_CANDIDATE_VALIDATION,
    declared_trials_N=10,
    minbtl_max_trials=20,
)


def _panel(n=260, k=12, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    cols = {}
    for i in range(k):
        drift = 0.0002 + i * 0.00001
        cols[f"A{i}"] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.00025, 0.008, n)))
    return pd.DataFrame(cols, index=idx)


def _records(panel):
    records = []
    available = panel.index[0].date()
    for i, asset in enumerate([c for c in panel.columns if c != "SPY"]):
        accession = f"{asset}-2020q1"
        records.append(
            EdgarXbrlRecord(
                asset=asset, cik=f"{i:010d}", accession=accession, form="10-Q",
                report_period_end=date(2019, 12, 31), filing_date=available,
                accepted_datetime=available, concept="StockholdersEquity", unit="USD",
                value=1000.0 + i * 10.0, available_asof=available, superseded_by="",
                amended_flag=False, missingness_reason="", denominator_policy="",
            )
        )
        records.append(
            EdgarXbrlRecord(
                asset=asset, cik=f"{i:010d}", accession=accession, form="10-Q",
                report_period_end=date(2019, 12, 31), filing_date=available,
                accepted_datetime=available, concept="MarketCapAnchor", unit="USD",
                value=500.0 + i * 5.0, available_asof=available, superseded_by="",
                amended_flag=False, missingness_reason="",
                denominator_policy="as_reported_shares_x_raw_close_at_avail",
            )
        )
    return records


def _fundamental_panel(panel, cfg=CFG):
    assets = [c for c in panel.columns if c != "SPY"]
    return build_fundamental_panel(_records(panel), panel, assets, warmup=cfg.warmup)


def test_fundamental_candidate_metrics_reports_schema_and_blinds_holdout():
    panel = _panel()
    m = fundamental_candidate_metrics(panel, _fundamental_panel(panel), CFG, vcfg=VCFG)
    for key in (
        "verdict", "universe", "dev", "weights", "holdout", "ensemble", "spy",
        "best_family", "margin", "passes", "validation", "power",
    ):
        assert key in m
    assert m["holdout"] is None
    assert m["passes"] is False
    assert set(m["weights"]) == set(CFG.families)


def test_fundamental_no_holdout_outputs_are_reserved_tail_invariant():
    panel = _panel()
    base = fundamental_candidate_metrics(panel, _fundamental_panel(panel), CFG, vcfg=VCFG)

    mutated = panel.copy()
    dev_end_abs = CFG.warmup + int((len(panel) - CFG.warmup) * 0.8)
    mutated.iloc[dev_end_abs:] *= 3.0
    after = fundamental_candidate_metrics(
        mutated,
        _fundamental_panel(mutated),
        CFG,
        vcfg=VCFG,
    )

    assert after["holdout"] is None
    assert after["verdict"] == base["verdict"]
    assert after["spy"] == base["spy"]
    assert after["ensemble"] == base["ensemble"]
    assert after["best_family"] == base["best_family"]
    assert after["weights"] == base["weights"]
    assert after["dev"]["fold_deltas"] == base["dev"]["fold_deltas"]


def test_fundamental_panel_tail_is_blinded_when_holdout_locked():
    panel = _panel()
    funda = _fundamental_panel(panel)
    base = fundamental_candidate_metrics(panel, funda, CFG, vcfg=VCFG)

    mutated_funda = funda.copy()
    dev_end = int(len(mutated_funda) * 0.8)
    mutated_funda.iloc[dev_end:] *= 10.0
    after = fundamental_candidate_metrics(panel, mutated_funda, CFG, vcfg=VCFG)

    assert after["holdout"] is None
    assert after["spy"] == base["spy"]
    assert after["ensemble"] == base["ensemble"]
    assert after["best_family"] == base["best_family"]
    assert after["weights"] == base["weights"]
    assert after["dev"]["fold_deltas"] == base["dev"]["fold_deltas"]


def test_fundamental_holdout_unlock_requires_verified_run_hash(monkeypatch):
    panel = _panel()
    monkeypatch.setattr(candidate_floor, "dev_gate", lambda *args, **kwargs: GateResult(True, []))
    with pytest.raises(ValueError, match="fundamental_candidate_run_hash"):
        fundamental_candidate_metrics(
            panel,
            _fundamental_panel(panel),
            CFG,
            prereg_hash="deadbeef",
            vcfg=VCFG,
            fixture_path=__file__,
        )

    # A valid hash is accepted by the unlock guard. This synthetic candidate may or may
    # not pass dev, so the assertion is only that the call does not reject the key.
    valid_hash = fundamental_candidate_run_hash(CFG, __file__)
    m = fundamental_candidate_metrics(
        panel,
        _fundamental_panel(panel),
        CFG,
        prereg_hash=valid_hash,
        vcfg=VCFG,
        fixture_path=__file__,
    )
    assert m["holdout"] is not None
    assert set(m["holdout"]) >= {
        "delta_lcb", "spy_lcb", "beats_parts", "beats_spy",
        "ensemble_sharpe", "spy_sharpe", "best_family_sharpe", "label",
    }


def test_fundamental_validation_N_flows_to_dsr_n_used():
    panel = _panel()
    base = fundamental_candidate_metrics(panel, _fundamental_panel(panel), CFG, vcfg=VCFG)
    bumped = fundamental_candidate_metrics(
        panel,
        _fundamental_panel(panel),
        CFG,
        vcfg=dataclasses.replace(VCFG, declared_trials_N=17),
    )
    assert base["validation"]["n_used"] == 10
    assert bumped["validation"]["n_used"] == 17


def test_fundamental_power_report_is_non_degenerate_on_dev_folds():
    panel = _panel()
    m = fundamental_candidate_metrics(panel, _fundamental_panel(panel), CFG, vcfg=VCFG)
    power = m["power"]
    assert power["folds"]
    assert power["power_limited"] is False
    for fold in power["folds"]:
        assert fold["min_positive_train"] > 0
        assert fold["median_positive_train"] > 0


def test_fundamental_candidate_rejects_unwarmed_fundamental_panel():
    panel = _panel()
    full_funda = build_fundamental_panel(
        _records(panel),
        panel,
        [c for c in panel.columns if c != "SPY"],
        warmup=0,
    )
    with pytest.raises(ValueError, match="cfg.warmup"):
        fundamental_candidate_metrics(panel, full_funda, CFG, vcfg=VCFG)
