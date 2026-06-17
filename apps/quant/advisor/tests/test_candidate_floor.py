import dataclasses

import numpy as np
import pandas as pd
import pytest

from advisor.research.candidate_prereg import (
    DEFAULT_CANDIDATE, CandidatePreReg, candidate_run_hash,
)
from advisor.research.candidate_floor import candidate_metrics, _verify_holdout_unlock
from advisor.research.candidate_validation_prereg import DEFAULT_CANDIDATE_VALIDATION

FIXTURE = "apps/quant/advisor/tests/fixtures/floor_prices.csv"

def _panel(n=1400, k=24, seed=2):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)

def test_candidate_metrics_reports_full_floor_schema():
    m = candidate_metrics(_panel(), DEFAULT_CANDIDATE, prereg_hash=None)
    for key in ("verdict", "universe", "dev", "weights", "holdout",
                "ensemble", "spy", "best_family", "margin", "passes", "validation", "power"):
        assert key in m
    assert m["verdict"] in {"DEV_FAILED", "UNSUPPORTED", "INCONCLUSIVE", "PASSED"}

def test_holdout_blinded_until_dev_passes():
    # With prereg_hash=None the holdout is never touched (leakage guard, rail #5).
    m = candidate_metrics(_panel(), DEFAULT_CANDIDATE, prereg_hash=None)
    assert m["holdout"] is None
    assert m["passes"] is False

def test_candidate_validation_N_flows_to_dsr_n_used():
    # Amendment F1 dead-field regression guard: bumping the candidate's declared_trials_N
    # (rail #4 secondary run) MUST change validation["n_used"] -> proves the field is LIVE,
    # not the dead field it was when Task 7 passed the floor's DEFAULT_VALIDATION.
    panel = _panel()
    base = candidate_metrics(panel, DEFAULT_CANDIDATE, prereg_hash=None)
    bumped = candidate_metrics(panel, DEFAULT_CANDIDATE, prereg_hash=None,
                               vcfg=dataclasses.replace(DEFAULT_CANDIDATE_VALIDATION,
                                                        declared_trials_N=90))
    assert base["validation"]["n_used"] == 45
    assert bumped["validation"]["n_used"] == 90

def test_power_report_present_and_flags_thin_value_fit():
    # Amendment F6: the power block reports per-fold value-fit sufficiency. On this short
    # positive-drift panel the value leg never fires (dead) -> power_limited True; on the
    # real fixture (Task 8) it measures the genuine fold-1 thinness.
    m = candidate_metrics(_panel(), DEFAULT_CANDIDATE, prereg_hash=None)
    power = m["power"]
    assert set(power) >= {"folds", "power_limited", "positive_floor"}
    assert power["folds"]                                   # non-empty per-fold report
    for fold in power["folds"]:
        assert set(fold) >= {"min_positive_train", "median_positive_train",
                             "nonzero_transformed_coverage", "test_obs"}
    assert power["power_limited"] is True                   # dead value leg on this panel

def test_holdout_unlock_requires_verified_run_hash():
    # Review F2: the reserved tail unlocks ONLY via candidate_run_hash(cfg, fixture) — a
    # non-null string is NOT enough (the frozen floor and the original mirror unlocked on any
    # truthy hash). Verify valid-hash, wrong-hash, and missing-fixture behavior directly.
    c = DEFAULT_CANDIDATE
    assert _verify_holdout_unlock(c, None, None) is False              # blinded
    assert _verify_holdout_unlock(c, candidate_run_hash(c, FIXTURE), FIXTURE) is True
    with pytest.raises(ValueError):                                    # wrong hash
        _verify_holdout_unlock(c, "deadbeef", FIXTURE)
    with pytest.raises(ValueError):                                    # hash without fixture
        _verify_holdout_unlock(c, "deadbeef", None)

def test_no_holdout_outputs_are_invariant_to_the_reserved_tail():
    # Review F1: prove the blinded path NEVER reads the reserved tail — mutating only the
    # post-dev (holdout) rows must leave every no-holdout output identical, INCLUDING `spy`.
    panel = _panel()
    cfg = DEFAULT_CANDIDATE
    base = candidate_metrics(panel, cfg, prereg_hash=None)
    n = len(panel)
    dev_end_abs = cfg.warmup + int((n - cfg.warmup) * 0.8)
    mutated = panel.copy()
    mutated.iloc[dev_end_abs:] *= 3.0          # perturb ONLY the holdout tail (all columns)
    after = candidate_metrics(mutated, cfg, prereg_hash=None)
    assert after["verdict"] == base["verdict"]
    assert after["spy"] == base["spy"]         # the F1 fix: SPY benchmark is dev-only
    assert after["ensemble"] == base["ensemble"]
    assert after["best_family"] == base["best_family"]
    assert after["weights"] == base["weights"]
    assert after["dev"]["fold_deltas"] == base["dev"]["fold_deltas"]

def test_candidate_prereg_declared_trials_N_is_vestigial_not_the_dsr_source():
    # Review F4 (pushback): CandidatePreReg.declared_trials_N is a VESTIGIAL duplicate — the
    # LIVE DSR trial count is CandidateValidationPreReg.declared_trials_N (Amendment F1). It is
    # deliberately kept (changing it would mutate the frozen, recorded candidate_hash AFTER the
    # run); this test pins it as a no-op so a future secondary run can't be trapped by it.
    panel = _panel()
    base = candidate_metrics(panel, DEFAULT_CANDIDATE, prereg_hash=None)
    bumped_wrong = candidate_metrics(
        panel,
        dataclasses.replace(DEFAULT_CANDIDATE, declared_trials_N=90),  # WRONG surface
        prereg_hash=None,
    )
    assert bumped_wrong["validation"]["n_used"] == base["validation"]["n_used"] == 45
