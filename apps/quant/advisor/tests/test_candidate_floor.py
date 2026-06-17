import dataclasses

import numpy as np
import pandas as pd

from advisor.research.candidate_prereg import DEFAULT_CANDIDATE
from advisor.research.candidate_floor import candidate_metrics
from advisor.research.candidate_validation_prereg import DEFAULT_CANDIDATE_VALIDATION

def _panel(n=1400, k=24, seed=2):
    rng = np.random.default_rng(seed)
    cols = {f"A{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for i in range(k)}
    cols["SPY"] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    return pd.DataFrame(cols)

def test_candidate_metrics_reports_full_floor_schema():
    m = candidate_metrics(_panel(), DEFAULT_CANDIDATE, prereg_hash="deadbeef")
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
