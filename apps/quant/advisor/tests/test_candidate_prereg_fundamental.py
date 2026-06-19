import dataclasses

import pytest

from advisor.research.candidate_prereg_fundamental import (
    DEFAULT_FUNDAMENTAL_CANDIDATE, fundamental_candidate_hash, fundamental_candidate_run_hash,
)
from advisor.research.candidate_validation_prereg_fundamental import (
    DEFAULT_FUNDAMENTAL_CANDIDATE_VALIDATION, fundamental_candidate_validation_hash,
)

# reuse the existing floor fixture for the byte-binding test (the EDGAR fixture is built in T7)
FIXTURE = "apps/quant/advisor/tests/fixtures/floor_prices.csv"


def test_fundamental_candidate_freezes_methodology():
    c = DEFAULT_FUNDAMENTAL_CANDIDATE
    assert c.families == ("fundamental_value", "momentum")
    assert c.value_metric == "book_to_price"
    assert c.fixture_source == "SEC_EDGAR_XBRL"
    assert 0.0 < c.orthogonality_tau <= 0.5
    # inherited floor params must match the floor so the bench is faithful
    assert c.folds == 5 and c.embargo == 5 and c.margin == 0.0
    assert c.dev_lcb == 0.90 and c.final_lcb == 0.95
    with pytest.raises(Exception):       # frozen dataclass
        c.margin = 0.5


def test_fundamental_candidate_hash_stable_and_sensitive():
    h0 = fundamental_candidate_hash(DEFAULT_FUNDAMENTAL_CANDIDATE)
    assert isinstance(h0, str) and len(h0) == 64
    assert fundamental_candidate_hash(DEFAULT_FUNDAMENTAL_CANDIDATE) == h0
    mutated = dataclasses.replace(DEFAULT_FUNDAMENTAL_CANDIDATE, orthogonality_tau=0.25)
    assert fundamental_candidate_hash(mutated) != h0


def test_fundamental_run_hash_binds_config_and_fixture_bytes():
    h0 = fundamental_candidate_run_hash(DEFAULT_FUNDAMENTAL_CANDIDATE, FIXTURE)
    assert isinstance(h0, str) and len(h0) == 64
    # not the fixture-blind methodology id
    assert h0 != fundamental_candidate_hash(DEFAULT_FUNDAMENTAL_CANDIDATE)
    mutated = dataclasses.replace(DEFAULT_FUNDAMENTAL_CANDIDATE, value_metric="earnings_yield")
    assert fundamental_candidate_run_hash(mutated, FIXTURE) != h0


def test_fundamental_validation_is_live_trial_surface():
    v = DEFAULT_FUNDAMENTAL_CANDIDATE_VALIDATION
    assert v.declared_trials_N == 45
    assert v.dsr_pass == 0.95
    assert v.declared_var_sr > 0
    h = fundamental_candidate_validation_hash(v)
    assert isinstance(h, str) and len(h) == 64
    mutated = dataclasses.replace(v, declared_trials_N=90)
    assert fundamental_candidate_validation_hash(mutated) != h
