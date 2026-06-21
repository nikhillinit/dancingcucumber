import dataclasses

import pytest

from advisor.research.candidate_prereg_lazy_prices import (
    DEFAULT_LAZY_PRICES_CANDIDATE, lazy_prices_candidate_hash, lazy_prices_candidate_run_hash,
)
from advisor.research.candidate_validation_prereg_lazy_prices import (
    DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION, lazy_prices_candidate_validation_hash,
)

# reuse the existing floor fixture for the byte-binding test (the lazy-prices fixture is built in D6)
FIXTURE = "apps/quant/advisor/tests/fixtures/floor_prices.csv"


def test_lazy_prices_candidate_freezes_methodology():
    c = DEFAULT_LAZY_PRICES_CANDIDATE
    assert c.families == ("lazy_prices", "momentum")
    assert c.similarity_metric == "cosine_tfidf"
    assert c.fixture_source == "SEC_EDGAR_FILING_TEXT"
    assert c.reporting_lag_days == 0          # filing text has no +90 lag
    assert 0.0 < c.orthogonality_tau <= 0.5
    assert c.folds == 5 and c.embargo == 5 and c.margin == 0.0
    assert c.dev_lcb == 0.90 and c.final_lcb == 0.95
    with pytest.raises(Exception):            # frozen dataclass
        c.margin = 0.5


def test_lazy_prices_candidate_hash_stable_and_sensitive():
    h0 = lazy_prices_candidate_hash(DEFAULT_LAZY_PRICES_CANDIDATE)
    assert isinstance(h0, str) and len(h0) == 64
    assert lazy_prices_candidate_hash(DEFAULT_LAZY_PRICES_CANDIDATE) == h0
    mutated = dataclasses.replace(DEFAULT_LAZY_PRICES_CANDIDATE, orthogonality_tau=0.25)
    assert lazy_prices_candidate_hash(mutated) != h0


def test_lazy_prices_run_hash_binds_config_and_fixture_bytes():
    h0 = lazy_prices_candidate_run_hash(DEFAULT_LAZY_PRICES_CANDIDATE, FIXTURE)
    assert isinstance(h0, str) and len(h0) == 64
    assert h0 != lazy_prices_candidate_hash(DEFAULT_LAZY_PRICES_CANDIDATE)
    mutated = dataclasses.replace(DEFAULT_LAZY_PRICES_CANDIDATE, similarity_metric="jaccard")
    assert lazy_prices_candidate_run_hash(mutated, FIXTURE) != h0


def test_lazy_prices_validation_is_live_trial_surface():
    v = DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION
    assert v.declared_trials_N == 45
    assert v.dsr_pass == 0.95
    assert v.declared_var_sr > 0
    h = lazy_prices_candidate_validation_hash(v)
    assert isinstance(h, str) and len(h) == 64
    mutated = dataclasses.replace(v, declared_trials_N=90)
    assert lazy_prices_candidate_validation_hash(mutated) != h


def test_methodology_hash_differs_from_run_hash():
    assert (lazy_prices_candidate_hash(DEFAULT_LAZY_PRICES_CANDIDATE)
            != lazy_prices_candidate_run_hash(DEFAULT_LAZY_PRICES_CANDIDATE, FIXTURE))
