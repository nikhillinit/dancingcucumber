import dataclasses

from advisor.research.candidate_validation_prereg import (
    CandidateValidationPreReg,
    DEFAULT_CANDIDATE_VALIDATION,
    candidate_validation_hash,
)


def test_default_candidate_validation_freezes_dsr_params():
    v = DEFAULT_CANDIDATE_VALIDATION
    # Field names mirror ValidationPreReg so validation_report reads them unchanged.
    assert v.declared_trials_N == 45      # candidate trial count (>= MinBTL budget)
    assert v.minbtl_max_trials == 45
    assert v.dsr_pass == 0.95
    assert v.tstat_hurdle == 3.0
    assert v.psr_benchmark_sr == 0.0
    # declared_var_sr is the PRE-REGISTERED constant (frozen in CANDIDATE_PREREG.md, its
    # hash recorded there). T8 calibrates the candidate's per-obs trial var_sr to JUSTIFY
    # reusing 1e-4 as conservative-and-stricter (1e-4 >= measured) -- it does NOT silently
    # mutate this default. If calibration ever required a different value, that is a
    # re-pre-registration event (new hash, documented) and this assert SHOULD break loudly.
    assert v.declared_var_sr == 1e-4


def test_candidate_validation_hash_is_stable_and_sensitive():
    h0 = candidate_validation_hash(DEFAULT_CANDIDATE_VALIDATION)
    assert isinstance(h0, str) and len(h0) == 64
    # Bumping the trial count (rail #4 secondary run) MUST re-hash.
    bumped = dataclasses.replace(DEFAULT_CANDIDATE_VALIDATION, declared_trials_N=90)
    assert candidate_validation_hash(bumped) != h0


def test_twin_exposes_every_attr_validation_report_reads():
    # Duck-typed twin (Amendment F1): validation_report reads exactly these six attrs
    # (backtest/validation.py:123-131). The dead-field regression guard (replace N=90 ->
    # validation["n_used"]==90 THROUGH candidate_metrics) lands with Task 7.
    v = DEFAULT_CANDIDATE_VALIDATION
    for attr in ("declared_var_sr", "declared_trials_N", "minbtl_max_trials",
                 "tstat_hurdle", "dsr_pass", "psr_benchmark_sr"):
        assert hasattr(v, attr)
