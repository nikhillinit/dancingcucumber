from advisor.research.candidate_prereg import (
    CandidatePreReg, DEFAULT_CANDIDATE, candidate_hash, candidate_run_hash,
)

FIXTURE = "apps/quant/advisor/tests/fixtures/floor_prices.csv"

def test_default_candidate_freezes_methodology():
    c = DEFAULT_CANDIDATE
    # Pre-committed methodology (rail #4) — these are the falsifiable choices.
    assert c.families == ("value", "momentum")
    assert c.value_skip == 126            # exclude the 6mo momentum window
    assert c.value_lookback == 270        # fixture-feasible intermediate reversal (see note)
    # Feasibility ceiling (rail #5): value must be live in EVERY dev fold. On the
    # 2015-2023 fixture, dev~=1654, fold_size~=330, fold-1 train ends ~325, so a
    # value_lookback >= ~325 leaves fold 1 with an all-NaN train -> dead leg ->
    # dev gate unwinnable for non-signal reasons. Keep margin for the percentile fit.
    assert c.value_lookback <= 300
    assert 0.0 < c.orthogonality_tau <= 0.5
    assert c.declared_trials_N >= 45      # inherits the conservative MinBTL budget
    # Inherited floor gate params must match the floor so the bench is faithful.
    assert c.folds == 5 and c.embargo == 5 and c.margin == 0.0
    assert c.final_lcb == 0.95 and c.dev_lcb == 0.90

def test_candidate_hash_is_stable_and_sensitive():
    import dataclasses
    h0 = candidate_hash(DEFAULT_CANDIDATE)
    assert isinstance(h0, str) and len(h0) == 64
    # Any methodology change must re-hash (no silent p-hacking).
    mutated = dataclasses.replace(DEFAULT_CANDIDATE, value_lookback=1260)
    assert candidate_hash(mutated) != h0

def test_candidate_run_hash_binds_config_and_fixture_bytes():
    import dataclasses
    # Amendment F2: the holdout-unlock key includes fixture bytes (mirrors config_hash).
    h0 = candidate_run_hash(DEFAULT_CANDIDATE, FIXTURE)
    assert isinstance(h0, str) and len(h0) == 64
    # It is NOT the fixture-blind methodology id.
    assert h0 != candidate_hash(DEFAULT_CANDIDATE)
    # A config change must re-hash (no silent re-spec after seeing the holdout).
    mutated = dataclasses.replace(DEFAULT_CANDIDATE, value_lookback=126)
    assert candidate_run_hash(mutated, FIXTURE) != h0
