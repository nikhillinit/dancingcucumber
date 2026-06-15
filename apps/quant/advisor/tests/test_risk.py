from advisor.risk.limits import (
    correlation_multiplier,
    position_limit,
    vol_adjusted_fraction,
)


def test_low_vol_gets_full_fraction():
    assert vol_adjusted_fraction(0.10) == 0.25


def test_high_vol_is_capped_low():
    assert vol_adjusted_fraction(0.60) == 0.10


def test_correlation_penalizes_crowded_and_rewards_diversifiers():
    assert correlation_multiplier(0.90) == 0.70
    assert correlation_multiplier(0.0) == 1.10
    assert correlation_multiplier(0.50) == 1.0


def test_position_limit_dollars():
    # net_liq 100k, fraction 0.25, multiplier 1.0 -> 25k
    assert position_limit(100_000, vol=0.10, correlation=0.5) == 25_000.0


def test_position_limit_never_negative():
    assert position_limit(0, vol=0.10, correlation=0.5) == 0.0
