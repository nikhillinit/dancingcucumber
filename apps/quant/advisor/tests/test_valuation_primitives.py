import pytest

from advisor.analysis.valuation_primitives import intrinsic_value_dcf, owner_earnings


def test_owner_earnings_basic():
    # NI 100 + D&A 50 - capex 40 - delta WC 10 = 100
    assert owner_earnings(100, 50, 40, 10) == 100


def test_dcf_zero_growth_zero_terminal_growth_closed_form():
    # base=100, g=0, tg=0, r=0.10, horizon=10, no margin of safety.
    # annuity(10y @10%)*100 + (100/0.10)/1.1^10 ~= 614.46 + 385.54 = 1000.0
    v = intrinsic_value_dcf(100, growth_rate=0.0, terminal_growth=0.0,
                            discount_rate=0.10, horizon=10, margin_of_safety=0.0)
    assert v == pytest.approx(1000.0, rel=1e-3)


def test_dcf_nonpositive_cashflow_is_zero():
    assert intrinsic_value_dcf(0) == 0.0
    assert intrinsic_value_dcf(-50) == 0.0


def test_dcf_higher_growth_yields_higher_value():
    low = intrinsic_value_dcf(100, growth_rate=0.02)
    high = intrinsic_value_dcf(100, growth_rate=0.10)
    assert high > low
