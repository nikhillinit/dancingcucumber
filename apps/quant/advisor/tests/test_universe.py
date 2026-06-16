from advisor.backtest.universe import classify_universe
from advisor.backtest.prereg import DEFAULT_CONFIG


def test_formal_when_broad():
    assert classify_universe([22, 25, 24, 23], DEFAULT_CONFIG) == "formal"


def test_micro_when_thin_but_runnable():
    assert classify_universe([14, 13, 15, 12], DEFAULT_CONFIG) == "micro"


def test_do_not_run_when_too_thin():
    assert classify_universe([14, 8, 20, 19], DEFAULT_CONFIG) == "do_not_run"
