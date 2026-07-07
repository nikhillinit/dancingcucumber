import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from advisor.backtest.ls_reversal_screen import decide, reversed_net_stream
from advisor.backtest.residual_screen import resid
from advisor.backtest.stats import book_sharpe
from advisor.research.ls_reversal_prereg import DEFAULT_LS_REVERSAL

RNG = np.random.default_rng(7)


def _streams(n=500):
    spy = pd.Series(RNG.normal(0.0004, 0.01, n))
    net = pd.Series(0.9 * spy.values + RNG.normal(-0.0002, 0.004, n))  # negative-alpha book
    turnover = pd.Series(RNG.uniform(0.0, 0.2, n))
    gross = pd.Series(RNG.uniform(0.8, 1.0, n))
    return net, turnover, gross, spy


def test_zero_cost_identity():
    net, turnover, gross, spy = _streams()
    cfg = replace(DEFAULT_LS_REVERSAL, cost_per_turn=0.0, borrow_rate_annual=0.0)
    rev, a = reversed_net_stream(net, turnover, gross, spy, cfg)
    res, a2 = resid(net, spy)
    assert a == a2
    assert book_sharpe(rev) == pytest.approx(-book_sharpe(res), abs=1e-12)


def test_hand_computed_costs():
    net = pd.Series([0.01, -0.01]); spy = pd.Series([0.0, 0.0])
    turnover = pd.Series([0.1, 0.2]); gross = pd.Series([1.0, 1.0])
    cfg = replace(DEFAULT_LS_REVERSAL, trading_days=252)
    rev, a = reversed_net_stream(net, turnover, gross, spy, cfg)
    b = 0.005 / 252
    # rev = -net - 2*turn*c - gross*b + a*spy (spy=0 kills the hedge term)
    assert rev.iloc[0] == pytest.approx(-0.01 - 2 * 0.1 * 0.0005 - b)
    assert rev.iloc[1] == pytest.approx(+0.01 - 2 * 0.2 * 0.0005 - b)


def test_costs_are_monotone_drag():
    net, turnover, gross, spy = _streams()
    lo, _ = reversed_net_stream(net, turnover, gross, spy, DEFAULT_LS_REVERSAL)
    hi, _ = reversed_net_stream(net, turnover, gross, spy,
                                replace(DEFAULT_LS_REVERSAL, borrow_rate_annual=0.05))
    assert book_sharpe(hi) < book_sharpe(lo)


def _family(precost, postcost):
    return {"precost_ir": precost, "postcost_reversed_ir": postcost, "beta": 0.9}


def test_tripwire_abort_suppresses_postcost():
    fams = {"value": _family(-0.10, 0.5),                       # drifted vs published -0.41
            "fundamental_value": _family(-0.32, 0.5),
            "lazy_prices": _family(-0.40, 0.5)}
    out = decide(fams, DEFAULT_LS_REVERSAL)
    assert out["verdict"] == "ABORT"
    assert all("postcost_reversed_ir" not in f for f in out["families"].values())


def test_pass_needs_two_of_three_at_tau():
    ok = {"value": _family(-0.41, 0.20), "fundamental_value": _family(-0.32, 0.20),
          "lazy_prices": _family(-0.40, 0.05)}
    assert decide(ok, DEFAULT_LS_REVERSAL)["verdict"] == "PASS"
    one = {"value": _family(-0.41, 0.35), "fundamental_value": _family(-0.32, 0.10),
           "lazy_prices": _family(-0.40, 0.10)}
    assert decide(one, DEFAULT_LS_REVERSAL)["verdict"] == "CLOSED"


def test_readings_wiring_builds_without_running_any_sweep():
    from advisor.backtest.ls_reversal_screen import _readings
    r = _readings(DEFAULT_LS_REVERSAL)
    assert set(r) == set(DEFAULT_LS_REVERSAL.families)
    for family, (panel, fam_cfg, raw_fn) in r.items():
        assert "SPY" in panel.columns
        assert callable(raw_fn)


def test_one_shot_lock(tmp_path):
    import json
    from advisor.backtest.ls_reversal_screen import enforce_one_shot
    rp = tmp_path / "result.json"
    enforce_one_shot(rp, None)                       # no prior result: allowed
    for locked in ("PASS", "CLOSED"):
        rp.write_text(json.dumps({"verdict": locked}))
        with pytest.raises(SystemExit):
            enforce_one_shot(rp, None)               # locked forever...
        with pytest.raises(SystemExit):
            enforce_one_shot(rp, "any reason")       # ...even with a reason
    rp.write_text(json.dumps({"verdict": "ABORT"}))
    with pytest.raises(SystemExit):
        enforce_one_shot(rp, None)                   # ABORT without reason: refused
    enforce_one_shot(rp, "fixed fixture path typo")  # ABORT + reason: allowed


def test_record_result_writes_reason(tmp_path):
    import json
    from advisor.backtest.ls_reversal_screen import record_result
    rp = tmp_path / "result.json"
    record_result(rp, {"verdict": "ABORT"}, None)
    assert json.loads(rp.read_text())["rerun_after_abort_reason"] is None
    record_result(rp, {"verdict": "CLOSED"}, "fixed fixture path typo")
    assert json.loads(rp.read_text())["rerun_after_abort_reason"] == "fixed fixture path typo"
