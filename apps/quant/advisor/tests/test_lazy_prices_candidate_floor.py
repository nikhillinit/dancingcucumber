import numpy as np
import pandas as pd

from advisor.research.candidate_prereg_lazy_prices import DEFAULT_LAZY_PRICES_CANDIDATE
from advisor.research.lazy_prices import build_lazy_prices_panel, SIMILARITY_CONCEPT
from advisor.research.candidate_floor import lazy_prices_candidate_metrics
from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from datetime import date


def _synthetic_panel(n=400):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    cols = {a: 100 + np.cumsum(rng.normal(0.05, 1.0, n)) for a in
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH",
             "III", "JJJ", "KKK", "LLL"]}
    cols["SPY"] = 400 + np.cumsum(rng.normal(0.04, 0.8, n))
    return pd.DataFrame(cols, index=idx).astype(float)


def _sim_records(panel, assets):
    recs = []
    for a in assets:
        for k, t in enumerate(panel.index[::60]):       # ~quarterly filings
            d = t.date()
            recs.append(EdgarXbrlRecord(
                asset=a, cik="0", accession=f"{a}-{k}", form="10-Q",
                report_period_end=d, filing_date=d, accepted_datetime=d,
                concept=SIMILARITY_CONCEPT, unit="ratio",
                value=0.80 + 0.1 * ((hash((a, k)) % 10) / 10.0),
                available_asof=d, superseded_by="", amended_flag=False,
                missingness_reason="", denominator_policy="cosine_tfidf_yoy_same_form",
            ))
    return recs


def test_metrics_blinded_by_default():
    cfg = DEFAULT_LAZY_PRICES_CANDIDATE
    panel = _synthetic_panel()
    assets = [c for c in panel.columns if c != "SPY"]
    recs = _sim_records(panel, assets)
    panel_lp = build_lazy_prices_panel(recs, panel, assets, warmup=cfg.warmup)
    m = lazy_prices_candidate_metrics(panel, panel_lp, cfg, prereg_hash=None)
    assert m["holdout"] is None          # blinded
    assert m["passes"] is False
    assert set(m).issuperset({"verdict", "universe", "dev", "weights",
                              "ensemble", "best_family", "validation", "power"})


def test_row_count_guard():
    cfg = DEFAULT_LAZY_PRICES_CANDIDATE
    panel = _synthetic_panel()
    bad = pd.DataFrame({c: [0.5] for c in panel.columns if c != "SPY"})  # 1 row
    try:
        lazy_prices_candidate_metrics(panel, bad, cfg, prereg_hash=None)
        assert False, "expected row-count ValueError"
    except ValueError:
        pass


def test_power_report_present():
    cfg = DEFAULT_LAZY_PRICES_CANDIDATE
    panel = _synthetic_panel()
    assets = [c for c in panel.columns if c != "SPY"]
    panel_lp = build_lazy_prices_panel(_sim_records(panel, assets), panel, assets,
                                       warmup=cfg.warmup)
    m = lazy_prices_candidate_metrics(panel, panel_lp, cfg, prereg_hash=None)
    assert "power_limited" in m["power"]
