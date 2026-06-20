from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord
from advisor.research.lazy_prices import (
    LAZY_PRICES, SIMILARITY_CONCEPT,
    compute_text_available_asof, audit_text_available_asof,
)


def _rec(asset, value, *, filing="2016-02-01", accepted="2016-02-03",
         period="2015-12-31", avail="2016-02-03"):
    return EdgarXbrlRecord(
        asset=asset, cik="0000320193", accession="acc-" + asset, form="10-K",
        report_period_end=date.fromisoformat(period),
        filing_date=date.fromisoformat(filing),
        accepted_datetime=date.fromisoformat(accepted),
        concept=SIMILARITY_CONCEPT, unit="ratio", value=value,
        available_asof=date.fromisoformat(avail),
        superseded_by="", amended_flag=False, missingness_reason="",
        denominator_policy="cosine_tfidf_yoy_same_form",
    )


def test_text_availability_has_no_reporting_lag():
    # filing-text availability = max(filing_date, accepted), NOT report_period_end+90
    got = compute_text_available_asof(date(2016, 2, 1), date(2016, 2, 3))
    assert got == date(2016, 2, 3)
    # +90 from 2015-12-31 would be ~2016-03-30 — must NOT be used
    assert got < date(2016, 3, 30)


def test_audit_passes_for_canonical_and_fails_for_lagged():
    ok = _rec("AAA", 0.95, avail="2016-02-03")
    assert audit_text_available_asof(ok) is True
    lagged = _rec("BBB", 0.95, avail="2016-03-30")   # someone wrongly applied +90
    assert audit_text_available_asof(lagged) is False


def test_lazy_prices_family_constant():
    assert LAZY_PRICES == "lazy_prices"

import pandas as pd
from advisor.research.lazy_prices import build_lazy_prices_panel


def _price_panel(dates):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"AAA": range(100, 100 + len(idx)),
                         "SPY": range(400, 400 + len(idx))}, index=idx).astype(float)


def test_panel_is_stepwise_and_pit():
    panel = _price_panel(["2016-01-04", "2016-02-03", "2016-02-04", "2016-03-01"])
    recs = [_rec("AAA", 0.92, accepted="2016-02-03", avail="2016-02-03")]
    out = build_lazy_prices_panel(recs, panel, assets=["AAA"])
    # before availability -> NaN; on/after -> the similarity, held forward
    assert pd.isna(out["AAA"].iloc[0])                  # 2016-01-04 (pre-filing)
    assert out["AAA"].iloc[1] == 0.92                   # 2016-02-03 (available)
    assert out["AAA"].iloc[2] == 0.92                   # held forward
    assert out["AAA"].iloc[3] == 0.92


def test_panel_unknown_asset_all_nan_and_warmup_slices():
    panel = _price_panel(["2016-01-04", "2016-02-03", "2016-02-04"])
    out = build_lazy_prices_panel([], panel, assets=["AAA"], warmup=1)
    assert len(out) == 2                                 # warmup row dropped
    assert out["AAA"].isna().all()                       # no records -> neutral
