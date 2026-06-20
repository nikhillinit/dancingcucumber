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

from advisor.research.lazy_prices import make_lazy_prices_raw


def test_raw_is_similarity_not_its_complement():
    # SIGN GUARD: HIGH similarity = non-changer = LONG leg. raw MUST equal similarity,
    # never 1-similarity (that would long the changers = the wrong/short leg ->
    # DEV_FAILED by construction).
    panel_lp = pd.DataFrame({"NONCHANGER": [0.97, 0.97], "CHANGER": [0.20, 0.20]})
    raw_fn = make_lazy_prices_raw(panel_lp)
    hi = raw_fn(LAZY_PRICES, pd.Series([100.0, 101.0], name="NONCHANGER"))
    lo = raw_fn(LAZY_PRICES, pd.Series([100.0, 101.0], name="CHANGER"))
    assert hi.iloc[0] == 0.97 and lo.iloc[0] == 0.20     # raw == similarity
    assert hi.mean() > lo.mean()                          # non-changer ranks long


def test_raw_dispatches_momentum_to_base_and_unknown_to_nan():
    panel_lp = pd.DataFrame({"AAA": [0.5, 0.5, 0.5]})
    raw_fn = make_lazy_prices_raw(panel_lp)
    # momentum delegates to the frozen price raw_metric (not the panel)
    mom = raw_fn("momentum", pd.Series([1.0] * 130, name="AAA"))
    assert len(mom) == 130
    # unknown asset on the lazy_prices family -> all-NaN neutral
    unk = raw_fn(LAZY_PRICES, pd.Series([1.0, 2.0], name="ZZZ"))
    assert unk.isna().all()


def test_raw_raises_on_warmup_mismatch():
    panel_lp = pd.DataFrame({"AAA": [0.5]})               # 1 row
    raw_fn = make_lazy_prices_raw(panel_lp)
    try:
        raw_fn(LAZY_PRICES, pd.Series([1.0, 2.0, 3.0], name="AAA"))  # 3 rows
        assert False, "expected ValueError on panel shorter than prices"
    except ValueError:
        pass

from advisor.research.lazy_prices import (
    dev_lazy_momentum_corr, dev_cross_sectional_dispersion,
)


def _multi_asset_panel(n=400, names=("AAA", "BBB", "CCC", "DDD")):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    cols = {a: [100.0 + i + j for i in range(n)] for j, a in enumerate(names)}
    cols["SPY"] = [400.0 + i for i in range(n)]
    return pd.DataFrame(cols, index=idx)


def _step_records(panel, level_by_asset):
    # one filing every ~60 trading days, each asset pinned to its own similarity LEVEL
    recs = []
    for a, lvl in level_by_asset.items():
        for k, t in enumerate(panel.index[::60]):
            d = str(t.date())
            recs.append(_rec(a, lvl + 0.01 * (k % 3), filing=d, accepted=d,
                             period=d, avail=d))
    return recs


def test_orthogonality_diagnostic_returns_corr_key():
    panel = _multi_asset_panel()
    recs = _step_records(panel, {"AAA": 0.9, "BBB": 0.5, "CCC": 0.3, "DDD": 0.7})
    panel_lp = build_lazy_prices_panel(recs, panel, warmup=200)
    c = dev_lazy_momentum_corr(panel, panel_lp, warmup=200, holdout_frac=0.2)
    assert "momentum" in c                       # report-only; NaN allowed if a leg is flat


def test_cross_sectional_dispersion_detects_level_collapse():
    # Per-asset transform encodes each name vs ITS OWN history, so a big cross-sectional
    # LEVEL gap (0.9 vs 0.3) does NOT guarantee cross-sectional conviction spread.
    panel = _multi_asset_panel()
    recs = _step_records(panel, {"AAA": 0.9, "BBB": 0.5, "CCC": 0.3, "DDD": 0.7})
    panel_lp = build_lazy_prices_panel(recs, panel, warmup=200)
    d = dev_cross_sectional_dispersion(panel, panel_lp, warmup=200, holdout_frac=0.2)
    assert "min_xs_std" in d and "median_xs_std" in d
    # documents (does not assert a threshold) the collapse the bench transform induces;
    # the VALUE is what READING_C_RESULT.md reports so a DEV_FAILED is interpretable.

from advisor.data.filing_text_fetch import tokenize, cosine_tfidf, build_similarity_row


def test_tokenize_is_deterministic_lowercase_words():
    assert tokenize("The QUICK brown fox; fox!") == ["the", "quick", "brown", "fox", "fox"]


def test_cosine_identical_is_one_and_disjoint_is_zero():
    a = tokenize("alpha beta gamma alpha")
    assert abs(cosine_tfidf(a, a) - 1.0) < 1e-9
    assert cosine_tfidf(tokenize("alpha beta"), tokenize("gamma delta")) == 0.0


def test_cosine_partial_overlap_in_unit_interval():
    s = cosine_tfidf(tokenize("alpha beta gamma"), tokenize("alpha beta delta"))
    assert 0.0 < s < 1.0


def test_build_similarity_row_sets_availability_without_lag():
    row = build_similarity_row(
        asset="AAA", cik="0000320193", accession="acc1", form="10-K",
        report_period_end="2015-12-31", filing_date="2016-02-01",
        accepted_datetime="2016-02-03T16:30:00", similarity=0.91,
    )
    assert row["concept"] == "FilingSimilarity"
    assert row["value"] == "0.91"
    assert row["available_asof"] == "2016-02-03"     # max(filing, accepted), no +90
    assert row["denominator_policy"] == "cosine_tfidf_yoy_same_form"
