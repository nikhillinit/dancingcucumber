from datetime import date

from advisor.data.edgar_xbrl_fetch import (
    accepted_map, build_asset_rows, close_asof, compute_available_asof,
    cumulative_split_factor, latest_facts_by_accn, merge_concept_facts,
)


def test_latest_facts_by_accn_drops_prior_year_comparatives():
    facts = [
        {"end": "2015-09-26", "val": 5578753000, "accn": "A1", "form": "10-K", "filed": "2016-10-26"},
        {"end": "2016-09-24", "val": 5336166000, "accn": "A1", "form": "10-K", "filed": "2016-10-26"},
        {"end": "2016-09-24", "val": None, "accn": "A2", "form": "10-Q", "filed": "x"},  # skipped
    ]
    out = latest_facts_by_accn(facts)
    assert out["A1"]["end"] == "2016-09-24" and out["A1"]["val"] == 5336166000
    assert "A2" not in out


def test_accepted_map_merges_recent_and_older_files():
    recent = {"filings": {"recent": {
        "accessionNumber": ["A1"], "acceptanceDateTime": ["2016-10-26T20:42:16.000Z"]}}}
    older = {"accessionNumber": ["A0"], "acceptanceDateTime": ["2015-01-02T09:00:00.000Z"]}
    m = accepted_map([recent, older])
    assert m["A1"] == "2016-10-26" and m["A0"] == "2015-01-02"


def test_compute_available_asof_lag_dominates_and_accepted_optional():
    pe, filed, acc = date(2016, 9, 24), date(2016, 10, 26), date(2016, 10, 26)
    assert compute_available_asof(pe, filed, acc) == date(2016, 12, 23)   # end + 90 dominates
    assert compute_available_asof(pe, filed, None) == date(2016, 12, 23)  # accepted missing -> fallback


def test_cumulative_split_factor_after_date():
    splits = [(date(2014, 6, 9), 7.0), (date(2020, 8, 31), 4.0)]
    assert cumulative_split_factor(splits, date(2016, 9, 24)) == 4.0   # only the 2020 split is after
    assert cumulative_split_factor(splits, date(2021, 1, 1)) == 1.0    # none after


def test_close_asof_returns_last_on_or_before():
    closes = [(date(2020, 3, 27), 24.0), (date(2020, 3, 30), 25.0), (date(2020, 4, 1), 26.0)]
    assert close_asof(closes, date(2020, 3, 30)) == 25.0
    assert close_asof(closes, date(2020, 3, 31)) == 25.0
    assert close_asof(closes, date(2020, 1, 1)) is None


def _eq(accn, end, val, filed="2020-02-01", form="10-K"):
    return {"end": end, "val": val, "accn": accn, "form": form, "filed": filed}


def test_build_asset_rows_emits_three_rows_with_real_dollar_anchor():
    # 2:1 split after period_end; split-adjusted close 25 -> real price 50 -> real mktcap 500
    equity = [_eq("A1", "2019-12-31", 1000.0)]
    shares = [_eq("A1", "2019-12-31", 10.0)]
    accepted = {"A1": "2020-02-01"}
    splits = [(date(2020, 6, 1), 2.0)]
    closes = [(date(2020, 3, 30), 25.0)]   # avail = 2019-12-31 + 90 = 2020-03-30
    rows = build_asset_rows("AAA", "0001", equity, shares, accepted, splits, closes)
    by = {r["concept"]: r for r in rows}
    assert set(by) == {"StockholdersEquity", "CommonStockSharesOutstanding", "MarketCapAnchor"}
    assert by["StockholdersEquity"]["available_asof"] == "2020-03-30"
    assert float(by["MarketCapAnchor"]["value"]) == 10.0 * 2.0 * 25.0   # = 500.0 real dollars
    assert by["MarketCapAnchor"]["unit"] == "USD"


def test_build_asset_rows_skips_anchor_when_shares_period_mismatches():
    # shares only present for the prior-year comparative (different end) -> equity row only, no anchor
    equity = [_eq("A1", "2019-12-31", 1000.0)]
    shares = [_eq("A1", "2018-12-31", 9.0)]   # mismatched end
    rows = build_asset_rows("AAA", "0001", equity, shares, {"A1": "2020-02-01"},
                            [], [(date(2020, 3, 30), 25.0)])
    concepts = {r["concept"] for r in rows}
    assert concepts == {"StockholdersEquity"}


def test_build_asset_rows_accepts_cover_date_shares():
    # dei EntityCommonStockSharesOutstanding reports at a cover date a few weeks AFTER
    # period_end; the window match must still emit the anchor (not require exact ==).
    equity = [_eq("A1", "2019-12-31", 1000.0)]
    shares = [_eq("A1", "2020-01-20", 10.0)]   # cover date, 20 days after period_end
    rows = build_asset_rows("AAA", "0001", equity, shares, {"A1": "2020-02-01"},
                            [], [(date(2020, 3, 30), 50.0)])
    by = {r["concept"]: r for r in rows}
    assert "MarketCapAnchor" in by
    assert float(by["MarketCapAnchor"]["value"]) == 10.0 * 1.0 * 50.0


def test_merge_concept_facts_prefers_primary_backfills_fallback():
    primary = [_eq("A1", "2019-12-31", 1000.0)]
    fallback = [_eq("A1", "2019-12-31", 1111.0), _eq("A2", "2018-12-31", 900.0)]
    merged = merge_concept_facts(primary, fallback)
    by = latest_facts_by_accn(merged)
    assert by["A1"]["val"] == 1000.0   # primary wins for A1
    assert by["A2"]["val"] == 900.0    # fallback backfills A2


def test_build_asset_rows_filters_forms_and_year_range():
    equity = [
        _eq("A1", "2019-12-31", 1000.0, form="8-K"),     # non-periodic -> skipped
        _eq("A2", "2009-12-31", 900.0, form="10-K"),     # out of [2014, 2023] -> skipped
    ]
    rows = build_asset_rows("AAA", "0001", equity, [], {}, [], [])
    assert rows == []
