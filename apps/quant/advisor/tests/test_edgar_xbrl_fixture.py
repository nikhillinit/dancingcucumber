from datetime import date

from advisor.data.edgar_xbrl_fixture import (
    FIELDS, audit_available_asof, compute_available_asof, coverage_in_window, load_fixture,
)

HEADER = ",".join(FIELDS)

# one complete 2016 10-K row + one row missing `value` (must be dropped, never filled)
GOOD_ROW = (
    "AAPL,0000320193,0000320193-16-000178,10-K,2016-09-24,2016-10-26,"
    "2016-10-26T16:31:00,StockholdersEquity,USD,128249000000,2016-12-23,"
    ",false,,as_reported_then_split_adjusted"
)
MISSING_VALUE_ROW = (
    "MSFT,0000789019,0000789019-16-000001,10-K,2016-06-30,2016-07-28,"
    "2016-07-28T16:00:00,StockholdersEquity,USD,,2016-09-28,"
    ",false,value_not_reported,as_reported_then_split_adjusted"
)


def _write(tmp_path, *rows):
    p = tmp_path / "edgar.csv"
    p.write_text("\n".join((HEADER, *rows)) + "\n", encoding="utf-8")
    return p


def test_schema_is_15_fields():
    assert len(FIELDS) == 15


def test_complete_row_parses_all_fields(tmp_path):
    recs = load_fixture(_write(tmp_path, GOOD_ROW))
    assert len(recs) == 1
    r = recs[0]
    assert r.asset == "AAPL"
    assert r.cik == "0000320193"
    assert r.accession == "0000320193-16-000178"
    assert r.form == "10-K"
    assert r.report_period_end == date(2016, 9, 24)
    assert r.filing_date == date(2016, 10, 26)
    assert r.accepted_datetime == date(2016, 10, 26)
    assert r.concept == "StockholdersEquity"
    assert r.unit == "USD"
    assert r.value == 128249000000.0
    assert r.available_asof == date(2016, 12, 23)
    assert r.superseded_by == ""
    assert r.amended_flag is False
    assert r.missingness_reason == ""
    assert r.denominator_policy == "as_reported_then_split_adjusted"


def test_missing_value_row_is_dropped_not_filled(tmp_path):
    recs = load_fixture(_write(tmp_path, GOOD_ROW, MISSING_VALUE_ROW))
    assert len(recs) == 1
    assert all(r.asset != "MSFT" for r in recs)


def test_available_asof_audit_matches_recompute(tmp_path):
    recs = load_fixture(_write(tmp_path, GOOD_ROW))
    assert audit_available_asof(recs[0])
    # report_period_end 2016-09-24 + 90d = 2016-12-23 dominates filing/accepted (2016-10-26)
    assert compute_available_asof(date(2016, 9, 24), date(2016, 10, 26), date(2016, 10, 26)) == date(2016, 12, 23)


def test_availability_is_in_window_not_fetch_date(tmp_path):
    # POSITIVE availability guard: a 2016 filing is usable in 2016, NOT a future fetch date.
    recs = load_fixture(_write(tmp_path, GOOD_ROW))
    assert recs[0].available_asof.year == 2016


def test_coverage_nondegenerate_within_backtest_window(tmp_path):
    recs = load_fixture(_write(tmp_path, GOOD_ROW))
    assert coverage_in_window(recs, date(2015, 1, 1), date(2023, 12, 31)) == 1.0
    # a row stamped outside the window (e.g. a fetch-date bug) is excluded
    assert coverage_in_window(recs, date(2015, 1, 1), date(2015, 12, 31)) == 0.0
