from datetime import date

from edgar_b2_classify import build_delisting_events, classify_event


def filing(form: str, filing_date: str, items: str = "") -> dict[str, str]:
    return {"form": form, "filing_date": filing_date, "items": items}


def test_bankruptcy_precedes_acquisition_and_performance() -> None:
    filings = [
        filing("8-K", "2020-01-05", "3.01"),
        filing("8-K", "2020-01-06", "2.01"),
        filing("8-K", "2020-01-07", "1.03"),
    ]

    assert classify_event(date(2020, 1, 15), filings) == "bankruptcy"


def test_acquisition_precedes_performance() -> None:
    filings = [
        filing("8-K", "2020-01-05", "3.01"),
        filing("8-K", "2020-01-06", "2.01"),
    ]

    assert classify_event(date(2020, 1, 15), filings) == "acquisition"


def test_ninety_day_window_excludes_filings_one_hundred_days_away() -> None:
    filings = [filing("8-K", "2020-04-10", "1.03")]

    assert classify_event(date(2020, 1, 1), filings) == "unknown"


def test_defm14a_in_prior_365_days_classifies_as_acquisition() -> None:
    filings = [filing("DEFM14A", "2019-01-02")]

    assert classify_event(date(2020, 1, 1), filings) == "acquisition"


def test_item_matching_uses_exact_tokens() -> None:
    filings = [filing("8-K", "2020-01-01", "11.03,3.01")]

    assert classify_event(date(2020, 1, 1), filings) == "performance"


def test_form25_cluster_within_30_days_uses_earliest_date() -> None:
    rows = [
        {"cik": "100", "company": "Example Corp", "form": "25-NSE", "date": "2020-01-31"},
        {"cik": "100", "company": "Example Corp", "form": "25", "date": "2020-01-01"},
        {"cik": "100", "company": "Example Corp", "form": "25-NSE", "date": "2020-03-05"},
    ]

    events = build_delisting_events(rows)

    assert len(events) == 2
    assert events[0].delist_date == date(2020, 1, 1)
    assert events[0].n_form25 == 2


def test_unknown_when_no_filing_qualifies() -> None:
    filings = [
        filing("8-K", "2020-01-01", "7.01"),
        filing("DEFM14A", "2018-12-31"),
    ]

    assert classify_event(date(2020, 1, 1), filings) == "unknown"
