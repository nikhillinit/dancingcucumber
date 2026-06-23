from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from advisor.source_integrity import (
    BridgeRow,
    EdgarDelisting,
    FilingEvent,
    QCDelisting,
    ReasonClass,
    assert_source_only_report,
    bridge_qc_to_edgar,
    classify_reason,
    evaluate_thresholds,
    extract_8k_items,
    load_qc_delistings_csv,
    parse_master_index,
    render_report,
    validate_qc_delisting_rows,
)
from advisor.source_integrity.bridge import AMBIGUOUS, MATCHED, UNMATCHED, normalize_company_name
from advisor.source_integrity.diagnostic import DiagnosticThresholds
from advisor.source_integrity.edgar import SecRateLimiter, build_master_index_url, fetch_text
from advisor.source_integrity.qc_export import QC_EXPORT_FIELDS


def test_b1_master_index_parser_keeps_form25_family_only():
    text = """Description:           Master Index of EDGAR Dissemination Feed
CIK|Company Name|Form Type|Date Filed|Filename
1001|Alpha Widgets Inc|25-NSE|2017-01-03|edgar/data/1001/0001001001-17-000001.txt
1002|Beta Corp|8-K|2017-01-04|edgar/data/1002/0001002002-17-000001.txt
1003|Gamma Group Ltd|25|2017-01-05|edgar/data/1003/0001003003-17-000001.txt
"""
    rows = parse_master_index(text)

    assert [(row.cik, row.company, row.form, row.accession) for row in rows] == [
        ("0000001001", "Alpha Widgets Inc", "25-NSE", "0001001001-17-000001"),
        ("0000001003", "Gamma Group Ltd", "25", "0001003003-17-000001"),
    ]
    assert build_master_index_url(2017, 1).endswith("/2017/QTR1/master.idx")


class _Response:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.data


def test_b1_fetch_requires_declared_user_agent_and_sets_headers():
    captured = {}

    def opener(req, timeout):
        captured["timeout"] = timeout
        captured["ua"] = req.get_header("User-agent")
        captured["host"] = req.get_header("Host")
        return _Response(b"ok")

    assert fetch_text(
        "https://www.sec.gov/Archives/edgar/full-index/2017/QTR1/master.idx",
        user_agent="AIHedgeFund research test@example.com",
        opener=opener,
    ) == "ok"
    assert captured == {
        "timeout": 60,
        "ua": "AIHedgeFund research test@example.com",
        "host": "www.sec.gov",
    }

    with pytest.raises(ValueError):
        fetch_text("https://www.sec.gov/x", user_agent="python", opener=opener)


def test_b1_rate_limiter_respects_ten_request_ceiling():
    now_values = iter([10.0, 10.0, 10.05, 10.05])
    sleeps = []
    limiter = SecRateLimiter()

    limiter.wait(now=lambda: next(now_values), sleep=sleeps.append)
    limiter.wait(now=lambda: next(now_values), sleep=sleeps.append)

    assert sleeps == pytest.approx([0.05])


def test_b2_extracts_items_and_applies_reason_precedence():
    delisting = EdgarDelisting(
        cik="42",
        company="Example Co",
        form="25-NSE",
        filing_date=date(2020, 5, 1),
        accession="acc",
        filename="acc.txt",
    )
    events = [
        FilingEvent("0000000042", "8-K", date(2020, 4, 20), text="Item 3.01 Notice"),
        FilingEvent("42", "8-K", date(2020, 4, 22), text="Item 2.01 Completion"),
        FilingEvent("42", "8-K", date(2020, 4, 23), text="ITEM 1.03 Bankruptcy"),
    ]

    assert extract_8k_items("Item 3.01\nITEM 2.01") == frozenset({"3.01", "2.01"})
    assert classify_reason(delisting, events) is ReasonClass.BANKRUPTCY
    assert classify_reason(delisting, events[:2]) is ReasonClass.ACQUISITION
    assert classify_reason(delisting, events[:1]) is ReasonClass.PERFORMANCE


def test_b2_defm14a_and_listing_rule_paths():
    base = EdgarDelisting(
        cik="7",
        company="Target Inc",
        form="25",
        filing_date=date(2020, 5, 1),
        accession="a",
        filename="a.txt",
    )
    assert classify_reason(
        base,
        [FilingEvent("7", "DEFM14A", date(2019, 12, 1))],
    ) is ReasonClass.ACQUISITION

    with_rule = EdgarDelisting(
        cik="7",
        company="Weak Listing Inc",
        form="25",
        filing_date=date(2020, 5, 1),
        accession="b",
        filename="b.txt",
        cited_rule="continued listing standard under 12d2-2(b)",
    )
    assert classify_reason(with_rule, []) is ReasonClass.PERFORMANCE


def test_b2_voluntary_and_procedural_listing_rules_are_unknown_without_8k():
    voluntary = EdgarDelisting(
        cik="7",
        company="Voluntary Exit Inc",
        form="25",
        filing_date=date(2020, 5, 1),
        accession="c",
        filename="c.txt",
        cited_rule="voluntary withdrawal under 12d2-2(c)",
    )
    procedural = EdgarDelisting(
        cik="7",
        company="Procedural Removal Inc",
        form="25",
        filing_date=date(2020, 5, 1),
        accession="d",
        filename="d.txt",
        cited_rule="removed under 12d2-2(d)",
    )

    assert classify_reason(voluntary, []) is ReasonClass.UNKNOWN
    assert classify_reason(procedural, []) is ReasonClass.UNKNOWN


def _qc_row(**overrides):
    row = {
        "qc_symbol": "ABC",
        "map_file_symbol": "abc",
        "company_name": "ABC Therapeutics Inc",
        "delisting_date": "2020-05-02",
        "last_eligible_date": "2020-04-30",
        "pit_eligible": "true",
        "momentum_decile": "1",
        "value_bucket": "high_bp",
        "negative_book": "false",
    }
    row.update(overrides)
    return row


def test_b3_qc_export_schema_accepts_pit_rows_and_rejects_scope_leaks(tmp_path):
    rows = validate_qc_delisting_rows([_qc_row()])
    assert rows == [
        QCDelisting(
            qc_symbol="ABC",
            map_file_symbol="abc",
            company_name="ABC Therapeutics Inc",
            delisting_date=date(2020, 5, 2),
            last_eligible_date=date(2020, 4, 30),
            momentum_decile=1,
            value_bucket="high_bp",
            negative_book=False,
        )
    ]

    leaked = _qc_row(**{"sha" + "rpe": "1.0"})
    with pytest.raises(ValueError, match="out-of-scope"):
        validate_qc_delisting_rows([leaked])

    with pytest.raises(ValueError, match="PIT-eligible"):
        validate_qc_delisting_rows([_qc_row(pit_eligible="false")])
    with pytest.raises(ValueError, match="1..10"):
        validate_qc_delisting_rows([_qc_row(momentum_decile="11")])
    with pytest.raises(ValueError, match="12-month lookback"):
        validate_qc_delisting_rows([_qc_row(last_eligible_date="2018-01-01")])

    csv_path = tmp_path / "qc.csv"
    csv_path.write_text(
        ",".join(QC_EXPORT_FIELDS)
        + "\n"
        + ",".join(_qc_row()[field] for field in QC_EXPORT_FIELDS)
        + "\n",
        encoding="utf-8",
    )
    assert load_qc_delistings_csv(csv_path)[0].qc_symbol == "ABC"


def test_b4_bridge_matches_by_normalized_name_and_date():
    qc = validate_qc_delisting_rows([_qc_row()])[0]
    edgar = EdgarDelisting(
        cik="101",
        company="ABC Therapeutics Corporation",
        form="25-NSE",
        filing_date=date(2020, 5, 1),
        accession="edgar-a",
        filename="edgar-a.txt",
    )

    assert normalize_company_name("The ABC Therapeutics, Inc. Common Stock") == "abc therapeutics"
    bridged = bridge_qc_to_edgar(
        [qc],
        [edgar],
        {"edgar-a": ReasonClass.PERFORMANCE},
    )

    assert bridged == [
        BridgeRow(
            qc_symbol="ABC",
            company_name="ABC Therapeutics Inc",
            delisting_date=date(2020, 5, 2),
            momentum_decile=1,
            value_bucket="high_bp",
            negative_book=False,
            bridge_status=MATCHED,
            cik="101",
            form25_date=date(2020, 5, 1),
            reason=ReasonClass.PERFORMANCE,
        )
    ]


def test_b4_bridge_marks_ambiguous_and_unmatched_cases():
    qc = validate_qc_delisting_rows([_qc_row()])[0]
    edgar_a = EdgarDelisting("1", "ABC Therapeutics Inc", "25", date(2020, 5, 1), "a", "a.txt")
    edgar_b = EdgarDelisting("2", "ABC Therapeutics Inc", "25", date(2020, 5, 1), "b", "b.txt")
    far = EdgarDelisting("3", "ABC Therapeutics Inc", "25", date(2020, 7, 1), "c", "c.txt")

    assert bridge_qc_to_edgar([qc], [edgar_a, edgar_b], {})[0].bridge_status == AMBIGUOUS
    assert bridge_qc_to_edgar([qc], [far], {})[0].bridge_status == UNMATCHED


def _bridge_row(decile: int, reason: ReasonClass, status: str = MATCHED) -> BridgeRow:
    return BridgeRow(
        qc_symbol=f"Q{decile}",
        company_name=f"Name {decile}",
        delisting_date=date(2020, 1, 1),
        momentum_decile=decile,
        value_bucket="bucket",
        negative_book=False,
        bridge_status=status,
        reason=reason,
    )


def _threshold_rows(*, total: int, classified: int, adverse: int,
                    decile1: int, decile10: int) -> list[BridgeRow]:
    rows: list[BridgeRow] = []
    rows.extend(_bridge_row(1, ReasonClass.PERFORMANCE) for _ in range(decile1))
    rows.extend(_bridge_row(10, ReasonClass.PERFORMANCE) for _ in range(decile10))
    remaining_adverse = adverse - decile1 - decile10
    rows.extend(_bridge_row(5, ReasonClass.BANKRUPTCY) for _ in range(remaining_adverse))
    rows.extend(
        _bridge_row(6, ReasonClass.ACQUISITION)
        for _ in range(classified - adverse)
    )
    rows.extend(
        _bridge_row(7, ReasonClass.UNKNOWN, UNMATCHED)
        for _ in range(total - classified)
    )
    return rows


def test_b5_threshold_counting_boundaries_and_unknowns_are_conservative():
    passing = evaluate_thresholds(
        _threshold_rows(total=100, classified=85, adverse=50, decile1=2, decile10=1)
    )
    assert passing.passed
    assert passing.mappability == 0.85
    assert passing.adverse_mass == 50
    assert passing.concentration_ratio == 2.0

    assert "mass" in evaluate_thresholds(
        _threshold_rows(total=100, classified=85, adverse=49, decile1=2, decile10=1)
    ).failed_thresholds
    assert "mappability" in evaluate_thresholds(
        _threshold_rows(total=100, classified=84, adverse=50, decile1=2, decile10=1)
    ).failed_thresholds
    assert "concentration" in evaluate_thresholds(
        _threshold_rows(total=400, classified=400, adverse=299, decile1=199, decile10=100)
    ).failed_thresholds
    assert evaluate_thresholds(
        _threshold_rows(total=400, classified=400, adverse=300, decile1=200, decile10=100)
    ).passed


def test_b5_rendered_artifact_stays_source_only():
    result = evaluate_thresholds(
        _threshold_rows(total=100, classified=84, adverse=50, decile1=2, decile10=1)
    )
    report = render_report(result)

    assert "Outcome: STOP" in report
    assert "mappability" in result.failed_thresholds
    assert_source_only_report(report)


def test_source_integrity_package_does_not_import_floor_or_reserved_tail_surfaces():
    package_dir = Path("apps/quant/advisor/source_integrity")
    blocked = [
        "advisor.back" + "test",
        "advisor.re" + "search.candidate",
        "Pre" + "RegConfig",
        "run" + "-floor",
        "HOLD" + "OUT",
    ]
    offenders = []
    for path in package_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in blocked:
            if token in text:
                offenders.append(f"{path}:{token}")

    assert offenders == []


def test_b5_thresholds_are_precommitted_values():
    t = DiagnosticThresholds()
    assert (t.adverse_mass, t.mappability, t.concentration) == (50, 0.85, 2.0)
