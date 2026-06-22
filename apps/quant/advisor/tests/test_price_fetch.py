from datetime import date

import pandas as pd

from advisor.data.price_fetch import build_price_fixture


def _price_frame(index: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Close": pd.Series(values, index=index, dtype="float64")})


def test_build_price_fixture_writes_floor_schema_and_drops_only_coverage_gaps(tmp_path):
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    frames = {
        "AAA": _price_frame(dates, [10.0, 11.0, 12.0]),
        "DROP": _price_frame(dates[:2], [20.0, 21.0]),
        "LOSER": _price_frame(dates, [30.0, 29.0, 28.0]),
        "SPY": _price_frame(dates, [100.0, 101.0, 102.0]),
    }
    calls = []

    def fake_getter(ticker: str, start: date, end: date) -> pd.DataFrame:
        calls.append((ticker, start, end))
        return frames[ticker].copy()

    out_path = tmp_path / "prices.csv"
    coverage = build_price_fixture(
        ["AAA", "DROP", "LOSER"],
        out_path,
        start="2020-01-02",
        end="2020-01-07",
        getter=fake_getter,
    )

    written = pd.read_csv(out_path, index_col=0, parse_dates=True)
    floor_header = pd.read_csv(
        "apps/quant/advisor/tests/fixtures/floor_prices.csv", nrows=0
    ).columns

    assert written.index.name == floor_header[0] == "Date"
    assert list(written.columns) == ["AAA", "LOSER", "SPY"]
    assert all(pd.api.types.is_float_dtype(written[c]) for c in written.columns)
    assert written["LOSER"].tolist() == [30.0, 29.0, 28.0]
    assert coverage["dropped"] == ["DROP"]
    assert coverage["coverage"]["DROP"] == 2
    assert coverage["coverage"]["LOSER"] == 3
    assert calls == [
        ("AAA", date(2020, 1, 2), date(2020, 1, 7)),
        ("DROP", date(2020, 1, 2), date(2020, 1, 7)),
        ("LOSER", date(2020, 1, 2), date(2020, 1, 7)),
        ("SPY", date(2020, 1, 2), date(2020, 1, 7)),
    ]
