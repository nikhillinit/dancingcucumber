from datetime import date

import pandas as pd

from advisor.data.fred_provider import FRED_SERIES_T10Y2Y, FredApiProvider


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_keyless_provider_returns_empty_series():
    s = FredApiProvider(api_key="").get_series(FRED_SERIES_T10Y2Y, date(2024, 1, 1), date(2024, 5, 1))
    assert isinstance(s, pd.Series) and s.empty


def test_parses_observations_and_bounds_end_at_as_of():
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = params
        return _Resp({"observations": [
            {"date": "2024-04-30", "value": "1.20"},
            {"date": "2024-05-01", "value": "."},  # FRED missing marker -> NaN
        ]})

    s = FredApiProvider(api_key="k", http_get=fake_get).get_series(
        FRED_SERIES_T10Y2Y, date(2024, 4, 1), date(2024, 5, 1))
    assert len(s) == 2
    assert float(s.iloc[0]) == 1.20
    assert pd.isna(s.iloc[1])
    assert calls["params"]["observation_end"] == "2024-05-01"  # as-of upper bound


def test_http_error_degrades_to_empty():
    def boom(url, params=None, timeout=None):
        raise RuntimeError("network down")

    s = FredApiProvider(api_key="k", http_get=boom).get_series(
        FRED_SERIES_T10Y2Y, date(2024, 4, 1), date(2024, 5, 1))
    assert s.empty
