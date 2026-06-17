from datetime import date

from advisor.data.news_provider import AlphaVantageNewsProvider, CompositeNewsProvider
from advisor.tests.fakes import FakeNewsProvider


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_composite_unions_and_dedupes():
    a = FakeNewsProvider(["Earnings beat", "Record profit"])
    b = FakeNewsProvider(["earnings  beat", "New product launch"])  # 1st dup (case/space)
    out = CompositeNewsProvider([a, b]).get_headlines("AAPL", date(2024, 5, 1))
    assert out == ["Earnings beat", "Record profit", "New product launch"]


def test_composite_all_empty_is_empty():
    assert CompositeNewsProvider([FakeNewsProvider([]), FakeNewsProvider([])]).get_headlines(
        "AAPL", date(2024, 5, 1)) == []


def test_composite_swallows_a_raising_source():
    good = FakeNewsProvider(["Strong guidance"])
    out = CompositeNewsProvider([FakeNewsProvider(raises=True), good]).get_headlines(
        "AAPL", date(2024, 5, 1))
    assert out == ["Strong guidance"]


def test_av_keyless_returns_empty():
    assert AlphaVantageNewsProvider(api_key="").get_headlines("AAPL", date(2024, 5, 1)) == []


def test_av_bounds_time_to_at_as_of_and_parses_titles():
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls["params"] = params
        return _Resp({"feed": [{"title": "Earnings beat"}, {"title": "Record profit"}, {"x": 1}]})

    out = AlphaVantageNewsProvider(api_key="k", http_get=fake_get).get_headlines(
        "AAPL", date(2024, 5, 1))
    assert out == ["Earnings beat", "Record profit"]
    assert calls["params"]["time_to"].startswith("20240501T")  # as-of upper bound
    assert calls["params"]["tickers"] == "AAPL"


def test_av_throttle_note_returns_empty():
    def throttled(url, params=None, timeout=None):
        return _Resp({"Note": "rate limit"})

    assert AlphaVantageNewsProvider(api_key="k", http_get=throttled).get_headlines(
        "AAPL", date(2024, 5, 1)) == []
