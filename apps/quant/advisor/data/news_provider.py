from __future__ import annotations

import os
from datetime import date
from typing import Callable, Protocol

_AV_URL = "https://www.alphavantage.co/query"


class NewsProvider(Protocol):
    def get_headlines(self, ticker: str, as_of: date) -> list[str]: ...


class CompositeNewsProvider:
    """Union + dedupe headlines across providers for redundancy.

    A keyless/down/raising source contributes nothing; only an all-empty result yields []
    (which the sentiment coro maps to neutral). Redundancy buys availability, NOT signal
    validity - the floor still blocks.
    """

    def __init__(self, providers: list[NewsProvider]) -> None:
        self._providers = list(providers)

    def get_headlines(self, ticker: str, as_of: date) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self._providers:
            try:
                headlines = p.get_headlines(ticker, as_of)
            except Exception:
                headlines = []
            for h in headlines or []:
                key = " ".join(str(h).lower().split())
                if key and key not in seen:
                    seen.add(key)
                    out.append(h)
        return out


class AlphaVantageNewsProvider:
    """Alpha Vantage NEWS_SENTIMENT adapter. Network-bound; unit-tested via injected http_get.

    Reads ALPHAVANTAGE_API_KEY from env. Caps ``time_to`` at as_of. Treats a missing key,
    a throttle response ({"Note"/"Information": ...}), or any error as unavailable -> [].
    """

    def __init__(self, api_key: str | None = None, http_get: Callable | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self._http_get = http_get

    def get_headlines(self, ticker: str, as_of: date) -> list[str]:
        if not self._api_key:
            return []
        getter = self._http_get
        if getter is None:  # pragma: no cover - network path
            import requests
            getter = requests.get
        try:
            resp = getter(_AV_URL, params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_to": as_of.strftime("%Y%m%dT2359"),  # as-of upper bound
                "limit": 50,
                "apikey": self._api_key,
            }, timeout=15)
            data = resp.json()
        except Exception:
            return []
        if not isinstance(data, dict) or "feed" not in data:  # throttle / error
            return []
        return [a["title"] for a in data.get("feed", []) if isinstance(a, dict) and a.get("title")]
