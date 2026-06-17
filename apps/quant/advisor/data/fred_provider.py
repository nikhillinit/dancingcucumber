from __future__ import annotations

import os
from datetime import date
from typing import Callable, Protocol

import pandas as pd

FRED_SERIES_T10Y2Y = "T10Y2Y"
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredProvider(Protocol):
    def get_series(self, series_id: str, start: date, end: date) -> pd.Series: ...


class FredApiProvider:
    """Thin FRED adapter. Network-bound; unit-tested via an injected ``http_get``.

    Reads FRED_API_KEY from the environment. Returns an empty Series when the key is
    missing or the request fails, so the macro coro degrades to neutral (spec section 10).
    """

    def __init__(self, api_key: str | None = None, http_get: Callable | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")
        self._http_get = http_get

    def get_series(self, series_id: str, start: date, end: date) -> pd.Series:
        if not self._api_key:
            return pd.Series(dtype="float64")
        getter = self._http_get
        if getter is None:  # pragma: no cover - network path
            import requests
            getter = requests.get
        try:
            resp = getter(_FRED_URL, params={
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),  # as-of upper bound
            }, timeout=15)
            observations = resp.json().get("observations", [])
        except Exception:
            return pd.Series(dtype="float64")
        idx: list = []
        vals: list[float] = []
        for o in observations:
            try:
                vals.append(float(o.get("value")))
            except (TypeError, ValueError):
                vals.append(float("nan"))  # FRED uses "." for missing
            idx.append(pd.Timestamp(o.get("date")))
        return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype="float64")
