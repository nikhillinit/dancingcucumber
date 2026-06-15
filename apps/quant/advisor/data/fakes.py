from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from advisor.data.provider import Fundamentals


@dataclass
class FakeProvider:
    fundamentals: dict[str, Fundamentals] = field(default_factory=dict)
    prices: dict[str, pd.Series] = field(default_factory=dict)

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        s = self.prices.get(ticker, pd.Series(dtype=float))
        return pd.DataFrame({"Close": s})

    def get_fundamentals_asof(self, ticker: str, as_of: date) -> Fundamentals | None:
        return self.fundamentals.get(ticker)
