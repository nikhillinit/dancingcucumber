from __future__ import annotations

import argparse
from datetime import date

from advisor.analysis.value_quality import evaluate
from advisor.backtest.walk_forward import disclosure_header
from advisor.data.provider import MarketDataProvider, YFinanceProvider


def run(provider: MarketDataProvider, ticker: str, as_of: date) -> str:
    f = provider.get_fundamentals_asof(ticker, as_of)
    if f is None:
        return f"{ticker}: no point-in-time fundamentals available as of {as_of}\n{disclosure_header()}"
    sig = evaluate(f, as_of)
    return (f"{ticker} [{sig.direction.value}] confidence={sig.confidence:.0f} :: {sig.reasoning}\n"
            f"{disclosure_header()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advisor", description="Single-family AI advisor (value/quality)")
    parser.add_argument("ticker")
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="YYYY-MM-DD point-in-time date")
    args = parser.parse_args(argv)
    print(run(YFinanceProvider(), args.ticker, date.fromisoformat(args.as_of)))
    return 0
