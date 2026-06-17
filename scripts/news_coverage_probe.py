"""Operator-run coverage probe (NOT a unit test - hits the network).

Usage:
    $env:ALPHAVANTAGE_API_KEY="..."   # never commit keys
    python scripts/news_coverage_probe.py AAPL MSFT NVDA --as-of 2024-05-01

Prints, per ticker, how many headlines each configured source returns. Empty counts
(missing key / throttle / no coverage) degrade to a neutral sentiment signal - they are
NOT failures. This is an availability comparison, not a signal-quality claim: news
sentiment remains report-only and cannot close the floor (spec section 6/10).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Make the advisor package importable when run standalone (pytest uses pytest.ini's
# pythonpath=apps/quant; this script is run directly, so bootstrap the same path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "quant"))

from advisor.data.news_provider import AlphaVantageNewsProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="news_coverage_probe")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--as-of", default=date.today().isoformat())
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)

    sources = {"alpha_vantage": AlphaVantageNewsProvider()}
    have_keys = {"alpha_vantage": bool(os.environ.get("ALPHAVANTAGE_API_KEY"))}
    print(f"as_of={as_of} (time_to capped at as_of) | keys: {have_keys}")
    for ticker in args.tickers:
        for name, src in sources.items():
            heads = src.get_headlines(ticker, as_of)
            sample = heads[0] if heads else "-"
            print(f"  {ticker:6s} {name:14s} {len(heads):3d} headlines | e.g. {sample!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
