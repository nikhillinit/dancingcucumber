from __future__ import annotations

import argparse
from datetime import date

from advisor.analysis.value_quality import evaluate
from advisor.backtest.walk_forward import disclosure_header
from advisor.data.provider import MarketDataProvider, YFinanceProvider


def run(provider: MarketDataProvider, ticker: str, as_of: date, critic=None) -> str:
    f = provider.get_fundamentals_asof(ticker, as_of)
    if f is None:
        return f"{ticker}: no point-in-time fundamentals available as of {as_of}\n{disclosure_header()}"
    sig = evaluate(f, as_of)
    line = f"{ticker} [{sig.direction.value}] confidence={sig.confidence:.0f} :: {sig.reasoning}"
    if critic is not None:
        verdict = critic(sig)
        line += f"\n  persona: {verdict.explanation}"
    return f"{line}\n{disclosure_header()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advisor", description="Single-family AI advisor (value/quality)")
    parser.add_argument("ticker")
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="YYYY-MM-DD point-in-time date")
    parser.add_argument("--explain", action="store_true",
                        help="append a persona explanation line (v1: read-only narration)")
    args = parser.parse_args(argv)
    critic = None
    if args.explain:
        from advisor.personas.overlay import PersonaVerdict
        critic = lambda sig: PersonaVerdict(1.0, f"{sig.direction.value} per value/quality family")
    print(run(YFinanceProvider(), args.ticker, date.fromisoformat(args.as_of), critic=critic))
    return 0
