from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

from advisor.analysis.news_scorer import lexicon_score
from advisor.analysis.value_quality import evaluate
from advisor.backtest.walk_forward import disclosure_header
from advisor.data.fred_provider import FredApiProvider, FredProvider
from advisor.data.news_provider import AlphaVantageNewsProvider, CompositeNewsProvider, NewsProvider
from advisor.data.provider import MarketDataProvider, YFinanceProvider
from advisor.pipeline.families import (
    close_series, make_macro_coro, make_momentum_coro, make_sentiment_coro,
    make_trend_coro, make_value_quality_coro,
)
from advisor.pipeline.run import run_pipeline

# Conservative, illustrative defaults for report-only sizing. These do NOT authorize
# real capital; the floor still blocks (spec section 6). Override on the CLI.
DEFAULT_NET_LIQ = 100_000.0
DEFAULT_VOL = 0.30
DEFAULT_CORRELATION = 0.50


def run(provider: MarketDataProvider, ticker: str, as_of: date, critic=None) -> str:
    """Single-family (value/quality) path - unchanged back-compat behavior."""
    f = provider.get_fundamentals_asof(ticker, as_of)
    if f is None:
        return f"{ticker}: no point-in-time fundamentals available as of {as_of}\n{disclosure_header()}"
    sig = evaluate(f, as_of)
    line = f"{ticker} [{sig.direction.value}] confidence={sig.confidence:.0f} :: {sig.reasoning}"
    if critic is not None:
        verdict = critic(sig)
        line += f"\n  persona: {verdict.explanation}"
    return f"{line}\n{disclosure_header()}"


def _latest_price(provider: MarketDataProvider, ticker: str, as_of: date) -> float:
    df = provider.get_prices(ticker, as_of - timedelta(days=10), as_of)
    s = close_series(df).dropna()
    return float(s.iloc[-1]) if len(s) else 0.0


def run_all(provider: MarketDataProvider, fred: FredProvider, news: NewsProvider,
            scorer, ticker: str, as_of: date, net_liq: float, vol: float,
            correlation: float, persona_critic=None) -> str:
    """Five-family report-only path: assembles all coros and calls run_pipeline."""
    coros = [
        make_value_quality_coro(provider, ticker),
        make_trend_coro(provider, ticker),
        make_momentum_coro(provider, ticker),
        make_macro_coro(fred),
        make_sentiment_coro(news, scorer, ticker),
    ]
    price = _latest_price(provider, ticker, as_of)
    decision = asyncio.run(run_pipeline(
        ticker, as_of, price=price, net_liq=net_liq, vol=vol, correlation=correlation,
        family_coros=coros, persona_critic=persona_critic))
    line = (f"{decision.ticker} [{decision.bundle_direction}] {decision.action} "
            f"qty={decision.quantity} :: {decision.reasoning}")
    return f"{line}\n{disclosure_header()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="advisor", description="AI advisor (report-only)")
    parser.add_argument("ticker")
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="YYYY-MM-DD point-in-time date")
    parser.add_argument("--families", choices=["value", "all"], default="value",
                        help="'value' (single-family, default) or 'all' (five-family pipeline)")
    parser.add_argument("--net-liq", type=float, default=DEFAULT_NET_LIQ,
                        help="illustrative net liquidation value for sizing (report-only)")
    parser.add_argument("--vol", type=float, default=DEFAULT_VOL,
                        help="illustrative annualized volatility for the position limit")
    parser.add_argument("--correlation", type=float, default=DEFAULT_CORRELATION,
                        help="illustrative correlation for the position limit")
    parser.add_argument("--explain", action="store_true",
                        help="append a persona explanation line (read-only narration)")
    args = parser.parse_args(argv)
    as_of = date.fromisoformat(args.as_of)

    if args.families == "all":
        provider = YFinanceProvider()
        fred = FredApiProvider()
        news = CompositeNewsProvider([AlphaVantageNewsProvider()])
        critic = None
        if args.explain:
            from advisor.personas.overlay import PersonaVerdict
            critic = lambda d: PersonaVerdict(1.0, f"{d.bundle_direction} per 5-family ensemble")
        print(run_all(provider, fred, news, lexicon_score, args.ticker, as_of,
                      net_liq=args.net_liq, vol=args.vol, correlation=args.correlation,
                      persona_critic=critic))
        return 0

    critic = None
    if args.explain:
        from advisor.personas.overlay import PersonaVerdict
        critic = lambda sig: PersonaVerdict(1.0, f"{sig.direction.value} per value/quality family")
    print(run(YFinanceProvider(), args.ticker, as_of, critic=critic))
    return 0
