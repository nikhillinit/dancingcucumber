from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Awaitable, Callable

from advisor.portfolio.allocator import allocate, ensemble_vote
from advisor.risk.limits import position_limit
from advisor.schemas import SignalBundle

FamilyCoro = Callable[[date], Awaitable]


@dataclass(frozen=True)
class Decision:
    ticker: str
    action: str
    quantity: int
    bundle_direction: str
    reasoning: str


async def run_pipeline(ticker: str, as_of: date, price: float, net_liq: float,
                       vol: float, correlation: float,
                       family_coros: list[FamilyCoro]) -> Decision:
    signals = await asyncio.gather(*(coro(as_of) for coro in family_coros))
    bundle = SignalBundle(ticker=ticker, as_of=as_of, signals=list(signals))
    direction, _ = ensemble_vote(bundle)
    limit = position_limit(net_liq, vol=vol, correlation=correlation)
    alloc = allocate(bundle, price=price, position_limit_dollars=limit)
    return Decision(ticker=ticker, action=alloc.action, quantity=alloc.quantity,
                    bundle_direction=direction.value, reasoning=alloc.reasoning)
