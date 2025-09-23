"""
Market Microstructure Analysis with Multi-Agent System
======================================================
Analyze order flow, liquidity, and market depth with parallel processing
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ray
from collections import deque, defaultdict
import websocket
import json
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for classification"""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    ICEBERG = "iceberg"
    DARK_POOL = "dark_pool"


@dataclass
class OrderFlowImbalance:
    """Order flow imbalance metrics"""
    symbol: str
    buy_volume: float
    sell_volume: float
    imbalance: float  # (buy - sell) / (buy + sell)
    buy_trades: int
    sell_trades: int
    avg_buy_size: float
    avg_sell_size: float
    large_buy_orders: int  # Institutional size
    large_sell_orders: int
    timestamp: datetime


@dataclass
class MarketDepth:
    """Level 2 market depth analysis"""
    symbol: str
    bid_levels: List[Tuple[float, float]]  # (price, size)
    ask_levels: List[Tuple[float, float]]
    total_bid_liquidity: float
    total_ask_liquidity: float
    bid_ask_ratio: float
    depth_imbalance: float
    support_levels: List[float]
    resistance_levels: List[float]
    timestamp: datetime


@dataclass
class MicrostructureSignal:
    """Aggregated microstructure signal"""
    symbol: str
    signal_type: str  # buy, sell, hold
    strength: float  # 0-1
    liquidity_score: float
    toxicity_score: float  # Adverse selection
    price_impact: float
    smart_money_flow: float
    timestamp: datetime


class OrderFlowAgent(ray.remote):
    """Agent for analyzing order flow"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.tick_buffer = deque(maxlen=10000)
        self.trade_classifier = TradeClassifier()

    async def analyze_tick_data(self, ticks: List[Dict]) -> OrderFlowImbalance:
        """Analyze tick-by-tick trade data"""
        buy_volume = 0
        sell_volume = 0
        buy_trades = 0
        sell_trades = 0
        buy_sizes = []
        sell_sizes = []
        large_threshold = 10000  # Define institutional size

        for tick in ticks:
            # Classify trade direction
            direction = self.trade_classifier.classify_trade(tick)

            if direction == 'buy':
                buy_volume += tick['size']
                buy_trades += 1
                buy_sizes.append(tick['size'])
            else:
                sell_volume += tick['size']
                sell_trades += 1
                sell_sizes.append(tick['size'])

        # Calculate metrics
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / (total_volume + 1) if total_volume > 0 else 0

        return OrderFlowImbalance(
            symbol=ticks[0]['symbol'] if ticks else 'UNKNOWN',
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            imbalance=imbalance,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            avg_buy_size=np.mean(buy_sizes) if buy_sizes else 0,
            avg_sell_size=np.mean(sell_sizes) if sell_sizes else 0,
            large_buy_orders=sum(1 for s in buy_sizes if s > large_threshold),
            large_sell_orders=sum(1 for s in sell_sizes if s > large_threshold),
            timestamp=datetime.now()
        )

    def detect_iceberg_orders(self, ticks: List[Dict]) -> List[Dict]:
        """Detect potential iceberg orders"""
        icebergs = []

        # Look for repeated executions at same price/size
        price_size_counts = defaultdict(int)

        for tick in ticks:
            key = (tick['price'], tick['size'])
            price_size_counts[key] += 1

            # If same price/size appears frequently, likely iceberg
            if price_size_counts[key] > 5:
                icebergs.append({
                    'price': tick['price'],
                    'size': tick['size'],
                    'count': price_size_counts[key],
                    'type': 'potential_iceberg'
                })

        return icebergs


class TradeClassifier:
    """Classify trades as buys or sells using various methods"""

    def classify_trade(self, tick: Dict) -> str:
        """Classify individual trade"""
        # Lee-Ready algorithm
        if 'bid' in tick and 'ask' in tick:
            mid = (tick['bid'] + tick['ask']) / 2

            if tick['price'] > mid:
                return 'buy'
            elif tick['price'] < mid:
                return 'sell'
            else:
                # Use tick rule for midpoint trades
                return self._tick_rule(tick)
        else:
            return self._tick_rule(tick)

    def _tick_rule(self, tick: Dict) -> str:
        """Simple tick rule classification"""
        if 'prev_price' in tick:
            if tick['price'] > tick['prev_price']:
                return 'buy'
            else:
                return 'sell'
        return 'neutral'


class MarketDepthAgent(ray.remote):
    """Agent for analyzing market depth and liquidity"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.depth_history = deque(maxlen=1000)

    async def analyze_order_book(self, order_book: Dict) -> MarketDepth:
        """Analyze level 2 order book data"""
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        # Calculate liquidity metrics
        total_bid_liquidity = sum(price * size for price, size in bids)
        total_ask_liquidity = sum(price * size for price, size in asks)

        bid_ask_ratio = total_bid_liquidity / (total_ask_liquidity + 1)
        depth_imbalance = (total_bid_liquidity - total_ask_liquidity) / (
            total_bid_liquidity + total_ask_liquidity + 1
        )

        # Identify support/resistance levels
        support_levels = self._identify_support_levels(bids)
        resistance_levels = self._identify_resistance_levels(asks)

        return MarketDepth(
            symbol=order_book.get('symbol', 'UNKNOWN'),
            bid_levels=bids[:10],  # Top 10 levels
            ask_levels=asks[:10],
            total_bid_liquidity=total_bid_liquidity,
            total_ask_liquidity=total_ask_liquidity,
            bid_ask_ratio=bid_ask_ratio,
            depth_imbalance=depth_imbalance,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            timestamp=datetime.now()
        )

    def _identify_support_levels(self, bids: List[Tuple[float, float]]) -> List[float]:
        """Identify significant support levels"""
        if not bids:
            return []

        # Find levels with large size
        sizes = [size for _, size in bids]
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        support_levels = []
        for price, size in bids:
            if size > mean_size + 2 * std_size:
                support_levels.append(price)

        return support_levels[:3]  # Top 3 levels

    def _identify_resistance_levels(self, asks: List[Tuple[float, float]]) -> List[float]:
        """Identify significant resistance levels"""
        if not asks:
            return []

        sizes = [size for _, size in asks]
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        resistance_levels = []
        for price, size in asks:
            if size > mean_size + 2 * std_size:
                resistance_levels.append(price)

        return resistance_levels[:3]


class LiquidityAgent(ray.remote):
    """Agent for analyzing market liquidity and toxicity"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.spread_history = deque(maxlen=1000)

    async def analyze_liquidity(
        self,
        order_book: Dict,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Analyze market liquidity metrics"""
        # Calculate various liquidity measures
        spread = self._calculate_spread(order_book)
        effective_spread = self._calculate_effective_spread(trades)
        realized_spread = self._calculate_realized_spread(trades)

        # Kyle's lambda (price impact)
        price_impact = self._calculate_price_impact(trades)

        # Amihud illiquidity
        amihud = self._calculate_amihud_illiquidity(trades)

        # Roll's implicit spread
        roll_spread = self._calculate_roll_spread(trades)

        # Toxicity (VPIN)
        toxicity = self._calculate_vpin(trades)

        return {
            'spread': spread,
            'effective_spread': effective_spread,
            'realized_spread': realized_spread,
            'price_impact': price_impact,
            'amihud_illiquidity': amihud,
            'roll_spread': roll_spread,
            'toxicity': toxicity,
            'liquidity_score': self._calculate_liquidity_score(spread, price_impact, amihud)
        }

    def _calculate_spread(self, order_book: Dict) -> float:
        """Calculate bid-ask spread"""
        if 'bids' in order_book and 'asks' in order_book:
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
                return (best_ask - best_bid) / ((best_ask + best_bid) / 2)
        return 0

    def _calculate_effective_spread(self, trades: List[Dict]) -> float:
        """Calculate effective spread"""
        if not trades:
            return 0

        spreads = []
        for trade in trades:
            if 'mid' in trade:
                eff_spread = 2 * abs(trade['price'] - trade['mid']) / trade['mid']
                spreads.append(eff_spread)

        return np.mean(spreads) if spreads else 0

    def _calculate_realized_spread(self, trades: List[Dict]) -> float:
        """Calculate realized spread (temporary vs permanent impact)"""
        if len(trades) < 2:
            return 0

        realized_spreads = []
        for i in range(len(trades) - 1):
            if 'mid' in trades[i]:
                current_trade = trades[i]
                future_mid = trades[i + 1].get('mid', current_trade['price'])

                realized = 2 * (current_trade['price'] - current_trade['mid']) * (
                    current_trade['price'] - future_mid
                ) / current_trade['mid']

                realized_spreads.append(realized)

        return np.mean(realized_spreads) if realized_spreads else 0

    def _calculate_price_impact(self, trades: List[Dict]) -> float:
        """Calculate Kyle's lambda (price impact coefficient)"""
        if len(trades) < 10:
            return 0

        # Regress price changes on signed volume
        price_changes = []
        signed_volumes = []

        for i in range(1, len(trades)):
            price_change = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
            signed_volume = trades[i]['size'] * (1 if trades[i].get('direction') == 'buy' else -1)

            price_changes.append(price_change)
            signed_volumes.append(signed_volume)

        if signed_volumes:
            # Simple linear regression
            x = np.array(signed_volumes)
            y = np.array(price_changes)

            if np.std(x) > 0:
                lambda_kyle = np.cov(x, y)[0, 1] / np.var(x)
                return abs(lambda_kyle)

        return 0

    def _calculate_amihud_illiquidity(self, trades: List[Dict]) -> float:
        """Calculate Amihud illiquidity measure"""
        if not trades:
            return 0

        illiquidity = []
        for trade in trades:
            if trade['size'] > 0:
                price_return = abs(trade.get('return', 0))
                volume = trade['size'] * trade['price']
                illiquidity.append(price_return / volume)

        return np.mean(illiquidity) * 1e6 if illiquidity else 0

    def _calculate_roll_spread(self, trades: List[Dict]) -> float:
        """Calculate Roll's implicit spread"""
        if len(trades) < 2:
            return 0

        price_changes = [
            trades[i]['price'] - trades[i-1]['price']
            for i in range(1, len(trades))
        ]

        if price_changes:
            cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
            if cov < 0:
                return 2 * np.sqrt(-cov)

        return 0

    def _calculate_vpin(self, trades: List[Dict]) -> float:
        """Calculate Volume-Synchronized Probability of Informed Trading"""
        if not trades:
            return 0

        # Simplified VPIN calculation
        bucket_size = 50  # trades per bucket
        buckets = [trades[i:i+bucket_size] for i in range(0, len(trades), bucket_size)]

        vpin_values = []
        for bucket in buckets:
            buy_volume = sum(t['size'] for t in bucket if t.get('direction') == 'buy')
            sell_volume = sum(t['size'] for t in bucket if t.get('direction') == 'sell')
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                vpin = abs(buy_volume - sell_volume) / total_volume
                vpin_values.append(vpin)

        return np.mean(vpin_values) if vpin_values else 0

    def _calculate_liquidity_score(
        self,
        spread: float,
        price_impact: float,
        amihud: float
    ) -> float:
        """Calculate overall liquidity score (0-1, higher is better)"""
        # Normalize and invert (lower spread/impact is better)
        spread_score = max(0, 1 - spread * 100)
        impact_score = max(0, 1 - price_impact * 1000)
        amihud_score = max(0, 1 - amihud)

        # Weighted average
        return 0.4 * spread_score + 0.3 * impact_score + 0.3 * amihud_score


class SmartMoneyFlowAgent(ray.remote):
    """Agent for detecting smart money (institutional) flow"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def analyze_smart_money(
        self,
        order_flow: OrderFlowImbalance,
        market_depth: MarketDepth,
        trades: List[Dict]
    ) -> float:
        """Detect and quantify smart money flow"""
        smart_money_score = 0

        # Large order detection
        if order_flow.large_buy_orders > order_flow.large_sell_orders * 2:
            smart_money_score += 0.3
        elif order_flow.large_sell_orders > order_flow.large_buy_orders * 2:
            smart_money_score -= 0.3

        # Order book imbalance at key levels
        if market_depth.depth_imbalance > 0.2:
            smart_money_score += 0.2

        # Time-weighted average price (TWAP) detection
        twap_score = self._detect_twap_execution(trades)
        smart_money_score += twap_score

        # Hidden liquidity detection
        hidden_score = self._detect_hidden_liquidity(market_depth, trades)
        smart_money_score += hidden_score

        return np.clip(smart_money_score, -1, 1)

    def _detect_twap_execution(self, trades: List[Dict]) -> float:
        """Detect TWAP-like execution patterns"""
        if len(trades) < 20:
            return 0

        # Look for consistent sized trades over time
        sizes = [t['size'] for t in trades[-20:]]
        times = [t.get('timestamp', i) for i, t in enumerate(trades[-20:])]

        # Check for regular intervals and similar sizes
        size_std = np.std(sizes)
        mean_size = np.mean(sizes)

        if size_std / mean_size < 0.1:  # Low variation in size
            return 0.2  # Likely algorithmic execution

        return 0

    def _detect_hidden_liquidity(
        self,
        market_depth: MarketDepth,
        trades: List[Dict]
    ) -> float:
        """Detect hidden/dark pool liquidity"""
        if not trades:
            return 0

        # Look for trades executed inside the spread
        inside_spread_trades = 0
        for trade in trades:
            if 'bid' in trade and 'ask' in trade:
                if trade['bid'] < trade['price'] < trade['ask']:
                    inside_spread_trades += 1

        if len(trades) > 0:
            hidden_ratio = inside_spread_trades / len(trades)
            if hidden_ratio > 0.3:
                return 0.2  # Significant hidden liquidity

        return 0


class MicrostructureOrchestrator:
    """Orchestrate multi-agent microstructure analysis"""

    def __init__(self, n_agents: int = 4):
        ray.init(ignore_reinit_error=True)

        self.order_flow_agent = OrderFlowAgent.remote("order_flow")
        self.market_depth_agent = MarketDepthAgent.remote("market_depth")
        self.liquidity_agent = LiquidityAgent.remote("liquidity")
        self.smart_money_agent = SmartMoneyFlowAgent.remote("smart_money")

    async def analyze_microstructure(
        self,
        symbol: str,
        ticks: List[Dict],
        order_book: Dict
    ) -> MicrostructureSignal:
        """Perform comprehensive microstructure analysis"""
        # Parallel analysis
        tasks = [
            self.order_flow_agent.analyze_tick_data.remote(ticks),
            self.market_depth_agent.analyze_order_book.remote(order_book),
            self.liquidity_agent.analyze_liquidity.remote(order_book, ticks),
        ]

        results = await asyncio.gather(*[
            asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            ) for task in tasks
        ])

        order_flow = results[0]
        market_depth = results[1]
        liquidity_metrics = results[2]

        # Smart money analysis
        smart_money_task = self.smart_money_agent.analyze_smart_money.remote(
            order_flow, market_depth, ticks
        )
        smart_money_flow = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, smart_money_task)
        )

        # Generate signal
        signal = self._generate_signal(
            order_flow,
            market_depth,
            liquidity_metrics,
            smart_money_flow
        )

        return signal

    def _generate_signal(
        self,
        order_flow: OrderFlowImbalance,
        market_depth: MarketDepth,
        liquidity: Dict[str, float],
        smart_money: float
    ) -> MicrostructureSignal:
        """Generate trading signal from microstructure analysis"""
        # Combine multiple signals
        score = 0

        # Order flow imbalance
        score += order_flow.imbalance * 0.3

        # Market depth imbalance
        score += market_depth.depth_imbalance * 0.2

        # Smart money flow
        score += smart_money * 0.4

        # Liquidity adjustment
        liquidity_score = liquidity.get('liquidity_score', 0.5)
        score *= liquidity_score

        # Determine signal
        if score > 0.2:
            signal_type = 'buy'
        elif score < -0.2:
            signal_type = 'sell'
        else:
            signal_type = 'hold'

        return MicrostructureSignal(
            symbol=order_flow.symbol,
            signal_type=signal_type,
            strength=abs(score),
            liquidity_score=liquidity_score,
            toxicity_score=liquidity.get('toxicity', 0),
            price_impact=liquidity.get('price_impact', 0),
            smart_money_flow=smart_money,
            timestamp=datetime.now()
        )

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of microstructure analyzer"""
    orchestrator = MicrostructureOrchestrator()

    # Simulated tick data
    ticks = [
        {'symbol': 'AAPL', 'price': 150.01, 'size': 100, 'bid': 150.00, 'ask': 150.02},
        {'symbol': 'AAPL', 'price': 150.02, 'size': 500, 'bid': 150.01, 'ask': 150.03},
        {'symbol': 'AAPL', 'price': 150.01, 'size': 1000, 'bid': 150.00, 'ask': 150.02},
    ] * 10

    # Simulated order book
    order_book = {
        'symbol': 'AAPL',
        'bids': [(150.00, 1000), (149.99, 2000), (149.98, 1500)],
        'asks': [(150.02, 800), (150.03, 1200), (150.04, 900)]
    }

    signal = await orchestrator.analyze_microstructure('AAPL', ticks, order_book)

    print(f"Microstructure Signal: {signal.signal_type}")
    print(f"Strength: {signal.strength:.2%}")
    print(f"Liquidity Score: {signal.liquidity_score:.2f}")
    print(f"Smart Money Flow: {signal.smart_money_flow:.2f}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())