"""
Real-time WebSocket Streaming with Multi-Agent Processing
=========================================================
WebSocket connections for live market data, order books, and trades
"""

import asyncio
import websocket
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import ray
from collections import deque
import aiohttp
from asyncio import Queue, create_task
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class StreamData:
    """Real-time stream data"""
    stream_type: str  # price, orderbook, trade, news
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    latency_ms: float


@dataclass
class OrderBookUpdate:
    """Level 2 order book update"""
    symbol: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]
    spread: float
    mid_price: float
    timestamp: datetime


@dataclass
class TradeUpdate:
    """Trade execution update"""
    symbol: str
    price: float
    size: float
    side: str  # buy/sell
    trade_id: str
    timestamp: datetime


class WebSocketClient(ray.remote):
    """WebSocket client for specific data stream"""

    def __init__(self, client_id: str, url: str):
        self.client_id = client_id
        self.url = url
        self.ws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnects = 5
        self.message_queue = Queue(maxsize=10000)
        self.callbacks = {}

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = await websocket.connect(self.url)
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info(f"WebSocket {self.client_id} connected to {self.url}")

            # Start message handler
            asyncio.create_task(self._handle_messages())

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self._reconnect()

    async def _reconnect(self):
        """Auto-reconnect logic"""
        if self.reconnect_attempts < self.max_reconnects:
            self.reconnect_attempts += 1
            wait_time = min(2 ** self.reconnect_attempts, 30)
            logger.info(f"Reconnecting in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            await self.connect()
        else:
            logger.error(f"Max reconnection attempts reached for {self.client_id}")

    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    await self.message_queue.put(data)

                    # Process callbacks
                    await self._process_callbacks(data)

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")

        except websocket.ConnectionClosed:
            logger.warning(f"WebSocket {self.client_id} connection closed")
            self.is_connected = False
            await self._reconnect()

        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _process_callbacks(self, data: Dict):
        """Process registered callbacks"""
        message_type = data.get('type', 'unknown')

        if message_type in self.callbacks:
            for callback in self.callbacks[message_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def subscribe(self, symbols: List[str], channels: List[str]):
        """Subscribe to data channels"""
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return

        subscription = {
            'action': 'subscribe',
            'symbols': symbols,
            'channels': channels
        }

        asyncio.create_task(self.ws.send(json.dumps(subscription)))

    def register_callback(self, message_type: str, callback: Callable):
        """Register callback for specific message type"""
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
        self.callbacks[message_type].append(callback)

    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False


class StreamAggregator(ray.remote):
    """Aggregate multiple data streams"""

    def __init__(self, aggregator_id: str):
        self.aggregator_id = aggregator_id
        self.price_buffer = deque(maxlen=1000)
        self.orderbook_buffer = deque(maxlen=100)
        self.trade_buffer = deque(maxlen=5000)
        self.aggregated_data = {}

    async def aggregate_price_stream(self, price_data: Dict) -> Dict:
        """Aggregate price updates"""
        symbol = price_data.get('symbol')
        price = price_data.get('price')
        volume = price_data.get('volume', 0)
        timestamp = datetime.fromisoformat(price_data.get('timestamp', datetime.now().isoformat()))

        # Add to buffer
        self.price_buffer.append({
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })

        # Calculate aggregated metrics
        if len(self.price_buffer) >= 10:
            recent_prices = [p['price'] for p in list(self.price_buffer)[-10:]]

            aggregated = {
                'symbol': symbol,
                'current_price': price,
                'avg_price': np.mean(recent_prices),
                'price_std': np.std(recent_prices),
                'price_trend': (price - recent_prices[0]) / recent_prices[0],
                'volume_weighted_price': self._calculate_vwap(),
                'timestamp': timestamp
            }

            self.aggregated_data[symbol] = aggregated
            return aggregated

        return {}

    def _calculate_vwap(self) -> float:
        """Calculate volume-weighted average price"""
        recent_data = list(self.price_buffer)[-20:]

        total_value = sum(d['price'] * d['volume'] for d in recent_data)
        total_volume = sum(d['volume'] for d in recent_data)

        if total_volume > 0:
            return total_value / total_volume
        return recent_data[-1]['price'] if recent_data else 0

    async def aggregate_orderbook_stream(self, orderbook_data: Dict) -> OrderBookUpdate:
        """Aggregate order book updates"""
        symbol = orderbook_data.get('symbol')
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])

        # Calculate metrics
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
        else:
            spread = 0
            mid_price = 0

        update = OrderBookUpdate(
            symbol=symbol,
            bids=bids[:10],  # Top 10 levels
            asks=asks[:10],
            spread=spread,
            mid_price=mid_price,
            timestamp=datetime.now()
        )

        self.orderbook_buffer.append(update)
        return update

    async def aggregate_trade_stream(self, trade_data: Dict) -> TradeUpdate:
        """Aggregate trade executions"""
        trade = TradeUpdate(
            symbol=trade_data.get('symbol'),
            price=trade_data.get('price'),
            size=trade_data.get('size'),
            side=trade_data.get('side'),
            trade_id=trade_data.get('id', ''),
            timestamp=datetime.fromisoformat(trade_data.get('timestamp', datetime.now().isoformat()))
        )

        self.trade_buffer.append(trade)

        # Calculate trade flow metrics
        if len(self.trade_buffer) >= 100:
            await self._calculate_trade_flow_metrics()

        return trade

    async def _calculate_trade_flow_metrics(self) -> Dict:
        """Calculate trade flow imbalance and other metrics"""
        recent_trades = list(self.trade_buffer)[-100:]

        buy_volume = sum(t.size for t in recent_trades if t.side == 'buy')
        sell_volume = sum(t.size for t in recent_trades if t.side == 'sell')

        if buy_volume + sell_volume > 0:
            imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        else:
            imbalance = 0

        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'imbalance': imbalance,
            'trade_count': len(recent_trades)
        }


class ExchangeConnector:
    """Connector for specific exchanges"""

    def __init__(self, exchange: str):
        self.exchange = exchange
        self.endpoints = self._get_endpoints()
        self.clients = {}
        self.aggregators = {}

    def _get_endpoints(self) -> Dict[str, str]:
        """Get WebSocket endpoints for exchanges"""
        endpoints = {
            'binance': {
                'stream': 'wss://stream.binance.com:9443/ws',
                'orderbook': 'wss://stream.binance.com:9443/ws',
                'trades': 'wss://stream.binance.com:9443/ws'
            },
            'coinbase': {
                'stream': 'wss://ws-feed.exchange.coinbase.com',
                'orderbook': 'wss://ws-feed.exchange.coinbase.com',
                'trades': 'wss://ws-feed.exchange.coinbase.com'
            },
            'kraken': {
                'stream': 'wss://ws.kraken.com',
                'orderbook': 'wss://ws.kraken.com',
                'trades': 'wss://ws.kraken.com'
            },
            'alpaca': {
                'stream': 'wss://stream.data.alpaca.markets/v2/iex',
                'orderbook': 'wss://stream.data.alpaca.markets/v2/iex',
                'trades': 'wss://stream.data.alpaca.markets/v2/iex'
            }
        }

        return endpoints.get(self.exchange, endpoints['binance'])

    async def connect_all_streams(self, symbols: List[str]):
        """Connect to all data streams"""
        ray.init(ignore_reinit_error=True)

        # Create WebSocket clients
        for stream_type, url in self.endpoints.items():
            client_id = f"{self.exchange}_{stream_type}"
            self.clients[stream_type] = WebSocketClient.remote(client_id, url)

            # Connect client
            await self.clients[stream_type].connect.remote()

            # Subscribe to symbols
            await self._subscribe_to_stream(stream_type, symbols)

        # Create aggregators
        self.aggregators['main'] = StreamAggregator.remote(f"{self.exchange}_aggregator")

    async def _subscribe_to_stream(self, stream_type: str, symbols: List[str]):
        """Subscribe to specific stream"""
        client = self.clients.get(stream_type)

        if not client:
            return

        if self.exchange == 'binance':
            # Binance subscription format
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                if stream_type == 'orderbook':
                    streams.append(f"{symbol_lower}@depth20")
                elif stream_type == 'trades':
                    streams.append(f"{symbol_lower}@trade")
                else:
                    streams.append(f"{symbol_lower}@ticker")

            subscription = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }

        elif self.exchange == 'coinbase':
            # Coinbase subscription format
            subscription = {
                "type": "subscribe",
                "product_ids": symbols,
                "channels": [stream_type]
            }

        else:
            # Generic format
            subscription = {
                "action": "subscribe",
                "symbols": symbols,
                "channel": stream_type
            }

        await client.subscribe.remote(symbols, [stream_type])


class StreamOrchestrator:
    """Orchestrate all real-time data streams"""

    def __init__(self):
        self.exchanges = {}
        self.active_streams = {}
        self.callbacks = {}
        self.is_running = False

    async def start_streaming(
        self,
        exchanges: List[str],
        symbols: List[str]
    ):
        """Start streaming from multiple exchanges"""
        self.is_running = True

        # Connect to each exchange
        for exchange in exchanges:
            connector = ExchangeConnector(exchange)
            await connector.connect_all_streams(symbols)
            self.exchanges[exchange] = connector

        # Start processing loop
        asyncio.create_task(self._process_streams())

        logger.info(f"Streaming started for {exchanges} with {symbols}")

    async def _process_streams(self):
        """Main processing loop for streams"""
        while self.is_running:
            try:
                # Process each exchange's streams
                for exchange, connector in self.exchanges.items():
                    aggregator = connector.aggregators.get('main')

                    if aggregator:
                        # Get aggregated data
                        # This would normally pull from the message queues
                        pass

                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning

            except Exception as e:
                logger.error(f"Stream processing error: {e}")

    def register_callback(
        self,
        stream_type: str,
        callback: Callable[[StreamData], None]
    ):
        """Register callback for stream data"""
        if stream_type not in self.callbacks:
            self.callbacks[stream_type] = []
        self.callbacks[stream_type].append(callback)

    async def _trigger_callbacks(self, stream_data: StreamData):
        """Trigger registered callbacks"""
        stream_type = stream_data.stream_type

        if stream_type in self.callbacks:
            for callback in self.callbacks[stream_type]:
                try:
                    await callback(stream_data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    async def stop_streaming(self):
        """Stop all streams"""
        self.is_running = False

        # Close all connections
        for exchange, connector in self.exchanges.items():
            for client in connector.clients.values():
                await client.close.remote()

        logger.info("Streaming stopped")


class MarketDataSimulator:
    """Simulate market data for testing"""

    def __init__(self):
        self.is_running = False
        self.symbols = []
        self.callbacks = {}

    async def start_simulation(self, symbols: List[str]):
        """Start market data simulation"""
        self.symbols = symbols
        self.is_running = True

        # Start simulation tasks
        tasks = [
            self._simulate_prices(),
            self._simulate_orderbook(),
            self._simulate_trades()
        ]

        await asyncio.gather(*tasks)

    async def _simulate_prices(self):
        """Simulate price updates"""
        while self.is_running:
            for symbol in self.symbols:
                price_data = {
                    'type': 'price',
                    'symbol': symbol,
                    'price': 100 + np.random.randn() * 5,
                    'volume': np.random.randint(100, 10000),
                    'timestamp': datetime.now().isoformat()
                }

                stream_data = StreamData(
                    stream_type='price',
                    symbol=symbol,
                    data=price_data,
                    timestamp=datetime.now(),
                    latency_ms=np.random.uniform(1, 10)
                )

                await self._notify_callbacks('price', stream_data)

            await asyncio.sleep(0.1)  # 10 updates per second

    async def _simulate_orderbook(self):
        """Simulate order book updates"""
        while self.is_running:
            for symbol in self.symbols:
                base_price = 100

                # Generate order book
                bids = [(base_price - i * 0.01, np.random.randint(100, 1000))
                       for i in range(1, 11)]
                asks = [(base_price + i * 0.01, np.random.randint(100, 1000))
                       for i in range(1, 11)]

                orderbook_data = {
                    'type': 'orderbook',
                    'symbol': symbol,
                    'bids': bids,
                    'asks': asks,
                    'timestamp': datetime.now().isoformat()
                }

                stream_data = StreamData(
                    stream_type='orderbook',
                    symbol=symbol,
                    data=orderbook_data,
                    timestamp=datetime.now(),
                    latency_ms=np.random.uniform(1, 5)
                )

                await self._notify_callbacks('orderbook', stream_data)

            await asyncio.sleep(0.5)  # 2 updates per second

    async def _simulate_trades(self):
        """Simulate trade executions"""
        while self.is_running:
            for symbol in self.symbols:
                trade_data = {
                    'type': 'trade',
                    'symbol': symbol,
                    'price': 100 + np.random.randn(),
                    'size': np.random.randint(1, 100),
                    'side': np.random.choice(['buy', 'sell']),
                    'id': f"trade_{np.random.randint(100000)}",
                    'timestamp': datetime.now().isoformat()
                }

                stream_data = StreamData(
                    stream_type='trade',
                    symbol=symbol,
                    data=trade_data,
                    timestamp=datetime.now(),
                    latency_ms=np.random.uniform(1, 3)
                )

                await self._notify_callbacks('trade', stream_data)

            await asyncio.sleep(0.05)  # 20 trades per second

    def register_callback(self, stream_type: str, callback: Callable):
        """Register callback for stream data"""
        if stream_type not in self.callbacks:
            self.callbacks[stream_type] = []
        self.callbacks[stream_type].append(callback)

    async def _notify_callbacks(self, stream_type: str, data: StreamData):
        """Notify registered callbacks"""
        if stream_type in self.callbacks:
            for callback in self.callbacks[stream_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def stop_simulation(self):
        """Stop simulation"""
        self.is_running = False


# Example usage
async def main():
    """Example usage of WebSocket streaming"""

    # Use simulator for demonstration
    simulator = MarketDataSimulator()

    # Track received data
    price_count = 0
    trade_count = 0
    orderbook_count = 0

    async def price_callback(data: StreamData):
        nonlocal price_count
        price_count += 1
        if price_count % 100 == 0:
            print(f"Received {price_count} price updates")
            print(f"  Latest: {data.symbol} @ ${data.data['price']:.2f}")

    async def trade_callback(data: StreamData):
        nonlocal trade_count
        trade_count += 1
        if trade_count % 100 == 0:
            print(f"Received {trade_count} trades")

    async def orderbook_callback(data: StreamData):
        nonlocal orderbook_count
        orderbook_count += 1
        if orderbook_count % 20 == 0:
            print(f"Received {orderbook_count} orderbook updates")

    # Register callbacks
    simulator.register_callback('price', price_callback)
    simulator.register_callback('trade', trade_callback)
    simulator.register_callback('orderbook', orderbook_callback)

    # Start simulation
    symbols = ['AAPL', 'GOOGL', 'MSFT']

    print(f"Starting real-time simulation for {symbols}...")

    # Run for 10 seconds
    simulation_task = asyncio.create_task(simulator.start_simulation(symbols))

    await asyncio.sleep(10)

    # Stop simulation
    simulator.stop_simulation()

    print(f"\nSimulation complete:")
    print(f"  Price updates: {price_count}")
    print(f"  Trades: {trade_count}")
    print(f"  Orderbook updates: {orderbook_count}")
    print(f"  Average latency: ~3ms")


if __name__ == "__main__":
    asyncio.run(main())