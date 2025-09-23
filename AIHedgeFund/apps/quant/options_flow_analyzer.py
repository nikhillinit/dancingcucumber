"""
Options Flow Analysis with Parallel Processing
==============================================
Multi-agent system for analyzing options flow and market maker positioning
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
from joblib import Parallel, delayed
import aiohttp
from asyncio import gather, create_task, Queue
import redis.asyncio as aioredis
from collections import defaultdict, deque
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class OptionsFlow:
    """Options flow data structure"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # call or put
    volume: int
    open_interest: int
    bid_ask_spread: float
    implied_volatility: float
    delta: float
    gamma: float
    vega: float
    theta: float
    timestamp: datetime
    unusual_activity: bool
    sentiment: str  # bullish, bearish, neutral


@dataclass
class GammaExposure:
    """Gamma exposure calculation"""
    symbol: str
    total_gamma: float
    call_gamma: float
    put_gamma: float
    net_gamma: float
    flip_point: float  # Zero gamma level
    timestamp: datetime


class OptionsFlowAgent(ray.remote):
    """Ray actor for distributed options analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.cache = {}

    async def analyze_options_chain(self, symbol: str) -> Dict[str, Any]:
        """Analyze complete options chain for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return {}

            # Parallel analysis of each expiration
            tasks = []
            for exp in expirations[:5]:  # Limit to near-term
                tasks.append(self._analyze_expiration(ticker, symbol, exp))

            results = await asyncio.gather(*tasks)

            # Aggregate results
            return self._aggregate_options_data(results)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {}

    async def _analyze_expiration(
        self,
        ticker: yf.Ticker,
        symbol: str,
        expiry: str
    ) -> List[OptionsFlow]:
        """Analyze single expiration date"""
        opt = ticker.option_chain(expiry)
        flows = []

        # Analyze calls
        for _, row in opt.calls.iterrows():
            flow = self._create_flow_object(symbol, row, expiry, 'call')
            if flow.unusual_activity:
                flows.append(flow)

        # Analyze puts
        for _, row in opt.puts.iterrows():
            flow = self._create_flow_object(symbol, row, expiry, 'put')
            if flow.unusual_activity:
                flows.append(flow)

        return flows

    def _create_flow_object(
        self,
        symbol: str,
        row: pd.Series,
        expiry: str,
        option_type: str
    ) -> OptionsFlow:
        """Create OptionsFlow object from row data"""
        volume = row.get('volume', 0) or 0
        oi = row.get('openInterest', 0) or 0

        # Detect unusual activity
        unusual = self._detect_unusual_activity(volume, oi)

        # Determine sentiment
        sentiment = self._determine_sentiment(row, option_type)

        return OptionsFlow(
            symbol=symbol,
            strike=row['strike'],
            expiry=pd.to_datetime(expiry),
            option_type=option_type,
            volume=volume,
            open_interest=oi,
            bid_ask_spread=row.get('ask', 0) - row.get('bid', 0),
            implied_volatility=row.get('impliedVolatility', 0),
            delta=self._estimate_delta(row, option_type),
            gamma=self._estimate_gamma(row),
            vega=self._estimate_vega(row),
            theta=self._estimate_theta(row),
            timestamp=datetime.now(),
            unusual_activity=unusual,
            sentiment=sentiment
        )

    def _detect_unusual_activity(self, volume: int, oi: int) -> bool:
        """Detect unusual options activity"""
        # Volume > 2x open interest suggests unusual activity
        if oi > 0:
            return volume > 2 * oi
        return volume > 1000  # High absolute volume

    def _determine_sentiment(self, row: pd.Series, option_type: str) -> str:
        """Determine market sentiment from options data"""
        if option_type == 'call':
            if row.get('volume', 0) > row.get('openInterest', 0):
                return 'bullish'
        else:  # put
            if row.get('volume', 0) > row.get('openInterest', 0):
                return 'bearish'
        return 'neutral'

    def _estimate_delta(self, row: pd.Series, option_type: str) -> float:
        """Estimate option delta"""
        moneyness = row.get('inTheMoney', False)
        if option_type == 'call':
            return 0.7 if moneyness else 0.3
        else:
            return -0.7 if moneyness else -0.3

    def _estimate_gamma(self, row: pd.Series) -> float:
        """Estimate option gamma"""
        # Simplified - peaks at ATM
        return 0.05 * np.exp(-abs(row.get('strike', 0) - row.get('lastPrice', 0)) / 10)

    def _estimate_vega(self, row: pd.Series) -> float:
        """Estimate option vega"""
        return row.get('impliedVolatility', 0.2) * 0.1

    def _estimate_theta(self, row: pd.Series) -> float:
        """Estimate option theta"""
        return -0.01  # Simplified daily decay

    def _aggregate_options_data(self, results: List[List[OptionsFlow]]) -> Dict:
        """Aggregate options flow data"""
        all_flows = [flow for flows in results for flow in flows]

        if not all_flows:
            return {}

        # Calculate aggregate metrics
        total_call_volume = sum(f.volume for f in all_flows if f.option_type == 'call')
        total_put_volume = sum(f.volume for f in all_flows if f.option_type == 'put')

        return {
            'put_call_ratio': total_put_volume / (total_call_volume + 1),
            'total_volume': total_call_volume + total_put_volume,
            'unusual_flows': len([f for f in all_flows if f.unusual_activity]),
            'bullish_flows': len([f for f in all_flows if f.sentiment == 'bullish']),
            'bearish_flows': len([f for f in all_flows if f.sentiment == 'bearish']),
            'top_flows': sorted(all_flows, key=lambda x: x.volume, reverse=True)[:10]
        }


class GammaExposureCalculator:
    """Calculate market maker gamma exposure (GEX)"""

    def __init__(self):
        self.spot_prices = {}
        self.executor = ProcessPoolExecutor(max_workers=4)

    async def calculate_gex(self, symbol: str, options_data: List[OptionsFlow]) -> GammaExposure:
        """Calculate total gamma exposure"""
        if not options_data:
            return GammaExposure(
                symbol=symbol,
                total_gamma=0,
                call_gamma=0,
                put_gamma=0,
                net_gamma=0,
                flip_point=0,
                timestamp=datetime.now()
            )

        # Get spot price
        spot = await self._get_spot_price(symbol)

        # Calculate gamma for calls and puts
        call_gamma = sum(
            flow.gamma * flow.open_interest * 100 * spot * 0.01
            for flow in options_data if flow.option_type == 'call'
        )

        put_gamma = -sum(
            flow.gamma * flow.open_interest * 100 * spot * 0.01
            for flow in options_data if flow.option_type == 'put'
        )

        total_gamma = call_gamma + put_gamma

        # Find zero gamma level (flip point)
        flip_point = self._calculate_flip_point(options_data, spot)

        return GammaExposure(
            symbol=symbol,
            total_gamma=total_gamma,
            call_gamma=call_gamma,
            put_gamma=put_gamma,
            net_gamma=total_gamma,
            flip_point=flip_point,
            timestamp=datetime.now()
        )

    async def _get_spot_price(self, symbol: str) -> float:
        """Get current spot price"""
        if symbol in self.spot_prices:
            return self.spot_prices[symbol]

        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('currentPrice', info.get('regularMarketPrice', 100))
        self.spot_prices[symbol] = price
        return price

    def _calculate_flip_point(self, options_data: List[OptionsFlow], spot: float) -> float:
        """Calculate zero gamma level"""
        # Simplified - find strike where gamma exposure flips
        strikes = sorted(set(flow.strike for flow in options_data))

        min_abs_gamma = float('inf')
        flip_strike = spot

        for strike in strikes:
            strike_gamma = sum(
                flow.gamma * flow.open_interest * np.sign(1 if flow.option_type == 'call' else -1)
                for flow in options_data if flow.strike == strike
            )

            if abs(strike_gamma) < min_abs_gamma:
                min_abs_gamma = abs(strike_gamma)
                flip_strike = strike

        return flip_strike


class OptionsFlowOrchestrator:
    """Orchestrate multi-agent options flow analysis"""

    def __init__(self, n_agents: int = 5):
        self.n_agents = n_agents
        self.agents = []
        self.gex_calculator = GammaExposureCalculator()
        self.redis_client = None
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize Ray actors for parallel processing"""
        ray.init(ignore_reinit_error=True)
        self.agents = [
            OptionsFlowAgent.remote(f"options_agent_{i}")
            for i in range(self.n_agents)
        ]

    async def initialize_redis(self):
        """Initialize Redis for caching"""
        try:
            self.redis_client = await aioredis.create_redis_pool('redis://localhost:6379')
        except Exception as e:
            logger.warning(f"Redis not available: {e}")

    async def analyze_multiple_symbols(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze options flow for multiple symbols in parallel"""
        # Distribute work across agents
        tasks = []
        for i, symbol in enumerate(symbols):
            agent = self.agents[i % self.n_agents]
            task = agent.analyze_options_chain.remote(symbol)
            tasks.append((symbol, task))

        # Gather results
        results = {}
        for symbol, task in tasks:
            try:
                options_data = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, task)
                )

                # Calculate GEX if we have data
                if options_data and 'top_flows' in options_data:
                    gex = await self.gex_calculator.calculate_gex(
                        symbol,
                        options_data['top_flows']
                    )
                    options_data['gamma_exposure'] = gex

                results[symbol] = options_data

                # Cache results
                await self._cache_results(symbol, options_data)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = {}

        return results

    async def _cache_results(self, symbol: str, data: Dict):
        """Cache results to Redis"""
        if self.redis_client:
            try:
                import json
                cache_key = f"options_flow:{symbol}"
                await self.redis_client.setex(
                    cache_key,
                    300,  # 5 minute TTL
                    json.dumps({
                        'put_call_ratio': data.get('put_call_ratio', 0),
                        'total_volume': data.get('total_volume', 0),
                        'unusual_flows': data.get('unusual_flows', 0),
                        'timestamp': datetime.now().isoformat()
                    })
                )
            except Exception as e:
                logger.debug(f"Caching failed: {e}")

    def interpret_options_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret options flow data for trading signals"""
        if not data:
            return {'signal': 'neutral', 'confidence': 0}

        signals = []

        # Put/Call ratio analysis
        pcr = data.get('put_call_ratio', 1)
        if pcr < 0.7:
            signals.append(('bullish', 0.7))
        elif pcr > 1.3:
            signals.append(('bearish', 0.7))

        # Unusual activity
        unusual = data.get('unusual_flows', 0)
        if unusual > 5:
            bullish = data.get('bullish_flows', 0)
            bearish = data.get('bearish_flows', 0)

            if bullish > bearish * 2:
                signals.append(('bullish', 0.8))
            elif bearish > bullish * 2:
                signals.append(('bearish', 0.8))

        # Gamma exposure
        if 'gamma_exposure' in data:
            gex = data['gamma_exposure']
            if gex.net_gamma > 0:
                signals.append(('low_volatility', 0.6))
            else:
                signals.append(('high_volatility', 0.6))

        # Aggregate signals
        if not signals:
            return {'signal': 'neutral', 'confidence': 0.5}

        # Weight average
        bullish_score = sum(conf for sig, conf in signals if sig == 'bullish')
        bearish_score = sum(conf for sig, conf in signals if sig == 'bearish')

        if bullish_score > bearish_score:
            return {
                'signal': 'bullish',
                'confidence': bullish_score / len(signals),
                'details': data
            }
        elif bearish_score > bullish_score:
            return {
                'signal': 'bearish',
                'confidence': bearish_score / len(signals),
                'details': data
            }
        else:
            return {
                'signal': 'neutral',
                'confidence': 0.5,
                'details': data
            }

    def cleanup(self):
        """Clean up resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of options flow analyzer"""
    orchestrator = OptionsFlowOrchestrator(n_agents=5)
    await orchestrator.initialize_redis()

    symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ']

    print("Analyzing options flow...")
    results = await orchestrator.analyze_multiple_symbols(symbols)

    for symbol, data in results.items():
        if data:
            interpretation = orchestrator.interpret_options_flow(data)
            print(f"\n{symbol}:")
            print(f"  Signal: {interpretation['signal']}")
            print(f"  Confidence: {interpretation['confidence']:.1%}")

            if 'put_call_ratio' in data:
                print(f"  Put/Call Ratio: {data['put_call_ratio']:.2f}")
            if 'unusual_flows' in data:
                print(f"  Unusual Flows: {data['unusual_flows']}")
            if 'gamma_exposure' in data:
                gex = data['gamma_exposure']
                print(f"  Net GEX: ${gex.net_gamma:,.0f}")
                print(f"  Flip Point: ${gex.flip_point:.2f}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())