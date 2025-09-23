"""
Statistical Arbitrage Engine with Multi-Agent Processing
========================================================
Pairs trading, cointegration, and mean reversion strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import ray
from joblib import Parallel, delayed
import asyncio
from scipy import stats
from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TradingPair:
    """Statistical arbitrage pair"""
    symbol1: str
    symbol2: str
    cointegration_score: float
    p_value: float
    hedge_ratio: float
    half_life: int
    spread_mean: float
    spread_std: float
    current_zscore: float
    sharpe_ratio: float
    timestamp: datetime


@dataclass
class ArbitrageSignal:
    """Arbitrage trading signal"""
    pair: TradingPair
    signal_type: str  # entry_long, entry_short, exit, hold
    confidence: float
    expected_return: float
    risk_score: float
    position_size: Tuple[float, float]  # (symbol1_size, symbol2_size)
    stop_loss: float
    take_profit: float
    timestamp: datetime


class CointegrationAgent(ray.remote):
    """Agent for cointegration analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.cointegrated_pairs = []

    async def find_cointegrated_pairs(
        self,
        price_data: Dict[str, pd.Series],
        threshold: float = 0.05
    ) -> List[TradingPair]:
        """Find cointegrated pairs using Engle-Granger test"""
        symbols = list(price_data.keys())
        pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pair = await self._test_cointegration(
                    symbols[i], symbols[j],
                    price_data[symbols[i]], price_data[symbols[j]],
                    threshold
                )

                if pair:
                    pairs.append(pair)

        # Sort by cointegration score
        pairs.sort(key=lambda x: x.p_value)
        self.cointegrated_pairs = pairs[:20]  # Keep top 20 pairs

        return self.cointegrated_pairs

    async def _test_cointegration(
        self,
        symbol1: str,
        symbol2: str,
        series1: pd.Series,
        series2: pd.Series,
        threshold: float
    ) -> Optional[TradingPair]:
        """Test cointegration between two series"""
        try:
            # Engle-Granger test
            score, p_value, _ = coint(series1, series2)

            if p_value < threshold:
                # Calculate hedge ratio using OLS
                model = OLS(series1, series2).fit()
                hedge_ratio = model.params[0]

                # Calculate spread
                spread = series1 - hedge_ratio * series2

                # Test spread stationarity
                adf_result = adfuller(spread)
                if adf_result[1] < threshold:  # Spread is stationary
                    # Calculate half-life
                    half_life = self._calculate_half_life(spread)

                    # Calculate current z-score
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    current_zscore = (spread.iloc[-1] - spread_mean) / spread_std

                    # Calculate Sharpe ratio
                    returns = spread.pct_change().dropna()
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

                    return TradingPair(
                        symbol1=symbol1,
                        symbol2=symbol2,
                        cointegration_score=score,
                        p_value=p_value,
                        hedge_ratio=hedge_ratio,
                        half_life=half_life,
                        spread_mean=spread_mean,
                        spread_std=spread_std,
                        current_zscore=current_zscore,
                        sharpe_ratio=sharpe,
                        timestamp=datetime.now()
                    )

        except Exception as e:
            logger.debug(f"Cointegration test failed for {symbol1}-{symbol2}: {e}")

        return None

    def _calculate_half_life(self, spread: pd.Series) -> int:
        """Calculate mean reversion half-life using Ornstein-Uhlenbeck"""
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align the series
        spread_lag = spread_lag[spread_diff.index]

        # OLS regression
        model = OLS(spread_diff, spread_lag).fit()
        theta = -model.params[0]

        if theta > 0:
            half_life = int(np.log(2) / theta)
        else:
            half_life = 100  # Default large value

        return min(half_life, 100)


class KalmanFilterAgent(ray.remote):
    """Agent for Kalman filter-based hedge ratio estimation"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.filters = {}

    def initialize_kalman_filter(self, pair_id: str) -> KalmanFilter:
        """Initialize Kalman filter for dynamic hedge ratio"""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # State vector [hedge_ratio, drift]
        kf.x = np.array([[1.0], [0.0]])

        # State transition matrix
        kf.F = np.array([[1, 1],
                        [0, 1]])

        # Measurement matrix
        kf.H = np.array([[1, 0]])

        # Process noise covariance
        kf.Q = np.array([[0.0001, 0],
                        [0, 0.00001]])

        # Measurement noise covariance
        kf.R = np.array([[0.001]])

        # Initial covariance
        kf.P = np.eye(2) * 0.1

        self.filters[pair_id] = kf
        return kf

    async def update_hedge_ratio(
        self,
        pair: TradingPair,
        price1: float,
        price2: float
    ) -> float:
        """Update hedge ratio using Kalman filter"""
        pair_id = f"{pair.symbol1}_{pair.symbol2}"

        if pair_id not in self.filters:
            kf = self.initialize_kalman_filter(pair_id)
        else:
            kf = self.filters[pair_id]

        # Predict
        kf.predict()

        # Update with measurement
        spread = price1 - kf.x[0, 0] * price2
        kf.update(spread)

        # Return updated hedge ratio
        return float(kf.x[0, 0])


class MeanReversionAgent(ray.remote):
    """Agent for mean reversion signal generation"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.entry_threshold = 2.0  # Z-score for entry
        self.exit_threshold = 0.5   # Z-score for exit
        self.stop_loss_multiplier = 3.0

    async def generate_signal(
        self,
        pair: TradingPair,
        current_prices: Dict[str, float]
    ) -> ArbitrageSignal:
        """Generate mean reversion trading signal"""
        # Calculate current spread
        price1 = current_prices[pair.symbol1]
        price2 = current_prices[pair.symbol2]
        current_spread = price1 - pair.hedge_ratio * price2

        # Update z-score
        current_zscore = (current_spread - pair.spread_mean) / pair.spread_std

        # Generate signal
        if abs(current_zscore) > self.entry_threshold:
            # Entry signal
            if current_zscore > self.entry_threshold:
                # Spread too high - short spread
                signal_type = 'entry_short'
                position_size = (-1.0, pair.hedge_ratio)  # Short symbol1, long symbol2
            else:
                # Spread too low - long spread
                signal_type = 'entry_long'
                position_size = (1.0, -pair.hedge_ratio)  # Long symbol1, short symbol2

            confidence = min(abs(current_zscore) / 3, 1.0)
            expected_return = abs(current_zscore - pair.spread_mean) * pair.spread_std / price1

        elif abs(current_zscore) < self.exit_threshold:
            # Exit signal
            signal_type = 'exit'
            position_size = (0, 0)
            confidence = 0.8
            expected_return = 0

        else:
            # Hold signal
            signal_type = 'hold'
            position_size = (0, 0)
            confidence = 0.5
            expected_return = 0

        # Calculate risk metrics
        risk_score = abs(current_zscore) / self.stop_loss_multiplier
        stop_loss = pair.spread_mean + (self.stop_loss_multiplier * pair.spread_std * np.sign(current_zscore))
        take_profit = pair.spread_mean

        return ArbitrageSignal(
            pair=pair,
            signal_type=signal_type,
            confidence=confidence,
            expected_return=expected_return,
            risk_score=risk_score,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now()
        )


class MultiLegArbitrageAgent(ray.remote):
    """Agent for multi-leg arbitrage strategies"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def find_triangular_arbitrage(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.Series]
    ) -> List[Dict[str, Any]]:
        """Find triangular arbitrage opportunities"""
        opportunities = []

        # For each triplet of symbols
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                for k in range(j + 1, len(symbols)):
                    opportunity = self._check_triangular_arbitrage(
                        symbols[i], symbols[j], symbols[k],
                        price_data
                    )

                    if opportunity:
                        opportunities.append(opportunity)

        return opportunities

    def _check_triangular_arbitrage(
        self,
        symbol1: str,
        symbol2: str,
        symbol3: str,
        price_data: Dict[str, pd.Series]
    ) -> Optional[Dict[str, Any]]:
        """Check for triangular arbitrage opportunity"""
        try:
            # Calculate synthetic price
            prices1 = price_data[symbol1].iloc[-100:]
            prices2 = price_data[symbol2].iloc[-100:]
            prices3 = price_data[symbol3].iloc[-100:]

            # Check if triangular relationship exists
            synthetic = prices1 / prices2 * prices3

            # Calculate deviation from parity
            deviation = (synthetic / prices1 - 1).iloc[-1]

            if abs(deviation) > 0.01:  # 1% threshold
                return {
                    'symbols': [symbol1, symbol2, symbol3],
                    'deviation': deviation,
                    'expected_profit': abs(deviation) * 0.7,  # Account for costs
                    'confidence': min(abs(deviation) * 10, 0.9),
                    'timestamp': datetime.now()
                }

        except Exception as e:
            logger.debug(f"Triangular arbitrage check failed: {e}")

        return None


class StatisticalArbitrageOrchestrator:
    """Orchestrate statistical arbitrage strategies"""

    def __init__(self, n_agents: int = 4):
        ray.init(ignore_reinit_error=True)

        self.cointegration_agent = CointegrationAgent.remote("cointegration")
        self.kalman_agent = KalmanFilterAgent.remote("kalman")
        self.mean_reversion_agent = MeanReversionAgent.remote("mean_reversion")
        self.multi_leg_agent = MultiLegArbitrageAgent.remote("multi_leg")

        self.active_pairs = []
        self.positions = {}

    async def find_arbitrage_opportunities(
        self,
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Find all arbitrage opportunities"""
        # Convert to price series
        price_series = {
            symbol: df['close'] if 'close' in df.columns else df.iloc[:, 0]
            for symbol, df in price_data.items()
        }

        # Find cointegrated pairs
        pairs_task = self.cointegration_agent.find_cointegrated_pairs.remote(
            price_series, threshold=0.05
        )
        cointegrated_pairs = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, pairs_task)
        )

        # Find triangular arbitrage
        symbols = list(price_data.keys())
        triangular_task = self.multi_leg_agent.find_triangular_arbitrage.remote(
            symbols, price_series
        )
        triangular_opportunities = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, triangular_task)
        )

        return {
            'pairs': cointegrated_pairs,
            'triangular': triangular_opportunities,
            'timestamp': datetime.now()
        }

    async def generate_trading_signals(
        self,
        pairs: List[TradingPair],
        current_prices: Dict[str, float]
    ) -> List[ArbitrageSignal]:
        """Generate trading signals for all pairs"""
        signals = []

        for pair in pairs:
            # Update hedge ratio with Kalman filter
            if pair.symbol1 in current_prices and pair.symbol2 in current_prices:
                updated_ratio_task = self.kalman_agent.update_hedge_ratio.remote(
                    pair,
                    current_prices[pair.symbol1],
                    current_prices[pair.symbol2]
                )
                pair.hedge_ratio = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, updated_ratio_task)
                )

                # Generate signal
                signal_task = self.mean_reversion_agent.generate_signal.remote(
                    pair, current_prices
                )
                signal = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, signal_task)
                )

                signals.append(signal)

        return signals

    def calculate_portfolio_metrics(
        self,
        signals: List[ArbitrageSignal]
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        if not signals:
            return {}

        # Aggregate metrics
        total_expected_return = sum(s.expected_return for s in signals if s.signal_type.startswith('entry'))
        avg_confidence = np.mean([s.confidence for s in signals])
        avg_risk = np.mean([s.risk_score for s in signals])

        # Count by signal type
        signal_counts = {}
        for signal in signals:
            signal_counts[signal.signal_type] = signal_counts.get(signal.signal_type, 0) + 1

        # Calculate diversification
        unique_symbols = set()
        for signal in signals:
            unique_symbols.add(signal.pair.symbol1)
            unique_symbols.add(signal.pair.symbol2)

        diversification = len(unique_symbols) / (len(signals) * 2) if signals else 0

        return {
            'total_expected_return': total_expected_return,
            'avg_confidence': avg_confidence,
            'avg_risk_score': avg_risk,
            'n_opportunities': len(signals),
            'signal_distribution': signal_counts,
            'diversification': diversification
        }

    async def execute_arbitrage_strategy(
        self,
        price_data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute complete arbitrage strategy"""
        # Find opportunities
        opportunities = await self.find_arbitrage_opportunities(price_data)

        # Generate signals
        signals = await self.generate_trading_signals(
            opportunities['pairs'][:10],  # Top 10 pairs
            current_prices
        )

        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(signals)

        # Filter high-confidence signals
        high_confidence_signals = [s for s in signals if s.confidence > 0.7]

        return {
            'opportunities': opportunities,
            'signals': high_confidence_signals,
            'metrics': metrics,
            'active_pairs': len(opportunities['pairs']),
            'timestamp': datetime.now()
        }

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of statistical arbitrage engine"""
    orchestrator = StatisticalArbitrageOrchestrator()

    # Generate sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    price_data = {}
    current_prices = {}

    for i, symbol in enumerate(symbols):
        # Simulated price data
        base_price = 100 + i * 50
        prices = base_price + np.cumsum(np.random.randn(500) * 2)
        price_data[symbol] = pd.DataFrame({'close': prices})
        current_prices[symbol] = prices[-1]

    # Execute strategy
    result = await orchestrator.execute_arbitrage_strategy(price_data, current_prices)

    print(f"Found {result['active_pairs']} cointegrated pairs")
    print(f"Generated {len(result['signals'])} high-confidence signals")

    if result['signals']:
        print("\nTop Arbitrage Signals:")
        for signal in result['signals'][:5]:
            print(f"  {signal.pair.symbol1}-{signal.pair.symbol2}: "
                  f"{signal.signal_type} (confidence: {signal.confidence:.1%})")

    print(f"\nPortfolio Metrics:")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())