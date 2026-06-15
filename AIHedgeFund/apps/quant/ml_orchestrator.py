"""
Unified ML Orchestrator
======================
Coordinates all ML models for optimal trading decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import supported ML components
from .finbert_sentiment_analyzer import SentimentEnhancedPersona
from .informer_predictor import InformerIntegration

logger = logging.getLogger(__name__)


@dataclass
class MLSignal:
    """Unified ML signal"""
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    model_source: str
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any]


class MLOrchestrator:
    """Orchestrates all ML models for trading decisions"""

    def __init__(self):
        # Initialize supported ML components
        self.sentiment = SentimentEnhancedPersona()
        self.informer = InformerIntegration()

        # Model weights for ensemble
        self.model_weights = {
            'sentiment': 0.50,
            'informer': 0.50
        }

        self.executor = ThreadPoolExecutor(max_workers=2)

    async def generate_comprehensive_signals(
        self,
        symbols: List[str],
        timeframe: str = 'daily'
    ) -> Dict[str, MLSignal]:
        """Generate signals from all ML models"""

        logger.info(f"Generating comprehensive ML signals for {symbols}")

        # Run all models in parallel
        tasks = []

        for symbol in symbols:
            tasks.append(self._generate_symbol_signals(symbol, timeframe))

        # Wait for all signals
        all_signals = await asyncio.gather(*tasks)

        # Combine signals by symbol
        combined_signals = {}
        for symbol, signals in zip(symbols, all_signals):
            combined_signals[symbol] = self._ensemble_signals(signals)

        return combined_signals

    async def _generate_symbol_signals(
        self,
        symbol: str,
        timeframe: str
    ) -> List[MLSignal]:
        """Generate signals for a single symbol from all models"""

        signals = []

        # Run models in parallel
        loop = asyncio.get_event_loop()

        # Sentiment signal
        sentiment_future = loop.run_in_executor(
            self.executor,
            self._get_sentiment_signal,
            symbol,
            timeframe
        )

        # Informer signal
        informer_future = loop.run_in_executor(
            self.executor,
            self._get_informer_signal,
            symbol,
            timeframe
        )

        # Collect all signals
        sentiment_signal = await sentiment_future
        informer_signal = await informer_future

        signals.extend([
            sentiment_signal,
            informer_signal
        ])

        return [s for s in signals if s is not None]

    def _get_sentiment_signal(self, symbol: str, timeframe: str) -> Optional[MLSignal]:
        """Get signal from sentiment analysis"""

        try:
            # Get sentiment-based signal
            signal = self.sentiment.signal_generator.generate_signal(symbol, timeframe)

            return MLSignal(
                symbol=symbol,
                action=signal['signal'],
                confidence=signal['confidence'],
                model_source='sentiment',
                reasoning=signal['reasoning'],
                timestamp=signal['timestamp'],
                metadata={
                    'sentiment': signal['current_sentiment'],
                    'momentum': signal['sentiment_momentum']
                }
            )

        except Exception as e:
            logger.error(f"Sentiment signal generation failed: {e}")
            return None

    def _get_informer_signal(self, symbol: str, timeframe: str) -> Optional[MLSignal]:
        """Get signal from Informer model"""

        try:
            # Generate Informer predictions
            # This would use real data in production
            predictions = pd.DataFrame({
                'prediction': np.random.randn(24) * 0.01 + 1.01
            })

            expected_return = predictions['prediction'].iloc[-1] - 1

            if expected_return > 0.01:
                action = 'buy'
            elif expected_return < -0.01:
                action = 'sell'
            else:
                action = 'hold'

            return MLSignal(
                symbol=symbol,
                action=action,
                confidence=min(abs(expected_return) * 50, 0.9),
                model_source='informer',
                reasoning=f"Informer predicts {expected_return:.2%} return",
                timestamp=datetime.now(),
                metadata={'prediction_horizon': 24}
            )

        except Exception as e:
            logger.error(f"Informer signal generation failed: {e}")
            return None

    def _ensemble_signals(self, signals: List[MLSignal]) -> MLSignal:
        """Combine multiple signals into ensemble signal"""

        if not signals:
            return MLSignal(
                symbol="",
                action="hold",
                confidence=0.0,
                model_source="ensemble",
                reasoning="No signals available",
                timestamp=datetime.now(),
                metadata={}
            )

        # Count votes for each action
        action_scores = {'buy': 0, 'sell': 0, 'hold': 0}
        total_weight = 0

        for signal in signals:
            weight = self.model_weights.get(signal.model_source, 0.1) * signal.confidence
            action_scores[signal.action] += weight
            total_weight += weight

        # Normalize scores
        if total_weight > 0:
            for action in action_scores:
                action_scores[action] /= total_weight

        # Get ensemble action
        ensemble_action = max(action_scores, key=action_scores.get)
        ensemble_confidence = action_scores[ensemble_action]

        # Generate reasoning
        model_votes = {}
        for signal in signals:
            if signal.model_source not in model_votes:
                model_votes[signal.model_source] = signal.action

        reasoning = f"Ensemble decision based on {len(signals)} models. "
        reasoning += f"Votes: {', '.join([f'{m}={a}' for m, a in model_votes.items()])}"

        return MLSignal(
            symbol=signals[0].symbol,
            action=ensemble_action,
            confidence=ensemble_confidence,
            model_source="ensemble",
            reasoning=reasoning,
            timestamp=datetime.now(),
            metadata={
                'action_scores': action_scores,
                'model_count': len(signals),
                'individual_signals': [
                    {
                        'model': s.model_source,
                        'action': s.action,
                        'confidence': s.confidence
                    }
                    for s in signals
                ]
            }
        )


class AdaptiveModelSelector:
    """Dynamically selects best models based on market conditions"""

    def __init__(self):
        self.orchestrator = MLOrchestrator()
        self.performance_history = {}
        self.market_regime = None

    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime"""

        # Calculate regime indicators
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        trend = (market_data['close'].iloc[-1] / market_data['close'].iloc[-20]) - 1

        # Classify regime
        if volatility > 0.03:
            regime = 'high_volatility'
        elif trend > 0.05:
            regime = 'bull'
        elif trend < -0.05:
            regime = 'bear'
        else:
            regime = 'sideways'

        self.market_regime = regime
        return regime

    def adjust_model_weights(self, regime: str):
        """Adjust model weights based on market regime"""

        regime_weights = {
            'bull': {
                'sentiment': 0.35,
                'informer': 0.65
            },
            'bear': {
                'sentiment': 0.55,
                'informer': 0.45
            },
            'high_volatility': {
                'sentiment': 0.50,
                'informer': 0.50
            },
            'sideways': {
                'sentiment': 0.45,
                'informer': 0.55
            }
        }

        self.orchestrator.model_weights = regime_weights.get(regime, self.orchestrator.model_weights)

        logger.info(f"Adjusted model weights for {regime} regime: {self.orchestrator.model_weights}")

    def track_performance(self, signal: MLSignal, actual_return: float):
        """Track model performance for continuous improvement"""

        model = signal.model_source

        if model not in self.performance_history:
            self.performance_history[model] = []

        # Calculate signal accuracy
        signal_correct = (
            (signal.action == 'buy' and actual_return > 0) or
            (signal.action == 'sell' and actual_return < 0) or
            (signal.action == 'hold' and abs(actual_return) < 0.01)
        )

        self.performance_history[model].append({
            'timestamp': signal.timestamp,
            'correct': signal_correct,
            'confidence': signal.confidence,
            'actual_return': actual_return,
            'regime': self.market_regime
        })

        # Update weights if enough history
        if len(self.performance_history[model]) >= 20:
            self._update_model_weight(model)

    def _update_model_weight(self, model: str):
        """Update model weight based on recent performance"""

        recent_performance = self.performance_history[model][-20:]
        accuracy = sum(p['correct'] for p in recent_performance) / len(recent_performance)

        # Adjust weight based on accuracy
        current_weight = self.orchestrator.model_weights[model]

        if accuracy > 0.6:
            new_weight = min(current_weight * 1.1, 0.4)
        elif accuracy < 0.4:
            new_weight = max(current_weight * 0.9, 0.05)
        else:
            new_weight = current_weight

        # Normalize weights
        total_adjustment = new_weight - current_weight
        self.orchestrator.model_weights[model] = new_weight

        # Distribute adjustment to other models
        other_models = [m for m in self.orchestrator.model_weights if m != model]
        for m in other_models:
            self.orchestrator.model_weights[m] -= total_adjustment / len(other_models)

        # Ensure weights sum to 1
        total = sum(self.orchestrator.model_weights.values())
        for m in self.orchestrator.model_weights:
            self.orchestrator.model_weights[m] /= total


class MLPortfolioManager:
    """Manages portfolio based on ML signals"""

    def __init__(self, initial_capital: float = 1000000):
        self.orchestrator = MLOrchestrator()
        self.selector = AdaptiveModelSelector()
        self.capital = initial_capital
        self.positions = {}
        self.performance_metrics = []

    async def execute_trading_day(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Execute a full trading day with ML signals"""

        # Detect market regime
        regime = self.selector.detect_market_regime(market_data[symbols[0]])

        # Adjust model weights
        self.selector.adjust_model_weights(regime)

        # Generate signals
        signals = await self.orchestrator.generate_comprehensive_signals(symbols)

        # Execute trades
        trades = []
        for symbol, signal in signals.items():
            trade = self._execute_trade(symbol, signal, market_data[symbol])
            if trade:
                trades.append(trade)

        # Update portfolio metrics
        portfolio_value = self._calculate_portfolio_value(market_data)

        return {
            'regime': regime,
            'signals': signals,
            'trades': trades,
            'portfolio_value': portfolio_value,
            'positions': self.positions.copy()
        }

    def _execute_trade(
        self,
        symbol: str,
        signal: MLSignal,
        market_data: pd.DataFrame
    ) -> Optional[Dict]:
        """Execute trade based on signal"""

        current_price = market_data['close'].iloc[-1]

        # Position sizing based on confidence
        position_size = self.capital * 0.1 * signal.confidence

        trade = None

        if signal.action == 'buy' and symbol not in self.positions:
            # Buy signal
            shares = int(position_size / current_price)
            if shares > 0 and position_size <= self.capital:
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': current_price,
                    'entry_time': signal.timestamp
                }
                self.capital -= shares * current_price

                trade = {
                    'symbol': symbol,
                    'action': 'buy',
                    'shares': shares,
                    'price': current_price,
                    'value': shares * current_price,
                    'signal': signal
                }

        elif signal.action == 'sell' and symbol in self.positions:
            # Sell signal
            position = self.positions[symbol]
            exit_value = position['shares'] * current_price
            pnl = exit_value - (position['shares'] * position['entry_price'])

            self.capital += exit_value
            del self.positions[symbol]

            trade = {
                'symbol': symbol,
                'action': 'sell',
                'shares': position['shares'],
                'price': current_price,
                'value': exit_value,
                'pnl': pnl,
                'signal': signal
            }

        return trade

    def _calculate_portfolio_value(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value"""

        total_value = self.capital

        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                total_value += position['shares'] * current_price

        return total_value


# Integration function
async def run_ml_trading_system(symbols: List[str], days: int = 30):
    """Run the complete ML trading system"""

    manager = MLPortfolioManager()

    results = []

    for day in range(days):
        logger.info(f"Trading day {day + 1}")

        # Get market data (placeholder - would use real data)
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = pd.DataFrame({
                'close': np.random.randn(100) * 2 + 100,
                'volume': np.random.randint(1000000, 10000000, 100)
            })

        # Execute trading day
        day_result = await manager.execute_trading_day(symbols, market_data)
        results.append(day_result)

        logger.info(f"Day {day + 1} portfolio value: ${day_result['portfolio_value']:,.2f}")

    return results
