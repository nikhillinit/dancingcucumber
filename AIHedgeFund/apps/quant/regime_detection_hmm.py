"""
Market Regime Detection with Hidden Markov Models
=================================================
Multi-agent system for detecting and adapting to market regimes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import ray
from joblib import Parallel, delayed
import asyncio
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime definition"""
    regime_id: int
    name: str
    volatility: float
    return_mean: float
    characteristics: Dict[str, float]
    probability: float
    timestamp: datetime


@dataclass
class RegimeTransition:
    """Regime transition probabilities"""
    from_regime: int
    to_regime: int
    probability: float
    expected_duration: int
    historical_frequency: float


class RegimeDetectionAgent(ray.remote):
    """Agent for regime detection using HMM"""

    def __init__(self, agent_id: str, n_regimes: int = 4):
        self.agent_id = agent_id
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "Bull Market",
            1: "Bear Market",
            2: "High Volatility",
            3: "Low Volatility"
        }

    def train_hmm(self, data: pd.DataFrame) -> hmm.GaussianHMM:
        """Train Hidden Markov Model on market data"""
        # Prepare features
        features = self._prepare_features(data)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

        self.model.fit(features_scaled)

        return self.model

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM training"""
        features = []

        # Returns
        returns = data['close'].pct_change().fillna(0)
        features.append(returns.values.reshape(-1, 1))

        # Volatility (rolling std)
        volatility = returns.rolling(window=20).std().fillna(0)
        features.append(volatility.values.reshape(-1, 1))

        # Volume changes
        if 'volume' in data.columns:
            volume_change = data['volume'].pct_change().fillna(0)
            features.append(volume_change.values.reshape(-1, 1))

        # Price momentum
        momentum = data['close'].pct_change(20).fillna(0)
        features.append(momentum.values.reshape(-1, 1))

        # Combine features
        return np.hstack(features)

    def detect_current_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if self.model is None:
            self.train_hmm(data)

        # Prepare recent data
        features = self._prepare_features(data.iloc[-100:])
        features_scaled = self.scaler.transform(features)

        # Predict regime
        _, posterior = self.model.decode(features_scaled, algorithm="viterbi")
        current_regime = posterior[-1]

        # Get regime probabilities
        regime_probs = self.model.predict_proba(features_scaled[-1:].reshape(1, -1))[0]

        # Calculate regime characteristics
        characteristics = self._calculate_regime_characteristics(data, current_regime)

        return MarketRegime(
            regime_id=current_regime,
            name=self.regime_names.get(current_regime, f"Regime {current_regime}"),
            volatility=characteristics['volatility'],
            return_mean=characteristics['return_mean'],
            characteristics=characteristics,
            probability=regime_probs[current_regime],
            timestamp=datetime.now()
        )

    def _calculate_regime_characteristics(
        self,
        data: pd.DataFrame,
        regime: int
    ) -> Dict[str, float]:
        """Calculate characteristics of a specific regime"""
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)

        # Get all regime predictions
        regimes = self.model.predict(features_scaled)

        # Filter data for specific regime
        regime_mask = regimes == regime
        regime_data = data[regime_mask]

        if len(regime_data) > 0:
            returns = regime_data['close'].pct_change()

            characteristics = {
                'volatility': returns.std() * np.sqrt(252),
                'return_mean': returns.mean() * 252,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(regime_data['close']),
                'avg_volume': regime_data['volume'].mean() if 'volume' in regime_data else 0,
                'duration_days': len(regime_data)
            }
        else:
            characteristics = {
                'volatility': 0,
                'return_mean': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_volume': 0,
                'duration_days': 0
            }

        return characteristics

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def predict_regime_transitions(self) -> List[RegimeTransition]:
        """Predict regime transition probabilities"""
        if self.model is None:
            return []

        transitions = []
        transition_matrix = self.model.transmat_

        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                prob = transition_matrix[i, j]

                # Calculate expected duration in regime
                if i == j:
                    expected_duration = int(1 / (1 - prob)) if prob < 1 else 100
                else:
                    expected_duration = 0

                transitions.append(RegimeTransition(
                    from_regime=i,
                    to_regime=j,
                    probability=prob,
                    expected_duration=expected_duration,
                    historical_frequency=prob  # Simplified
                ))

        return transitions


class AdaptiveStrategyAgent(ray.remote):
    """Agent that adapts strategy based on regime"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.regime_strategies = {
            0: self._bull_market_strategy,
            1: self._bear_market_strategy,
            2: self._high_volatility_strategy,
            3: self._low_volatility_strategy
        }

    async def adapt_strategy(
        self,
        regime: MarketRegime,
        portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Adapt trading strategy based on detected regime"""
        strategy_func = self.regime_strategies.get(
            regime.regime_id,
            self._default_strategy
        )

        return await strategy_func(regime, portfolio)

    async def _bull_market_strategy(
        self,
        regime: MarketRegime,
        portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Strategy for bull market regime"""
        return {
            'allocation': {
                'stocks': 0.8,
                'bonds': 0.1,
                'cash': 0.1
            },
            'leverage': 1.2,
            'risk_limit': 0.02,
            'strategy_type': 'momentum',
            'position_sizing': 'kelly',
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'rebalance_frequency': 'weekly'
        }

    async def _bear_market_strategy(
        self,
        regime: MarketRegime,
        portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Strategy for bear market regime"""
        return {
            'allocation': {
                'stocks': 0.3,
                'bonds': 0.5,
                'cash': 0.2
            },
            'leverage': 0.5,
            'risk_limit': 0.01,
            'strategy_type': 'defensive',
            'position_sizing': 'equal_weight',
            'stop_loss': 0.03,
            'take_profit': 0.08,
            'rebalance_frequency': 'daily',
            'hedging': True
        }

    async def _high_volatility_strategy(
        self,
        regime: MarketRegime,
        portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Strategy for high volatility regime"""
        return {
            'allocation': {
                'stocks': 0.4,
                'bonds': 0.3,
                'cash': 0.2,
                'volatility_etf': 0.1
            },
            'leverage': 0.7,
            'risk_limit': 0.015,
            'strategy_type': 'mean_reversion',
            'position_sizing': 'volatility_adjusted',
            'stop_loss': 0.04,
            'take_profit': 0.10,
            'rebalance_frequency': 'daily',
            'options_hedging': True
        }

    async def _low_volatility_strategy(
        self,
        regime: MarketRegime,
        portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Strategy for low volatility regime"""
        return {
            'allocation': {
                'stocks': 0.7,
                'bonds': 0.2,
                'cash': 0.1
            },
            'leverage': 1.0,
            'risk_limit': 0.025,
            'strategy_type': 'carry_trade',
            'position_sizing': 'risk_parity',
            'stop_loss': 0.06,
            'take_profit': 0.12,
            'rebalance_frequency': 'monthly'
        }

    async def _default_strategy(
        self,
        regime: MarketRegime,
        portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Default balanced strategy"""
        return {
            'allocation': {
                'stocks': 0.5,
                'bonds': 0.3,
                'cash': 0.2
            },
            'leverage': 1.0,
            'risk_limit': 0.02,
            'strategy_type': 'balanced',
            'position_sizing': 'equal_weight',
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'rebalance_frequency': 'weekly'
        }


class RegimeMonitoringAgent(ray.remote):
    """Agent for continuous regime monitoring"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.regime_history = []
        self.transition_alerts = []

    async def monitor_regime_changes(
        self,
        current_regime: MarketRegime,
        previous_regime: Optional[MarketRegime]
    ) -> Dict[str, Any]:
        """Monitor for regime changes and generate alerts"""
        alerts = []

        if previous_regime and current_regime.regime_id != previous_regime.regime_id:
            # Regime change detected
            alerts.append({
                'type': 'regime_change',
                'severity': 'high',
                'from_regime': previous_regime.name,
                'to_regime': current_regime.name,
                'probability': current_regime.probability,
                'timestamp': datetime.now()
            })

            # Check for dangerous transitions
            if self._is_dangerous_transition(previous_regime.regime_id, current_regime.regime_id):
                alerts.append({
                    'type': 'dangerous_transition',
                    'severity': 'critical',
                    'message': f"High-risk transition from {previous_regime.name} to {current_regime.name}",
                    'recommended_action': 'reduce_exposure',
                    'timestamp': datetime.now()
                })

        # Check regime stability
        if current_regime.probability < 0.6:
            alerts.append({
                'type': 'regime_uncertainty',
                'severity': 'medium',
                'message': f"Low confidence in {current_regime.name} regime ({current_regime.probability:.1%})",
                'timestamp': datetime.now()
            })

        self.regime_history.append(current_regime)
        self.transition_alerts.extend(alerts)

        return {
            'current_regime': current_regime,
            'alerts': alerts,
            'regime_stability': self._calculate_regime_stability(),
            'transition_probability': self._estimate_transition_probability(current_regime)
        }

    def _is_dangerous_transition(self, from_regime: int, to_regime: int) -> bool:
        """Check if transition is dangerous"""
        dangerous_transitions = [
            (0, 1),  # Bull to Bear
            (3, 2),  # Low Vol to High Vol
        ]
        return (from_regime, to_regime) in dangerous_transitions

    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability score"""
        if len(self.regime_history) < 10:
            return 0.5

        # Check how often regime changed in recent history
        recent_regimes = [r.regime_id for r in self.regime_history[-10:]]
        changes = sum(1 for i in range(1, len(recent_regimes))
                     if recent_regimes[i] != recent_regimes[i-1])

        stability = 1 - (changes / len(recent_regimes))
        return stability

    def _estimate_transition_probability(self, current_regime: MarketRegime) -> Dict[str, float]:
        """Estimate probability of transitioning to other regimes"""
        # Simplified estimation
        transition_probs = {}
        remaining_prob = 1 - current_regime.probability

        for regime_id in range(4):
            if regime_id != current_regime.regime_id:
                transition_probs[f"to_regime_{regime_id}"] = remaining_prob / 3

        return transition_probs


class RegimeDetectionOrchestrator:
    """Orchestrate regime detection and adaptation"""

    def __init__(self):
        ray.init(ignore_reinit_error=True)

        self.detection_agent = RegimeDetectionAgent.remote("detection", n_regimes=4)
        self.strategy_agent = AdaptiveStrategyAgent.remote("strategy")
        self.monitoring_agent = RegimeMonitoringAgent.remote("monitoring")

        self.current_regime = None
        self.previous_regime = None

    async def analyze_regime(
        self,
        market_data: pd.DataFrame,
        portfolio: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Complete regime analysis pipeline"""
        # Train HMM if needed
        await self._train_if_needed(market_data)

        # Detect current regime
        regime_task = self.detection_agent.detect_current_regime.remote(market_data)
        current_regime = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, regime_task)
        )

        # Get transition probabilities
        transitions_task = self.detection_agent.predict_regime_transitions.remote()
        transitions = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, transitions_task)
        )

        # Adapt strategy
        if portfolio:
            strategy_task = self.strategy_agent.adapt_strategy.remote(current_regime, portfolio)
            adapted_strategy = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, strategy_task)
            )
        else:
            adapted_strategy = {}

        # Monitor for changes
        monitoring_task = self.monitoring_agent.monitor_regime_changes.remote(
            current_regime, self.previous_regime
        )
        monitoring_result = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, monitoring_task)
        )

        # Update regime history
        self.previous_regime = self.current_regime
        self.current_regime = current_regime

        return {
            'current_regime': current_regime,
            'transitions': transitions,
            'adapted_strategy': adapted_strategy,
            'monitoring': monitoring_result,
            'timestamp': datetime.now()
        }

    async def _train_if_needed(self, data: pd.DataFrame):
        """Train HMM model if not already trained"""
        # Check if model exists (simplified)
        train_task = self.detection_agent.train_hmm.remote(data)
        await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, train_task)
        )

    def get_regime_recommendations(self, regime: MarketRegime) -> List[str]:
        """Get trading recommendations based on regime"""
        recommendations = []

        if regime.regime_id == 0:  # Bull
            recommendations.extend([
                "Increase equity exposure",
                "Focus on growth stocks",
                "Consider leveraged positions",
                "Reduce hedging"
            ])
        elif regime.regime_id == 1:  # Bear
            recommendations.extend([
                "Reduce equity exposure",
                "Increase cash position",
                "Consider defensive sectors",
                "Implement hedging strategies"
            ])
        elif regime.regime_id == 2:  # High Vol
            recommendations.extend([
                "Reduce position sizes",
                "Increase stop-loss levels",
                "Consider volatility trading",
                "Diversify across uncorrelated assets"
            ])
        elif regime.regime_id == 3:  # Low Vol
            recommendations.extend([
                "Consider carry trades",
                "Increase position sizes cautiously",
                "Look for mean reversion opportunities",
                "Reduce cash holdings"
            ])

        return recommendations

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of regime detection"""
    orchestrator = RegimeDetectionOrchestrator()

    # Generate sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(len(dates)) * 2) + 100,
        'volume': np.random.random(len(dates)) * 1000000
    }, index=dates)

    # Sample portfolio
    portfolio = {'AAPL': 10000, 'GOOGL': 8000, 'MSFT': 7000}

    # Analyze regime
    result = await orchestrator.analyze_regime(data, portfolio)

    print(f"\nCurrent Regime: {result['current_regime'].name}")
    print(f"Confidence: {result['current_regime'].probability:.1%}")
    print(f"Volatility: {result['current_regime'].volatility:.1%}")
    print(f"Expected Return: {result['current_regime'].return_mean:.1%}")

    print("\nAdapted Strategy:")
    for key, value in result['adapted_strategy'].items():
        print(f"  {key}: {value}")

    print("\nRecommendations:")
    recommendations = orchestrator.get_regime_recommendations(result['current_regime'])
    for rec in recommendations:
        print(f"  - {rec}")

    if result['monitoring']['alerts']:
        print("\nAlerts:")
        for alert in result['monitoring']['alerts']:
            print(f"  {alert['type']}: {alert.get('message', '')}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())