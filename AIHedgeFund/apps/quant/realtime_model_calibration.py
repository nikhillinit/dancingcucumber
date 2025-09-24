"""
Real-Time Model Calibration Framework with Multi-Agent Processing
=================================================================
Online learning and adaptive model calibration for daily portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import ray
import asyncio
from collections import deque
from scipy import stats
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    accuracy: float
    mse: float
    sharpe_ratio: float
    hit_rate: float
    profit_factor: float
    max_drawdown: float
    decay_rate: float
    last_update: datetime
    n_predictions: int


@dataclass
class DriftDetection:
    """Concept drift detection result"""
    drift_detected: bool
    drift_type: str  # sudden, gradual, incremental, recurring
    drift_magnitude: float
    confidence: float
    affected_features: List[str]
    recommended_action: str
    timestamp: datetime


@dataclass
class ModelUpdate:
    """Model update information"""
    model_id: str
    update_type: str  # weights, architecture, hyperparameters
    performance_before: float
    performance_after: float
    update_size: int  # Number of samples used
    learning_rate: float
    timestamp: datetime


@dataclass
class RegimeState:
    """Market regime state"""
    regime_id: int
    regime_name: str
    volatility_level: str  # low, medium, high
    trend_direction: str  # bullish, bearish, sideways
    correlation_state: str  # normal, breakdown, crisis
    confidence: float
    timestamp: datetime


@ray.remote
class OnlineLearningAgent:
    """Agent for online model learning and updates"""

    def __init__(self, agent_id: str, model_type: str = 'neural'):
        self.agent_id = agent_id
        self.model_type = model_type
        self.model = self._initialize_model()
        self.optimizer = None
        self.buffer_size = 1000
        self.experience_buffer = deque(maxlen=self.buffer_size)
        self.update_frequency = 100  # Update every N samples
        self.learning_rate = 0.001

    def _initialize_model(self):
        """Initialize the online learning model"""
        if self.model_type == 'neural':
            model = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            return model
        else:
            # Could add other model types
            return None

    async def update_model(
        self,
        new_data: pd.DataFrame,
        targets: np.ndarray,
        update_type: str = 'incremental'
    ) -> ModelUpdate:
        """Update model with new data"""

        # Store in buffer
        for i in range(len(new_data)):
            self.experience_buffer.append((new_data.iloc[i].values, targets[i]))

        # Check if update needed
        if len(self.experience_buffer) % self.update_frequency == 0:
            return await self._perform_update(update_type)

        return ModelUpdate(
            model_id=self.agent_id,
            update_type='none',
            performance_before=0,
            performance_after=0,
            update_size=0,
            learning_rate=self.learning_rate,
            timestamp=datetime.now()
        )

    async def _perform_update(self, update_type: str) -> ModelUpdate:
        """Perform model update"""
        if self.model_type == 'neural' and self.model is not None:
            # Sample from buffer
            batch_size = min(32, len(self.experience_buffer))
            batch = list(self.experience_buffer)[-batch_size:]

            X = torch.FloatTensor([x for x, _ in batch])
            y = torch.FloatTensor([y for _, y in batch]).unsqueeze(1)

            # Calculate performance before
            self.model.eval()
            with torch.no_grad():
                pred_before = self.model(X)
                loss_before = nn.MSELoss()(pred_before, y).item()

            # Update model
            self.model.train()
            epochs = 5 if update_type == 'incremental' else 20

            for _ in range(epochs):
                self.optimizer.zero_grad()
                predictions = self.model(X)
                loss = nn.MSELoss()(predictions, y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

            # Calculate performance after
            self.model.eval()
            with torch.no_grad():
                pred_after = self.model(X)
                loss_after = nn.MSELoss()(pred_after, y).item()

            return ModelUpdate(
                model_id=self.agent_id,
                update_type=update_type,
                performance_before=loss_before,
                performance_after=loss_after,
                update_size=batch_size,
                learning_rate=self.learning_rate,
                timestamp=datetime.now()
            )

        return ModelUpdate(
            model_id=self.agent_id,
            update_type='failed',
            performance_before=0,
            performance_after=0,
            update_size=0,
            learning_rate=0,
            timestamp=datetime.now()
        )

    async def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with current model"""
        if self.model_type == 'neural' and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(features)
                predictions = self.model(X).numpy()
            return predictions
        return np.zeros(len(features))

    def adjust_learning_rate(self, performance_trend: float):
        """Dynamically adjust learning rate based on performance"""
        if performance_trend < 0:  # Performance degrading
            self.learning_rate *= 0.9
        elif performance_trend > 0.1:  # Significant improvement
            self.learning_rate *= 1.1

        # Keep in reasonable range
        self.learning_rate = max(0.0001, min(0.01, self.learning_rate))

        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate


@ray.remote
class DriftDetectionAgent:
    """Agent for detecting concept drift in data streams"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.window_size = 100
        self.reference_window = deque(maxlen=self.window_size)
        self.detection_window = deque(maxlen=self.window_size)
        self.drift_threshold = 0.05
        self.warning_threshold = 0.01

    async def detect_drift(
        self,
        new_data: pd.DataFrame,
        feature_names: List[str]
    ) -> DriftDetection:
        """Detect concept drift in data stream"""

        # Update windows
        for _, row in new_data.iterrows():
            self.detection_window.append(row.values)

            # Move old detection to reference
            if len(self.detection_window) == self.window_size:
                self.reference_window.append(self.detection_window[0])

        # Need full windows for detection
        if len(self.reference_window) < self.window_size // 2:
            return DriftDetection(
                drift_detected=False,
                drift_type='none',
                drift_magnitude=0,
                confidence=0,
                affected_features=[],
                recommended_action='continue',
                timestamp=datetime.now()
            )

        # Convert to arrays
        ref_data = np.array(list(self.reference_window))
        det_data = np.array(list(self.detection_window))

        # Multiple drift detection methods
        drift_scores = {}

        # 1. Kolmogorov-Smirnov test for each feature
        ks_drifts = self._ks_test_drift(ref_data, det_data, feature_names)
        drift_scores['ks'] = ks_drifts

        # 2. Page-Hinkley test
        ph_drift = self._page_hinkley_test(ref_data, det_data)
        drift_scores['page_hinkley'] = ph_drift

        # 3. ADWIN (Adaptive Windowing)
        adwin_drift = self._adwin_test(ref_data, det_data)
        drift_scores['adwin'] = adwin_drift

        # 4. Statistical moments comparison
        moment_drift = self._moment_drift(ref_data, det_data)
        drift_scores['moments'] = moment_drift

        # Aggregate drift detection
        drift_detected, drift_type, magnitude, affected = self._aggregate_drift_detection(
            drift_scores, feature_names
        )

        # Determine action
        if drift_detected:
            if magnitude > 0.3:
                action = 'full_retrain'
            elif magnitude > 0.15:
                action = 'partial_retrain'
            else:
                action = 'adapt_weights'
        else:
            action = 'continue'

        return DriftDetection(
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_magnitude=magnitude,
            confidence=min(0.95, magnitude * 2),
            affected_features=affected,
            recommended_action=action,
            timestamp=datetime.now()
        )

    def _ks_test_drift(
        self,
        ref_data: np.ndarray,
        det_data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Kolmogorov-Smirnov test for distribution drift"""
        drifts = {}

        for i, feature in enumerate(feature_names[:min(len(feature_names), ref_data.shape[1])]):
            _, p_value = stats.ks_2samp(ref_data[:, i], det_data[:, i])
            drifts[feature] = 1 - p_value  # Convert to drift score

        return drifts

    def _page_hinkley_test(
        self,
        ref_data: np.ndarray,
        det_data: np.ndarray,
        delta: float = 0.005
    ) -> float:
        """Page-Hinkley test for mean shift"""
        # Simplified implementation
        ref_mean = np.mean(ref_data, axis=0)
        cumsum = 0
        min_val = 0
        drift_score = 0

        for sample in det_data:
            cumsum += np.mean(sample - ref_mean - delta)
            min_val = min(min_val, cumsum)
            ph_stat = cumsum - min_val

            if ph_stat > self.drift_threshold * len(det_data):
                drift_score = ph_stat / (self.drift_threshold * len(det_data))
                break

        return min(1.0, drift_score)

    def _adwin_test(
        self,
        ref_data: np.ndarray,
        det_data: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """ADWIN adaptive windowing test"""
        # Simplified ADWIN implementation
        all_data = np.vstack([ref_data, det_data])
        n = len(all_data)

        max_drift = 0
        for split_point in range(len(ref_data), len(ref_data) + len(det_data) // 2):
            w1 = all_data[:split_point]
            w2 = all_data[split_point:]

            mean1 = np.mean(w1)
            mean2 = np.mean(w2)

            # Calculate drift magnitude
            drift = abs(mean1 - mean2) / (np.std(all_data) + 1e-8)
            max_drift = max(max_drift, drift)

        return min(1.0, max_drift / 2)

    def _moment_drift(
        self,
        ref_data: np.ndarray,
        det_data: np.ndarray
    ) -> float:
        """Detect drift in statistical moments"""
        # Compare mean, variance, skewness, kurtosis
        ref_moments = [
            np.mean(ref_data),
            np.var(ref_data),
            stats.skew(ref_data.flatten()),
            stats.kurtosis(ref_data.flatten())
        ]

        det_moments = [
            np.mean(det_data),
            np.var(det_data),
            stats.skew(det_data.flatten()),
            stats.kurtosis(det_data.flatten())
        ]

        # Weighted difference
        weights = [0.3, 0.3, 0.2, 0.2]
        drift_score = 0

        for i, (ref_m, det_m, w) in enumerate(zip(ref_moments, det_moments, weights)):
            if i == 0:  # Mean
                diff = abs(ref_m - det_m) / (abs(ref_m) + 1e-8)
            elif i == 1:  # Variance
                diff = abs(np.log(det_m + 1e-8) - np.log(ref_m + 1e-8))
            else:  # Skewness, Kurtosis
                diff = abs(det_m - ref_m) / 10

            drift_score += w * min(1.0, diff)

        return drift_score

    def _aggregate_drift_detection(
        self,
        drift_scores: Dict[str, Any],
        feature_names: List[str]
    ) -> Tuple[bool, str, float, List[str]]:
        """Aggregate multiple drift detection methods"""
        # KS test results
        ks_drifts = drift_scores.get('ks', {})
        max_ks_drift = max(ks_drifts.values()) if ks_drifts else 0
        affected_features = [f for f, v in ks_drifts.items() if v > self.drift_threshold]

        # Other drift scores
        ph_drift = drift_scores.get('page_hinkley', 0)
        adwin_drift = drift_scores.get('adwin', 0)
        moment_drift = drift_scores.get('moments', 0)

        # Weighted average
        magnitude = (max_ks_drift * 0.3 + ph_drift * 0.3 + adwin_drift * 0.2 + moment_drift * 0.2)

        # Determine drift type
        if magnitude > self.drift_threshold:
            drift_detected = True
            if ph_drift > 0.5:
                drift_type = 'sudden'
            elif moment_drift > 0.3:
                drift_type = 'gradual'
            else:
                drift_type = 'incremental'
        else:
            drift_detected = False
            drift_type = 'none'

        return drift_detected, drift_type, magnitude, affected_features


@ray.remote
class RegimeDetectionAgent:
    """Agent for real-time regime detection"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.regime_history = deque(maxlen=100)
        self.current_regime = None
        self.regime_models = {}

    async def detect_regime(
        self,
        market_data: pd.DataFrame,
        volatility_window: int = 20
    ) -> RegimeState:
        """Detect current market regime"""

        if len(market_data) < volatility_window:
            return self._default_regime()

        # Calculate regime indicators
        returns = market_data['close'].pct_change().dropna()

        # 1. Volatility regime
        volatility = returns.rolling(volatility_window).std().iloc[-1]
        volatility_percentile = stats.percentileofscore(
            returns.rolling(volatility_window).std().dropna(), volatility
        )

        if volatility_percentile < 30:
            volatility_level = 'low'
        elif volatility_percentile < 70:
            volatility_level = 'medium'
        else:
            volatility_level = 'high'

        # 2. Trend regime
        sma_short = market_data['close'].rolling(20).mean().iloc[-1]
        sma_long = market_data['close'].rolling(50).mean().iloc[-1] if len(market_data) >= 50 else sma_short
        current_price = market_data['close'].iloc[-1]

        if current_price > sma_short > sma_long:
            trend_direction = 'bullish'
        elif current_price < sma_short < sma_long:
            trend_direction = 'bearish'
        else:
            trend_direction = 'sideways'

        # 3. Correlation regime (simplified - would use multiple assets in production)
        recent_returns = returns.iloc[-volatility_window:]
        autocorrelation = recent_returns.autocorr(lag=1) if len(recent_returns) > 1 else 0

        if abs(autocorrelation) < 0.2:
            correlation_state = 'normal'
        elif abs(autocorrelation) < 0.5:
            correlation_state = 'breakdown'
        else:
            correlation_state = 'crisis'

        # 4. Determine regime ID and name
        regime_id, regime_name = self._classify_regime(
            volatility_level, trend_direction, correlation_state
        )

        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            volatility_percentile, abs(autocorrelation)
        )

        regime = RegimeState(
            regime_id=regime_id,
            regime_name=regime_name,
            volatility_level=volatility_level,
            trend_direction=trend_direction,
            correlation_state=correlation_state,
            confidence=confidence,
            timestamp=datetime.now()
        )

        # Update history
        self.regime_history.append(regime)
        self.current_regime = regime

        return regime

    def _default_regime(self) -> RegimeState:
        """Return default regime when insufficient data"""
        return RegimeState(
            regime_id=0,
            regime_name='unknown',
            volatility_level='medium',
            trend_direction='sideways',
            correlation_state='normal',
            confidence=0.5,
            timestamp=datetime.now()
        )

    def _classify_regime(
        self,
        volatility: str,
        trend: str,
        correlation: str
    ) -> Tuple[int, str]:
        """Classify regime based on characteristics"""
        # Simplified regime classification
        regime_map = {
            ('low', 'bullish', 'normal'): (1, 'steady_growth'),
            ('low', 'bearish', 'normal'): (2, 'orderly_decline'),
            ('low', 'sideways', 'normal'): (3, 'range_bound'),
            ('medium', 'bullish', 'normal'): (4, 'normal_bull'),
            ('medium', 'bearish', 'normal'): (5, 'normal_bear'),
            ('medium', 'sideways', 'normal'): (6, 'choppy'),
            ('high', 'bullish', 'normal'): (7, 'volatile_bull'),
            ('high', 'bearish', 'normal'): (8, 'volatile_bear'),
            ('high', 'sideways', 'breakdown'): (9, 'uncertainty'),
            ('high', 'bearish', 'crisis'): (10, 'crisis')
        }

        key = (volatility, trend, correlation)
        return regime_map.get(key, (0, 'unclassified'))

    def _calculate_regime_confidence(
        self,
        volatility_percentile: float,
        correlation: float
    ) -> float:
        """Calculate confidence in regime detection"""
        # Higher confidence for extreme values
        vol_confidence = abs(volatility_percentile - 50) / 50
        corr_confidence = abs(correlation)

        confidence = (vol_confidence + corr_confidence) / 2
        return min(0.95, max(0.5, confidence))

    async def get_regime_specific_params(
        self,
        regime: RegimeState
    ) -> Dict[str, Any]:
        """Get regime-specific model parameters"""
        params = {
            'steady_growth': {
                'risk_level': 0.8,
                'position_size': 1.2,
                'stop_loss': 0.03,
                'take_profit': 0.05
            },
            'crisis': {
                'risk_level': 0.3,
                'position_size': 0.5,
                'stop_loss': 0.01,
                'take_profit': 0.02
            },
            'volatile_bull': {
                'risk_level': 0.6,
                'position_size': 0.8,
                'stop_loss': 0.02,
                'take_profit': 0.04
            }
        }

        return params.get(regime.regime_name, {
            'risk_level': 0.5,
            'position_size': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.03
        })


@ray.remote
class ModelPerformanceMonitor:
    """Agent for monitoring model performance"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.performance_history = deque(maxlen=1000)
        self.decay_window = 20

    async def evaluate_model_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_id: str
    ) -> ModelPerformance:
        """Evaluate model performance metrics"""

        if len(predictions) == 0 or len(actuals) == 0:
            return self._empty_performance(model_id)

        # Basic metrics
        mse = mean_squared_error(actuals, predictions)

        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions))
            actual_direction = np.sign(np.diff(actuals))
            hit_rate = np.mean(pred_direction == actual_direction)
        else:
            hit_rate = 0.5

        # Trading metrics (simplified)
        returns = (predictions - actuals) / (actuals + 1e-8)
        sharpe_ratio = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / (losses + 1e-8)

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = abs(drawdown.min())

        # Calculate performance decay rate
        decay_rate = self._calculate_decay_rate(model_id)

        performance = ModelPerformance(
            model_id=model_id,
            accuracy=hit_rate,
            mse=mse,
            sharpe_ratio=sharpe_ratio,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            decay_rate=decay_rate,
            last_update=datetime.now(),
            n_predictions=len(predictions)
        )

        self.performance_history.append(performance)

        return performance

    def _empty_performance(self, model_id: str) -> ModelPerformance:
        """Return empty performance metrics"""
        return ModelPerformance(
            model_id=model_id,
            accuracy=0,
            mse=float('inf'),
            sharpe_ratio=0,
            hit_rate=0.5,
            profit_factor=1,
            max_drawdown=0,
            decay_rate=0,
            last_update=datetime.now(),
            n_predictions=0
        )

    def _calculate_decay_rate(self, model_id: str) -> float:
        """Calculate model performance decay rate"""
        # Get recent performance for this model
        recent_performance = [
            p for p in list(self.performance_history)[-self.decay_window:]
            if p.model_id == model_id
        ]

        if len(recent_performance) < 2:
            return 0

        # Calculate trend in accuracy
        accuracies = [p.accuracy for p in recent_performance]
        x = np.arange(len(accuracies))

        if len(x) > 1:
            slope = np.polyfit(x, accuracies, 1)[0]
            decay_rate = -slope if slope < 0 else 0
        else:
            decay_rate = 0

        return decay_rate


class RealtimeCalibrationOrchestrator:
    """Orchestrate real-time model calibration with multi-agent processing"""

    def __init__(self, n_models: int = 3):
        ray.init(ignore_reinit_error=True)

        # Initialize agents
        self.online_learning_agents = [
            OnlineLearningAgent.remote(f"model_{i}", "neural")
            for i in range(n_models)
        ]
        self.drift_detector = DriftDetectionAgent.remote("drift_detector")
        self.regime_detector = RegimeDetectionAgent.remote("regime_detector")
        self.performance_monitor = ModelPerformanceMonitor.remote("monitor")

        # Calibration state
        self.current_regime = None
        self.last_drift_detection = None
        self.model_performances = {}

    async def calibrate_models(
        self,
        market_data: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        actuals: np.ndarray
    ) -> Dict[str, Any]:
        """Run complete model calibration pipeline"""

        calibration_tasks = []

        # 1. Detect regime
        regime_task = self.regime_detector.detect_regime.remote(market_data)
        calibration_tasks.append(('regime', regime_task))

        # 2. Detect drift
        feature_names = market_data.columns.tolist()
        drift_task = self.drift_detector.detect_drift.remote(market_data, feature_names)
        calibration_tasks.append(('drift', drift_task))

        # 3. Evaluate model performances
        for model_id, preds in predictions.items():
            perf_task = self.performance_monitor.evaluate_model_performance.remote(
                preds, actuals, model_id
            )
            calibration_tasks.append((f'performance_{model_id}', perf_task))

        # Gather results
        results = {}
        for task_type, task in calibration_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )
            results[task_type] = result

        # Extract key results
        regime = results.get('regime')
        drift = results.get('drift')
        performances = {
            k.replace('performance_', ''): v
            for k, v in results.items()
            if k.startswith('performance_')
        }

        # Update models based on drift and performance
        update_tasks = []

        if drift and drift.drift_detected:
            # Drift detected - update all models
            update_type = 'full' if drift.drift_magnitude > 0.3 else 'incremental'

            for i, agent in enumerate(self.online_learning_agents):
                update_task = agent.update_model.remote(
                    market_data, actuals, update_type
                )
                update_tasks.append((f"model_{i}", update_task))
        else:
            # Regular incremental updates for poor performers
            for model_id, performance in performances.items():
                if performance.accuracy < 0.55 or performance.decay_rate > 0.01:
                    # Find corresponding agent
                    agent_idx = int(model_id.split('_')[-1]) if '_' in model_id else 0
                    if agent_idx < len(self.online_learning_agents):
                        agent = self.online_learning_agents[agent_idx]
                        update_task = agent.update_model.remote(
                            market_data, actuals, 'incremental'
                        )
                        update_tasks.append((model_id, update_task))

        # Gather update results
        updates = {}
        for model_id, task in update_tasks:
            update = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )
            updates[model_id] = update

        # Get regime-specific parameters
        if regime:
            regime_params_task = self.regime_detector.get_regime_specific_params.remote(regime)
            regime_params = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, regime_params_task)
            )
        else:
            regime_params = {}

        # Store state
        self.current_regime = regime
        self.last_drift_detection = drift
        self.model_performances = performances

        return {
            'regime': regime,
            'drift': drift,
            'performances': performances,
            'updates': updates,
            'regime_params': regime_params,
            'calibration_summary': self._generate_calibration_summary(regime, drift, performances, updates),
            'timestamp': datetime.now()
        }

    def _generate_calibration_summary(
        self,
        regime: Optional[RegimeState],
        drift: Optional[DriftDetection],
        performances: Dict[str, ModelPerformance],
        updates: Dict[str, ModelUpdate]
    ) -> Dict[str, Any]:
        """Generate calibration summary"""
        summary = {
            'current_regime': regime.regime_name if regime else 'unknown',
            'regime_confidence': regime.confidence if regime else 0,
            'drift_detected': drift.drift_detected if drift else False,
            'drift_magnitude': drift.drift_magnitude if drift else 0,
            'avg_model_accuracy': np.mean([p.accuracy for p in performances.values()]) if performances else 0,
            'best_model': max(performances.keys(), key=lambda k: performances[k].sharpe_ratio) if performances else None,
            'models_updated': len(updates),
            'recommended_actions': []
        }

        # Generate recommendations
        if drift and drift.drift_detected:
            summary['recommended_actions'].append(f"Retrain models due to {drift.drift_type} drift")

        if regime and regime.volatility_level == 'high':
            summary['recommended_actions'].append("Reduce position sizes in high volatility")

        if performances:
            poor_models = [m for m, p in performances.items() if p.accuracy < 0.52]
            if poor_models:
                summary['recommended_actions'].append(f"Replace models: {', '.join(poor_models)}")

        return summary

    async def predict_with_calibration(
        self,
        features: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Make predictions with calibrated models"""
        predictions = {}

        for i, agent in enumerate(self.online_learning_agents):
            pred_task = agent.predict.remote(features.values)
            pred = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, pred_task)
            )
            predictions[f"model_{i}"] = pred

        return predictions

    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get current calibration metrics"""
        return {
            'current_regime': self.current_regime.regime_name if self.current_regime else 'unknown',
            'drift_status': 'detected' if self.last_drift_detection and self.last_drift_detection.drift_detected else 'none',
            'model_performances': {
                model_id: {
                    'accuracy': perf.accuracy,
                    'sharpe': perf.sharpe_ratio,
                    'decay_rate': perf.decay_rate
                }
                for model_id, perf in self.model_performances.items()
            },
            'timestamp': datetime.now()
        }

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of real-time model calibration"""
    orchestrator = RealtimeCalibrationOrchestrator(n_models=3)

    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    market_data = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.2,
        'high': prices + np.abs(np.random.randn(100) * 0.3),
        'low': prices - np.abs(np.random.randn(100) * 0.3),
        'close': prices,
        'volume': np.random.gamma(2, 100000, 100)
    }, index=dates)

    # Simulate predictions from different models
    predictions = {
        'model_0': prices[-50:] + np.random.randn(50) * 0.5,
        'model_1': prices[-50:] + np.random.randn(50) * 0.7,
        'model_2': prices[-50:] + np.random.randn(50) * 0.3
    }

    actuals = prices[-50:] + np.random.randn(50) * 0.2

    # Run calibration
    print("Running real-time model calibration...")
    results = await orchestrator.calibrate_models(
        market_data,
        predictions,
        actuals
    )

    # Display results
    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)

    # Regime
    if results['regime']:
        regime = results['regime']
        print(f"\nCurrent Regime: {regime.regime_name}")
        print(f"  Volatility: {regime.volatility_level}")
        print(f"  Trend: {regime.trend_direction}")
        print(f"  Confidence: {regime.confidence:.1%}")

    # Drift
    if results['drift']:
        drift = results['drift']
        print(f"\nDrift Detection:")
        print(f"  Drift Detected: {drift.drift_detected}")
        if drift.drift_detected:
            print(f"  Type: {drift.drift_type}")
            print(f"  Magnitude: {drift.drift_magnitude:.3f}")
            print(f"  Action: {drift.recommended_action}")

    # Model performances
    print(f"\nModel Performances:")
    for model_id, perf in results['performances'].items():
        print(f"  {model_id}:")
        print(f"    Accuracy: {perf.accuracy:.1%}")
        print(f"    Sharpe: {perf.sharpe_ratio:.2f}")
        print(f"    Decay Rate: {perf.decay_rate:.3f}")

    # Updates
    if results['updates']:
        print(f"\nModel Updates:")
        for model_id, update in results['updates'].items():
            if update.update_type != 'none':
                print(f"  {model_id}: {update.update_type}")
                print(f"    Performance: {update.performance_before:.3f} -> {update.performance_after:.3f}")

    # Summary
    summary = results['calibration_summary']
    print(f"\nCalibration Summary:")
    print(f"  Average Accuracy: {summary['avg_model_accuracy']:.1%}")
    print(f"  Best Model: {summary['best_model']}")

    if summary['recommended_actions']:
        print(f"  Recommended Actions:")
        for action in summary['recommended_actions']:
            print(f"    - {action}")

    # Make new predictions
    print("\nMaking calibrated predictions...")
    new_features = pd.DataFrame(np.random.randn(10, 5))
    calibrated_predictions = await orchestrator.predict_with_calibration(new_features)

    print(f"Generated {len(calibrated_predictions)} model predictions")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())