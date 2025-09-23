"""
Advanced Signal Validation Framework with Multi-Agent Processing
================================================================
Sophisticated signal ensemble methods with dynamic weighting and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.ensemble import IsolationForest
import ray
import asyncio
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class SignalValidation:
    """Validated trading signal with confidence metrics"""
    original_signal: Any
    validation_score: float
    confidence_interval: Tuple[float, float]
    bayesian_weight: float
    regime_adjustment: float
    decay_factor: float
    quality_score: float
    false_positive_probability: float
    expected_accuracy: float
    timestamp: datetime


@dataclass
class EnsembleSignal:
    """Ensemble signal from multiple sources"""
    symbol: str
    direction: str
    ensemble_confidence: float
    component_signals: List[Any]
    weights: List[float]
    correlation_penalty: float
    regime_factor: float
    timestamp: datetime


@ray.remote
class BayesianEnsembleAgent:
    """Agent for Bayesian Model Averaging"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.prior_weights = {}
        self.posterior_weights = {}
        self.performance_history = deque(maxlen=1000)

    async def calculate_bayesian_weights(
        self,
        signals: List[Dict[str, Any]],
        historical_performance: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate Bayesian weights for signal sources"""

        # Initialize priors (uniform if no history)
        n_sources = len(signals)
        if not self.prior_weights:
            self.prior_weights = {
                sig['source']: 1.0 / n_sources
                for sig in signals
            }

        # Calculate likelihood based on historical performance
        likelihoods = {}
        for signal in signals:
            source = signal['source']
            if source in historical_performance:
                perf = historical_performance[source]
                # Use beta distribution for binary outcomes
                successes = sum(1 for p in perf if p > 0)
                failures = len(perf) - successes

                # Beta posterior parameters
                alpha = successes + 1
                beta = failures + 1

                # Expected value as likelihood
                likelihoods[source] = alpha / (alpha + beta)
            else:
                likelihoods[source] = 0.5  # No information prior

        # Calculate posterior weights (Bayes' rule)
        posteriors = {}
        total_posterior = 0

        for source in likelihoods:
            posterior = likelihoods[source] * self.prior_weights.get(source, 1/n_sources)
            posteriors[source] = posterior
            total_posterior += posterior

        # Normalize
        if total_posterior > 0:
            for source in posteriors:
                posteriors[source] /= total_posterior

        # Update priors for next iteration
        self.prior_weights = posteriors.copy()

        return posteriors

    async def ensemble_signals(
        self,
        signals: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> EnsembleSignal:
        """Create ensemble signal with Bayesian weights"""

        # Aggregate directional signals
        long_weight = 0
        short_weight = 0

        for signal in signals:
            source = signal['source']
            weight = weights.get(source, 0)
            confidence = signal.get('confidence', 0.5)

            if signal['direction'] == 'long':
                long_weight += weight * confidence
            else:
                short_weight += weight * confidence

        # Determine ensemble direction
        if long_weight > short_weight:
            direction = 'long'
            ensemble_confidence = long_weight / (long_weight + short_weight)
        else:
            direction = 'short'
            ensemble_confidence = short_weight / (long_weight + short_weight)

        return EnsembleSignal(
            symbol=signals[0].get('symbol', 'UNKNOWN'),
            direction=direction,
            ensemble_confidence=ensemble_confidence,
            component_signals=signals,
            weights=list(weights.values()),
            correlation_penalty=0,  # Will be calculated separately
            regime_factor=1.0,
            timestamp=datetime.now()
        )


@ray.remote
class SignalQualityAgent:
    """Agent for signal quality assessment"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.quality_history = deque(maxlen=1000)
        self.anomaly_detector = IsolationForest(contamination=0.1)

    async def assess_signal_quality(
        self,
        signal: Dict[str, Any],
        historical_signals: List[Dict[str, Any]]
    ) -> float:
        """Assess quality and reliability of signal"""

        quality_scores = []

        # 1. Confidence consistency check
        if 'confidence' in signal:
            conf = signal['confidence']
            # Penalize extreme confidence (overconfidence)
            if conf > 0.95 or conf < 0.05:
                quality_scores.append(0.5)
            else:
                quality_scores.append(1.0)

        # 2. Signal stability (not flipping too frequently)
        if historical_signals:
            recent_directions = [s.get('direction') for s in historical_signals[-10:]]
            flips = sum(1 for i in range(1, len(recent_directions))
                       if recent_directions[i] != recent_directions[i-1])
            stability_score = 1.0 - (flips / max(len(recent_directions)-1, 1))
            quality_scores.append(stability_score)

        # 3. Anomaly detection
        if len(historical_signals) > 20:
            # Extract features for anomaly detection
            features = self._extract_signal_features(signal)
            historical_features = [self._extract_signal_features(s)
                                  for s in historical_signals[-100:]]

            if historical_features:
                # Fit and predict
                try:
                    self.anomaly_detector.fit(historical_features)
                    is_anomaly = self.anomaly_detector.predict([features])[0]
                    anomaly_score = 1.0 if is_anomaly == 1 else 0.7
                    quality_scores.append(anomaly_score)
                except:
                    quality_scores.append(0.8)

        # 4. Signal strength assessment
        if 'strength' in signal or 'magnitude' in signal:
            strength = signal.get('strength', signal.get('magnitude', 0))
            # Normalize strength to [0, 1]
            strength_score = min(abs(strength), 1.0)
            quality_scores.append(strength_score)

        # Calculate weighted average quality score
        if quality_scores:
            quality_score = np.mean(quality_scores)
        else:
            quality_score = 0.5  # Default neutral quality

        return quality_score

    def _extract_signal_features(self, signal: Dict[str, Any]) -> List[float]:
        """Extract numerical features from signal"""
        features = []

        # Basic features
        features.append(signal.get('confidence', 0.5))
        features.append(1.0 if signal.get('direction') == 'long' else -1.0)
        features.append(signal.get('strength', 0))
        features.append(signal.get('expected_return', 0))
        features.append(signal.get('risk_score', 0.5))

        # Time-based features
        if 'timestamp' in signal:
            hour = signal['timestamp'].hour
            features.append(hour / 24.0)  # Normalized hour

        return features

    async def detect_false_positives(
        self,
        signal: Dict[str, Any],
        outcome_history: List[Tuple[Dict, bool]]
    ) -> float:
        """Estimate false positive probability"""

        if not outcome_history:
            return 0.5  # No history, assume 50%

        # Find similar historical signals
        similar_signals = []
        for hist_signal, outcome in outcome_history:
            similarity = self._calculate_signal_similarity(signal, hist_signal)
            if similarity > 0.7:  # Threshold for similarity
                similar_signals.append((similarity, outcome))

        if not similar_signals:
            return 0.5  # No similar signals found

        # Weight outcomes by similarity
        false_positives = 0
        total_weight = 0

        for similarity, outcome in similar_signals:
            weight = similarity
            if not outcome:  # False positive
                false_positives += weight
            total_weight += weight

        if total_weight > 0:
            fp_probability = false_positives / total_weight
        else:
            fp_probability = 0.5

        return fp_probability

    def _calculate_signal_similarity(
        self,
        signal1: Dict[str, Any],
        signal2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two signals"""

        similarities = []

        # Direction match
        if signal1.get('direction') == signal2.get('direction'):
            similarities.append(1.0)
        else:
            similarities.append(0.0)

        # Confidence similarity
        conf1 = signal1.get('confidence', 0.5)
        conf2 = signal2.get('confidence', 0.5)
        conf_sim = 1.0 - abs(conf1 - conf2)
        similarities.append(conf_sim)

        # Symbol match (if available)
        if 'symbol' in signal1 and 'symbol' in signal2:
            if signal1['symbol'] == signal2['symbol']:
                similarities.append(1.0)
            else:
                similarities.append(0.0)

        return np.mean(similarities) if similarities else 0.0


@ray.remote
class DynamicWeightingAgent:
    """Agent for dynamic correlation-based weighting"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.correlation_matrix = None
        self.weight_history = deque(maxlen=100)

    async def calculate_dynamic_weights(
        self,
        signals: List[Dict[str, Any]],
        performance_matrix: pd.DataFrame
    ) -> List[float]:
        """Calculate dynamic weights based on correlation and performance"""

        n_signals = len(signals)

        # 1. Calculate correlation matrix
        if performance_matrix is not None and len(performance_matrix) > 10:
            corr_matrix = performance_matrix.corr()

            # 2. Apply correlation penalty (diversification bonus)
            correlation_penalties = []
            for i in range(n_signals):
                # Average correlation with other signals
                if i < len(corr_matrix):
                    avg_corr = corr_matrix.iloc[i].drop(corr_matrix.index[i]).mean()
                    # Lower weight for highly correlated signals
                    penalty = 1.0 - abs(avg_corr) * 0.5
                    correlation_penalties.append(max(penalty, 0.3))
                else:
                    correlation_penalties.append(1.0)
        else:
            correlation_penalties = [1.0] * n_signals

        # 3. Performance-based weights
        performance_weights = []
        for signal in signals:
            # Use recent performance if available
            if 'recent_performance' in signal:
                perf = signal['recent_performance']
                # Sigmoid transformation to [0, 1]
                weight = 1 / (1 + np.exp(-5 * perf))
            else:
                weight = 0.5
            performance_weights.append(weight)

        # 4. Confidence-based weights
        confidence_weights = [s.get('confidence', 0.5) for s in signals]

        # 5. Combine all weight factors
        final_weights = []
        for i in range(n_signals):
            combined_weight = (
                correlation_penalties[i] * 0.3 +
                performance_weights[i] * 0.5 +
                confidence_weights[i] * 0.2
            )
            final_weights.append(combined_weight)

        # Normalize weights
        total_weight = sum(final_weights)
        if total_weight > 0:
            final_weights = [w / total_weight for w in final_weights]
        else:
            final_weights = [1.0 / n_signals] * n_signals

        return final_weights

    async def apply_regime_adjustment(
        self,
        weights: List[float],
        current_regime: str,
        signal_regime_performance: Dict[str, Dict[str, float]]
    ) -> List[float]:
        """Adjust weights based on regime performance"""

        adjusted_weights = []

        for i, weight in enumerate(weights):
            # Get regime-specific performance multiplier
            signal_id = f"signal_{i}"

            if signal_id in signal_regime_performance:
                regime_perf = signal_regime_performance[signal_id].get(current_regime, 0.5)
                # Boost or reduce weight based on regime performance
                multiplier = 0.5 + regime_perf  # Range [0.5, 1.5]
            else:
                multiplier = 1.0

            adjusted_weights.append(weight * multiplier)

        # Renormalize
        total = sum(adjusted_weights)
        if total > 0:
            adjusted_weights = [w / total for w in adjusted_weights]

        return adjusted_weights


@ray.remote
class SignalDecayAgent:
    """Agent for signal decay modeling"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.decay_rates = {}

    async def calculate_signal_decay(
        self,
        signal: Dict[str, Any],
        signal_type: str
    ) -> float:
        """Calculate time-based signal decay"""

        # Define decay rates for different signal types
        decay_rates = {
            'technical': 0.1,      # Fast decay (hours)
            'fundamental': 0.01,   # Slow decay (days)
            'sentiment': 0.05,     # Medium decay
            'options_flow': 0.2,   # Very fast decay
            'microstructure': 0.3, # Ultra-fast decay
            'macro': 0.005        # Very slow decay
        }

        decay_rate = decay_rates.get(signal_type, 0.05)

        # Calculate time since signal generation
        if 'timestamp' in signal:
            time_elapsed = (datetime.now() - signal['timestamp']).total_seconds() / 3600  # Hours
        else:
            time_elapsed = 0

        # Exponential decay
        decay_factor = np.exp(-decay_rate * time_elapsed)

        # Apply minimum threshold
        decay_factor = max(decay_factor, 0.1)

        return decay_factor

    async def update_decay_rates(
        self,
        signal_outcomes: List[Tuple[str, float, float]]
    ) -> Dict[str, float]:
        """Update decay rates based on observed outcomes"""

        # signal_outcomes: [(signal_type, time_to_outcome, accuracy)]

        updated_rates = {}

        # Group by signal type
        type_outcomes = {}
        for sig_type, time_to_outcome, accuracy in signal_outcomes:
            if sig_type not in type_outcomes:
                type_outcomes[sig_type] = []
            type_outcomes[sig_type].append((time_to_outcome, accuracy))

        # Calculate optimal decay rate for each type
        for sig_type, outcomes in type_outcomes.items():
            if len(outcomes) > 10:
                # Find the time where accuracy drops below 60%
                times = [t for t, _ in outcomes]
                accuracies = [a for _, a in outcomes]

                # Fit exponential decay curve
                try:
                    from scipy.optimize import curve_fit

                    def exp_decay(t, rate):
                        return np.exp(-rate * t)

                    popt, _ = curve_fit(exp_decay, times, accuracies, p0=[0.1])
                    updated_rates[sig_type] = popt[0]
                except:
                    updated_rates[sig_type] = 0.05  # Default

        # Update internal rates
        self.decay_rates.update(updated_rates)

        return updated_rates


@ray.remote
class ConfidenceIntervalAgent:
    """Agent for confidence interval estimation"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.calibration_history = deque(maxlen=1000)

    async def estimate_confidence_interval(
        self,
        signal: Dict[str, Any],
        historical_errors: List[float]
    ) -> Tuple[float, float]:
        """Estimate confidence interval for signal prediction"""

        base_confidence = signal.get('confidence', 0.5)
        predicted_value = signal.get('expected_return', 0)

        # Calculate standard error from historical errors
        if historical_errors and len(historical_errors) > 10:
            std_error = np.std(historical_errors)

            # Adjust for signal confidence
            adjusted_std = std_error * (2 - base_confidence)
        else:
            # Use default based on signal type
            adjusted_std = 0.02  # 2% default

        # Calculate confidence interval (95%)
        z_score = 1.96
        lower_bound = predicted_value - z_score * adjusted_std
        upper_bound = predicted_value + z_score * adjusted_std

        # Apply signal-specific adjustments
        if 'volatility' in signal:
            vol_adjustment = signal['volatility'] * 0.5
            lower_bound -= vol_adjustment
            upper_bound += vol_adjustment

        return (lower_bound, upper_bound)

    async def calibrate_confidence(
        self,
        predictions: List[Tuple[float, float, float]]
    ) -> Dict[str, float]:
        """Calibrate confidence scores to actual probabilities"""

        # predictions: [(predicted_conf, actual_outcome, predicted_value)]

        if len(predictions) < 50:
            return {'calibration_factor': 1.0}

        # Bin predictions by confidence level
        bins = np.linspace(0, 1, 11)
        calibration_curve = []

        for i in range(len(bins) - 1):
            bin_mask = [
                bins[i] <= conf < bins[i+1]
                for conf, _, _ in predictions
            ]

            if sum(bin_mask) > 0:
                bin_predictions = [
                    (conf, outcome, value)
                    for (conf, outcome, value), mask in zip(predictions, bin_mask)
                    if mask
                ]

                avg_predicted_conf = np.mean([conf for conf, _, _ in bin_predictions])
                actual_success_rate = np.mean([
                    1 if outcome > 0 else 0
                    for _, outcome, _ in bin_predictions
                ])

                calibration_curve.append((avg_predicted_conf, actual_success_rate))

        # Fit calibration function
        if calibration_curve:
            x = [point[0] for point in calibration_curve]
            y = [point[1] for point in calibration_curve]

            # Linear regression for calibration
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                calibration_factor = slope
            else:
                calibration_factor = 1.0
        else:
            calibration_factor = 1.0

        return {
            'calibration_factor': calibration_factor,
            'calibration_curve': calibration_curve
        }


class AdvancedSignalValidationOrchestrator:
    """Orchestrate advanced signal validation with multi-agent processing"""

    def __init__(self):
        ray.init(ignore_reinit_error=True)

        # Initialize agents
        self.bayesian_agent = BayesianEnsembleAgent.remote("bayesian")
        self.quality_agent = SignalQualityAgent.remote("quality")
        self.weighting_agent = DynamicWeightingAgent.remote("weighting")
        self.decay_agent = SignalDecayAgent.remote("decay")
        self.confidence_agent = ConfidenceIntervalAgent.remote("confidence")

        # Performance tracking
        self.performance_history = {}
        self.signal_history = deque(maxlen=10000)
        self.validation_metrics = {}

    async def validate_signals(
        self,
        signals: List[Dict[str, Any]],
        market_regime: Optional[str] = None
    ) -> List[SignalValidation]:
        """Validate and enhance signals through multi-agent processing"""

        validated_signals = []

        # Parallel validation tasks
        validation_tasks = []

        for signal in signals:
            # Quality assessment
            quality_task = self.quality_agent.assess_signal_quality.remote(
                signal, list(self.signal_history)
            )

            # False positive detection
            fp_task = self.quality_agent.detect_false_positives.remote(
                signal, self._get_outcome_history()
            )

            # Decay calculation
            signal_type = signal.get('type', 'technical')
            decay_task = self.decay_agent.calculate_signal_decay.remote(
                signal, signal_type
            )

            # Confidence interval
            ci_task = self.confidence_agent.estimate_confidence_interval.remote(
                signal, self._get_historical_errors(signal.get('source', 'unknown'))
            )

            validation_tasks.append({
                'signal': signal,
                'quality': quality_task,
                'fp': fp_task,
                'decay': decay_task,
                'ci': ci_task
            })

        # Gather results
        for task_set in validation_tasks:
            quality_score = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task_set['quality'])
            )
            fp_prob = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task_set['fp'])
            )
            decay_factor = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task_set['decay'])
            )
            confidence_interval = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task_set['ci'])
            )

            # Calculate Bayesian weight (simplified for single signal)
            bayesian_weight = quality_score * (1 - fp_prob) * decay_factor

            # Expected accuracy
            expected_accuracy = quality_score * (1 - fp_prob)

            # Create validated signal
            validation = SignalValidation(
                original_signal=task_set['signal'],
                validation_score=quality_score * decay_factor,
                confidence_interval=confidence_interval,
                bayesian_weight=bayesian_weight,
                regime_adjustment=1.0 if not market_regime else self._get_regime_multiplier(market_regime),
                decay_factor=decay_factor,
                quality_score=quality_score,
                false_positive_probability=fp_prob,
                expected_accuracy=expected_accuracy,
                timestamp=datetime.now()
            )

            validated_signals.append(validation)

            # Store in history
            self.signal_history.append(task_set['signal'])

        return validated_signals

    async def create_ensemble_signal(
        self,
        validated_signals: List[SignalValidation],
        symbol: str
    ) -> EnsembleSignal:
        """Create ensemble signal from validated components"""

        # Group signals by symbol
        symbol_signals = [
            vs for vs in validated_signals
            if vs.original_signal.get('symbol') == symbol
        ]

        if not symbol_signals:
            return None

        # Prepare for Bayesian ensemble
        signals_dict = [
            {
                'source': vs.original_signal.get('source', 'unknown'),
                'direction': vs.original_signal.get('direction', 'neutral'),
                'confidence': vs.expected_accuracy,
                'symbol': symbol
            }
            for vs in symbol_signals
        ]

        # Calculate Bayesian weights
        historical_perf = self._get_source_performance()
        weights_task = self.bayesian_agent.calculate_bayesian_weights.remote(
            signals_dict, historical_perf
        )
        weights = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, weights_task)
        )

        # Create ensemble
        ensemble_task = self.bayesian_agent.ensemble_signals.remote(
            signals_dict, weights
        )
        ensemble = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, ensemble_task)
        )

        # Apply dynamic correlation adjustment
        if len(self.performance_history) > 10:
            perf_matrix = pd.DataFrame(self.performance_history)
            dynamic_weights_task = self.weighting_agent.calculate_dynamic_weights.remote(
                signals_dict, perf_matrix
            )
            dynamic_weights = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, dynamic_weights_task)
            )

            # Blend Bayesian and dynamic weights
            final_weights = [
                0.7 * list(weights.values())[i] + 0.3 * dynamic_weights[i]
                for i in range(len(signals_dict))
            ]

            # Normalize
            total = sum(final_weights)
            if total > 0:
                final_weights = [w / total for w in final_weights]

            ensemble.weights = final_weights

        return ensemble

    def _get_outcome_history(self) -> List[Tuple[Dict, bool]]:
        """Get historical signal outcomes for false positive detection"""
        # Placeholder - would connect to actual outcome tracking
        return []

    def _get_historical_errors(self, source: str) -> List[float]:
        """Get historical prediction errors for a source"""
        # Placeholder - would connect to actual error tracking
        return [np.random.normal(0, 0.01) for _ in range(100)]

    def _get_source_performance(self) -> Dict[str, List[float]]:
        """Get historical performance by source"""
        # Placeholder - would connect to actual performance tracking
        return {
            'technical': [np.random.uniform(-0.02, 0.03) for _ in range(100)],
            'fundamental': [np.random.uniform(-0.01, 0.02) for _ in range(100)],
            'sentiment': [np.random.uniform(-0.015, 0.025) for _ in range(100)]
        }

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get regime-specific multiplier"""
        multipliers = {
            'bull': 1.2,
            'bear': 0.8,
            'volatile': 0.9,
            'stable': 1.1
        }
        return multipliers.get(regime, 1.0)

    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation performance metrics"""
        if not self.signal_history:
            return {}

        recent_signals = list(self.signal_history)[-100:]

        return {
            'total_signals_validated': len(self.signal_history),
            'avg_quality_score': np.mean([
                s.get('quality_score', 0.5) for s in recent_signals
            ]),
            'avg_false_positive_rate': np.mean([
                s.get('fp_probability', 0.5) for s in recent_signals
            ]),
            'signal_diversity': len(set(s.get('source') for s in recent_signals)),
            'timestamp': datetime.now()
        }

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of advanced signal validation"""
    orchestrator = AdvancedSignalValidationOrchestrator()

    # Generate sample signals from different sources
    sample_signals = [
        {
            'symbol': 'AAPL',
            'source': 'technical',
            'type': 'technical',
            'direction': 'long',
            'confidence': 0.75,
            'strength': 0.8,
            'expected_return': 0.02,
            'risk_score': 0.3,
            'timestamp': datetime.now()
        },
        {
            'symbol': 'AAPL',
            'source': 'sentiment',
            'type': 'sentiment',
            'direction': 'long',
            'confidence': 0.65,
            'strength': 0.6,
            'expected_return': 0.015,
            'risk_score': 0.4,
            'timestamp': datetime.now() - timedelta(hours=1)
        },
        {
            'symbol': 'AAPL',
            'source': 'options_flow',
            'type': 'options_flow',
            'direction': 'short',
            'confidence': 0.55,
            'strength': 0.7,
            'expected_return': -0.01,
            'risk_score': 0.5,
            'timestamp': datetime.now() - timedelta(minutes=30)
        }
    ]

    # Validate signals
    print("Validating signals...")
    validated = await orchestrator.validate_signals(sample_signals, market_regime='bull')

    print(f"\nValidated {len(validated)} signals:")
    for val in validated:
        print(f"  Source: {val.original_signal['source']}")
        print(f"    Quality Score: {val.quality_score:.3f}")
        print(f"    False Positive Prob: {val.false_positive_probability:.3f}")
        print(f"    Decay Factor: {val.decay_factor:.3f}")
        print(f"    Expected Accuracy: {val.expected_accuracy:.3f}")
        print(f"    Confidence Interval: [{val.confidence_interval[0]:.3f}, {val.confidence_interval[1]:.3f}]")

    # Create ensemble signal
    print("\nCreating ensemble signal...")
    ensemble = await orchestrator.create_ensemble_signal(validated, 'AAPL')

    if ensemble:
        print(f"Ensemble Signal for {ensemble.symbol}:")
        print(f"  Direction: {ensemble.direction}")
        print(f"  Confidence: {ensemble.ensemble_confidence:.3f}")
        print(f"  Component Weights: {[f'{w:.3f}' for w in ensemble.weights]}")

    # Get validation metrics
    metrics = orchestrator.get_validation_metrics()
    print("\nValidation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())