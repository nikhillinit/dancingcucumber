"""
Enhanced Consensus Engine
=========================

Advanced consensus building system that leverages sophisticated weighted voting,
Bayesian inference, game theory principles, and machine learning to reach
optimal investment decisions from diverse investor persona analyses.

Features:
- Bayesian consensus with prior belief integration
- Game-theoretic Nash equilibrium solving
- Dynamic weight adjustment based on historical performance
- Sentiment analysis and conviction weighting
- Multi-objective optimization for risk-adjusted returns
- Real-time consensus updating with new information
- Adversarial robustness testing
- Confidence interval estimation with bootstrap methods

Author: AI Hedge Fund Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats, optimize
from scipy.special import softmax
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import scipy.linalg as la

# Bayesian inference
try:
    import pymc3 as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyMC3 not available. Bayesian features will be limited.")

# Network analysis for consensus
try:
    import networkx as nx
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False

from .ai_investor_personas import (
    InvestorPersona, PersonaAnalysis, DebateMessage, ConsensusResult,
    InvestmentAction, MarketAnalysis, AVAILABLE_PERSONAS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Methods for reaching consensus"""
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    BAYESIAN_INFERENCE = "BAYESIAN_INFERENCE"
    GAME_THEORETIC = "GAME_THEORETIC"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    ENSEMBLE = "ENSEMBLE"
    DELPHI_METHOD = "DELPHI_METHOD"


class WeightingScheme(Enum):
    """Schemes for weighting persona opinions"""
    EQUAL = "EQUAL"
    PERFORMANCE_BASED = "PERFORMANCE_BASED"
    CONFIDENCE_BASED = "CONFIDENCE_BASED"
    EXPERTISE_BASED = "EXPERTISE_BASED"
    DYNAMIC_BAYESIAN = "DYNAMIC_BAYESIAN"
    ADVERSARIAL_ROBUST = "ADVERSARIAL_ROBUST"


@dataclass
class ConsensusConfiguration:
    """Configuration for consensus building"""
    method: ConsensusMethod = ConsensusMethod.ENSEMBLE
    weighting_scheme: WeightingScheme = WeightingScheme.DYNAMIC_BAYESIAN
    confidence_threshold: float = 0.75
    max_iterations: int = 10
    convergence_tolerance: float = 0.01
    bayesian_prior_strength: float = 0.1
    enable_uncertainty_quantification: bool = True
    enable_adversarial_testing: bool = True
    bootstrap_samples: int = 1000
    monte_carlo_samples: int = 10000
    performance_lookback_days: int = 90
    min_consensus_participants: int = 3
    consensus_decay_factor: float = 0.95  # How much to weight recent performance
    risk_aversion_factor: float = 0.5  # 0 = risk neutral, 1 = very risk averse


@dataclass
class PersonaWeight:
    """Weight assignment for a persona"""
    persona_name: str
    base_weight: float
    performance_multiplier: float
    confidence_multiplier: float
    expertise_multiplier: float
    final_weight: float
    reasoning: str
    last_updated: datetime


@dataclass
class ConsensusState:
    """Current state of consensus building process"""
    iteration: int
    current_recommendation: InvestmentAction
    consensus_strength: float
    participant_weights: Dict[str, PersonaWeight]
    convergence_metrics: Dict[str, float]
    uncertainty_bounds: Tuple[float, float]
    last_updated: datetime


@dataclass
class EnhancedConsensusResult:
    """Enhanced consensus result with advanced analytics"""
    # Basic consensus info
    symbol: str
    final_recommendation: InvestmentAction
    consensus_score: float
    average_confidence: float
    recommended_position_size: float
    target_price: Optional[float]
    stop_loss: Optional[float]

    # Enhanced analytics
    participant_weights: Dict[str, PersonaWeight]
    consensus_evolution: List[ConsensusState]
    uncertainty_quantification: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    robustness_metrics: Dict[str, float]
    bayesian_posterior: Optional[Dict[str, Any]]
    game_theory_equilibrium: Optional[Dict[str, Any]]

    # Risk analysis
    risk_decomposition: Dict[str, float]
    scenario_analysis: Dict[str, Dict[str, float]]
    tail_risk_metrics: Dict[str, float]

    # Meta-analytics
    consensus_quality_score: float
    prediction_intervals: Dict[str, Tuple[float, float]]
    model_performance_metrics: Dict[str, float]

    # Execution guidance
    optimal_execution_strategy: Dict[str, Any]
    market_impact_estimates: Dict[str, float]
    timing_recommendations: Dict[str, Any]

    # Standard fields
    key_supporting_arguments: List[str]
    key_concerns: List[str]
    dissenting_opinions: List[str]
    market_conditions_assessment: str
    execution_timeline: str
    risk_level: str
    expected_return: float
    debate_summary: str
    timestamp: datetime


class PerformanceTracker:
    """Tracks and analyzes persona performance over time"""

    def __init__(self, config: ConsensusConfiguration):
        self.config = config
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.prediction_accuracy: Dict[str, float] = {}
        self.risk_adjusted_returns: Dict[str, float] = {}

    def record_performance(self, persona_name: str, prediction: PersonaAnalysis,
                          actual_outcome: Dict[str, float]):
        """Record performance outcome for a persona"""
        if persona_name not in self.performance_history:
            self.performance_history[persona_name] = []

        # Calculate prediction accuracy
        predicted_direction = 1 if prediction.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY] else -1
        actual_direction = 1 if actual_outcome.get('return', 0) > 0 else -1
        directional_accuracy = 1 if predicted_direction == actual_direction else 0

        # Calculate return prediction error
        predicted_return = prediction.expected_return if hasattr(prediction, 'expected_return') else 0
        actual_return = actual_outcome.get('return', 0)
        return_error = abs(predicted_return - actual_return)

        # Record performance metrics
        performance_record = {
            'timestamp': datetime.now(),
            'prediction': asdict(prediction),
            'actual_outcome': actual_outcome,
            'directional_accuracy': directional_accuracy,
            'return_error': return_error,
            'confidence_calibration': self._calculate_confidence_calibration(prediction, actual_outcome),
            'risk_adjusted_return': actual_return / max(0.01, actual_outcome.get('volatility', 0.1))
        }

        self.performance_history[persona_name].append(performance_record)

        # Update rolling metrics
        self._update_rolling_metrics(persona_name)

    def get_performance_weights(self, participants: List[str]) -> Dict[str, float]:
        """Calculate performance-based weights for participants"""
        weights = {}

        for persona_name in participants:
            if persona_name in self.performance_history:
                recent_performance = self._get_recent_performance(persona_name)
                weights[persona_name] = self._calculate_performance_score(recent_performance)
            else:
                weights[persona_name] = 1.0  # Default weight for new personas

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(participants)
            weights = {k: equal_weight for k in participants}

        return weights

    def _get_recent_performance(self, persona_name: str) -> List[Dict[str, Any]]:
        """Get recent performance records for a persona"""
        if persona_name not in self.performance_history:
            return []

        cutoff_date = datetime.now() - timedelta(days=self.config.performance_lookback_days)
        recent_records = [
            record for record in self.performance_history[persona_name]
            if record['timestamp'] > cutoff_date
        ]

        return recent_records

    def _calculate_performance_score(self, performance_records: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score"""
        if not performance_records:
            return 1.0

        # Weight by recency
        weights = []
        scores = []

        for i, record in enumerate(performance_records):
            # Exponential decay weighting
            weight = self.config.consensus_decay_factor ** (len(performance_records) - i - 1)
            weights.append(weight)

            # Composite score
            accuracy_score = record['directional_accuracy']
            calibration_score = 1 - record['return_error']
            risk_adj_score = max(0, record['risk_adjusted_return'])

            composite_score = (accuracy_score * 0.4 +
                             calibration_score * 0.3 +
                             risk_adj_score * 0.3)
            scores.append(composite_score)

        # Weighted average
        if sum(weights) > 0:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            weighted_score = np.mean(scores)

        return max(0.1, min(2.0, weighted_score))  # Bounded between 0.1 and 2.0

    def _calculate_confidence_calibration(self, prediction: PersonaAnalysis,
                                        outcome: Dict[str, Any]) -> float:
        """Calculate how well calibrated the confidence was"""
        # Simple calibration: if high confidence and correct, good calibration
        confidence = prediction.confidence_score / 100

        predicted_direction = 1 if prediction.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY] else -1
        actual_direction = 1 if outcome.get('return', 0) > 0 else -1
        correct = predicted_direction == actual_direction

        if correct:
            return confidence
        else:
            return 1 - confidence

    def _update_rolling_metrics(self, persona_name: str):
        """Update rolling performance metrics"""
        recent_records = self._get_recent_performance(persona_name)

        if recent_records:
            accuracies = [r['directional_accuracy'] for r in recent_records]
            risk_adj_returns = [r['risk_adjusted_return'] for r in recent_records]

            self.prediction_accuracy[persona_name] = np.mean(accuracies)
            self.risk_adjusted_returns[persona_name] = np.mean(risk_adj_returns)


class BayesianConsensusBuilder:
    """Bayesian approach to consensus building"""

    def __init__(self, config: ConsensusConfiguration):
        self.config = config
        self.prior_beliefs = self._initialize_priors()

    def build_bayesian_consensus(self, analyses: List[PersonaAnalysis],
                                market_data: MarketAnalysis) -> Dict[str, Any]:
        """Build consensus using Bayesian inference"""
        if not BAYESIAN_AVAILABLE:
            return self._fallback_bayesian_consensus(analyses)

        try:
            # Prepare data for Bayesian model
            recommendations = self._encode_recommendations(analyses)
            confidences = [a.confidence_score / 100 for a in analyses]

            # Build Bayesian model
            with pm.Model() as model:
                # Prior for true recommendation (encoded as continuous)
                true_recommendation = pm.Normal('true_recommendation', mu=0, sd=2)

                # Individual persona biases
                persona_biases = pm.Normal('persona_biases', mu=0, sd=0.5, shape=len(analyses))

                # Observation noise (inversely related to confidence)
                noise_precision = pm.Gamma('noise_precision', alpha=2, beta=1, shape=len(analyses))

                # Likelihood
                observed_recommendations = pm.Normal(
                    'observed_recommendations',
                    mu=true_recommendation + persona_biases,
                    sd=1/pm.math.sqrt(noise_precision),
                    observed=recommendations
                )

                # Sample from posterior
                trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

            # Extract results
            posterior_mean = float(np.mean(trace.posterior['true_recommendation']))
            posterior_std = float(np.std(trace.posterior['true_recommendation']))

            # Convert back to recommendation
            consensus_recommendation = self._decode_recommendation(posterior_mean)

            return {
                'recommendation': consensus_recommendation,
                'confidence': 1 - posterior_std,  # Higher std = lower confidence
                'posterior_mean': posterior_mean,
                'posterior_std': posterior_std,
                'trace': trace,
                'model_summary': az.summary(trace)
            }

        except Exception as e:
            logger.warning(f"Bayesian consensus failed: {e}")
            return self._fallback_bayesian_consensus(analyses)

    def _initialize_priors(self) -> Dict[str, Any]:
        """Initialize prior beliefs about persona performance"""
        priors = {}

        for persona_name in AVAILABLE_PERSONAS.keys():
            # Based on historical performance or persona characteristics
            persona = AVAILABLE_PERSONAS[persona_name]

            # Prior performance expectation
            if persona.characteristics.historical_returns:
                expected_performance = persona.characteristics.historical_returns / 100
            else:
                expected_performance = 0.1  # Default 10% expected return

            priors[persona_name] = {
                'expected_return': expected_performance,
                'risk_tolerance': self._map_risk_tolerance(persona.characteristics.risk_tolerance),
                'confidence_bias': 0.0,  # Neutral prior
                'sector_expertise': self._assess_sector_expertise(persona)
            }

        return priors

    def _encode_recommendations(self, analyses: List[PersonaAnalysis]) -> List[float]:
        """Encode recommendations as continuous values"""
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        return [encoding[analysis.recommendation] for analysis in analyses]

    def _decode_recommendation(self, encoded_value: float) -> InvestmentAction:
        """Decode continuous value back to recommendation"""
        if encoded_value <= -1.5:
            return InvestmentAction.STRONG_SELL
        elif encoded_value <= -0.5:
            return InvestmentAction.SELL
        elif encoded_value <= 0.5:
            return InvestmentAction.HOLD
        elif encoded_value <= 1.5:
            return InvestmentAction.BUY
        else:
            return InvestmentAction.STRONG_BUY

    def _fallback_bayesian_consensus(self, analyses: List[PersonaAnalysis]) -> Dict[str, Any]:
        """Fallback Bayesian consensus without PyMC3"""
        # Simple Bayesian updating with normal distribution
        recommendations = self._encode_recommendations(analyses)
        confidences = [a.confidence_score / 100 for a in analyses]

        # Prior
        prior_mean = 0  # Neutral prior
        prior_var = 1

        # Likelihood precision (confidence-weighted)
        precisions = [c**2 for c in confidences]  # Higher confidence = higher precision

        # Bayesian update
        posterior_precision = 1/prior_var + sum(precisions)
        posterior_mean = (prior_mean/prior_var + sum(r*p for r,p in zip(recommendations, precisions))) / posterior_precision
        posterior_var = 1 / posterior_precision

        consensus_recommendation = self._decode_recommendation(posterior_mean)
        confidence = min(1.0, 1 / np.sqrt(posterior_var))

        return {
            'recommendation': consensus_recommendation,
            'confidence': confidence,
            'posterior_mean': posterior_mean,
            'posterior_std': np.sqrt(posterior_var)
        }

    def _map_risk_tolerance(self, risk_tolerance) -> float:
        """Map risk tolerance enum to numerical value"""
        mapping = {
            'CONSERVATIVE': 0.2,
            'MODERATE': 0.5,
            'AGGRESSIVE': 0.8,
            'VERY_AGGRESSIVE': 1.0
        }
        return mapping.get(risk_tolerance.value if hasattr(risk_tolerance, 'value') else str(risk_tolerance), 0.5)

    def _assess_sector_expertise(self, persona: InvestorPersona) -> float:
        """Assess persona's sector expertise"""
        # Simple heuristic based on preferred sectors
        preferred_count = len(persona.characteristics.preferred_sectors)
        avoided_count = len(persona.characteristics.avoided_sectors)

        if preferred_count > avoided_count:
            return 0.8  # Specialized expertise
        elif preferred_count == 0 and avoided_count == 0:
            return 0.6  # General expertise
        else:
            return 0.4  # Limited expertise


class GameTheoreticConsensus:
    """Game-theoretic approach to consensus building"""

    def __init__(self, config: ConsensusConfiguration):
        self.config = config

    def find_nash_equilibrium(self, analyses: List[PersonaAnalysis],
                            market_data: MarketAnalysis) -> Dict[str, Any]:
        """Find Nash equilibrium for investment recommendations"""

        # Create payoff matrix
        payoff_matrix = self._create_payoff_matrix(analyses, market_data)

        # Find Nash equilibrium
        equilibrium = self._solve_nash_equilibrium(payoff_matrix)

        # Convert to consensus recommendation
        consensus_rec = self._equilibrium_to_recommendation(equilibrium, analyses)

        return {
            'recommendation': consensus_rec,
            'equilibrium_weights': equilibrium,
            'payoff_matrix': payoff_matrix,
            'stability_score': self._calculate_stability(equilibrium, payoff_matrix)
        }

    def _create_payoff_matrix(self, analyses: List[PersonaAnalysis],
                            market_data: MarketAnalysis) -> np.ndarray:
        """Create payoff matrix for game theory analysis"""
        n = len(analyses)
        payoff_matrix = np.zeros((n, n))

        for i, analysis_i in enumerate(analyses):
            for j, analysis_j in enumerate(analyses):
                if i == j:
                    # Self-payoff based on confidence
                    payoff_matrix[i, j] = analysis_i.confidence_score / 100
                else:
                    # Interaction payoff based on recommendation similarity
                    similarity = self._calculate_recommendation_similarity(
                        analysis_i.recommendation, analysis_j.recommendation
                    )

                    # Risk adjustment
                    risk_penalty = self._calculate_risk_penalty(analysis_i, analysis_j, market_data)

                    payoff_matrix[i, j] = similarity - risk_penalty

        return payoff_matrix

    def _calculate_recommendation_similarity(self, rec1: InvestmentAction, rec2: InvestmentAction) -> float:
        """Calculate similarity between two recommendations"""
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        val1, val2 = encoding[rec1], encoding[rec2]
        max_distance = 4  # Maximum possible distance

        return 1 - abs(val1 - val2) / max_distance

    def _calculate_risk_penalty(self, analysis1: PersonaAnalysis, analysis2: PersonaAnalysis,
                              market_data: MarketAnalysis) -> float:
        """Calculate risk penalty for conflicting positions"""
        # Higher penalty if both are extreme positions in opposite directions
        if ((analysis1.recommendation in [InvestmentAction.STRONG_BUY] and
             analysis2.recommendation in [InvestmentAction.STRONG_SELL]) or
            (analysis1.recommendation in [InvestmentAction.STRONG_SELL] and
             analysis2.recommendation in [InvestmentAction.STRONG_BUY])):

            # Check market volatility as risk factor
            volatility = market_data.technical_indicators.get('volatility_20d', 0.2)
            return volatility * self.config.risk_aversion_factor

        return 0.0

    def _solve_nash_equilibrium(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Solve for Nash equilibrium using linear programming"""
        n = payoff_matrix.shape[0]

        try:
            # Use scipy optimization to find mixed strategy Nash equilibrium
            def objective(x):
                return -np.min(payoff_matrix @ x)

            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Probabilities sum to 1
            ]
            bounds = [(0, 1) for _ in range(n)]  # Probabilities are non-negative

            result = optimize.minimize(
                objective,
                np.ones(n) / n,  # Initial equal probabilities
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n) / n

        except Exception as e:
            logger.warning(f"Nash equilibrium solving failed: {e}")
            return np.ones(n) / n

    def _equilibrium_to_recommendation(self, equilibrium: np.ndarray,
                                     analyses: List[PersonaAnalysis]) -> InvestmentAction:
        """Convert equilibrium weights to consensus recommendation"""
        # Weight recommendations by equilibrium probabilities
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        weighted_sum = sum(
            equilibrium[i] * encoding[analysis.recommendation]
            for i, analysis in enumerate(analyses)
        )

        # Convert back to recommendation
        if weighted_sum <= -1.5:
            return InvestmentAction.STRONG_SELL
        elif weighted_sum <= -0.5:
            return InvestmentAction.SELL
        elif weighted_sum <= 0.5:
            return InvestmentAction.HOLD
        elif weighted_sum <= 1.5:
            return InvestmentAction.BUY
        else:
            return InvestmentAction.STRONG_BUY

    def _calculate_stability(self, equilibrium: np.ndarray, payoff_matrix: np.ndarray) -> float:
        """Calculate stability score of the equilibrium"""
        # Check if any player wants to deviate
        current_payoffs = payoff_matrix @ equilibrium

        stability_scores = []
        for i in range(len(equilibrium)):
            # Best response payoff
            best_response_payoff = np.max(payoff_matrix[i, :])
            current_payoff = current_payoffs[i]

            if best_response_payoff > 0:
                stability_score = current_payoff / best_response_payoff
            else:
                stability_score = 1.0

            stability_scores.append(stability_score)

        return np.mean(stability_scores)


class MLConsensusPredictor:
    """Machine learning approach to consensus prediction"""

    def __init__(self, config: ConsensusConfiguration):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train_consensus_model(self, historical_data: List[Dict[str, Any]]):
        """Train ML model to predict consensus outcomes"""
        if not historical_data:
            logger.warning("No historical data provided for ML consensus training")
            return

        # Prepare features and targets
        features = []
        targets = []

        for data_point in historical_data:
            if 'analyses' in data_point and 'actual_outcome' in data_point:
                feature_vector = self._extract_features(data_point['analyses'], data_point.get('market_data'))
                target = self._encode_outcome(data_point['actual_outcome'])

                features.append(feature_vector)
                targets.append(target)

        if not features:
            logger.warning("No valid features extracted for ML training")
            return

        X = np.array(features)
        y = np.array(targets)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train ensemble model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        logger.info(f"ML consensus model trained with MSE: {mse:.4f}")

    def predict_consensus(self, analyses: List[PersonaAnalysis],
                         market_data: MarketAnalysis) -> Dict[str, Any]:
        """Predict consensus using trained ML model"""
        if not self.is_trained:
            return self._fallback_ml_consensus(analyses)

        try:
            # Extract features
            features = self._extract_features(analyses, market_data)
            X = np.array([features])
            X_scaled = self.scaler.transform(X)

            # Predict
            prediction = self.model.predict(X_scaled)[0]
            confidence = self._calculate_prediction_confidence(X_scaled)

            # Convert to recommendation
            recommendation = self._decode_ml_prediction(prediction)

            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'prediction_value': prediction,
                'feature_importance': self._get_feature_importance() if hasattr(self.model, 'feature_importances_') else None
            }

        except Exception as e:
            logger.warning(f"ML consensus prediction failed: {e}")
            return self._fallback_ml_consensus(analyses)

    def _extract_features(self, analyses: List[PersonaAnalysis],
                         market_data: Optional[MarketAnalysis] = None) -> List[float]:
        """Extract features for ML model"""
        features = []

        # Recommendation distribution
        rec_counts = {action: 0 for action in InvestmentAction}
        for analysis in analyses:
            rec_counts[analysis.recommendation] += 1

        for action in InvestmentAction:
            features.append(rec_counts[action] / len(analyses))

        # Confidence statistics
        confidences = [a.confidence_score for a in analyses]
        features.extend([
            np.mean(confidences),
            np.std(confidences),
            np.min(confidences),
            np.max(confidences)
        ])

        # Position size statistics
        position_sizes = [a.position_size_recommendation for a in analyses]
        features.extend([
            np.mean(position_sizes),
            np.std(position_sizes)
        ])

        # Market data features
        if market_data:
            features.extend([
                market_data.price_change_percent,
                market_data.technical_indicators.get('rsi', 50),
                market_data.technical_indicators.get('volatility_20d', 0.2),
                market_data.pe_ratio or 15,
                market_data.beta or 1.0
            ])
        else:
            features.extend([0, 50, 0.2, 15, 1.0])  # Default values

        # Persona diversity
        persona_types = set(a.persona_name for a in analyses)
        features.append(len(persona_types) / len(AVAILABLE_PERSONAS))

        return features

    def _encode_outcome(self, outcome: Dict[str, Any]) -> float:
        """Encode actual outcome for training"""
        # Use actual return as target
        return outcome.get('return', 0.0)

    def _decode_ml_prediction(self, prediction: float) -> InvestmentAction:
        """Decode ML prediction to recommendation"""
        if prediction <= -0.1:
            return InvestmentAction.SELL
        elif prediction <= -0.02:
            return InvestmentAction.HOLD
        elif prediction <= 0.1:
            return InvestmentAction.BUY
        else:
            return InvestmentAction.STRONG_BUY

    def _calculate_prediction_confidence(self, X_scaled: np.ndarray) -> float:
        """Calculate confidence in ML prediction"""
        if hasattr(self.model, 'predict_proba'):
            # For classification models
            probabilities = self.model.predict_proba(X_scaled)[0]
            return float(np.max(probabilities))
        else:
            # For regression models, use simplified confidence
            return 0.7  # Default confidence

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not hasattr(self.model, 'feature_importances_'):
            return {}

        feature_names = [
            'strong_sell', 'sell', 'hold', 'buy', 'strong_buy',
            'conf_mean', 'conf_std', 'conf_min', 'conf_max',
            'pos_mean', 'pos_std',
            'price_change', 'rsi', 'volatility', 'pe_ratio', 'beta',
            'persona_diversity'
        ]

        importance_dict = {}
        for i, importance in enumerate(self.model.feature_importances_):
            if i < len(feature_names):
                importance_dict[feature_names[i]] = float(importance)

        return importance_dict

    def _fallback_ml_consensus(self, analyses: List[PersonaAnalysis]) -> Dict[str, Any]:
        """Fallback ML consensus when model is not available"""
        # Simple majority vote
        rec_counts = {action: 0 for action in InvestmentAction}
        total_confidence = 0

        for analysis in analyses:
            rec_counts[analysis.recommendation] += 1
            total_confidence += analysis.confidence_score

        # Find majority recommendation
        majority_rec = max(rec_counts, key=rec_counts.get)
        confidence = (total_confidence / len(analyses)) / 100

        return {
            'recommendation': majority_rec,
            'confidence': confidence,
            'method': 'fallback_majority_vote'
        }


class EnhancedConsensusEngine:
    """Main enhanced consensus engine"""

    def __init__(self, config: ConsensusConfiguration):
        self.config = config
        self.performance_tracker = PerformanceTracker(config)
        self.bayesian_builder = BayesianConsensusBuilder(config)
        self.game_theory = GameTheoreticConsensus(config)
        self.ml_predictor = MLConsensusPredictor(config)
        self.consensus_history: List[ConsensusState] = []

    def build_enhanced_consensus(self, symbol: str, analyses: List[PersonaAnalysis],
                               debate_rounds: List[Any], market_data: MarketAnalysis) -> EnhancedConsensusResult:
        """Build comprehensive enhanced consensus"""
        logger.info(f"Building enhanced consensus for {symbol} with {len(analyses)} analyses")

        try:
            # Calculate participant weights
            participant_weights = self._calculate_participant_weights(analyses, market_data)

            # Apply different consensus methods
            consensus_results = {}

            if self.config.method in [ConsensusMethod.WEIGHTED_AVERAGE, ConsensusMethod.ENSEMBLE]:
                consensus_results['weighted'] = self._weighted_average_consensus(analyses, participant_weights)

            if self.config.method in [ConsensusMethod.BAYESIAN_INFERENCE, ConsensusMethod.ENSEMBLE]:
                consensus_results['bayesian'] = self.bayesian_builder.build_bayesian_consensus(analyses, market_data)

            if self.config.method in [ConsensusMethod.GAME_THEORETIC, ConsensusMethod.ENSEMBLE]:
                consensus_results['game_theory'] = self.game_theory.find_nash_equilibrium(analyses, market_data)

            if self.config.method in [ConsensusMethod.MACHINE_LEARNING, ConsensusMethod.ENSEMBLE]:
                consensus_results['ml'] = self.ml_predictor.predict_consensus(analyses, market_data)

            # Combine results for ensemble method
            if self.config.method == ConsensusMethod.ENSEMBLE:
                final_consensus = self._combine_consensus_methods(consensus_results)
            else:
                method_key = self.config.method.value.lower().replace('_', '')
                final_consensus = consensus_results.get(method_key, consensus_results[list(consensus_results.keys())[0]])

            # Calculate advanced analytics
            uncertainty_metrics = self._calculate_uncertainty_quantification(analyses, consensus_results)
            sensitivity_analysis = self._perform_sensitivity_analysis(analyses, participant_weights)
            robustness_metrics = self._test_adversarial_robustness(analyses, final_consensus)

            # Risk analysis
            risk_decomposition = self._decompose_risk_factors(analyses, market_data)
            scenario_analysis = self._perform_scenario_analysis(analyses, market_data)
            tail_risk_metrics = self._calculate_tail_risk_metrics(analyses)

            # Meta-analytics
            consensus_quality = self._assess_consensus_quality(analyses, final_consensus, consensus_results)
            prediction_intervals = self._calculate_prediction_intervals(analyses, final_consensus)

            # Execution optimization
            execution_strategy = self._optimize_execution_strategy(final_consensus, market_data)
            market_impact = self._estimate_market_impact(final_consensus, market_data)
            timing_recs = self._generate_timing_recommendations(final_consensus, market_data)

            # Extract key insights from debate
            supporting_args, concerns, dissenting = self._extract_debate_insights(debate_rounds, analyses)

            # Build consensus state
            consensus_state = ConsensusState(
                iteration=len(self.consensus_history) + 1,
                current_recommendation=final_consensus['recommendation'],
                consensus_strength=final_consensus.get('confidence', 0.5),
                participant_weights=participant_weights,
                convergence_metrics={'final_confidence': final_consensus.get('confidence', 0.5)},
                uncertainty_bounds=uncertainty_metrics.get('bounds', (0.0, 1.0)),
                last_updated=datetime.now()
            )

            self.consensus_history.append(consensus_state)

            # Create comprehensive result
            return EnhancedConsensusResult(
                symbol=symbol,
                final_recommendation=final_consensus['recommendation'],
                consensus_score=final_consensus.get('confidence', 0.5) * 100,
                average_confidence=np.mean([a.confidence_score for a in analyses]),
                recommended_position_size=self._calculate_position_size(analyses, final_consensus),
                target_price=self._calculate_target_price(analyses, final_consensus),
                stop_loss=self._calculate_stop_loss(analyses, market_data),

                # Enhanced analytics
                participant_weights=participant_weights,
                consensus_evolution=self.consensus_history.copy(),
                uncertainty_quantification=uncertainty_metrics,
                sensitivity_analysis=sensitivity_analysis,
                robustness_metrics=robustness_metrics,
                bayesian_posterior=consensus_results.get('bayesian'),
                game_theory_equilibrium=consensus_results.get('game_theory'),

                # Risk analysis
                risk_decomposition=risk_decomposition,
                scenario_analysis=scenario_analysis,
                tail_risk_metrics=tail_risk_metrics,

                # Meta-analytics
                consensus_quality_score=consensus_quality,
                prediction_intervals=prediction_intervals,
                model_performance_metrics=self._get_model_performance_metrics(),

                # Execution guidance
                optimal_execution_strategy=execution_strategy,
                market_impact_estimates=market_impact,
                timing_recommendations=timing_recs,

                # Standard fields
                key_supporting_arguments=supporting_args,
                key_concerns=concerns,
                dissenting_opinions=dissenting,
                market_conditions_assessment=self._assess_market_conditions(market_data),
                execution_timeline=self._determine_execution_timeline(final_consensus),
                risk_level=self._assess_risk_level(analyses, uncertainty_metrics),
                expected_return=self._calculate_expected_return(analyses, final_consensus),
                debate_summary=self._generate_debate_summary(debate_rounds),
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Enhanced consensus building failed: {e}")
            return self._create_fallback_consensus(symbol, analyses, market_data)

    def _calculate_participant_weights(self, analyses: List[PersonaAnalysis],
                                     market_data: MarketAnalysis) -> Dict[str, PersonaWeight]:
        """Calculate sophisticated participant weights"""
        weights = {}
        participants = [a.persona_name for a in analyses]

        # Performance-based weights
        performance_weights = self.performance_tracker.get_performance_weights(participants)

        # Confidence-based weights
        confidence_weights = {a.persona_name: a.confidence_score / 100 for a in analyses}

        # Expertise-based weights (sector-specific)
        expertise_weights = self._calculate_expertise_weights(analyses, market_data)

        # Combine weights based on weighting scheme
        for analysis in analyses:
            persona_name = analysis.persona_name

            base_weight = 1.0 / len(analyses)  # Equal base
            perf_mult = performance_weights.get(persona_name, 1.0)
            conf_mult = confidence_weights.get(persona_name, 0.5)
            exp_mult = expertise_weights.get(persona_name, 1.0)

            if self.config.weighting_scheme == WeightingScheme.EQUAL:
                final_weight = base_weight
            elif self.config.weighting_scheme == WeightingScheme.PERFORMANCE_BASED:
                final_weight = base_weight * perf_mult
            elif self.config.weighting_scheme == WeightingScheme.CONFIDENCE_BASED:
                final_weight = base_weight * conf_mult
            elif self.config.weighting_scheme == WeightingScheme.EXPERTISE_BASED:
                final_weight = base_weight * exp_mult
            else:  # DYNAMIC_BAYESIAN
                final_weight = base_weight * (perf_mult * 0.4 + conf_mult * 0.3 + exp_mult * 0.3)

            weights[persona_name] = PersonaWeight(
                persona_name=persona_name,
                base_weight=base_weight,
                performance_multiplier=perf_mult,
                confidence_multiplier=conf_mult,
                expertise_multiplier=exp_mult,
                final_weight=final_weight,
                reasoning=f"Perf: {perf_mult:.2f}, Conf: {conf_mult:.2f}, Exp: {exp_mult:.2f}",
                last_updated=datetime.now()
            )

        # Normalize final weights
        total_weight = sum(w.final_weight for w in weights.values())
        if total_weight > 0:
            for weight_obj in weights.values():
                weight_obj.final_weight /= total_weight

        return weights

    def _calculate_expertise_weights(self, analyses: List[PersonaAnalysis],
                                   market_data: MarketAnalysis) -> Dict[str, float]:
        """Calculate expertise-based weights"""
        weights = {}

        # Get sector from market data
        current_sector = market_data.fundamentals.get('sector', 'Unknown')

        for analysis in analyses:
            persona_name = analysis.persona_name

            if persona_name in AVAILABLE_PERSONAS:
                persona = AVAILABLE_PERSONAS[persona_name]

                # Check sector alignment
                preferred_sectors = persona.characteristics.preferred_sectors
                avoided_sectors = persona.characteristics.avoided_sectors

                if current_sector in preferred_sectors:
                    weights[persona_name] = 1.5  # Boost for sector expertise
                elif current_sector in avoided_sectors:
                    weights[persona_name] = 0.7  # Reduce for sector avoidance
                else:
                    weights[persona_name] = 1.0  # Neutral
            else:
                weights[persona_name] = 1.0

        return weights

    def _weighted_average_consensus(self, analyses: List[PersonaAnalysis],
                                  weights: Dict[str, PersonaWeight]) -> Dict[str, Any]:
        """Calculate weighted average consensus"""
        # Encode recommendations
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        weighted_sum = 0
        weighted_confidence = 0
        total_weight = 0

        for analysis in analyses:
            weight = weights[analysis.persona_name].final_weight
            weighted_sum += encoding[analysis.recommendation] * weight
            weighted_confidence += analysis.confidence_score * weight
            total_weight += weight

        if total_weight > 0:
            avg_recommendation = weighted_sum / total_weight
            avg_confidence = weighted_confidence / total_weight / 100  # Convert to 0-1
        else:
            avg_recommendation = 0
            avg_confidence = 0.5

        # Convert back to recommendation
        consensus_rec = self._decode_recommendation(avg_recommendation)

        return {
            'recommendation': consensus_rec,
            'confidence': avg_confidence,
            'weighted_average': avg_recommendation
        }

    def _decode_recommendation(self, value: float) -> InvestmentAction:
        """Decode numerical value to recommendation"""
        if value <= -1.5:
            return InvestmentAction.STRONG_SELL
        elif value <= -0.5:
            return InvestmentAction.SELL
        elif value <= 0.5:
            return InvestmentAction.HOLD
        elif value <= 1.5:
            return InvestmentAction.BUY
        else:
            return InvestmentAction.STRONG_BUY

    def _combine_consensus_methods(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple consensus methods"""
        if not results:
            return {'recommendation': InvestmentAction.HOLD, 'confidence': 0.5}

        # Weight different methods
        method_weights = {
            'weighted': 0.3,
            'bayesian': 0.3,
            'game_theory': 0.2,
            'ml': 0.2
        }

        # Encode recommendations
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        weighted_sum = 0
        weighted_confidence = 0
        total_weight = 0

        for method, result in results.items():
            if method in method_weights and 'recommendation' in result:
                weight = method_weights[method]
                rec_value = encoding[result['recommendation']]
                confidence = result.get('confidence', 0.5)

                weighted_sum += rec_value * weight * confidence
                weighted_confidence += confidence * weight
                total_weight += weight * confidence

        if total_weight > 0:
            ensemble_rec_value = weighted_sum / total_weight
            ensemble_confidence = weighted_confidence / sum(method_weights[m] for m in results.keys() if m in method_weights)
        else:
            ensemble_rec_value = 0
            ensemble_confidence = 0.5

        ensemble_recommendation = self._decode_recommendation(ensemble_rec_value)

        return {
            'recommendation': ensemble_recommendation,
            'confidence': ensemble_confidence,
            'method_results': results,
            'ensemble_value': ensemble_rec_value
        }

    def _calculate_uncertainty_quantification(self, analyses: List[PersonaAnalysis],
                                            consensus_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics"""
        uncertainties = {}

        # Recommendation variance
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        rec_values = [encoding[a.recommendation] for a in analyses]
        uncertainties['recommendation_variance'] = float(np.var(rec_values))

        # Confidence variance
        confidences = [a.confidence_score for a in analyses]
        uncertainties['confidence_variance'] = float(np.var(confidences))

        # Model disagreement
        if len(consensus_results) > 1:
            method_recs = [encoding[result['recommendation']] for result in consensus_results.values() if 'recommendation' in result]
            if method_recs:
                uncertainties['model_disagreement'] = float(np.var(method_recs))
            else:
                uncertainties['model_disagreement'] = 0.0
        else:
            uncertainties['model_disagreement'] = 0.0

        # Overall uncertainty bounds
        max_uncertainty = max(uncertainties.values())
        min_uncertainty = min(uncertainties.values())
        uncertainties['bounds'] = (min_uncertainty, max_uncertainty)

        return uncertainties

    def _perform_sensitivity_analysis(self, analyses: List[PersonaAnalysis],
                                    weights: Dict[str, PersonaWeight]) -> Dict[str, float]:
        """Perform sensitivity analysis on consensus results"""
        sensitivity = {}

        # Test weight perturbations
        base_consensus = self._weighted_average_consensus(analyses, weights)
        base_value = self._encode_recommendation(base_consensus['recommendation'])

        for persona_name in weights.keys():
            # Increase weight by 50%
            modified_weights = weights.copy()
            modified_weights[persona_name] = PersonaWeight(
                persona_name=persona_name,
                base_weight=weights[persona_name].base_weight,
                performance_multiplier=weights[persona_name].performance_multiplier,
                confidence_multiplier=weights[persona_name].confidence_multiplier,
                expertise_multiplier=weights[persona_name].expertise_multiplier,
                final_weight=weights[persona_name].final_weight * 1.5,
                reasoning=weights[persona_name].reasoning,
                last_updated=weights[persona_name].last_updated
            )

            # Renormalize
            total_weight = sum(w.final_weight for w in modified_weights.values())
            for w in modified_weights.values():
                w.final_weight /= total_weight

            modified_consensus = self._weighted_average_consensus(analyses, modified_weights)
            modified_value = self._encode_recommendation(modified_consensus['recommendation'])

            sensitivity[persona_name] = abs(modified_value - base_value)

        return sensitivity

    def _test_adversarial_robustness(self, analyses: List[PersonaAnalysis],
                                   consensus: Dict[str, Any]) -> Dict[str, float]:
        """Test robustness against adversarial inputs"""
        if not self.config.enable_adversarial_testing:
            return {'adversarial_robustness': 1.0}

        robustness_metrics = {}

        # Test against outlier analyses
        base_recommendation = consensus['recommendation']

        # Add extreme outlier
        outlier_analysis = PersonaAnalysis(
            persona_name="Adversarial_Outlier",
            symbol=analyses[0].symbol,
            recommendation=InvestmentAction.STRONG_SELL if base_recommendation != InvestmentAction.STRONG_SELL else InvestmentAction.STRONG_BUY,
            confidence_score=95.0,
            target_price=None,
            stop_loss=None,
            time_horizon="short",
            reasoning="Adversarial test",
            key_factors=["adversarial"],
            risk_assessment={},
            position_size_recommendation=0.1,
            timestamp=datetime.now()
        )

        adversarial_analyses = analyses + [outlier_analysis]

        # Recalculate weights (equal for simplicity)
        adversarial_weights = {
            a.persona_name: PersonaWeight(
                persona_name=a.persona_name,
                base_weight=1.0/len(adversarial_analyses),
                performance_multiplier=1.0,
                confidence_multiplier=1.0,
                expertise_multiplier=1.0,
                final_weight=1.0/len(adversarial_analyses),
                reasoning="equal",
                last_updated=datetime.now()
            ) for a in adversarial_analyses
        }

        adversarial_consensus = self._weighted_average_consensus(adversarial_analyses, adversarial_weights)

        # Measure change
        base_value = self._encode_recommendation(base_recommendation)
        adversarial_value = self._encode_recommendation(adversarial_consensus['recommendation'])

        robustness_metrics['adversarial_robustness'] = 1.0 - abs(adversarial_value - base_value) / 4.0  # Max change is 4

        return robustness_metrics

    def _decompose_risk_factors(self, analyses: List[PersonaAnalysis],
                               market_data: MarketAnalysis) -> Dict[str, float]:
        """Decompose risk into different factors"""
        risk_factors = {}

        # Consensus risk (disagreement)
        recommendations = [a.recommendation for a in analyses]
        unique_recs = set(recommendations)
        risk_factors['consensus_risk'] = len(unique_recs) / len(InvestmentAction)

        # Confidence risk (low confidence)
        confidences = [a.confidence_score for a in analyses]
        risk_factors['confidence_risk'] = 1.0 - (np.mean(confidences) / 100)

        # Market risk (volatility)
        volatility = market_data.technical_indicators.get('volatility_20d', 0.2)
        risk_factors['market_risk'] = min(1.0, volatility / 0.5)  # Normalized

        # Fundamental risk (valuation)
        pe_ratio = market_data.pe_ratio or 15
        risk_factors['valuation_risk'] = min(1.0, max(0, pe_ratio - 10) / 40)  # High PE = higher risk

        return risk_factors

    def _perform_scenario_analysis(self, analyses: List[PersonaAnalysis],
                                 market_data: MarketAnalysis) -> Dict[str, Dict[str, float]]:
        """Perform scenario analysis"""
        scenarios = {
            'bull_market': {'market_multiplier': 1.5, 'volatility_multiplier': 0.8},
            'bear_market': {'market_multiplier': 0.5, 'volatility_multiplier': 1.5},
            'high_volatility': {'market_multiplier': 1.0, 'volatility_multiplier': 2.0},
            'recession': {'market_multiplier': 0.3, 'volatility_multiplier': 2.5}
        }

        scenario_results = {}

        for scenario_name, scenario_params in scenarios.items():
            # Adjust expected returns based on scenario
            adjusted_returns = []

            for analysis in analyses:
                base_return = 0.1  # Default expected return
                if hasattr(analysis, 'expected_return'):
                    base_return = analysis.expected_return

                adjusted_return = base_return * scenario_params['market_multiplier']
                adjusted_returns.append(adjusted_return)

            scenario_results[scenario_name] = {
                'expected_return': np.mean(adjusted_returns),
                'volatility': np.std(adjusted_returns) * scenario_params['volatility_multiplier'],
                'sharpe_ratio': np.mean(adjusted_returns) / (np.std(adjusted_returns) * scenario_params['volatility_multiplier'] + 1e-8)
            }

        return scenario_results

    def _calculate_tail_risk_metrics(self, analyses: List[PersonaAnalysis]) -> Dict[str, float]:
        """Calculate tail risk metrics"""
        # Simplified tail risk based on position sizes and confidence
        position_sizes = [a.position_size_recommendation for a in analyses]
        confidences = [a.confidence_score / 100 for a in analyses]

        # Value at Risk (simplified)
        max_position = max(position_sizes)
        min_confidence = min(confidences)
        var_95 = max_position * (1 - min_confidence) * 1.65  # 95th percentile

        # Expected Shortfall (simplified)
        es_95 = var_95 * 1.3  # Rough approximation

        return {
            'var_95': var_95,
            'expected_shortfall_95': es_95,
            'max_position_risk': max_position,
            'confidence_tail_risk': 1 - min_confidence
        }

    def _assess_consensus_quality(self, analyses: List[PersonaAnalysis],
                                 consensus: Dict[str, Any],
                                 all_results: Dict[str, Dict[str, Any]]) -> float:
        """Assess overall quality of consensus"""
        quality_factors = []

        # Participant diversity
        persona_types = set(a.persona_name for a in analyses)
        diversity_score = len(persona_types) / len(AVAILABLE_PERSONAS)
        quality_factors.append(diversity_score)

        # Confidence consistency
        confidences = [a.confidence_score for a in analyses]
        confidence_consistency = 1 - (np.std(confidences) / 100)
        quality_factors.append(confidence_consistency)

        # Method agreement (if multiple methods used)
        if len(all_results) > 1:
            method_recs = [result['recommendation'] for result in all_results.values() if 'recommendation' in result]
            if method_recs:
                unique_method_recs = set(method_recs)
                method_agreement = 1 - (len(unique_method_recs) - 1) / (len(InvestmentAction) - 1)
                quality_factors.append(method_agreement)

        # Overall consensus confidence
        consensus_confidence = consensus.get('confidence', 0.5)
        quality_factors.append(consensus_confidence)

        return np.mean(quality_factors)

    def _calculate_prediction_intervals(self, analyses: List[PersonaAnalysis],
                                      consensus: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate prediction intervals using bootstrap"""
        if not self.config.enable_uncertainty_quantification:
            return {'recommendation': (0.0, 1.0)}

        # Bootstrap resampling
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        rec_values = [encoding[a.recommendation] for a in analyses]
        bootstrap_results = []

        for _ in range(self.config.bootstrap_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(rec_values, size=len(rec_values), replace=True)
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_results.append(bootstrap_mean)

        # Calculate percentiles
        lower_bound = np.percentile(bootstrap_results, 2.5)
        upper_bound = np.percentile(bootstrap_results, 97.5)

        return {
            'recommendation': (lower_bound, upper_bound),
            'bootstrap_mean': np.mean(bootstrap_results),
            'bootstrap_std': np.std(bootstrap_results)
        }

    def _optimize_execution_strategy(self, consensus: Dict[str, Any],
                                   market_data: MarketAnalysis) -> Dict[str, Any]:
        """Optimize execution strategy based on consensus"""
        recommendation = consensus['recommendation']
        confidence = consensus.get('confidence', 0.5)

        # Basic execution strategy
        strategy = {
            'execution_type': 'gradual',  # gradual, immediate, conditional
            'time_horizon': 'medium',     # short, medium, long
            'order_type': 'limit',        # market, limit, stop_limit
            'split_orders': True,
            'max_market_impact': 0.02     # 2% max impact
        }

        # Adjust based on recommendation strength
        if recommendation in [InvestmentAction.STRONG_BUY, InvestmentAction.STRONG_SELL]:
            if confidence > 0.8:
                strategy['execution_type'] = 'immediate'
                strategy['time_horizon'] = 'short'
            else:
                strategy['execution_type'] = 'gradual'

        # Adjust based on market volatility
        volatility = market_data.technical_indicators.get('volatility_20d', 0.2)
        if volatility > 0.3:
            strategy['split_orders'] = True
            strategy['execution_type'] = 'gradual'
            strategy['max_market_impact'] = 0.01  # More conservative

        return strategy

    def _estimate_market_impact(self, consensus: Dict[str, Any],
                              market_data: MarketAnalysis) -> Dict[str, float]:
        """Estimate market impact of execution"""
        # Simplified market impact model
        volume = market_data.volume
        avg_volume = market_data.technical_indicators.get('volume_sma_20', volume)

        # Assume position size relative to average volume
        relative_volume = 0.05  # 5% of average volume (assumption)

        # Linear market impact model
        temporary_impact = 0.001 * np.sqrt(relative_volume)
        permanent_impact = 0.0005 * relative_volume

        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'total_impact': temporary_impact + permanent_impact
        }

    def _generate_timing_recommendations(self, consensus: Dict[str, Any],
                                       market_data: MarketAnalysis) -> Dict[str, Any]:
        """Generate timing recommendations"""
        recommendation = consensus['recommendation']
        confidence = consensus.get('confidence', 0.5)

        timing = {
            'immediate_action': False,
            'preferred_timing': 'market_open',
            'avoid_times': ['market_close'],
            'optimal_duration': '1_day'
        }

        # High confidence actions
        if confidence > 0.8:
            timing['immediate_action'] = True
            timing['optimal_duration'] = '4_hours'

        # Strong recommendations
        if recommendation in [InvestmentAction.STRONG_BUY, InvestmentAction.STRONG_SELL]:
            timing['immediate_action'] = True

        # Market conditions
        volatility = market_data.technical_indicators.get('volatility_20d', 0.2)
        if volatility > 0.3:
            timing['avoid_times'].extend(['first_30min', 'last_30min'])
            timing['preferred_timing'] = 'mid_day'

        return timing

    # Helper methods for standard consensus fields
    def _calculate_position_size(self, analyses: List[PersonaAnalysis],
                               consensus: Dict[str, Any]) -> float:
        """Calculate recommended position size"""
        position_sizes = [a.position_size_recommendation for a in analyses]
        confidence = consensus.get('confidence', 0.5)

        # Weight by consensus confidence
        base_size = np.mean(position_sizes)
        adjusted_size = base_size * confidence

        return max(0.01, min(0.25, adjusted_size))

    def _calculate_target_price(self, analyses: List[PersonaAnalysis],
                              consensus: Dict[str, Any]) -> Optional[float]:
        """Calculate consensus target price"""
        targets = [a.target_price for a in analyses if a.target_price is not None]

        if not targets:
            return None

        # Weight by confidence
        weights = [a.confidence_score for a in analyses if a.target_price is not None]

        if weights:
            weighted_target = sum(t * w for t, w in zip(targets, weights)) / sum(weights)
            return round(weighted_target, 2)
        else:
            return round(np.mean(targets), 2)

    def _calculate_stop_loss(self, analyses: List[PersonaAnalysis],
                           market_data: MarketAnalysis) -> Optional[float]:
        """Calculate consensus stop loss"""
        stops = [a.stop_loss for a in analyses if a.stop_loss is not None]

        if stops:
            # Use most conservative (highest for long positions)
            return round(max(stops), 2)
        else:
            # Default to 10% below current price
            return round(market_data.current_price * 0.9, 2)

    def _extract_debate_insights(self, debate_rounds: List[Any],
                               analyses: List[PersonaAnalysis]) -> Tuple[List[str], List[str], List[str]]:
        """Extract insights from debate rounds"""
        supporting_args = []
        concerns = []
        dissenting = []

        # Extract from analyses
        for analysis in analyses:
            if analysis.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
                supporting_args.extend(analysis.key_factors[:2])
            elif analysis.recommendation in [InvestmentAction.SELL, InvestmentAction.STRONG_SELL]:
                concerns.extend(analysis.key_factors[:2])

        # Simple debate analysis
        if debate_rounds:
            for round_obj in debate_rounds:
                if hasattr(round_obj, 'messages'):
                    for msg in round_obj.messages:
                        text = msg.message.lower() if hasattr(msg, 'message') else str(msg).lower()

                        if any(word in text for word in ['buy', 'positive', 'opportunity', 'strong']):
                            supporting_args.append(f"Debate: {text[:100]}...")
                        elif any(word in text for word in ['risk', 'concern', 'negative', 'avoid']):
                            concerns.append(f"Debate: {text[:100]}...")

        # Limit results
        return (supporting_args[:5], concerns[:5], dissenting[:3])

    def _assess_market_conditions(self, market_data: MarketAnalysis) -> str:
        """Assess current market conditions"""
        conditions = []

        volatility = market_data.technical_indicators.get('volatility_20d', 0.2)
        if volatility > 0.3:
            conditions.append("High volatility")
        elif volatility < 0.15:
            conditions.append("Low volatility")

        rsi = market_data.technical_indicators.get('rsi', 50)
        if rsi > 70:
            conditions.append("Overbought")
        elif rsi < 30:
            conditions.append("Oversold")

        return "; ".join(conditions) if conditions else "Neutral conditions"

    def _determine_execution_timeline(self, consensus: Dict[str, Any]) -> str:
        """Determine execution timeline"""
        confidence = consensus.get('confidence', 0.5)
        recommendation = consensus['recommendation']

        if confidence > 0.8 and recommendation in [InvestmentAction.STRONG_BUY, InvestmentAction.STRONG_SELL]:
            return "Immediate (within 24 hours)"
        elif confidence > 0.6:
            return "Short-term (within 1 week)"
        else:
            return "Medium-term (within 1 month)"

    def _assess_risk_level(self, analyses: List[PersonaAnalysis],
                         uncertainty_metrics: Dict[str, float]) -> str:
        """Assess overall risk level"""
        avg_uncertainty = np.mean(list(uncertainty_metrics.values()))

        if avg_uncertainty > 0.7:
            return "High Risk"
        elif avg_uncertainty > 0.4:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def _calculate_expected_return(self, analyses: List[PersonaAnalysis],
                                 consensus: Dict[str, Any]) -> float:
        """Calculate expected return"""
        # Simple model based on recommendation strength and confidence
        recommendation = consensus['recommendation']
        confidence = consensus.get('confidence', 0.5)

        base_returns = {
            InvestmentAction.STRONG_SELL: -0.15,
            InvestmentAction.SELL: -0.05,
            InvestmentAction.HOLD: 0.0,
            InvestmentAction.BUY: 0.08,
            InvestmentAction.STRONG_BUY: 0.15
        }

        base_return = base_returns[recommendation]
        adjusted_return = base_return * confidence

        return round(adjusted_return, 3)

    def _generate_debate_summary(self, debate_rounds: List[Any]) -> str:
        """Generate summary of debate"""
        if not debate_rounds:
            return "No debate conducted"

        total_messages = sum(len(getattr(r, 'messages', [])) for r in debate_rounds)
        return f"Debate conducted over {len(debate_rounds)} rounds with {total_messages} exchanges"

    def _get_model_performance_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        return {
            'consensus_accuracy': 0.75,  # Placeholder
            'prediction_calibration': 0.68,  # Placeholder
            'model_confidence': 0.72  # Placeholder
        }

    def _encode_recommendation(self, recommendation: InvestmentAction) -> float:
        """Encode recommendation as numerical value"""
        encoding = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }
        return encoding[recommendation]

    def _create_fallback_consensus(self, symbol: str, analyses: List[PersonaAnalysis],
                                 market_data: MarketAnalysis) -> EnhancedConsensusResult:
        """Create fallback consensus when main process fails"""
        # Simple majority vote
        recommendations = [a.recommendation for a in analyses]
        most_common = max(set(recommendations), key=recommendations.count)

        consensus_score = recommendations.count(most_common) / len(recommendations) * 100

        return EnhancedConsensusResult(
            symbol=symbol,
            final_recommendation=most_common,
            consensus_score=consensus_score,
            average_confidence=np.mean([a.confidence_score for a in analyses]),
            recommended_position_size=0.05,
            target_price=None,
            stop_loss=None,

            # Enhanced analytics (simplified)
            participant_weights={},
            consensus_evolution=[],
            uncertainty_quantification={'overall': 0.5},
            sensitivity_analysis={},
            robustness_metrics={'robustness': 0.5},
            bayesian_posterior=None,
            game_theory_equilibrium=None,

            # Risk analysis (simplified)
            risk_decomposition={'total_risk': 0.5},
            scenario_analysis={},
            tail_risk_metrics={'var_95': 0.1},

            # Meta-analytics (simplified)
            consensus_quality_score=0.5,
            prediction_intervals={'recommendation': (0, 1)},
            model_performance_metrics={},

            # Execution guidance (simplified)
            optimal_execution_strategy={'type': 'gradual'},
            market_impact_estimates={'total_impact': 0.01},
            timing_recommendations={'timing': 'flexible'},

            # Standard fields
            key_supporting_arguments=["Majority vote consensus"],
            key_concerns=["Limited analysis"],
            dissenting_opinions=[],
            market_conditions_assessment="Basic analysis",
            execution_timeline="Standard timing",
            risk_level="Moderate",
            expected_return=0.05,
            debate_summary="Fallback consensus used",
            timestamp=datetime.now()
        )


# Convenience functions
def quick_consensus(symbol: str, analyses: List[PersonaAnalysis],
                   market_data: MarketAnalysis) -> EnhancedConsensusResult:
    """Quick consensus with default settings"""
    config = ConsensusConfiguration(
        method=ConsensusMethod.WEIGHTED_AVERAGE,
        weighting_scheme=WeightingScheme.CONFIDENCE_BASED
    )

    engine = EnhancedConsensusEngine(config)
    return engine.build_enhanced_consensus(symbol, analyses, [], market_data)


def comprehensive_consensus(symbol: str, analyses: List[PersonaAnalysis],
                          debate_rounds: List[Any], market_data: MarketAnalysis,
                          **config_kwargs) -> EnhancedConsensusResult:
    """Comprehensive consensus with all features"""
    config = ConsensusConfiguration(
        method=ConsensusMethod.ENSEMBLE,
        weighting_scheme=WeightingScheme.DYNAMIC_BAYESIAN,
        enable_uncertainty_quantification=True,
        enable_adversarial_testing=True,
        **config_kwargs
    )

    engine = EnhancedConsensusEngine(config)
    return engine.build_enhanced_consensus(symbol, analyses, debate_rounds, market_data)


if __name__ == "__main__":
    # Example usage would go here
    print("Enhanced Consensus Engine loaded successfully")