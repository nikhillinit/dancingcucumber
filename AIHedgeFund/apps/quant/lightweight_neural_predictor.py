"""
Lightweight Neural Predictor with Batch Processing
==================================================
CPU-optimized neural networks for price prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from joblib import Parallel, delayed
import multiprocessing as mp
import asyncio
from concurrent.futures import ProcessPoolExecutor
import pickle
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Enhanced prediction result with confidence intervals"""
    symbol: str
    predicted_price: float
    confidence_interval: Tuple[float, float]
    direction: str  # up, down, neutral
    confidence_score: float
    feature_importance: Dict[str, float]
    model_type: str
    prediction_horizon: int
    timestamp: datetime
    processing_time: float
    metadata: Dict[str, Any]


class LightweightNeuralPredictor:
    """
    CPU-optimized neural predictor with ensemble methods
    Uses sklearn's MLPRegressor and gradient boosting for efficiency
    """

    def __init__(
        self,
        n_models: int = 5,
        batch_size: int = 32,
        n_jobs: int = -1
    ):
        self.n_models = n_models
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()

        # Initialize ensemble of lightweight models
        self.models = self._initialize_models()
        self.scaler = RobustScaler()
        self.feature_names = None
        self.is_fitted = False

        # Cache for predictions
        self.prediction_cache = {}

        logger.info(f"Initialized LightweightNeuralPredictor with {n_models} models")

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize diverse set of lightweight models"""
        models = {
            'mlp_small': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                batch_size=self.batch_size,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            ),
            'mlp_deep': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='tanh',
                solver='lbfgs',
                max_iter=500,
                early_stopping=True,
                random_state=43
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=self.n_jobs,
                tree_method='hist',
                random_state=44
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                thread_count=self.n_jobs,
                verbose=False,
                random_state=45
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                n_jobs=self.n_jobs,
                verbose=-1,
                random_state=46
            )
        }

        # Create voting ensemble
        models['ensemble'] = VotingRegressor(
            estimators=[(name, model) for name, model in models.items() if name != 'ensemble'],
            n_jobs=self.n_jobs
        )

        return models

    def prepare_features(
        self,
        data: pd.DataFrame,
        lookback: int = 20
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for neural network
        Optimized for batch processing
        """
        features = []
        feature_names = []

        # Price features
        for lag in range(1, min(lookback + 1, len(data))):
            features.append(data['close'].shift(lag).iloc[-1])
            feature_names.append(f'price_lag_{lag}')

        # Returns
        for period in [1, 5, 10, 20]:
            if len(data) > period:
                returns = data['close'].pct_change(period).iloc[-1]
                features.append(returns)
                feature_names.append(f'return_{period}d')

        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(data) >= period:
                ma = data['close'].rolling(period).mean().iloc[-1]
                features.append(data['close'].iloc[-1] / ma - 1)
                feature_names.append(f'price_to_ma_{period}')

        # Volatility
        for period in [5, 10, 20]:
            if len(data) > period:
                vol = data['close'].pct_change().rolling(period).std().iloc[-1]
                features.append(vol)
                feature_names.append(f'volatility_{period}d')

        # Volume features
        if 'volume' in data.columns:
            for period in [5, 10, 20]:
                if len(data) >= period:
                    vol_ratio = data['volume'].iloc[-1] / data['volume'].rolling(period).mean().iloc[-1]
                    features.append(vol_ratio)
                    feature_names.append(f'volume_ratio_{period}')

        # Technical indicators (fast computation)
        if len(data) >= 14:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1] / 100)
            feature_names.append('rsi_14')

        # Time features
        if isinstance(data.index, pd.DatetimeIndex):
            features.append(data.index[-1].dayofweek / 6)
            features.append(data.index[-1].day / 31)
            features.append(data.index[-1].month / 12)
            feature_names.extend(['day_of_week', 'day_of_month', 'month'])

        return np.array(features).reshape(1, -1), feature_names

    async def train_parallel(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.2
    ):
        """Train all models in parallel"""
        import time
        start_time = time.time()

        # Split data
        split_idx = int(len(X_train) * (1 - validation_split))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train models in parallel
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=self.n_jobs)

        training_tasks = []
        for name, model in self.models.items():
            if name != 'ensemble':  # Train ensemble after base models
                task = loop.run_in_executor(
                    executor,
                    self._train_single_model,
                    name, model, X_train_scaled, y_train
                )
                training_tasks.append((name, task))

        # Wait for all models to train
        trained_models = {}
        for name, task in training_tasks:
            trained_model = await task
            trained_models[name] = trained_model
            logger.info(f"Trained {name} model")

        # Update models
        self.models.update(trained_models)

        # Train ensemble
        self.models['ensemble'].fit(X_train_scaled, y_train)

        self.is_fitted = True
        training_time = time.time() - start_time

        # Validate models
        validation_scores = self._validate_models(X_val_scaled, y_val)

        executor.shutdown()

        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Validation scores: {validation_scores}")

        return validation_scores

    def _train_single_model(
        self,
        name: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """Train a single model (for parallel processing)"""
        model.fit(X, y)
        return model

    def _validate_models(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Validate all models"""
        scores = {}

        for name, model in self.models.items():
            predictions = model.predict(X_val)
            mse = np.mean((predictions - y_val) ** 2)
            mae = np.mean(np.abs(predictions - y_val))
            scores[name] = {'mse': mse, 'mae': mae}

        return scores

    def predict_batch(
        self,
        X: np.ndarray,
        return_confidence: bool = True
    ) -> List[PredictionResult]:
        """Predict for batch of samples"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        import time
        start_time = time.time()

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            preds = model.predict(X_scaled)
            predictions[name] = preds

        # Calculate ensemble prediction and confidence
        results = []
        for i in range(X.shape[0]):
            sample_preds = [predictions[name][i] for name in predictions]

            # Ensemble prediction
            ensemble_pred = predictions['ensemble'][i]

            # Confidence based on agreement
            std_dev = np.std(sample_preds)
            confidence = 1 / (1 + std_dev)

            # Confidence interval
            lower_ci = np.percentile(sample_preds, 5)
            upper_ci = np.percentile(sample_preds, 95)

            # Direction
            if ensemble_pred > 0.01:
                direction = 'up'
            elif ensemble_pred < -0.01:
                direction = 'down'
            else:
                direction = 'neutral'

            # Feature importance (simplified)
            feature_importance = self._get_feature_importance()

            result = PredictionResult(
                symbol='UNKNOWN',  # Set by caller
                predicted_price=ensemble_pred,
                confidence_interval=(lower_ci, upper_ci),
                direction=direction,
                confidence_score=confidence,
                feature_importance=feature_importance,
                model_type='ensemble',
                prediction_horizon=1,
                timestamp=datetime.now(),
                processing_time=time.time() - start_time,
                metadata={
                    'model_predictions': {name: float(pred) for name, pred in zip(predictions.keys(), sample_preds)},
                    'std_dev': std_dev
                }
            )
            results.append(result)

        return results

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from tree models"""
        importance = {}

        if self.feature_names:
            # Get importance from XGBoost
            if 'xgboost' in self.models and hasattr(self.models['xgboost'], 'feature_importances_'):
                xgb_importance = self.models['xgboost'].feature_importances_
                for i, name in enumerate(self.feature_names[:len(xgb_importance)]):
                    importance[name] = float(xgb_importance[i])

            # Get importance from LightGBM
            if 'lightgbm' in self.models and hasattr(self.models['lightgbm'], 'feature_importances_'):
                lgb_importance = self.models['lightgbm'].feature_importances_
                for i, name in enumerate(self.feature_names[:len(lgb_importance)]):
                    if name in importance:
                        importance[name] = (importance[name] + float(lgb_importance[i])) / 2
                    else:
                        importance[name] = float(lgb_importance[i])

        # Normalize importance
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}

        return importance

    async def predict_multi_horizon(
        self,
        data: pd.DataFrame,
        horizons: List[int] = [1, 5, 10, 20]
    ) -> Dict[int, PredictionResult]:
        """Predict for multiple time horizons"""
        results = {}

        # Prepare base features
        X_base, feature_names = self.prepare_features(data)
        self.feature_names = feature_names

        # Create tasks for different horizons
        tasks = []
        for horizon in horizons:
            # Modify features for different horizons
            X_horizon = self._adjust_features_for_horizon(X_base, horizon)
            task = self._predict_horizon_async(X_horizon, horizon)
            tasks.append((horizon, task))

        # Execute predictions in parallel
        for horizon, task in tasks:
            result = await task
            result.prediction_horizon = horizon
            results[horizon] = result

        return results

    async def _predict_horizon_async(
        self,
        X: np.ndarray,
        horizon: int
    ) -> PredictionResult:
        """Async prediction for a specific horizon"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_batch, X, True)

    def _adjust_features_for_horizon(
        self,
        X: np.ndarray,
        horizon: int
    ) -> np.ndarray:
        """Adjust features based on prediction horizon"""
        # Simple adjustment - scale features by horizon
        X_adjusted = X.copy()

        # Adjust time-dependent features
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                if 'volatility' in name:
                    X_adjusted[0, i] *= np.sqrt(horizon)
                elif 'return' in name:
                    X_adjusted[0, i] *= horizon

        return X_adjusted

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {name: [] for name in self.models.keys()}

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train and evaluate each model
            for name, model in self.models.items():
                if name == 'ensemble':
                    continue  # Skip ensemble for CV

                # Clone and train model
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train_scaled, y_train)

                # Evaluate
                predictions = model_clone.predict(X_val_scaled)
                mse = np.mean((predictions - y_val) ** 2)
                cv_scores[name].append(mse)

        return cv_scores

    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.models = saved_data['models']
            self.scaler = saved_data['scaler']
            self.feature_names = saved_data['feature_names']
            self.is_fitted = saved_data['is_fitted']

        logger.info(f"Model loaded from {filepath}")


class BatchPredictor:
    """
    Batch prediction processor for multiple symbols
    """

    def __init__(self, predictor: LightweightNeuralPredictor):
        self.predictor = predictor
        self.executor = ProcessPoolExecutor(max_workers=mp.cpu_count())

    async def predict_symbols(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        horizons: List[int] = [1, 5, 20]
    ) -> Dict[str, Dict[int, PredictionResult]]:
        """Predict for multiple symbols in parallel"""
        tasks = []

        for symbol, data in symbol_data.items():
            task = self._predict_symbol(symbol, data, horizons)
            tasks.append((symbol, task))

        results = {}
        for symbol, task in tasks:
            symbol_results = await task
            # Update symbol in results
            for horizon, result in symbol_results.items():
                result.symbol = symbol
            results[symbol] = symbol_results

        return results

    async def _predict_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        horizons: List[int]
    ) -> Dict[int, PredictionResult]:
        """Predict for a single symbol"""
        return await self.predictor.predict_multi_horizon(data, horizons)


# Numba-optimized functions for speed
from numba import jit

@jit(nopython=True)
def calculate_returns(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Fast return calculation"""
    n = len(prices)
    n_periods = len(periods)
    returns = np.zeros(n_periods)

    for i, period in enumerate(periods):
        if n > period:
            returns[i] = (prices[-1] / prices[-period-1]) - 1

    return returns

@jit(nopython=True)
def calculate_volatility(returns: np.ndarray, window: int) -> float:
    """Fast volatility calculation"""
    if len(returns) < window:
        return 0.0

    recent_returns = returns[-window:]
    mean = np.mean(recent_returns)
    variance = np.mean((recent_returns - mean) ** 2)
    return np.sqrt(variance * 252)  # Annualized


# Example usage
async def main():
    """Example usage of lightweight neural predictor"""
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.random(len(dates)) * 100 + 100,
        'high': np.random.random(len(dates)) * 100 + 105,
        'low': np.random.random(len(dates)) * 100 + 95,
        'close': np.cumsum(np.random.randn(len(dates)) * 2) + 100,
        'volume': np.random.random(len(dates)) * 1000000
    }, index=dates)

    # Initialize predictor
    predictor = LightweightNeuralPredictor(n_models=5, batch_size=32)

    # Prepare training data
    X_list = []
    y_list = []

    for i in range(50, len(data) - 1):
        X, feature_names = predictor.prepare_features(data.iloc[:i])
        y = data['close'].iloc[i+1] / data['close'].iloc[i] - 1  # Next day return
        X_list.append(X[0])
        y_list.append(y)

    predictor.feature_names = feature_names

    X_train = np.array(X_list)
    y_train = np.array(y_list)

    # Train models
    print("Training models...")
    validation_scores = await predictor.train_parallel(X_train, y_train)

    # Make predictions
    print("\nMaking predictions...")
    latest_data = data.iloc[-50:]
    predictions = await predictor.predict_multi_horizon(
        latest_data,
        horizons=[1, 5, 10, 20]
    )

    for horizon, result in predictions.items():
        print(f"\nHorizon {horizon} days:")
        print(f"  Predicted: {result.predicted_price:.4f}")
        print(f"  Confidence: {result.confidence_score:.2%}")
        print(f"  Direction: {result.direction}")
        print(f"  CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")

    # Batch prediction for multiple symbols
    print("\nBatch prediction for multiple symbols...")
    batch_predictor = BatchPredictor(predictor)

    symbol_data = {
        'AAPL': data,
        'GOOGL': data * 1.5,
        'MSFT': data * 0.8
    }

    batch_results = await batch_predictor.predict_symbols(symbol_data)

    for symbol, horizons in batch_results.items():
        print(f"\n{symbol}:")
        for horizon, result in horizons.items():
            print(f"  {horizon}d: {result.direction} ({result.confidence_score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())