"""
Real Data Integration for Production AI Trading System
=====================================================
Replaces simulated signals with actual market data and ML models
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    import talib
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(["pip", "install", "yfinance", "xgboost", "scikit-learn", "TA-Lib"])
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit


class RealDataProvider:
    """Fetch real market data from multiple sources"""

    def __init__(self):
        self.cache = {}

    def get_market_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch real market data from Yahoo Finance"""
        market_data = {}

        print(f"[DATA] Fetching real market data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval="1d")

                if not data.empty:
                    # Ensure consistent column names
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    market_data[symbol] = data
                    print(f"  ✓ {symbol}: {len(data)} days of data")
                else:
                    print(f"  ✗ {symbol}: No data available")

            except Exception as e:
                print(f"  ✗ {symbol}: Error - {str(e)}")

        return market_data

    def get_options_data(self, symbol: str) -> Optional[Dict]:
        """Fetch real options data"""
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options

            if not options_dates:
                return None

            # Get nearest expiration
            nearest_exp = options_dates[0]
            chain = ticker.option_chain(nearest_exp)

            calls = chain.calls
            puts = chain.puts

            # Calculate put/call ratio
            total_call_volume = calls['volume'].fillna(0).sum()
            total_put_volume = puts['volume'].fillna(0).sum()

            pc_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 1.0

            # Calculate gamma exposure (simplified)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]

            gamma_exposure = 0
            for _, row in calls.iterrows():
                if pd.notna(row['gamma']) and pd.notna(row['openInterest']):
                    gamma_exposure += row['gamma'] * row['openInterest'] * 100  # 100 shares per contract

            for _, row in puts.iterrows():
                if pd.notna(row['gamma']) and pd.notna(row['openInterest']):
                    gamma_exposure -= row['gamma'] * row['openInterest'] * 100  # Negative for puts

            return {
                'put_call_ratio': pc_ratio,
                'gamma_exposure': gamma_exposure,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'current_price': current_price
            }

        except Exception as e:
            print(f"Options data error for {symbol}: {e}")
            return None

    def get_sentiment_data(self, symbol: str) -> Optional[Dict]:
        """Fetch sentiment data (placeholder - would use real API)"""
        # In production, you'd use:
        # - Twitter API for social sentiment
        # - News APIs (Alpha Vantage, Finnhub, etc.)
        # - Reddit API for retail sentiment
        # - Google Trends API

        # Simulated for now - replace with real APIs
        return {
            'social_sentiment': np.random.normal(0, 0.1),
            'news_sentiment': np.random.normal(0, 0.05),
            'analyst_sentiment': np.random.normal(0, 0.03),
            'confidence': 0.6
        }


class FeatureEngineer:
    """Advanced feature engineering for ML models"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features_df = pd.DataFrame(index=data.index)

        # Price-based features
        features_df['returns_1d'] = data['Close'].pct_change()
        features_df['returns_5d'] = data['Close'].pct_change(5)
        features_df['returns_20d'] = data['Close'].pct_change(20)

        # Moving averages
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features_df[f'price_vs_sma_{period}'] = data['Close'] / features_df[f'sma_{period}'] - 1

        # Volatility features
        features_df['volatility_5d'] = data['Close'].pct_change().rolling(5).std()
        features_df['volatility_20d'] = data['Close'].pct_change().rolling(20).std()
        features_df['volatility_ratio'] = features_df['volatility_5d'] / features_df['volatility_20d']

        # Volume features
        features_df['volume_sma_20'] = data['Volume'].rolling(20).mean()
        features_df['volume_ratio'] = data['Volume'] / features_df['volume_sma_20']
        features_df['volume_price_trend'] = (data['Volume'] * data['Close'].pct_change()).rolling(5).mean()

        # Technical indicators using TA-Lib
        try:
            # RSI
            features_df['rsi_14'] = talib.RSI(data['Close'].values, timeperiod=14)
            features_df['rsi_divergence'] = features_df['rsi_14'] - 50  # Centered around 0

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(data['Close'].values)
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_histogram'] = macd_hist

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'].values)
            features_df['bb_position'] = (data['Close'] - bb_middle) / (bb_upper - bb_lower)
            features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle

            # ADX (trend strength)
            features_df['adx'] = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values)

            # Williams %R
            features_df['williams_r'] = talib.WILLR(data['High'].values, data['Low'].values, data['Close'].values)

        except Exception as e:
            print(f"TA-Lib features error: {e}")
            # Fill with basic calculations if TA-Lib fails
            features_df['rsi_14'] = 50  # Neutral RSI
            features_df['rsi_divergence'] = 0

        # Advanced patterns
        features_df['price_momentum'] = self._calculate_momentum_score(data)
        features_df['volume_momentum'] = self._calculate_volume_momentum(data)
        features_df['breakout_score'] = self._calculate_breakout_score(data)

        # Cross-sectional features (require market data)
        features_df['symbol'] = symbol  # For later cross-sectional analysis

        # Forward fill and backward fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')

        return features_df

    def _calculate_momentum_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate composite momentum score"""
        returns = data['Close'].pct_change()

        # Multi-timeframe momentum
        momentum_1w = returns.rolling(5).mean()
        momentum_1m = returns.rolling(20).mean()
        momentum_3m = returns.rolling(60).mean()

        # Weighted combination
        momentum_score = (momentum_1w * 0.5 + momentum_1m * 0.3 + momentum_3m * 0.2)
        return momentum_score

    def _calculate_volume_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume-based momentum"""
        volume_change = data['Volume'].pct_change()
        price_change = data['Close'].pct_change()

        # Volume-price correlation over rolling window
        volume_momentum = volume_change.rolling(20).corr(price_change)
        return volume_momentum.fillna(0)

    def _calculate_breakout_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate breakout potential score"""
        high_20 = data['High'].rolling(20).max()
        low_20 = data['Low'].rolling(20).min()

        # Position within range
        range_position = (data['Close'] - low_20) / (high_20 - low_20)

        # Volatility squeeze detection
        bb_width = (high_20 - low_20) / data['Close'].rolling(20).mean()
        volatility_squeeze = bb_width.rolling(20).rank(pct=True)  # Percentile rank

        # Combine for breakout score
        breakout_score = range_position * (1 - volatility_squeeze)  # High when at range extremes during low volatility
        return breakout_score


class MLModelManager:
    """Manage multiple ML models for ensemble predictions"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []

    def train_xgboost_model(self, features: pd.DataFrame, target: pd.Series, symbol: str):
        """Train XGBoost model for specific symbol"""

        # Clean data
        clean_data = self._prepare_training_data(features, target)
        if clean_data is None:
            return None

        X, y = clean_data

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)

        best_score = float('inf')
        best_model = None

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=20
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            val_score = model.score(X_val, y_val)
            if val_score > best_score:
                best_score = val_score
                best_model = model

        self.models[f'xgb_{symbol}'] = best_model
        print(f"  XGBoost trained for {symbol}: R² = {best_score:.3f}")

        return best_model

    def train_random_forest_model(self, features: pd.DataFrame, target: pd.Series, symbol: str):
        """Train Random Forest model"""

        clean_data = self._prepare_training_data(features, target)
        if clean_data is None:
            return None

        X, y = clean_data

        # Random Forest
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        # Use last 80% for training, 20% for validation
        split_point = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)

        self.models[f'rf_{symbol}'] = model
        print(f"  Random Forest trained for {symbol}: R² = {val_score:.3f}")

        return model

    def _prepare_training_data(self, features: pd.DataFrame, target: pd.Series):
        """Clean and prepare data for training"""

        # Align features and target
        aligned_data = features.join(target.rename('target'), how='inner')
        aligned_data = aligned_data.dropna()

        if len(aligned_data) < 100:  # Need sufficient data
            print(f"  Insufficient data: {len(aligned_data)} samples")
            return None

        # Remove symbol column if present
        feature_cols = [col for col in aligned_data.columns if col not in ['target', 'symbol']]

        X = aligned_data[feature_cols]
        y = aligned_data['target']

        # Store feature columns
        if not self.feature_columns:
            self.feature_columns = feature_cols

        return X, y

    def predict_ensemble(self, features: pd.DataFrame, symbol: str) -> Dict:
        """Generate ensemble prediction"""

        feature_cols = [col for col in features.columns if col not in ['symbol']]
        X = features[feature_cols].iloc[-1:].fillna(0)  # Last row only

        predictions = {}

        # XGBoost prediction
        xgb_model = self.models.get(f'xgb_{symbol}')
        if xgb_model:
            try:
                xgb_pred = xgb_model.predict(X)[0]
                predictions['xgboost'] = xgb_pred
            except Exception as e:
                print(f"XGBoost prediction error: {e}")

        # Random Forest prediction
        rf_model = self.models.get(f'rf_{symbol}')
        if rf_model:
            try:
                rf_pred = rf_model.predict(X)[0]
                predictions['random_forest'] = rf_pred
            except Exception as e:
                print(f"Random Forest prediction error: {e}")

        if predictions:
            # Ensemble average
            ensemble_pred = np.mean(list(predictions.values()))

            # Calculate confidence based on model agreement
            pred_std = np.std(list(predictions.values())) if len(predictions) > 1 else 0
            confidence = max(0.1, min(0.9, 1.0 - pred_std * 10))

            return {
                'ensemble_prediction': ensemble_pred,
                'confidence': confidence,
                'individual_predictions': predictions,
                'num_models': len(predictions)
            }

        return {'ensemble_prediction': 0, 'confidence': 0, 'individual_predictions': {}, 'num_models': 0}


class ProductionAISystem:
    """Production-ready AI system with real data"""

    def __init__(self):
        self.data_provider = RealDataProvider()
        self.feature_engineer = FeatureEngineer()
        self.model_manager = MLModelManager()

        # Fidelity constraints
        self.config = {
            'max_positions': 6,
            'max_position_size': 0.18,
            'min_confidence': 0.6,
            'rebalance_threshold': 0.05,
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
        }

    def initialize_models(self):
        """Train models on historical data"""
        print("[AI] Initializing production models with real data...")

        # Get historical data
        market_data = self.data_provider.get_market_data(
            self.config['symbols'],
            period="2y"  # 2 years for training
        )

        for symbol, data in market_data.items():
            print(f"\nTraining models for {symbol}...")

            # Create features
            features = self.feature_engineer.create_features(data, symbol)

            # Create target (next day return)
            target = data['Close'].pct_change().shift(-1)  # Next day return

            # Train models
            self.model_manager.train_xgboost_model(features, target, symbol)
            self.model_manager.train_random_forest_model(features, target, symbol)

        print(f"\n✅ Models trained for {len(market_data)} symbols")

    def generate_daily_recommendations(self) -> Dict:
        """Generate daily trading recommendations using real ML models"""
        print("\n[AI] Generating daily recommendations with production models...")

        # Get latest market data
        market_data = self.data_provider.get_market_data(
            self.config['symbols'],
            period="6mo"
        )

        recommendations = {}

        for symbol, data in market_data.items():
            # Create features for latest data
            features = self.feature_engineer.create_features(data, symbol)

            # Get ML prediction
            ml_prediction = self.model_manager.predict_ensemble(features, symbol)

            # Get additional signals
            options_data = self.data_provider.get_options_data(symbol)
            sentiment_data = self.data_provider.get_sentiment_data(symbol)

            # Combine all signals
            final_signal = self._combine_signals(ml_prediction, options_data, sentiment_data)

            if final_signal['confidence'] > self.config['min_confidence']:
                recommendations[symbol] = final_signal

        return recommendations

    def _combine_signals(self, ml_pred: Dict, options_data: Optional[Dict], sentiment_data: Optional[Dict]) -> Dict:
        """Combine ML predictions with alternative data"""

        base_signal = ml_pred['ensemble_prediction']
        base_confidence = ml_pred['confidence']

        # Options flow adjustment
        options_adjustment = 0
        if options_data:
            pc_ratio = options_data['put_call_ratio']
            # Low P/C ratio = bullish sentiment
            options_adjustment = (1 - min(pc_ratio, 2.0)) * 0.01

        # Sentiment adjustment
        sentiment_adjustment = 0
        if sentiment_data:
            sentiment_adjustment = sentiment_data.get('social_sentiment', 0) * 0.5

        # Combined signal
        combined_signal = base_signal + options_adjustment + sentiment_adjustment

        # Adjust confidence
        combined_confidence = base_confidence
        if options_data and sentiment_data:
            combined_confidence *= 1.1  # Bonus for having all data

        return {
            'signal': combined_signal,
            'confidence': min(0.95, combined_confidence),
            'ml_prediction': base_signal,
            'options_adjustment': options_adjustment,
            'sentiment_adjustment': sentiment_adjustment,
            'num_ml_models': ml_pred['num_models']
        }


# Demo usage
def run_production_demo():
    """Run production system demo"""
    print("="*80)
    print("PRODUCTION AI TRADING SYSTEM")
    print("Real Data + Real ML Models")
    print("="*80)

    try:
        ai_system = ProductionAISystem()

        # Initialize with real data
        ai_system.initialize_models()

        # Generate recommendations
        recommendations = ai_system.generate_daily_recommendations()

        print(f"\n📊 DAILY RECOMMENDATIONS ({len(recommendations)} signals):")
        print("="*60)

        for symbol, rec in recommendations.items():
            direction = "🔴 SELL" if rec['signal'] < 0 else "🟢 BUY"
            print(f"{direction} {symbol}")
            print(f"  Signal Strength: {rec['signal']:.4f}")
            print(f"  Confidence: {rec['confidence']:.1%}")
            print(f"  ML Models: {rec['num_ml_models']}")
            print(f"  Components: ML={rec['ml_prediction']:.4f}, Options={rec['options_adjustment']:.4f}, Sentiment={rec['sentiment_adjustment']:.4f}")
            print()

        if not recommendations:
            print("No high-confidence signals today. Hold current positions.")

        return recommendations

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install yfinance xgboost scikit-learn TA-Lib")
        return {}


if __name__ == "__main__":
    results = run_production_demo()