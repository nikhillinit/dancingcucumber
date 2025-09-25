"""
Regime-Specific Training System
==============================
Train separate models for different market conditions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RegimeSpecificTrainer:
    def __init__(self):
        self.universe = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM']
        self.regimes = {
            'bull_market': {'vix_max': 20, 'spy_trend': 'up'},
            'bear_market': {'vix_min': 25, 'spy_trend': 'down'},
            'high_volatility': {'vix_min': 30},
            'low_volatility': {'vix_max': 15},
            'crisis': {'vix_min': 40}
        }

    def download_market_data(self, years_back=10):
        """Download comprehensive market data"""
        print(f"📥 Downloading {years_back} years of market data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)

        # Download stock data
        stock_data = {}
        for symbol in self.universe:
            try:
                data = yf.download(symbol, start=start_date, end=end_date)
                if not data.empty:
                    stock_data[symbol] = data
                    print(f"✅ {symbol}: {len(data)} days")
            except Exception as e:
                print(f"❌ {symbol}: {e}")

        # Download market indicators
        indicators = {}
        try:
            indicators['VIX'] = yf.download('^VIX', start=start_date, end=end_date)['Close']
            indicators['SPY'] = yf.download('SPY', start=start_date, end=end_date)['Close']
            print(f"✅ Market indicators downloaded")
        except Exception as e:
            print(f"❌ Market indicators: {e}")

        return stock_data, indicators

    def classify_market_regimes(self, indicators):
        """Classify each day into market regimes"""
        print("🏷️  Classifying market regimes...")

        if 'VIX' not in indicators or 'SPY' not in indicators:
            print("❌ Missing market indicators for regime classification")
            return {}

        regimes = pd.DataFrame(index=indicators['VIX'].index)
        regimes['vix'] = indicators['VIX']
        regimes['spy'] = indicators['SPY']
        regimes['spy_trend'] = regimes['spy'].pct_change(20)  # 20-day trend

        # Classify regimes
        regimes['regime'] = 'normal'

        # Crisis periods (VIX > 40)
        regimes.loc[regimes['vix'] > 40, 'regime'] = 'crisis'

        # High volatility (VIX > 30 but not crisis)
        regimes.loc[(regimes['vix'] > 30) & (regimes['regime'] == 'normal'), 'regime'] = 'high_volatility'

        # Low volatility (VIX < 15)
        regimes.loc[regimes['vix'] < 15, 'regime'] = 'low_volatility'

        # Bull market (VIX < 20 and positive trend)
        regimes.loc[(regimes['vix'] < 20) & (regimes['spy_trend'] > 0.02) &
                   (regimes['regime'] == 'normal'), 'regime'] = 'bull_market'

        # Bear market (VIX > 25 and negative trend)
        regimes.loc[(regimes['vix'] > 25) & (regimes['spy_trend'] < -0.02) &
                   (regimes['regime'] == 'normal'), 'regime'] = 'bear_market'

        regime_counts = regimes['regime'].value_counts()
        print("\n📊 Regime Distribution:")
        for regime, count in regime_counts.items():
            percentage = count / len(regimes) * 100
            print(f"{regime:15}: {count:4d} days ({percentage:5.1f}%)")

        return regimes

    def train_regime_models(self, stock_data, regime_data):
        """Train separate models for each market regime"""
        print("\n🤖 Training regime-specific models...")

        regime_models = {}

        for regime_type in regime_data['regime'].unique():
            if regime_type == 'normal':
                continue  # Skip generic normal periods

            print(f"\n🎯 Training {regime_type} model...")

            # Get dates for this regime
            regime_dates = regime_data[regime_data['regime'] == regime_type].index

            # Create training dataset for this regime
            regime_features = []
            regime_targets = []

            for symbol in stock_data:
                stock = stock_data[symbol]

                for date in regime_dates:
                    if date in stock.index:
                        try:
                            # Get features (current day)
                            idx = stock.index.get_loc(date)
                            if idx < 20 or idx >= len(stock) - 5:  # Need lookback and forward data
                                continue

                            features = self.extract_features(stock, idx)
                            target = self.calculate_target(stock, idx)

                            regime_features.append(features)
                            regime_targets.append(target)

                        except Exception as e:
                            continue

            if len(regime_features) < 50:  # Need minimum samples
                print(f"⚠️  Insufficient data for {regime_type} ({len(regime_features)} samples)")
                continue

            # Train simple model for this regime
            model = self.train_simple_model(regime_features, regime_targets)
            regime_models[regime_type] = {
                'model': model,
                'samples': len(regime_features),
                'accuracy': self.evaluate_model(model, regime_features, regime_targets)
            }

            print(f"✅ {regime_type} model: {len(regime_features)} samples, "
                  f"{regime_models[regime_type]['accuracy']:.1%} accuracy")

        return regime_models

    def extract_features(self, stock_data, idx):
        """Extract features for a specific date"""
        # Get last 20 days of data
        window = stock_data.iloc[idx-20:idx]

        features = []

        # Price features
        returns = window['Close'].pct_change().dropna()
        features.extend([
            returns.mean(),           # Average return
            returns.std(),           # Volatility
            window['Close'].iloc[-1] / window['Close'].iloc[0] - 1,  # 20-day return
            window['Close'].iloc[-1] / window['Close'].iloc[-5] - 1,  # 5-day return
        ])

        # Volume features
        vol_ratio = window['Volume'].iloc[-1] / window['Volume'].mean()
        features.append(vol_ratio)

        # Technical indicators
        rsi = self.calculate_simple_rsi(window['Close'])
        features.append(rsi)

        return features

    def calculate_target(self, stock_data, idx):
        """Calculate target (future return)"""
        # Predict 5-day forward return
        current_price = stock_data['Close'].iloc[idx]
        future_price = stock_data['Close'].iloc[idx + 5]
        return (future_price / current_price - 1) > 0.01  # Binary: >1% gain

    def calculate_simple_rsi(self, prices, period=14):
        """Calculate simple RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def train_simple_model(self, features, targets):
        """Train a simple logistic regression-like model"""
        features_array = np.array(features)
        targets_array = np.array(targets, dtype=int)

        # Simple feature normalization
        feature_means = np.mean(features_array, axis=0)
        feature_stds = np.std(features_array, axis=0) + 1e-8  # Avoid division by zero

        normalized_features = (features_array - feature_means) / feature_stds

        # Simple linear model weights (using correlation as proxy)
        weights = []
        for i in range(normalized_features.shape[1]):
            corr = np.corrcoef(normalized_features[:, i], targets_array)[0, 1]
            weights.append(corr if not np.isnan(corr) else 0)

        return {
            'weights': np.array(weights),
            'means': feature_means,
            'stds': feature_stds,
            'threshold': 0.5
        }

    def evaluate_model(self, model, features, targets):
        """Evaluate model accuracy"""
        features_array = np.array(features)
        targets_array = np.array(targets)

        # Normalize features
        normalized = (features_array - model['means']) / model['stds']

        # Make predictions
        scores = np.dot(normalized, model['weights'])
        predictions = (scores > model['threshold']).astype(int)

        accuracy = np.mean(predictions == targets_array)
        return accuracy

    def current_regime_prediction(self, regime_models, indicators):
        """Make prediction based on current market regime"""
        print("\n🎯 Generating regime-aware predictions...")

        # Determine current regime
        current_vix = indicators['VIX'].iloc[-1] if len(indicators['VIX']) > 0 else 20
        spy_trend = indicators['SPY'].pct_change(20).iloc[-1] if len(indicators['SPY']) > 20 else 0

        current_regime = self.classify_current_regime(current_vix, spy_trend)
        print(f"Current Market Regime: {current_regime} (VIX: {current_vix:.1f})")

        # Use appropriate model
        if current_regime in regime_models:
            model_info = regime_models[current_regime]
            print(f"Using {current_regime} model (Accuracy: {model_info['accuracy']:.1%})")

            # Generate predictions for current portfolio
            predictions = self.generate_regime_predictions(current_regime, model_info)
            return predictions
        else:
            print(f"No specific model for {current_regime}, using general strategy")
            return self.generate_general_predictions(current_vix, spy_trend)

    def classify_current_regime(self, vix, spy_trend):
        """Classify current market regime"""
        if vix > 40:
            return 'crisis'
        elif vix > 30:
            return 'high_volatility'
        elif vix < 15:
            return 'low_volatility'
        elif vix < 20 and spy_trend > 0.02:
            return 'bull_market'
        elif vix > 25 and spy_trend < -0.02:
            return 'bear_market'
        else:
            return 'normal'

    def generate_regime_predictions(self, regime, model_info):
        """Generate predictions using regime-specific model"""
        predictions = {}

        # Download current data for predictions
        current_data = {}
        for symbol in self.universe:
            try:
                data = yf.download(symbol, period='60d')  # Last 60 days
                if not data.empty:
                    current_data[symbol] = data
            except:
                continue

        for symbol in current_data:
            stock = current_data[symbol]

            try:
                # Extract current features
                features = self.extract_features(stock, len(stock) - 1)

                # Normalize features
                normalized = (np.array(features) - model_info['model']['means']) / model_info['model']['stds']

                # Make prediction
                score = np.dot(normalized, model_info['model']['weights'])
                probability = 1 / (1 + np.exp(-score))  # Sigmoid

                # Determine action and position size
                if probability > 0.65:
                    action = 'BUY'
                    position = min(15, probability * 20)
                elif probability < 0.35:
                    action = 'SELL'
                    position = min(10, (1 - probability) * 15)
                else:
                    action = 'HOLD'
                    position = 5

                predictions[symbol] = {
                    'action': action,
                    'probability': probability,
                    'position_size': position,
                    'regime': regime,
                    'confidence': model_info['accuracy']
                }

            except Exception as e:
                print(f"Error predicting {symbol}: {e}")
                continue

        return predictions

    def generate_general_predictions(self, vix, spy_trend):
        """Generate general predictions when no specific regime model exists"""
        predictions = {}

        for symbol in self.universe:
            # Simple heuristic based on market conditions
            if vix < 20 and spy_trend > 0:  # Calm bull market
                action = 'BUY'
                position = 10
                confidence = 0.6
            elif vix > 30:  # High volatility
                action = 'HOLD'
                position = 5
                confidence = 0.4
            else:  # Neutral conditions
                action = 'HOLD'
                position = 7
                confidence = 0.5

            predictions[symbol] = {
                'action': action,
                'probability': 0.5,
                'position_size': position,
                'regime': 'general',
                'confidence': confidence
            }

        return predictions

def main():
    print("🎯 REGIME-SPECIFIC TRAINING SYSTEM")
    print("=" * 50)

    trainer = RegimeSpecificTrainer()

    # Download historical data
    stock_data, indicators = trainer.download_market_data(years_back=10)

    if not indicators:
        print("❌ Cannot proceed without market indicators")
        return

    # Classify market regimes
    regime_data = trainer.classify_market_regimes(indicators)

    # Train regime-specific models
    regime_models = trainer.train_regime_models(stock_data, regime_data)

    # Generate current predictions
    predictions = trainer.current_regime_prediction(regime_models, indicators)

    print("\n🚀 REGIME-AWARE PREDICTIONS:")
    print("-" * 50)

    total_allocation = 0
    for symbol, pred in predictions.items():
        print(f"{symbol:6}: {pred['action']:4} - {pred['position_size']:4.1f}% allocation")
        print(f"        Regime: {pred['regime']:15} | Confidence: {pred['confidence']:.1%}")
        total_allocation += pred['position_size']

    print(f"\nTotal Allocation: {total_allocation:.1f}% | Cash: {100-total_allocation:.1f}%")

    return predictions, regime_models

if __name__ == "__main__":
    main()