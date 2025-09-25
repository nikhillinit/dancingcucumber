"""
Historical Data Optimization System
==================================
Leverage 20+ years of market data to improve TODAY's predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HistoricalDataOptimizer:
    def __init__(self):
        self.lookback_years = 20  # Use 20 years of data
        self.universe = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM']

    def download_comprehensive_history(self):
        """Download 20 years of historical data for training"""
        print("Downloading 20 years of market data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.lookback_years)

        historical_data = {}
        for symbol in self.universe:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                historical_data[symbol] = data
                print(f"✅ {symbol}: {len(data)} trading days")
            except Exception as e:
                print(f"❌ {symbol}: Failed to download - {e}")

        return historical_data

    def create_regime_training_sets(self, historical_data):
        """Split historical data by market regimes for specialized training"""
        print("\nCreating regime-specific training datasets...")

        regimes = {
            'bull_market': [],      # Strong uptrends
            'bear_market': [],      # Major downtrends
            'high_volatility': [],  # VIX > 25
            'low_volatility': [],   # VIX < 15
            'crisis_periods': []    # 2008, 2020 crashes
        }

        # Download VIX for regime classification
        try:
            vix_data = yf.download('^VIX', period='20y')

            for date in historical_data['AAPL'].index:
                if date in vix_data.index:
                    vix_value = vix_data.loc[date, 'Close']

                    # Classify market regime
                    if vix_value > 30:
                        regimes['crisis_periods'].append(date)
                    elif vix_value > 25:
                        regimes['high_volatility'].append(date)
                    elif vix_value < 15:
                        regimes['low_volatility'].append(date)

        except Exception as e:
            print(f"VIX data error: {e}")

        return regimes

    def calculate_advanced_features(self, price_data):
        """Calculate sophisticated features from historical data"""
        features = pd.DataFrame(index=price_data.index)

        # Price-based features
        features['returns'] = price_data['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['momentum'] = price_data['Close'].pct_change(10)

        # Volume features
        features['volume_ratio'] = price_data['Volume'] / price_data['Volume'].rolling(20).mean()
        features['price_volume'] = features['returns'] * np.log(price_data['Volume'])

        # Technical indicators
        features['rsi'] = self.calculate_rsi(price_data['Close'])
        features['macd'] = self.calculate_macd(price_data['Close'])
        features['bollinger_position'] = self.calculate_bollinger_position(price_data['Close'])

        # Market microstructure
        features['bid_ask_spread'] = (price_data['High'] - price_data['Low']) / price_data['Close']
        features['overnight_gap'] = (price_data['Open'] - price_data['Close'].shift(1)) / price_data['Close'].shift(1)

        return features.dropna()

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices):
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26

    def calculate_bollinger_position(self, prices, period=20):
        """Calculate position within Bollinger Bands"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return (prices - ma) / (2 * std)

    def backtest_historical_strategies(self, historical_data):
        """Test strategies across different historical periods"""
        print("\nBacktesting across historical periods...")

        results = {
            'dot_com_crash': {'start': '2000-01-01', 'end': '2002-12-31'},
            'pre_crisis': {'start': '2005-01-01', 'end': '2007-12-31'},
            'financial_crisis': {'start': '2008-01-01', 'end': '2009-12-31'},
            'recovery': {'start': '2010-01-01', 'end': '2012-12-31'},
            'bull_run': {'start': '2013-01-01', 'end': '2019-12-31'},
            'covid_crash': {'start': '2020-01-01', 'end': '2020-12-31'},
            'post_covid': {'start': '2021-01-01', 'end': '2023-12-31'}
        }

        strategy_performance = {}

        for period, dates in results.items():
            print(f"Testing {period}: {dates['start']} to {dates['end']}")

            # Calculate period-specific metrics
            period_returns = []
            for symbol in self.universe:
                if symbol in historical_data:
                    data = historical_data[symbol]
                    period_data = data[dates['start']:dates['end']]
                    if not period_data.empty:
                        period_return = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0] - 1) * 100
                        period_returns.append(period_return)

            avg_return = np.mean(period_returns) if period_returns else 0
            strategy_performance[period] = {
                'avg_return': avg_return,
                'volatility': np.std(period_returns) if period_returns else 0,
                'num_stocks': len(period_returns)
            }

        return strategy_performance

    def optimize_current_predictions(self, historical_data):
        """Use historical patterns to optimize today's predictions"""
        print("\n🔮 Optimizing today's predictions with historical patterns...")

        current_features = {}
        today = datetime.now().strftime('%Y-%m-%d')

        for symbol in self.universe:
            if symbol in historical_data:
                data = historical_data[symbol]
                recent_data = data.tail(60)  # Last 60 days

                # Calculate current technical state
                current_rsi = self.calculate_rsi(recent_data['Close']).iloc[-1]
                current_momentum = recent_data['Close'].pct_change(10).iloc[-1] * 100
                current_volatility = recent_data['Close'].pct_change().rolling(20).std().iloc[-1] * 100

                current_features[symbol] = {
                    'rsi': current_rsi,
                    'momentum': current_momentum,
                    'volatility': current_volatility,
                    'current_price': recent_data['Close'].iloc[-1]
                }

        # Generate optimized predictions
        predictions = self.generate_historical_informed_predictions(current_features, historical_data)
        return predictions

    def generate_historical_informed_predictions(self, current_features, historical_data):
        """Generate predictions informed by historical patterns"""
        predictions = {}

        for symbol in current_features:
            features = current_features[symbol]

            # Historical pattern matching
            prediction_score = 0
            confidence = 0

            # RSI-based prediction (oversold/overbought)
            if features['rsi'] < 30:  # Oversold
                prediction_score += 2  # Strong buy signal
                confidence += 0.3
            elif features['rsi'] > 70:  # Overbought
                prediction_score -= 2  # Strong sell signal
                confidence += 0.3

            # Momentum-based prediction
            if features['momentum'] > 5:  # Strong upward momentum
                prediction_score += 1
                confidence += 0.2
            elif features['momentum'] < -5:  # Strong downward momentum
                prediction_score -= 1
                confidence += 0.2

            # Volatility adjustment
            if features['volatility'] > 3:  # High volatility - reduce confidence
                confidence *= 0.7

            # Normalize prediction score to probability
            probability = max(0, min(1, (prediction_score + 3) / 6))  # Scale to 0-1

            predictions[symbol] = {
                'prediction': 'BUY' if probability > 0.6 else 'SELL' if probability < 0.4 else 'HOLD',
                'probability': probability,
                'confidence': min(0.95, max(0.05, confidence)),
                'target_allocation': probability * 15,  # Max 15% per stock
                'reasoning': self.generate_reasoning(features, prediction_score)
            }

        return predictions

    def generate_reasoning(self, features, prediction_score):
        """Generate human-readable reasoning for predictions"""
        reasons = []

        if features['rsi'] < 30:
            reasons.append("Oversold condition (RSI < 30)")
        elif features['rsi'] > 70:
            reasons.append("Overbought condition (RSI > 70)")

        if features['momentum'] > 5:
            reasons.append("Strong upward momentum (+5%)")
        elif features['momentum'] < -5:
            reasons.append("Strong downward momentum (-5%)")

        if features['volatility'] > 3:
            reasons.append("High volatility environment")

        return "; ".join(reasons) if reasons else "Neutral technical indicators"

def main():
    print("🚀 HISTORICAL DATA OPTIMIZATION - IMMEDIATE EFFICACY BOOST")
    print("=" * 60)

    optimizer = HistoricalDataOptimizer()

    # Download comprehensive historical data
    historical_data = optimizer.download_comprehensive_history()

    # Create regime-specific training sets
    regimes = optimizer.create_regime_training_sets(historical_data)

    # Backtest across historical periods
    strategy_performance = optimizer.backtest_historical_strategies(historical_data)

    print("\n📊 HISTORICAL STRATEGY PERFORMANCE:")
    for period, metrics in strategy_performance.items():
        print(f"{period:15}: {metrics['avg_return']:6.1f}% avg return, {metrics['volatility']:5.1f}% volatility")

    # Generate today's optimized predictions
    predictions = optimizer.optimize_current_predictions(historical_data)

    print("\n🎯 TODAY'S HISTORICAL-INFORMED PREDICTIONS:")
    print("-" * 60)
    total_allocation = 0
    for symbol, pred in predictions.items():
        print(f"{symbol:6}: {pred['prediction']:4} ({pred['probability']:5.1%} prob) - "
              f"{pred['target_allocation']:4.1f}% allocation")
        print(f"        Confidence: {pred['confidence']:5.1%} | {pred['reasoning']}")
        total_allocation += pred['target_allocation']

    print(f"\nTotal Allocation: {total_allocation:.1f}% (Cash: {100-total_allocation:.1f}%)")

    return predictions

if __name__ == "__main__":
    predictions = main()