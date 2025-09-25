"""
Walk-Forward Validation System
=============================
Test strategy performance using rolling historical windows
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WalkForwardValidator:
    def __init__(self, lookback_months=12, rebalance_frequency=30):
        self.lookback_months = lookback_months  # Training window
        self.rebalance_frequency = rebalance_frequency  # Days between rebalancing
        self.universe = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM']

    def run_walk_forward_test(self, start_date='2019-01-01', end_date='2024-01-01'):
        """Run walk-forward validation across time periods"""
        print(f"🔄 Running Walk-Forward Validation: {start_date} to {end_date}")
        print("=" * 60)

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        results = []
        current_date = start

        while current_date < end:
            # Define training and testing windows
            training_start = current_date - timedelta(days=self.lookback_months * 30)
            training_end = current_date
            testing_start = current_date
            testing_end = current_date + timedelta(days=self.rebalance_frequency)

            print(f"\nTesting Period: {testing_start.strftime('%Y-%m-%d')} to {testing_end.strftime('%Y-%m-%d')}")

            # Get period performance
            period_result = self.test_period(training_start, training_end, testing_start, testing_end)
            if period_result:
                results.append(period_result)

            current_date = testing_end

        return self.analyze_results(results)

    def test_period(self, train_start, train_end, test_start, test_end):
        """Test strategy performance for a specific period"""
        try:
            # Download training data
            training_data = self.download_period_data(train_start, train_end)
            if not training_data:
                return None

            # Generate predictions based on training data
            predictions = self.generate_period_predictions(training_data, test_start)

            # Download testing data to measure actual performance
            testing_data = self.download_period_data(test_start, test_end)
            if not testing_data:
                return None

            # Calculate actual returns
            actual_returns = self.calculate_actual_returns(testing_data)

            # Calculate strategy performance
            strategy_return = self.calculate_strategy_return(predictions, actual_returns)

            # Calculate benchmark (S&P 500) return
            benchmark_return = self.get_benchmark_return(test_start, test_end)

            return {
                'date': test_start,
                'strategy_return': strategy_return,
                'benchmark_return': benchmark_return,
                'alpha': strategy_return - benchmark_return,
                'predictions': predictions,
                'actual_returns': actual_returns
            }

        except Exception as e:
            print(f"Error in period {test_start}: {e}")
            return None

    def download_period_data(self, start_date, end_date):
        """Download market data for specific period"""
        data = {}
        for symbol in self.universe:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[symbol] = hist
            except:
                continue
        return data

    def generate_period_predictions(self, training_data, prediction_date):
        """Generate predictions based on training data patterns"""
        predictions = {}

        for symbol in training_data:
            data = training_data[symbol]
            if len(data) < 30:  # Need minimum data
                continue

            # Calculate technical indicators
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else returns.std()
            momentum = data['Close'].pct_change(10).iloc[-1] if len(data) > 10 else 0

            # Simple momentum + mean reversion strategy
            recent_return = data['Close'].pct_change(5).iloc[-1] if len(data) > 5 else 0

            # Prediction logic
            if momentum > 0.05 and recent_return > 0.02:  # Strong momentum
                prediction = 'BUY'
                confidence = min(0.8, abs(momentum) * 10)
            elif momentum < -0.05 and recent_return < -0.02:  # Strong negative momentum
                prediction = 'SELL'
                confidence = min(0.8, abs(momentum) * 10)
            else:
                prediction = 'HOLD'
                confidence = 0.3

            # Position sizing based on inverse volatility
            position_size = min(15, max(2, 10 / (volatility * 100))) if volatility > 0 else 5

            predictions[symbol] = {
                'action': prediction,
                'confidence': confidence,
                'position_size': position_size,
                'momentum': momentum,
                'volatility': volatility
            }

        return predictions

    def calculate_actual_returns(self, testing_data):
        """Calculate actual returns for the testing period"""
        returns = {}
        for symbol in testing_data:
            data = testing_data[symbol]
            if len(data) >= 2:
                returns[symbol] = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        return returns

    def calculate_strategy_return(self, predictions, actual_returns):
        """Calculate strategy return based on predictions and actual results"""
        total_return = 0
        total_weight = 0

        for symbol in predictions:
            if symbol in actual_returns:
                pred = predictions[symbol]
                actual = actual_returns[symbol]

                # Apply position sizing
                weight = pred['position_size'] / 100  # Convert to decimal

                # Apply prediction direction
                if pred['action'] == 'BUY':
                    total_return += weight * actual
                elif pred['action'] == 'SELL':
                    total_return += weight * (-actual)  # Short position
                # HOLD positions contribute 0

                total_weight += weight

        # Normalize by total weight
        return total_return / total_weight if total_weight > 0 else 0

    def get_benchmark_return(self, start_date, end_date):
        """Get S&P 500 return for the period"""
        try:
            spy = yf.Ticker('SPY')
            data = spy.history(start=start_date, end=end_date)
            if len(data) >= 2:
                return (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        except:
            pass
        return 0

    def analyze_results(self, results):
        """Analyze walk-forward validation results"""
        if not results:
            return None

        df = pd.DataFrame(results)

        analysis = {
            'total_periods': len(results),
            'win_rate': (df['alpha'] > 0).mean(),
            'avg_strategy_return': df['strategy_return'].mean(),
            'avg_benchmark_return': df['benchmark_return'].mean(),
            'avg_alpha': df['alpha'].mean(),
            'alpha_volatility': df['alpha'].std(),
            'best_period': df.loc[df['alpha'].idxmax()],
            'worst_period': df.loc[df['alpha'].idxmin()],
            'sharpe_ratio': df['alpha'].mean() / df['alpha'].std() if df['alpha'].std() > 0 else 0
        }

        self.print_analysis(analysis, df)
        return analysis

    def print_analysis(self, analysis, df):
        """Print detailed analysis results"""
        print("\n" + "=" * 60)
        print("📊 WALK-FORWARD VALIDATION RESULTS")
        print("=" * 60)

        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"Total Test Periods:     {analysis['total_periods']}")
        print(f"Win Rate:              {analysis['win_rate']:.1%}")
        print(f"Average Alpha:         {analysis['avg_alpha']:.2%}")
        print(f"Alpha Volatility:      {analysis['alpha_volatility']:.2%}")
        print(f"Sharpe Ratio:          {analysis['sharpe_ratio']:.2f}")

        print(f"\n📈 RETURNS COMPARISON:")
        print(f"Strategy Return:       {analysis['avg_strategy_return']:.2%}")
        print(f"Benchmark Return:      {analysis['avg_benchmark_return']:.2%}")
        print(f"Outperformance:        {analysis['avg_alpha']:.2%}")

        print(f"\n🏆 BEST PERIOD:")
        best = analysis['best_period']
        print(f"Date: {best['date'].strftime('%Y-%m-%d')}")
        print(f"Alpha: {best['alpha']:.2%}")
        print(f"Strategy: {best['strategy_return']:.2%}, Benchmark: {best['benchmark_return']:.2%}")

        print(f"\n📉 WORST PERIOD:")
        worst = analysis['worst_period']
        print(f"Date: {worst['date'].strftime('%Y-%m-%d')}")
        print(f"Alpha: {worst['alpha']:.2%}")
        print(f"Strategy: {worst['strategy_return']:.2%}, Benchmark: {worst['benchmark_return']:.2%}")

        # Monthly performance breakdown
        if len(df) > 12:
            monthly_alpha = df.set_index('date')['alpha'].resample('M').mean()
            print(f"\n📅 MONTHLY ALPHA PERFORMANCE:")
            for date, alpha in monthly_alpha.items():
                print(f"{date.strftime('%Y-%m')}: {alpha:.2%}")

def main():
    validator = WalkForwardValidator()

    # Run comprehensive walk-forward test
    results = validator.run_walk_forward_test(start_date='2020-01-01', end_date='2024-01-01')

    if results:
        print(f"\n🚀 HISTORICAL VALIDATION COMPLETE!")
        print(f"Average Alpha: {results['avg_alpha']:.2%}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        if results['avg_alpha'] > 0.02:  # 2% average outperformance
            print("✅ STRATEGY VALIDATED - Ready for deployment!")
        else:
            print("⚠️  STRATEGY NEEDS IMPROVEMENT - Consider refinements")

if __name__ == "__main__":
    main()