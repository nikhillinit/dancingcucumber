"""
Simple Backtest Test - AI Hedge Fund vs S&P 500
================================================
Test without external dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Import benchmark evaluator
from AIHedgeFund.apps.quant.benchmark_evaluation import BenchmarkEvaluator, PerformanceMetrics


def generate_market_data(symbols, days=252):
    """Generate realistic market data"""
    market_data = {}

    for i, symbol in enumerate(symbols):
        # Different characteristics for each stock
        base_return = 0.0003 + i * 0.00005  # Different returns
        volatility = 0.015 + i * 0.003  # Different volatilities

        # Generate price series with realistic properties
        returns = np.random.normal(base_return, volatility, days)

        # Add momentum
        returns = pd.Series(returns).ewm(span=5).mean().values

        # Add some trends
        trend = np.linspace(0, 0.0001 * i, days)
        returns = returns + trend

        # Calculate prices
        prices = 100 * (1 + i * 0.5) * np.exp(np.cumsum(returns))

        # Create OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        market_data[symbol] = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.005, 0.005, days)),
            'High': prices * (1 + np.random.uniform(0, 0.01, days)),
            'Low': prices * (1 - np.random.uniform(0, 0.01, days)),
            'Close': prices,
            'Volume': np.random.gamma(2, 1000000, days)
        }, index=dates)

    return market_data


def simulate_ai_trading(market_data, initial_capital=100000):
    """Simulate AI trading with enhanced strategies"""

    portfolio_value = initial_capital
    cash = initial_capital
    positions = {}
    portfolio_history = []
    trades = []

    # Get all dates
    all_dates = sorted(list(market_data[list(market_data.keys())[0]].index))

    print(f"📊 Simulating AI trading for {len(all_dates)} days...")

    for day_idx, date in enumerate(all_dates):
        # Calculate signals for each symbol
        signals = {}

        for symbol, data in market_data.items():
            if day_idx < 20:  # Need history
                continue

            # Get historical data up to current date
            hist_data = data.iloc[:day_idx+1]

            # Multiple signal generators (simulating our advanced components)

            # 1. Momentum signal
            returns = hist_data['Close'].pct_change().dropna()
            momentum = returns.rolling(10).mean().iloc[-1] if len(returns) > 10 else 0

            # 2. Mean reversion signal
            if len(returns) > 20:
                zscore = (hist_data['Close'].iloc[-1] - hist_data['Close'].rolling(20).mean().iloc[-1]) / hist_data['Close'].rolling(20).std().iloc[-1]
                mean_reversion = -zscore * 0.1
            else:
                mean_reversion = 0

            # 3. Volume signal
            volume_ratio = hist_data['Volume'].iloc[-1] / hist_data['Volume'].rolling(20).mean().iloc[-1] if len(hist_data) > 20 else 1
            volume_signal = (volume_ratio - 1) * 0.05

            # 4. Volatility regime adjustment (simulating regime detection)
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0.01
            vol_adjustment = max(0.5, min(1.5, 0.02 / volatility))  # Reduce size in high vol

            # 5. Combine signals with weights (simulating ensemble)
            raw_signal = (
                momentum * 0.35 +           # Momentum
                mean_reversion * 0.25 +      # Mean reversion
                volume_signal * 0.15 +       # Volume
                np.random.normal(0, 0.001)  # Small random component (ML prediction noise)
            ) * vol_adjustment

            # 6. Apply signal validation threshold (simulating validation framework)
            confidence = min(0.95, abs(raw_signal) * 50)

            if confidence > 0.55:  # Only trade high confidence signals
                signals[symbol] = {
                    'direction': 'long' if raw_signal > 0 else 'short',
                    'strength': abs(raw_signal),
                    'confidence': confidence
                }

        # Portfolio optimization (simulating our optimizer)
        if signals:
            # Risk budget (limit total exposure)
            max_exposure = 0.8
            total_confidence = sum(s['confidence'] for s in signals.values())

            # Allocate based on confidence
            target_positions = {}
            for symbol, signal in signals.items():
                weight = (signal['confidence'] / total_confidence) * max_exposure
                if signal['direction'] == 'short':
                    weight = -weight * 0.5  # Reduce short exposure

                target_positions[symbol] = weight

            # Execute trades
            for symbol, target_weight in target_positions.items():
                current_price = market_data[symbol].loc[date, 'Close']
                target_value = portfolio_value * target_weight
                target_shares = int(target_value / current_price)

                current_shares = positions.get(symbol, 0)
                shares_to_trade = target_shares - current_shares

                if abs(shares_to_trade) > 0:
                    # Transaction costs
                    trade_value = abs(shares_to_trade * current_price)
                    transaction_cost = trade_value * 0.001  # 0.1% transaction cost
                    cash -= transaction_cost

                    # Update position
                    positions[symbol] = target_shares
                    cash -= shares_to_trade * current_price

                    # Record trade
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'shares': shares_to_trade,
                        'price': current_price,
                        'value': shares_to_trade * current_price,
                        'action': 'buy' if shares_to_trade > 0 else 'sell'
                    })

        # Calculate portfolio value
        portfolio_value = cash
        for symbol, shares in positions.items():
            if shares != 0:
                current_price = market_data[symbol].loc[date, 'Close']
                portfolio_value += shares * current_price

        portfolio_history.append(portfolio_value)

        # Progress update
        if (day_idx + 1) % 50 == 0:
            returns_so_far = (portfolio_value / initial_capital - 1) * 100
            print(f"  Day {day_idx + 1}/{len(all_dates)}: Portfolio ${portfolio_value:,.0f} ({returns_so_far:+.1f}%)")

    # Create portfolio series
    portfolio_series = pd.Series(portfolio_history, index=all_dates, name='AI_Portfolio')

    # Calculate some trade statistics
    if trades:
        df_trades = pd.DataFrame(trades)
        winning_trades = df_trades[df_trades['value'] > 0]
        losing_trades = df_trades[df_trades['value'] < 0]

        # Calculate P&L for closed positions
        trade_pnl = []
        position_tracker = {}

        for trade in trades:
            symbol = trade['symbol']
            if symbol not in position_tracker:
                position_tracker[symbol] = []

            position_tracker[symbol].append(trade)

            # Simple P&L calculation (not perfect but good enough for testing)
            if len(position_tracker[symbol]) > 1:
                prev_trade = position_tracker[symbol][-2]
                if prev_trade['action'] != trade['action']:  # Position flip
                    pnl = (trade['price'] - prev_trade['price']) * min(abs(trade['shares']), abs(prev_trade['shares']))
                    if prev_trade['action'] == 'sell':
                        pnl = -pnl
                    trade_pnl.append({'pnl': pnl})

        # Convert to list of dicts for metric calculation
        trades_with_pnl = trade_pnl if trade_pnl else [{'pnl': 0}]
    else:
        trades_with_pnl = []

    print(f"\n✅ Simulation complete!")
    print(f"  Final value: ${portfolio_value:,.2f}")
    print(f"  Total return: {(portfolio_value/initial_capital - 1)*100:.2f}%")
    print(f"  Total trades: {len(trades)}")

    return portfolio_series, trades_with_pnl


def simulate_sp500(start_date, end_date, initial_value=100000):
    """Simulate S&P 500 performance"""
    days = (end_date - start_date).days

    # Historical S&P 500 characteristics
    annual_return = 0.10  # 10% historical average
    annual_volatility = 0.16  # 16% historical volatility

    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)

    # Generate returns
    returns = np.random.normal(daily_return, daily_volatility, days)

    # Add some autocorrelation (momentum)
    returns = pd.Series(returns).ewm(span=3).mean().values

    # Calculate prices
    prices = initial_value * np.exp(np.cumsum(returns))

    # Create series
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sp500 = pd.Series(prices, index=dates[:len(prices)], name='S&P500')

    return sp500


def evaluate_performance(portfolio_series, sp500_series, trades):
    """Evaluate portfolio performance against S&P 500"""

    # Align series
    common_dates = portfolio_series.index.intersection(sp500_series.index)
    portfolio = portfolio_series[common_dates]
    sp500 = sp500_series[common_dates]

    # Calculate returns
    portfolio_returns = portfolio.pct_change().dropna()
    sp500_returns = sp500.pct_change().dropna()

    # Create evaluator
    evaluator = BenchmarkEvaluator()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(portfolio_returns, sp500_returns, trades)

    return metrics


def print_performance_report(metrics):
    """Print formatted performance report"""

    print("\n" + "="*60)
    print("📈 AI HEDGE FUND PERFORMANCE REPORT vs S&P 500")
    print("="*60)

    # Determine verdict
    score = 0
    if metrics.sharpe_ratio > 1.5: score += 3
    elif metrics.sharpe_ratio > 1.0: score += 2
    elif metrics.sharpe_ratio > 0.5: score += 1

    if metrics.alpha > 0.05: score += 3
    elif metrics.alpha > 0.02: score += 2
    elif metrics.alpha > 0: score += 1

    if metrics.information_ratio > 0.5: score += 2
    elif metrics.information_ratio > 0: score += 1

    if score >= 7:
        verdict = "🌟 EXCELLENT - Significantly outperforms S&P 500"
    elif score >= 4:
        verdict = "✅ GOOD - Outperforms S&P 500"
    elif score >= 2:
        verdict = "📊 MODERATE - Comparable to S&P 500"
    else:
        verdict = "⚠️ UNDERPERFORMING - Below S&P 500 benchmark"

    print(f"\n🎯 VERDICT: {verdict}")

    print("\n📊 RETURNS:")
    print(f"  Annualized Return: {metrics.annualized_return:.2%}")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Alpha (excess return): {metrics.alpha:.2%}")
    print(f"  Beta (market correlation): {metrics.beta:.2f}")

    print("\n📈 RISK-ADJUSTED PERFORMANCE:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    sharpe_interpretation = "Excellent" if metrics.sharpe_ratio > 1.5 else "Good" if metrics.sharpe_ratio > 1.0 else "Moderate" if metrics.sharpe_ratio > 0.5 else "Poor"
    print(f"    ({sharpe_interpretation} - S&P typically ~0.8-1.0)")

    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Information Ratio: {metrics.information_ratio:.2f}")
    info_interpretation = "Excellent" if metrics.information_ratio > 0.5 else "Good" if metrics.information_ratio > 0.25 else "Moderate" if metrics.information_ratio > 0 else "Poor"
    print(f"    ({info_interpretation} - Consistency of alpha)")

    print("\n⚠️ RISK METRICS:")
    print(f"  Volatility: {metrics.volatility:.2%} annual")
    vol_interpretation = "Low" if metrics.volatility < 0.15 else "Moderate" if metrics.volatility < 0.25 else "High"
    print(f"    ({vol_interpretation} risk)")

    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    dd_interpretation = "Excellent" if metrics.max_drawdown > -0.1 else "Good" if metrics.max_drawdown > -0.2 else "Moderate" if metrics.max_drawdown > -0.3 else "High"
    print(f"    ({dd_interpretation} risk control)")

    print(f"  Downside Volatility: {metrics.downside_volatility:.2%}")

    print("\n📊 MARKET CAPTURE:")
    print(f"  Upside Capture: {metrics.upside_capture:.1%}")
    print(f"  Downside Capture: {metrics.downside_capture:.1%}")
    print(f"  Capture Ratio: {metrics.capture_ratio:.2f}")
    capture_interpretation = "Excellent" if metrics.capture_ratio > 1.5 else "Good" if metrics.capture_ratio > 1.2 else "Moderate" if metrics.capture_ratio > 1.0 else "Poor"
    print(f"    ({capture_interpretation} - Want high upside, low downside)")

    print("\n📊 TRADING PERFORMANCE:")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    win_interpretation = "Excellent" if metrics.win_rate > 0.6 else "Good" if metrics.win_rate > 0.55 else "Moderate" if metrics.win_rate > 0.5 else "Poor"
    print(f"    ({win_interpretation})")

    print("\n✅ STATISTICAL SIGNIFICANCE:")
    print(f"  T-Statistic: {metrics.t_statistic:.3f}")
    print(f"  P-Value: {metrics.p_value:.4f}")

    if metrics.p_value < 0.01:
        significance = "✅ HIGHLY SIGNIFICANT (99% confidence)"
    elif metrics.p_value < 0.05:
        significance = "✅ STATISTICALLY SIGNIFICANT (95% confidence)"
    elif metrics.p_value < 0.10:
        significance = "📊 MARGINALLY SIGNIFICANT (90% confidence)"
    else:
        significance = "❌ NOT STATISTICALLY SIGNIFICANT"

    print(f"  Result: {significance}")

    # Summary recommendations
    print("\n💡 KEY INSIGHTS:")

    insights = []
    if metrics.sharpe_ratio > 1.5:
        insights.append("✅ Excellent risk-adjusted returns")
    if metrics.alpha > 0.05:
        insights.append("✅ Strong alpha generation")
    if metrics.max_drawdown > -0.15:
        insights.append("✅ Good drawdown control")
    if metrics.information_ratio > 0.5:
        insights.append("✅ Consistent outperformance")
    if metrics.win_rate > 0.55:
        insights.append("✅ High win rate")

    if metrics.volatility > 0.25:
        insights.append("⚠️ High volatility - consider risk reduction")
    if metrics.beta > 1.5:
        insights.append("⚠️ High market correlation - need more diversification")
    if metrics.max_drawdown < -0.25:
        insights.append("⚠️ Large drawdowns - improve risk management")

    for insight in insights:
        print(f"  • {insight}")

    # Final recommendation
    print("\n" + "="*60)
    if metrics.sharpe_ratio > 1.5 and metrics.alpha > 0.05 and metrics.p_value < 0.05:
        print("🚀 RECOMMENDATION: Ready for paper trading!")
        print("   Start with small capital and monitor performance closely.")
    elif metrics.sharpe_ratio > 1.0 and metrics.alpha > 0:
        print("📈 RECOMMENDATION: Promising results, continue optimization.")
        print("   Focus on improving signal quality and risk management.")
    elif metrics.sharpe_ratio > 0.5:
        print("📊 RECOMMENDATION: Moderate performance, needs improvement.")
        print("   Review signal generation and portfolio optimization.")
    else:
        print("⚠️ RECOMMENDATION: Significant improvements needed.")
        print("   Re-evaluate strategy and consider parameter tuning.")

    print("="*60)


def main():
    """Run the complete test"""
    print("="*60)
    print("🤖 AI HEDGE FUND BACKTESTING SIMULATION")
    print("="*60)

    # Parameters
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
    trading_days = 252  # One year
    initial_capital = 100000

    print(f"\n📊 Test Parameters:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Trading Days: {trading_days}")
    print(f"  Initial Capital: ${initial_capital:,}")

    # Generate market data
    print(f"\n📈 Generating market data...")
    market_data = generate_market_data(symbols, trading_days)

    # Run AI trading simulation
    print(f"\n🤖 Running AI trading simulation...")
    portfolio, trades = simulate_ai_trading(market_data, initial_capital)

    # Generate S&P 500 benchmark
    print(f"\n📊 Generating S&P 500 benchmark...")
    sp500 = simulate_sp500(portfolio.index[0], portfolio.index[-1], initial_capital)

    # Evaluate performance
    print(f"\n📈 Evaluating performance...")
    metrics = evaluate_performance(portfolio, sp500, trades)

    # Print report
    print_performance_report(metrics)

    print("\n✅ Test complete!")

    return metrics


if __name__ == "__main__":
    metrics = main()