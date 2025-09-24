"""
Standalone Backtest Test - AI Hedge Fund vs S&P 500
=====================================================
Fully self-contained test with no external dependencies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
    information_ratio: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    downside_volatility: float
    upside_capture: float
    downside_capture: float
    capture_ratio: float
    t_statistic: float
    p_value: float


class BenchmarkEvaluator:
    """Simple benchmark evaluator"""

    def calculate_metrics(self, returns: pd.Series, benchmark_returns: pd.Series, trades: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        # Basic calculations
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio (assume 2% risk-free rate)
        risk_free = 0.02 / 252
        excess_returns = returns - risk_free
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Alpha and Beta
        if len(benchmark_returns) > 0 and benchmark_returns.std() > 0:
            # Calculate beta using covariance
            covariance = returns.cov(benchmark_returns)
            variance = benchmark_returns.var()
            beta = covariance / variance if variance > 0 else 1

            # Calculate benchmark total return
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1

            # Alpha = Portfolio return - (Risk free + Beta * (Market return - Risk free))
            alpha = annualized_return - (0.02 + beta * (benchmark_annualized - 0.02))
        else:
            beta = 1
            alpha = 0

        # Information ratio
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio and downside volatility
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
        sortino_ratio = (annualized_return - 0.02) / downside_volatility

        # Market capture ratios
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0

        if up_market.any():
            upside_capture = (returns[up_market].mean() / benchmark_returns[up_market].mean()) if benchmark_returns[up_market].mean() != 0 else 0
        else:
            upside_capture = 0

        if down_market.any():
            downside_capture = (returns[down_market].mean() / benchmark_returns[down_market].mean()) if benchmark_returns[down_market].mean() != 0 else 0
        else:
            downside_capture = 0

        capture_ratio = upside_capture / abs(downside_capture) if downside_capture != 0 else upside_capture

        # Statistical significance (t-test for alpha)
        if len(active_returns) > 0:
            t_statistic = (active_returns.mean() * np.sqrt(len(active_returns))) / active_returns.std() if active_returns.std() > 0 else 0
            # Approximate p-value manually without scipy
            # Using simplified normal approximation
            if abs(t_statistic) > 2.576:
                p_value = 0.01  # 99% confidence
            elif abs(t_statistic) > 1.96:
                p_value = 0.05  # 95% confidence
            elif abs(t_statistic) > 1.645:
                p_value = 0.10  # 90% confidence
            else:
                p_value = 1.0 - min(0.9, abs(t_statistic) / 2)  # Rough approximation
        else:
            t_statistic = 0
            p_value = 1

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            downside_volatility=downside_volatility,
            upside_capture=upside_capture,
            downside_capture=downside_capture,
            capture_ratio=capture_ratio,
            t_statistic=t_statistic,
            p_value=p_value
        )


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
    """Simulate AI trading - Fidelity-style daily rebalancing, long-only positions"""

    portfolio_value = initial_capital
    cash = initial_capital
    positions = {}
    portfolio_history = []
    trades = []
    daily_trades = 0

    # Get all dates
    all_dates = sorted(list(market_data[list(market_data.keys())[0]].index))

    print(f"[INFO] Simulating Fidelity-compatible AI trading for {len(all_dates)} days...")
    print(f"[INFO] Strategy: Daily portfolio rebalancing, long positions only, no derivatives")

    for day_idx, date in enumerate(all_dates):
        # Skip first 30 days to build history
        if day_idx < 30:
            portfolio_history.append(portfolio_value)
            continue

        # === DAILY SIGNAL GENERATION ===
        signals = {}

        for symbol, data in market_data.items():
            # Get historical data up to current date
            hist_data = data.iloc[:day_idx+1]
            returns = hist_data['Close'].pct_change().dropna()

            # Enhanced signal generation for daily trading

            # 1. Multi-timeframe momentum (5, 10, 20 day)
            momentum_5d = returns.rolling(5).mean().iloc[-1] if len(returns) >= 5 else 0
            momentum_10d = returns.rolling(10).mean().iloc[-1] if len(returns) >= 10 else 0
            momentum_20d = returns.rolling(20).mean().iloc[-1] if len(returns) >= 20 else 0
            momentum_signal = (momentum_5d * 0.5 + momentum_10d * 0.3 + momentum_20d * 0.2)

            # 2. RSI-style mean reversion
            price_change = (hist_data['Close'].iloc[-1] / hist_data['Close'].rolling(14).mean().iloc[-1] - 1) if len(hist_data) >= 14 else 0
            rsi_signal = -price_change * 0.3 if abs(price_change) > 0.05 else 0  # Only strong reversals

            # 3. Volume confirmation
            volume_sma = hist_data['Volume'].rolling(20).mean().iloc[-1] if len(hist_data) >= 20 else hist_data['Volume'].iloc[-1]
            volume_signal = min(0.1, (hist_data['Volume'].iloc[-1] / volume_sma - 1) * 0.1)

            # 4. Volatility adjustment
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.02
            vol_adjustment = max(0.3, min(1.0, 0.02 / volatility))  # Conservative in high vol

            # 5. Combined signal (only long positions for Fidelity)
            raw_signal = (
                momentum_signal * 0.5 +      # Momentum weight
                rsi_signal * 0.2 +          # Mean reversion
                volume_signal * 0.3         # Volume confirmation
            ) * vol_adjustment

            # 6. Signal validation - only take LONG positions with high confidence
            if raw_signal > 0.002:  # Positive signal threshold (0.2% daily return expectation)
                confidence = min(0.9, raw_signal * 100)
                if confidence > 0.4:  # Higher threshold for daily trading
                    signals[symbol] = {
                        'direction': 'long',  # Only long positions
                        'strength': raw_signal,
                        'confidence': confidence
                    }

        # === DAILY PORTFOLIO REBALANCING ===
        if signals:
            # Conservative position sizing for Fidelity account
            max_portfolio_exposure = 0.95  # Keep 5% cash
            max_single_position = 0.15     # Max 15% in any single stock

            # Select top signals for daily portfolio
            sorted_signals = sorted(signals.items(), key=lambda x: x[1]['confidence'], reverse=True)
            top_signals = dict(sorted_signals[:5])  # Max 5 positions per day

            # Calculate position sizes
            total_confidence = sum(s['confidence'] for s in top_signals.values())
            target_positions = {}

            for symbol, signal in top_signals.items():
                # Weight by confidence, capped at max single position
                base_weight = (signal['confidence'] / total_confidence) * max_portfolio_exposure
                position_weight = min(base_weight, max_single_position)
                target_positions[symbol] = position_weight

            # === EXECUTE DAILY TRADES ===
            trades_today = []

            # Close positions not in today's signals
            for symbol in list(positions.keys()):
                if symbol not in target_positions and positions[symbol] > 0:
                    # Sell entire position
                    shares_to_sell = positions[symbol]
                    current_price = market_data[symbol].iloc[day_idx]['Close']

                    cash += shares_to_sell * current_price * 0.999  # 0.1% transaction cost
                    positions[symbol] = 0

                    trades_today.append({
                        'date': date,
                        'symbol': symbol,
                        'shares': -shares_to_sell,
                        'price': current_price,
                        'value': -shares_to_sell * current_price,
                        'action': 'sell'
                    })

            # Open/adjust positions in today's signals
            for symbol, target_weight in target_positions.items():
                current_price = market_data[symbol].iloc[day_idx]['Close']
                target_value = portfolio_value * target_weight
                target_shares = int(target_value / current_price)

                current_shares = positions.get(symbol, 0)
                shares_to_trade = target_shares - current_shares

                if abs(shares_to_trade) > 0:
                    # Execute trade with transaction costs
                    trade_value = shares_to_trade * current_price
                    cash -= trade_value * 1.001  # 0.1% transaction cost
                    positions[symbol] = target_shares

                    trades_today.append({
                        'date': date,
                        'symbol': symbol,
                        'shares': shares_to_trade,
                        'price': current_price,
                        'value': trade_value,
                        'action': 'buy' if shares_to_trade > 0 else 'sell'
                    })

            trades.extend(trades_today)
            daily_trades = len(trades_today)
        else:
            # No signals - hold current positions
            daily_trades = 0

        # Calculate portfolio value
        portfolio_value = cash
        for symbol, shares in positions.items():
            if shares != 0:
                # Use iloc to access by position instead of date
                current_price = market_data[symbol].iloc[day_idx]['Close']
                portfolio_value += shares * current_price

        portfolio_history.append(portfolio_value)

        # Progress update with trade info
        if (day_idx + 1) % 50 == 0:
            returns_so_far = (portfolio_value / initial_capital - 1) * 100
            num_positions = len([p for p in positions.values() if p > 0])
            print(f"  Day {day_idx + 1}/{len(all_dates)}: ${portfolio_value:,.0f} ({returns_so_far:+.1f}%) | {num_positions} positions | {daily_trades} trades today")

    # Create portfolio series
    portfolio_series = pd.Series(portfolio_history, index=all_dates, name='AI_Portfolio')

    # Calculate some trade statistics
    if trades:
        df_trades = pd.DataFrame(trades)

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

    print(f"\n[SUCCESS] Simulation complete!")
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
    print("[REPORT] AI HEDGE FUND PERFORMANCE REPORT vs S&P 500")
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
        verdict = "[EXCELLENT] Significantly outperforms S&P 500"
    elif score >= 4:
        verdict = "[SUCCESS] GOOD - Outperforms S&P 500"
    elif score >= 2:
        verdict = "[MODERATE] Comparable to S&P 500"
    else:
        verdict = "[WARNING] UNDERPERFORMING - Below S&P 500 benchmark"

    print(f"\n[VERDICT]: {verdict}")

    print("\n[INFO] RETURNS:")
    print(f"  Annualized Return: {metrics.annualized_return:.2%}")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Alpha (excess return): {metrics.alpha:.2%}")
    print(f"  Beta (market correlation): {metrics.beta:.2f}")

    print("\n[REPORT] RISK-ADJUSTED PERFORMANCE:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    sharpe_interpretation = "Excellent" if metrics.sharpe_ratio > 1.5 else "Good" if metrics.sharpe_ratio > 1.0 else "Moderate" if metrics.sharpe_ratio > 0.5 else "Poor"
    print(f"    ({sharpe_interpretation} - S&P typically ~0.8-1.0)")

    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Information Ratio: {metrics.information_ratio:.2f}")
    info_interpretation = "Excellent" if metrics.information_ratio > 0.5 else "Good" if metrics.information_ratio > 0.25 else "Moderate" if metrics.information_ratio > 0 else "Poor"
    print(f"    ({info_interpretation} - Consistency of alpha)")

    print("\n[WARNING] RISK METRICS:")
    print(f"  Volatility: {metrics.volatility:.2%} annual")
    vol_interpretation = "Low" if metrics.volatility < 0.15 else "Moderate" if metrics.volatility < 0.25 else "High"
    print(f"    ({vol_interpretation} risk)")

    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    dd_interpretation = "Excellent" if metrics.max_drawdown > -0.1 else "Good" if metrics.max_drawdown > -0.2 else "Moderate" if metrics.max_drawdown > -0.3 else "High"
    print(f"    ({dd_interpretation} risk control)")

    print(f"  Downside Volatility: {metrics.downside_volatility:.2%}")

    print("\n[INFO] MARKET CAPTURE:")
    print(f"  Upside Capture: {metrics.upside_capture:.1%}")
    print(f"  Downside Capture: {metrics.downside_capture:.1%}")
    print(f"  Capture Ratio: {metrics.capture_ratio:.2f}")
    capture_interpretation = "Excellent" if metrics.capture_ratio > 1.5 else "Good" if metrics.capture_ratio > 1.2 else "Moderate" if metrics.capture_ratio > 1.0 else "Poor"
    print(f"    ({capture_interpretation} - Want high upside, low downside)")

    print("\n[INFO] TRADING PERFORMANCE:")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    win_interpretation = "Excellent" if metrics.win_rate > 0.6 else "Good" if metrics.win_rate > 0.55 else "Moderate" if metrics.win_rate > 0.5 else "Poor"
    print(f"    ({win_interpretation})")

    print("\n[SUCCESS] STATISTICAL SIGNIFICANCE:")
    print(f"  T-Statistic: {metrics.t_statistic:.3f}")
    print(f"  P-Value: {metrics.p_value:.4f}")

    if metrics.p_value < 0.01:
        significance = "[SUCCESS] HIGHLY SIGNIFICANT (99% confidence)"
    elif metrics.p_value < 0.05:
        significance = "[SUCCESS] STATISTICALLY SIGNIFICANT (95% confidence)"
    elif metrics.p_value < 0.10:
        significance = "[MODERATE] MARGINALLY SIGNIFICANT (90% confidence)"
    else:
        significance = "[FAIL] NOT STATISTICALLY SIGNIFICANT"

    print(f"  Result: {significance}")

    # Summary recommendations
    print("\n[INSIGHTS] KEY INSIGHTS:")

    insights = []
    if metrics.sharpe_ratio > 1.5:
        insights.append("[SUCCESS] Excellent risk-adjusted returns")
    if metrics.alpha > 0.05:
        insights.append("[SUCCESS] Strong alpha generation")
    if metrics.max_drawdown > -0.15:
        insights.append("[SUCCESS] Good drawdown control")
    if metrics.information_ratio > 0.5:
        insights.append("[SUCCESS] Consistent outperformance")
    if metrics.win_rate > 0.55:
        insights.append("[SUCCESS] High win rate")

    if metrics.volatility > 0.25:
        insights.append("[WARNING] High volatility - consider risk reduction")
    if metrics.beta > 1.5:
        insights.append("[WARNING] High market correlation - need more diversification")
    if metrics.max_drawdown < -0.25:
        insights.append("[WARNING] Large drawdowns - improve risk management")

    for insight in insights:
        print(f"  • {insight}")

    # Final recommendation
    print("\n" + "="*60)
    if metrics.sharpe_ratio > 1.5 and metrics.alpha > 0.05 and metrics.p_value < 0.05:
        print("[READY] RECOMMENDATION: Ready for paper trading!")
        print("   Start with small capital and monitor performance closely.")
    elif metrics.sharpe_ratio > 1.0 and metrics.alpha > 0:
        print("[REPORT] RECOMMENDATION: Promising results, continue optimization.")
        print("   Focus on improving signal quality and risk management.")
    elif metrics.sharpe_ratio > 0.5:
        print("[INFO] RECOMMENDATION: Moderate performance, needs improvement.")
        print("   Review signal generation and portfolio optimization.")
    else:
        print("[WARNING] RECOMMENDATION: Significant improvements needed.")
        print("   Re-evaluate strategy and consider parameter tuning.")

    print("="*60)


def main():
    """Run the complete test"""
    print("="*60)
    print("[AI] AI HEDGE FUND BACKTESTING SIMULATION")
    print("="*60)

    # Parameters - Fidelity-tradeable large cap stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'PG', 'UNH', 'HD']
    trading_days = 252  # One year
    initial_capital = 100000

    print(f"\n[INFO] Test Parameters:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Trading Days: {trading_days}")
    print(f"  Initial Capital: ${initial_capital:,}")

    # Generate market data
    print(f"\n[REPORT] Generating market data...")
    market_data = generate_market_data(symbols, trading_days)

    # Run AI trading simulation
    print(f"\n[AI] Running AI trading simulation...")
    portfolio, trades = simulate_ai_trading(market_data, initial_capital)

    # Generate S&P 500 benchmark
    print(f"\n[INFO] Generating S&P 500 benchmark...")
    sp500 = simulate_sp500(portfolio.index[0], portfolio.index[-1], initial_capital)

    # Evaluate performance
    print(f"\n[REPORT] Evaluating performance...")
    metrics = evaluate_performance(portfolio, sp500, trades)

    # Print report
    print_performance_report(metrics)

    print("\n[SUCCESS] Test complete!")

    return metrics


if __name__ == "__main__":
    metrics = main()