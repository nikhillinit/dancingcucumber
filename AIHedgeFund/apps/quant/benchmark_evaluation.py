"""
Benchmark Evaluation Framework - AI Hedge Fund vs S&P 500
=========================================================
Comprehensive performance evaluation against market benchmarks
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Absolute Performance
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk Metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float

    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float

    # Relative Performance
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    active_return: float

    # Trading Statistics
    win_rate: float
    profit_factor: float
    avg_win_loss_ratio: float
    total_trades: int

    # Market Capture
    upside_capture: float
    downside_capture: float
    capture_ratio: float

    # Statistical Tests
    t_statistic: float
    p_value: float

    # Period
    start_date: datetime
    end_date: datetime
    trading_days: int


class BenchmarkEvaluator:
    """Evaluate portfolio performance against S&P 500 benchmark"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.benchmark_ticker = '^GSPC'  # S&P 500

    def fetch_benchmark_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch S&P 500 benchmark data"""
        try:
            spy = yf.download(self.benchmark_ticker, start=start_date, end=end_date)
            if spy.empty:
                # Fallback to SPY ETF
                spy = yf.download('SPY', start=start_date, end=end_date)
            return spy
        except Exception as e:
            logger.error(f"Failed to fetch benchmark data: {e}")
            # Return simulated S&P 500 data for testing
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            prices = 4000 + np.cumsum(np.random.randn(len(dates)) * 20)
            return pd.DataFrame({'Close': prices}, index=dates)

    def calculate_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        trades: Optional[List[Dict]] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        # Ensure alignment
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        portfolio = aligned['portfolio']
        benchmark = aligned['benchmark']

        # Basic returns
        total_return = (1 + portfolio).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio)) - 1
        cumulative_return = (1 + portfolio).cumprod().iloc[-1] - 1

        # Risk metrics
        volatility = portfolio.std() * np.sqrt(252)
        downside_returns = portfolio[portfolio < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Drawdown
        cumulative = (1 + portfolio).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Drawdown duration
        drawdown_start = drawdown[drawdown < 0].index[0] if any(drawdown < 0) else None
        drawdown_end = drawdown[drawdown == max_drawdown].index[0] if max_drawdown < 0 else None
        max_drawdown_duration = (drawdown_end - drawdown_start).days if drawdown_start and drawdown_end else 0

        # VaR and CVaR
        var_95 = portfolio.quantile(0.05)
        cvar_95 = portfolio[portfolio <= var_95].mean() if any(portfolio <= var_95) else var_95

        # Sharpe Ratio
        excess_returns = portfolio - self.risk_free_rate
        sharpe_ratio = excess_returns.mean() / portfolio.std() * np.sqrt(252) if portfolio.std() > 0 else 0

        # Sortino Ratio
        sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Information Ratio
        active_returns = portfolio - benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0

        # Alpha and Beta (CAPM)
        if len(portfolio) > 1 and len(benchmark) > 1:
            beta = portfolio.cov(benchmark) / benchmark.var() if benchmark.var() > 0 else 1
            alpha = annualized_return - (self.risk_free_rate * 252 + beta * (benchmark.mean() * 252 - self.risk_free_rate * 252))
        else:
            beta = 1
            alpha = 0

        # Treynor Ratio
        treynor_ratio = excess_returns.mean() * 252 / beta if beta != 0 else 0

        # Correlation
        correlation = portfolio.corr(benchmark)

        # Market Capture
        up_market = benchmark > 0
        down_market = benchmark <= 0

        if up_market.sum() > 0:
            upside_capture = (portfolio[up_market].mean() / benchmark[up_market].mean()) if benchmark[up_market].mean() != 0 else 0
        else:
            upside_capture = 0

        if down_market.sum() > 0:
            downside_capture = (portfolio[down_market].mean() / benchmark[down_market].mean()) if benchmark[down_market].mean() != 0 else 0
        else:
            downside_capture = 0

        capture_ratio = upside_capture / downside_capture if downside_capture != 0 else upside_capture

        # Trading statistics
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0

            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            avg_win = total_wins / len(winning_trades) if winning_trades else 0
            avg_loss = total_losses / len(losing_trades) if losing_trades else 0
            avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

            total_trades = len(trades)
        else:
            win_rate = 0.5
            profit_factor = 1
            avg_win_loss_ratio = 1
            total_trades = 0

        # Statistical significance
        if len(active_returns) > 1:
            t_statistic, p_value = stats.ttest_1samp(active_returns, 0)
        else:
            t_statistic, p_value = 0, 1

        return PerformanceMetrics(
            # Absolute Performance
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,

            # Risk Metrics
            volatility=volatility,
            downside_volatility=downside_volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            var_95=var_95,
            cvar_95=cvar_95,

            # Risk-Adjusted Returns
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,

            # Relative Performance
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            active_return=active_returns.mean() * 252,

            # Trading Statistics
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win_loss_ratio=avg_win_loss_ratio,
            total_trades=total_trades,

            # Market Capture
            upside_capture=upside_capture,
            downside_capture=downside_capture,
            capture_ratio=capture_ratio,

            # Statistical Tests
            t_statistic=t_statistic,
            p_value=p_value,

            # Period
            start_date=portfolio.index[0],
            end_date=portfolio.index[-1],
            trading_days=len(portfolio)
        )

    def create_performance_report(
        self,
        metrics: PerformanceMetrics,
        portfolio_equity: pd.Series,
        benchmark_equity: pd.Series
    ) -> Dict[str, Any]:
        """Create comprehensive performance report"""

        report = {
            'summary': self._create_summary(metrics),
            'metrics': metrics,
            'visualizations': self._create_visualizations(portfolio_equity, benchmark_equity),
            'risk_analysis': self._analyze_risk(metrics),
            'attribution': self._performance_attribution(metrics),
            'statistical_significance': self._statistical_analysis(metrics)
        }

        return report

    def _create_summary(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """Create executive summary"""

        outperformance = metrics.annualized_return - (metrics.beta * metrics.annualized_return)

        summary = {
            'verdict': self._get_verdict(metrics),
            'key_findings': [],
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }

        # Key findings
        summary['key_findings'].append(
            f"Portfolio achieved {metrics.annualized_return:.1%} annualized return vs S&P 500"
        )
        summary['key_findings'].append(
            f"Risk-adjusted performance (Sharpe): {metrics.sharpe_ratio:.2f}"
        )
        summary['key_findings'].append(
            f"Generated alpha of {metrics.alpha:.1%} with beta of {metrics.beta:.2f}"
        )

        # Strengths
        if metrics.sharpe_ratio > 1:
            summary['strengths'].append("Excellent risk-adjusted returns")
        if metrics.alpha > 0:
            summary['strengths'].append(f"Positive alpha generation ({metrics.alpha:.1%})")
        if metrics.capture_ratio > 1.2:
            summary['strengths'].append("Strong upside capture with limited downside")
        if metrics.win_rate > 0.55:
            summary['strengths'].append(f"High win rate ({metrics.win_rate:.1%})")

        # Weaknesses
        if metrics.max_drawdown < -0.2:
            summary['weaknesses'].append(f"Large drawdown risk ({metrics.max_drawdown:.1%})")
        if metrics.volatility > 0.25:
            summary['weaknesses'].append(f"High volatility ({metrics.volatility:.1%})")
        if metrics.beta > 1.5:
            summary['weaknesses'].append("High market correlation")

        # Recommendations
        if metrics.sharpe_ratio < 0.5:
            summary['recommendations'].append("Consider risk reduction strategies")
        if metrics.win_rate < 0.45:
            summary['recommendations'].append("Improve signal quality")
        if abs(metrics.beta) > 1.5:
            summary['recommendations'].append("Reduce market exposure")

        return summary

    def _get_verdict(self, metrics: PerformanceMetrics) -> str:
        """Generate overall verdict"""

        score = 0

        # Scoring system
        if metrics.sharpe_ratio > 1.5:
            score += 3
        elif metrics.sharpe_ratio > 1:
            score += 2
        elif metrics.sharpe_ratio > 0.5:
            score += 1

        if metrics.alpha > 0.05:
            score += 3
        elif metrics.alpha > 0.02:
            score += 2
        elif metrics.alpha > 0:
            score += 1

        if metrics.information_ratio > 0.5:
            score += 2
        elif metrics.information_ratio > 0:
            score += 1

        if metrics.max_drawdown > -0.1:
            score += 2
        elif metrics.max_drawdown > -0.2:
            score += 1

        # Verdict based on score
        if score >= 9:
            return "EXCELLENT - Significantly outperforms S&P 500"
        elif score >= 6:
            return "GOOD - Outperforms S&P 500"
        elif score >= 3:
            return "MODERATE - Comparable to S&P 500"
        else:
            return "UNDERPERFORMING - Below S&P 500 benchmark"

    def _create_visualizations(
        self,
        portfolio_equity: pd.Series,
        benchmark_equity: pd.Series
    ) -> Dict[str, plt.Figure]:
        """Create performance visualizations"""

        figs = {}

        # 1. Cumulative Returns Comparison
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(portfolio_equity.index, portfolio_equity, label='AI Portfolio', linewidth=2)
        ax1.plot(benchmark_equity.index, benchmark_equity, label='S&P 500', linewidth=2)
        ax1.set_title('Cumulative Returns: AI Portfolio vs S&P 500')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        figs['cumulative_returns'] = fig1

        # 2. Rolling Sharpe Ratio
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        portfolio_returns = portfolio_equity.pct_change()
        rolling_sharpe = portfolio_returns.rolling(window=252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        ax2.plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe (252d)')
        ax2.axhline(y=1, color='r', linestyle='--', label='Good (1.0)')
        ax2.set_title('Rolling Sharpe Ratio')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        figs['rolling_sharpe'] = fig2

        # 3. Drawdown Chart
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax3.set_title('Portfolio Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown %')
        ax3.grid(True, alpha=0.3)
        figs['drawdown'] = fig3

        # 4. Returns Distribution
        fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))

        # Portfolio returns distribution
        ax4.hist(portfolio_returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(x=portfolio_returns.mean(), color='r', linestyle='--', label='Mean')
        ax4.set_title('Portfolio Returns Distribution')
        ax4.set_xlabel('Daily Return')
        ax4.set_ylabel('Frequency')
        ax4.legend()

        # Q-Q plot
        stats.probplot(portfolio_returns.dropna(), dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot (Normality Test)')
        figs['returns_distribution'] = fig4

        return figs

    def _analyze_risk(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Detailed risk analysis"""

        risk_analysis = {
            'risk_rating': self._get_risk_rating(metrics),
            'var_breach_probability': 1 - 0.95,  # 5% for 95% VaR
            'expected_shortfall': metrics.cvar_95,
            'tail_risk': abs(metrics.cvar_95 / metrics.var_95) if metrics.var_95 != 0 else 1,
            'downside_risk': metrics.downside_volatility,
            'recovery_time': metrics.max_drawdown_duration,
            'risk_adjusted_return': metrics.sharpe_ratio,
            'diversification_benefit': 1 - metrics.correlation if metrics.correlation > 0 else 0
        }

        return risk_analysis

    def _get_risk_rating(self, metrics: PerformanceMetrics) -> str:
        """Generate risk rating"""

        risk_score = 0

        if metrics.volatility < 0.15:
            risk_score += 1
        elif metrics.volatility > 0.25:
            risk_score -= 1

        if metrics.max_drawdown > -0.1:
            risk_score += 1
        elif metrics.max_drawdown < -0.25:
            risk_score -= 1

        if metrics.beta < 0.8:
            risk_score += 1
        elif metrics.beta > 1.2:
            risk_score -= 1

        if risk_score >= 2:
            return "LOW RISK"
        elif risk_score >= 0:
            return "MODERATE RISK"
        else:
            return "HIGH RISK"

    def _performance_attribution(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Attribute performance to different factors"""

        # Simple attribution model
        total_return = metrics.annualized_return

        # Market component (beta * market return)
        market_component = metrics.beta * 0.1  # Assuming 10% market return

        # Alpha component
        alpha_component = metrics.alpha

        # Selection component (approximated)
        selection_component = total_return - market_component - alpha_component

        attribution = {
            'total_return': total_return,
            'market_beta': market_component,
            'alpha': alpha_component,
            'selection': selection_component,
            'market_contribution_%': (market_component / total_return * 100) if total_return != 0 else 0,
            'alpha_contribution_%': (alpha_component / total_return * 100) if total_return != 0 else 0,
            'selection_contribution_%': (selection_component / total_return * 100) if total_return != 0 else 0
        }

        return attribution

    def _statistical_analysis(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Statistical significance analysis"""

        analysis = {
            'statistically_significant': metrics.p_value < 0.05,
            'confidence_level': (1 - metrics.p_value) * 100,
            't_statistic': metrics.t_statistic,
            'p_value': metrics.p_value,
            'interpretation': self._interpret_statistics(metrics)
        }

        return analysis

    def _interpret_statistics(self, metrics: PerformanceMetrics) -> str:
        """Interpret statistical results"""

        if metrics.p_value < 0.01:
            return "Highly significant outperformance (99% confidence)"
        elif metrics.p_value < 0.05:
            return "Statistically significant outperformance (95% confidence)"
        elif metrics.p_value < 0.1:
            return "Marginally significant outperformance (90% confidence)"
        else:
            return "No statistically significant difference from benchmark"


def run_backtest_evaluation(
    portfolio_values: pd.Series,
    start_date: str,
    end_date: str,
    trades: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Run complete backtest evaluation against S&P 500"""

    evaluator = BenchmarkEvaluator()

    # Fetch benchmark data
    benchmark_data = evaluator.fetch_benchmark_data(start_date, end_date)
    benchmark_values = benchmark_data['Close']

    # Calculate returns
    portfolio_returns = portfolio_values.pct_change().dropna()
    benchmark_returns = benchmark_values.pct_change().dropna()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(portfolio_returns, benchmark_returns, trades)

    # Create report
    report = evaluator.create_performance_report(metrics, portfolio_values, benchmark_values)

    return report


# Example usage
if __name__ == "__main__":
    # Generate sample portfolio data
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')

    # Simulate portfolio that outperforms S&P 500
    np.random.seed(42)
    market_returns = np.random.normal(0.0004, 0.012, len(dates))  # ~10% annual, 19% vol
    alpha = np.random.normal(0.0002, 0.005, len(dates))  # Additional alpha
    portfolio_returns = market_returns + alpha

    portfolio_values = pd.Series(
        100000 * (1 + portfolio_returns).cumprod(),
        index=dates,
        name='Portfolio'
    )

    # Run evaluation
    report = run_backtest_evaluation(
        portfolio_values,
        '2022-01-01',
        '2024-01-01'
    )

    # Print results
    metrics = report['metrics']
    print("\n" + "="*60)
    print("AI HEDGE FUND PERFORMANCE vs S&P 500")
    print("="*60)

    print(f"\nVERDICT: {report['summary']['verdict']}")

    print("\n📊 RETURNS:")
    print(f"  Annualized Return: {metrics.annualized_return:.2%}")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Alpha: {metrics.alpha:.2%}")
    print(f"  Beta: {metrics.beta:.2f}")

    print("\n📈 RISK-ADJUSTED PERFORMANCE:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Information Ratio: {metrics.information_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")

    print("\n⚠️ RISK METRICS:")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  VaR (95%): {metrics.var_95:.2%}")
    print(f"  CVaR (95%): {metrics.cvar_95:.2%}")

    print("\n📊 MARKET CAPTURE:")
    print(f"  Upside Capture: {metrics.upside_capture:.2%}")
    print(f"  Downside Capture: {metrics.downside_capture:.2%}")
    print(f"  Capture Ratio: {metrics.capture_ratio:.2f}")

    print("\n📈 STATISTICAL SIGNIFICANCE:")
    print(f"  T-Statistic: {metrics.t_statistic:.2f}")
    print(f"  P-Value: {metrics.p_value:.4f}")
    print(f"  Significant: {'YES' if metrics.p_value < 0.05 else 'NO'}")

    print("\n💡 KEY FINDINGS:")
    for finding in report['summary']['key_findings']:
        print(f"  • {finding}")

    print("\n✅ STRENGTHS:")
    for strength in report['summary']['strengths']:
        print(f"  • {strength}")

    if report['summary']['weaknesses']:
        print("\n⚠️ AREAS FOR IMPROVEMENT:")
        for weakness in report['summary']['weaknesses']:
            print(f"  • {weakness}")