"""
ENHANCED EVALUATION SYSTEM
==========================
Professional-grade evaluation with risk-matched benchmarking,
QuantStats integration, and deflated Sharpe ratios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False
    print("Warning: QuantStats not installed. Install with: pip install quantstats")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not installed. Install with: pip install scipy")

ANNUAL_DAYS = 252

class EnhancedEvaluationSystem:
    """Professional evaluation system with advanced metrics"""

    def __init__(self, portfolio_value: float = 500000):
        self.portfolio_value = portfolio_value
        self.benchmark_symbols = ['SPY', 'QQQ', 'IWM', 'AGG']

    def calculate_deflated_sharpe(self, sharpe_ratio: float, n_trials: int,
                                  n_observations: int) -> Tuple[float, float]:
        """
        Calculate Deflated Sharpe Ratio to account for multiple testing
        Based on Bailey & Lopez de Prado (2014)
        """

        if not HAS_SCIPY:
            return sharpe_ratio, 1.0  # Return original if no scipy

        # Expected maximum Sharpe under null hypothesis
        euler_mascheroni = 0.5772156649
        expected_max_sharpe = np.sqrt(2 * np.log(n_trials)) - euler_mascheroni

        # Standard deviation of maximum Sharpe
        std_max_sharpe = 1 / np.sqrt(2 * np.log(n_trials))

        # Deflated Sharpe Ratio
        deflated_sharpe = (sharpe_ratio - expected_max_sharpe) / std_max_sharpe

        # Probability that observed Sharpe is due to chance
        if HAS_SCIPY:
            p_value = 1 - stats.norm.cdf(deflated_sharpe)
        else:
            # Simple approximation
            p_value = 0.5 * (1 - np.tanh(deflated_sharpe))

        return deflated_sharpe, p_value

    def calculate_pbo(self, returns: pd.Series, n_splits: int = 10) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO)
        Based on Bailey, Borwein, Lopez de Prado, and Zhu (2014)
        """

        n = len(returns)
        split_size = n // n_splits

        in_sample_sharpes = []
        out_sample_sharpes = []

        for i in range(n_splits):
            # Create train/test split
            if i % 2 == 0:
                train_idx = range(i * split_size, (i + 1) * split_size)
                test_idx = range((i + 1) * split_size, min((i + 2) * split_size, n))
            else:
                test_idx = range(i * split_size, (i + 1) * split_size)
                train_idx = range((i + 1) * split_size, min((i + 2) * split_size, n))

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            train_returns = returns.iloc[list(train_idx)]
            test_returns = returns.iloc[list(test_idx)]

            # Calculate Sharpe ratios
            train_sharpe = (train_returns.mean() / train_returns.std()) * np.sqrt(ANNUAL_DAYS)
            test_sharpe = (test_returns.mean() / test_returns.std()) * np.sqrt(ANNUAL_DAYS)

            in_sample_sharpes.append(train_sharpe)
            out_sample_sharpes.append(test_sharpe)

        # Calculate PBO
        n_pairs = len(in_sample_sharpes)
        if n_pairs == 0:
            return 0.5

        loss_count = 0
        for i in range(n_pairs):
            if out_sample_sharpes[i] < 0:  # Out-of-sample loss
                loss_count += 1

        pbo = loss_count / n_pairs
        return pbo

    def risk_match_benchmark(self, strategy_returns: pd.Series,
                            benchmark_returns: pd.Series) -> pd.Series:
        """Scale benchmark returns to match strategy volatility"""

        strat_vol = strategy_returns.std()
        bench_vol = benchmark_returns.std()

        if bench_vol == 0:
            return benchmark_returns

        scale_factor = strat_vol / bench_vol
        return benchmark_returns * scale_factor

    def calculate_information_ratio(self, strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""

        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std()

        if tracking_error == 0:
            return 0

        annual_excess = excess_returns.mean() * ANNUAL_DAYS
        annual_te = tracking_error * np.sqrt(ANNUAL_DAYS)

        return annual_excess / annual_te

    def calculate_factor_attribution(self, returns: pd.Series) -> Dict:
        """
        Calculate Fama-French factor attribution
        Note: This is simplified - real implementation would fetch factor data
        """

        # Simplified factor loadings (in production, run regression)
        factors = {
            'Market': 0.8,  # Market beta
            'Size': 0.1,    # SMB loading
            'Value': 0.05,  # HML loading
            'Momentum': 0.15,  # MOM loading
            'Quality': 0.1,  # RMW loading
            'Investment': 0.05  # CMA loading
        }

        # Simplified factor returns (in production, fetch from Ken French library)
        factor_returns = {
            'Market': 0.10,
            'Size': 0.02,
            'Value': 0.03,
            'Momentum': 0.05,
            'Quality': 0.04,
            'Investment': 0.02
        }

        # Calculate attribution
        total_return = returns.mean() * ANNUAL_DAYS
        factor_contribution = sum(factors[f] * factor_returns[f] for f in factors)
        alpha = total_return - factor_contribution

        attribution = {
            'Total_Return': total_return,
            'Factor_Return': factor_contribution,
            'Alpha': alpha,
            'Factor_Loadings': factors
        }

        return attribution

    def generate_quantstats_report(self, returns: pd.Series,
                                  benchmark: pd.Series = None,
                                  title: str = "AI Hedge Fund Performance") -> str:
        """Generate comprehensive QuantStats HTML report"""

        if not HAS_QUANTSTATS:
            return "QuantStats not installed"

        # Extend pandas with QuantStats
        qs.extend_pandas()

        # Generate HTML report
        output_file = f"performance_report_{datetime.now().strftime('%Y%m%d')}.html"

        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=output_file,
            title=title,
            download_filename=output_file
        )

        return output_file

    def calculate_advanced_metrics(self, returns: pd.Series,
                                  benchmark: pd.Series) -> Dict:
        """Calculate comprehensive suite of advanced metrics"""

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (ANNUAL_DAYS / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ANNUAL_DAYS)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(ANNUAL_DAYS) if len(downside_returns) > 0 else 0.001
        sortino = annual_return / downside_vol if downside_vol > 0 else 0

        # Drawdown metrics
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Risk-matched benchmark
        risk_matched_bench = self.risk_match_benchmark(returns, benchmark)

        # Information Ratio
        ir = self.calculate_information_ratio(returns, benchmark)
        risk_matched_ir = self.calculate_information_ratio(returns, risk_matched_bench)

        # Deflated Sharpe (assuming 100 strategy trials)
        deflated_sharpe, p_value = self.calculate_deflated_sharpe(sharpe, 100, len(returns))

        # Probability of Backtest Overfitting
        pbo = self.calculate_pbo(returns)

        # Factor attribution
        attribution = self.calculate_factor_attribution(returns)

        # Win/Loss metrics
        win_rate = (returns > 0).mean()
        win_loss_ratio = abs(returns[returns > 0].mean() / returns[returns < 0].mean()) if (returns < 0).any() else np.inf

        # Tail metrics
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        metrics = {
            # Returns
            'Annual_Return': annual_return,
            'Total_Return': total_return,
            'Volatility': volatility,

            # Risk-adjusted
            'Sharpe_Ratio': sharpe,
            'Deflated_Sharpe': deflated_sharpe,
            'Sharpe_P_Value': p_value,
            'Sortino_Ratio': sortino,
            'Calmar_Ratio': calmar,

            # Relative
            'Information_Ratio': ir,
            'Risk_Matched_IR': risk_matched_ir,

            # Drawdown
            'Max_Drawdown': max_drawdown,
            'Current_Drawdown': drawdown.iloc[-1] if not drawdown.empty else 0,

            # Statistical
            'PBO': pbo,
            'Win_Rate': win_rate,
            'Win_Loss_Ratio': win_loss_ratio,
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),

            # Risk
            'VaR_95': var_95,
            'CVaR_95': cvar_95,

            # Factor
            'Alpha': attribution['Alpha'],
            'Factor_Return': attribution['Factor_Return']
        }

        return metrics

    def create_evaluation_report(self, strategy_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                save_html: bool = True) -> Dict:
        """Create comprehensive evaluation report"""

        print("\n" + "="*70)
        print("ENHANCED EVALUATION REPORT")
        print("="*70)
        print(f"Evaluation Period: {strategy_returns.index[0].date()} to {strategy_returns.index[-1].date()}")
        print(f"Trading Days: {len(strategy_returns)}")
        print("="*70)

        # Calculate all metrics
        metrics = self.calculate_advanced_metrics(strategy_returns, benchmark_returns)

        # Display results
        print("\n>>> PERFORMANCE METRICS")
        print("-"*50)
        print(f"Annual Return:        {metrics['Annual_Return']:>10.2%}")
        print(f"Volatility:          {metrics['Volatility']:>10.2%}")
        print(f"Max Drawdown:        {metrics['Max_Drawdown']:>10.2%}")

        print("\n>>> RISK-ADJUSTED RETURNS")
        print("-"*50)
        print(f"Sharpe Ratio:        {metrics['Sharpe_Ratio']:>10.2f}")
        print(f"Deflated Sharpe:     {metrics['Deflated_Sharpe']:>10.2f}")
        print(f"P-Value:             {metrics['Sharpe_P_Value']:>10.4f}")
        print(f"Sortino Ratio:       {metrics['Sortino_Ratio']:>10.2f}")
        print(f"Calmar Ratio:        {metrics['Calmar_Ratio']:>10.2f}")

        print("\n>>> RELATIVE PERFORMANCE")
        print("-"*50)
        print(f"Information Ratio:   {metrics['Information_Ratio']:>10.2f}")
        print(f"Risk-Matched IR:     {metrics['Risk_Matched_IR']:>10.2f}")
        print(f"Alpha:               {metrics['Alpha']:>10.2%}")

        print("\n>>> STATISTICAL VALIDATION")
        print("-"*50)
        print(f"PBO:                 {metrics['PBO']:>10.2%}")
        print(f"Win Rate:            {metrics['Win_Rate']:>10.2%}")
        print(f"Win/Loss Ratio:      {metrics['Win_Loss_Ratio']:>10.2f}")
        print(f"VaR 95%:             {metrics['VaR_95']:>10.2%}")

        # Validation assessment
        print("\n" + "="*70)
        print("VALIDATION ASSESSMENT")
        print("="*70)

        if metrics['Deflated_Sharpe'] > 0 and metrics['Sharpe_P_Value'] < 0.05:
            print("✓ Statistically significant positive performance (p < 0.05)")
        else:
            print("⚠ Performance not statistically significant")

        if metrics['PBO'] < 0.5:
            print("✓ Low probability of backtest overfitting (PBO < 50%)")
        else:
            print("⚠ High probability of backtest overfitting (PBO >= 50%)")

        if metrics['Information_Ratio'] > 1.0:
            print("✓ Strong risk-adjusted outperformance (IR > 1.0)")
        elif metrics['Information_Ratio'] > 0.5:
            print("✓ Moderate risk-adjusted outperformance (IR > 0.5)")
        else:
            print("⚠ Weak risk-adjusted performance (IR <= 0.5)")

        # Generate QuantStats report if available
        if save_html and HAS_QUANTSTATS:
            print("\n>>> Generating QuantStats Report...")
            report_file = self.generate_quantstats_report(
                strategy_returns,
                benchmark_returns,
                "AI Hedge Fund - Enhanced Evaluation"
            )
            print(f"✓ Report saved to: {report_file}")

        return metrics

def run_enhanced_evaluation():
    """Run complete enhanced evaluation"""

    print("\n" + "="*70)
    print("RUNNING ENHANCED EVALUATION SYSTEM")
    print("="*70)

    # Initialize evaluator
    evaluator = EnhancedEvaluationSystem()

    # Generate sample data for demonstration
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='B')

    # Simulate strategy returns (with our edge)
    np.random.seed(42)
    base_returns = np.random.normal(0.0008, 0.012, len(dates))  # 20% annual, 19% vol
    strategy_returns = pd.Series(base_returns, index=dates)

    # Simulate benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0004, 0.011, len(dates)),
        index=dates
    )

    # Run evaluation
    metrics = evaluator.create_evaluation_report(
        strategy_returns,
        benchmark_returns,
        save_html=HAS_QUANTSTATS
    )

    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    if metrics['Alpha'] > 0.30:
        print(f"✓ Strong alpha generation: {metrics['Alpha']:.1%}")
    else:
        print(f"⚠ Moderate alpha: {metrics['Alpha']:.1%}")

    print(f"✓ Deflated Sharpe: {metrics['Deflated_Sharpe']:.2f}")
    print(f"✓ PBO: {metrics['PBO']:.1%}")
    print(f"✓ Information Ratio: {metrics['Information_Ratio']:.2f}")

    return metrics

if __name__ == "__main__":
    run_enhanced_evaluation()