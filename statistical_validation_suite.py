"""
STATISTICAL VALIDATION SUITE
============================
Implements MinTRL, Block Bootstrap CIs, and improved PBO calculation
Addresses remaining statistical vulnerabilities
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidationSuite:
    """Advanced statistical validation for trading strategies"""

    def __init__(self):
        self.confidence_level = 0.95
        self.block_size = 21  # Trading days (approximately 1 month)

    def calculate_mintrl(self, sharpe_ratio: float, target_confidence: float = 0.95,
                        skewness: float = 0, kurtosis: float = 0) -> float:
        """
        Calculate Minimum Track Record Length (MinTRL)
        Based on Bailey & Lopez de Prado (2012)

        Returns: Number of months needed for statistical confidence
        """

        # Adjust for non-normality
        if skewness != 0 or kurtosis != 0:
            # Cornish-Fisher expansion for non-normal distributions
            z_cf = self._cornish_fisher_quantile(target_confidence, skewness, kurtosis)
        else:
            z_cf = stats.norm.ppf(target_confidence)

        # MinTRL formula (in years)
        mintrl_years = (z_cf / sharpe_ratio) ** 2

        # Convert to months
        mintrl_months = mintrl_years * 12

        return mintrl_months

    def _cornish_fisher_quantile(self, p: float, skew: float, kurt: float) -> float:
        """Cornish-Fisher expansion for non-normal quantiles"""

        z = stats.norm.ppf(p)

        # Cornish-Fisher expansion terms
        cf_adjustment = (
            z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * (kurt - 3) / 24 -
            (2*z**3 - 5*z) * skew**2 / 36
        )

        return cf_adjustment

    def block_bootstrap_confidence_intervals(self, returns: pd.Series,
                                           n_bootstrap: int = 10000,
                                           metrics: List[str] = None) -> Dict:
        """
        Calculate block bootstrap confidence intervals
        Preserves autocorrelation structure
        """

        if metrics is None:
            metrics = ['sharpe', 'sortino', 'max_drawdown', 'calmar']

        n = len(returns)
        n_blocks = n // self.block_size

        bootstrap_results = {metric: [] for metric in metrics}

        for _ in range(n_bootstrap):
            # Create bootstrap sample using blocks
            bootstrap_sample = []

            for _ in range(n_blocks):
                # Random block start
                start_idx = np.random.randint(0, n - self.block_size + 1)
                block = returns.iloc[start_idx:start_idx + self.block_size]
                bootstrap_sample.extend(block.values)

            bootstrap_returns = pd.Series(bootstrap_sample[:n])

            # Calculate metrics for this bootstrap sample
            if 'sharpe' in metrics:
                sharpe = (bootstrap_returns.mean() / bootstrap_returns.std()) * np.sqrt(252)
                bootstrap_results['sharpe'].append(sharpe)

            if 'sortino' in metrics:
                downside = bootstrap_returns[bootstrap_returns < 0].std()
                sortino = (bootstrap_returns.mean() / downside) * np.sqrt(252) if downside > 0 else 0
                bootstrap_results['sortino'].append(sortino)

            if 'max_drawdown' in metrics:
                cum_returns = (1 + bootstrap_returns).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (cum_returns - running_max) / running_max
                max_dd = drawdown.min()
                bootstrap_results['max_drawdown'].append(max_dd)

            if 'calmar' in metrics:
                annual_return = (1 + bootstrap_returns).prod() ** (252/len(bootstrap_returns)) - 1
                max_dd = bootstrap_results['max_drawdown'][-1] if 'max_drawdown' in bootstrap_results else 0
                calmar = annual_return / abs(max_dd) if max_dd < 0 else 0
                bootstrap_results['calmar'].append(calmar)

        # Calculate confidence intervals
        confidence_intervals = {}

        for metric, values in bootstrap_results.items():
            values_array = np.array(values)
            lower = np.percentile(values_array, (1 - self.confidence_level) / 2 * 100)
            upper = np.percentile(values_array, (1 + self.confidence_level) / 2 * 100)
            median = np.median(values_array)

            confidence_intervals[metric] = {
                'lower_95': lower,
                'median': median,
                'upper_95': upper,
                'std_error': np.std(values_array)
            }

        return confidence_intervals

    def improved_pbo_calculation(self, returns: pd.Series,
                                 n_splits: int = 16,
                                 model_complexity: int = 1) -> Dict:
        """
        Improved Probability of Backtest Overfitting
        Accounts for model complexity and uses more splits
        """

        n = len(returns)
        split_size = n // n_splits

        # Track in-sample vs out-of-sample performance
        is_performance = []
        oos_performance = []

        for split in range(n_splits - 1):
            # Define train and test periods
            if split % 2 == 0:
                train_start = split * split_size
                train_end = (split + 1) * split_size
                test_start = train_end
                test_end = min((split + 2) * split_size, n)
            else:
                test_start = split * split_size
                test_end = (split + 1) * split_size
                train_start = test_end
                train_end = min((split + 2) * split_size, n)

            if train_end > n or test_end > n:
                continue

            train_returns = returns.iloc[train_start:train_end]
            test_returns = returns.iloc[test_start:test_end]

            # Calculate Sharpe ratios
            train_sharpe = (train_returns.mean() / train_returns.std()) * np.sqrt(252)
            test_sharpe = (test_returns.mean() / test_returns.std()) * np.sqrt(252)

            is_performance.append(train_sharpe)
            oos_performance.append(test_sharpe)

        # Calculate PBO
        n_pairs = len(is_performance)
        if n_pairs == 0:
            return {'pbo': 0.5, 'n_pairs': 0}

        # Count how often out-of-sample underperforms
        underperform_count = sum(1 for i in range(n_pairs) if oos_performance[i] < 0)

        # Adjust for model complexity
        # More complex models get penalty
        complexity_penalty = 0.05 * np.log(model_complexity + 1)

        pbo = (underperform_count / n_pairs) + complexity_penalty

        # Calculate additional metrics
        is_mean = np.mean(is_performance)
        oos_mean = np.mean(oos_performance)
        degradation = (is_mean - oos_mean) / is_mean if is_mean != 0 else 0

        return {
            'pbo': min(pbo, 1.0),
            'n_pairs': n_pairs,
            'is_sharpe_mean': is_mean,
            'oos_sharpe_mean': oos_mean,
            'performance_degradation': degradation,
            'complexity_penalty': complexity_penalty
        }

    def calculate_factor_neutral_alpha(self, returns: pd.Series,
                                      factor_returns: Dict[str, pd.Series]) -> Dict:
        """
        Calculate alpha after removing factor exposures
        Uses Fama-French 6-factor model
        """

        # Align all series
        aligned_data = pd.DataFrame({'strategy': returns})

        for factor_name, factor_series in factor_returns.items():
            aligned_data[factor_name] = factor_series

        aligned_data = aligned_data.dropna()

        if len(aligned_data) < 60:  # Need sufficient data
            return {'error': 'Insufficient data for factor regression'}

        # Run regression: Strategy returns ~ Factors
        from sklearn.linear_model import LinearRegression

        X = aligned_data[list(factor_returns.keys())]
        y = aligned_data['strategy']

        model = LinearRegression()
        model.fit(X, y)

        # Calculate residuals (alpha)
        predicted = model.predict(X)
        residuals = y - predicted

        # Annualized alpha
        alpha_daily = residuals.mean()
        alpha_annual = alpha_daily * 252

        # Residual risk
        residual_vol = residuals.std() * np.sqrt(252)

        # Information Ratio of residuals
        residual_ir = alpha_annual / residual_vol if residual_vol > 0 else 0

        # Factor loadings
        loadings = dict(zip(factor_returns.keys(), model.coef_))

        # R-squared (how much variance is explained by factors)
        r_squared = model.score(X, y)

        return {
            'alpha_annual': alpha_annual,
            'residual_volatility': residual_vol,
            'residual_ir': residual_ir,
            'factor_loadings': loadings,
            'r_squared': r_squared,
            'unexplained_variance': 1 - r_squared
        }

    def generate_blue_sheet(self, returns: pd.Series,
                           benchmark_returns: pd.Series,
                           signals_metadata: Dict) -> Dict:
        """
        Generate comprehensive one-page blue sheet with all metrics
        """

        # Basic performance metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252/len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Calculate skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # MinTRL
        mintrl = self.calculate_mintrl(sharpe, skewness=skewness, kurtosis=kurtosis)

        # Bootstrap confidence intervals
        ci_results = self.block_bootstrap_confidence_intervals(returns)

        # Improved PBO
        pbo_results = self.improved_pbo_calculation(returns, n_splits=16)

        # Create factor returns (simplified - in production, use real data)
        factor_returns = {
            'MKT': benchmark_returns - 0.0001,  # Market minus risk-free
            'SMB': pd.Series(np.random.normal(0.0001, 0.005, len(returns)), index=returns.index),
            'HML': pd.Series(np.random.normal(0.0001, 0.004, len(returns)), index=returns.index),
            'MOM': pd.Series(np.random.normal(0.0002, 0.006, len(returns)), index=returns.index),
            'RMW': pd.Series(np.random.normal(0.0001, 0.003, len(returns)), index=returns.index),
            'CMA': pd.Series(np.random.normal(0.0001, 0.003, len(returns)), index=returns.index)
        }

        # Factor-neutral alpha
        factor_analysis = self.calculate_factor_neutral_alpha(returns, factor_returns)

        # Compile blue sheet
        blue_sheet = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'period_start': returns.index[0].isoformat() if hasattr(returns.index[0], 'isoformat') else str(returns.index[0]),
                'period_end': returns.index[-1].isoformat() if hasattr(returns.index[-1], 'isoformat') else str(returns.index[-1]),
                'n_days': len(returns),
                'confidence_level': self.confidence_level
            },
            'performance': {
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'statistical_validation': {
                'mintrl_months': mintrl,
                'pbo': pbo_results['pbo'],
                'pbo_details': pbo_results
            },
            'confidence_intervals': ci_results,
            'factor_analysis': factor_analysis,
            'signals_metadata': signals_metadata
        }

        return blue_sheet

    def format_blue_sheet_markdown(self, blue_sheet: Dict) -> str:
        """Format blue sheet as markdown for easy reading"""

        md = "# TRADING STRATEGY BLUE SHEET\n\n"
        md += f"Generated: {blue_sheet['metadata']['generated']}\n"
        md += f"Period: {blue_sheet['metadata']['period_start']} to {blue_sheet['metadata']['period_end']}\n"
        md += f"Days: {blue_sheet['metadata']['n_days']}\n\n"

        md += "## Performance Summary\n"
        perf = blue_sheet['performance']
        md += f"- **Annual Return**: {perf['annual_return']:.2%}\n"
        md += f"- **Volatility**: {perf['volatility']:.2%}\n"
        md += f"- **Sharpe Ratio**: {perf['sharpe_ratio']:.2f}\n"
        md += f"- **Skewness**: {perf['skewness']:.2f}\n"
        md += f"- **Kurtosis**: {perf['kurtosis']:.2f}\n\n"

        md += "## Statistical Validation\n"
        stats = blue_sheet['statistical_validation']
        md += f"- **MinTRL**: {stats['mintrl_months']:.1f} months\n"
        md += f"- **PBO**: {stats['pbo']:.1%}\n"
        md += f"- **IS/OOS Degradation**: {stats['pbo_details']['performance_degradation']:.1%}\n\n"

        md += "## 95% Confidence Intervals (Block Bootstrap)\n"
        ci = blue_sheet['confidence_intervals']
        for metric, values in ci.items():
            md += f"- **{metric}**: [{values['lower_95']:.2f}, {values['upper_95']:.2f}]\n"
        md += "\n"

        md += "## Factor Analysis\n"
        factor = blue_sheet['factor_analysis']
        if 'error' not in factor:
            md += f"- **Residual Alpha**: {factor['alpha_annual']:.2%} p.a.\n"
            md += f"- **Residual IR**: {factor['residual_ir']:.2f}\n"
            md += f"- **R² (Factor)**: {factor['r_squared']:.2%}\n"
            md += f"- **Unexplained**: {factor['unexplained_variance']:.2%}\n\n"

            md += "### Factor Loadings\n"
            for fname, loading in factor['factor_loadings'].items():
                md += f"- **{fname}**: {loading:.3f}\n"

        return md


def demonstrate_validation_suite():
    """Demonstrate the statistical validation suite"""

    print("\n" + "="*70)
    print("STATISTICAL VALIDATION SUITE DEMONSTRATION")
    print("="*70)

    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='B')

    # Simulate strategy with edge
    returns = pd.Series(np.random.normal(0.001, 0.012, 500), index=dates)
    benchmark = pd.Series(np.random.normal(0.0004, 0.010, 500), index=dates)

    # Initialize validator
    validator = StatisticalValidationSuite()

    # Calculate MinTRL
    sharpe = 2.0
    mintrl = validator.calculate_mintrl(sharpe, skewness=-0.5, kurtosis=3.5)
    print(f"\nMinTRL for Sharpe={sharpe:.2f}: {mintrl:.1f} months")

    # Calculate bootstrap CIs
    print("\nCalculating bootstrap confidence intervals...")
    cis = validator.block_bootstrap_confidence_intervals(returns, n_bootstrap=1000)

    print("\n95% Confidence Intervals:")
    for metric, values in cis.items():
        print(f"{metric:15s}: [{values['lower_95']:>7.2f}, {values['upper_95']:>7.2f}]")

    # Calculate improved PBO
    pbo_results = validator.improved_pbo_calculation(returns)
    print(f"\nImproved PBO: {pbo_results['pbo']:.1%}")
    print(f"IS/OOS Degradation: {pbo_results['performance_degradation']:.1%}")

    # Generate blue sheet
    signals_metadata = {
        'effective_breadth': 3.2,
        'sources': 6,
        'avg_lag_days': 14
    }

    blue_sheet = validator.generate_blue_sheet(returns, benchmark, signals_metadata)

    # Save blue sheet
    with open('blue_sheet.json', 'w') as f:
        json.dump(blue_sheet, f, indent=2, default=str)

    # Generate markdown
    md = validator.format_blue_sheet_markdown(blue_sheet)
    with open('blue_sheet.md', 'w') as f:
        f.write(md)

    print("\n✓ Blue sheet saved to blue_sheet.json and blue_sheet.md")

    # Final assessment
    print("\n" + "="*70)
    print("VALIDATION ASSESSMENT")
    print("="*70)

    if pbo_results['pbo'] < 0.3:
        print("✓ PBO < 30% - Low overfitting risk")
    else:
        print("⚠ PBO >= 30% - Moderate overfitting risk")

    if mintrl < 36:
        print(f"✓ MinTRL < 36 months - Reasonable validation period")
    else:
        print(f"⚠ MinTRL = {mintrl:.0f} months - Long validation needed")

    return validator


if __name__ == "__main__":
    demonstrate_validation_suite()