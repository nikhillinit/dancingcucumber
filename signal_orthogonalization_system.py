"""
SIGNAL ORTHOGONALIZATION & EFFECTIVE BREADTH SYSTEM
===================================================
Ensures multiple intelligence sources provide independent value
Implements Fundamental Law of Active Management principles
"""

import numpy as np
import pandas as pd
from scipy import linalg
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SignalOrthogonalizationSystem:
    """Orthogonalize signals and calculate effective breadth"""

    def __init__(self):
        self.signal_sources = [
            'congressional',
            'form4',
            'sec_filings',
            'fed_speeches',
            'options_flow',
            'earnings_calls'
        ]

    def calculate_signal_correlation(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between signal sources"""

        # Calculate returns for each signal
        signal_returns = {}

        for source in self.signal_sources:
            if source in signals_df.columns:
                # Calculate signal returns (next day return given signal)
                signal_returns[source] = signals_df[source].shift(1)

        # Create correlation matrix
        returns_df = pd.DataFrame(signal_returns)
        correlation_matrix = returns_df.corr()

        return correlation_matrix

    def orthogonalize_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Orthogonalize signals using Gram-Schmidt process
        Ensures each signal provides independent information
        """

        # Convert to numpy array
        signals_array = signals_df[self.signal_sources].values

        # Remove NaN rows
        valid_mask = ~np.isnan(signals_array).any(axis=1)
        clean_signals = signals_array[valid_mask]

        if len(clean_signals) < 2:
            return signals_df

        # Gram-Schmidt orthogonalization
        orthogonal_signals = np.zeros_like(clean_signals)

        for i in range(clean_signals.shape[1]):
            # Start with original signal
            orthogonal_signals[:, i] = clean_signals[:, i]

            # Remove projections onto previous orthogonal signals
            for j in range(i):
                projection = np.dot(clean_signals[:, i], orthogonal_signals[:, j])
                projection /= np.dot(orthogonal_signals[:, j], orthogonal_signals[:, j]) + 1e-10
                orthogonal_signals[:, i] -= projection * orthogonal_signals[:, j]

            # Normalize
            norm = np.linalg.norm(orthogonal_signals[:, i])
            if norm > 0:
                orthogonal_signals[:, i] /= norm

        # Create output dataframe
        orthogonal_df = pd.DataFrame(
            orthogonal_signals,
            columns=[f"{s}_orthogonal" for s in self.signal_sources],
            index=signals_df.index[valid_mask]
        )

        return orthogonal_df

    def calculate_effective_breadth(self, signals_df: pd.DataFrame,
                                   returns: pd.Series) -> Dict:
        """
        Calculate Effective Number of Bets (ENB)
        Based on Fundamental Law: IR ≈ IC × √(Breadth)
        """

        # Calculate correlation matrix
        corr_matrix = self.calculate_signal_correlation(signals_df)

        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(corr_matrix.fillna(0))
        eigenvalues = eigenvalues[eigenvalues > 0]  # Keep only positive

        # Method 1: Effective Number of Bets (ENB)
        # Based on eigenvalue decomposition
        if len(eigenvalues) > 0:
            enb = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        else:
            enb = 1

        # Method 2: Diversification Ratio
        # Sum of volatilities / Portfolio volatility
        signal_vols = signals_df[self.signal_sources].std()
        portfolio_vol = signals_df[self.signal_sources].mean(axis=1).std()

        if portfolio_vol > 0:
            diversification_ratio = signal_vols.sum() / (len(signal_vols) * portfolio_vol)
        else:
            diversification_ratio = 1

        # Method 3: Principal Component Analysis
        # How many components explain 95% of variance
        if len(eigenvalues) > 0:
            eigenvalues_sorted = np.sort(eigenvalues)[::-1]
            cumsum = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted)
            n_components_95 = np.argmax(cumsum >= 0.95) + 1
        else:
            n_components_95 = 1

        # Information Coefficient (IC) for each source
        ic_scores = {}
        for source in self.signal_sources:
            if source in signals_df.columns:
                # Correlation between signal and forward returns
                signal = signals_df[source].shift(1)  # Lagged signal
                valid_mask = ~(signal.isna() | returns.isna())

                if valid_mask.sum() > 10:
                    ic = signal[valid_mask].corr(returns[valid_mask])
                    ic_scores[source] = ic
                else:
                    ic_scores[source] = 0

        # Transfer Coefficient (TC)
        # How well we can implement the signals
        tc = 0.8  # Assume 80% implementation efficiency

        # Calculate expected Information Ratio
        mean_ic = np.mean(list(ic_scores.values()))
        expected_ir = mean_ic * np.sqrt(enb) * tc

        results = {
            'effective_breadth': enb,
            'diversification_ratio': diversification_ratio,
            'n_components_95': n_components_95,
            'correlation_matrix': corr_matrix,
            'eigenvalues': eigenvalues,
            'ic_scores': ic_scores,
            'mean_ic': mean_ic,
            'transfer_coefficient': tc,
            'expected_ir': expected_ir,
            'raw_breadth': len(self.signal_sources)
        }

        return results

    def perform_ablation_study(self, signals_df: pd.DataFrame,
                              returns: pd.Series) -> Dict:
        """
        Ablation study: Remove each source and measure impact
        Shows marginal contribution of each source
        """

        # Baseline with all signals
        all_signals = signals_df[self.signal_sources].mean(axis=1)
        baseline_ic = all_signals.shift(1).corr(returns)

        ablation_results = {}

        for source_to_remove in self.signal_sources:
            # Create signal without this source
            remaining_sources = [s for s in self.signal_sources if s != source_to_remove]

            if remaining_sources:
                reduced_signal = signals_df[remaining_sources].mean(axis=1)
                reduced_ic = reduced_signal.shift(1).corr(returns)

                # Calculate impact
                ic_loss = baseline_ic - reduced_ic
                percentage_impact = (ic_loss / baseline_ic) * 100 if baseline_ic != 0 else 0

                ablation_results[source_to_remove] = {
                    'ic_without': reduced_ic,
                    'ic_loss': ic_loss,
                    'percentage_impact': percentage_impact
                }

        return {
            'baseline_ic': baseline_ic,
            'ablation_results': ablation_results
        }

    def calculate_signal_independence(self, signals_df: pd.DataFrame) -> Dict:
        """
        Calculate how independent each signal source is
        Using Variance Inflation Factor (VIF) and uniqueness scores
        """

        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Prepare data
        signals_clean = signals_df[self.signal_sources].dropna()

        if len(signals_clean) < 10:
            return {}

        # Calculate VIF for each signal
        vif_scores = {}
        for i, source in enumerate(self.signal_sources):
            if source in signals_clean.columns:
                vif = variance_inflation_factor(signals_clean.values, i)
                vif_scores[source] = vif

        # Calculate uniqueness (1 - R² from regressing on other signals)
        uniqueness_scores = {}

        for target in self.signal_sources:
            if target in signals_clean.columns:
                # Regress target on all other signals
                X = signals_clean.drop(columns=[target])
                y = signals_clean[target]

                # Simple OLS R²
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()

                try:
                    model.fit(X, y)
                    r_squared = model.score(X, y)
                    uniqueness = 1 - r_squared
                    uniqueness_scores[target] = uniqueness
                except:
                    uniqueness_scores[target] = 1.0

        return {
            'vif_scores': vif_scores,
            'uniqueness_scores': uniqueness_scores,
            'average_vif': np.mean(list(vif_scores.values())),
            'average_uniqueness': np.mean(list(uniqueness_scores.values()))
        }

    def generate_independence_report(self, signals_df: pd.DataFrame,
                                    returns: pd.Series) -> None:
        """Generate comprehensive independence report"""

        print("\n" + "="*70)
        print("SIGNAL INDEPENDENCE & EFFECTIVE BREADTH REPORT")
        print("="*70)

        # 1. Correlation Analysis
        print("\n>>> SIGNAL CORRELATION MATRIX")
        print("-"*50)
        corr_matrix = self.calculate_signal_correlation(signals_df)
        print(corr_matrix.round(3))

        # 2. Effective Breadth
        print("\n>>> EFFECTIVE BREADTH ANALYSIS")
        print("-"*50)
        breadth_results = self.calculate_effective_breadth(signals_df, returns)

        print(f"Raw Breadth (# sources):     {breadth_results['raw_breadth']}")
        print(f"Effective Breadth (ENB):      {breadth_results['effective_breadth']:.2f}")
        print(f"Diversification Ratio:        {breadth_results['diversification_ratio']:.2f}")
        print(f"Components for 95% variance: {breadth_results['n_components_95']}")

        # 3. Information Coefficients
        print("\n>>> INFORMATION COEFFICIENTS (IC)")
        print("-"*50)
        for source, ic in breadth_results['ic_scores'].items():
            print(f"{source:20s}: {ic:>7.4f}")
        print("-"*50)
        print(f"Mean IC:              {breadth_results['mean_ic']:>7.4f}")
        print(f"Transfer Coefficient: {breadth_results['transfer_coefficient']:>7.4f}")
        print(f"Expected IR:          {breadth_results['expected_ir']:>7.4f}")

        # 4. Ablation Study
        print("\n>>> ABLATION STUDY (Source Impact)")
        print("-"*50)
        ablation = self.perform_ablation_study(signals_df, returns)
        print(f"Baseline IC: {ablation['baseline_ic']:.4f}")
        print("\nImpact of removing each source:")

        for source, results in ablation['ablation_results'].items():
            print(f"  Remove {source:15s}: IC loss = {results['ic_loss']:>7.4f} ({results['percentage_impact']:>5.1f}%)")

        # 5. Signal Independence
        print("\n>>> SIGNAL INDEPENDENCE METRICS")
        print("-"*50)
        independence = self.calculate_signal_independence(signals_df)

        if independence:
            print("Variance Inflation Factors (VIF):")
            for source, vif in independence['vif_scores'].items():
                status = "✓" if vif < 5 else "⚠" if vif < 10 else "✗"
                print(f"  {source:20s}: {vif:>6.2f} {status}")

            print("\nUniqueness Scores (1 - R²):")
            for source, unique in independence['uniqueness_scores'].items():
                status = "✓" if unique > 0.5 else "⚠" if unique > 0.3 else "✗"
                print(f"  {source:20s}: {unique:>6.2%} {status}")

        # 6. Key Findings
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)

        effective_vs_raw = breadth_results['effective_breadth'] / breadth_results['raw_breadth']

        if effective_vs_raw > 0.7:
            print("✓ Sources are reasonably independent (>70% effective)")
        elif effective_vs_raw > 0.5:
            print("⚠ Moderate correlation between sources (50-70% effective)")
        else:
            print("✗ High correlation - sources not independent (<50% effective)")

        if breadth_results['expected_ir'] > 1.0:
            print(f"✓ Strong expected IR: {breadth_results['expected_ir']:.2f}")
        else:
            print(f"⚠ Moderate expected IR: {breadth_results['expected_ir']:.2f}")

        print(f"\n→ Effective breadth is {effective_vs_raw:.1%} of raw breadth")
        print(f"→ {breadth_results['n_components_95']} independent factors explain 95% of variance")
        print(f"→ Sources are NOT multiplicative but follow √(breadth) scaling")


def demonstrate_orthogonalization():
    """Demonstrate orthogonalization system"""

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='B')

    # Generate correlated signals (realistic scenario)
    base_signal = np.random.randn(500)

    signals_df = pd.DataFrame({
        'congressional': base_signal + np.random.randn(500) * 0.5,
        'form4': base_signal * 0.7 + np.random.randn(500) * 0.6,
        'sec_filings': base_signal * 0.5 + np.random.randn(500) * 0.7,
        'fed_speeches': np.random.randn(500),  # Independent
        'options_flow': base_signal * 0.3 + np.random.randn(500) * 0.8,
        'earnings_calls': base_signal * 0.4 + np.random.randn(500) * 0.7
    }, index=dates)

    # Generate returns correlated with signals
    returns = (signals_df.mean(axis=1) * 0.3 + np.random.randn(500) * 0.9) * 0.01

    # Run analysis
    system = SignalOrthogonalizationSystem()
    system.generate_independence_report(signals_df, returns)

    # Show orthogonalized signals
    print("\n>>> ORTHOGONALIZED SIGNALS")
    print("-"*50)
    orthogonal = system.orthogonalize_signals(signals_df)
    print("Original correlation (mean):", signals_df.corr().values[np.triu_indices(6, 1)].mean())
    print("Orthogonal correlation (mean):", orthogonal.corr().values[np.triu_indices(6, 1)].mean())

    return system


if __name__ == "__main__":
    demonstrate_orthogonalization()