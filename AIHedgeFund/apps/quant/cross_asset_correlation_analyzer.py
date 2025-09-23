"""
Cross-Asset Correlation Analyzer with Multi-Agent Processing
============================================================
Multi-asset correlation analysis with regime detection and lead-lag relationships
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ray
import asyncio
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Cross-asset correlation matrix"""
    assets: List[str]
    correlations: np.ndarray
    rolling_window: int
    correlation_type: str  # pearson, spearman, kendall
    timestamp: datetime

    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets"""
        idx1 = self.assets.index(asset1)
        idx2 = self.assets.index(asset2)
        return self.correlations[idx1, idx2]


@dataclass
class CorrelationRegime:
    """Correlation regime state"""
    regime_id: int
    name: str  # risk-on, risk-off, transition, decorrelated
    avg_correlation: float
    dispersion: float
    stability_score: float
    dominant_factors: List[str]
    timestamp: datetime


@dataclass
class LeadLagRelationship:
    """Lead-lag relationship between assets"""
    leader: str
    follower: str
    lag_periods: int
    correlation: float
    granger_causality_pvalue: float
    confidence: float
    timestamp: datetime


@dataclass
class CrossAssetSignal:
    """Cross-asset correlation signal"""
    signal_type: str  # correlation_breakdown, regime_change, lead_signal
    affected_assets: List[str]
    strength: float
    confidence: float
    expected_impact: Dict[str, float]
    action_required: str
    timestamp: datetime


@ray.remote
class CorrelationCalculatorAgent:
    """Agent for correlation calculations"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.correlation_cache = {}

    async def calculate_rolling_correlation(
        self,
        returns_data: Dict[str, pd.Series],
        window: int = 60,
        method: str = 'pearson'
    ) -> List[CorrelationMatrix]:
        """Calculate rolling correlation matrices"""
        assets = list(returns_data.keys())
        n_assets = len(assets)

        # Align all series
        aligned_data = pd.DataFrame(returns_data).dropna()

        # Calculate rolling correlations
        correlation_matrices = []

        for i in range(window, len(aligned_data)):
            window_data = aligned_data.iloc[i-window:i]

            if method == 'pearson':
                corr_matrix = window_data.corr().values
            elif method == 'spearman':
                corr_matrix = window_data.corr(method='spearman').values
            elif method == 'kendall':
                corr_matrix = window_data.corr(method='kendall').values
            else:
                corr_matrix = window_data.corr().values

            correlation_matrices.append(CorrelationMatrix(
                assets=assets,
                correlations=corr_matrix,
                rolling_window=window,
                correlation_type=method,
                timestamp=aligned_data.index[i]
            ))

        return correlation_matrices

    async def detect_correlation_breakdown(
        self,
        current_corr: CorrelationMatrix,
        historical_corr: List[CorrelationMatrix],
        threshold: float = 2.0
    ) -> List[Tuple[str, str, float]]:
        """Detect significant correlation changes"""
        breakdowns = []

        # Calculate historical average and std
        historical_values = {}

        for i, asset1 in enumerate(current_corr.assets):
            for j, asset2 in enumerate(current_corr.assets):
                if i < j:
                    pair = f"{asset1}_{asset2}"
                    historical_values[pair] = [
                        hist.correlations[i, j] for hist in historical_corr
                    ]

        # Check for breakdowns
        for i, asset1 in enumerate(current_corr.assets):
            for j, asset2 in enumerate(current_corr.assets):
                if i < j:
                    pair = f"{asset1}_{asset2}"
                    hist_vals = historical_values[pair]

                    mean_corr = np.mean(hist_vals)
                    std_corr = np.std(hist_vals)

                    if std_corr > 0:
                        z_score = abs(current_corr.correlations[i, j] - mean_corr) / std_corr

                        if z_score > threshold:
                            breakdowns.append((
                                asset1,
                                asset2,
                                z_score
                            ))

        return breakdowns


@ray.remote
class RegimeDetectionAgent:
    """Agent for correlation regime detection"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.current_regime = None
        self.regime_history = deque(maxlen=100)

    async def identify_correlation_regime(
        self,
        corr_matrix: CorrelationMatrix,
        factor_loadings: Optional[np.ndarray] = None
    ) -> CorrelationRegime:
        """Identify current correlation regime"""
        # Calculate average correlation
        n = len(corr_matrix.assets)
        upper_triangle = np.triu_indices(n, k=1)
        correlations = corr_matrix.correlations[upper_triangle]

        avg_correlation = np.mean(correlations)
        dispersion = np.std(correlations)

        # Determine regime
        if avg_correlation > 0.6:
            regime_id = 0
            regime_name = "risk-on"
        elif avg_correlation < 0.2:
            regime_id = 1
            regime_name = "decorrelated"
        elif dispersion > 0.3:
            regime_id = 2
            regime_name = "transition"
        else:
            regime_id = 3
            regime_name = "risk-off"

        # Calculate stability
        if self.regime_history:
            recent_regimes = [r.regime_id for r in list(self.regime_history)[-10:]]
            stability_score = recent_regimes.count(regime_id) / len(recent_regimes)
        else:
            stability_score = 1.0

        # Identify dominant factors
        dominant_factors = []
        if factor_loadings is not None:
            # First 3 principal components
            for i in range(min(3, factor_loadings.shape[1])):
                factor_assets = [
                    corr_matrix.assets[j]
                    for j in np.argsort(np.abs(factor_loadings[:, i]))[-3:]
                ]
                dominant_factors.append(f"PC{i+1}: {', '.join(factor_assets)}")

        regime = CorrelationRegime(
            regime_id=regime_id,
            name=regime_name,
            avg_correlation=avg_correlation,
            dispersion=dispersion,
            stability_score=stability_score,
            dominant_factors=dominant_factors,
            timestamp=corr_matrix.timestamp
        )

        self.regime_history.append(regime)
        self.current_regime = regime

        return regime


@ray.remote
class LeadLagAnalyzerAgent:
    """Agent for lead-lag relationship analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def detect_lead_lag_relationships(
        self,
        price_data: Dict[str, pd.Series],
        max_lag: int = 10
    ) -> List[LeadLagRelationship]:
        """Detect lead-lag relationships using cross-correlation"""
        relationships = []
        assets = list(price_data.keys())

        # Calculate returns
        returns = {
            asset: prices.pct_change().dropna()
            for asset, prices in price_data.items()
        }

        # Test all pairs
        for i, leader in enumerate(assets):
            for j, follower in enumerate(assets):
                if i != j:
                    relationship = await self._test_lead_lag(
                        returns[leader],
                        returns[follower],
                        leader,
                        follower,
                        max_lag
                    )

                    if relationship and relationship.confidence > 0.7:
                        relationships.append(relationship)

        # Sort by confidence
        relationships.sort(key=lambda x: x.confidence, reverse=True)

        return relationships[:20]  # Top 20 relationships

    async def _test_lead_lag(
        self,
        leader_returns: pd.Series,
        follower_returns: pd.Series,
        leader_name: str,
        follower_name: str,
        max_lag: int
    ) -> Optional[LeadLagRelationship]:
        """Test for lead-lag relationship"""
        try:
            # Align series
            aligned = pd.DataFrame({
                'leader': leader_returns,
                'follower': follower_returns
            }).dropna()

            if len(aligned) < 100:
                return None

            # Calculate cross-correlations
            correlations = []
            for lag in range(1, max_lag + 1):
                shifted_leader = aligned['leader'].shift(lag)
                valid_idx = ~shifted_leader.isna()

                if valid_idx.sum() > 50:
                    corr, _ = pearsonr(
                        shifted_leader[valid_idx],
                        aligned['follower'][valid_idx]
                    )
                    correlations.append((lag, abs(corr)))

            if not correlations:
                return None

            # Find optimal lag
            optimal_lag, max_corr = max(correlations, key=lambda x: x[1])

            # Granger causality test (simplified)
            from statsmodels.tsa.stattools import grangercausalitytests

            test_data = aligned[['follower', 'leader']].values
            try:
                result = grangercausalitytests(test_data, maxlag=optimal_lag, verbose=False)
                p_value = result[optimal_lag][0]['ssr_ftest'][1]
            except:
                p_value = 1.0

            # Calculate confidence
            confidence = max_corr * (1 - p_value) if p_value < 0.05 else 0

            if confidence > 0:
                return LeadLagRelationship(
                    leader=leader_name,
                    follower=follower_name,
                    lag_periods=optimal_lag,
                    correlation=max_corr,
                    granger_causality_pvalue=p_value,
                    confidence=confidence,
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.debug(f"Lead-lag test failed: {e}")

        return None


@ray.remote
class MultiAssetFactorAgent:
    """Agent for multi-asset factor analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.pca = PCA(n_components=5)
        self.scaler = StandardScaler()

    async def extract_common_factors(
        self,
        returns_data: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Extract common factors using PCA"""
        # Prepare data
        returns_df = pd.DataFrame(returns_data).dropna()

        # Standardize
        returns_scaled = self.scaler.fit_transform(returns_df)

        # PCA
        factors = self.pca.fit_transform(returns_scaled)
        loadings = self.pca.components_.T

        # Calculate factor contributions
        explained_variance = self.pca.explained_variance_ratio_

        # Identify factor interpretations
        factor_interpretations = []
        for i in range(min(3, loadings.shape[1])):
            top_assets_idx = np.argsort(np.abs(loadings[:, i]))[-5:]
            top_assets = [returns_df.columns[idx] for idx in top_assets_idx]

            factor_interpretations.append({
                'factor': f'PC{i+1}',
                'variance_explained': explained_variance[i],
                'top_contributors': top_assets,
                'loadings': loadings[:, i]
            })

        return {
            'factors': factors,
            'loadings': loadings,
            'explained_variance': explained_variance,
            'interpretations': factor_interpretations,
            'n_factors_90pct': np.argmax(np.cumsum(explained_variance) > 0.9) + 1
        }

    async def calculate_factor_exposures(
        self,
        asset_returns: pd.Series,
        factor_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate asset's exposure to common factors"""
        exposures = {}

        for i in range(factor_returns.shape[1]):
            # Regression of asset returns on factor
            valid_idx = ~np.isnan(factor_returns[:, i])

            if valid_idx.sum() > 20:
                X = factor_returns[valid_idx, i].reshape(-1, 1)
                y = asset_returns.values[valid_idx]

                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)

                exposures[f'factor_{i+1}'] = float(model.coef_[0])
                exposures[f'factor_{i+1}_r2'] = float(model.score(X, y))

        return exposures


@ray.remote
class CurrencyImpactAgent:
    """Agent for currency impact analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.base_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CNY']

    async def analyze_currency_impact(
        self,
        asset_returns: Dict[str, pd.Series],
        currency_returns: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Analyze currency impact on asset returns"""
        impacts = {}

        for asset, returns in asset_returns.items():
            asset_impacts = {}

            for currency, fx_returns in currency_returns.items():
                # Align series
                aligned = pd.DataFrame({
                    'asset': returns,
                    'fx': fx_returns
                }).dropna()

                if len(aligned) > 50:
                    # Calculate correlation
                    corr, p_value = pearsonr(aligned['asset'], aligned['fx'])

                    # Calculate beta (sensitivity)
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    X = aligned[['fx']].values
                    y = aligned['asset'].values
                    model.fit(X, y)

                    beta = float(model.coef_[0])

                    asset_impacts[currency] = {
                        'correlation': corr,
                        'beta': beta,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

            impacts[asset] = asset_impacts

        # Find dominant currency influences
        dominant_influences = {}
        for asset, currency_impacts in impacts.items():
            sorted_impacts = sorted(
                currency_impacts.items(),
                key=lambda x: abs(x[1]['correlation']),
                reverse=True
            )

            dominant_influences[asset] = {
                'primary_currency': sorted_impacts[0][0] if sorted_impacts else None,
                'correlation': sorted_impacts[0][1]['correlation'] if sorted_impacts else 0
            }

        return {
            'detailed_impacts': impacts,
            'dominant_influences': dominant_influences,
            'timestamp': datetime.now()
        }


class CrossAssetCorrelationOrchestrator:
    """Orchestrate cross-asset correlation analysis"""

    def __init__(self, n_agents: int = 5):
        ray.init(ignore_reinit_error=True)

        self.correlation_agent = CorrelationCalculatorAgent.remote("correlation")
        self.regime_agent = RegimeDetectionAgent.remote("regime")
        self.leadlag_agent = LeadLagAnalyzerAgent.remote("leadlag")
        self.factor_agent = MultiAssetFactorAgent.remote("factor")
        self.currency_agent = CurrencyImpactAgent.remote("currency")

        self.correlation_history = deque(maxlen=100)
        self.current_regime = None

    async def analyze_correlations(
        self,
        price_data: Dict[str, pd.DataFrame],
        currency_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """Comprehensive correlation analysis"""
        # Calculate returns
        returns_data = {
            symbol: df['close'].pct_change().dropna() if 'close' in df.columns else df.iloc[:, 0].pct_change().dropna()
            for symbol, df in price_data.items()
        }

        # Rolling correlations
        corr_task = self.correlation_agent.calculate_rolling_correlation.remote(
            returns_data, window=60, method='pearson'
        )
        correlation_matrices = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, corr_task)
        )

        if correlation_matrices:
            self.correlation_history.extend(correlation_matrices[-10:])
            current_correlation = correlation_matrices[-1]

            # Detect correlation breakdowns
            if len(self.correlation_history) > 20:
                breakdown_task = self.correlation_agent.detect_correlation_breakdown.remote(
                    current_correlation,
                    list(self.correlation_history)[:-1],
                    threshold=2.0
                )
                breakdowns = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, breakdown_task)
                )
            else:
                breakdowns = []

            # Factor analysis
            factor_task = self.factor_agent.extract_common_factors.remote(returns_data)
            factor_analysis = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, factor_task)
            )

            # Regime detection
            regime_task = self.regime_agent.identify_correlation_regime.remote(
                current_correlation,
                factor_analysis['loadings']
            )
            current_regime = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, regime_task)
            )
            self.current_regime = current_regime

            # Lead-lag relationships
            leadlag_task = self.leadlag_agent.detect_lead_lag_relationships.remote(
                {k: v['close'] if 'close' in v.columns else v.iloc[:, 0]
                 for k, v in price_data.items()},
                max_lag=10
            )
            leadlag_relationships = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, leadlag_task)
            )

            # Currency impact
            currency_impact = None
            if currency_data:
                currency_returns = {
                    symbol: df['close'].pct_change().dropna() if 'close' in df.columns else df.iloc[:, 0].pct_change().dropna()
                    for symbol, df in currency_data.items()
                }

                currency_task = self.currency_agent.analyze_currency_impact.remote(
                    returns_data, currency_returns
                )
                currency_impact = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, currency_task)
                )

            return {
                'current_correlation': current_correlation,
                'correlation_breakdowns': breakdowns,
                'regime': current_regime,
                'factor_analysis': factor_analysis,
                'leadlag_relationships': leadlag_relationships,
                'currency_impact': currency_impact,
                'timestamp': datetime.now()
            }

        return {}

    def generate_signals(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[CrossAssetSignal]:
        """Generate trading signals from correlation analysis"""
        signals = []

        # Correlation breakdown signals
        if analysis_results.get('correlation_breakdowns'):
            for asset1, asset2, z_score in analysis_results['correlation_breakdowns']:
                signal = CrossAssetSignal(
                    signal_type='correlation_breakdown',
                    affected_assets=[asset1, asset2],
                    strength=min(z_score / 3, 1.0),
                    confidence=0.8,
                    expected_impact={
                        asset1: -0.02 * z_score,
                        asset2: -0.02 * z_score
                    },
                    action_required='reduce_correlation_exposure',
                    timestamp=datetime.now()
                )
                signals.append(signal)

        # Regime change signals
        if analysis_results.get('regime'):
            regime = analysis_results['regime']
            if regime.stability_score < 0.3:
                signal = CrossAssetSignal(
                    signal_type='regime_change',
                    affected_assets=analysis_results['current_correlation'].assets,
                    strength=1.0 - regime.stability_score,
                    confidence=0.7,
                    expected_impact={
                        asset: 0.05 if regime.name == 'risk-off' else -0.03
                        for asset in analysis_results['current_correlation'].assets
                    },
                    action_required=f'adjust_for_{regime.name}_regime',
                    timestamp=datetime.now()
                )
                signals.append(signal)

        # Lead-lag signals
        if analysis_results.get('leadlag_relationships'):
            for relationship in analysis_results['leadlag_relationships'][:5]:
                if relationship.confidence > 0.8:
                    signal = CrossAssetSignal(
                        signal_type='lead_signal',
                        affected_assets=[relationship.leader, relationship.follower],
                        strength=relationship.correlation,
                        confidence=relationship.confidence,
                        expected_impact={
                            relationship.follower: 0.01 * relationship.correlation
                        },
                        action_required=f'monitor_{relationship.leader}_for_{relationship.follower}_signal',
                        timestamp=datetime.now()
                    )
                    signals.append(signal)

        return signals

    def get_portfolio_correlation_metrics(
        self,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate portfolio-level correlation metrics"""
        if not analysis_results.get('current_correlation'):
            return {}

        corr_matrix = analysis_results['current_correlation']
        n = len(corr_matrix.assets)

        # Average pairwise correlation
        upper_triangle = np.triu_indices(n, k=1)
        avg_correlation = np.mean(corr_matrix.correlations[upper_triangle])

        # Effective number of independent bets (Effective N)
        eigenvalues = np.linalg.eigvalsh(corr_matrix.correlations)
        eigenvalues = eigenvalues[eigenvalues > 0]
        effective_n = np.exp(-np.sum(eigenvalues * np.log(eigenvalues)) / n) if len(eigenvalues) > 0 else 1

        # Diversification ratio
        equal_weight = np.ones(n) / n
        portfolio_variance = equal_weight @ corr_matrix.correlations @ equal_weight.T
        diversification_ratio = 1 / np.sqrt(n * portfolio_variance) if portfolio_variance > 0 else 1

        # Concentration risk (HHI)
        if analysis_results.get('factor_analysis'):
            explained_var = analysis_results['factor_analysis']['explained_variance']
            hhi = np.sum(explained_var ** 2)
        else:
            hhi = 1 / n

        return {
            'avg_correlation': avg_correlation,
            'effective_n': effective_n,
            'diversification_ratio': diversification_ratio,
            'concentration_hhi': hhi,
            'regime': analysis_results['regime'].name if analysis_results.get('regime') else 'unknown'
        }

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of cross-asset correlation analyzer"""
    orchestrator = CrossAssetCorrelationOrchestrator()

    # Generate sample multi-asset data
    assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'UUP', 'XLE', 'XLF', 'VNQ']
    price_data = {}

    for i, asset in enumerate(assets):
        # Create correlated price series
        base_returns = np.random.randn(500) * 0.02

        # Add some correlation structure
        if i > 0:
            correlation = 0.3 + np.random.random() * 0.4
            base_returns = correlation * base_returns + (1 - correlation) * np.random.randn(500) * 0.02

        prices = 100 * np.exp(np.cumsum(base_returns))
        price_data[asset] = pd.DataFrame({'close': prices})

    # Currency data
    currency_data = {
        'DXY': pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(500) * 0.5)}),
        'EURUSD': pd.DataFrame({'close': 1.1 + np.cumsum(np.random.randn(500) * 0.01)})
    }

    # Analyze correlations
    results = await orchestrator.analyze_correlations(price_data, currency_data)

    # Generate signals
    signals = orchestrator.generate_signals(results)

    # Portfolio metrics
    metrics = orchestrator.get_portfolio_correlation_metrics(results)

    print("Cross-Asset Correlation Analysis")
    print("=" * 50)

    if results.get('regime'):
        regime = results['regime']
        print(f"\nCurrent Regime: {regime.name}")
        print(f"  Average Correlation: {regime.avg_correlation:.3f}")
        print(f"  Stability Score: {regime.stability_score:.1%}")

    if results.get('correlation_breakdowns'):
        print(f"\nCorrelation Breakdowns Detected: {len(results['correlation_breakdowns'])}")
        for asset1, asset2, z_score in results['correlation_breakdowns'][:3]:
            print(f"  {asset1}-{asset2}: Z-score = {z_score:.2f}")

    if results.get('leadlag_relationships'):
        print(f"\nLead-Lag Relationships: {len(results['leadlag_relationships'])}")
        for rel in results['leadlag_relationships'][:3]:
            print(f"  {rel.leader} -> {rel.follower} (lag: {rel.lag_periods}, conf: {rel.confidence:.1%})")

    if signals:
        print(f"\nGenerated Signals: {len(signals)}")
        for signal in signals[:3]:
            print(f"  {signal.signal_type}: {', '.join(signal.affected_assets)} (conf: {signal.confidence:.1%})")

    print("\nPortfolio Correlation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())