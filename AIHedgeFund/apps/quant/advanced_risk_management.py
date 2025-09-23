"""
Advanced Risk Management System with Parallel Processing
========================================================
VaR, CVaR, stress testing, and dynamic position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
import ray
from joblib import Parallel, delayed
import asyncio
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    timestamp: datetime


@dataclass
class StressTestResult:
    """Stress test scenario result"""
    scenario_name: str
    portfolio_loss: float
    var_breach: bool
    affected_positions: List[str]
    recovery_time: int  # Days to recover
    probability: float
    severity: str  # low, medium, high, extreme
    recommendations: List[str]


@dataclass
class PositionLimit:
    """Dynamic position limits"""
    symbol: str
    max_position: float
    current_position: float
    risk_budget: float
    var_contribution: float
    optimal_size: float
    kelly_fraction: float
    stop_loss: float
    take_profit: float


class VaRCalculator(ray.remote):
    """Agent for Value at Risk calculations"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.confidence_levels = [0.95, 0.99]

    async def calculate_historical_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Historical VaR and CVaR"""
        if len(returns) < 100:
            return 0, 0

        # VaR
        var = np.percentile(returns, (1 - confidence_level) * 100)

        # CVaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()

        return abs(var), abs(cvar)

    async def calculate_parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Parametric VaR assuming normal distribution"""
        if len(returns) < 30:
            return 0, 0

        mean = returns.mean()
        std = returns.std()

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)

        # VaR
        var = mean + z_score * std

        # CVaR for normal distribution
        cvar = mean - std * stats.norm.pdf(z_score) / (1 - confidence_level)

        return abs(var), abs(cvar)

    async def calculate_monte_carlo_var(
        self,
        returns: pd.Series,
        n_simulations: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR"""
        if len(returns) < 50:
            return 0, 0

        mean = returns.mean()
        std = returns.std()

        # Generate simulations
        simulated_returns = np.random.normal(mean, std, n_simulations)

        # Calculate VaR and CVaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        cvar = simulated_returns[simulated_returns <= var].mean()

        return abs(var), abs(cvar)

    async def calculate_cornish_fisher_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Cornish-Fisher VaR (adjusts for skewness and kurtosis)"""
        if len(returns) < 100:
            return 0

        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Cornish-Fisher expansion
        z = stats.norm.ppf(1 - confidence_level)
        cf_z = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36

        var = mean + cf_z * std

        return abs(var)


class StressTestingAgent(ray.remote):
    """Agent for stress testing and scenario analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.scenarios = self._define_scenarios()

    def _define_scenarios(self) -> Dict[str, Dict]:
        """Define stress test scenarios"""
        return {
            'market_crash': {
                'equity_shock': -0.20,
                'volatility_multiplier': 3.0,
                'correlation_increase': 0.3,
                'probability': 0.05,
                'severity': 'extreme'
            },
            'flash_crash': {
                'equity_shock': -0.10,
                'volatility_multiplier': 5.0,
                'correlation_increase': 0.5,
                'probability': 0.10,
                'severity': 'high'
            },
            'recession': {
                'equity_shock': -0.30,
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.2,
                'probability': 0.15,
                'severity': 'high'
            },
            'interest_rate_shock': {
                'equity_shock': -0.05,
                'volatility_multiplier': 1.5,
                'correlation_increase': 0.1,
                'probability': 0.20,
                'severity': 'medium'
            },
            'geopolitical_crisis': {
                'equity_shock': -0.15,
                'volatility_multiplier': 2.5,
                'correlation_increase': 0.4,
                'probability': 0.10,
                'severity': 'high'
            },
            'liquidity_crisis': {
                'equity_shock': -0.08,
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.6,
                'probability': 0.08,
                'severity': 'high'
            }
        }

    async def run_stress_test(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame,
        scenario_name: str
    ) -> StressTestResult:
        """Run a specific stress test scenario"""
        scenario = self.scenarios.get(scenario_name, self.scenarios['market_crash'])

        # Apply shocks
        shocked_returns = self._apply_shocks(market_data, scenario)

        # Calculate portfolio loss
        portfolio_loss = self._calculate_portfolio_loss(portfolio, shocked_returns)

        # Check VaR breach
        var_breach = portfolio_loss > 0.10  # 10% threshold

        # Identify affected positions
        affected = self._identify_affected_positions(portfolio, shocked_returns)

        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(portfolio_loss, scenario)

        # Generate recommendations
        recommendations = self._generate_recommendations(scenario_name, portfolio_loss)

        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_loss=portfolio_loss,
            var_breach=var_breach,
            affected_positions=affected,
            recovery_time=recovery_time,
            probability=scenario['probability'],
            severity=scenario['severity'],
            recommendations=recommendations
        )

    def _apply_shocks(self, market_data: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """Apply scenario shocks to market data"""
        shocked_data = market_data.copy()

        # Apply equity shock
        if 'close' in shocked_data.columns:
            shocked_data['close'] *= (1 + scenario['equity_shock'])

        # Increase volatility
        returns = shocked_data['close'].pct_change()
        shocked_vol = returns.std() * scenario['volatility_multiplier']

        # Add noise with increased volatility
        noise = np.random.normal(0, shocked_vol, len(shocked_data))
        shocked_data['close'] += noise

        return shocked_data

    def _calculate_portfolio_loss(
        self,
        portfolio: Dict[str, float],
        shocked_data: pd.DataFrame
    ) -> float:
        """Calculate portfolio loss under scenario"""
        initial_value = sum(portfolio.values())

        # Simplified - calculate percentage loss
        avg_shock = shocked_data['close'].pct_change().mean()
        portfolio_loss = abs(avg_shock) * np.random.uniform(0.8, 1.2)

        return min(portfolio_loss, 0.5)  # Cap at 50% loss

    def _identify_affected_positions(
        self,
        portfolio: Dict[str, float],
        shocked_data: pd.DataFrame
    ) -> List[str]:
        """Identify most affected positions"""
        # Return top affected positions (simplified)
        positions = list(portfolio.keys())
        n_affected = min(len(positions), max(1, len(positions) // 3))
        return positions[:n_affected]

    def _estimate_recovery_time(self, loss: float, scenario: Dict) -> int:
        """Estimate recovery time in days"""
        base_recovery = {
            'low': 5,
            'medium': 20,
            'high': 60,
            'extreme': 180
        }

        severity = scenario.get('severity', 'medium')
        base_time = base_recovery.get(severity, 30)

        # Adjust based on loss magnitude
        recovery_time = int(base_time * (1 + loss))

        return recovery_time

    def _generate_recommendations(self, scenario: str, loss: float) -> List[str]:
        """Generate scenario-specific recommendations"""
        recommendations = []

        if loss > 0.20:
            recommendations.append("Immediately reduce leverage")
            recommendations.append("Increase cash position to 30%")
            recommendations.append("Implement stop-loss orders")

        if 'crash' in scenario:
            recommendations.append("Activate hedging strategies")
            recommendations.append("Reduce high-beta positions")
            recommendations.append("Consider defensive sectors")

        if 'liquidity' in scenario:
            recommendations.append("Focus on liquid assets only")
            recommendations.append("Reduce position sizes")
            recommendations.append("Maintain higher cash reserves")

        if 'interest' in scenario:
            recommendations.append("Rotate to rate-defensive sectors")
            recommendations.append("Consider floating rate instruments")
            recommendations.append("Reduce duration risk")

        return recommendations


class PositionSizingAgent(ray.remote):
    """Agent for dynamic position sizing"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.max_position_pct = 0.10  # Max 10% per position
        self.max_sector_pct = 0.30    # Max 30% per sector

    async def calculate_optimal_position_sizes(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_budget: float,
        current_positions: Dict[str, float]
    ) -> Dict[str, PositionLimit]:
        """Calculate optimal position sizes using risk parity"""
        symbols = list(expected_returns.index)
        position_limits = {}

        for symbol in symbols:
            # Kelly Criterion
            kelly = self._calculate_kelly_fraction(
                expected_returns[symbol],
                covariance_matrix.loc[symbol, symbol]
            )

            # Risk parity weight
            risk_parity = self._calculate_risk_parity_weight(
                symbol,
                covariance_matrix
            )

            # VaR contribution
            var_contribution = self._calculate_var_contribution(
                symbol,
                current_positions.get(symbol, 0),
                covariance_matrix
            )

            # Optimal size (blend of methods)
            optimal_size = 0.4 * kelly + 0.6 * risk_parity

            # Apply constraints
            max_position = min(
                risk_budget * self.max_position_pct,
                optimal_size * risk_budget
            )

            # Calculate stop-loss and take-profit
            volatility = np.sqrt(covariance_matrix.loc[symbol, symbol])
            stop_loss = -2 * volatility * np.sqrt(252)
            take_profit = 3 * volatility * np.sqrt(252)

            position_limits[symbol] = PositionLimit(
                symbol=symbol,
                max_position=max_position,
                current_position=current_positions.get(symbol, 0),
                risk_budget=risk_budget,
                var_contribution=var_contribution,
                optimal_size=optimal_size,
                kelly_fraction=kelly,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        return position_limits

    def _calculate_kelly_fraction(self, expected_return: float, variance: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        if variance <= 0:
            return 0

        # Kelly formula: f = μ/σ²
        kelly = expected_return / variance

        # Apply Kelly fraction (typically 0.25 for safety)
        kelly_fraction = min(max(kelly * 0.25, 0), 0.25)

        return kelly_fraction

    def _calculate_risk_parity_weight(
        self,
        symbol: str,
        covariance_matrix: pd.DataFrame
    ) -> float:
        """Calculate risk parity weight"""
        # Inverse volatility weighting (simplified risk parity)
        volatility = np.sqrt(covariance_matrix.loc[symbol, symbol])

        if volatility > 0:
            inv_vol = 1 / volatility
        else:
            inv_vol = 0

        # Normalize
        total_inv_vol = sum(1 / np.sqrt(covariance_matrix.loc[s, s])
                           for s in covariance_matrix.index
                           if covariance_matrix.loc[s, s] > 0)

        if total_inv_vol > 0:
            return inv_vol / total_inv_vol
        else:
            return 1 / len(covariance_matrix)

    def _calculate_var_contribution(
        self,
        symbol: str,
        position: float,
        covariance_matrix: pd.DataFrame
    ) -> float:
        """Calculate marginal VaR contribution"""
        if position == 0:
            return 0

        # Marginal VaR
        portfolio_variance = position**2 * covariance_matrix.loc[symbol, symbol]
        portfolio_std = np.sqrt(portfolio_variance)

        # 95% VaR contribution
        var_contribution = 1.645 * portfolio_std

        return var_contribution


class CorrelationMonitor(ray.remote):
    """Agent for monitoring correlation dynamics"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.correlation_history = deque(maxlen=100)
        self.breakdown_threshold = 0.3

    async def analyze_correlations(
        self,
        returns_data: pd.DataFrame,
        lookback_window: int = 60
    ) -> Dict[str, Any]:
        """Analyze correlation structure and detect breakdowns"""
        # Calculate correlation matrix
        current_corr = returns_data.iloc[-lookback_window:].corr()

        # Rolling correlation
        rolling_corr = self._calculate_rolling_correlation(returns_data)

        # Detect correlation breakdowns
        breakdowns = self._detect_correlation_breakdowns(current_corr, rolling_corr)

        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(current_corr)

        # Identify correlation clusters
        clusters = self._identify_correlation_clusters(current_corr)

        self.correlation_history.append(current_corr)

        return {
            'current_correlation': current_corr,
            'correlation_risk': correlation_risk,
            'breakdowns': breakdowns,
            'clusters': clusters,
            'avg_correlation': current_corr.values[np.triu_indices_from(current_corr.values, k=1)].mean(),
            'max_correlation': current_corr.values[np.triu_indices_from(current_corr.values, k=1)].max(),
            'timestamp': datetime.now()
        }

    def _calculate_rolling_correlation(
        self,
        returns_data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """Calculate rolling correlation"""
        return returns_data.rolling(window).corr()

    def _detect_correlation_breakdowns(
        self,
        current_corr: pd.DataFrame,
        rolling_corr: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """Detect significant correlation changes"""
        breakdowns = []

        if len(self.correlation_history) < 2:
            return breakdowns

        previous_corr = self.correlation_history[-2]

        for i in range(len(current_corr)):
            for j in range(i + 1, len(current_corr)):
                symbol1 = current_corr.index[i]
                symbol2 = current_corr.index[j]

                current_val = current_corr.iloc[i, j]
                previous_val = previous_corr.iloc[i, j]

                change = abs(current_val - previous_val)

                if change > self.breakdown_threshold:
                    breakdowns.append((symbol1, symbol2, change))

        return breakdowns

    def _calculate_correlation_risk(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate overall correlation risk"""
        # Average absolute correlation
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        avg_corr = np.mean(np.abs(upper_triangle))

        # Correlation concentration
        high_corr_count = np.sum(np.abs(upper_triangle) > 0.7)
        corr_concentration = high_corr_count / len(upper_triangle) if len(upper_triangle) > 0 else 0

        # Combined risk score
        correlation_risk = 0.6 * avg_corr + 0.4 * corr_concentration

        return min(correlation_risk, 1.0)

    def _identify_correlation_clusters(self, corr_matrix: pd.DataFrame) -> List[List[str]]:
        """Identify highly correlated clusters"""
        from sklearn.cluster import AgglomerativeClustering

        # Use correlation as distance
        distance_matrix = 1 - np.abs(corr_matrix.values)

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.3,
            linkage='average',
            metric='precomputed'
        )

        labels = clustering.fit_predict(distance_matrix)

        # Group symbols by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(corr_matrix.index[i])

        return list(clusters.values())


class RiskManagementOrchestrator:
    """Orchestrate comprehensive risk management"""

    def __init__(self):
        ray.init(ignore_reinit_error=True)

        self.var_calculator = VaRCalculator.remote("var")
        self.stress_tester = StressTestingAgent.remote("stress")
        self.position_sizer = PositionSizingAgent.remote("sizing")
        self.correlation_monitor = CorrelationMonitor.remote("correlation")

        self.risk_limits = {
            'max_var_95': 0.02,  # 2% daily VaR limit
            'max_var_99': 0.05,  # 5% daily VaR limit
            'max_drawdown': 0.15,  # 15% max drawdown
            'max_leverage': 2.0,   # 2x leverage limit
            'min_sharpe': 0.5      # Minimum Sharpe ratio
        }

    async def calculate_portfolio_risk(
        self,
        portfolio: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio, returns_data)

        # VaR calculations in parallel
        tasks = [
            self.var_calculator.calculate_historical_var.remote(portfolio_returns, 0.95),
            self.var_calculator.calculate_historical_var.remote(portfolio_returns, 0.99),
            self.var_calculator.calculate_parametric_var.remote(portfolio_returns, 0.95),
            self.var_calculator.calculate_monte_carlo_var.remote(portfolio_returns, 0.95),
        ]

        results = await asyncio.gather(*[
            asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            ) for task in tasks
        ])

        # Aggregate VaR metrics
        var_95 = np.mean([results[0][0], results[2][0], results[3][0]])
        cvar_95 = np.mean([results[0][1], results[2][1], results[3][1]])
        var_99 = results[1][0]
        cvar_99 = results[1][1]

        # Other risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        current_drawdown = self._calculate_current_drawdown(portfolio_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_returns, max_drawdown)

        # Market risk
        market_returns = returns_data.mean(axis=1)
        beta = self._calculate_beta(portfolio_returns, market_returns)

        # Correlation risk
        corr_task = self.correlation_monitor.analyze_correlations.remote(returns_data)
        corr_analysis = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, corr_task)
        )
        correlation_risk = corr_analysis['correlation_risk']

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            beta=beta,
            correlation_risk=correlation_risk,
            liquidity_risk=0.1,  # Simplified
            concentration_risk=self._calculate_concentration_risk(portfolio),
            timestamp=datetime.now()
        )

    async def run_stress_tests(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame
    ) -> List[StressTestResult]:
        """Run all stress test scenarios"""
        scenarios = ['market_crash', 'flash_crash', 'recession',
                    'interest_rate_shock', 'geopolitical_crisis', 'liquidity_crisis']

        tasks = [
            self.stress_tester.run_stress_test.remote(portfolio, market_data, scenario)
            for scenario in scenarios
        ]

        results = await asyncio.gather(*[
            asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            ) for task in tasks
        ])

        return results

    async def optimize_position_sizes(
        self,
        expected_returns: pd.Series,
        returns_data: pd.DataFrame,
        risk_budget: float,
        current_positions: Dict[str, float]
    ) -> Dict[str, PositionLimit]:
        """Optimize position sizes based on risk"""
        # Calculate covariance matrix
        covariance_matrix = returns_data.cov()

        # Get optimal sizes
        position_task = self.position_sizer.calculate_optimal_position_sizes.remote(
            expected_returns,
            covariance_matrix,
            risk_budget,
            current_positions
        )

        position_limits = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, position_task)
        )

        return position_limits

    def _calculate_portfolio_returns(
        self,
        portfolio: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns"""
        weights = pd.Series(portfolio)
        weights = weights / weights.sum()  # Normalize

        # Filter returns data for portfolio symbols
        portfolio_returns = returns_data[list(portfolio.keys())].dot(weights)

        return portfolio_returns

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_current_drawdown(self, returns: pd.Series) -> float:
        """Calculate current drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        current_dd = (cumulative.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        return abs(current_dd)

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free / 252
        if returns.std() > 0:
            return excess_returns.mean() / returns.std() * np.sqrt(252)
        return 0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                return excess_returns.mean() / downside_std * np.sqrt(252)
        return 0

    def _calculate_calmar_ratio(self, returns: pd.Series, max_dd: float) -> float:
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        if max_dd > 0:
            return annual_return / max_dd
        return 0

    def _calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        if len(portfolio_returns) != len(market_returns):
            return 1.0

        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance > 0:
            return covariance / market_variance
        return 1.0

    def _calculate_concentration_risk(self, portfolio: Dict[str, float]) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        total_value = sum(portfolio.values())
        if total_value <= 0:
            return 1.0

        weights = [value / total_value for value in portfolio.values()]
        herfindahl = sum(w**2 for w in weights)

        return herfindahl

    def check_risk_limits(self, metrics: RiskMetrics) -> Dict[str, bool]:
        """Check if risk limits are breached"""
        breaches = {}

        breaches['var_95'] = metrics.var_95 > self.risk_limits['max_var_95']
        breaches['var_99'] = metrics.var_99 > self.risk_limits['max_var_99']
        breaches['drawdown'] = metrics.max_drawdown > self.risk_limits['max_drawdown']
        breaches['sharpe'] = metrics.sharpe_ratio < self.risk_limits['min_sharpe']

        return breaches

    def generate_risk_report(
        self,
        metrics: RiskMetrics,
        stress_results: List[StressTestResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        # Check limit breaches
        breaches = self.check_risk_limits(metrics)

        # Worst case scenario
        worst_scenario = max(stress_results, key=lambda x: x.portfolio_loss)

        # Risk score (0-100)
        risk_score = (
            metrics.var_95 * 20 +
            metrics.max_drawdown * 30 +
            (1 - metrics.sharpe_ratio) * 20 +
            metrics.correlation_risk * 15 +
            metrics.concentration_risk * 15
        ) * 100

        return {
            'risk_score': min(risk_score, 100),
            'risk_level': 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low',
            'metrics': metrics,
            'limit_breaches': breaches,
            'worst_scenario': worst_scenario,
            'recommendations': self._generate_recommendations(metrics, breaches),
            'timestamp': datetime.now()
        }

    def _generate_recommendations(
        self,
        metrics: RiskMetrics,
        breaches: Dict[str, bool]
    ) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        if breaches.get('var_95'):
            recommendations.append("Reduce position sizes to lower VaR")

        if breaches.get('drawdown'):
            recommendations.append("Implement stricter stop-losses")

        if breaches.get('sharpe'):
            recommendations.append("Review strategy performance and optimize")

        if metrics.correlation_risk > 0.7:
            recommendations.append("Diversify to reduce correlation risk")

        if metrics.concentration_risk > 0.3:
            recommendations.append("Rebalance to reduce concentration")

        if metrics.current_drawdown > 0.05:
            recommendations.append("Consider reducing exposure during drawdown")

        return recommendations

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of risk management system"""
    orchestrator = RiskManagementOrchestrator()

    # Sample portfolio
    portfolio = {
        'AAPL': 50000,
        'GOOGL': 30000,
        'MSFT': 40000,
        'AMZN': 35000,
        'TSLA': 25000
    }

    # Generate sample returns data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    returns_data = pd.DataFrame({
        symbol: np.random.normal(0.0005, 0.02, len(dates))
        for symbol in portfolio.keys()
    }, index=dates)

    # Calculate risk metrics
    print("Calculating portfolio risk metrics...")
    metrics = await orchestrator.calculate_portfolio_risk(portfolio, returns_data)

    print(f"\nRisk Metrics:")
    print(f"  VaR (95%): {metrics.var_95:.2%}")
    print(f"  CVaR (95%): {metrics.cvar_95:.2%}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Beta: {metrics.beta:.2f}")

    # Run stress tests
    print("\nRunning stress tests...")
    stress_results = await orchestrator.run_stress_tests(
        portfolio,
        pd.DataFrame(returns_data.mean(axis=1), columns=['close'])
    )

    for result in stress_results:
        print(f"\n{result.scenario_name}:")
        print(f"  Loss: {result.portfolio_loss:.2%}")
        print(f"  Severity: {result.severity}")
        print(f"  Recovery: {result.recovery_time} days")

    # Optimize position sizes
    print("\nOptimizing position sizes...")
    expected_returns = pd.Series({s: np.random.uniform(0.05, 0.15) for s in portfolio.keys()})
    position_limits = await orchestrator.optimize_position_sizes(
        expected_returns,
        returns_data,
        risk_budget=sum(portfolio.values()),
        current_positions=portfolio
    )

    for symbol, limit in position_limits.items():
        print(f"\n{symbol}:")
        print(f"  Optimal Size: ${limit.optimal_size * limit.risk_budget:,.0f}")
        print(f"  Max Position: ${limit.max_position:,.0f}")
        print(f"  Kelly Fraction: {limit.kelly_fraction:.2%}")

    # Generate risk report
    report = orchestrator.generate_risk_report(metrics, stress_results)
    print(f"\nRisk Report:")
    print(f"  Risk Score: {report['risk_score']:.1f}/100")
    print(f"  Risk Level: {report['risk_level']}")
    print(f"  Recommendations:")
    for rec in report['recommendations']:
        print(f"    - {rec}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())