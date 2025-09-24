"""
Dynamic Portfolio Optimization with Multi-Agent Processing
==========================================================
Real-time portfolio optimization with risk budgeting and multi-objective optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, LinearConstraint
from scipy import stats
import cvxpy as cp
import ray
import asyncio
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n: float
    turnover: float
    transaction_cost: float
    confidence_level: float
    timestamp: datetime


@dataclass
class RiskBudget:
    """Risk budget allocation"""
    total_risk_budget: float
    asset_risk_budgets: Dict[str, float]
    factor_risk_budgets: Dict[str, float]
    current_risk_usage: float
    available_risk: float
    risk_utilization: float
    timestamp: datetime


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    max_position: float = 0.2
    min_position: float = -0.1
    max_turnover: float = 0.5
    max_leverage: float = 1.5
    sector_limits: Dict[str, float] = field(default_factory=dict)
    esg_minimum: float = 0.0
    liquidity_minimum: float = 0.7


@ray.remote
class BlackLittermanAgent:
    """Agent for Black-Litterman optimization with ML views"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.tau = 0.05  # Confidence in equilibrium
        self.risk_aversion = 2.5

    async def optimize_portfolio(
        self,
        market_data: pd.DataFrame,
        ml_predictions: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        market_cap_weights: Optional[Dict[str, float]] = None
    ) -> PortfolioAllocation:
        """Run Black-Litterman optimization with ML views"""

        assets = list(ml_predictions.keys())
        n_assets = len(assets)

        # Calculate equilibrium returns
        if market_cap_weights:
            equilibrium_returns = self._calculate_equilibrium_returns(
                covariance_matrix, market_cap_weights
            )
        else:
            # Use equal weights as prior
            equilibrium_returns = pd.Series(
                np.ones(n_assets) * 0.08 / 252,  # 8% annual
                index=assets
            )

        # Create view matrix (P) and view returns (Q)
        P, Q, omega = self._create_ml_views(ml_predictions, assets)

        # Calculate posterior returns
        posterior_returns = self._calculate_posterior_returns(
            equilibrium_returns, covariance_matrix, P, Q, omega
        )

        # Optimize portfolio with posterior returns
        optimal_weights = self._optimize_with_constraints(
            posterior_returns, covariance_matrix
        )

        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, posterior_returns)
        portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
        portfolio_risk = np.sqrt(portfolio_variance)

        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

        # Calculate diversification ratio
        individual_risks = np.sqrt(np.diag(covariance_matrix.values))
        diversification = np.dot(np.abs(optimal_weights), individual_risks) / portfolio_risk if portfolio_risk > 0 else 1

        return PortfolioAllocation(
            weights={assets[i]: optimal_weights[i] for i in range(n_assets)},
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification,
            effective_n=1 / np.sum(optimal_weights ** 2) if np.sum(optimal_weights ** 2) > 0 else n_assets,
            turnover=0,  # Will be calculated separately
            transaction_cost=0,
            confidence_level=self._calculate_confidence(omega),
            timestamp=datetime.now()
        )

    def _calculate_equilibrium_returns(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate implied equilibrium returns"""
        weights = np.array(list(market_weights.values()))
        equilibrium = self.risk_aversion * np.dot(cov_matrix.values, weights)
        return pd.Series(equilibrium, index=cov_matrix.index)

    def _create_ml_views(
        self,
        ml_predictions: Dict[str, float],
        assets: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create view matrix from ML predictions"""
        n_assets = len(assets)
        n_views = len(ml_predictions)

        # View matrix (each ML prediction is a view)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)

        for i, (asset, prediction) in enumerate(ml_predictions.items()):
            if asset in assets:
                asset_idx = assets.index(asset)
                P[i, asset_idx] = 1
                Q[i] = prediction

        # Uncertainty matrix (diagonal with ML confidence as inverse)
        omega_diag = np.array([
            0.01 * abs(pred) for pred in ml_predictions.values()
        ])
        omega = np.diag(omega_diag)

        return P, Q, omega

    def _calculate_posterior_returns(
        self,
        prior_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray
    ) -> np.ndarray:
        """Calculate Black-Litterman posterior returns"""
        sigma = cov_matrix.values
        pi = prior_returns.values

        # Scaling matrix
        tau_sigma = self.tau * sigma

        # Posterior covariance
        inv_omega = np.linalg.inv(omega)
        posterior_cov_inv = np.linalg.inv(tau_sigma) + P.T @ inv_omega @ P
        posterior_cov = np.linalg.inv(posterior_cov_inv)

        # Posterior mean
        posterior_mean = posterior_cov @ (
            np.linalg.inv(tau_sigma) @ pi + P.T @ inv_omega @ Q
        )

        return posterior_mean

    def _optimize_with_constraints(
        self,
        expected_returns: np.ndarray,
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Optimize portfolio with constraints"""
        n_assets = len(expected_returns)

        # Use quadratic programming
        def objective(w):
            return -np.dot(w, expected_returns) + self.risk_aversion * np.dot(w, np.dot(cov_matrix, w))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds (allow short selling up to 10%)
        bounds = [(-0.1, 0.3) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else x0

    def _calculate_confidence(self, omega: np.ndarray) -> float:
        """Calculate confidence level based on view uncertainty"""
        avg_uncertainty = np.mean(np.diag(omega))
        confidence = 1 / (1 + avg_uncertainty * 100)
        return min(0.95, max(0.5, confidence))


@ray.remote
class RiskBudgetingAgent:
    """Agent for dynamic risk budgeting"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.risk_history = deque(maxlen=100)

    async def calculate_risk_budget(
        self,
        portfolio: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        target_risk: float = 0.15,
        time_horizon: int = 1
    ) -> RiskBudget:
        """Calculate and allocate risk budget"""

        assets = list(portfolio.keys())
        weights = np.array(list(portfolio.values()))

        # Calculate portfolio risk
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix.values, weights))
        portfolio_risk = np.sqrt(portfolio_variance * time_horizon)

        # Calculate risk contributions
        marginal_contributions = np.dot(covariance_matrix.values, weights)
        risk_contributions = weights * marginal_contributions / portfolio_risk if portfolio_risk > 0 else np.zeros_like(weights)

        # Risk budgets per asset
        asset_risk_budgets = {
            assets[i]: risk_contributions[i] * target_risk
            for i in range(len(assets))
        }

        # Factor risk budgets (simplified - in production would use factor model)
        factor_risk_budgets = self._calculate_factor_risk_budgets(
            weights, covariance_matrix, target_risk
        )

        # Calculate utilization
        current_risk_usage = portfolio_risk
        available_risk = max(0, target_risk - current_risk_usage)
        risk_utilization = current_risk_usage / target_risk if target_risk > 0 else 0

        return RiskBudget(
            total_risk_budget=target_risk,
            asset_risk_budgets=asset_risk_budgets,
            factor_risk_budgets=factor_risk_budgets,
            current_risk_usage=current_risk_usage,
            available_risk=available_risk,
            risk_utilization=risk_utilization,
            timestamp=datetime.now()
        )

    def _calculate_factor_risk_budgets(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame,
        target_risk: float
    ) -> Dict[str, float]:
        """Calculate factor-based risk budgets"""
        # Simplified factor model
        factors = {
            'market': 0.4,
            'size': 0.2,
            'value': 0.2,
            'momentum': 0.1,
            'quality': 0.1
        }

        # Allocate target risk to factors
        factor_budgets = {
            factor: allocation * target_risk
            for factor, allocation in factors.items()
        }

        return factor_budgets

    async def optimize_risk_parity(
        self,
        assets: List[str],
        covariance_matrix: pd.DataFrame,
        risk_budgets: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Optimize portfolio for risk parity"""

        n_assets = len(assets)

        if risk_budgets is None:
            # Equal risk contribution
            risk_budgets = {asset: 1/n_assets for asset in assets}

        # Risk parity optimization
        def risk_parity_objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix.values, weights)))
            marginal_contributions = np.dot(covariance_matrix.values, weights)
            risk_contributions = weights * marginal_contributions / portfolio_risk if portfolio_risk > 0 else np.zeros_like(weights)

            # Target risk contributions
            target_contributions = np.array(list(risk_budgets.values()))

            # Minimize squared differences
            return np.sum((risk_contributions - target_contributions) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w}  # Non-negative weights
        ]

        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return {assets[i]: result.x[i] for i in range(n_assets)}
        else:
            return {asset: 1/n_assets for asset in assets}


@ray.remote
class MultiObjectiveOptimizer:
    """Agent for multi-objective portfolio optimization"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def optimize_multi_objective(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objectives: Dict[str, float],
        constraints: OptimizationConstraints
    ) -> PortfolioAllocation:
        """Optimize portfolio with multiple objectives"""

        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()

        # Decision variables
        weights = cp.Variable(n_assets)

        # Primary objective: Maximize risk-adjusted return
        portfolio_return = expected_returns.values @ weights
        portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
        portfolio_risk = cp.sqrt(portfolio_variance)

        # Multiple objectives with weights
        objective_terms = []

        # 1. Return maximization
        if 'return' in objectives:
            objective_terms.append(objectives['return'] * portfolio_return)

        # 2. Risk minimization
        if 'risk' in objectives:
            objective_terms.append(-objectives['risk'] * portfolio_risk)

        # 3. Diversification maximization (negative HHI)
        if 'diversification' in objectives:
            hhi = cp.sum_squares(weights)
            objective_terms.append(-objectives['diversification'] * hhi)

        # 4. Transaction cost minimization (if rebalancing)
        if 'transaction_cost' in objectives and 'current_weights' in objectives:
            current = objectives['current_weights']
            turnover = cp.norm(weights - current, 1)
            objective_terms.append(-objectives['transaction_cost'] * turnover)

        # Combine objectives
        objective = cp.Maximize(cp.sum(objective_terms))

        # Constraints
        constraint_list = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= constraints.min_position,  # Min position
            weights <= constraints.max_position,  # Max position
        ]

        # Leverage constraint
        if constraints.max_leverage > 1:
            constraint_list.append(cp.norm(weights, 1) <= constraints.max_leverage)

        # Sector constraints
        if constraints.sector_limits:
            # Would need sector mapping in production
            pass

        # Solve
        problem = cp.Problem(objective, constraint_list)

        try:
            problem.solve(solver=cp.OSQP)

            if weights.value is not None:
                optimal_weights = weights.value
            else:
                # Fallback to equal weights
                optimal_weights = np.ones(n_assets) / n_assets
        except:
            optimal_weights = np.ones(n_assets) / n_assets

        # Calculate metrics
        portfolio_return = np.dot(optimal_weights, expected_returns.values)
        portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix.values, optimal_weights))
        portfolio_risk = np.sqrt(portfolio_variance)

        return PortfolioAllocation(
            weights={assets[i]: optimal_weights[i] for i in range(n_assets)},
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
            diversification_ratio=self._calculate_diversification(optimal_weights, covariance_matrix),
            effective_n=1 / np.sum(optimal_weights ** 2) if np.sum(optimal_weights ** 2) > 0 else n_assets,
            turnover=0,
            transaction_cost=0,
            confidence_level=0.8,
            timestamp=datetime.now()
        )

    def _calculate_diversification(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate diversification ratio"""
        individual_risks = np.sqrt(np.diag(cov_matrix.values))
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))

        if portfolio_risk > 0:
            return np.dot(np.abs(weights), individual_risks) / portfolio_risk
        return 1


@ray.remote
class TransactionCostOptimizer:
    """Agent for transaction cost optimization"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.cost_model = {
            'commission': 0.001,
            'spread': 0.0005,
            'market_impact': 0.0001
        }

    async def optimize_with_costs(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        turnover_limit: float = 0.5
    ) -> PortfolioAllocation:
        """Optimize portfolio considering transaction costs"""

        assets = list(target_weights.keys())
        n_assets = len(assets)

        # Current and target weight arrays
        current = np.array([current_weights.get(asset, 0) for asset in assets])
        target = np.array([target_weights[asset] for asset in assets])

        # Calculate transaction costs
        trades = target - current
        transaction_costs = self._calculate_transaction_costs(trades, assets)

        # Net returns after costs
        net_returns = expected_returns.values - transaction_costs

        # Optimize with turnover constraint
        def objective(w):
            portfolio_return = np.dot(w, net_returns)
            portfolio_risk = np.sqrt(np.dot(w, np.dot(covariance_matrix.values, w)))
            sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            return -sharpe

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: turnover_limit - np.sum(np.abs(w - current))}
        ]

        # Bounds
        bounds = [(0, 0.3) for _ in range(n_assets)]

        # Optimize
        result = minimize(objective, target, method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x if result.success else target

        # Calculate final metrics
        final_trades = optimal_weights - current
        final_costs = self._calculate_total_cost(final_trades)
        turnover = np.sum(np.abs(final_trades))

        portfolio_return = np.dot(optimal_weights, expected_returns.values) - final_costs
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix.values, optimal_weights)))

        return PortfolioAllocation(
            weights={assets[i]: optimal_weights[i] for i in range(n_assets)},
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
            diversification_ratio=1,
            effective_n=1 / np.sum(optimal_weights ** 2) if np.sum(optimal_weights ** 2) > 0 else n_assets,
            turnover=turnover,
            transaction_cost=final_costs,
            confidence_level=0.85,
            timestamp=datetime.now()
        )

    def _calculate_transaction_costs(
        self,
        trades: np.ndarray,
        assets: List[str]
    ) -> np.ndarray:
        """Calculate transaction costs per asset"""
        costs = np.zeros_like(trades)

        for i, trade in enumerate(trades):
            if trade != 0:
                # Commission
                costs[i] += self.cost_model['commission'] * abs(trade)
                # Spread
                costs[i] += self.cost_model['spread'] * abs(trade)
                # Market impact (square root model)
                costs[i] += self.cost_model['market_impact'] * np.sqrt(abs(trade))

        return costs

    def _calculate_total_cost(self, trades: np.ndarray) -> float:
        """Calculate total transaction cost"""
        total_cost = 0

        for trade in trades:
            if trade != 0:
                total_cost += self.cost_model['commission'] * abs(trade)
                total_cost += self.cost_model['spread'] * abs(trade)
                total_cost += self.cost_model['market_impact'] * np.sqrt(abs(trade))

        return total_cost


class DynamicPortfolioOrchestrator:
    """Orchestrate dynamic portfolio optimization with multi-agent processing"""

    def __init__(self):
        ray.init(ignore_reinit_error=True)

        # Initialize optimization agents
        self.black_litterman_agent = BlackLittermanAgent.remote("black_litterman")
        self.risk_budgeting_agent = RiskBudgetingAgent.remote("risk_budget")
        self.multi_objective_agent = MultiObjectiveOptimizer.remote("multi_objective")
        self.transaction_cost_agent = TransactionCostOptimizer.remote("transaction_cost")

        # Portfolio tracking
        self.current_portfolio = {}
        self.optimization_history = deque(maxlen=100)

    async def optimize_daily_portfolio(
        self,
        market_data: Dict[str, pd.DataFrame],
        ml_predictions: Dict[str, float],
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, Any]:
        """Run complete daily portfolio optimization"""

        if constraints is None:
            constraints = OptimizationConstraints()

        # Prepare data
        returns_data = self._calculate_returns(market_data)
        covariance_matrix = self._calculate_covariance(returns_data)
        expected_returns = pd.Series(ml_predictions)

        # Parallel optimization tasks
        optimization_tasks = []

        # 1. Black-Litterman optimization
        bl_task = self.black_litterman_agent.optimize_portfolio.remote(
            returns_data, ml_predictions, covariance_matrix
        )
        optimization_tasks.append(('black_litterman', bl_task))

        # 2. Risk parity optimization
        assets = list(ml_predictions.keys())
        rp_task = self.risk_budgeting_agent.optimize_risk_parity.remote(
            assets, covariance_matrix
        )
        optimization_tasks.append(('risk_parity', rp_task))

        # 3. Multi-objective optimization
        objectives = {
            'return': 0.4,
            'risk': 0.3,
            'diversification': 0.2,
            'transaction_cost': 0.1,
            'current_weights': self.current_portfolio
        }
        mo_task = self.multi_objective_agent.optimize_multi_objective.remote(
            expected_returns, covariance_matrix, objectives, constraints
        )
        optimization_tasks.append(('multi_objective', mo_task))

        # Gather results
        optimization_results = {}

        for opt_type, task in optimization_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )
            optimization_results[opt_type] = result

        # Select best portfolio (ensemble approach)
        best_portfolio = self._select_best_portfolio(optimization_results)

        # Optimize for transaction costs if rebalancing
        if self.current_portfolio:
            tc_task = self.transaction_cost_agent.optimize_with_costs.remote(
                best_portfolio.weights,
                self.current_portfolio,
                expected_returns,
                covariance_matrix
            )
            final_portfolio = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, tc_task)
            )
        else:
            final_portfolio = best_portfolio

        # Calculate risk budget
        risk_budget_task = self.risk_budgeting_agent.calculate_risk_budget.remote(
            final_portfolio.weights,
            covariance_matrix,
            target_risk=0.15
        )
        risk_budget = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, risk_budget_task)
        )

        # Update current portfolio
        self.current_portfolio = final_portfolio.weights
        self.optimization_history.append(final_portfolio)

        return {
            'optimal_portfolio': final_portfolio,
            'risk_budget': risk_budget,
            'optimization_methods': optimization_results,
            'rebalancing_required': final_portfolio.turnover > 0.05,
            'timestamp': datetime.now()
        }

    def _calculate_returns(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns matrix"""
        returns_dict = {}

        for symbol, df in market_data.items():
            if 'close' in df.columns:
                returns_dict[symbol] = df['close'].pct_change().dropna()

        return pd.DataFrame(returns_dict)

    def _calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix with shrinkage"""
        # Simple covariance
        simple_cov = returns.cov()

        # Shrinkage target (identity matrix scaled)
        n_assets = len(returns.columns)
        avg_var = np.diag(simple_cov.values).mean()
        shrinkage_target = np.eye(n_assets) * avg_var

        # Ledoit-Wolf shrinkage
        shrinkage_intensity = 0.1
        shrunk_cov = (1 - shrinkage_intensity) * simple_cov + shrinkage_intensity * shrinkage_target

        return pd.DataFrame(shrunk_cov, index=simple_cov.index, columns=simple_cov.columns)

    def _select_best_portfolio(
        self,
        results: Dict[str, Any]
    ) -> PortfolioAllocation:
        """Select best portfolio from multiple optimizations"""

        # Score each portfolio
        scores = {}

        for method, portfolio in results.items():
            if isinstance(portfolio, PortfolioAllocation):
                # Score based on Sharpe, diversification, and confidence
                score = (
                    portfolio.sharpe_ratio * 0.4 +
                    portfolio.diversification_ratio * 0.3 +
                    portfolio.confidence_level * 0.3
                )
                scores[method] = (score, portfolio)
            elif isinstance(portfolio, dict):  # Risk parity returns dict
                # Create PortfolioAllocation object
                allocation = PortfolioAllocation(
                    weights=portfolio,
                    expected_return=0,
                    expected_risk=0,
                    sharpe_ratio=0,
                    diversification_ratio=1,
                    effective_n=len(portfolio),
                    turnover=0,
                    transaction_cost=0,
                    confidence_level=0.7,
                    timestamp=datetime.now()
                )
                scores[method] = (0.7, allocation)  # Default score

        # Select highest scoring
        if scores:
            best_method = max(scores.keys(), key=lambda k: scores[k][0])
            return scores[best_method][1]
        else:
            # Return equal weights as fallback
            n_assets = 5
            equal_weights = {f"Asset_{i}": 1/n_assets for i in range(n_assets)}
            return PortfolioAllocation(
                weights=equal_weights,
                expected_return=0,
                expected_risk=0.15,
                sharpe_ratio=0,
                diversification_ratio=1,
                effective_n=n_assets,
                turnover=0,
                transaction_cost=0,
                confidence_level=0.5,
                timestamp=datetime.now()
            )

    def get_portfolio_recommendations(self) -> Dict[str, Any]:
        """Get portfolio recommendations for daily trading"""

        if not self.current_portfolio:
            return {"status": "No portfolio optimized yet"}

        # Sort by weight
        sorted_positions = sorted(
            self.current_portfolio.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        recommendations = {
            'long_positions': [(s, w) for s, w in sorted_positions if w > 0.01],
            'short_positions': [(s, w) for s, w in sorted_positions if w < -0.01],
            'total_long': sum(w for _, w in sorted_positions if w > 0),
            'total_short': abs(sum(w for _, w in sorted_positions if w < 0)),
            'net_exposure': sum(w for _, w in sorted_positions),
            'n_positions': sum(1 for _, w in sorted_positions if abs(w) > 0.01)
        }

        return recommendations

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of dynamic portfolio optimization"""
    orchestrator = DynamicPortfolioOrchestrator()

    # Generate sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    market_data = {}

    for symbol in symbols:
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        prices = 100 + np.cumsum(np.random.randn(252) * 2)

        df = pd.DataFrame({
            'open': prices + np.random.randn(252),
            'high': prices + np.abs(np.random.randn(252) * 2),
            'low': prices - np.abs(np.random.randn(252) * 2),
            'close': prices,
            'volume': np.random.gamma(2, 1000000, 252)
        }, index=dates)

        market_data[symbol] = df

    # ML predictions (expected daily returns)
    ml_predictions = {
        'AAPL': 0.002,
        'GOOGL': 0.0015,
        'MSFT': 0.0018,
        'AMZN': 0.001,
        'META': 0.0012
    }

    # Run optimization
    print("Running dynamic portfolio optimization...")
    results = await orchestrator.optimize_daily_portfolio(
        market_data,
        ml_predictions
    )

    # Display results
    print("\n" + "="*50)
    print("OPTIMAL DAILY PORTFOLIO")
    print("="*50)

    optimal = results['optimal_portfolio']
    print(f"\nExpected Return: {optimal.expected_return:.2%}")
    print(f"Expected Risk: {optimal.expected_risk:.2%}")
    print(f"Sharpe Ratio: {optimal.sharpe_ratio:.2f}")
    print(f"Diversification Ratio: {optimal.diversification_ratio:.2f}")

    print("\nOptimal Weights:")
    for asset, weight in optimal.weights.items():
        if abs(weight) > 0.01:
            print(f"  {asset}: {weight:.1%}")

    # Risk budget
    risk_budget = results['risk_budget']
    print(f"\nRisk Budget:")
    print(f"  Total Budget: {risk_budget.total_risk_budget:.1%}")
    print(f"  Current Usage: {risk_budget.current_risk_usage:.1%}")
    print(f"  Utilization: {risk_budget.risk_utilization:.1%}")

    # Recommendations
    recommendations = orchestrator.get_portfolio_recommendations()
    print(f"\nTrading Recommendations:")
    print(f"  Long Positions: {len(recommendations['long_positions'])}")
    print(f"  Short Positions: {len(recommendations['short_positions'])}")
    print(f"  Net Exposure: {recommendations['net_exposure']:.1%}")

    if recommendations['long_positions']:
        print(f"\nTop Long Positions:")
        for symbol, weight in recommendations['long_positions'][:3]:
            print(f"    {symbol}: {weight:.1%}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())