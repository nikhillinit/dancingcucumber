"""
Advanced Backtesting Engine with Walk-Forward Analysis
=======================================================
Comprehensive backtesting with Monte Carlo simulation and statistical validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
import ray
import asyncio
from collections import deque
import warnings
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 100000
    commission: float = 0.001
    slippage: float = 0.001
    max_position_size: float = 0.2
    max_positions: int = 10
    rebalance_frequency: str = 'daily'
    risk_free_rate: float = 0.02
    margin_requirement: float = 0.25
    short_selling_allowed: bool = True
    use_stops: bool = True
    stop_loss: float = 0.05
    take_profit: float = 0.10


@dataclass
class BacktestResult:
    """Backtest result with comprehensive metrics"""
    # Returns
    total_return: float
    annual_return: float
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    recovery_time: float
    # Trade statistics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    # Statistical tests
    t_statistic: float
    p_value: float
    skewness: float
    kurtosis: float
    # Additional metrics
    information_ratio: float
    alpha: float
    beta: float
    # Portfolio
    equity_curve: pd.Series
    positions_history: pd.DataFrame
    trades_history: pd.DataFrame


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    in_sample_results: List[BacktestResult]
    out_sample_results: List[BacktestResult]
    optimization_params: List[Dict[str, Any]]
    stability_score: float
    overfitting_score: float
    robustness_score: float


class PortfolioSimulator:
    """Portfolio simulation engine"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset portfolio state"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.equity = self.config.initial_capital
        self.equity_curve = [self.config.initial_capital]
        self.trades = []
        self.position_history = []

    def execute_signal(
        self,
        signal: Dict[str, Any],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute trading signal"""
        symbol = signal['symbol']
        direction = signal['direction']
        size = signal.get('size', 0.1)

        # Calculate position size
        position_value = self.equity * min(size, self.config.max_position_size)

        # Check if we can execute
        if direction == 'long' and self.cash > position_value:
            return self._open_long_position(symbol, position_value, current_prices[symbol])
        elif direction == 'short' and self.config.short_selling_allowed:
            return self._open_short_position(symbol, position_value, current_prices[symbol])
        elif direction == 'close' and symbol in self.positions:
            return self._close_position(symbol, current_prices[symbol])

        return {'executed': False}

    def _open_long_position(self, symbol: str, value: float, price: float) -> Dict[str, Any]:
        """Open long position"""
        # Apply slippage and commission
        actual_price = price * (1 + self.config.slippage)
        commission = value * self.config.commission
        shares = (value - commission) / actual_price

        # Update portfolio
        self.cash -= value
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': actual_price,
            'direction': 'long',
            'value': value,
            'stop_loss': actual_price * (1 - self.config.stop_loss) if self.config.use_stops else None,
            'take_profit': actual_price * (1 + self.config.take_profit) if self.config.use_stops else None
        }

        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'buy',
            'shares': shares,
            'price': actual_price,
            'commission': commission,
            'value': value
        }
        self.trades.append(trade)

        return {'executed': True, 'trade': trade}

    def _open_short_position(self, symbol: str, value: float, price: float) -> Dict[str, Any]:
        """Open short position"""
        # Apply slippage and commission
        actual_price = price * (1 - self.config.slippage)
        commission = value * self.config.commission
        shares = (value - commission) / actual_price

        # Update portfolio
        self.cash += value - commission  # Receive cash from short sale
        self.positions[symbol] = {
            'shares': -shares,
            'entry_price': actual_price,
            'direction': 'short',
            'value': value,
            'stop_loss': actual_price * (1 + self.config.stop_loss) if self.config.use_stops else None,
            'take_profit': actual_price * (1 - self.config.take_profit) if self.config.use_stops else None
        }

        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'sell_short',
            'shares': shares,
            'price': actual_price,
            'commission': commission,
            'value': value
        }
        self.trades.append(trade)

        return {'executed': True, 'trade': trade}

    def _close_position(self, symbol: str, price: float) -> Dict[str, Any]:
        """Close position"""
        position = self.positions[symbol]
        shares = position['shares']

        if shares > 0:  # Long position
            actual_price = price * (1 - self.config.slippage)
            value = shares * actual_price
            commission = value * self.config.commission
            self.cash += value - commission
            action = 'sell'
        else:  # Short position
            actual_price = price * (1 + self.config.slippage)
            value = abs(shares) * actual_price
            commission = value * self.config.commission
            self.cash -= value + commission
            action = 'buy_to_cover'

        # Calculate P&L
        if shares > 0:
            pnl = (actual_price - position['entry_price']) * shares - commission
        else:
            pnl = (position['entry_price'] - actual_price) * abs(shares) - commission

        # Remove position
        del self.positions[symbol]

        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'shares': abs(shares),
            'price': actual_price,
            'commission': commission,
            'pnl': pnl
        }
        self.trades.append(trade)

        return {'executed': True, 'trade': trade, 'pnl': pnl}

    def update_portfolio(self, current_prices: Dict[str, float]):
        """Update portfolio value"""
        portfolio_value = self.cash

        # Check stops and update values
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                shares = position['shares']

                # Check stop loss and take profit
                if self.config.use_stops:
                    if position['direction'] == 'long':
                        if position['stop_loss'] and current_price <= position['stop_loss']:
                            positions_to_close.append(symbol)
                        elif position['take_profit'] and current_price >= position['take_profit']:
                            positions_to_close.append(symbol)
                    else:  # short
                        if position['stop_loss'] and current_price >= position['stop_loss']:
                            positions_to_close.append(symbol)
                        elif position['take_profit'] and current_price <= position['take_profit']:
                            positions_to_close.append(symbol)

                # Update position value
                if shares > 0:
                    portfolio_value += shares * current_price
                else:
                    # Short position: initial value - (current - entry) * shares
                    portfolio_value -= abs(shares) * (current_price - position['entry_price'])

        # Close positions that hit stops
        for symbol in positions_to_close:
            self._close_position(symbol, current_prices[symbol])

        self.equity = portfolio_value
        self.equity_curve.append(portfolio_value)

        # Record position snapshot
        self.position_history.append({
            'timestamp': datetime.now(),
            'equity': self.equity,
            'cash': self.cash,
            'n_positions': len(self.positions),
            'positions': self.positions.copy()
        })


@ray.remote
class WalkForwardAgent:
    """Agent for walk-forward analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.optimization_results = []

    async def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        window_size: int = 252,
        step_size: int = 63,
        optimization_metric: str = 'sharpe_ratio'
    ) -> WalkForwardResult:
        """Run walk-forward analysis"""

        in_sample_results = []
        out_sample_results = []
        optimization_params = []

        # Walk forward through data
        for start in range(0, len(data) - window_size - step_size, step_size):
            # In-sample period
            in_sample_data = data.iloc[start:start + window_size]

            # Out-of-sample period
            out_sample_data = data.iloc[start + window_size:start + window_size + step_size]

            # Optimize parameters on in-sample
            best_params = await self._optimize_parameters(
                in_sample_data,
                strategy_func,
                param_ranges,
                optimization_metric
            )
            optimization_params.append(best_params)

            # Test on in-sample
            in_sample_result = await self._backtest_strategy(
                in_sample_data,
                strategy_func,
                best_params
            )
            in_sample_results.append(in_sample_result)

            # Test on out-of-sample
            out_sample_result = await self._backtest_strategy(
                out_sample_data,
                strategy_func,
                best_params
            )
            out_sample_results.append(out_sample_result)

        # Calculate walk-forward metrics
        stability_score = self._calculate_stability(in_sample_results, out_sample_results)
        overfitting_score = self._calculate_overfitting(in_sample_results, out_sample_results)
        robustness_score = self._calculate_robustness(out_sample_results)

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_sample_results=out_sample_results,
            optimization_params=optimization_params,
            stability_score=stability_score,
            overfitting_score=overfitting_score,
            robustness_score=robustness_score
        )

    async def _optimize_parameters(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        metric: str
    ) -> Dict[str, float]:
        """Optimize strategy parameters"""

        def objective(params):
            # Convert array to dict
            param_dict = {
                key: params[i]
                for i, key in enumerate(param_ranges.keys())
            }

            # Run backtest
            result = asyncio.run(self._backtest_strategy(data, strategy_func, param_dict))

            # Return negative metric for minimization
            return -getattr(result, metric, 0)

        # Initial guess
        x0 = [(r[0] + r[1]) / 2 for r in param_ranges.values()]

        # Bounds
        bounds = list(param_ranges.values())

        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        # Convert back to dict
        best_params = {
            key: result.x[i]
            for i, key in enumerate(param_ranges.keys())
        }

        return best_params

    async def _backtest_strategy(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict[str, float]
    ) -> BacktestResult:
        """Run single backtest"""

        config = BacktestConfig()
        simulator = PortfolioSimulator(config)

        # Generate signals
        signals = strategy_func(data, **params)

        # Simulate trading
        for i, (timestamp, row) in enumerate(data.iterrows()):
            if i < len(signals):
                signal = signals[i]
                current_prices = {'symbol': row['close']}
                simulator.execute_signal(signal, current_prices)

            # Update portfolio
            simulator.update_portfolio({'symbol': row['close']})

        # Calculate metrics
        returns = pd.Series(simulator.equity_curve).pct_change().dropna()

        return BacktestResult(
            total_return=(simulator.equity / config.initial_capital - 1),
            annual_return=returns.mean() * 252,
            volatility=returns.std() * np.sqrt(252),
            sharpe_ratio=(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            sortino_ratio=self._calculate_sortino(returns),
            calmar_ratio=self._calculate_calmar(returns, simulator.equity_curve),
            max_drawdown=self._calculate_max_drawdown(simulator.equity_curve),
            max_drawdown_duration=0,  # Simplified
            recovery_time=0,  # Simplified
            total_trades=len(simulator.trades),
            win_rate=self._calculate_win_rate(simulator.trades),
            avg_win=0,  # Simplified
            avg_loss=0,  # Simplified
            profit_factor=1.0,  # Simplified
            t_statistic=0,  # Simplified
            p_value=0,  # Simplified
            skewness=returns.skew(),
            kurtosis=returns.kurtosis(),
            information_ratio=0,  # Simplified
            alpha=0,  # Simplified
            beta=0,  # Simplified
            equity_curve=pd.Series(simulator.equity_curve),
            positions_history=pd.DataFrame(simulator.position_history),
            trades_history=pd.DataFrame(simulator.trades)
        )

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                return (returns.mean() / downside_std) * np.sqrt(252)
        return 0

    def _calculate_calmar(self, returns: pd.Series, equity_curve: List[float]) -> float:
        """Calculate Calmar ratio"""
        max_dd = self._calculate_max_drawdown(equity_curve)
        if max_dd > 0:
            return (returns.mean() * 252) / max_dd
        return 0

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0

        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        if not trades:
            return 0

        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return wins / len(trades)

    def _calculate_stability(
        self,
        in_sample: List[BacktestResult],
        out_sample: List[BacktestResult]
    ) -> float:
        """Calculate stability score"""
        if not in_sample or not out_sample:
            return 0

        # Compare Sharpe ratios
        in_sharpes = [r.sharpe_ratio for r in in_sample]
        out_sharpes = [r.sharpe_ratio for r in out_sample]

        if len(in_sharpes) > 1 and len(out_sharpes) > 1:
            correlation = np.corrcoef(in_sharpes[:len(out_sharpes)], out_sharpes[:len(in_sharpes)])[0, 1]
            return max(0, correlation)
        return 0

    def _calculate_overfitting(
        self,
        in_sample: List[BacktestResult],
        out_sample: List[BacktestResult]
    ) -> float:
        """Calculate overfitting score (0 = no overfitting, 1 = severe)"""
        if not in_sample or not out_sample:
            return 0

        in_avg = np.mean([r.sharpe_ratio for r in in_sample])
        out_avg = np.mean([r.sharpe_ratio for r in out_sample])

        if in_avg > 0:
            degradation = (in_avg - out_avg) / in_avg
            return max(0, min(1, degradation))
        return 0

    def _calculate_robustness(self, out_sample: List[BacktestResult]) -> float:
        """Calculate robustness score"""
        if not out_sample:
            return 0

        # Check consistency of out-of-sample results
        sharpes = [r.sharpe_ratio for r in out_sample]
        if len(sharpes) > 1:
            # Low variance = high robustness
            cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) > 0 else 1
            return max(0, 1 - cv)
        return 0.5


@ray.remote
class MonteCarloAgent:
    """Agent for Monte Carlo simulation"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def run_monte_carlo(
        self,
        base_returns: pd.Series,
        n_simulations: int = 1000,
        n_periods: int = 252
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""

        # Extract return characteristics
        mean_return = base_returns.mean()
        std_return = base_returns.std()
        skew = base_returns.skew()
        kurt = base_returns.kurtosis()

        # Run simulations
        simulation_results = []

        for _ in range(n_simulations):
            # Generate returns with similar characteristics
            if abs(skew) > 0.5 or abs(kurt) > 1:
                # Use bootstrapping for non-normal distributions
                simulated_returns = np.random.choice(base_returns, size=n_periods, replace=True)
            else:
                # Use normal distribution
                simulated_returns = np.random.normal(mean_return, std_return, n_periods)

            # Calculate portfolio value
            portfolio_value = 100000 * np.exp(np.cumsum(simulated_returns))
            final_value = portfolio_value[-1]

            simulation_results.append({
                'final_value': final_value,
                'max_drawdown': self._calculate_path_drawdown(portfolio_value),
                'volatility': np.std(simulated_returns) * np.sqrt(252)
            })

        # Calculate statistics
        final_values = [r['final_value'] for r in simulation_results]
        drawdowns = [r['max_drawdown'] for r in simulation_results]

        return {
            'expected_value': np.mean(final_values),
            'median_value': np.median(final_values),
            'var_95': np.percentile(final_values, 5),
            'cvar_95': np.mean([v for v in final_values if v <= np.percentile(final_values, 5)]),
            'probability_of_profit': sum(1 for v in final_values if v > 100000) / n_simulations,
            'expected_max_drawdown': np.mean(drawdowns),
            'worst_case_drawdown': np.max(drawdowns),
            'percentiles': {
                '5%': np.percentile(final_values, 5),
                '25%': np.percentile(final_values, 25),
                '50%': np.percentile(final_values, 50),
                '75%': np.percentile(final_values, 75),
                '95%': np.percentile(final_values, 95)
            }
        }

    def _calculate_path_drawdown(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown for a path"""
        peak = values[0]
        max_dd = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd


class AdvancedBacktestOrchestrator:
    """Orchestrate advanced backtesting with multiple agents"""

    def __init__(self):
        ray.init(ignore_reinit_error=True)

        # Initialize agents
        self.walk_forward_agents = [
            WalkForwardAgent.remote(f"wf_{i}") for i in range(3)
        ]
        self.monte_carlo_agents = [
            MonteCarloAgent.remote(f"mc_{i}") for i in range(3)
        ]

        self.backtest_results = {}
        self.monte_carlo_results = {}

    async def run_comprehensive_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        param_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Run comprehensive backtesting suite"""

        backtest_tasks = []

        # Walk-forward analysis for each asset
        for i, (symbol, df) in enumerate(data.items()):
            agent = self.walk_forward_agents[i % len(self.walk_forward_agents)]

            task = agent.run_walk_forward.remote(
                df,
                strategy_func,
                param_ranges,
                window_size=252,
                step_size=63
            )
            backtest_tasks.append((symbol, 'walk_forward', task))

        # Gather walk-forward results
        walk_forward_results = {}
        for symbol, analysis_type, task in backtest_tasks:
            if analysis_type == 'walk_forward':
                result = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, task)
                )
                walk_forward_results[symbol] = result

        # Run Monte Carlo simulations
        monte_carlo_tasks = []

        for symbol, wf_result in walk_forward_results.items():
            if wf_result.out_sample_results:
                # Get returns from out-of-sample results
                returns = pd.concat([
                    r.equity_curve.pct_change().dropna()
                    for r in wf_result.out_sample_results
                ])

                agent = self.monte_carlo_agents[len(monte_carlo_tasks) % len(self.monte_carlo_agents)]
                task = agent.run_monte_carlo.remote(returns, n_simulations=1000)
                monte_carlo_tasks.append((symbol, task))

        # Gather Monte Carlo results
        monte_carlo_results = {}
        for symbol, task in monte_carlo_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )
            monte_carlo_results[symbol] = result

        # Statistical validation
        validation_results = self._run_statistical_tests(walk_forward_results)

        return {
            'walk_forward': walk_forward_results,
            'monte_carlo': monte_carlo_results,
            'statistical_validation': validation_results,
            'summary': self._generate_summary(walk_forward_results, monte_carlo_results)
        }

    def _run_statistical_tests(
        self,
        results: Dict[str, WalkForwardResult]
    ) -> Dict[str, Any]:
        """Run statistical validation tests"""

        tests = {}

        for symbol, wf_result in results.items():
            if wf_result.out_sample_results:
                returns = []
                for result in wf_result.out_sample_results:
                    if len(result.equity_curve) > 1:
                        returns.extend(result.equity_curve.pct_change().dropna().tolist())

                if len(returns) > 30:
                    # T-test for positive returns
                    t_stat, p_value = stats.ttest_1samp(returns, 0)

                    # Jarque-Bera test for normality
                    jb_stat, jb_pvalue = stats.jarque_bera(returns)

                    # Ljung-Box test for autocorrelation
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_result = acorr_ljungbox(returns, lags=10)

                    tests[symbol] = {
                        't_test': {'statistic': t_stat, 'p_value': p_value},
                        'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue},
                        'ljung_box': {'statistic': lb_result.lb_stat.values[0], 'p_value': lb_result.lb_pvalue.values[0]},
                        'significant_alpha': p_value < 0.05
                    }

        return tests

    def _generate_summary(
        self,
        walk_forward: Dict[str, WalkForwardResult],
        monte_carlo: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Generate summary statistics"""

        summary = {
            'best_symbol': None,
            'best_sharpe': 0,
            'avg_stability': 0,
            'avg_overfitting': 0,
            'avg_robustness': 0,
            'monte_carlo_profit_probability': 0
        }

        # Walk-forward summary
        stabilities = []
        overfittings = []
        robustnesses = []

        for symbol, wf_result in walk_forward.items():
            stabilities.append(wf_result.stability_score)
            overfittings.append(wf_result.overfitting_score)
            robustnesses.append(wf_result.robustness_score)

            # Find best Sharpe
            if wf_result.out_sample_results:
                avg_sharpe = np.mean([r.sharpe_ratio for r in wf_result.out_sample_results])
                if avg_sharpe > summary['best_sharpe']:
                    summary['best_sharpe'] = avg_sharpe
                    summary['best_symbol'] = symbol

        summary['avg_stability'] = np.mean(stabilities) if stabilities else 0
        summary['avg_overfitting'] = np.mean(overfittings) if overfittings else 0
        summary['avg_robustness'] = np.mean(robustnesses) if robustnesses else 0

        # Monte Carlo summary
        if monte_carlo:
            profit_probs = [mc.get('probability_of_profit', 0) for mc in monte_carlo.values()]
            summary['monte_carlo_profit_probability'] = np.mean(profit_probs)

        return summary

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example strategy function
def example_strategy(data: pd.DataFrame, **params) -> List[Dict[str, Any]]:
    """Example strategy for backtesting"""
    signals = []

    # Simple MA crossover
    short_ma = data['close'].rolling(params.get('short_period', 20)).mean()
    long_ma = data['close'].rolling(params.get('long_period', 50)).mean()

    for i in range(len(data)):
        if i < params.get('long_period', 50):
            signals.append({'symbol': 'TEST', 'direction': 'hold', 'size': 0})
        elif short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
            signals.append({'symbol': 'TEST', 'direction': 'long', 'size': 0.1})
        elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
            signals.append({'symbol': 'TEST', 'direction': 'short', 'size': 0.1})
        else:
            signals.append({'symbol': 'TEST', 'direction': 'hold', 'size': 0})

    return signals


# Example usage
async def main():
    """Example usage of advanced backtesting"""
    orchestrator = AdvancedBacktestOrchestrator()

    # Generate sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = {}

    for symbol in symbols:
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
        prices = 100 + np.cumsum(np.random.randn(1000) * 2)

        df = pd.DataFrame({
            'open': prices + np.random.randn(1000),
            'high': prices + np.abs(np.random.randn(1000) * 2),
            'low': prices - np.abs(np.random.randn(1000) * 2),
            'close': prices,
            'volume': np.random.gamma(2, 1000000, 1000)
        }, index=dates)

        data[symbol] = df

    # Define parameter ranges for optimization
    param_ranges = {
        'short_period': (10, 30),
        'long_period': (40, 100)
    }

    # Run comprehensive backtest
    print("Running comprehensive backtest...")
    results = await orchestrator.run_comprehensive_backtest(
        data,
        example_strategy,
        param_ranges
    )

    # Display results
    print("\nWalk-Forward Analysis Results:")
    for symbol, wf_result in results['walk_forward'].items():
        print(f"\n{symbol}:")
        print(f"  Stability Score: {wf_result.stability_score:.3f}")
        print(f"  Overfitting Score: {wf_result.overfitting_score:.3f}")
        print(f"  Robustness Score: {wf_result.robustness_score:.3f}")

    print("\nMonte Carlo Simulation Results:")
    for symbol, mc_result in results['monte_carlo'].items():
        print(f"\n{symbol}:")
        print(f"  Expected Value: ${mc_result['expected_value']:,.0f}")
        print(f"  VaR 95%: ${mc_result['var_95']:,.0f}")
        print(f"  Probability of Profit: {mc_result['probability_of_profit']:.1%}")

    print("\nSummary:")
    summary = results['summary']
    print(f"  Best Symbol: {summary['best_symbol']}")
    print(f"  Best Sharpe: {summary['best_sharpe']:.2f}")
    print(f"  Avg Stability: {summary['avg_stability']:.1%}")
    print(f"  Avg Overfitting: {summary['avg_overfitting']:.1%}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())