"""
Execution Algorithms Suite with Multi-Agent Processing
======================================================
Advanced execution algorithms for optimal trade execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import ray
import asyncio
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Trading order specification"""
    symbol: str
    side: str  # buy, sell
    quantity: float
    order_type: str  # market, limit, stop
    limit_price: Optional[float]
    time_in_force: str  # GTC, IOC, FOK, GTD
    algo_strategy: str  # VWAP, TWAP, IS, POV, etc.
    start_time: datetime
    end_time: datetime
    urgency: float  # 0-1 scale
    constraints: Dict[str, Any]


@dataclass
class ExecutionSlice:
    """Single execution slice"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    venue: str  # exchange, dark pool, etc.
    fees: float
    market_impact: float


@dataclass
class ExecutionPlan:
    """Complete execution plan"""
    order: Order
    slices: List[Dict[str, Any]]
    expected_cost: float
    expected_slippage: float
    expected_market_impact: float
    risk_score: float
    strategy_params: Dict[str, Any]


@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    order_id: str
    symbol: str
    total_quantity: float
    avg_price: float
    benchmark_price: float  # arrival price, VWAP, etc.
    slippage: float
    market_impact: float
    implementation_shortfall: float
    total_fees: float
    execution_time: float
    timestamp: datetime


@ray.remote
class VWAPAgent:
    """Volume Weighted Average Price execution agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.historical_volumes = {}

    async def create_vwap_schedule(
        self,
        order: Order,
        volume_profile: pd.DataFrame,
        participation_rate: float = 0.1
    ) -> ExecutionPlan:
        """Create VWAP execution schedule"""
        # Analyze historical volume pattern
        if 'volume' in volume_profile.columns:
            volumes = volume_profile['volume'].values
        else:
            volumes = np.random.gamma(2, 1000, len(volume_profile))

        # Calculate time buckets
        total_time = (order.end_time - order.start_time).total_seconds() / 60
        n_buckets = min(int(total_time / 5), 20)  # 5-minute buckets

        # Create volume distribution
        volume_dist = self._calculate_volume_distribution(volumes, n_buckets)

        # Generate execution slices
        slices = []
        remaining_qty = order.quantity
        current_time = order.start_time

        for i, vol_pct in enumerate(volume_dist):
            slice_qty = min(order.quantity * vol_pct, remaining_qty)

            if slice_qty > 0:
                slice_time = current_time + timedelta(minutes=5*i)

                slices.append({
                    'time': slice_time,
                    'quantity': slice_qty,
                    'min_price': None,
                    'max_price': None,
                    'urgency': order.urgency * (1 + i/n_buckets),  # Increase urgency over time
                    'venue_preference': 'lit' if vol_pct > 0.05 else 'dark'
                })

                remaining_qty -= slice_qty

        # Calculate expected metrics
        expected_slippage = self._estimate_slippage(order, participation_rate)
        expected_impact = self._estimate_market_impact(order, participation_rate)

        return ExecutionPlan(
            order=order,
            slices=slices,
            expected_cost=order.quantity * expected_slippage,
            expected_slippage=expected_slippage,
            expected_market_impact=expected_impact,
            risk_score=expected_impact / 0.01,  # Normalize to basis points
            strategy_params={
                'participation_rate': participation_rate,
                'n_slices': len(slices),
                'adaptive': True
            }
        )

    def _calculate_volume_distribution(
        self,
        volumes: np.ndarray,
        n_buckets: int
    ) -> np.ndarray:
        """Calculate intraday volume distribution"""
        # Simple U-shape distribution (realistic for equities)
        x = np.linspace(0, 1, n_buckets)
        u_shape = 1.5 - 2 * np.abs(x - 0.5)
        u_shape = u_shape / u_shape.sum()

        # Add some noise
        noise = np.random.normal(0, 0.02, n_buckets)
        distribution = np.maximum(u_shape + noise, 0.01)
        distribution = distribution / distribution.sum()

        return distribution

    def _estimate_slippage(self, order: Order, participation_rate: float) -> float:
        """Estimate expected slippage"""
        # Simplified slippage model
        base_slippage = 0.0005  # 5 bps base
        urgency_factor = 1 + order.urgency
        size_factor = 1 + np.log1p(order.quantity / 10000)
        participation_factor = 1 + participation_rate * 2

        return base_slippage * urgency_factor * size_factor * participation_factor

    def _estimate_market_impact(self, order: Order, participation_rate: float) -> float:
        """Estimate market impact using square-root model"""
        # Almgren-Chriss square-root impact model
        daily_volume = 1000000  # Assumed average daily volume
        sigma = 0.02  # Daily volatility

        # Permanent impact
        permanent_impact = 0.1 * sigma * np.sqrt(order.quantity / daily_volume)

        # Temporary impact
        temporary_impact = 0.3 * sigma * np.sqrt(participation_rate)

        return permanent_impact + temporary_impact


@ray.remote
class TWAPAgent:
    """Time Weighted Average Price execution agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def create_twap_schedule(
        self,
        order: Order,
        randomize: bool = True
    ) -> ExecutionPlan:
        """Create TWAP execution schedule"""
        # Calculate time slices
        total_time = (order.end_time - order.start_time).total_seconds() / 60
        n_slices = min(int(total_time / 2), 30)  # 2-minute slices

        # Equal distribution with optional randomization
        if randomize:
            # Add 20% randomization to avoid detection
            weights = np.random.uniform(0.8, 1.2, n_slices)
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_slices) / n_slices

        # Generate slices
        slices = []
        current_time = order.start_time

        for i, weight in enumerate(weights):
            slice_time = current_time + timedelta(minutes=2*i)
            slice_qty = order.quantity * weight

            slices.append({
                'time': slice_time,
                'quantity': slice_qty,
                'min_price': None,
                'max_price': None,
                'urgency': order.urgency,
                'venue_preference': 'any'
            })

        # Simple cost estimates
        expected_slippage = 0.0003 * (1 + order.urgency)
        expected_impact = 0.0002 * np.sqrt(order.quantity / 10000)

        return ExecutionPlan(
            order=order,
            slices=slices,
            expected_cost=order.quantity * expected_slippage,
            expected_slippage=expected_slippage,
            expected_market_impact=expected_impact,
            risk_score=0.3,
            strategy_params={
                'n_slices': n_slices,
                'randomized': randomize,
                'slice_duration': 2  # minutes
            }
        )


@ray.remote
class ImplementationShortfallAgent:
    """Implementation Shortfall minimization agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def optimize_execution(
        self,
        order: Order,
        market_params: Dict[str, float]
    ) -> ExecutionPlan:
        """Optimize execution to minimize implementation shortfall"""
        # Almgren-Chriss optimal execution
        T = (order.end_time - order.start_time).total_seconds() / 3600  # Hours
        n_slices = min(int(T * 12), 20)  # 5-minute slices

        # Market parameters
        sigma = market_params.get('volatility', 0.02)
        lambda_t = market_params.get('temporary_impact', 0.1)
        eta = market_params.get('permanent_impact', 0.05)
        gamma = market_params.get('risk_aversion', 1.0)

        # Optimize trajectory
        trajectory = self._optimize_trajectory(
            order.quantity, n_slices, T, sigma, lambda_t, eta, gamma
        )

        # Convert to slices
        slices = []
        current_time = order.start_time

        for i, (qty, urgency) in enumerate(trajectory):
            slice_time = current_time + timedelta(hours=T*i/n_slices)

            slices.append({
                'time': slice_time,
                'quantity': qty,
                'min_price': None,
                'max_price': None,
                'urgency': urgency,
                'venue_preference': 'smart'  # Smart order routing
            })

        # Calculate expected costs
        expected_cost = self._calculate_expected_cost(
            trajectory, sigma, lambda_t, eta, T
        )

        return ExecutionPlan(
            order=order,
            slices=slices,
            expected_cost=expected_cost['total'],
            expected_slippage=expected_cost['slippage'],
            expected_market_impact=expected_cost['impact'],
            risk_score=expected_cost['risk'],
            strategy_params={
                'optimization': 'almgren_chriss',
                'risk_aversion': gamma,
                'adaptive': True
            }
        )

    def _optimize_trajectory(
        self,
        total_qty: float,
        n_slices: int,
        T: float,
        sigma: float,
        lambda_t: float,
        eta: float,
        gamma: float
    ) -> List[Tuple[float, float]]:
        """Optimize execution trajectory using Almgren-Chriss"""
        # Simplified optimal trajectory
        kappa = np.sqrt(lambda_t * gamma * sigma**2)
        tau = T / n_slices

        # Exponential decay trajectory
        trajectory = []
        remaining = total_qty

        for i in range(n_slices):
            # Optimal trade size
            if i < n_slices - 1:
                trade_size = remaining * (1 - np.exp(-kappa * tau))
                urgency = 0.5 + 0.5 * (i / n_slices)
            else:
                trade_size = remaining  # Complete remaining
                urgency = 1.0

            trajectory.append((trade_size, urgency))
            remaining -= trade_size

        return trajectory

    def _calculate_expected_cost(
        self,
        trajectory: List[Tuple[float, float]],
        sigma: float,
        lambda_t: float,
        eta: float,
        T: float
    ) -> Dict[str, float]:
        """Calculate expected execution costs"""
        total_qty = sum(qty for qty, _ in trajectory)
        n = len(trajectory)

        # Permanent impact cost
        permanent_cost = eta * total_qty**2 / 2

        # Temporary impact cost
        temporary_cost = sum(
            lambda_t * qty**2 / n
            for qty, _ in trajectory
        )

        # Volatility risk
        risk = sigma * np.sqrt(T) * total_qty * 0.1

        return {
            'total': permanent_cost + temporary_cost + risk,
            'slippage': (permanent_cost + temporary_cost) / total_qty,
            'impact': permanent_cost / total_qty,
            'risk': risk / total_qty
        }


@ray.remote
class SmartOrderRoutingAgent:
    """Smart Order Routing agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.venue_statistics = {}

    async def route_order(
        self,
        order_slice: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine optimal routing for order slice"""
        venues = self._get_available_venues(order_slice)
        venue_scores = {}

        for venue in venues:
            score = await self._score_venue(
                venue,
                order_slice,
                market_data
            )
            venue_scores[venue] = score

        # Select optimal venue(s)
        sorted_venues = sorted(
            venue_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )

        # Split across multiple venues if beneficial
        routing_plan = self._create_routing_plan(
            order_slice,
            sorted_venues,
            market_data
        )

        return routing_plan

    def _get_available_venues(self, order_slice: Dict[str, Any]) -> List[str]:
        """Get list of available execution venues"""
        base_venues = ['NYSE', 'NASDAQ', 'BATS', 'IEX']

        if order_slice.get('venue_preference') == 'dark':
            base_venues.extend(['SIGMA', 'CROSSFINDER', 'MS_POOL'])
        elif order_slice.get('venue_preference') == 'smart':
            base_venues.extend(['EDGX', 'ARCA', 'BZX'])

        return base_venues

    async def _score_venue(
        self,
        venue: str,
        order_slice: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Score venue for order execution"""
        # Scoring factors
        liquidity_score = np.random.uniform(0.5, 1.0)  # Simulated
        cost_score = 1.0 - np.random.uniform(0.0001, 0.001)  # Fee structure
        speed_score = 1.0 - np.random.uniform(0.001, 0.01)  # Latency
        fill_rate_score = np.random.uniform(0.7, 0.99)  # Historical fill rate

        # Dark pool specific
        if 'POOL' in venue or venue in ['SIGMA', 'CROSSFINDER']:
            information_leakage_score = 0.9  # Lower info leakage
            size_score = 1.0 if order_slice['quantity'] > 1000 else 0.5
        else:
            information_leakage_score = 0.6
            size_score = 0.8

        # Weight factors based on order characteristics
        if order_slice.get('urgency', 0.5) > 0.7:
            weights = {'speed': 0.4, 'liquidity': 0.3, 'fill': 0.2, 'cost': 0.1}
        else:
            weights = {'cost': 0.3, 'liquidity': 0.3, 'info': 0.2, 'fill': 0.2}

        total_score = (
            weights.get('liquidity', 0.25) * liquidity_score +
            weights.get('cost', 0.25) * cost_score +
            weights.get('speed', 0.25) * speed_score +
            weights.get('fill', 0.15) * fill_rate_score +
            weights.get('info', 0.1) * information_leakage_score
        )

        return {
            'total_score': total_score,
            'liquidity': liquidity_score,
            'cost': cost_score,
            'speed': speed_score,
            'fill_rate': fill_rate_score,
            'info_leakage': information_leakage_score
        }

    def _create_routing_plan(
        self,
        order_slice: Dict[str, Any],
        sorted_venues: List[Tuple[str, Dict[str, float]]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create multi-venue routing plan"""
        total_qty = order_slice['quantity']
        routing = []

        # Use top 3 venues with decreasing allocation
        allocations = [0.6, 0.3, 0.1]

        for i, (venue, scores) in enumerate(sorted_venues[:3]):
            if i < len(allocations):
                venue_qty = total_qty * allocations[i]

                routing.append({
                    'venue': venue,
                    'quantity': venue_qty,
                    'expected_fill_rate': scores['fill_rate'],
                    'expected_cost': venue_qty * 0.0001 * (2 - scores['cost']),
                    'priority': i + 1
                })

        return {
            'order_slice': order_slice,
            'routing': routing,
            'expected_fill': sum(r['quantity'] * r['expected_fill_rate'] for r in routing),
            'expected_total_cost': sum(r['expected_cost'] for r in routing),
            'timestamp': datetime.now()
        }


@ray.remote
class DarkPoolAggregatorAgent:
    """Dark Pool aggregation agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.dark_pools = [
            'SIGMA', 'CROSSFINDER', 'MS_POOL', 'UBS_ATS',
            'BARX', 'JPM_X', 'CITI_MATCH', 'LEVEL_ATS'
        ]

    async def find_dark_liquidity(
        self,
        order: Order,
        min_size: float = 100
    ) -> List[Dict[str, Any]]:
        """Find available dark pool liquidity"""
        opportunities = []

        for pool in self.dark_pools:
            # Simulate liquidity check
            available_liquidity = await self._check_pool_liquidity(
                pool, order.symbol
            )

            if available_liquidity > min_size:
                opportunity = {
                    'venue': pool,
                    'symbol': order.symbol,
                    'side': order.side,
                    'available_qty': available_liquidity,
                    'min_qty': min_size,
                    'price_improvement': np.random.uniform(0.0001, 0.0005),
                    'info_leakage_risk': self._assess_info_leakage_risk(pool),
                    'timestamp': datetime.now()
                }
                opportunities.append(opportunity)

        # Sort by quality score
        opportunities.sort(
            key=lambda x: x['price_improvement'] - x['info_leakage_risk'],
            reverse=True
        )

        return opportunities

    async def _check_pool_liquidity(
        self,
        pool: str,
        symbol: str
    ) -> float:
        """Check available liquidity in dark pool"""
        # Simulated liquidity (in production, would query actual pool)
        base_liquidity = np.random.gamma(2, 5000)

        # Adjust for pool characteristics
        if pool in ['SIGMA', 'CROSSFINDER']:
            liquidity_multiplier = 1.5  # Larger pools
        elif pool in ['LEVEL_ATS', 'CITI_MATCH']:
            liquidity_multiplier = 0.8
        else:
            liquidity_multiplier = 1.0

        return base_liquidity * liquidity_multiplier

    def _assess_info_leakage_risk(self, pool: str) -> float:
        """Assess information leakage risk for dark pool"""
        # Risk scores (0-1, lower is better)
        risk_scores = {
            'IEX': 0.1,  # Very low
            'SIGMA': 0.2,
            'CROSSFINDER': 0.25,
            'MS_POOL': 0.3,
            'UBS_ATS': 0.35,
            'BARX': 0.35,
            'JPM_X': 0.3,
            'CITI_MATCH': 0.4,
            'LEVEL_ATS': 0.3
        }
        return risk_scores.get(pool, 0.5)


class ExecutionOrchestrator:
    """Orchestrate execution algorithms"""

    def __init__(self):
        ray.init(ignore_reinit_error=True)

        self.vwap_agent = VWAPAgent.remote("vwap")
        self.twap_agent = TWAPAgent.remote("twap")
        self.is_agent = ImplementationShortfallAgent.remote("is")
        self.sor_agent = SmartOrderRoutingAgent.remote("sor")
        self.dark_pool_agent = DarkPoolAggregatorAgent.remote("dark")

        self.active_orders = {}
        self.execution_history = deque(maxlen=1000)

    async def create_execution_plan(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> ExecutionPlan:
        """Create optimal execution plan based on order characteristics"""
        # Select strategy based on order
        if order.algo_strategy == 'VWAP':
            plan = await self._execute_vwap(order, market_data)
        elif order.algo_strategy == 'TWAP':
            plan = await self._execute_twap(order)
        elif order.algo_strategy == 'IS':
            plan = await self._execute_is(order, market_data)
        else:
            # Default to VWAP
            plan = await self._execute_vwap(order, market_data)

        # Check for dark pool opportunities
        if order.quantity > 1000:
            dark_task = self.dark_pool_agent.find_dark_liquidity.remote(order)
            dark_opportunities = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, dark_task)
            )

            if dark_opportunities:
                plan = self._incorporate_dark_liquidity(plan, dark_opportunities)

        return plan

    async def _execute_vwap(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> ExecutionPlan:
        """Execute using VWAP strategy"""
        volume_profile = market_data.get('volume_profile', pd.DataFrame())

        vwap_task = self.vwap_agent.create_vwap_schedule.remote(
            order, volume_profile, participation_rate=0.1
        )

        plan = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, vwap_task)
        )

        return plan

    async def _execute_twap(self, order: Order) -> ExecutionPlan:
        """Execute using TWAP strategy"""
        twap_task = self.twap_agent.create_twap_schedule.remote(
            order, randomize=True
        )

        plan = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, twap_task)
        )

        return plan

    async def _execute_is(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> ExecutionPlan:
        """Execute using Implementation Shortfall strategy"""
        market_params = {
            'volatility': market_data.get('volatility', 0.02),
            'temporary_impact': 0.1,
            'permanent_impact': 0.05,
            'risk_aversion': 1.0
        }

        is_task = self.is_agent.optimize_execution.remote(
            order, market_params
        )

        plan = await asyncio.wrap_future(
            asyncio.get_event_loop().run_in_executor(None, ray.get, is_task)
        )

        return plan

    def _incorporate_dark_liquidity(
        self,
        plan: ExecutionPlan,
        dark_opportunities: List[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Incorporate dark pool liquidity into execution plan"""
        if not dark_opportunities:
            return plan

        # Allocate up to 30% to dark pools
        dark_allocation = min(plan.order.quantity * 0.3,
                             sum(d['available_qty'] for d in dark_opportunities[:3]))

        if dark_allocation > 100:
            # Reduce lit market slices proportionally
            reduction_factor = (plan.order.quantity - dark_allocation) / plan.order.quantity

            for slice_data in plan.slices:
                slice_data['quantity'] *= reduction_factor

            # Add dark pool slice at the beginning
            dark_slice = {
                'time': plan.order.start_time,
                'quantity': dark_allocation,
                'venue_preference': 'dark',
                'pools': [d['venue'] for d in dark_opportunities[:3]],
                'urgency': 0.3
            }

            plan.slices.insert(0, dark_slice)

            # Update expected costs
            price_improvement = np.mean([d['price_improvement'] for d in dark_opportunities[:3]])
            plan.expected_cost *= (1 - price_improvement * dark_allocation / plan.order.quantity)

        return plan

    async def execute_order(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> ExecutionMetrics:
        """Execute order and return metrics"""
        # Create execution plan
        plan = await self.create_execution_plan(order, market_data)

        # Simulate execution
        executions = []
        total_quantity = 0
        total_cost = 0
        arrival_price = market_data.get('current_price', 100)

        for slice_data in plan.slices:
            # Route slice
            sor_task = self.sor_agent.route_order.remote(
                slice_data, market_data
            )
            routing = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, sor_task)
            )

            # Simulate fills
            for route in routing['routing']:
                fill_price = arrival_price * (1 + np.random.normal(0, 0.001))
                fill_qty = route['quantity'] * route['expected_fill_rate']

                execution = ExecutionSlice(
                    order_id=f"{order.symbol}_{datetime.now().timestamp()}",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    timestamp=datetime.now(),
                    venue=route['venue'],
                    fees=route['expected_cost'],
                    market_impact=plan.expected_market_impact * fill_qty / order.quantity
                )

                executions.append(execution)
                total_quantity += fill_qty
                total_cost += fill_qty * fill_price + route['expected_cost']

        # Calculate metrics
        avg_price = total_cost / total_quantity if total_quantity > 0 else arrival_price
        slippage = (avg_price - arrival_price) / arrival_price
        implementation_shortfall = slippage + plan.expected_market_impact

        metrics = ExecutionMetrics(
            order_id=f"{order.symbol}_{order.start_time}",
            symbol=order.symbol,
            total_quantity=total_quantity,
            avg_price=avg_price,
            benchmark_price=arrival_price,
            slippage=slippage,
            market_impact=plan.expected_market_impact,
            implementation_shortfall=implementation_shortfall,
            total_fees=sum(e.fees for e in executions),
            execution_time=(datetime.now() - order.start_time).total_seconds(),
            timestamp=datetime.now()
        )

        self.execution_history.append(metrics)
        return metrics

    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get execution performance analytics"""
        if not self.execution_history:
            return {}

        metrics_list = list(self.execution_history)

        return {
            'total_orders': len(metrics_list),
            'avg_slippage': np.mean([m.slippage for m in metrics_list]),
            'avg_market_impact': np.mean([m.market_impact for m in metrics_list]),
            'avg_implementation_shortfall': np.mean([m.implementation_shortfall for m in metrics_list]),
            'total_fees': sum(m.total_fees for m in metrics_list),
            'avg_fill_rate': np.mean([m.total_quantity for m in metrics_list]),
            'timestamp': datetime.now()
        }

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of execution algorithms"""
    orchestrator = ExecutionOrchestrator()

    # Create sample order
    order = Order(
        symbol='AAPL',
        side='buy',
        quantity=10000,
        order_type='limit',
        limit_price=150.00,
        time_in_force='GTC',
        algo_strategy='VWAP',
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2),
        urgency=0.6,
        constraints={'max_participation': 0.15}
    )

    # Market data
    market_data = {
        'current_price': 149.50,
        'volatility': 0.02,
        'volume_profile': pd.DataFrame({
            'volume': np.random.gamma(2, 100000, 100)
        })
    }

    # Execute order
    metrics = await orchestrator.execute_order(order, market_data)

    print("Execution Complete")
    print("=" * 50)
    print(f"Symbol: {metrics.symbol}")
    print(f"Quantity Filled: {metrics.total_quantity:,.0f}")
    print(f"Average Price: ${metrics.avg_price:.2f}")
    print(f"Arrival Price: ${metrics.benchmark_price:.2f}")
    print(f"Slippage: {metrics.slippage:.2%}")
    print(f"Market Impact: {metrics.market_impact:.2%}")
    print(f"Implementation Shortfall: {metrics.implementation_shortfall:.2%}")
    print(f"Total Fees: ${metrics.total_fees:.2f}")
    print(f"Execution Time: {metrics.execution_time:.1f} seconds")

    # Analytics
    analytics = orchestrator.get_execution_analytics()
    print("\nExecution Analytics:")
    for key, value in analytics.items():
        if isinstance(value, float):
            if 'slippage' in key or 'impact' in key or 'shortfall' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())