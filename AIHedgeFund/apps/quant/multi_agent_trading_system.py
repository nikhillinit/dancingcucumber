"""
Multi-Agent Trading System with Specialized Strategies
======================================================
Autonomous agents with different trading personalities and strategies
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import ray
from ray import serve
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import redis
from collections import deque
import json

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRASH = "crash"


class AgentPersonality(Enum):
    """Trading agent personalities"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    CONTRARIAN = "contrarian"
    TREND_FOLLOWER = "trend_follower"
    SCALPER = "scalper"
    SWING_TRADER = "swing_trader"
    VALUE_INVESTOR = "value_investor"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 0


@dataclass
class TradingDecision:
    """Trading decision from an agent"""
    agent_id: str
    symbol: str
    action: str  # buy, sell, hold
    size: float
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float
    time_horizon: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingAgentBase(Agent, ABC):
    """Base class for all trading agents"""

    def __init__(
        self,
        unique_id: str,
        model: 'MultiAgentTradingModel',
        personality: AgentPersonality,
        risk_tolerance: float,
        capital: float
    ):
        super().__init__(unique_id, model)
        self.personality = personality
        self.risk_tolerance = risk_tolerance
        self.capital = capital
        self.portfolio = {}
        self.message_inbox = deque(maxlen=100)
        self.performance_history = []
        self.current_regime = MarketRegime.SIDEWAYS
        self.learning_rate = 0.1
        self.memory = deque(maxlen=1000)

    @abstractmethod
    async def analyze_market(self, market_data: pd.DataFrame) -> TradingDecision:
        """Analyze market and make trading decision"""
        pass

    @abstractmethod
    def update_strategy(self, feedback: Dict[str, Any]):
        """Update strategy based on performance feedback"""
        pass

    def send_message(self, recipient_id: str, message_type: str, content: Dict):
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.unique_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )
        self.model.message_broker.route_message(message)

    def receive_message(self, message: AgentMessage):
        """Receive and process message"""
        self.message_inbox.append(message)
        self._process_message(message)

    def _process_message(self, message: AgentMessage):
        """Process received message"""
        if message.message_type == "market_signal":
            self.memory.append(message.content)
        elif message.message_type == "regime_change":
            self.current_regime = MarketRegime(message.content['regime'])

    def step(self):
        """Agent step in simulation"""
        # Process messages
        while self.message_inbox:
            message = self.message_inbox.popleft()
            self._process_message(message)


class MomentumTradingAgent(TradingAgentBase):
    """Agent specialized in momentum trading"""

    def __init__(self, unique_id: str, model: 'MultiAgentTradingModel'):
        super().__init__(
            unique_id=unique_id,
            model=model,
            personality=AgentPersonality.TREND_FOLLOWER,
            risk_tolerance=0.7,
            capital=100000
        )
        self.momentum_window = 20
        self.strength_threshold = 0.6

    async def analyze_market(self, market_data: pd.DataFrame) -> TradingDecision:
        """Momentum-based market analysis"""
        # Calculate momentum indicators
        returns = market_data['close'].pct_change(self.momentum_window)
        momentum_score = returns.iloc[-1]

        # RSI calculation
        rsi = self._calculate_rsi(market_data['close'])

        # Volume analysis
        volume_trend = market_data['volume'].rolling(10).mean().iloc[-1] / \
                      market_data['volume'].rolling(50).mean().iloc[-1]

        # Make decision
        if momentum_score > 0.02 and rsi < 70 and volume_trend > 1.2:
            action = "buy"
            confidence = min(momentum_score * 10, 0.9)
        elif momentum_score < -0.02 and rsi > 30:
            action = "sell"
            confidence = min(abs(momentum_score) * 10, 0.9)
        else:
            action = "hold"
            confidence = 0.3

        return TradingDecision(
            agent_id=self.unique_id,
            symbol=market_data.attrs.get('symbol', 'UNKNOWN'),
            action=action,
            size=self._calculate_position_size(confidence),
            confidence=confidence,
            reasoning=f"Momentum: {momentum_score:.3f}, RSI: {rsi:.1f}",
            risk_score=1 - confidence,
            expected_return=momentum_score * confidence,
            time_horizon="short",
            metadata={'momentum': momentum_score, 'rsi': rsi}
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on Kelly Criterion"""
        kelly_fraction = confidence - (1 - confidence) / 2  # Simplified Kelly
        position_size = min(kelly_fraction * self.capital * self.risk_tolerance, self.capital * 0.1)
        return max(position_size, 0)

    def update_strategy(self, feedback: Dict[str, Any]):
        """Adjust momentum parameters based on performance"""
        if feedback['success_rate'] < 0.4:
            self.momentum_window = min(self.momentum_window + 5, 50)
            self.strength_threshold = min(self.strength_threshold + 0.05, 0.8)
        elif feedback['success_rate'] > 0.6:
            self.momentum_window = max(self.momentum_window - 2, 10)


class ValueInvestingAgent(TradingAgentBase):
    """Agent focused on value investing principles"""

    def __init__(self, unique_id: str, model: 'MultiAgentTradingModel'):
        super().__init__(
            unique_id=unique_id,
            model=model,
            personality=AgentPersonality.VALUE_INVESTOR,
            risk_tolerance=0.4,
            capital=200000
        )
        self.valuation_metrics = ['pe_ratio', 'pb_ratio', 'debt_equity']

    async def analyze_market(self, market_data: pd.DataFrame) -> TradingDecision:
        """Value-based analysis"""
        # Calculate value metrics
        current_price = market_data['close'].iloc[-1]
        sma_200 = market_data['close'].rolling(200).mean().iloc[-1]
        price_to_sma = current_price / sma_200

        # Simulate fundamental analysis
        intrinsic_value = self._calculate_intrinsic_value(market_data)
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value

        if margin_of_safety > 0.2:
            action = "buy"
            confidence = min(margin_of_safety * 2, 0.9)
        elif margin_of_safety < -0.2:
            action = "sell"
            confidence = min(abs(margin_of_safety) * 2, 0.9)
        else:
            action = "hold"
            confidence = 0.4

        return TradingDecision(
            agent_id=self.unique_id,
            symbol=market_data.attrs.get('symbol', 'UNKNOWN'),
            action=action,
            size=self._calculate_position_size(confidence, margin_of_safety),
            confidence=confidence,
            reasoning=f"Margin of safety: {margin_of_safety:.2%}",
            risk_score=max(0, 1 - margin_of_safety),
            expected_return=margin_of_safety,
            time_horizon="long",
            metadata={'intrinsic_value': intrinsic_value, 'margin_of_safety': margin_of_safety}
        )

    def _calculate_intrinsic_value(self, market_data: pd.DataFrame) -> float:
        """Simplified intrinsic value calculation"""
        # This is a simplified model - in reality would use DCF, earnings, etc.
        avg_price = market_data['close'].mean()
        volatility = market_data['close'].std()
        return avg_price * (1 + np.random.normal(0, 0.1))

    def _calculate_position_size(self, confidence: float, margin_of_safety: float) -> float:
        """Conservative position sizing for value investing"""
        base_size = self.capital * 0.05  # Max 5% per position
        return base_size * confidence * (1 + margin_of_safety)

    def update_strategy(self, feedback: Dict[str, Any]):
        """Adjust value thresholds"""
        if feedback.get('avg_return', 0) < 0:
            self.risk_tolerance *= 0.95


class ArbitrageAgent(TradingAgentBase):
    """Agent specialized in arbitrage opportunities"""

    def __init__(self, unique_id: str, model: 'MultiAgentTradingModel'):
        super().__init__(
            unique_id=unique_id,
            model=model,
            personality=AgentPersonality.SCALPER,
            risk_tolerance=0.3,
            capital=150000
        )
        self.min_spread = 0.001
        self.execution_speed = 0.1

    async def analyze_market(self, market_data: pd.DataFrame) -> TradingDecision:
        """Identify arbitrage opportunities"""
        # Detect price inefficiencies
        bid_ask_spread = self._calculate_spread(market_data)
        price_deviation = self._detect_deviation(market_data)

        if bid_ask_spread > self.min_spread:
            action = "buy"  # Simplified - would actually be buy/sell pair
            confidence = min(bid_ask_spread * 100, 0.8)
        else:
            action = "hold"
            confidence = 0.2

        return TradingDecision(
            agent_id=self.unique_id,
            symbol=market_data.attrs.get('symbol', 'UNKNOWN'),
            action=action,
            size=self.capital * 0.2,  # Use larger size for arbitrage
            confidence=confidence,
            reasoning=f"Spread: {bid_ask_spread:.4f}",
            risk_score=0.1,  # Low risk for arbitrage
            expected_return=bid_ask_spread,
            time_horizon="immediate",
            metadata={'spread': bid_ask_spread}
        )

    def _calculate_spread(self, market_data: pd.DataFrame) -> float:
        """Calculate bid-ask spread"""
        # Simplified calculation
        high_low_spread = (market_data['high'].iloc[-1] - market_data['low'].iloc[-1]) / market_data['close'].iloc[-1]
        return high_low_spread

    def _detect_deviation(self, market_data: pd.DataFrame) -> float:
        """Detect price deviation from fair value"""
        return np.random.random() * 0.01

    def update_strategy(self, feedback: Dict[str, Any]):
        """Adjust arbitrage parameters"""
        if feedback.get('execution_success', 0) < 0.8:
            self.min_spread *= 1.1


@ray.remote
class SentimentAnalysisAgent(TradingAgentBase):
    """Ray actor for distributed sentiment analysis"""

    def __init__(self):
        self.sentiment_cache = {}

    async def analyze_sentiment(self, symbol: str, news_data: List[str]) -> Dict:
        """Analyze sentiment from news data"""
        # Simplified sentiment analysis
        sentiment_scores = []
        for news in news_data:
            # In production, would use actual NLP model
            score = np.random.random() * 2 - 1  # -1 to 1
            sentiment_scores.append(score)

        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0

        return {
            'symbol': symbol,
            'sentiment': avg_sentiment,
            'confidence': 1 - np.std(sentiment_scores) if sentiment_scores else 0,
            'n_articles': len(news_data)
        }


class MessageBroker:
    """Central message broker for agent communication"""

    def __init__(self):
        self.agents = {}
        self.message_log = deque(maxlen=10000)
        self.topic_subscriptions = {}

    def register_agent(self, agent: TradingAgentBase):
        """Register an agent with the broker"""
        self.agents[agent.unique_id] = agent

    def route_message(self, message: AgentMessage):
        """Route message to recipient"""
        self.message_log.append(message)

        if message.recipient_id == "broadcast":
            # Broadcast to all agents
            for agent in self.agents.values():
                agent.receive_message(message)
        elif message.recipient_id in self.agents:
            self.agents[message.recipient_id].receive_message(message)

    def publish_topic(self, topic: str, content: Dict):
        """Publish to a topic"""
        if topic in self.topic_subscriptions:
            for agent_id in self.topic_subscriptions[topic]:
                message = AgentMessage(
                    sender_id="broker",
                    recipient_id=agent_id,
                    message_type=topic,
                    content=content,
                    timestamp=datetime.now()
                )
                self.route_message(message)


class MultiAgentTradingModel(Model):
    """
    Main multi-agent trading system model
    Coordinates multiple specialized trading agents
    """

    def __init__(
        self,
        n_momentum_agents: int = 3,
        n_value_agents: int = 2,
        n_arbitrage_agents: int = 2
    ):
        self.num_agents = n_momentum_agents + n_value_agents + n_arbitrage_agents
        self.schedule = RandomActivation(self)
        self.message_broker = MessageBroker()
        self.current_step = 0

        # Initialize Redis for inter-agent communication
        self.redis_client = redis.Redis(host='localhost', port=6379)

        # Create communication network
        self.comm_network = nx.Graph()

        # Initialize different types of agents
        agent_id = 0

        # Momentum agents
        for i in range(n_momentum_agents):
            agent = MomentumTradingAgent(f"momentum_{agent_id}", self)
            self.schedule.add(agent)
            self.message_broker.register_agent(agent)
            self.comm_network.add_node(agent.unique_id, agent_type='momentum')
            agent_id += 1

        # Value agents
        for i in range(n_value_agents):
            agent = ValueInvestingAgent(f"value_{agent_id}", self)
            self.schedule.add(agent)
            self.message_broker.register_agent(agent)
            self.comm_network.add_node(agent.unique_id, agent_type='value')
            agent_id += 1

        # Arbitrage agents
        for i in range(n_arbitrage_agents):
            agent = ArbitrageAgent(f"arbitrage_{agent_id}", self)
            self.schedule.add(agent)
            self.message_broker.register_agent(agent)
            self.comm_network.add_node(agent.unique_id, agent_type='arbitrage')
            agent_id += 1

        # Create communication edges (agents can communicate)
        self._create_communication_network()

        # Initialize Ray sentiment analysis agents
        self.sentiment_agents = [
            SentimentAnalysisAgent.remote() for _ in range(3)
        ]

        # Data collector for monitoring
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Capital": self._compute_total_capital,
                "Active_Positions": self._count_active_positions,
                "Message_Volume": lambda m: len(m.message_broker.message_log)
            },
            agent_reporters={
                "Capital": "capital",
                "Portfolio_Size": lambda a: len(a.portfolio)
            }
        )

        logger.info(f"Initialized MultiAgentTradingModel with {self.num_agents} agents")

    def _create_communication_network(self):
        """Create communication network between agents"""
        # Connect agents of same type
        agents_by_type = {}
        for node, data in self.comm_network.nodes(data=True):
            agent_type = data['agent_type']
            if agent_type not in agents_by_type:
                agents_by_type[agent_type] = []
            agents_by_type[agent_type].append(node)

        # Create edges within same type
        for agent_type, agents in agents_by_type.items():
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    self.comm_network.add_edge(agents[i], agents[j])

        # Create some cross-type connections
        if 'momentum' in agents_by_type and 'value' in agents_by_type:
            for m_agent in agents_by_type['momentum'][:1]:
                for v_agent in agents_by_type['value'][:1]:
                    self.comm_network.add_edge(m_agent, v_agent)

    async def execute_trading_round(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> List[TradingDecision]:
        """Execute one trading round with all agents"""
        all_decisions = []

        # Detect market regime
        regime = self._detect_market_regime(market_data)
        self.message_broker.publish_topic(
            "regime_change",
            {'regime': regime.value, 'timestamp': datetime.now().isoformat()}
        )

        # Get sentiment analysis in parallel
        sentiment_tasks = []
        for symbol, data in market_data.items():
            # Distribute sentiment analysis across Ray agents
            agent_idx = hash(symbol) % len(self.sentiment_agents)
            task = self.sentiment_agents[agent_idx].analyze_sentiment.remote(
                symbol, []  # Would pass actual news data
            )
            sentiment_tasks.append(task)

        sentiment_results = await asyncio.gather(*[
            asyncio.wrap_future(asyncio.get_event_loop().run_in_executor(None, ray.get, task))
            for task in sentiment_tasks
        ])

        # Broadcast sentiment to all agents
        for sentiment in sentiment_results:
            self.message_broker.publish_topic("sentiment_update", sentiment)

        # Each agent analyzes market and makes decisions
        agent_tasks = []
        for agent in self.schedule.agents:
            for symbol, data in market_data.items():
                data.attrs['symbol'] = symbol
                task = agent.analyze_market(data)
                agent_tasks.append(task)

        # Execute all agent analyses in parallel
        decisions = await asyncio.gather(*agent_tasks)
        all_decisions.extend(decisions)

        # Agents communicate decisions
        for decision in decisions:
            self.message_broker.publish_topic(
                "trading_signal",
                decision.__dict__
            )

        # Consensus building
        consensus = self._build_consensus(all_decisions)

        return consensus

    def _detect_market_regime(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> MarketRegime:
        """Detect current market regime"""
        # Simplified regime detection
        volatilities = []
        trends = []

        for symbol, data in market_data.items():
            volatilities.append(data['close'].pct_change().std())
            trends.append(data['close'].iloc[-1] / data['close'].iloc[0] - 1)

        avg_volatility = np.mean(volatilities)
        avg_trend = np.mean(trends)

        if avg_volatility > 0.03:
            if avg_trend < -0.1:
                return MarketRegime.CRASH
            return MarketRegime.VOLATILE
        elif avg_trend > 0.05:
            return MarketRegime.BULL
        elif avg_trend < -0.05:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _build_consensus(
        self,
        decisions: List[TradingDecision]
    ) -> List[TradingDecision]:
        """Build consensus from multiple agent decisions"""
        # Group decisions by symbol
        symbol_decisions = {}
        for decision in decisions:
            if decision.symbol not in symbol_decisions:
                symbol_decisions[decision.symbol] = []
            symbol_decisions[decision.symbol].append(decision)

        consensus_decisions = []
        for symbol, symbol_decs in symbol_decisions.items():
            # Weight decisions by confidence and agent type
            buy_score = sum(d.confidence for d in symbol_decs if d.action == 'buy')
            sell_score = sum(d.confidence for d in symbol_decs if d.action == 'sell')
            hold_score = sum(d.confidence for d in symbol_decs if d.action == 'hold')

            total_score = buy_score + sell_score + hold_score

            if total_score > 0:
                if buy_score > sell_score and buy_score > hold_score:
                    consensus_action = 'buy'
                    consensus_confidence = buy_score / total_score
                elif sell_score > buy_score and sell_score > hold_score:
                    consensus_action = 'sell'
                    consensus_confidence = sell_score / total_score
                else:
                    consensus_action = 'hold'
                    consensus_confidence = hold_score / total_score

                consensus_decisions.append(TradingDecision(
                    agent_id='consensus',
                    symbol=symbol,
                    action=consensus_action,
                    size=np.mean([d.size for d in symbol_decs]),
                    confidence=consensus_confidence,
                    reasoning=f"Consensus from {len(symbol_decs)} agents",
                    risk_score=np.mean([d.risk_score for d in symbol_decs]),
                    expected_return=np.mean([d.expected_return for d in symbol_decs]),
                    time_horizon='mixed',
                    metadata={'n_agents': len(symbol_decs)}
                ))

        return consensus_decisions

    def step(self):
        """Advance model by one step"""
        self.current_step += 1
        self.schedule.step()
        self.datacollector.collect(self)

    def _compute_total_capital(self) -> float:
        """Compute total capital across all agents"""
        return sum(agent.capital for agent in self.schedule.agents)

    def _count_active_positions(self) -> int:
        """Count total active positions"""
        return sum(len(agent.portfolio) for agent in self.schedule.agents)


# Example usage
async def run_multi_agent_system():
    """Run the multi-agent trading system"""
    # Initialize model
    model = MultiAgentTradingModel(
        n_momentum_agents=3,
        n_value_agents=2,
        n_arbitrage_agents=2
    )

    # Simulate market data
    market_data = {
        'AAPL': pd.DataFrame({
            'close': np.random.random(100) * 100 + 100,
            'high': np.random.random(100) * 100 + 105,
            'low': np.random.random(100) * 100 + 95,
            'volume': np.random.random(100) * 1000000
        }),
        'GOOGL': pd.DataFrame({
            'close': np.random.random(100) * 100 + 150,
            'high': np.random.random(100) * 100 + 155,
            'low': np.random.random(100) * 100 + 145,
            'volume': np.random.random(100) * 1000000
        })
    }

    # Execute trading round
    decisions = await model.execute_trading_round(market_data)

    print("Trading Decisions:")
    for decision in decisions:
        print(f"{decision.symbol}: {decision.action} "
              f"(confidence: {decision.confidence:.2f}, "
              f"size: ${decision.size:.2f})")

    # Run simulation steps
    for i in range(10):
        model.step()

    # Get results
    df = model.datacollector.get_model_vars_dataframe()
    print(f"\nSimulation Results:")
    print(f"Final Total Capital: ${df['Total_Capital'].iloc[-1]:,.2f}")
    print(f"Active Positions: {df['Active_Positions'].iloc[-1]}")
    print(f"Message Volume: {df['Message_Volume'].iloc[-1]}")


if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Run the system
    asyncio.run(run_multi_agent_system())

    # Shutdown Ray
    ray.shutdown()