"""
Reinforcement Learning Trading Agents with Multi-Agent Processing
=================================================================
Advanced RL agents for adaptive trading strategies using DQN, PPO, and A3C
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import ray
import asyncio
from collections import deque, namedtuple
import gym
from gym import spaces
import random
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Experience replay buffer
Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class RLAction:
    """RL agent action"""
    symbol: str
    action_type: str  # buy, sell, hold
    position_size: float
    confidence: float
    expected_reward: float
    risk_estimate: float
    timestamp: datetime


@dataclass
class RLPerformance:
    """RL agent performance metrics"""
    total_reward: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    trades_executed: int
    avg_holding_period: float
    timestamp: datetime


class TradingEnvironment(gym.Env):
    """Trading environment for RL agents"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        max_position: float = 0.2
    ):
        super().__init__()

        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        # State: [prices, indicators, position, cash]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )

        # Action: [hold, buy, sell] × [small, medium, large]
        self.action_space = spaces.Discrete(9)

        self.reset()

    def reset(self):
        """Reset environment"""
        self.current_step = 50  # Start after having history
        self.cash = self.initial_capital
        self.position = 0
        self.portfolio_value = self.initial_capital
        self.trades = []

        return self._get_state()

    def step(self, action):
        """Execute action and return next state"""
        # Decode action
        action_type, position_size = self._decode_action(action)

        # Get current price
        current_price = self.data['close'].iloc[self.current_step]

        # Execute trade
        reward = self._execute_trade(action_type, position_size, current_price)

        # Move to next step
        self.current_step += 1

        # Check if done
        done = self.current_step >= len(self.data) - 1

        # Get next state
        next_state = self._get_state()

        # Calculate portfolio value
        self.portfolio_value = self.cash + self.position * current_price

        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash
        }

        return next_state, reward, done, info

    def _get_state(self):
        """Get current state representation"""
        # Price features
        window = 20
        prices = self.data['close'].iloc[self.current_step-window:self.current_step].values
        returns = np.diff(prices) / prices[:-1]

        # Technical indicators
        sma = prices.mean()
        std = prices.std()
        rsi = self._calculate_rsi(prices)

        # Normalize prices
        norm_prices = (prices - sma) / std

        # Position and cash info
        position_ratio = self.position * prices[-1] / self.portfolio_value if self.portfolio_value > 0 else 0
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 1

        # Combine features
        state = np.concatenate([
            norm_prices[-10:],  # Recent prices
            returns[-10:],      # Recent returns
            [rsi, position_ratio, cash_ratio, std/sma]  # Additional features
        ])

        # Pad if necessary
        if len(state) < 50:
            state = np.pad(state, (0, 50 - len(state)), 'constant')

        return state.astype(np.float32)

    def _decode_action(self, action):
        """Decode discrete action to trade parameters"""
        action_types = ['hold', 'buy', 'sell']
        position_sizes = [0.05, 0.1, 0.2]  # Small, medium, large

        action_type = action_types[action // 3]
        position_size = position_sizes[action % 3]

        return action_type, position_size

    def _execute_trade(self, action_type, position_size, current_price):
        """Execute trade and calculate reward"""
        prev_value = self.portfolio_value
        trade_amount = self.portfolio_value * position_size

        if action_type == 'buy' and self.cash > trade_amount:
            # Buy
            shares = (trade_amount * (1 - self.transaction_cost)) / current_price
            self.position += shares
            self.cash -= trade_amount
            self.trades.append(('buy', shares, current_price))

        elif action_type == 'sell' and self.position > 0:
            # Sell
            shares_to_sell = min(self.position, trade_amount / current_price)
            self.cash += shares_to_sell * current_price * (1 - self.transaction_cost)
            self.position -= shares_to_sell
            self.trades.append(('sell', shares_to_sell, current_price))

        # Calculate new portfolio value
        new_value = self.cash + self.position * current_price

        # Reward is the return
        reward = (new_value - prev_value) / prev_value

        # Add risk-adjusted component
        if self.position > 0:
            position_ratio = self.position * current_price / new_value
            risk_penalty = -0.001 * (position_ratio - 0.5) ** 2 if position_ratio > 0.5 else 0
            reward += risk_penalty

        return reward

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period:
            return 0.5

        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        return rsi / 100  # Normalize to [0, 1]


class DQNNetwork(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)

        # Dueling DQN architecture
        self.value_stream = nn.Linear(hidden_dim // 2, 1)
        self.advantage_stream = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class PPONetwork(nn.Module):
    """Proximal Policy Optimization network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        policy = self.policy(shared)
        value = self.value(shared)
        return policy, value


class A3CNetwork(nn.Module):
    """Asynchronous Advantage Actor-Critic network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic (value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))

        # LSTM for temporal dependencies
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1) if x.shape[1] == 1 else x[:, -1, :]

        x = F.relu(self.fc2(x))

        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)

        return policy, value, hidden


@ray.remote
class DQNAgent:
    """Deep Q-Network agent"""

    def __init__(
        self,
        agent_id: str,
        state_dim: int = 50,
        action_dim: int = 9,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = 32

        # Update target network
        self.update_target_network()

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append(Experience(state, action, reward, next_state, done))

    async def train(self, batch_size: Optional[int] = None):
        """Train DQN"""
        if len(self.memory) < self.batch_size:
            return 0

        batch_size = batch_size or self.batch_size
        batch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Double DQN
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


@ray.remote
class PPOAgent:
    """Proximal Policy Optimization agent"""

    def __init__(
        self,
        agent_id: str,
        state_dim: int = 50,
        action_dim: int = 9,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param

        # Network
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state):
        """Select action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, value = self.network(state_tensor)

            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    async def train(self, epochs: int = 4):
        """Train PPO"""
        if len(self.states) < 32:
            return 0

        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        log_probs = torch.FloatTensor(self.log_probs)
        dones = torch.FloatTensor(self.dones)

        # Calculate advantages using GAE
        advantages = self._calculate_gae(rewards, values, dones)

        # Calculate returns
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_loss = 0
        for _ in range(epochs):
            # Get current policy and value
            policy, value = self.network(states)

            dist = Categorical(policy)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate ratio
            ratio = torch.exp(new_log_probs - log_probs)

            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(value.squeeze(), returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        return total_loss / epochs

    def _calculate_gae(self, rewards, values, dones):
        """Calculate Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages)


@ray.remote
class A3CAgent:
    """Asynchronous Advantage Actor-Critic agent"""

    def __init__(
        self,
        agent_id: str,
        state_dim: int = 50,
        action_dim: int = 9,
        learning_rate: float = 0.001,
        gamma: float = 0.99
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Network
        self.network = A3CNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Hidden state for LSTM
        self.hidden = None

    def select_action(self, state):
        """Select action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, value, self.hidden = self.network(state_tensor, self.hidden)

            dist = Categorical(policy)
            action = dist.sample()

        return action.item(), value.item()

    async def train_on_trajectory(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        dones: List[bool]
    ):
        """Train on a trajectory of experiences"""
        if len(states) == 0:
            return 0

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)

        # Forward pass
        policies, values, _ = self.network(states_tensor)

        # Calculate returns
        returns = self._calculate_returns(rewards, dones)
        returns_tensor = torch.FloatTensor(returns)

        # Calculate advantages
        advantages = returns_tensor - values.squeeze()

        # Policy loss
        dist = Categorical(policies)
        log_probs = dist.log_prob(actions_tensor)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns_tensor)

        # Entropy for exploration
        entropy = dist.entropy().mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset hidden state after training
        self.hidden = None

        return loss.item()

    def _calculate_returns(self, rewards, dones):
        """Calculate discounted returns"""
        returns = []
        R = 0

        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        return returns


class RLTradingOrchestrator:
    """Orchestrate multiple RL agents with different strategies"""

    def __init__(self, n_agents: int = 3):
        ray.init(ignore_reinit_error=True)

        # Initialize different types of agents
        self.dqn_agents = [
            DQNAgent.remote(f"dqn_{i}") for i in range(n_agents)
        ]
        self.ppo_agents = [
            PPOAgent.remote(f"ppo_{i}") for i in range(n_agents)
        ]
        self.a3c_agents = [
            A3CAgent.remote(f"a3c_{i}") for i in range(n_agents)
        ]

        # Performance tracking
        self.agent_performances = {}
        self.ensemble_actions = deque(maxlen=1000)

    async def train_agents(
        self,
        market_data: Dict[str, pd.DataFrame],
        episodes: int = 100
    ) -> Dict[str, Any]:
        """Train all RL agents in parallel"""

        training_tasks = []
        training_results = {}

        # Train each agent type on different data
        symbols = list(market_data.keys())

        # DQN agents
        for i, agent in enumerate(self.dqn_agents):
            symbol = symbols[i % len(symbols)]
            env = TradingEnvironment(market_data[symbol])

            task = self._train_dqn_agent(agent, env, episodes)
            training_tasks.append(('dqn', symbol, task))

        # PPO agents
        for i, agent in enumerate(self.ppo_agents):
            symbol = symbols[(i + 1) % len(symbols)]
            env = TradingEnvironment(market_data[symbol])

            task = self._train_ppo_agent(agent, env, episodes)
            training_tasks.append(('ppo', symbol, task))

        # A3C agents
        for i, agent in enumerate(self.a3c_agents):
            symbol = symbols[(i + 2) % len(symbols)]
            env = TradingEnvironment(market_data[symbol])

            task = self._train_a3c_agent(agent, env, episodes)
            training_tasks.append(('a3c', symbol, task))

        # Gather results
        for agent_type, symbol, task in training_tasks:
            result = await task
            key = f"{agent_type}_{symbol}"
            training_results[key] = result
            self.agent_performances[key] = result

        return training_results

    async def _train_dqn_agent(
        self,
        agent,
        env: TradingEnvironment,
        episodes: int
    ) -> Dict[str, Any]:
        """Train single DQN agent"""
        total_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action = ray.get(agent.select_action.remote(state, training=True))

                # Step environment
                next_state, reward, done, _ = env.step(action)

                # Remember experience
                ray.get(agent.remember.remote(state, action, reward, next_state, done))

                # Train
                if episode % 4 == 0:
                    await asyncio.wrap_future(
                        asyncio.get_event_loop().run_in_executor(None, ray.get, agent.train.remote())
                    )

                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

            # Update target network
            if episode % 10 == 0:
                ray.get(agent.update_target_network.remote())

        return {
            'avg_reward': np.mean(total_rewards),
            'final_reward': total_rewards[-1],
            'best_reward': max(total_rewards)
        }

    async def _train_ppo_agent(
        self,
        agent,
        env: TradingEnvironment,
        episodes: int
    ) -> Dict[str, Any]:
        """Train single PPO agent"""
        total_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action, log_prob, value = ray.get(agent.select_action.remote(state))

                # Step environment
                next_state, reward, done, _ = env.step(action)

                # Store transition
                ray.get(agent.store_transition.remote(
                    state, action, reward, value, log_prob, done
                ))

                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

            # Train after each episode
            if episode % 4 == 0:
                await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, agent.train.remote())
                )

        return {
            'avg_reward': np.mean(total_rewards),
            'final_reward': total_rewards[-1],
            'best_reward': max(total_rewards)
        }

    async def _train_a3c_agent(
        self,
        agent,
        env: TradingEnvironment,
        episodes: int
    ) -> Dict[str, Any]:
        """Train single A3C agent"""
        total_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            states, actions, rewards, dones = [], [], [], []

            while not done:
                # Select action
                action, _ = ray.get(agent.select_action.remote(state))

                # Step environment
                next_state, reward, done, _ = env.step(action)

                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

            # Train on trajectory
            await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(
                    None, ray.get,
                    agent.train_on_trajectory.remote(states, actions, rewards, dones)
                )
            )

        return {
            'avg_reward': np.mean(total_rewards),
            'final_reward': total_rewards[-1],
            'best_reward': max(total_rewards)
        }

    async def generate_ensemble_action(
        self,
        state: np.ndarray,
        symbol: str
    ) -> RLAction:
        """Generate action from ensemble of RL agents"""

        action_votes = []

        # Get actions from each agent type
        for agent in self.dqn_agents:
            action = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(
                    None, ray.get,
                    agent.select_action.remote(state, training=False)
                )
            )
            action_votes.append(action)

        for agent in self.ppo_agents:
            action, _, _ = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(
                    None, ray.get,
                    agent.select_action.remote(state)
                )
            )
            action_votes.append(action)

        for agent in self.a3c_agents:
            action, _ = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(
                    None, ray.get,
                    agent.select_action.remote(state)
                )
            )
            action_votes.append(action)

        # Majority voting
        from collections import Counter
        action_counts = Counter(action_votes)
        ensemble_action = action_counts.most_common(1)[0][0]

        # Decode action
        action_types = ['hold', 'buy', 'sell']
        position_sizes = [0.05, 0.1, 0.2]

        action_type = action_types[ensemble_action // 3]
        position_size = position_sizes[ensemble_action % 3]

        # Calculate confidence based on agreement
        confidence = action_counts[ensemble_action] / len(action_votes)

        return RLAction(
            symbol=symbol,
            action_type=action_type,
            position_size=position_size,
            confidence=confidence,
            expected_reward=0.01 * confidence,  # Simplified
            risk_estimate=1 - confidence,
            timestamp=datetime.now()
        )

    def get_agent_performances(self) -> Dict[str, RLPerformance]:
        """Get performance metrics for all agents"""
        performances = {}

        for agent_id, metrics in self.agent_performances.items():
            perf = RLPerformance(
                total_reward=metrics.get('avg_reward', 0),
                win_rate=0.6,  # Placeholder - would calculate from trades
                sharpe_ratio=metrics.get('avg_reward', 0) / 0.1,  # Simplified
                max_drawdown=0.1,  # Placeholder
                trades_executed=100,  # Placeholder
                avg_holding_period=24,  # hours
                timestamp=datetime.now()
            )
            performances[agent_id] = perf

        return performances

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of RL trading agents"""
    orchestrator = RLTradingOrchestrator(n_agents=2)

    # Generate sample market data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    market_data = {}

    for symbol in symbols:
        # Create synthetic price data
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1h')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        df = pd.DataFrame({
            'open': prices + np.random.randn(1000) * 0.2,
            'high': prices + np.abs(np.random.randn(1000) * 0.3),
            'low': prices - np.abs(np.random.randn(1000) * 0.3),
            'close': prices,
            'volume': np.random.gamma(2, 1000000, 1000)
        }, index=dates)

        market_data[symbol] = df

    # Train agents
    print("Training RL agents...")
    training_results = await orchestrator.train_agents(market_data, episodes=10)

    print("\nTraining Results:")
    for agent_id, metrics in training_results.items():
        print(f"  {agent_id}:")
        print(f"    Avg Reward: {metrics.get('avg_reward', 0):.4f}")
        print(f"    Best Reward: {metrics.get('best_reward', 0):.4f}")

    # Generate ensemble action
    print("\nGenerating ensemble actions...")
    test_state = np.random.randn(50).astype(np.float32)

    for symbol in symbols:
        action = await orchestrator.generate_ensemble_action(test_state, symbol)
        print(f"  {symbol}: {action.action_type} "
              f"(size: {action.position_size:.1%}, conf: {action.confidence:.1%})")

    # Get performances
    performances = orchestrator.get_agent_performances()
    print("\nAgent Performances:")
    for agent_id, perf in performances.items():
        print(f"  {agent_id}:")
        print(f"    Total Reward: {perf.total_reward:.4f}")
        print(f"    Sharpe Ratio: {perf.sharpe_ratio:.2f}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())