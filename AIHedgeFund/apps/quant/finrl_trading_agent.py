"""
FinRL Deep Reinforcement Learning Trading Agent
==============================================
State-of-the-art DRL algorithms for portfolio management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# FinRL imports
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# Stable Baselines3
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

logger = logging.getLogger(__name__)


@dataclass
class DRLConfig:
    """Configuration for DRL trading agent"""
    algorithm: str = "PPO"  # PPO, A2C, DDPG, TD3, SAC
    total_timesteps: int = 100000
    learning_rate: float = 0.0003
    batch_size: int = 64
    buffer_size: int = 100000
    learning_starts: int = 100
    tau: float = 0.005
    gamma: float = 0.99
    seed: int = 42
    net_arch: List[int] = None

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [400, 300]


class FinRLTradingAgent:
    """Advanced DRL Trading Agent using FinRL"""

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        config: Optional[DRLConfig] = None
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.config = config or DRLConfig()
        self.model = None
        self.env = None
        self.processed_data = None

    def prepare_data(self) -> pd.DataFrame:
        """Download and preprocess market data"""
        logger.info(f"Downloading data for {self.symbols}")

        # Download data
        downloader = YahooDownloader(
            start_date=self.start_date,
            end_date=self.end_date,
            ticker_list=self.symbols
        )

        raw_data = downloader.fetch_data()

        # Feature engineering
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False
        )

        self.processed_data = fe.preprocess_data(raw_data)
        return self.processed_data

    def create_trading_env(self, data: pd.DataFrame) -> StockTradingEnv:
        """Create trading environment"""

        stock_dimension = len(self.symbols)
        state_space = (
            1 + 2 * stock_dimension +
            len(INDICATORS) * stock_dimension
        )

        env_config = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": [0] * stock_dimension,
            "buy_cost_pct": [0.001] * stock_dimension,
            "sell_cost_pct": [0.001] * stock_dimension,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }

        self.env = StockTradingEnv(
            df=data,
            **env_config
        )

        return self.env

    def train(
        self,
        train_data: Optional[pd.DataFrame] = None,
        val_data: Optional[pd.DataFrame] = None
    ):
        """Train the DRL agent"""

        if train_data is None:
            train_data = self.processed_data

        # Create environment
        train_env = self.create_trading_env(train_data)
        train_env = DummyVecEnv([lambda: train_env])

        # Select algorithm
        algorithm_map = {
            "PPO": PPO,
            "A2C": A2C,
            "DDPG": DDPG,
            "TD3": TD3,
            "SAC": SAC
        }

        AlgorithmClass = algorithm_map[self.config.algorithm]

        # Initialize model
        model_kwargs = {
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "seed": self.config.seed,
            "verbose": 1
        }

        if self.config.algorithm in ["PPO", "A2C"]:
            model_kwargs["n_steps"] = 2048
            model_kwargs["ent_coef"] = 0.01
        else:
            model_kwargs["buffer_size"] = self.config.buffer_size
            model_kwargs["learning_starts"] = self.config.learning_starts
            model_kwargs["tau"] = self.config.tau

        self.model = AlgorithmClass(
            "MlpPolicy",
            train_env,
            policy_kwargs={"net_arch": self.config.net_arch},
            **model_kwargs
        )

        # Setup callbacks
        callbacks = []

        if val_data is not None:
            val_env = self.create_trading_env(val_data)
            val_env = DummyVecEnv([lambda: val_env])

            eval_callback = EvalCallback(
                val_env,
                best_model_save_path="./models/",
                log_path="./logs/",
                eval_freq=10000,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./models/checkpoints/",
            name_prefix=f"{self.config.algorithm}_model"
        )
        callbacks.append(checkpoint_callback)

        # Train model
        logger.info(f"Training {self.config.algorithm} for {self.config.total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks
        )

        return self.model

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""

        test_env = self.create_trading_env(test_data)

        obs = test_env.reset()
        actions = []
        rewards = []

        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)

            actions.append(action)
            rewards.append(reward)

            if done:
                break

        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': test_data['date'].unique()[:len(actions)],
            'actions': actions,
            'rewards': rewards,
            'cumulative_return': np.cumprod(1 + np.array(rewards)) - 1
        })

        return results

    def backtest(self, test_data: pd.DataFrame) -> Dict:
        """Run backtest and calculate metrics"""

        results = self.predict(test_data)

        # Calculate metrics
        total_return = results['cumulative_return'].iloc[-1]
        sharpe_ratio = np.sqrt(252) * results['rewards'].mean() / results['rewards'].std()
        max_drawdown = (results['cumulative_return'].cummax() - results['cumulative_return']).max()

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (results['rewards'] > 0).mean(),
            'avg_reward': results['rewards'].mean(),
            'algorithm': self.config.algorithm
        }

        return metrics

    def ensemble_predict(
        self,
        test_data: pd.DataFrame,
        models: List[str] = ["PPO", "A2C", "SAC"]
    ) -> pd.DataFrame:
        """Ensemble prediction using multiple algorithms"""

        ensemble_actions = []

        for algo in models:
            self.config.algorithm = algo
            self.train()
            results = self.predict(test_data)
            ensemble_actions.append(results['actions'].values)

        # Weighted average of actions
        weights = [0.4, 0.3, 0.3]  # PPO, A2C, SAC weights
        final_actions = np.average(ensemble_actions, axis=0, weights=weights)

        return pd.DataFrame({
            'timestamp': test_data['date'].unique()[:len(final_actions)],
            'ensemble_actions': final_actions.tolist()
        })


class MultiAgentTradingSystem:
    """Multiple DRL agents for different market conditions"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.agents = {}

    def create_specialized_agents(self):
        """Create agents for different market regimes"""

        # Bull market agent (aggressive)
        bull_config = DRLConfig(
            algorithm="PPO",
            learning_rate=0.0005,
            gamma=0.95
        )
        self.agents['bull'] = FinRLTradingAgent(
            self.symbols,
            "2020-01-01",
            "2021-12-31",
            bull_config
        )

        # Bear market agent (conservative)
        bear_config = DRLConfig(
            algorithm="SAC",
            learning_rate=0.0001,
            gamma=0.99
        )
        self.agents['bear'] = FinRLTradingAgent(
            self.symbols,
            "2020-01-01",
            "2021-12-31",
            bear_config
        )

        # Sideways market agent (neutral)
        sideways_config = DRLConfig(
            algorithm="A2C",
            learning_rate=0.0003,
            gamma=0.97
        )
        self.agents['sideways'] = FinRLTradingAgent(
            self.symbols,
            "2020-01-01",
            "2021-12-31",
            sideways_config
        )

    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""

        # Simple regime detection based on moving averages
        prices = data.groupby('date')['close'].mean()
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()

        recent_ma20 = ma_20.iloc[-1]
        recent_ma50 = ma_50.iloc[-1]
        recent_price = prices.iloc[-1]

        if recent_price > recent_ma20 > recent_ma50:
            return 'bull'
        elif recent_price < recent_ma20 < recent_ma50:
            return 'bear'
        else:
            return 'sideways'

    def adaptive_trading(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adaptive trading based on market regime"""

        regime = self.detect_market_regime(data)
        logger.info(f"Detected market regime: {regime}")

        # Select appropriate agent
        agent = self.agents[regime]

        # Generate predictions
        predictions = agent.predict(data)
        predictions['market_regime'] = regime

        return predictions