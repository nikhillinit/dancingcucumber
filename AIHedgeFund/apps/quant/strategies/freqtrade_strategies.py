"""
Freqtrade-inspired strategies with multiple technical indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import pandas_ta as ta
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    """Configuration for trading strategies"""
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_ma: int = 20
    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 25


class FreqtradeStrategy:
    """Base class for Freqtrade-style strategies"""

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        # Momentum
        df['rsi'] = ta.rsi(df['close'], length=self.config.rsi_period)

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=self.config.bb_period, std=self.config.bb_std)
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # MACD
        macd = ta.macd(df['close'],
                      fast=self.config.macd_fast,
                      slow=self.config.macd_slow,
                      signal=self.config.macd_signal)
        df['macd'] = macd[f'MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']
        df['macd_signal'] = macd[f'MACDs_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']
        df['macd_hist'] = macd[f'MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']

        # Volume
        df['volume_ma'] = ta.sma(df['volume'], length=self.config.volume_ma)
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.config.atr_period)
        df['atr_ratio'] = df['atr'] / df['close']

        # Trend
        adx = ta.adx(df['high'], df['low'], df['close'], length=self.config.adx_period)
        df['adx'] = adx[f'ADX_{self.config.adx_period}']
        df['plus_di'] = adx[f'DMP_{self.config.adx_period}']
        df['minus_di'] = adx[f'DMN_{self.config.adx_period}']

        # EMA
        df['ema_fast'] = ta.ema(df['close'], length=12)
        df['ema_slow'] = ta.ema(df['close'], length=26)
        df['ema_200'] = ta.ema(df['close'], length=200)

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

        # Williams %R
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])

        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])

        # Ichimoku
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])[0]
        df['ichimoku_base'] = ichimoku['ISB_9']
        df['ichimoku_conv'] = ichimoku['ISA_9']

        return df

    def populate_buy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy signals - override in subclasses"""
        raise NotImplementedError

    def populate_sell_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sell signals - override in subclasses"""
        raise NotImplementedError


class BBRSIStrategy(FreqtradeStrategy):
    """Bollinger Bands + RSI Strategy"""

    def populate_buy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['buy_signal'] = (
            (df['close'] < df['bb_lower']) &
            (df['rsi'] < self.config.rsi_oversold) &
            (df['volume'] > df['volume_ma'])
        ).astype(int)
        return df

    def populate_sell_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sell_signal'] = (
            (df['close'] > df['bb_upper']) &
            (df['rsi'] > self.config.rsi_overbought)
        ).astype(int)
        return df


class MACDStrategy(FreqtradeStrategy):
    """MACD Crossover with ADX filter"""

    def populate_buy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['buy_signal'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1)) &
            (df['adx'] > self.config.adx_threshold) &
            (df['plus_di'] > df['minus_di'])
        ).astype(int)
        return df

    def populate_sell_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sell_signal'] = (
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)
        return df


class EMAScalpingStrategy(FreqtradeStrategy):
    """Fast EMA scalping for high-frequency trading"""

    def populate_buy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['buy_signal'] = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &
            (df['rsi'] > 30) &
            (df['rsi'] < 70) &
            (df['volume'] > df['volume_ma'] * 1.5)
        ).astype(int)
        return df

    def populate_sell_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sell_signal'] = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        ).astype(int)
        return df


class MultiIndicatorStrategy(FreqtradeStrategy):
    """Complex strategy using multiple confirmations"""

    def populate_buy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['buy_signal'] = (
            # Trend confirmation
            (df['close'] > df['ema_200']) &
            (df['adx'] > 20) &

            # Momentum
            ((df['rsi'] > 30) & (df['rsi'] < 50)) &
            (df['macd'] > df['macd_signal']) &

            # Volume confirmation
            (df['volume'] > df['volume_ma'] * 1.2) &

            # Price action
            (df['close'] > df['open']) &

            # Volatility filter
            (df['atr_ratio'] > 0.01) & (df['atr_ratio'] < 0.05)
        ).astype(int)
        return df

    def populate_sell_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sell_signal'] = (
            (
                # Take profit
                (df['rsi'] > 70) |

                # Stop loss
                ((df['close'] < df['bb_lower']) & (df['macd'] < df['macd_signal'])) |

                # Trend reversal
                ((df['ema_fast'] < df['ema_slow']) & (df['adx'] > 25))
            )
        ).astype(int)
        return df


class IchimokuCloudStrategy(FreqtradeStrategy):
    """Ichimoku Cloud based strategy"""

    def populate_buy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['buy_signal'] = (
            # Price above cloud
            (df['close'] > df['ichimoku_base']) &
            (df['close'] > df['ichimoku_conv']) &

            # Conversion line crosses above base line
            (df['ichimoku_conv'] > df['ichimoku_base']) &
            (df['ichimoku_conv'].shift(1) <= df['ichimoku_base'].shift(1)) &

            # Volume confirmation
            (df['volume'] > df['volume_ma'])
        ).astype(int)
        return df

    def populate_sell_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sell_signal'] = (
            # Price below cloud
            (df['close'] < df['ichimoku_base']) |
            (df['close'] < df['ichimoku_conv'])
        ).astype(int)
        return df


def run_strategy_backtest(
    df: pd.DataFrame,
    strategy: FreqtradeStrategy,
    initial_cash: float = 100000,
    position_size: float = 0.95,
    commission: float = 0.001
) -> Dict:
    """Run backtest for a given strategy"""

    # Populate indicators
    df = strategy.populate_indicators(df)
    df = strategy.populate_buy_trend(df)
    df = strategy.populate_sell_trend(df)

    # Simple backtest logic
    cash = initial_cash
    position = 0
    trades = []
    equity = []

    for i in range(len(df)):
        equity.append(cash + position * df['close'].iloc[i])

        if df['buy_signal'].iloc[i] and position == 0:
            # Buy
            size = (cash * position_size) / df['close'].iloc[i]
            cost = size * df['close'].iloc[i] * (1 + commission)
            if cost <= cash:
                position = size
                cash -= cost
                trades.append({
                    'type': 'buy',
                    'time': df.index[i],
                    'price': df['close'].iloc[i],
                    'size': size
                })

        elif df['sell_signal'].iloc[i] and position > 0:
            # Sell
            proceeds = position * df['close'].iloc[i] * (1 - commission)
            cash += proceeds
            trades.append({
                'type': 'sell',
                'time': df.index[i],
                'price': df['close'].iloc[i],
                'size': position
            })
            position = 0

    # Final equity
    final_equity = cash + position * df['close'].iloc[-1]

    # Calculate metrics
    equity_series = pd.Series(equity, index=df.index)
    returns = equity_series.pct_change().dropna()

    return {
        'total_return': (final_equity - initial_cash) / initial_cash,
        'final_equity': final_equity,
        'num_trades': len(trades),
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
        'max_drawdown': (equity_series / equity_series.cummax() - 1).min(),
        'win_rate': calculate_win_rate(trades, df),
        'trades': trades,
        'equity_curve': equity_series.to_list()
    }


def calculate_win_rate(trades: list, df: pd.DataFrame) -> float:
    """Calculate win rate from trades"""
    if len(trades) < 2:
        return 0

    wins = 0
    total = 0

    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            buy = trades[i]
            sell = trades[i + 1]
            if sell['price'] > buy['price']:
                wins += 1
            total += 1

    return wins / total if total > 0 else 0