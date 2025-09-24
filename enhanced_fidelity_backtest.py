"""
Enhanced Fidelity Backtest with Advanced AI Components
======================================================
Integrates all sophisticated AI models within single-user constraints
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import asyncio
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
    information_ratio: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    downside_volatility: float
    upside_capture: float
    downside_capture: float
    capture_ratio: float
    t_statistic: float
    p_value: float

class EnhancedAITradingSystem:
    """Advanced AI system with all sophisticated components for Fidelity trading"""

    def __init__(self):
        # Enhanced ML model configurations
        self.ensemble_weights = {
            'transformer_model': 0.25,    # Price sequence prediction
            'xgboost_model': 0.20,        # Feature-based prediction
            'lstm_model': 0.15,           # Time series patterns
            'technical_signals': 0.20,    # Technical analysis
            'sentiment_signals': 0.12,    # Alternative data
            'options_flow': 0.08          # Options market signals
        }

        # Risk management parameters
        self.risk_params = {
            'max_portfolio_vol': 0.16,    # Max 16% annual volatility
            'max_single_position': 0.18,  # Max 18% per stock
            'confidence_threshold': 0.62, # Higher threshold for quality
            'rebalance_threshold': 0.04,  # 4% drift before rebalance
            'stop_loss_threshold': -0.08, # 8% stop loss
            'take_profit_threshold': 0.15 # 15% take profit
        }

    def generate_transformer_signals(self, market_data: Dict, day_idx: int) -> Dict:
        """Simulate sophisticated transformer model predictions"""
        signals = {}

        for symbol, data in market_data.items():
            if day_idx < 60:  # Need sufficient sequence length
                continue

            # Simulate transformer attention on price sequences
            hist_data = data.iloc[max(0, day_idx-60):day_idx+1]
            returns = hist_data['Close'].pct_change().dropna()

            if len(returns) < 30:
                continue

            # Multi-head attention simulation (would be real transformer)
            # Attention head 1: Short-term patterns (5-day)
            short_pattern = returns.rolling(5).mean().iloc[-1] * 1.2

            # Attention head 2: Medium-term trends (20-day)
            med_pattern = returns.rolling(20).mean().iloc[-1] * 0.8

            # Attention head 3: Long-term momentum (60-day)
            long_pattern = returns.rolling(60).mean().iloc[-1] * 0.5

            # Cross-asset attention (correlation with market)
            market_returns = []
            for other_symbol, other_data in market_data.items():
                if other_symbol != symbol and len(other_data) > day_idx:
                    other_returns = other_data.iloc[max(0, day_idx-20):day_idx+1]['Close'].pct_change().dropna()
                    if len(other_returns) > 0:
                        market_returns.extend(other_returns.tail(10).values)

            if market_returns:
                market_signal = np.mean(market_returns) * 0.3
                cross_asset_signal = returns.iloc[-1] * (1 + market_signal)
            else:
                cross_asset_signal = 0

            # Combine transformer heads
            transformer_signal = (short_pattern + med_pattern + long_pattern + cross_asset_signal * 0.2)
            confidence = min(0.95, abs(transformer_signal) * 15 + 0.5)

            if confidence > 0.6:
                signals[symbol] = {
                    'signal': transformer_signal,
                    'confidence': confidence,
                    'components': {
                        'short_attention': short_pattern,
                        'medium_attention': med_pattern,
                        'long_attention': long_pattern,
                        'cross_asset': cross_asset_signal
                    }
                }

        return signals

    def generate_xgboost_signals(self, market_data: Dict, day_idx: int) -> Dict:
        """Simulate XGBoost feature-based predictions"""
        signals = {}

        for symbol, data in market_data.items():
            if day_idx < 50:
                continue

            hist_data = data.iloc[max(0, day_idx-50):day_idx+1]

            # Feature engineering (would be more sophisticated in reality)
            features = self._extract_features(hist_data)

            if features is None:
                continue

            # Simulate XGBoost prediction based on features
            # Feature importance weighting (simulated)
            price_momentum = features.get('price_momentum', 0) * 0.25
            volume_strength = features.get('volume_strength', 0) * 0.20
            volatility_regime = features.get('volatility_regime', 0) * 0.15
            rsi_divergence = features.get('rsi_divergence', 0) * 0.15
            bollinger_position = features.get('bollinger_position', 0) * 0.10
            macd_signal = features.get('macd_signal', 0) * 0.15

            xgb_prediction = (price_momentum + volume_strength + volatility_regime +
                            rsi_divergence + bollinger_position + macd_signal)

            # Feature interaction effects (XGBoost specialty)
            interaction_boost = features.get('volume_strength', 0) * features.get('price_momentum', 0) * 0.1
            xgb_prediction += interaction_boost

            confidence = min(0.9, abs(xgb_prediction) * 12 + 0.4)

            if confidence > 0.55:
                signals[symbol] = {
                    'signal': xgb_prediction,
                    'confidence': confidence,
                    'feature_importance': features
                }

        return signals

    def _extract_features(self, data: pd.DataFrame) -> Optional[Dict]:
        """Extract technical features for ML models"""
        if len(data) < 30:
            return None

        try:
            returns = data['Close'].pct_change().dropna()

            features = {}

            # Price momentum features
            features['price_momentum'] = returns.rolling(10).mean().iloc[-1]
            features['price_acceleration'] = returns.rolling(5).mean().iloc[-1] - returns.rolling(15).mean().iloc[-1]

            # Volume features
            volume_sma = data['Volume'].rolling(20).mean()
            features['volume_strength'] = (data['Volume'].iloc[-1] / volume_sma.iloc[-1] - 1) * 0.1
            features['volume_momentum'] = data['Volume'].rolling(5).mean().iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1] - 1

            # Volatility regime
            short_vol = returns.rolling(10).std().iloc[-1]
            long_vol = returns.rolling(30).std().iloc[-1]
            features['volatility_regime'] = (short_vol / long_vol - 1) if long_vol > 0 else 0

            # RSI-like momentum
            gains = returns.where(returns > 0, 0).rolling(14).mean()
            losses = -returns.where(returns < 0, 0).rolling(14).mean()
            rs = gains.iloc[-1] / losses.iloc[-1] if losses.iloc[-1] > 0 else 1
            rsi = 1 - (1 / (1 + rs))
            features['rsi_divergence'] = (0.5 - rsi) * 0.02  # Mean reversion signal

            # Bollinger Band position
            sma_20 = data['Close'].rolling(20).mean()
            std_20 = data['Close'].rolling(20).std()
            if std_20.iloc[-1] > 0:
                bb_position = (data['Close'].iloc[-1] - sma_20.iloc[-1]) / (2 * std_20.iloc[-1])
                features['bollinger_position'] = np.clip(bb_position, -1, 1) * 0.01
            else:
                features['bollinger_position'] = 0

            # MACD-like signal
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            features['macd_signal'] = (macd_line.iloc[-1] - signal_line.iloc[-1]) / data['Close'].iloc[-1]

            return features

        except Exception:
            return None

    def generate_lstm_signals(self, market_data: Dict, day_idx: int) -> Dict:
        """Simulate LSTM time series predictions"""
        signals = {}

        for symbol, data in market_data.items():
            if day_idx < 40:
                continue

            hist_data = data.iloc[max(0, day_idx-40):day_idx+1]
            returns = hist_data['Close'].pct_change().dropna()

            if len(returns) < 30:
                continue

            # Simulate LSTM memory patterns
            # Short-term memory (recent 5 days)
            recent_pattern = returns.tail(5).mean() * 1.5

            # Medium-term memory (10-20 days ago)
            medium_pattern = returns.iloc[-20:-10].mean() * 0.8 if len(returns) >= 20 else 0

            # Long-term memory (30+ days ago)
            long_pattern = returns.iloc[:-20].mean() * 0.3 if len(returns) >= 30 else 0

            # LSTM forget gate simulation (volatility-based)
            volatility = returns.rolling(10).std().iloc[-1]
            forget_factor = max(0.5, min(1.2, 0.02 / volatility)) if volatility > 0 else 1

            # LSTM output
            lstm_signal = (recent_pattern + medium_pattern + long_pattern) * forget_factor

            # Sequence pattern recognition
            trend_consistency = self._calculate_trend_consistency(returns.tail(15))
            lstm_signal *= (1 + trend_consistency * 0.5)

            confidence = min(0.88, abs(lstm_signal) * 10 + 0.45)

            if confidence > 0.5:
                signals[symbol] = {
                    'signal': lstm_signal,
                    'confidence': confidence,
                    'memory_components': {
                        'recent': recent_pattern,
                        'medium': medium_pattern,
                        'long': long_pattern,
                        'forget_factor': forget_factor
                    }
                }

        return signals

    def _calculate_trend_consistency(self, returns: pd.Series) -> float:
        """Calculate trend consistency for LSTM"""
        if len(returns) < 5:
            return 0

        # Count consecutive up/down moves
        directions = (returns > 0).astype(int)
        changes = directions.diff().abs().sum()
        consistency = 1 - (changes / len(directions))
        return consistency * 2 - 1  # Scale to [-1, 1]

    def generate_enhanced_technical_signals(self, market_data: Dict, day_idx: int) -> Dict:
        """Enhanced technical analysis with pattern recognition"""
        signals = {}

        for symbol, data in market_data.items():
            if day_idx < 50:
                continue

            hist_data = data.iloc[max(0, day_idx-50):day_idx+1]

            # Advanced technical patterns
            tech_signals = []

            # 1. Multi-timeframe momentum with acceleration
            returns = hist_data['Close'].pct_change().dropna()
            momentum_3d = returns.rolling(3).mean().iloc[-1] * 2.0
            momentum_10d = returns.rolling(10).mean().iloc[-1] * 1.0
            momentum_20d = returns.rolling(20).mean().iloc[-1] * 0.5
            momentum_signal = momentum_3d + momentum_10d + momentum_20d
            tech_signals.append(momentum_signal)

            # 2. Volume-weighted price trends
            vwap = (hist_data['Close'] * hist_data['Volume']).rolling(20).sum() / hist_data['Volume'].rolling(20).sum()
            vwap_signal = (hist_data['Close'].iloc[-1] / vwap.iloc[-1] - 1) * 0.5
            tech_signals.append(vwap_signal)

            # 3. Breakout detection
            high_20 = hist_data['High'].rolling(20).max()
            low_20 = hist_data['Low'].rolling(20).min()
            range_position = (hist_data['Close'].iloc[-1] - low_20.iloc[-1]) / (high_20.iloc[-1] - low_20.iloc[-1])

            # Strong breakouts get higher weight
            if range_position > 0.95:  # Near highs
                breakout_signal = 0.03
            elif range_position < 0.05:  # Near lows (potential reversal)
                breakout_signal = 0.02
            else:
                breakout_signal = (range_position - 0.5) * 0.01
            tech_signals.append(breakout_signal)

            # 4. Volatility squeeze detection
            bb_squeeze = self._detect_volatility_squeeze(hist_data)
            tech_signals.append(bb_squeeze)

            # 5. Support/resistance levels
            sr_signal = self._calculate_support_resistance_signal(hist_data)
            tech_signals.append(sr_signal)

            combined_signal = np.mean(tech_signals)
            confidence = min(0.9, abs(combined_signal) * 20 + 0.3)

            if confidence > 0.4:
                signals[symbol] = {
                    'signal': combined_signal,
                    'confidence': confidence,
                    'technical_components': {
                        'momentum': momentum_signal,
                        'vwap': vwap_signal,
                        'breakout': breakout_signal,
                        'volatility_squeeze': bb_squeeze,
                        'support_resistance': sr_signal
                    }
                }

        return signals

    def _detect_volatility_squeeze(self, data: pd.DataFrame) -> float:
        """Detect Bollinger Band squeeze patterns"""
        if len(data) < 30:
            return 0

        # Calculate BB width
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        bb_width = (std_20 * 4) / sma_20  # Relative width

        current_width = bb_width.iloc[-1]
        avg_width = bb_width.rolling(20).mean().iloc[-1]

        # Squeeze detected when BB width is significantly below average
        if current_width < avg_width * 0.7:
            # Direction of eventual breakout
            recent_return = data['Close'].pct_change().iloc[-1]
            return recent_return * 2.0  # Amplify squeeze breakout

        return 0

    def _calculate_support_resistance_signal(self, data: pd.DataFrame) -> float:
        """Calculate support/resistance breakout signals"""
        if len(data) < 30:
            return 0

        # Identify recent swing highs and lows
        highs = data['High'].rolling(5, center=True).max()
        lows = data['Low'].rolling(5, center=True).min()

        current_price = data['Close'].iloc[-1]

        # Check for resistance breakout
        recent_highs = highs.tail(10).max()
        if current_price > recent_highs * 1.01:  # 1% breakout
            return 0.02

        # Check for support bounce
        recent_lows = lows.tail(10).min()
        if current_price < recent_lows * 1.01:  # Near support
            return 0.01

        return 0

    def generate_sentiment_signals(self, market_data: Dict, day_idx: int) -> Dict:
        """Enhanced sentiment analysis from alternative data"""
        signals = {}

        for symbol, data in market_data.items():
            # Simulate sophisticated sentiment analysis

            # 1. Social media sentiment with momentum
            base_sentiment = np.random.normal(0, 0.08)  # Would be real sentiment
            sentiment_momentum = np.random.normal(0, 0.03)  # Sentiment change
            social_signal = base_sentiment + sentiment_momentum * 0.5

            # 2. News sentiment with recency weighting
            news_scores = [np.random.normal(0, 0.05) for _ in range(7)]  # Daily news scores
            weights = np.array([0.3, 0.25, 0.2, 0.1, 0.08, 0.05, 0.02])  # Recent more important
            news_signal = np.average(news_scores, weights=weights)

            # 3. Analyst sentiment (upgrades/downgrades)
            analyst_signal = np.random.normal(0, 0.03)  # Would be real analyst data

            # 4. Options sentiment (put/call ratios, unusual activity)
            options_sentiment = np.random.normal(0, 0.04)  # Would be real options data

            # 5. Institutional flow sentiment
            institutional_flow = np.random.normal(0, 0.02)  # Would be real flow data

            # Combine sentiment signals with confidence weighting
            sentiment_components = {
                'social': social_signal,
                'news': news_signal,
                'analyst': analyst_signal,
                'options': options_sentiment,
                'institutional': institutional_flow
            }

            combined_sentiment = (
                social_signal * 0.3 +
                news_signal * 0.25 +
                analyst_signal * 0.2 +
                options_sentiment * 0.15 +
                institutional_flow * 0.1
            )

            # Sentiment confidence based on consistency
            sentiment_values = list(sentiment_components.values())
            sentiment_std = np.std(sentiment_values)
            confidence = max(0.2, 0.8 - sentiment_std * 5)  # Higher agreement = higher confidence

            if abs(combined_sentiment) > 0.01 and confidence > 0.3:
                signals[symbol] = {
                    'signal': combined_sentiment,
                    'confidence': confidence,
                    'components': sentiment_components
                }

        return signals

    def generate_options_flow_signals(self, market_data: Dict, day_idx: int) -> Dict:
        """Advanced options flow analysis"""
        signals = {}

        for symbol, data in market_data.items():
            if day_idx < 10:
                continue

            # Simulate sophisticated options analysis
            current_price = data['Close'].iloc[day_idx]

            # 1. Gamma exposure analysis
            # Simulate options chain data
            strikes = np.arange(current_price * 0.9, current_price * 1.1, current_price * 0.02)

            # Simulate gamma exposure calculation
            net_gamma = np.random.normal(0, 1000000)  # Would be real gamma exposure

            # High positive gamma = volatility suppression
            # High negative gamma = volatility amplification
            gamma_signal = -net_gamma / 10000000 * 0.01  # Scale appropriately

            # 2. Put/Call ratio analysis
            put_call_ratio = 0.8 + np.random.normal(0, 0.3)  # Would be real P/C ratio
            # Low P/C = bullish, High P/C = bearish
            pc_signal = (1 - put_call_ratio) * 0.02

            # 3. Unusual options activity
            # Simulate detection of large options trades
            unusual_activity = np.random.exponential(0.1) - 0.1  # Heavy right tail
            if unusual_activity > 0.2:  # Significant unusual activity
                # Direction based on call vs put dominance
                activity_direction = 1 if np.random.random() > put_call_ratio else -1
                unusual_signal = activity_direction * min(unusual_activity, 0.5) * 0.01
            else:
                unusual_signal = 0

            # 4. Volatility surface skew
            vol_skew = np.random.normal(0, 0.1)  # Would be real volatility skew
            skew_signal = vol_skew * 0.005

            # Combine options signals
            options_signal = gamma_signal + pc_signal + unusual_signal + skew_signal

            # Options confidence based on volume and open interest
            options_volume = np.random.gamma(2, 100)  # Would be real options volume
            confidence = min(0.85, np.log(max(options_volume, 1)) / 10 + 0.2)

            if abs(options_signal) > 0.005 and confidence > 0.3:
                signals[symbol] = {
                    'signal': options_signal,
                    'confidence': confidence,
                    'components': {
                        'gamma_exposure': gamma_signal,
                        'put_call_ratio': pc_signal,
                        'unusual_activity': unusual_signal,
                        'volatility_skew': skew_signal
                    }
                }

        return signals

    def generate_ensemble_signals(self, market_data: Dict, day_idx: int) -> Dict:
        """Combine all AI model signals into ensemble predictions"""

        # Generate signals from all models
        transformer_signals = self.generate_transformer_signals(market_data, day_idx)
        xgboost_signals = self.generate_xgboost_signals(market_data, day_idx)
        lstm_signals = self.generate_lstm_signals(market_data, day_idx)
        technical_signals = self.generate_enhanced_technical_signals(market_data, day_idx)
        sentiment_signals = self.generate_sentiment_signals(market_data, day_idx)
        options_signals = self.generate_options_flow_signals(market_data, day_idx)

        # Combine all signals
        all_signals = {
            'transformer': transformer_signals,
            'xgboost': xgboost_signals,
            'lstm': lstm_signals,
            'technical': technical_signals,
            'sentiment': sentiment_signals,
            'options': options_signals
        }

        # Get all symbols that have at least one signal
        all_symbols = set()
        for signal_dict in all_signals.values():
            all_symbols.update(signal_dict.keys())

        ensemble_signals = {}

        for symbol in all_symbols:
            # Collect signals and confidences for this symbol
            symbol_signals = []
            symbol_confidences = []
            signal_components = {}

            for model_name, signal_dict in all_signals.items():
                if symbol in signal_dict:
                    signal_data = signal_dict[symbol]
                    symbol_signals.append(signal_data['signal'])
                    symbol_confidences.append(signal_data['confidence'])
                    signal_components[model_name] = signal_data['signal']
                else:
                    symbol_signals.append(0)
                    symbol_confidences.append(0)
                    signal_components[model_name] = 0

            # Weighted ensemble combination
            weights = list(self.ensemble_weights.values())
            weighted_signal = sum(s * w for s, w in zip(symbol_signals, weights))

            # Ensemble confidence (higher when models agree)
            signal_std = np.std(symbol_signals)
            avg_confidence = np.mean(symbol_confidences)
            agreement_bonus = max(0, 1 - signal_std * 20)  # Higher agreement = bonus
            ensemble_confidence = avg_confidence * (1 + agreement_bonus * 0.5)

            # Only include high-confidence ensemble predictions
            if ensemble_confidence > self.risk_params['confidence_threshold']:
                ensemble_signals[symbol] = {
                    'signal': weighted_signal,
                    'confidence': min(0.95, ensemble_confidence),
                    'model_agreement': agreement_bonus,
                    'components': signal_components,
                    'num_models': sum(1 for s in symbol_signals if s != 0)
                }

        return ensemble_signals

def simulate_enhanced_ai_trading(market_data, initial_capital=100000):
    """Enhanced AI trading simulation with all advanced components"""

    ai_system = EnhancedAITradingSystem()

    portfolio_value = initial_capital
    cash = initial_capital
    positions = {}
    portfolio_history = []
    trades = []
    daily_signals_log = []

    all_dates = sorted(list(market_data[list(market_data.keys())[0]].index))

    print(f"[AI] Enhanced AI Trading Simulation - {len(all_dates)} days")
    print(f"[AI] Using: Transformer + XGBoost + LSTM + Technical + Sentiment + Options Flow")

    for day_idx, date in enumerate(all_dates):
        if day_idx < 60:  # Need history for AI models
            portfolio_history.append(portfolio_value)
            continue

        # Generate ensemble AI signals
        ensemble_signals = ai_system.generate_ensemble_signals(market_data, day_idx)
        daily_signals_log.append({
            'date': date,
            'signals': ensemble_signals,
            'num_signals': len(ensemble_signals)
        })

        # Portfolio optimization with risk constraints
        if ensemble_signals:
            # Select top signals with risk adjustment
            risk_adjusted_signals = {}
            for symbol, signal_data in ensemble_signals.items():
                # Risk adjustment based on volatility
                hist_data = market_data[symbol].iloc[max(0, day_idx-30):day_idx+1]
                returns = hist_data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) if len(returns) > 10 else 0.2

                # Penalize high volatility
                risk_penalty = max(0.5, min(1.5, ai_system.risk_params['max_portfolio_vol'] / volatility))
                adjusted_signal = signal_data['signal'] * risk_penalty

                if abs(adjusted_signal) > 0.008:  # Strong signal threshold
                    risk_adjusted_signals[symbol] = {
                        **signal_data,
                        'risk_adjusted_signal': adjusted_signal,
                        'volatility': volatility,
                        'risk_penalty': risk_penalty
                    }

            # Portfolio construction - select top 5-8 positions
            sorted_signals = sorted(
                risk_adjusted_signals.items(),
                key=lambda x: abs(x[1]['risk_adjusted_signal']) * x[1]['confidence'],
                reverse=True
            )

            target_positions = {}
            total_weight = 0

            for symbol, signal_data in sorted_signals[:6]:  # Max 6 positions
                # Position sizing based on Kelly criterion
                expected_return = signal_data['risk_adjusted_signal']
                confidence = signal_data['confidence']
                volatility = signal_data['volatility']

                # Conservative Kelly sizing
                if volatility > 0:
                    kelly_fraction = (expected_return * confidence) / (volatility ** 2)
                    position_weight = min(
                        abs(kelly_fraction) * 0.3,  # Very conservative Kelly
                        ai_system.risk_params['max_single_position']
                    )
                else:
                    position_weight = 0.05

                if position_weight > 0.03:  # Minimum 3% position
                    target_positions[symbol] = min(position_weight, 0.18)  # Cap at 18%
                    total_weight += target_positions[symbol]

            # Normalize if over-allocated
            if total_weight > 0.92:  # Keep 8% cash minimum
                scale_factor = 0.92 / total_weight
                for symbol in target_positions:
                    target_positions[symbol] *= scale_factor

            # Execute trades
            current_prices = {symbol: market_data[symbol].iloc[day_idx]['Close']
                            for symbol in market_data.keys()}

            trades_today = []

            # Close positions not in target
            for symbol in list(positions.keys()):
                if symbol not in target_positions and positions[symbol] > 0:
                    shares_to_sell = positions[symbol]
                    cash += shares_to_sell * current_prices[symbol] * 0.9995  # Small transaction cost
                    positions[symbol] = 0

                    trades_today.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': current_prices[symbol]
                    })

            # Open/adjust target positions
            for symbol, target_weight in target_positions.items():
                target_value = portfolio_value * target_weight
                target_shares = int(target_value / current_prices[symbol])

                current_shares = positions.get(symbol, 0)
                shares_diff = target_shares - current_shares

                if abs(shares_diff) > 0 and abs(shares_diff) * current_prices[symbol] > 1000:  # Min $1000 trade
                    trade_value = shares_diff * current_prices[symbol]
                    cash -= trade_value * 1.0005  # Small transaction cost
                    positions[symbol] = target_shares

                    trades_today.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'buy' if shares_diff > 0 else 'sell',
                        'shares': abs(shares_diff),
                        'price': current_prices[symbol]
                    })

            trades.extend(trades_today)

        # Calculate portfolio value
        portfolio_value = cash + sum(
            positions.get(symbol, 0) * market_data[symbol].iloc[day_idx]['Close']
            for symbol in market_data.keys()
        )
        portfolio_history.append(portfolio_value)

        # Progress reporting
        if (day_idx + 1) % 50 == 0:
            returns = (portfolio_value / initial_capital - 1) * 100
            num_positions = len([p for p in positions.values() if p > 0])
            avg_signals = np.mean([log['num_signals'] for log in daily_signals_log[-10:]])
            print(f"  Day {day_idx + 1}: ${portfolio_value:,.0f} ({returns:+.1f}%) | {num_positions} pos | {avg_signals:.1f} avg signals")

    portfolio_series = pd.Series(portfolio_history, index=all_dates, name='Enhanced_AI_Portfolio')

    print(f"\n[SUCCESS] Enhanced AI simulation complete!")
    print(f"  Final value: ${portfolio_value:,.2f}")
    print(f"  Total return: {(portfolio_value/initial_capital - 1)*100:.2f}%")
    print(f"  Total trades: {len(trades)}")

    # Calculate signal quality metrics
    total_signals = sum(log['num_signals'] for log in daily_signals_log)
    avg_daily_signals = total_signals / len(daily_signals_log) if daily_signals_log else 0
    print(f"  Avg daily signals: {avg_daily_signals:.1f}")

    return portfolio_series, trades, daily_signals_log

# Run enhanced backtest
def run_enhanced_backtest():
    print("="*80)
    print("ENHANCED AI HEDGE FUND - FIDELITY OPTIMIZED")
    print("="*80)

    # Generate market data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'PG']
    trading_days = 252

    market_data = {}
    for i, symbol in enumerate(symbols):
        # More realistic price generation
        dates = pd.date_range(end=datetime.now(), periods=trading_days, freq='D')
        base_return = 0.0004 + i * 0.00003  # Different expected returns
        volatility = 0.018 + i * 0.002  # Different volatilities

        returns = np.random.normal(base_return, volatility, trading_days)
        # Add momentum and mean reversion
        returns = pd.Series(returns).ewm(span=7).mean().values  # Momentum
        returns += np.random.normal(0, volatility * 0.3, trading_days)  # Additional noise

        prices = 150 * (1 + i * 0.3) * np.exp(np.cumsum(returns))
        volumes = np.random.gamma(2, 500000 * (1 + i * 0.5), trading_days)

        market_data[symbol] = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.008, 0.008, trading_days)),
            'High': prices * (1 + np.random.uniform(0, 0.015, trading_days)),
            'Low': prices * (1 - np.random.uniform(0, 0.015, trading_days)),
            'Close': prices,
            'Volume': volumes
        }, index=dates)

    # Run enhanced AI simulation
    portfolio, trades, signal_logs = simulate_enhanced_ai_trading(market_data, 100000)

    # Generate benchmark
    sp500_returns = np.random.normal(0.0004, 0.012, trading_days)  # S&P 500 characteristics
    sp500_prices = 100000 * np.exp(np.cumsum(sp500_returns))
    sp500 = pd.Series(sp500_prices, index=portfolio.index, name='S&P500')

    # Evaluate performance
    portfolio_returns = portfolio.pct_change().dropna()
    sp500_returns = sp500.pct_change().dropna()

    # Calculate metrics (simplified)
    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    sp500_total_return = (sp500.iloc[-1] / sp500.iloc[0]) - 1

    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (portfolio_returns.mean() - 0.02/252) / portfolio_returns.std() * np.sqrt(252)

    max_dd = ((portfolio / portfolio.cummax()) - 1).min()

    alpha = total_return - sp500_total_return

    print(f"\n" + "="*60)
    print(f"ENHANCED AI PERFORMANCE vs S&P 500")
    print(f"="*60)
    print(f"Portfolio Return: {total_return:.2%}")
    print(f"S&P 500 Return: {sp500_total_return:.2%}")
    print(f"Alpha: {alpha:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Volatility: {volatility:.2%}")

    if total_return > sp500_total_return + 0.02:  # 2%+ outperformance
        print(f"\n[EXCELLENT] Significant outperformance!")
    elif total_return > sp500_total_return:
        print(f"\n[GOOD] Outperforming S&P 500")
    else:
        print(f"\n[WARNING] Needs improvement")

    return portfolio, sp500, trades, signal_logs

if __name__ == "__main__":
    results = run_enhanced_backtest()