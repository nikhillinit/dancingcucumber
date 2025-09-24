"""
Single-User AI Trading System for Fidelity
==========================================
Leverages all advanced AI components within individual trading constraints:
- Daily trades only (no high-frequency)
- Long positions only (no derivatives/shorts)
- Real-time data processing
- Ensemble ML predictions
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our advanced AI components
from concurrent.futures import ThreadPoolExecutor
import json


class SingleUserAITradingSystem:
    """
    Complete AI trading system optimized for single-user Fidelity account
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []

        # AI Components
        self.ensemble_predictor = EnsembleMLPredictor()
        self.signal_validator = AdvancedSignalValidator()
        self.portfolio_optimizer = DailyPortfolioOptimizer()
        self.risk_manager = SingleUserRiskManager()
        self.alternative_data_analyzer = AlternativeDataAnalyzer()

        # Configuration for single-user constraints
        self.config = {
            'max_positions': 8,           # Max 8 stocks to manage easily
            'max_single_position': 0.20,  # Max 20% per stock
            'cash_buffer': 0.05,          # Keep 5% cash
            'rebalance_threshold': 0.05,  # Rebalance if >5% drift
            'confidence_threshold': 0.65, # High confidence trades only
            'max_daily_trades': 12,       # Reasonable for manual execution
        }

    async def generate_daily_signals(self, market_data: Dict) -> Dict:
        """
        Generate comprehensive daily trading signals using all AI components
        """
        print("[AI] Generating daily signals with advanced AI components...")

        # Run all AI components in parallel for efficiency
        tasks = [
            self.ensemble_predictor.predict_daily_returns(market_data),
            self.alternative_data_analyzer.get_sentiment_signals(market_data),
            self.signal_validator.validate_market_regime(market_data),
            self.risk_manager.assess_market_conditions(market_data)
        ]

        results = await asyncio.gather(*tasks)
        ml_predictions, sentiment_signals, regime_data, risk_assessment = results

        # Combine all signals into unified recommendations
        unified_signals = {}

        for symbol in market_data.keys():
            # 1. ML Ensemble Prediction (40% weight)
            ml_signal = ml_predictions.get(symbol, {})
            ml_score = ml_signal.get('expected_return', 0) * 0.4
            ml_confidence = ml_signal.get('confidence', 0)

            # 2. Alternative Data Sentiment (25% weight)
            sentiment = sentiment_signals.get(symbol, {})
            sentiment_score = sentiment.get('composite_sentiment', 0) * 0.25
            sentiment_confidence = sentiment.get('confidence', 0)

            # 3. Technical Pattern Recognition (20% weight)
            tech_signal = self._advanced_technical_analysis(market_data[symbol])
            tech_score = tech_signal['signal'] * 0.2
            tech_confidence = tech_signal['confidence']

            # 4. Options Flow Analysis (15% weight)
            options_signal = self._analyze_options_flow(symbol, market_data[symbol])
            options_score = options_signal['signal'] * 0.15
            options_confidence = options_signal['confidence']

            # Combine signals with regime adjustment
            regime_multiplier = regime_data.get('market_regime_multiplier', 1.0)

            combined_signal = (ml_score + sentiment_score + tech_score + options_score) * regime_multiplier
            combined_confidence = np.mean([ml_confidence, sentiment_confidence, tech_confidence, options_confidence])

            # Apply risk constraints
            risk_adjustment = risk_assessment.get(symbol, {}).get('risk_multiplier', 1.0)
            final_signal = combined_signal * risk_adjustment

            if final_signal > 0 and combined_confidence > self.config['confidence_threshold']:
                unified_signals[symbol] = {
                    'signal': final_signal,
                    'confidence': combined_confidence,
                    'components': {
                        'ml_prediction': ml_score / 0.4,
                        'sentiment': sentiment_score / 0.25,
                        'technical': tech_score / 0.2,
                        'options_flow': options_score / 0.15
                    },
                    'expected_return': ml_signal.get('expected_return', 0),
                    'risk_score': risk_assessment.get(symbol, {}).get('risk_score', 0.5)
                }

        return unified_signals

    def _advanced_technical_analysis(self, price_data: pd.DataFrame) -> Dict:
        """Advanced technical analysis combining multiple indicators"""

        if len(price_data) < 50:
            return {'signal': 0, 'confidence': 0}

        signals = []

        # 1. Multi-timeframe momentum
        returns_5d = price_data['Close'].pct_change(5).iloc[-1]
        returns_20d = price_data['Close'].pct_change(20).iloc[-1]
        momentum_signal = (returns_5d * 0.7 + returns_20d * 0.3) * 2
        signals.append(momentum_signal)

        # 2. Volume-price trend
        volume_ma = price_data['Volume'].rolling(20).mean()
        price_change = price_data['Close'].pct_change()
        volume_ratio = price_data['Volume'].iloc[-1] / volume_ma.iloc[-1]
        vpt_signal = price_change.iloc[-1] * min(volume_ratio, 2.0)
        signals.append(vpt_signal)

        # 3. Volatility breakout
        volatility = price_data['Close'].rolling(20).std()
        vol_percentile = (volatility.iloc[-1] - volatility.rolling(60).mean().iloc[-1]) / volatility.rolling(60).std().iloc[-1]
        breakout_signal = price_change.iloc[-1] * (1 + max(0, vol_percentile * 0.5))
        signals.append(breakout_signal)

        # 4. Support/Resistance
        high_20 = price_data['High'].rolling(20).max()
        low_20 = price_data['Low'].rolling(20).min()
        price_position = (price_data['Close'].iloc[-1] - low_20.iloc[-1]) / (high_20.iloc[-1] - low_20.iloc[-1])
        sr_signal = (price_position - 0.5) * 0.02  # Favor breakouts
        signals.append(sr_signal)

        combined_signal = np.mean(signals)
        confidence = min(0.95, abs(combined_signal) * 20 + 0.3)

        return {
            'signal': combined_signal,
            'confidence': confidence
        }

    def _analyze_options_flow(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Simulate options flow analysis (would use real data in production)"""

        # Simulate institutional options activity
        price_change = price_data['Close'].pct_change(5).iloc[-1]
        volume_surge = price_data['Volume'].iloc[-1] / price_data['Volume'].rolling(20).mean().iloc[-1]

        # Simulate gamma exposure effect
        gamma_signal = price_change * min(volume_surge, 3.0) * 0.1

        # Simulate put/call ratio effect
        simulated_pcr = 0.8 + np.random.normal(0, 0.2)  # Would be real data
        pcr_signal = (1 - simulated_pcr) * 0.05  # Lower P/C ratio = bullish

        combined_signal = gamma_signal + pcr_signal
        confidence = min(0.8, abs(combined_signal) * 15 + 0.2)

        return {
            'signal': combined_signal,
            'confidence': confidence
        }

    async def optimize_daily_portfolio(self, signals: Dict, current_portfolio: Dict) -> Dict:
        """
        Optimize portfolio for single daily rebalancing
        """
        print("[AI] Optimizing portfolio allocation...")

        # Sort signals by risk-adjusted return
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1]['signal'] / max(x[1]['risk_score'], 0.1),
            reverse=True
        )

        # Select top positions within constraints
        selected_positions = {}
        total_weight = 0

        for symbol, signal_data in sorted_signals:
            if len(selected_positions) >= self.config['max_positions']:
                break

            # Calculate optimal position size
            expected_return = signal_data['expected_return']
            risk_score = signal_data['risk_score']
            confidence = signal_data['confidence']

            # Kelly Criterion adjusted for confidence
            if risk_score > 0:
                kelly_weight = (expected_return * confidence) / risk_score
                position_weight = min(
                    kelly_weight * 0.5,  # Conservative Kelly
                    self.config['max_single_position']
                )
            else:
                position_weight = 0.05  # Minimum position

            if position_weight > 0.02:  # Minimum 2% position
                selected_positions[symbol] = {
                    'target_weight': position_weight,
                    'signal_strength': signal_data['signal'],
                    'confidence': confidence,
                    'expected_return': expected_return
                }
                total_weight += position_weight

        # Normalize weights to fit cash constraints
        max_total_weight = 1.0 - self.config['cash_buffer']
        if total_weight > max_total_weight:
            scale_factor = max_total_weight / total_weight
            for symbol in selected_positions:
                selected_positions[symbol]['target_weight'] *= scale_factor

        return selected_positions

    async def execute_daily_trades(self, target_portfolio: Dict, current_prices: Dict) -> List[Dict]:
        """
        Generate trade recommendations for manual execution
        """
        print("[AI] Generating trade recommendations...")

        trades = []
        current_value = self.cash + sum(
            self.positions.get(symbol, 0) * current_prices[symbol]
            for symbol in current_prices
        )

        # Close positions not in target portfolio
        for symbol, current_shares in self.positions.items():
            if symbol not in target_portfolio and current_shares > 0:
                trades.append({
                    'action': 'SELL',
                    'symbol': symbol,
                    'shares': current_shares,
                    'reason': 'Exit position - not in target portfolio',
                    'priority': 'HIGH',
                    'estimated_value': current_shares * current_prices[symbol]
                })

        # Enter/adjust target positions
        for symbol, target_data in target_portfolio.items():
            target_weight = target_data['target_weight']
            target_value = current_value * target_weight
            target_shares = int(target_value / current_prices[symbol])

            current_shares = self.positions.get(symbol, 0)
            shares_diff = target_shares - current_shares

            if abs(shares_diff) > 0:
                action = 'BUY' if shares_diff > 0 else 'SELL'
                trades.append({
                    'action': action,
                    'symbol': symbol,
                    'shares': abs(shares_diff),
                    'reason': f'Target allocation: {target_weight:.1%}',
                    'priority': 'MEDIUM' if abs(shares_diff) < 50 else 'HIGH',
                    'estimated_value': abs(shares_diff) * current_prices[symbol],
                    'confidence': target_data['confidence'],
                    'expected_return': target_data['expected_return']
                })

        # Sort trades by priority and value
        trades.sort(key=lambda x: (x['priority'] == 'HIGH', x['estimated_value']), reverse=True)

        # Limit to max daily trades
        if len(trades) > self.config['max_daily_trades']:
            trades = trades[:self.config['max_daily_trades']]
            print(f"[WARNING] Limited to {self.config['max_daily_trades']} trades. Consider increasing limit.")

        return trades

    async def run_daily_analysis(self, market_data: Dict) -> Dict:
        """
        Complete daily analysis and trade generation
        """
        print("\n" + "="*60)
        print(f"[AI] DAILY ANALYSIS - {datetime.now().strftime('%Y-%m-%d')}")
        print("="*60)

        try:
            # 1. Generate signals
            signals = await self.generate_daily_signals(market_data)
            print(f"[SUCCESS] Generated {len(signals)} high-confidence signals")

            # 2. Optimize portfolio
            current_portfolio = {symbol: shares for symbol, shares in self.positions.items() if shares > 0}
            target_portfolio = await self.optimize_daily_portfolio(signals, current_portfolio)
            print(f"[SUCCESS] Optimized portfolio with {len(target_portfolio)} target positions")

            # 3. Generate trades
            current_prices = {symbol: data['Close'].iloc[-1] for symbol, data in market_data.items()}
            trades = await self.execute_daily_trades(target_portfolio, current_prices)
            print(f"[SUCCESS] Generated {len(trades)} trade recommendations")

            # 4. Risk assessment
            portfolio_risk = await self._assess_portfolio_risk(target_portfolio, market_data)

            return {
                'signals': signals,
                'target_portfolio': target_portfolio,
                'trade_recommendations': trades,
                'risk_assessment': portfolio_risk,
                'market_analysis': {
                    'total_signals': len(signals),
                    'avg_confidence': np.mean([s['confidence'] for s in signals.values()]) if signals else 0,
                    'expected_portfolio_return': sum(p['expected_return'] * p['target_weight']
                                                   for p in target_portfolio.values()) if target_portfolio else 0
                }
            }

        except Exception as e:
            print(f"[ERROR] Analysis failed: {str(e)}")
            return {'error': str(e)}

    async def _assess_portfolio_risk(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Assess overall portfolio risk"""

        if not portfolio:
            return {'portfolio_risk': 0, 'risk_level': 'LOW'}

        # Calculate portfolio volatility
        returns_data = {}
        for symbol in portfolio.keys():
            returns_data[symbol] = market_data[symbol]['Close'].pct_change().dropna()

        portfolio_weights = [data['target_weight'] for data in portfolio.values()]
        symbols = list(portfolio.keys())

        # Simple correlation-based risk calc
        total_risk = 0
        for i, symbol1 in enumerate(symbols):
            weight1 = portfolio_weights[i]
            vol1 = returns_data[symbol1].std() * np.sqrt(252)  # Annualized volatility

            total_risk += (weight1 * vol1) ** 2

            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                weight2 = portfolio_weights[j]
                vol2 = returns_data[symbol2].std() * np.sqrt(252)
                correlation = returns_data[symbol1].corr(returns_data[symbol2])

                total_risk += 2 * weight1 * weight2 * vol1 * vol2 * correlation

        portfolio_volatility = np.sqrt(total_risk)

        risk_level = 'LOW' if portfolio_volatility < 0.15 else 'MEDIUM' if portfolio_volatility < 0.25 else 'HIGH'

        return {
            'portfolio_risk': portfolio_volatility,
            'risk_level': risk_level,
            'diversification_score': len(portfolio) / self.config['max_positions'],
            'max_position_weight': max(p['target_weight'] for p in portfolio.values()) if portfolio else 0
        }


# Supporting AI Component Classes
class EnsembleMLPredictor:
    """Ensemble ML predictor using multiple models"""

    async def predict_daily_returns(self, market_data: Dict) -> Dict:
        """Predict next day returns using ensemble of ML models"""
        predictions = {}

        for symbol, data in market_data.items():
            if len(data) < 60:  # Need sufficient history
                continue

            # Simulate ensemble predictions (would use real ML models)

            # 1. Momentum model
            momentum_pred = self._momentum_model(data)

            # 2. Mean reversion model
            mean_rev_pred = self._mean_reversion_model(data)

            # 3. Volume-price model
            vol_price_pred = self._volume_price_model(data)

            # 4. Volatility regime model
            vol_regime_pred = self._volatility_regime_model(data)

            # Ensemble combination with dynamic weights
            ensemble_pred = (
                momentum_pred * 0.3 +
                mean_rev_pred * 0.25 +
                vol_price_pred * 0.25 +
                vol_regime_pred * 0.2
            )

            # Calculate confidence based on model agreement
            model_preds = [momentum_pred, mean_rev_pred, vol_price_pred, vol_regime_pred]
            model_std = np.std(model_preds)
            confidence = max(0.1, 1.0 - model_std * 10)  # Higher agreement = higher confidence

            predictions[symbol] = {
                'expected_return': ensemble_pred,
                'confidence': confidence,
                'model_components': {
                    'momentum': momentum_pred,
                    'mean_reversion': mean_rev_pred,
                    'volume_price': vol_price_pred,
                    'volatility_regime': vol_regime_pred
                }
            }

        return predictions

    def _momentum_model(self, data: pd.DataFrame) -> float:
        """Simulate momentum-based prediction"""
        returns = data['Close'].pct_change().dropna()
        momentum_5d = returns.rolling(5).mean().iloc[-1] if len(returns) >= 5 else 0
        momentum_20d = returns.rolling(20).mean().iloc[-1] if len(returns) >= 20 else 0
        return momentum_5d * 0.6 + momentum_20d * 0.4

    def _mean_reversion_model(self, data: pd.DataFrame) -> float:
        """Simulate mean reversion prediction"""
        if len(data) < 20:
            return 0
        price_change = (data['Close'].iloc[-1] / data['Close'].rolling(20).mean().iloc[-1] - 1)
        return -price_change * 0.3  # Expect reversion

    def _volume_price_model(self, data: pd.DataFrame) -> float:
        """Simulate volume-price relationship prediction"""
        if len(data) < 10:
            return 0
        returns = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        correlation = returns.rolling(10).corr(volume_change).iloc[-1]
        return correlation * returns.iloc[-1] if not np.isnan(correlation) else 0

    def _volatility_regime_model(self, data: pd.DataFrame) -> float:
        """Simulate volatility regime prediction"""
        if len(data) < 30:
            return 0
        returns = data['Close'].pct_change().dropna()
        current_vol = returns.rolling(10).std().iloc[-1]
        long_vol = returns.rolling(30).std().iloc[-1]
        vol_ratio = current_vol / long_vol if long_vol > 0 else 1
        # In high vol regimes, expect continuation
        return returns.iloc[-1] * min(vol_ratio, 2.0) * 0.5


class AdvancedSignalValidator:
    """Advanced signal validation and regime detection"""

    async def validate_market_regime(self, market_data: Dict) -> Dict:
        """Detect current market regime and adjust signals"""

        # Aggregate market indicators
        market_returns = []
        market_volumes = []

        for symbol, data in market_data.items():
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                market_returns.extend(returns.tail(20).values)
                market_volumes.extend(data['Volume'].tail(20).values)

        if not market_returns:
            return {'market_regime_multiplier': 1.0, 'regime': 'UNKNOWN'}

        # Detect regime characteristics
        avg_return = np.mean(market_returns)
        volatility = np.std(market_returns) * np.sqrt(252)
        volume_trend = np.mean(np.diff(market_volumes)) / np.mean(market_volumes)

        # Classify regime
        if avg_return > 0.001 and volatility < 0.2:
            regime = 'BULL_LOW_VOL'
            multiplier = 1.2  # Favorable for momentum
        elif avg_return > 0.001 and volatility >= 0.2:
            regime = 'BULL_HIGH_VOL'
            multiplier = 1.0  # Neutral
        elif avg_return <= 0.001 and volatility < 0.2:
            regime = 'BEAR_LOW_VOL'
            multiplier = 0.8  # Reduce risk
        else:
            regime = 'BEAR_HIGH_VOL'
            multiplier = 0.5  # High risk, reduce exposure

        return {
            'market_regime_multiplier': multiplier,
            'regime': regime,
            'market_volatility': volatility,
            'market_trend': avg_return * 252,  # Annualized
            'volume_trend': volume_trend
        }


class DailyPortfolioOptimizer:
    """Portfolio optimization for daily rebalancing"""

    def optimize_weights(self, expected_returns: Dict, risk_scores: Dict, constraints: Dict) -> Dict:
        """Optimize portfolio weights using risk-adjusted returns"""

        symbols = list(expected_returns.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return {}

        # Simple optimization: weight by risk-adjusted returns
        risk_adjusted_returns = {}
        for symbol in symbols:
            expected_ret = expected_returns[symbol]
            risk = max(risk_scores.get(symbol, 0.1), 0.01)
            risk_adjusted_returns[symbol] = expected_ret / risk

        # Normalize to sum to maximum exposure
        total_score = sum(max(0, score) for score in risk_adjusted_returns.values())
        max_exposure = constraints.get('max_exposure', 0.95)
        max_position = constraints.get('max_position', 0.2)

        optimized_weights = {}
        if total_score > 0:
            for symbol in symbols:
                if risk_adjusted_returns[symbol] > 0:
                    raw_weight = (risk_adjusted_returns[symbol] / total_score) * max_exposure
                    optimized_weights[symbol] = min(raw_weight, max_position)

        return optimized_weights


class SingleUserRiskManager:
    """Risk management for single user trading"""

    async def assess_market_conditions(self, market_data: Dict) -> Dict:
        """Assess market-wide risk conditions"""

        risk_scores = {}

        for symbol, data in market_data.items():
            if len(data) < 30:
                risk_scores[symbol] = {'risk_score': 0.5, 'risk_multiplier': 1.0}
                continue

            # Calculate various risk metrics
            returns = data['Close'].pct_change().dropna()

            # 1. Volatility risk
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            vol_risk = min(1.0, volatility / 0.3)  # Normalize to 30% vol

            # 2. Drawdown risk
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.rolling(window=60, min_periods=1).max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            dd_risk = min(1.0, max_drawdown / 0.2)  # Normalize to 20% drawdown

            # 3. Liquidity risk (volume trend)
            volume_trend = data['Volume'].rolling(10).mean().iloc[-1] / data['Volume'].rolling(30).mean().iloc[-1]
            liquidity_risk = max(0.1, min(1.0, 2.0 - volume_trend))

            # Combined risk score
            combined_risk = (vol_risk * 0.4 + dd_risk * 0.4 + liquidity_risk * 0.2)
            risk_multiplier = max(0.3, 1.5 - combined_risk)  # Reduce exposure for high risk

            risk_scores[symbol] = {
                'risk_score': combined_risk,
                'risk_multiplier': risk_multiplier,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'volume_ratio': volume_trend
            }

        return risk_scores


class AlternativeDataAnalyzer:
    """Alternative data analysis for enhanced signals"""

    async def get_sentiment_signals(self, market_data: Dict) -> Dict:
        """Generate sentiment-based signals (simulated for demo)"""

        sentiment_signals = {}

        for symbol, data in market_data.items():
            # Simulate various alternative data sources

            # 1. Social media sentiment (simulated)
            social_sentiment = np.random.normal(0, 0.1)  # Would be real sentiment data

            # 2. News sentiment (simulated)
            news_sentiment = np.random.normal(0, 0.05)   # Would be real news analysis

            # 3. Analyst revision trend (simulated)
            analyst_trend = np.random.normal(0, 0.03)    # Would be real analyst data

            # 4. Insider trading activity (simulated)
            insider_signal = np.random.normal(0, 0.02)   # Would be real insider data

            # 5. Economic indicators correlation (simulated)
            econ_correlation = np.random.normal(0, 0.02) # Would be real economic data

            # Combine alternative data signals
            composite_sentiment = (
                social_sentiment * 0.3 +
                news_sentiment * 0.25 +
                analyst_trend * 0.2 +
                insider_signal * 0.15 +
                econ_correlation * 0.1
            )

            # Calculate confidence based on data quality
            confidence = min(0.9, abs(composite_sentiment) * 5 + 0.4)

            sentiment_signals[symbol] = {
                'composite_sentiment': composite_sentiment,
                'confidence': confidence,
                'components': {
                    'social_sentiment': social_sentiment,
                    'news_sentiment': news_sentiment,
                    'analyst_trend': analyst_trend,
                    'insider_signal': insider_signal,
                    'economic_correlation': econ_correlation
                }
            }

        return sentiment_signals


# Demo usage
async def run_single_user_demo():
    """Demonstrate the single-user AI trading system"""

    print("="*80)
    print("SINGLE-USER AI TRADING SYSTEM DEMO")
    print("Optimized for Fidelity Account Constraints")
    print("="*80)

    # Initialize system
    ai_system = SingleUserAITradingSystem(initial_capital=100000)

    # Generate sample market data (would be real data in production)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
    market_data = {}

    for symbol in symbols:
        # Generate realistic sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        volumes = np.random.gamma(2, 1000000, 100)

        market_data[symbol] = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            'High': prices * (1 + np.random.uniform(0, 0.02, 100)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, 100)),
            'Close': prices,
            'Volume': volumes
        }, index=dates)

    # Run daily analysis
    results = await ai_system.run_daily_analysis(market_data)

    if 'error' not in results:
        print("\n[RESULTS] Daily Analysis Complete")
        print(f"Market Analysis: {results['market_analysis']}")
        print(f"\nTrade Recommendations ({len(results['trade_recommendations'])}):")

        for trade in results['trade_recommendations'][:5]:  # Show top 5
            print(f"  {trade['action']} {trade['shares']} shares of {trade['symbol']}")
            print(f"    Reason: {trade['reason']}")
            print(f"    Expected Value: ${trade['estimated_value']:,.0f}")
            if 'confidence' in trade:
                print(f"    Confidence: {trade['confidence']:.1%}")
            print()

        print(f"Portfolio Risk Assessment: {results['risk_assessment']['risk_level']} risk")
        print(f"Expected Portfolio Return: {results['market_analysis']['expected_portfolio_return']:.2%}")

    return results


if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(run_single_user_demo())