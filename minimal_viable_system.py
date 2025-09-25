"""
MINIMAL VIABLE TRADING SYSTEM
=============================
Simplest possible implementation that can actually trade
Uses only free, accessible data sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List
import os

class MinimalViableSystem:
    """Minimal system that can actually generate real signals"""

    def __init__(self):
        self.universe = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLV', 'XLK']
        self.positions = {}
        self.capital = 10000  # Start small for testing

    def get_real_price_data(self, symbols: List[str], period: str = '1mo') -> pd.DataFrame:
        """Get REAL price data from Yahoo Finance"""

        try:
            # Download real data
            data = yf.download(symbols, period=period, interval='1d', progress=False)

            if len(symbols) == 1:
                # Single symbol - restructure
                df = pd.DataFrame(data)
                df.columns = pd.MultiIndex.from_product([df.columns, symbols])
            else:
                df = data

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def calculate_simple_momentum(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate simple momentum signal (REAL)"""

        close_prices = prices['Close'] if 'Close' in prices.columns else prices

        # 20-day momentum
        momentum = close_prices.pct_change(20)

        # Rank across universe
        ranked = momentum.rank(axis=1, pct=True)

        return ranked.iloc[-1]  # Most recent day

    def get_simple_sentiment(self, symbol: str) -> float:
        """Get basic sentiment from free sources"""

        # For MVP, use price-based sentiment proxy
        # In production, add news API

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Simple sentiment based on analyst recommendations
            if 'recommendationMean' in info:
                # 1=Strong Buy, 5=Strong Sell
                rec = info['recommendationMean']
                sentiment = (5 - rec) / 4  # Convert to 0-1 scale
            else:
                sentiment = 0.5  # Neutral

            return sentiment

        except:
            return 0.5  # Default neutral

    def get_volume_signal(self, symbol: str) -> float:
        """Get volume-based signal (unusual volume)"""

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo')

            if len(hist) < 20:
                return 0.5

            # Current volume vs 20-day average
            current_vol = hist['Volume'].iloc[-1]
            avg_vol = hist['Volume'].iloc[-20:].mean()

            if avg_vol > 0:
                volume_ratio = current_vol / avg_vol
                # Cap between 0 and 2, then scale to 0-1
                signal = min(2, max(0, volume_ratio)) / 2
            else:
                signal = 0.5

            return signal

        except:
            return 0.5

    def generate_daily_signals(self) -> Dict[str, float]:
        """Generate real trading signals for today"""

        print("\n" + "="*60)
        print("GENERATING REAL SIGNALS")
        print(f"Time: {datetime.now()}")
        print("="*60)

        signals = {}

        # Get real price data
        print("\n1. Fetching real price data...")
        prices = self.get_real_price_data(self.universe, period='2mo')

        if prices.empty:
            print("ERROR: Could not fetch price data")
            return {}

        # Calculate momentum
        print("2. Calculating momentum signals...")
        momentum = self.calculate_simple_momentum(prices)

        # Get sentiment and volume for each symbol
        print("3. Analyzing individual stocks...")

        for symbol in self.universe:
            print(f"   Analyzing {symbol}...")

            # Momentum score
            mom_score = momentum.get(symbol, 0.5)

            # Sentiment score
            sent_score = self.get_simple_sentiment(symbol)

            # Volume score
            vol_score = self.get_volume_signal(symbol)

            # Combine signals (simple average for MVP)
            combined = (mom_score * 0.5 + sent_score * 0.25 + vol_score * 0.25)

            signals[symbol] = combined

            print(f"      Momentum: {mom_score:.2f}, Sentiment: {sent_score:.2f}, Volume: {vol_score:.2f}")
            print(f"      Combined: {combined:.2f}")

        return signals

    def signals_to_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Convert signals to portfolio weights"""

        # Simple approach: Top 4 stocks, equal weight
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)

        weights = {}
        n_positions = 4

        for i, (symbol, signal) in enumerate(sorted_signals[:n_positions]):
            if signal > 0.5:  # Only positive signals
                weights[symbol] = 1.0 / n_positions

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        return weights

    def generate_orders(self, target_weights: Dict[str, float]) -> List[Dict]:
        """Generate specific orders to execute"""

        orders = []

        # Get current prices
        current_prices = {}
        for symbol in target_weights:
            ticker = yf.Ticker(symbol)
            current_prices[symbol] = ticker.info.get('regularMarketPrice', 100)

        # Calculate target positions
        for symbol, target_weight in target_weights.items():
            target_value = self.capital * target_weight
            target_shares = int(target_value / current_prices[symbol])

            current_shares = self.positions.get(symbol, 0)
            shares_to_trade = target_shares - current_shares

            if abs(shares_to_trade) > 0:
                orders.append({
                    'symbol': symbol,
                    'action': 'BUY' if shares_to_trade > 0 else 'SELL',
                    'shares': abs(shares_to_trade),
                    'price': current_prices[symbol],
                    'value': abs(shares_to_trade * current_prices[symbol])
                })

        return orders

    def run_daily_process(self) -> Dict:
        """Complete daily trading process"""

        # 1. Generate signals
        signals = self.generate_daily_signals()

        if not signals:
            print("No signals generated")
            return {}

        # 2. Convert to weights
        print("\n" + "="*60)
        print("PORTFOLIO ALLOCATION")
        print("="*60)

        weights = self.signals_to_weights(signals)

        for symbol, weight in weights.items():
            print(f"{symbol}: {weight:.1%}")

        # 3. Generate orders
        print("\n" + "="*60)
        print("ORDERS TO EXECUTE")
        print("="*60)

        orders = self.generate_orders(weights)

        for order in orders:
            print(f"{order['action']} {order['shares']} {order['symbol']} @ ${order['price']:.2f}")

        return {
            'signals': signals,
            'weights': weights,
            'orders': orders,
            'timestamp': datetime.now().isoformat()
        }


def setup_paper_trading():
    """Set up paper trading workflow"""

    print("\n" + "="*70)
    print("PAPER TRADING SETUP")
    print("="*70)

    # 1. Check dependencies
    print("\n1. Checking dependencies...")

    required = ['yfinance', 'pandas', 'numpy']
    missing = []

    for module in required:
        try:
            __import__(module)
            print(f"   ✓ {module} installed")
        except ImportError:
            print(f"   ✗ {module} MISSING")
            missing.append(module)

    if missing:
        print(f"\nInstall missing: pip install {' '.join(missing)}")
        return False

    # 2. Test data connection
    print("\n2. Testing data connection...")

    try:
        test_data = yf.download('SPY', period='5d', progress=False)
        if not test_data.empty:
            print("   ✓ Yahoo Finance connection working")
        else:
            print("   ✗ Could not fetch data")
            return False
    except Exception as e:
        print(f"   ✗ Connection error: {e}")
        return False

    # 3. Initialize system
    print("\n3. Initializing minimal viable system...")

    mvs = MinimalViableSystem()

    # 4. Run test
    print("\n4. Running test signal generation...")

    results = mvs.run_daily_process()

    if results:
        print("\n✅ SYSTEM READY FOR PAPER TRADING")

        # Save results
        with open('paper_trades.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"   Results saved to paper_trades.json")
        return True
    else:
        print("\n✗ System test failed")
        return False


if __name__ == "__main__":
    # Run setup
    if setup_paper_trading():
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Run this daily at 9:00 AM ET")
        print("2. Record signals in spreadsheet")
        print("3. Paper trade for 30 days")
        print("4. Compare performance to SPY")
        print("5. Graduate to small real positions if profitable")