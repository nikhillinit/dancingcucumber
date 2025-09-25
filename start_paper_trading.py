"""
START PAPER TRADING - Simple Working System
===========================================
Works with basic Python libraries only
"""

import json
import csv
from datetime import datetime, timedelta
import random
import os

class SimplePaperTradingSystem:
    """Ultra-simple system to start paper trading TODAY"""

    def __init__(self):
        self.universe = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLV', 'XLK', 'GLD']
        self.capital = 10000
        self.positions = {}
        self.trades_file = 'paper_trades.csv'

    def generate_mock_signals(self):
        """Generate signals for demonstration - replace with real data later"""

        signals = {}

        print("\n📊 GENERATING SIGNALS")
        print("-" * 40)

        for symbol in self.universe:
            # Mock signal (0-1 score)
            # In production: use real momentum, volume, sentiment
            momentum = random.uniform(0.3, 0.9)
            volume = random.uniform(0.2, 0.8)
            sentiment = random.uniform(0.4, 0.7)

            # Combine signals
            combined = (momentum * 0.5 + volume * 0.25 + sentiment * 0.25)
            signals[symbol] = combined

            print(f"{symbol}: {combined:.3f} (Mom:{momentum:.2f} Vol:{volume:.2f} Sent:{sentiment:.2f})")

        return signals

    def calculate_portfolio_weights(self, signals):
        """Convert signals to weights"""

        # Take top 4 signals
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)

        weights = {}
        n_positions = 4

        print("\n💼 PORTFOLIO ALLOCATION")
        print("-" * 40)

        for symbol, signal in sorted_signals[:n_positions]:
            if signal > 0.5:  # Only positive signals
                weights[symbol] = 0.25  # Equal weight for simplicity
                print(f"{symbol}: 25.0% (signal: {signal:.3f})")

        return weights

    def calculate_orders(self, weights):
        """Calculate what to buy/sell"""

        orders = []

        print("\n📝 ORDERS TO EXECUTE")
        print("-" * 40)

        # Mock prices (in production: fetch real prices)
        mock_prices = {
            'SPY': 450, 'QQQ': 380, 'IWM': 190, 'XLF': 40,
            'XLE': 85, 'XLV': 140, 'XLK': 180, 'GLD': 185
        }

        for symbol, target_weight in weights.items():
            target_value = self.capital * target_weight
            price = mock_prices.get(symbol, 100)
            shares = int(target_value / price)

            if shares > 0:
                orders.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': shares * price
                })

                print(f"BUY {shares} {symbol} @ ${price} = ${shares * price:,.0f}")

        return orders

    def save_trade_log(self, signals, weights, orders):
        """Save to CSV for tracking"""

        # Create or append to CSV
        file_exists = os.path.exists(self.trades_file)

        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Write header
                writer.writerow(['Date', 'Time', 'Symbol', 'Action', 'Shares', 'Price', 'Value', 'Signal'])

            # Write trades
            for order in orders:
                writer.writerow([
                    datetime.now().date(),
                    datetime.now().strftime('%H:%M:%S'),
                    order['symbol'],
                    order['action'],
                    order['shares'],
                    order['price'],
                    order['value'],
                    signals.get(order['symbol'], 0)
                ])

        print(f"\n💾 Trades saved to {self.trades_file}")

    def run_daily_analysis(self):
        """Complete daily workflow"""

        print("\n" + "="*50)
        print("🚀 PAPER TRADING SYSTEM - DAILY RUN")
        print("="*50)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Capital: ${self.capital:,}")
        print("="*50)

        # 1. Generate signals
        signals = self.generate_mock_signals()

        # 2. Calculate weights
        weights = self.calculate_portfolio_weights(signals)

        # 3. Generate orders
        orders = self.calculate_orders(weights)

        # 4. Save log
        self.save_trade_log(signals, weights, orders)

        # 5. Summary
        print("\n✅ DAILY ANALYSIS COMPLETE")
        print("-" * 40)
        print(f"Signals generated: {len(signals)}")
        print(f"Positions selected: {len(weights)}")
        print(f"Orders created: {len(orders)}")

        return {
            'date': datetime.now().isoformat(),
            'signals': signals,
            'weights': weights,
            'orders': orders
        }

def create_tracking_spreadsheet():
    """Create Excel-compatible CSV for manual tracking"""

    filename = 'paper_trading_tracker.csv'

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Date', 'SPY Signal', 'QQQ Signal', 'IWM Signal',
                'Portfolio Value', 'Daily P&L', 'Total P&L',
                'SPY Benchmark', 'Alpha', 'Notes'
            ])
        print(f"📊 Created tracking spreadsheet: {filename}")

    return filename

def setup_daily_routine():
    """Set up everything for daily paper trading"""

    print("\n" + "="*60)
    print("📋 SETTING UP PAPER TRADING ROUTINE")
    print("="*60)

    # 1. Create system
    system = SimplePaperTradingSystem()

    # 2. Create tracking spreadsheet
    tracker = create_tracking_spreadsheet()

    # 3. Run first analysis
    results = system.run_daily_analysis()

    # 4. Instructions
    print("\n" + "="*60)
    print("📌 DAILY ROUTINE INSTRUCTIONS")
    print("="*60)
    print("\n1️⃣  RUN EVERY MORNING (9:00 AM):")
    print("   python start_paper_trading.py")

    print("\n2️⃣  TRACK IN SPREADSHEET:")
    print(f"   Open: {tracker}")
    print("   Record daily P&L (hypothetical)")

    print("\n3️⃣  COMPARE TO BENCHMARK:")
    print("   Track SPY daily change")
    print("   Calculate your alpha (your return - SPY return)")

    print("\n4️⃣  AFTER 30 DAYS:")
    print("   • If profitable vs SPY → consider real trading")
    print("   • If unprofitable → refine signals first")

    print("\n" + "="*60)
    print("💡 NEXT STEPS TO IMPROVE")
    print("="*60)
    print("\n1. Add real price data (install yfinance when possible)")
    print("2. Add real momentum calculation")
    print("3. Add SEC filing alerts")
    print("4. Add volume analysis")
    print("5. Connect to broker (after profitable paper trading)")

    return results

if __name__ == "__main__":
    # START PAPER TRADING NOW
    results = setup_daily_routine()

    # Save results
    with open('latest_signals.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ Paper trading system ready!")
    print("📁 Files created:")
    print("   • paper_trades.csv (trade log)")
    print("   • paper_trading_tracker.csv (P&L tracking)")
    print("   • latest_signals.json (today's signals)")
    print("\n⏰ Run this script daily at 9:00 AM to track performance")