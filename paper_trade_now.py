"""
START PAPER TRADING NOW - Clean Version
========================================
Simple system to begin paper trading immediately
"""

import json
import csv
from datetime import datetime
import random
import os

def run_paper_trading():
    """Run paper trading analysis"""

    print("\n" + "="*60)
    print("PAPER TRADING SYSTEM - DAILY RUN")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Capital: $10,000")
    print("="*60)

    # Define universe
    universe = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLV', 'XLK', 'GLD']

    # Generate signals
    print("\n[SIGNALS] Generating trading signals...")
    print("-" * 40)

    signals = {}
    for symbol in universe:
        # Mock signals - replace with real data later
        momentum = random.uniform(0.3, 0.9)
        volume = random.uniform(0.2, 0.8)
        sentiment = random.uniform(0.4, 0.7)
        combined = (momentum * 0.5 + volume * 0.25 + sentiment * 0.25)
        signals[symbol] = combined
        print(f"{symbol}: {combined:.3f}")

    # Select top positions
    print("\n[PORTFOLIO] Selected positions:")
    print("-" * 40)

    sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    selected = []

    for symbol, signal in sorted_signals[:4]:
        if signal > 0.5:
            selected.append(symbol)
            print(f"{symbol}: 25% allocation (signal: {signal:.3f})")

    # Calculate orders
    print("\n[ORDERS] Orders to execute:")
    print("-" * 40)

    mock_prices = {
        'SPY': 450, 'QQQ': 380, 'IWM': 190, 'XLF': 40,
        'XLE': 85, 'XLV': 140, 'XLK': 180, 'GLD': 185
    }

    orders = []
    for symbol in selected:
        price = mock_prices.get(symbol, 100)
        shares = int(2500 / price)  # $2,500 per position
        value = shares * price

        orders.append({
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'value': value
        })

        print(f"BUY {shares} {symbol} @ ${price} = ${value:,.0f}")

    # Save to CSV
    csv_file = 'paper_trades.csv'
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['Date', 'Time', 'Symbol', 'Shares', 'Price', 'Signal'])

        for order in orders:
            writer.writerow([
                datetime.now().date(),
                datetime.now().strftime('%H:%M'),
                order['symbol'],
                order['shares'],
                order['price'],
                signals[order['symbol']]
            ])

    print(f"\n[SAVED] Trades saved to {csv_file}")

    # Create tracking sheet if needed
    tracker_file = 'tracking.csv'
    if not os.path.exists(tracker_file):
        with open(tracker_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Portfolio Value', 'Daily P&L', 'Total P&L', 'SPY Change', 'Alpha'])
        print(f"[CREATED] Tracking sheet: {tracker_file}")

    print("\n" + "="*60)
    print("COMPLETE - Daily analysis finished")
    print("="*60)

    return {
        'signals': signals,
        'selected': selected,
        'orders': orders
    }

if __name__ == "__main__":
    # Run the system
    results = run_paper_trading()

    # Save results
    with open('today_signals.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nNEXT STEPS:")
    print("1. Run this daily: python paper_trade_now.py")
    print("2. Track results in tracking.csv")
    print("3. After 30 days, analyze performance")
    print("4. If profitable, move to real trading with small capital")