"""
FIDELITY AUTOMATED TRADING SYSTEM
=================================
Fully automated trading execution for Fidelity brokerage
Uses unofficial fidelity-api package for browser automation
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fidelity_trading.log'),
        logging.StreamHandler()
    ]
)

class FidelityAutomatedTrading:
    """Automated trading system for Fidelity brokerage"""

    def __init__(self, config_file: str = 'fidelity_config.json'):
        """Initialize Fidelity automated trading"""

        self.logger = logging.getLogger(__name__)
        self.config = self.load_config(config_file)
        self.fidelity_session = None

        # Trading parameters
        self.max_position_size = 0.15  # 15% max per position
        self.min_trade_value = 100  # Minimum $100 per trade
        self.rebalance_threshold = 0.0025  # 25 bps threshold

        # Safety parameters
        self.max_daily_trades = 20
        self.require_confirmation = self.config.get('require_confirmation', True)
        self.dry_run = self.config.get('dry_run', True)

        self.logger.info("Fidelity Automated Trading System Initialized")
        if self.dry_run:
            self.logger.warning("DRY RUN MODE - No actual trades will be executed")

    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                "username": "",  # Set your Fidelity username
                "password": "",  # Use environment variable in production
                "account_number": "",  # Your account number
                "require_confirmation": True,
                "dry_run": True,  # Start in dry run mode for safety
                "max_retries": 3,
                "timeout": 30
            }

            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)

            self.logger.warning(f"Created default config at {config_file}")
            self.logger.warning("Please update with your Fidelity credentials")
            return default_config

    def setup_fidelity_api(self):
        """Setup Fidelity API connection using playwright automation"""

        try:
            # This would use the fidelity-api package
            # pip install fidelity-api
            from fidelity import FidelityAutomation

            self.logger.info("Connecting to Fidelity...")

            # Initialize automation
            self.fidelity_session = FidelityAutomation(
                username=self.config['username'],
                password=os.environ.get('FIDELITY_PASSWORD', self.config['password']),
                account=self.config['account_number'],
                headless=True  # Run browser in background
            )

            # Login with 2FA support
            self.fidelity_session.login()

            self.logger.info("Successfully connected to Fidelity")
            return True

        except ImportError:
            self.logger.warning("fidelity-api package not installed")
            self.logger.info("Install with: pip install fidelity-api")

            # Fallback to manual implementation
            return self.setup_manual_automation()

        except Exception as e:
            self.logger.error(f"Failed to connect to Fidelity: {e}")
            return False

    def setup_manual_automation(self):
        """Manual browser automation fallback using Playwright"""

        try:
            from playwright.sync_api import sync_playwright

            self.logger.info("Setting up manual Playwright automation...")

            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=True)
            self.page = browser.new_page()

            # Navigate to Fidelity
            self.page.goto("https://digital.fidelity.com/prgw/digital/login/full-page")

            # Login process
            self.page.fill('input[name="username"]', self.config['username'])
            self.page.fill('input[name="password"]',
                          os.environ.get('FIDELITY_PASSWORD', self.config['password']))
            self.page.click('button[type="submit"]')

            # Wait for 2FA if needed
            self.page.wait_for_timeout(5000)

            self.logger.info("Manual automation setup complete")
            return True

        except ImportError:
            self.logger.error("Playwright not installed")
            self.logger.info("Install with: pip install playwright")
            self.logger.info("Then run: playwright install chromium")
            return False

        except Exception as e:
            self.logger.error(f"Manual automation failed: {e}")
            return False

    def get_account_positions(self) -> Dict:
        """Get current account positions"""

        if self.dry_run:
            # Return mock positions for dry run
            return {
                'AAPL': {'shares': 100, 'value': 19000},
                'GOOGL': {'shares': 50, 'value': 7500},
                'MSFT': {'shares': 75, 'value': 28000},
                'Cash': {'value': 445500}
            }

        try:
            if self.fidelity_session:
                positions = self.fidelity_session.get_positions()
                return positions
            else:
                self.logger.warning("No active session, returning mock positions")
                return {}

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {}

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current market prices for symbols"""

        prices = {}

        for symbol in symbols:
            if self.dry_run:
                # Mock prices for dry run
                mock_prices = {
                    'AAPL': 190.00, 'GOOGL': 150.00, 'MSFT': 375.00,
                    'AMZN': 180.00, 'TSLA': 250.00, 'NVDA': 850.00,
                    'META': 500.00, 'JPM': 200.00
                }
                prices[symbol] = mock_prices.get(symbol, 100.00)
            else:
                try:
                    if self.fidelity_session:
                        price = self.fidelity_session.get_quote(symbol)
                        prices[symbol] = price
                    else:
                        prices[symbol] = 100.00  # Default fallback

                except Exception as e:
                    self.logger.error(f"Failed to get price for {symbol}: {e}")
                    prices[symbol] = 100.00

        return prices

    def calculate_orders(self, target_weights: Dict[str, float],
                        portfolio_value: float) -> List[Dict]:
        """Calculate orders needed to reach target weights"""

        orders = []

        # Get current positions
        positions = self.get_account_positions()
        cash_available = positions.get('Cash', {}).get('value', 0)

        # Get current prices
        symbols = list(target_weights.keys())
        prices = self.get_current_prices(symbols)

        # Calculate current weights
        current_weights = {}
        for symbol in symbols:
            if symbol in positions:
                current_value = positions[symbol].get('value', 0)
                current_weights[symbol] = current_value / portfolio_value
            else:
                current_weights[symbol] = 0

        # Calculate trades needed
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight

            # Check rebalance threshold
            if abs(weight_diff) < self.rebalance_threshold:
                continue

            # Calculate trade value and shares
            trade_value = weight_diff * portfolio_value

            if abs(trade_value) < self.min_trade_value:
                continue

            price = prices.get(symbol, 100)
            shares = int(trade_value / price)

            if shares == 0:
                continue

            order = {
                'symbol': symbol,
                'action': 'BUY' if shares > 0 else 'SELL',
                'shares': abs(shares),
                'price': price,
                'value': abs(shares * price),
                'order_type': 'MARKET',  # Use market orders at open
                'time_in_force': 'DAY'
            }

            orders.append(order)

        # Sort by value (largest trades first)
        orders.sort(key=lambda x: x['value'], reverse=True)

        # Limit number of daily trades
        if len(orders) > self.max_daily_trades:
            self.logger.warning(f"Limiting trades from {len(orders)} to {self.max_daily_trades}")
            orders = orders[:self.max_daily_trades]

        return orders

    def execute_order(self, order: Dict) -> bool:
        """Execute a single order"""

        symbol = order['symbol']
        action = order['action']
        shares = order['shares']

        self.logger.info(f"Executing: {action} {shares} shares of {symbol}")

        if self.dry_run:
            self.logger.info(f"DRY RUN: Would execute {action} {shares} {symbol}")
            return True

        try:
            if self.fidelity_session:
                # Execute through API
                if action == 'BUY':
                    result = self.fidelity_session.buy(
                        symbol=symbol,
                        quantity=shares,
                        order_type='MARKET',
                        time_in_force='DAY'
                    )
                else:
                    result = self.fidelity_session.sell(
                        symbol=symbol,
                        quantity=shares,
                        order_type='MARKET',
                        time_in_force='DAY'
                    )

                self.logger.info(f"Order executed: {result}")
                return True

            else:
                self.logger.warning("No active session for order execution")
                return False

        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return False

    def execute_daily_rebalance(self, ai_recommendations: Dict[str, float]) -> Dict:
        """Execute complete daily rebalancing based on AI recommendations"""

        self.logger.info("="*60)
        self.logger.info("STARTING DAILY REBALANCE")
        self.logger.info(f"Time: {datetime.now()}")
        self.logger.info("="*60)

        # Get portfolio value
        positions = self.get_account_positions()
        portfolio_value = sum(pos.get('value', 0) for pos in positions.values())

        self.logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")

        # Calculate orders
        orders = self.calculate_orders(ai_recommendations, portfolio_value)

        if not orders:
            self.logger.info("No trades needed - portfolio is balanced")
            return {'status': 'balanced', 'trades': 0}

        self.logger.info(f"Generated {len(orders)} orders")

        # Display orders for review
        print("\n" + "="*60)
        print("PROPOSED TRADES")
        print("="*60)

        total_buy_value = 0
        total_sell_value = 0

        for i, order in enumerate(orders, 1):
            print(f"{i}. {order['action']} {order['shares']} {order['symbol']} @ ${order['price']:.2f} = ${order['value']:,.2f}")

            if order['action'] == 'BUY':
                total_buy_value += order['value']
            else:
                total_sell_value += order['value']

        print("-"*60)
        print(f"Total BUY:  ${total_buy_value:,.2f}")
        print(f"Total SELL: ${total_sell_value:,.2f}")
        print(f"Net Change: ${total_buy_value - total_sell_value:,.2f}")
        print("="*60)

        # Confirm execution
        if self.require_confirmation and not self.dry_run:
            response = input("\nExecute these trades? (yes/no): ")
            if response.lower() != 'yes':
                self.logger.info("Trade execution cancelled by user")
                return {'status': 'cancelled', 'trades': 0}

        # Execute orders
        successful = 0
        failed = 0

        for order in orders:
            if self.execute_order(order):
                successful += 1
                time.sleep(1)  # Brief pause between orders
            else:
                failed += 1

        # Summary
        self.logger.info("="*60)
        self.logger.info("REBALANCE COMPLETE")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info("="*60)

        return {
            'status': 'complete',
            'trades': successful,
            'failed': failed,
            'portfolio_value': portfolio_value
        }

    def automated_morning_routine(self):
        """Complete automated morning trading routine"""

        self.logger.info("\n" + "="*70)
        self.logger.info("AUTOMATED MORNING TRADING ROUTINE")
        self.logger.info("="*70)

        # Step 1: Connect to Fidelity
        self.logger.info("Step 1: Connecting to Fidelity...")
        if not self.setup_fidelity_api():
            self.logger.error("Failed to connect to Fidelity")
            return False

        # Step 2: Get AI recommendations
        self.logger.info("Step 2: Getting AI recommendations...")
        from ultimate_hedge_fund_system import UltimateHedgeFundSystem

        ai_system = UltimateHedgeFundSystem()
        report = ai_system.run_complete_daily_analysis()

        # Extract weights from report
        target_weights = {}
        if 'final_portfolio' in report:
            for asset in report['final_portfolio']['allocations']:
                symbol = asset['symbol']
                weight = asset['weight']
                if weight > 0:
                    target_weights[symbol] = weight

        self.logger.info(f"Received recommendations for {len(target_weights)} assets")

        # Step 3: Execute rebalance
        self.logger.info("Step 3: Executing rebalance...")
        result = self.execute_daily_rebalance(target_weights)

        # Step 4: Log results
        self.logger.info("Step 4: Logging results...")
        self.save_trade_log(result, target_weights)

        return result

    def save_trade_log(self, result: Dict, weights: Dict):
        """Save trade log for record keeping"""

        log_entry = {
            'date': datetime.now().isoformat(),
            'status': result['status'],
            'trades': result.get('trades', 0),
            'portfolio_value': result.get('portfolio_value', 0),
            'target_weights': weights
        }

        log_file = f"trade_logs/trades_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs('trade_logs', exist_ok=True)

        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)

        self.logger.info(f"Trade log saved to {log_file}")


def main():
    """Main execution function"""

    # Initialize system
    trader = FidelityAutomatedTrading()

    # Check market hours
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)

    if now.hour < 9:
        print(f"Market opens at 9:30 AM. Current time: {now.strftime('%H:%M')}")
        print("Waiting for market open...")

    # Run automated routine
    print("\n" + "="*70)
    print("FIDELITY AUTOMATED TRADING SYSTEM")
    print("="*70)
    print(f"Mode: {'DRY RUN' if trader.dry_run else 'LIVE TRADING'}")
    print(f"Confirmation Required: {trader.require_confirmation}")
    print("="*70)

    # Execute morning routine
    result = trader.automated_morning_routine()

    if result:
        print("\n✓ Daily trading routine completed successfully")
    else:
        print("\n✗ Trading routine encountered errors")


if __name__ == "__main__":
    main()