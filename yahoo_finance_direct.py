"""
DIRECT YAHOO FINANCE API ACCESS
================================
Access Yahoo Finance data directly without yfinance library
Using the same endpoints that yfinance uses
"""

import requests
import json
from datetime import datetime, timedelta
import time

class YahooFinanceDirect:
    """Direct access to Yahoo Finance API endpoints"""

    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_quote(self, symbol):
        """Get real-time quote data"""

        url = f"{self.base_url}/v8/finance/chart/{symbol}"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()

                result = data['chart']['result'][0]
                meta = result['meta']

                quote = {
                    'symbol': symbol,
                    'price': meta['regularMarketPrice'],
                    'previousClose': meta['previousClose'],
                    'volume': meta.get('regularMarketVolume', 0),
                    'change': meta['regularMarketPrice'] - meta['previousClose'],
                    'changePercent': ((meta['regularMarketPrice'] - meta['previousClose']) / meta['previousClose']) * 100,
                    'marketTime': datetime.fromtimestamp(meta['regularMarketTime']),
                    'bid': meta.get('bid', 0),
                    'ask': meta.get('ask', 0),
                    'dayHigh': meta.get('regularMarketDayHigh', 0),
                    'dayLow': meta.get('regularMarketDayLow', 0)
                }

                return quote
            else:
                return None

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def get_historical(self, symbol, period='1mo', interval='1d'):
        """Get historical price data

        Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """

        url = f"{self.base_url}/v8/finance/chart/{symbol}"
        params = {
            'range': period,
            'interval': interval
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()

                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]

                prices = []
                for i in range(len(timestamps)):
                    prices.append({
                        'date': datetime.fromtimestamp(timestamps[i]),
                        'open': quotes['open'][i],
                        'high': quotes['high'][i],
                        'low': quotes['low'][i],
                        'close': quotes['close'][i],
                        'volume': quotes['volume'][i]
                    })

                return prices
            else:
                return None

        except Exception as e:
            print(f"Error fetching historical for {symbol}: {e}")
            return None

    def get_options(self, symbol):
        """Get options chain data"""

        url = f"{self.base_url}/v7/finance/options/{symbol}"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()

                option_chain = data['optionChain']['result'][0]

                # Get expiration dates
                expirations = option_chain.get('expirationDates', [])

                if expirations:
                    # Get options for first expiration
                    strikes = option_chain.get('strikes', [])
                    calls = option_chain.get('options', [{}])[0].get('calls', [])
                    puts = option_chain.get('options', [{}])[0].get('puts', [])

                    return {
                        'expirations': [datetime.fromtimestamp(exp) for exp in expirations],
                        'strikes': strikes,
                        'calls': calls[:5],  # First 5 for demo
                        'puts': puts[:5]
                    }

            return None

        except Exception as e:
            print(f"Error fetching options for {symbol}: {e}")
            return None

    def get_multiple_quotes(self, symbols):
        """Get quotes for multiple symbols at once"""

        symbols_str = ','.join(symbols)
        url = f"{self.base_url}/v7/finance/quote"
        params = {'symbols': symbols_str}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()

                quotes = {}
                for quote in data['quoteResponse']['result']:
                    symbol = quote['symbol']
                    quotes[symbol] = {
                        'price': quote.get('regularMarketPrice', 0),
                        'change': quote.get('regularMarketChange', 0),
                        'changePercent': quote.get('regularMarketChangePercent', 0),
                        'volume': quote.get('regularMarketVolume', 0),
                        'marketCap': quote.get('marketCap', 0),
                        'pe': quote.get('trailingPE', 0)
                    }

                return quotes

        except Exception as e:
            print(f"Error fetching multiple quotes: {e}")
            return None


def test_direct_api():
    """Test direct Yahoo Finance API access"""

    print("\n" + "="*80)
    print("YAHOO FINANCE DIRECT API TEST")
    print("="*80)

    yf = YahooFinanceDirect()

    # Test symbols
    symbols = ['NVDA', 'MSFT', 'PLTR', 'QQQ', 'AAPL']

    print("\n1. REAL-TIME QUOTES (Direct API)")
    print("-"*50)

    for symbol in symbols:
        quote = yf.get_quote(symbol)
        if quote:
            print(f"{symbol:5s}: ${quote['price']:>7.2f} ({quote['changePercent']:+.2f}%) | Vol: {quote['volume']:,}")
        time.sleep(0.1)  # Be nice to Yahoo

    print("\n2. BULK QUOTES (More Efficient)")
    print("-"*50)

    bulk_quotes = yf.get_multiple_quotes(symbols)
    if bulk_quotes:
        for symbol, data in bulk_quotes.items():
            print(f"{symbol:5s}: ${data['price']:>7.2f} ({data['changePercent']:+.2f}%)")

    print("\n3. INTRADAY DATA (5-min bars)")
    print("-"*50)

    nvda_intraday = yf.get_historical('NVDA', period='1d', interval='5m')
    if nvda_intraday:
        print(f"NVDA 5-min bars: {len(nvda_intraday)} data points")
        latest = nvda_intraday[-1]
        print(f"Latest: {latest['date']} - ${latest['close']:.2f}")

    print("\n4. HISTORICAL DATA (Daily)")
    print("-"*50)

    msft_daily = yf.get_historical('MSFT', period='1mo', interval='1d')
    if msft_daily:
        print(f"MSFT daily data: {len(msft_daily)} days")
        print(f"Date range: {msft_daily[0]['date'].date()} to {msft_daily[-1]['date'].date()}")

    print("\n5. OPTIONS CHAIN")
    print("-"*50)

    options = yf.get_options('NVDA')
    if options:
        print(f"Expiration dates: {len(options['expirations'])}")
        print(f"First expiry: {options['expirations'][0].date()}")
        print(f"Strikes available: {len(options['strikes'])}")

    print("\n" + "="*80)
    print("DIRECT API ADVANTAGES")
    print("="*80)

    print("\nPROS:")
    print("  ✓ No library dependencies")
    print("  ✓ Lightweight and fast")
    print("  ✓ Full control over requests")
    print("  ✓ Can parallelize requests")
    print("  ✓ Works immediately")

    print("\nUSE CASES:")
    print("  • Real-time quotes: Every 1-5 seconds")
    print("  • Intraday data: 1-min bars for day trading")
    print("  • Options flow: Monitor unusual activity")
    print("  • Bulk updates: 100+ symbols at once")

    print("\nIMPLEMENTATION:")
    print("  1. Use for real-time quote updates")
    print("  2. Cache historical data locally")
    print("  3. Run every minute during market hours")
    print("  4. Store in database for backtesting")

    return yf


if __name__ == "__main__":
    print("Testing direct Yahoo Finance API access...")
    print("No library installation needed!")

    api = test_direct_api()

    print("\n" + "="*80)
    print("READY TO USE IN YOUR AI HEDGE FUND")
    print("="*80)
    print("This direct API approach works immediately!")
    print("No yfinance installation required")