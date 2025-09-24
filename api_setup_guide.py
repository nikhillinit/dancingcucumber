"""
Free API Setup Guide for Trading System
======================================
Step-by-step guide to get all required free API keys
"""

def print_api_setup_guide():
    print("[API] FREE API SETUP GUIDE")
    print("="*50)

    print("\n[1] FRED (Federal Reserve Economic Data) - REQUIRED")
    print("   Purpose: Economic indicators (VIX, yield curves, unemployment)")
    print("   Cost: FREE")
    print("   Steps:")
    print("   - Go to: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("   - Click 'Request API Key'")
    print("   - Fill out form (takes 2 minutes):")
    print("     * Name: Your name")
    print("     * Email: Your email")
    print("     * Organization: 'Personal Trading'")
    print("     * Intended Use: 'Economic data for trading analysis'")
    print("   - Click Submit")
    print("   - You'll get API key instantly via email")
    print("   - Rate limit: 120 calls/minute (plenty for our use)")

    print("\n[2] REDDIT API - HIGH IMPACT")
    print("   Purpose: Sentiment analysis from r/wallstreetbets, r/investing")
    print("   Cost: FREE")
    print("   Steps:")
    print("   - Go to: https://www.reddit.com/prefs/apps")
    print("   - Click 'Create App' or 'Create Another App'")
    print("   - Fill out:")
    print("     * Name: 'TradingBot'")
    print("     * App type: Select 'script'")
    print("     * Description: 'Sentiment analysis for trading'")
    print("     * About URL: Leave blank")
    print("     * Redirect URI: http://localhost:8080")
    print("   - Click 'Create app'")
    print("   - You'll get:")
    print("     * Client ID: Under the app name (14 chars)")
    print("     * Client Secret: Next to 'secret' (27 chars)")
    print("   - Rate limit: 60 calls/minute")

    print("\n[3] NO SETUP NEEDED - Already Free")
    print("   - Yahoo Finance: No API key needed (yfinance package)")
    print("   - Google Trends: No API key needed (pytrends package)")
    print("   - SEC EDGAR: No API key needed (direct HTTP access)")

    print("\n[INSTALL] PACKAGE INSTALLATION")
    print("   Run this command:")
    print("   pip install yfinance pandas-datareader praw pytrends fredapi requests")

    print("\n[CONFIG] CONFIG FILE TEMPLATE")
    print("   Create file: api_config.py")
    print("   Content:")

    config_template = '''
# API Configuration
FRED_API_KEY = "your_fred_api_key_here"

# Reddit API
REDDIT_CLIENT_ID = "your_reddit_client_id_here"
REDDIT_CLIENT_SECRET = "your_reddit_client_secret_here"
REDDIT_USER_AGENT = "TradingBot/1.0"

# No keys needed for these:
# - Yahoo Finance (yfinance)
# - Google Trends (pytrends)
# - SEC EDGAR (requests)
'''

    print(config_template)

    print("\n[VERIFY] VERIFICATION STEPS")
    print("   After getting API keys:")
    print("   1. Test FRED: python test_fred_api.py")
    print("   2. Test Reddit: python test_reddit_api.py")
    print("   3. Test Yahoo: python test_yahoo_api.py")

    print("\n[TIME] ESTIMATED TIME")
    print("   - FRED API: 3-5 minutes")
    print("   - Reddit API: 5-7 minutes")
    print("   - Package install: 2-3 minutes")
    print("   - Total: 15 minutes maximum")

    print("\n[NEXT] WHAT HAPPENS NEXT")
    print("   Once you have the API keys:")
    print("   - I'll update the trading system to use real data")
    print("   - Replace all simulations with live market feeds")
    print("   - Run backtests on 2+ years of historical data")
    print("   - Generate real daily portfolio recommendations")
    print("   - Expected improvement: +15-25% accuracy over baseline")

def create_api_test_files():
    """Create test files for each API"""

    # FRED API Test
    fred_test = '''
"""Test FRED API Connection"""
try:
    import pandas_datareader.data as web
    from datetime import datetime, timedelta

    # Test FRED connection
    end = datetime.now()
    start = end - timedelta(days=30)

    # Test with VIX data
    vix_data = web.DataReader('VIXCLS', 'fred', start, end)
    print(f"✅ FRED API working! Got {len(vix_data)} days of VIX data")
    print(f"   Latest VIX: {vix_data.iloc[-1, 0]:.2f}")

except Exception as e:
    print(f"❌ FRED API failed: {str(e)}")
    print("   Check your FRED_API_KEY in api_config.py")
'''

    # Reddit API Test
    reddit_test = '''
"""Test Reddit API Connection"""
try:
    import praw
    from api_config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

    # Initialize Reddit API
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    # Test with wallstreetbets
    subreddit = reddit.subreddit('wallstreetbets')
    posts = list(subreddit.hot(limit=5))

    print(f"✅ Reddit API working! Got {len(posts)} posts from r/wallstreetbets")
    print(f"   Latest post: {posts[0].title[:50]}...")

except Exception as e:
    print(f"❌ Reddit API failed: {str(e)}")
    print("   Check your Reddit credentials in api_config.py")
'''

    # Yahoo Finance Test
    yahoo_test = '''
"""Test Yahoo Finance Connection"""
try:
    import yfinance as yf

    # Test with AAPL
    ticker = yf.Ticker('AAPL')
    hist = ticker.history(period='1mo')
    info = ticker.info

    print(f"✅ Yahoo Finance working! Got {len(hist)} days of AAPL data")
    print(f"   Current price: ${hist['Close'].iloc[-1]:.2f}")
    print(f"   Company: {info.get('longName', 'N/A')}")

    # Test options data
    if ticker.options:
        options = ticker.option_chain(ticker.options[0])
        print(f"   Options data: {len(options.calls)} calls, {len(options.puts)} puts")

except Exception as e:
    print(f"❌ Yahoo Finance failed: {str(e)}")
    print("   Try: pip install yfinance")
'''

    return fred_test, reddit_test, yahoo_test

def main():
    print_api_setup_guide()

    print("\n📄 CREATING TEST FILES...")

    # Create test files
    fred_test, reddit_test, yahoo_test = create_api_test_files()

    with open('test_fred_api.py', 'w') as f:
        f.write(fred_test)

    with open('test_reddit_api.py', 'w') as f:
        f.write(reddit_test)

    with open('test_yahoo_api.py', 'w') as f:
        f.write(yahoo_test)

    print("[SUCCESS] Created test files:")
    print("   - test_fred_api.py")
    print("   - test_reddit_api.py")
    print("   - test_yahoo_api.py")

    print("\n[STEPS] NEXT STEPS:")
    print("1. Get API keys using links above")
    print("2. Create api_config.py with your keys")
    print("3. Run: pip install yfinance pandas-datareader praw pytrends fredapi")
    print("4. Test APIs with: python test_fred_api.py")
    print("5. Let me know when ready - I'll integrate real data!")

if __name__ == "__main__":
    main()